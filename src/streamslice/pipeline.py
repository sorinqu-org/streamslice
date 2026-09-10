from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .audio_analysis import analyze_audio
from .config import project_path
from .context_analysis import review_candidate_context
from .curation import curate
from .discovery import discover_candidates
from .gemini import GeminiClient
from .identity import resolve_creator_identity
from .layout_analysis import analyze_dynamic_layout
from .media import cut_source_clip, detect_audio_activity, fingerprint, probe, stream_id
from .metadata import generate_metadata
from .models import Candidate, Word
from .overlay_analysis import detect_overlay_masks
from .process import require_binary
from .proxy import ensure_proxy
from .render import normalized_recommendation, prepare_render_props, render_clip
from .subtitles import build_subtitles
from .transcription import align_words_to_audio_activity, transcribe

LOGGER = logging.getLogger(__name__)


def doctor(config: dict[str, Any]) -> dict[str, Any]:
    tools = {
        name: require_binary(name)
        for name in ("ffmpeg", "ffprobe", "rclone", "node", "npm")
    }
    tools["chromium"] = require_binary(config["render"]["chromium_executable"])
    models = ensure_proxy(config)
    return {"tools": tools, "models": models, "configured_models": config["models"]}


def process_video(source: str | Path, config: dict[str, Any]) -> Path:
    started = time.monotonic()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file() or source_path.suffix.casefold() != ".mp4":
        raise FileNotFoundError(f"Input MP4 does not exist: {source_path}")
    ensure_proxy(config)
    info = probe(source_path)
    identifier = stream_id(source_path)
    source_fingerprint = fingerprint(source_path)
    work_root = project_path(config, config["runtime"]["work_dir"]) / identifier
    work_root.mkdir(parents=True, exist_ok=True)
    output_root = Path(config["output"]["root"]).expanduser().resolve() / identifier
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = {
        "stream_id": identifier,
        "source": str(source_path),
        "source_probe": info,
        "model": config["models"],
        "status": "processing",
        "started_at_unix": time.time(),
        "clips": [],
    }
    _write_json(manifest_path, manifest)
    client = GeminiClient(config)
    render_enabled = bool(config["render"].get("enabled", True))
    creator_identity = resolve_creator_identity(
        source_path,
        config=config,
        client=client,
        work_dir=work_root,
    )
    manifest["creator"] = creator_identity
    _write_json(manifest_path, manifest)

    with ThreadPoolExecutor(max_workers=2) as pool:
        transcript_future = pool.submit(
            transcribe,
            source_path,
            duration=info["duration"],
            work_dir=work_root,
            config=config,
            client=client,
        )
        peaks_future = pool.submit(
            analyze_audio,
            source_path,
            duration=info["duration"],
            work_dir=work_root,
            config=config,
        )
        words = transcript_future.result()
        peaks = peaks_future.result()

    candidates = discover_candidates(
        words,
        peaks,
        duration=info["duration"],
        work_dir=work_root,
        config=config,
        client=client,
        creator_identity=creator_identity,
    )
    candidates = review_candidate_context(
        source_path,
        words,
        candidates,
        work_dir=work_root,
        config=config,
        client=client,
        creator_identity=creator_identity,
    )
    highlights = curate(
        candidates,
        duration=info["duration"],
        work_dir=work_root,
        config=config,
        client=client,
        creator_identity=creator_identity,
    )
    highlights = _remove_repeated_scenes(highlights, candidates, words, info["duration"], config)
    for name in ("transcript.json", "audio-peaks.json", "selection.json", "candidates.json"):
        shutil.copy2(work_root / name, output_root / name)

    remotion_dir = project_path(config, config["render"]["remotion_dir"])
    for index, highlight in enumerate(highlights, start=1):
        clip_name = f"clip-{index:02d}"
        clip_dir = output_root / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)
        source_clip = work_root / "source-clips" / f"{clip_name}-source.mp4"
        source_clip_meta = source_clip.with_suffix(".json")
        expected_source_clip = {
            "source": str(source_path),
            "source_fingerprint": source_fingerprint,
            "start_time": round(highlight.start_time, 6),
            "end_time": round(highlight.end_time, 6),
        }
        reuse_source = False
        if source_clip.is_file() and source_clip_meta.is_file():
            try:
                cached_source_clip = json.loads(source_clip_meta.read_text(encoding="utf-8"))
                source_info = probe(source_clip)
                has_video = any(
                    item.get("codec_type") == "video" for item in source_info["streams"]
                )
                has_audio = any(
                    item.get("codec_type") == "audio" for item in source_info["streams"]
                )
                reuse_source = (
                    cached_source_clip == expected_source_clip
                    and has_video
                    and has_audio
                    and math.isclose(
                        source_info["duration"],
                        highlight.duration,
                        abs_tol=float(config["runtime"]["duration_tolerance_seconds"]),
                    )
                )
            except Exception:
                reuse_source = False
        if not reuse_source:
            source_clip.unlink(missing_ok=True)
            source_clip_meta.unlink(missing_ok=True)
            cut_source_clip(source_path, highlight, source_clip, config)
            _write_json(source_clip_meta, expected_source_clip)
        # Re-transcribe each cut independently. The chunk transcript is used for
        # discovery; exact-cut audio plus deterministic activity checks are the
        # source of truth for clip-local caption timing.
        clip_transcript_dir = clip_dir / "clip-transcript"
        local_words = transcribe(
            source_clip,
            duration=highlight.duration,
            work_dir=clip_transcript_dir,
            config=config,
            client=client,
        )
        audio_activity = detect_audio_activity(source_clip, duration=highlight.duration)
        caption_words, caption_alignment = align_words_to_audio_activity(
            local_words,
            audio_activity,
            highlight.duration,
        )
        caption_candidate = Candidate(
            start_time=0.0,
            end_time=highlight.duration,
            highlight_reason=highlight.highlight_reason,
            emotion_score=highlight.emotion_score,
            camera_layout_recommendation=highlight.camera_layout_recommendation,
        )
        subtitle_words = build_subtitles(caption_words, caption_candidate, clip_dir, config)
        (clip_dir / "caption-source.json").write_text(
            json.dumps(
                {
                    "source": "local",
                    "reason": "exact-cut transcript validated against local audio activity",
                    "alignment": caption_alignment,
                    "audio_activity_intervals": len(audio_activity),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        highlight.camera_layout_recommendation = analyze_dynamic_layout(
            source_clip,
            duration=highlight.duration,
            words=subtitle_words,
            candidate=highlight,
            output_dir=clip_dir,
            config=config,
            client=client,
        )
        transcript_text = " ".join(item["text"] for item in subtitle_words)
        metadata = generate_metadata(
            highlight,
            transcript_text,
            clip_dir,
            config,
            client,
            creator_identity=creator_identity,
        )
        overlay_masks = detect_overlay_masks(
            source_clip,
            duration=highlight.duration,
            output_dir=clip_dir,
            config=config,
            client=client,
        )
        output_video = clip_dir / f"{clip_name}.mp4"
        props_path = clip_dir / "remotion-props.json"
        if not render_enabled:
            bundle_source = clip_dir / "source.mp4"
            if (
                not bundle_source.is_file()
                or bundle_source.stat().st_size != source_clip.stat().st_size
            ):
                shutil.copy2(source_clip, bundle_source)
            _write_json(clip_dir / "selection.json", highlight.to_dict())
            prepare_render_props(
                source_name="source.mp4",
                candidate=highlight,
                words=subtitle_words,
                overlay_masks=overlay_masks,
                props_path=props_path,
                config=config,
                title=metadata.get("title", ""),
            )
            clip_record = {
                "index": index,
                "selection": highlight.to_dict(),
                "status": "render_pending",
                "bundle_source": str(bundle_source),
                "bundle_source_sha256": _sha256(bundle_source),
                "expected_video": str(output_video),
                "metadata": metadata,
            }
            manifest["clips"].append(clip_record)
            _write_json(manifest_path, manifest)
            continue
        expected_render_version = int(config["render"].get("render_version", 1))
        if output_video.is_file() and props_path.is_file():
            try:
                existing_props = json.loads(props_path.read_text(encoding="utf-8"))
                existing_version = int(existing_props.get("renderVersion", 0))
                existing_masks = existing_props.get("overlayMasks", [])
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                existing_version = 0
                existing_masks = []
            existing_words = existing_props.get("words", [])
            existing_recommendation = existing_props.get("layoutRecommendation", {})
            expected_recommendation = _normalized_recommendation(highlight)
            if (
                existing_version != expected_render_version
                or existing_masks != overlay_masks
                or existing_words != subtitle_words
                or existing_recommendation != expected_recommendation
            ):
                output_video.unlink(missing_ok=True)
        if output_video.is_file():
            try:
                render_info = probe(output_video)
            except Exception:
                output_video.unlink(missing_ok=True)
                render_info = render_clip(
                    source_clip,
                    highlight,
                    subtitle_words,
                    overlay_masks,
                    output_video,
                    job_id=f"{identifier}/{clip_name}",
                    config=config,
                )
        else:
            render_info = render_clip(
                source_clip,
                highlight,
                subtitle_words,
                overlay_masks,
                output_video,
                job_id=f"{identifier}/{clip_name}",
                config=config,
            )
        clip_record = {
            "index": index,
            "selection": highlight.to_dict(),
            "video": str(output_video),
            "probe": render_info,
            "metadata": metadata,
        }
        manifest["clips"].append(clip_record)
        _write_json(manifest_path, manifest)

    if highlights and not render_enabled:
        manifest["status"] = "awaiting_local_render"
        _write_json(
            output_root / "render-job.json",
            {
                "job_version": 1,
                "job_id": identifier,
                "created_at_unix": time.time(),
                "source": str(source_path),
                "source_fingerprint": source_fingerprint,
                "render_version": int(config["render"].get("render_version", 1)),
                "clips": [
                    {
                        "index": item["index"],
                        "directory": f"clip-{item['index']:02d}",
                        "source": f"clip-{item['index']:02d}/source.mp4",
                        "source_sha256": item["bundle_source_sha256"],
                        "output": f"clip-{item['index']:02d}/clip-{item['index']:02d}.mp4",
                    }
                    for item in manifest["clips"]
                ],
            },
        )
    else:
        manifest["status"] = "complete" if highlights else "complete_without_renders"
    manifest["elapsed_seconds"] = round(time.monotonic() - started, 3)
    manifest["completed_at_unix"] = time.time()
    _write_json(manifest_path, manifest)
    if not config["output"]["keep_intermediates"]:
        shutil.rmtree(work_root, ignore_errors=True)
        shutil.rmtree(remotion_dir / "public" / "jobs" / identifier, ignore_errors=True)
    LOGGER.info("Completed %s in %.1f seconds", identifier, manifest["elapsed_seconds"])
    return output_root


def process_downloaded(config: dict[str, Any]) -> list[Path]:
    local_dir = Path(config["sync"]["local_dir"]).expanduser().resolve()
    outputs: list[Path] = []
    for source in sorted(local_dir.rglob("*.mp4")):
        if source.name.endswith(".partial"):
            continue
        outputs.append(process_video(source, config))
    return outputs


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_recommendation(candidate: Candidate) -> dict[str, Any]:
    return normalized_recommendation(candidate)


def _remove_repeated_scenes(
    selected: list[Any],
    candidates: list[Any],
    words: list[Word],
    duration: float,
    config: dict[str, Any],
) -> list[Any]:
    """Prevent five clips from carrying the same spoken scene at different offsets."""
    minimum = float(config["selection"]["min_duration_seconds"])
    target = min(int(config["selection"]["final_count"]), len(candidates))
    accepted: list[Any] = []
    signatures: list[str] = []

    def signature(item: Any) -> str:
        return " ".join(
            word.text.casefold().strip(".,!?…")
            for word in words
            if item.start_time <= word.start <= item.end_time
        )

    def unique(item: Any) -> bool:
        value = signature(item)
        if not value:
            return True
        return all(SequenceMatcher(None, value, previous).ratio() < 0.78 for previous in signatures)

    for item in selected:
        if unique(item):
            accepted.append(item)
            signatures.append(signature(item))
            continue
        replacements = sorted(
            candidates, key=lambda candidate: candidate.emotion_score, reverse=True
        )
        replacement = next(
            (
                candidate
                for candidate in replacements
                if minimum <= candidate.duration <= 90
                and all(
                    min(candidate.end_time, current.end_time)
                    - max(candidate.start_time, current.start_time)
                    <= 1
                    for current in accepted
                )
                and unique(candidate)
            ),
            None,
        )
        if replacement is not None:
            accepted.append(replacement)
            signatures.append(signature(replacement))
    # Keep the requested cardinality only from context-approved candidates.
    for candidate in sorted(candidates, key=lambda item: item.emotion_score, reverse=True):
        if len(accepted) >= target:
            break
        if candidate in accepted or not unique(candidate):
            continue
        if all(
            min(candidate.end_time, current.end_time)
            - max(candidate.start_time, current.start_time)
            <= 1
            for current in accepted
        ):
            accepted.append(candidate)
            signatures.append(signature(candidate))
    accepted.sort(key=lambda item: item.start_time)
    return accepted
