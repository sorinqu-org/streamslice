#!/usr/bin/env python3
"""Measure how well a rendered clip frames the streamer's face.

The webcam crop is supposed to follow the face. Reading the filtergraph cannot
tell you whether it does, so this detects the face on rendered stills and reports
where it sits *inside its own band* — measuring against the whole frame is
meaningless for a split layout, where the webcam only occupies the top strip.

    PYTHONPATH=src python3 tools/audit_face_measure.py .work/probe/after
    PYTHONPATH=src python3 tools/audit_face_measure.py .work/probe/* --json

Band geometry comes from the template recorded in the render directory, so the
numbers stay correct when a template changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from streamslice.face_tracking import DEFAULT_MODEL_PATH
from streamslice.templates import load_template

LOGGER = logging.getLogger(__name__)

# A face this far from the centre of its band reads as a framing mistake.
CENTRE_TOLERANCE = 0.08


@dataclass(frozen=True, slots=True)
class Measurement:
    still: str
    band: str
    found: bool
    confidence: float = 0.0
    centre_x: float = 0.0
    centre_y: float = 0.0
    offset_x: float = 0.0
    centred: bool = False
    clipped: str = ""
    band_px: str = ""


def _detect(image) -> tuple[float, float, float, float, float] | None:
    height, width = image.shape[:2]
    if width < 32 or height < 32:
        return None
    detector = cv2.FaceDetectorYN.create(
        str(DEFAULT_MODEL_PATH), "", (width, height), 0.5, 0.3, 5000
    )
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: row[-1])
    return (
        float(best[0]),
        float(best[1]),
        float(best[2]),
        float(best[3]),
        float(best[-1]),
    )


def _bands(clip_dir: Path, config: dict) -> list[tuple[str, float, float]]:
    """Return (source, top, bottom) fractions for the layout that was rendered."""
    plan_file = clip_dir / "edit-plan.json"
    if not plan_file.is_file():
        return [("full", 0.0, 1.0)]
    payload = json.loads(plan_file.read_text(encoding="utf-8"))
    template = load_template(payload.get("template", "classic-split"), config)
    cuts = payload.get("plan", {}).get("cuts") or [{}]
    spec = template.layout(str(cuts[0].get("layout") or "split"))
    bands: list[tuple[str, float, float]] = []
    offset = 0.0
    for band in spec.bands:
        bands.append((band.source, offset, offset + band.height))
        offset += band.height
    return bands


def measure_dir(clip_dir: Path, config: dict) -> list[Measurement]:
    bands = _bands(clip_dir, config)
    results: list[Measurement] = []
    for still in sorted(clip_dir.glob("still-*.png")):
        image = cv2.imread(str(still))
        if image is None:
            results.append(Measurement(still.name, "?", False))
            continue
        height = image.shape[0]
        # Only bands that can contain the streamer are worth judging; a face
        # detected in a gameplay band belongs to the game, not the streamer.
        for source, top, bottom in bands:
            if source == "gameplay":
                continue
            crop = image[int(top * height) : int(bottom * height), :, :]
            band_h, band_w = crop.shape[:2]
            found = _detect(crop)
            if found is None:
                results.append(Measurement(still.name, source, False, band_px=f"{band_w}x{band_h}"))
                continue
            x, y, face_w, face_h, confidence = found
            centre_x = (x + face_w / 2.0) / band_w
            centre_y = (y + face_h / 2.0) / band_h
            clipped = "".join(
                flag
                for flag, hit in (
                    ("L", x < 1),
                    ("R", x + face_w > band_w - 1),
                    ("T", y < 1),
                    ("B", y + face_h > band_h - 1),
                )
                if hit
            )
            results.append(
                Measurement(
                    still=still.name,
                    band=source,
                    found=True,
                    confidence=round(confidence, 3),
                    centre_x=round(centre_x, 4),
                    centre_y=round(centre_y, 4),
                    offset_x=round(centre_x - 0.5, 4),
                    centred=abs(centre_x - 0.5) <= CENTRE_TOLERANCE,
                    clipped=clipped,
                    band_px=f"{band_w}x{band_h}",
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="render directories containing still-*.png")
    parser.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    config: dict = {}
    payload: dict[str, list[dict]] = {}
    failures = 0
    header = f"{'render':<26} {'band':<9} {'still':<13} {'cx':<8} {'off':<8} {'clip':<5} conf"
    if not args.json:
        print(header)
        print("-" * len(header))

    for raw in args.dirs:
        clip_dir = Path(raw)
        if not clip_dir.is_dir():
            continue
        results = measure_dir(clip_dir, config)
        payload[clip_dir.name] = [asdict(item) for item in results]
        for item in results:
            if not item.found or not item.centred or item.clipped:
                failures += 1
            if args.json:
                continue
            if item.found:
                print(
                    f"{clip_dir.name[:25]:<26} {item.band:<9} {item.still:<13} "
                    f"{item.centre_x:<8.3f} {item.offset_x:<+8.3f} "
                    f"{item.clipped or '-':<5} {item.confidence:.2f}"
                )
            else:
                print(
                    f"{clip_dir.name[:25]:<26} {item.band:<9} {item.still:<13} "
                    f"{'NO FACE':<8} {'':<8} {'':<5}"
                )

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        total = sum(len(items) for items in payload.values())
        print(f"\nmeasurements: {total}, off-target: {failures} (tolerance ±{CENTRE_TOLERANCE})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
