from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from .process import require_binary, run

LOGGER = logging.getLogger(__name__)


def remove_music_if_blocked(video_path: str | Path, clean_video_path: str | Path) -> Path:
    """Strip background music track using Demucs (if PyTorch is available) or FFmpeg speech filter,

    and recombine with the original video track.
    """
    src_video = Path(video_path).resolve()
    dst_video = Path(clean_video_path).resolve()
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    if not src_video.is_file():
        raise FileNotFoundError(f"Source video not found: {src_video}")

    ffmpeg_bin = require_binary("ffmpeg")

    try:
        import torch
        import torchaudio
        import torchaudio.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Starting Demucs AI music removal on %s (device=%s)", src_video.name, device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            temp_audio_in = tmp_dir_path / "original_audio.wav"
            temp_audio_clean = tmp_dir_path / "vocal_clean.wav"

            # 1. Extract audio to 44.1kHz Stereo WAV for Demucs
            cmd_extract = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src_video),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                str(temp_audio_in),
            ]
            run(cmd_extract)

            # 2. Load audio and separate vocals via Demucs
            waveform, sample_rate = torchaudio.load(str(temp_audio_in))
            if sample_rate != 44100:
                waveform = F.resample(waveform, sample_rate, 44100)
                sample_rate = 44100

            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB
            model = bundle.get_model().to(device)
            model.eval()

            waveform_in = waveform.unsqueeze(0).to(device)
            ref = waveform_in.mean(0)
            mean = ref.mean()
            std = ref.std()
            normalized_in = (waveform_in - mean) / (std + 1e-8)

            with torch.inference_mode():
                sources = model(normalized_in)
                sources = (sources * (std + 1e-8)) + mean

            vocals = sources[0, 3].detach().cpu()
            torchaudio.save(str(temp_audio_clean), vocals, sample_rate)

            # 3. Mux cleaned vocal audio with original video track
            cmd_mux = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src_video),
                "-i",
                str(temp_audio_clean),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(dst_video),
            ]
            run(cmd_mux)

        LOGGER.info("Cleaned video created via Demucs: %s", dst_video)
        return dst_video

    except ImportError:
        LOGGER.warning(
            "PyTorch/Torchaudio not available in environment, using FFmpeg vocal isolation filter"
        )
        cmd_ffmpeg_clean = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src_video),
            "-af",
            "highpass=f=120,lowpass=f=7500,afftdn=nf=-25,dynaudnorm=f=150:g=15",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(dst_video),
        ]
        run(cmd_ffmpeg_clean)
        LOGGER.info("Cleaned video created via FFmpeg audio filter: %s", dst_video)
        return dst_video
