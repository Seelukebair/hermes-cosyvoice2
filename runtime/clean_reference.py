#!/usr/bin/env python3
"""Isolate dialogue and conservatively compact long pauses in a voice reference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


MIN_SECONDS = 10.0
MAX_SECONDS = 15.0


def run(command: list[str], timeout: int) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "command failed").strip()[-500:])


def duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--separator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit("input WAV does not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cosyvoice-clean-") as temporary:
        work = Path(temporary)
        run(
            [
                str(args.separator),
                str(args.input),
                "--model_filename", args.model,
                "--model_file_dir", str(args.model_dir),
                "--output_dir", str(work),
                "--output_format", "WAV",
                "--single_stem", "Vocals",
                "--sample_rate", "44100",
                "--use_native_fp16",
            ],
            240,
        )
        stems = sorted(work.glob("*Vocals*.wav"))
        if len(stems) != 1:
            raise RuntimeError(f"expected one vocal stem, found {len(stems)}")

        separated = work / "separated.wav"
        compacted = work / "compacted.wav"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(stems[0]), "-ac", "1", "-ar", "24000",
            "-c:a", "pcm_s16le", str(separated),
        ], 45)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(separated), "-af",
            "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-40dB:"
            "stop_periods=-1:stop_duration=0.35:stop_threshold=-40dB:stop_silence=0.15",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(compacted),
        ], 45)

        selected = compacted if MIN_SECONDS <= duration(compacted) <= MAX_SECONDS else separated
        selected_duration = duration(selected)
        if not MIN_SECONDS <= selected_duration <= MAX_SECONDS:
            raise RuntimeError(f"cleaned duration is outside {MIN_SECONDS}-{MAX_SECONDS} seconds")
        shutil.copy2(selected, args.output)

    print(json.dumps({
        "status": "cleaned",
        "method": "bs_roformer_with_conservative_pause_compaction",
        "duration_seconds": round(selected_duration, 3),
        "pause_compaction_applied": selected == compacted,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
