#!/usr/bin/env python3
"""Generate repeated fixture-matched A/B samples for saved voice profiles."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import wave
from pathlib import Path


PROMPTS = (
    "Good evening. The equipment check is complete, and every system is ready for tonight's countdown.",
    "The next song is number seven.",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:17870")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for profile_id in args.profile:
        for repeat in range(1, args.repeats + 1):
            for index, text in enumerate(PROMPTS, start=1):
                request = urllib.request.Request(
                args.url.rstrip("/") + "/synthesize",
                data=json.dumps(
                    {
                        "text": text,
                        "voice": profile_id,
                        "speed": args.speed,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = response.read()
                elapsed_ms = round((time.monotonic() - started) * 1000, 2)
                output = args.output_dir / f"{profile_id}-r{repeat}-p{index}.wav"
                output.write_bytes(payload)
                with wave.open(str(output), "rb") as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                results.append(
                    {
                        "profile_id": profile_id,
                        "repeat": repeat,
                        "prompt": index,
                        "elapsed_ms": elapsed_ms,
                        "duration_seconds": round(duration, 3),
                        "bytes": len(payload),
                        "output": str(output),
                    }
                )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
