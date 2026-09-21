#!/usr/bin/env python3
"""Report basic PCM validity metrics for generated TTS samples."""

from __future__ import annotations

import argparse
import json
import wave

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        with wave.open(path, "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
        if width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        result = {
            "path": path,
            "duration_seconds": round(len(samples) / channels / rate, 3),
            "sample_rate": rate,
            "channels": channels,
            "peak": round(float(np.max(np.abs(samples))), 6),
            "rms": round(float(np.sqrt(np.mean(samples ** 2))), 6),
            "dc_offset": round(float(np.mean(samples)), 6),
            "clipped_samples": int(np.count_nonzero(np.abs(samples) >= 0.9999)),
            "finite": bool(np.isfinite(samples).all()),
        }
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
