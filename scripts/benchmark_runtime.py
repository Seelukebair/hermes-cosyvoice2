#!/usr/bin/env python3
"""Measure CosyVoice HTTP latency with fixed, non-private text fixtures."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import urllib.request
import wave
from pathlib import Path


FIXTURES = {
    "short": "Systems are online and ready.",
    "medium": (
        "The workshop is secure, the cameras are reporting normally, and all "
        "local services are ready for your next instruction."
    ),
    "long": (
        "I checked the local service path, confirmed the active voice profile, "
        "and reviewed the current system load. The response pipeline is healthy. "
        "I will continue watching the relevant services and report a specific "
        "failure if any part of the request path stops responding."
    ),
}


def request_once(base_url: str, path: str, text: str, speed: float) -> dict:
    payload = json.dumps(
        {"text": text, "voice": "default", "speed": speed, "instruct": ""}
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=330) as response:
        first = response.read(45)
        first_byte = time.monotonic()
        remainder = response.read()
        finished = time.monotonic()
        content_type = response.headers.get_content_type()
        status = response.status
    data = first + remainder
    duration = None
    if data.startswith(b"RIFF") and len(data) >= 44:
        try:
            with wave.open(io.BytesIO(data), "rb") as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
        except (EOFError, wave.Error):
            # Streaming WAV headers intentionally carry zero frames.
            sample_rate = int.from_bytes(data[24:28], "little")
            channels = int.from_bytes(data[22:24], "little")
            width = int.from_bytes(data[34:36], "little") // 8
            if sample_rate and channels and width:
                duration = (len(data) - 44) / (sample_rate * channels * width)
        if duration == 0 and len(data) > 44:
            sample_rate = int.from_bytes(data[24:28], "little")
            channels = int.from_bytes(data[22:24], "little")
            width = int.from_bytes(data[34:36], "little") // 8
            if sample_rate and channels and width:
                duration = (len(data) - 44) / (sample_rate * channels * width)
    return {
        "status": status,
        "content_type": content_type,
        "first_audio_seconds": round(first_byte - started, 4),
        "total_seconds": round(finished - started, 4),
        "bytes": len(data),
        "audio_seconds": round(duration, 4) if duration is not None else None,
        "rtf": round((finished - started) / duration, 4) if duration else None,
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for fixture in FIXTURES:
        selected = [row for row in rows if row["fixture"] == fixture]
        summary[fixture] = {
            key: round(statistics.median(row[key] for row in selected), 4)
            for key in ("first_audio_seconds", "total_seconds", "rtf")
            if all(row[key] is not None for row in selected)
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:17870")
    parser.add_argument("--path", default="/synthesize")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    rows = []
    for fixture, text in FIXTURES.items():
        for repetition in range(1, args.repeats + 1):
            result = request_once(args.base_url, args.path, text, args.speed)
            rows.append({"fixture": fixture, "repetition": repetition, **result})
    report = {
        "base_url": args.base_url,
        "path": args.path,
        "repeats": args.repeats,
        "speed": args.speed,
        "results": rows,
        "median": summarize(rows),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
