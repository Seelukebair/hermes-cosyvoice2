#!/usr/bin/env python3
"""Measure first PCM byte and total time from the local streaming endpoint."""

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:17870/synthesize-stream")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    for index in range(args.runs):
        body = json.dumps(
            {"text": "Systems are online and ready.", "voice": "default", "speed": 1.2}
        ).encode()
        request = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"}
        )
        started = time.monotonic()
        first_audio = None
        byte_count = 0
        with urllib.request.urlopen(request, timeout=90) as response:
            while chunk := response.read(4096):
                byte_count += len(chunk)
                if byte_count > 44 and first_audio is None:
                    first_audio = time.monotonic() - started
        print(
            json.dumps(
                {
                    "run": index + 1,
                    "first_audio_s": round(first_audio or 0, 3),
                    "total_s": round(time.monotonic() - started, 3),
                    "bytes": byte_count,
                }
            )
        )


if __name__ == "__main__":
    main()
