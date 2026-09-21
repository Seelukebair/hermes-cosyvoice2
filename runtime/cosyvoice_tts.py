#!/usr/bin/env python3
"""Hermes command-provider adapter for the local CosyVoice2 service."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--url", default="http://127.0.0.1:17870/synthesize")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("input text is empty")
    payload = json.dumps({"text": text, "speed": args.speed, "voice": args.voice}).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            audio = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"CosyVoice HTTP {exc.code}: {detail}") from exc
    if len(audio) <= 44 or audio[:4] != b"RIFF":
        raise RuntimeError("CosyVoice returned invalid WAV data")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(audio)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
