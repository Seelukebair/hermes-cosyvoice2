#!/usr/bin/env python3
"""Small command adapter for local Whisper transcription."""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default=os.environ.get("COSYVOICE_WHISPER_MODEL", "base"))
    args = parser.parse_args()

    import whisper

    model = whisper.load_model(args.model, device="cpu")
    result = model.transcribe(args.input, fp16=False)
    text = " ".join(str(result.get("text") or "").split())
    print(json.dumps({"text": text, "model": args.model}, ensure_ascii=True))
    return 0 if text else 2


if __name__ == "__main__":
    raise SystemExit(main())
