#!/usr/bin/env python3
"""Install the pinned CosyVoice demo asset as a safe bootstrap profile."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path


PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-wav", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-text")
    prompt.add_argument("--prompt-text-base64")
    parser.add_argument("--profile-id", default="bundled-stock")
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.reference_wav.is_file():
        raise SystemExit(f"reference WAV not found: {args.reference_wav}")
    prompt_text = args.prompt_text
    if args.prompt_text_base64:
        prompt_text = base64.b64decode(args.prompt_text_base64, validate=True).decode("utf-8")
    if not prompt_text or not prompt_text.strip():
        raise SystemExit("prompt text must not be empty")
    if not PROFILE_ID.fullmatch(args.profile_id):
        raise SystemExit("profile id must match [a-z0-9][a-z0-9-]{0,63}")

    state_path = args.root / "state.json"
    profile_dir = args.root / "profiles" / args.profile_id
    candidate_dir = args.root / "candidates" / args.profile_id
    if candidate_dir.exists():
        raise SystemExit(f"refusing ambiguous candidate/profile id: {args.profile_id}")

    args.backup_dir.mkdir(parents=True, exist_ok=False)
    if state_path.exists():
        shutil.copy2(state_path, args.backup_dir / "state.json")
    if profile_dir.exists():
        shutil.copytree(profile_dir, args.backup_dir / args.profile_id)

    profile_dir.mkdir(parents=True, exist_ok=True)
    with args.reference_wav.open("rb") as source:
        write_atomic(profile_dir / "reference.wav", source.read())

    now = int(time.time())
    metadata = {
        "id": args.profile_id,
        "name": "Bundled stock voice",
        "status": "saved",
        "created_at": now,
        "reference_wav": "reference.wav",
        "prompt_text": prompt_text.strip(),
        "delivery_prompt": "",
        "source": {
            "kind": "bundled_test_asset",
            "path": str(args.reference_wav),
        },
        "transcript": {
            "text": prompt_text.strip(),
            "verified": True,
            "verification_source": "pinned_cosyvoice_service_configuration",
        },
        "validation": {
            "status": "bundled_stock_asset",
            "reason": "Bootstrap profile from the pinned CosyVoice demo asset.",
        },
    }
    write_atomic(
        profile_dir / "profile.json",
        (json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(),
    )

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    state["default_profile"] = args.profile_id
    state["updated_at"] = now
    state.setdefault("session_profile", None)
    state.setdefault("session_candidate", False)
    write_atomic(
        state_path,
        (json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps({"installed": args.profile_id, "backup": str(args.backup_dir)}))


if __name__ == "__main__":
    main()
