#!/usr/bin/env python3
"""Atomically register the packaged CosyVoice2 command provider in Hermes."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path

import yaml


def provider(prefix: Path, speed: float) -> dict:
    return {
        "type": "command",
        "command": (
            f"{prefix}/venv/bin/python {prefix}/app/cosyvoice_tts.py "
            f"--input {{input_path}} --output {{output_path}} --voice {{voice}} --speed {speed:.2f}"
        ),
        "output_format": "wav",
        "voice": "default",
        "voice_compatible": False,
        "timeout": 180,
    }


def atomic_yaml(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            yaml.safe_dump(payload, output, sort_keys=False, allow_unicode=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path.home() / ".hermes" / "config.yaml")
    parser.add_argument("--prefix", type=Path, default=Path("/srv/cosyvoice2"))
    parser.add_argument("--speed", type=float, default=1.10)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0.5 <= args.speed <= 2.0:
        raise SystemExit("--speed must be between 0.5 and 2.0")
    if not args.config.is_file():
        raise SystemExit(f"Hermes config not found: {args.config}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    tts = config.setdefault("tts", {})
    tts.setdefault("providers", {})["cosyvoice2"] = provider(args.prefix, args.speed)
    if args.activate:
        tts["provider"] = "cosyvoice2"
    if args.dry_run:
        print(yaml.safe_dump({"tts": tts}, sort_keys=False, allow_unicode=True))
        return 0

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_root = Path(os.environ.get("JARVIS_BACKUP_ROOT", "/srv/jarvis-backups"))
    day = dt.datetime.now().astimezone().strftime("%Y-%m-%d")
    backup_dir = backup_root / day / "cosyvoice2" / f"{stamp}-provider-registration"
    backup_dir.mkdir(parents=True, mode=0o700)
    os.chmod(backup_dir, 0o700)
    backup = backup_dir / "config.yaml"
    shutil.copy2(args.config, backup)
    atomic_yaml(args.config, config)
    print(f"registered cosyvoice2; rollback: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
