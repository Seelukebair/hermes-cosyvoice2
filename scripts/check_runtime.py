#!/usr/bin/env python3
"""Read-only health and revision check for the packaged CosyVoice2 service."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:17870/health")
    parser.add_argument("--expect-source")
    parser.add_argument("--expect-model")
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(args.url, timeout=5) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2
    checks = {
        "ready": payload.get("ready") is True,
        "source_revision": not args.expect_source or payload.get("source_revision") == args.expect_source,
        "model_revision": not args.expect_model or payload.get("model_revision") == args.expect_model,
    }
    result = {"ok": all(checks.values()), "checks": checks, "health": payload}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

