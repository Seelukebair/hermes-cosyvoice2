#!/usr/bin/env python3
"""Exercise one continuous synthesis session without logging text or secrets."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request


def request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> bytes:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers, method=method),
        timeout=120,
    ) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-env", required=True)
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"missing token environment variable: {args.token_env}")

    base_url = args.url.rstrip("/") + "/synthesis-sessions"
    created = json.loads(
        request(
            base_url,
            token,
            method="POST",
            payload={
                "text": "The authenticated bridge starts one response.",
                "voice": "default",
                "speed": 1.2,
            },
        )
    )
    session_id = created["session_id"]
    audio: dict[str, bytes] = {}
    failure: list[BaseException] = []

    def receive_audio() -> None:
        try:
            audio["wav"] = request(f"{base_url}/{session_id}/audio", token)
        except BaseException as error:  # surfaced on the main thread below
            failure.append(error)

    started = time.monotonic()
    receiver = threading.Thread(target=receive_audio, daemon=True)
    receiver.start()
    time.sleep(0.3)
    request(
        f"{base_url}/{session_id}/text",
        token,
        method="POST",
        payload={"text": "A later sentence stays inside that response."},
    )
    request(f"{base_url}/{session_id}/finish", token, method="POST")
    receiver.join(timeout=120)
    if receiver.is_alive():
        raise SystemExit("audio response did not finish within 120 seconds")
    if failure:
        raise failure[0]

    wav = audio["wav"]
    pcm = wav[44:]
    first_nonzero = next(
        (offset for offset in range(0, len(pcm) - 1, 2) if pcm[offset : offset + 2] != b"\x00\x00"),
        len(pcm),
    )
    result = {
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        "bytes": len(wav),
        "pcm_bytes": max(0, len(wav) - 44),
        "riff_headers": wav.count(b"RIFF"),
        "valid_wav": wav[:4] == b"RIFF" and wav[8:12] == b"WAVE",
        "leading_silence_ms": round((first_nonzero / 2) / 24000 * 1000, 2),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid_wav"] and result["riff_headers"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
