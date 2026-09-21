#!/usr/bin/env python3
"""Update an HA Assist pipeline through its supported WebSocket API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import websocket


def command(ws: websocket.WebSocket, message_id: int, payload: dict) -> dict:
    ws.send(json.dumps({"id": message_id, **payload}))
    while True:
        response = json.loads(ws.recv())
        if response.get("id") == message_id:
            if not response.get("success"):
                raise RuntimeError(json.dumps(response.get("error", response)))
            return response.get("result")


def update_payload(pipeline: dict) -> dict:
    fields = {
        "conversation_engine",
        "conversation_language",
        "language",
        "name",
        "stt_engine",
        "stt_language",
        "tts_engine",
        "tts_language",
        "tts_voice",
        "wake_word_entity",
        "wake_word_id",
        "prefer_local_intents",
        "acknowledge_media_id",
    }
    return {
        "type": "assist_pipeline/pipeline/update",
        "pipeline_id": pipeline["id"],
        **{key: pipeline[key] for key in fields if key in pipeline},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ha-url")
    parser.add_argument("--token")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--tts-engine", required=True)
    parser.add_argument("--tts-voice", default="default")
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    env_values: dict[str, str] = {}
    if args.env_file:
        for raw in args.env_file.read_text(encoding="utf-8").splitlines():
            if raw and not raw.lstrip().startswith("#") and "=" in raw:
                key, value = raw.split("=", 1)
                env_values[key] = value
    args.ha_url = args.ha_url or os.environ.get("HASS_URL") or env_values.get("HASS_URL")
    args.token = (
        args.token
        or os.environ.get("HASS_TOKEN")
        or env_values.get("HASS_TOKEN")
        or env_values.get("JARVISEXT_HOME_ASSISTANT_TOKEN")
    )
    if not args.ha_url or not args.token:
        raise SystemExit("HASS_URL and HASS_TOKEN are required")
    if args.backup.exists():
        raise SystemExit(f"backup already exists: {args.backup}")

    parsed = urlsplit(args.ha_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = urlunsplit((scheme, parsed.netloc, "/api/websocket", "", ""))
    ws = websocket.create_connection(ws_url, timeout=30)
    try:
        if json.loads(ws.recv()).get("type") != "auth_required":
            raise RuntimeError("Home Assistant did not request authentication")
        ws.send(json.dumps({"type": "auth", "access_token": args.token}))
        if json.loads(ws.recv()).get("type") != "auth_ok":
            raise RuntimeError("Home Assistant authentication failed")

        listing = command(ws, 1, {"type": "assist_pipeline/pipeline/list"})
        items = listing.get("pipelines", listing.get("items", listing)) if isinstance(listing, dict) else listing
        original = next((item for item in items if item.get("id") == args.pipeline_id), None)
        if original is None:
            raise RuntimeError(f"pipeline not found: {args.pipeline_id}")
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(json.dumps(original, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(args.backup, 0o600)

        desired = dict(original, tts_engine=args.tts_engine, tts_voice=args.tts_voice)
        command(ws, 2, update_payload(desired))
        verified_listing = command(ws, 3, {"type": "assist_pipeline/pipeline/list"})
        verified_items = (
            verified_listing.get("pipelines", verified_listing.get("items", verified_listing))
            if isinstance(verified_listing, dict)
            else verified_listing
        )
        verified = next((item for item in verified_items if item.get("id") == args.pipeline_id), None)
        if not verified or verified.get("tts_engine") != args.tts_engine or verified.get("tts_voice") != args.tts_voice:
            command(ws, 4, update_payload(original))
            raise RuntimeError("pipeline read-back mismatch; original restored")
        print(json.dumps({"pipeline_id": args.pipeline_id, "tts_engine": verified["tts_engine"], "tts_voice": verified["tts_voice"]}))
    finally:
        ws.close()


if __name__ == "__main__":
    main()
