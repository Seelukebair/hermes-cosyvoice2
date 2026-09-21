#!/usr/bin/env python3
"""Authenticated, HA-only bridge to Jarvis's loopback CosyVoice API."""

from __future__ import annotations

import hmac
import http.client
import json
import os
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LISTEN_HOST = os.environ.get("JARVIS_COSYVOICE_PROXY_HOST", "192.168.1.10")
LISTEN_PORT = int(os.environ.get("JARVIS_COSYVOICE_PROXY_PORT", "17871"))
DEFAULT_BACKEND_URL = "http://127.0.0.1:17870"
BACKEND_URL = os.environ.get("JARVIS_COSYVOICE_BACKEND_URL", DEFAULT_BACKEND_URL)
ALLOWED_CLIENT = os.environ.get("JARVIS_COSYVOICE_ALLOWED_CLIENT", "192.168.1.11")
TOKEN = os.environ["JARVIS_COSYVOICE_PROXY_TOKEN"]
PROFILE_STATE = Path(
    os.environ.get(
        "JARVIS_COSYVOICE_PROFILE_STATE",
        "/srv/cosyvoice2/data/voice_profiles/state.json",
    )
)
MAX_BODY = 65536
BACKEND_TIMEOUT = 330
STREAM_READ_SIZE = 16 * 1024
MAX_ERROR_BODY = 4096
PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _positive_env_int(name: str, default: int) -> int:
    """Read a positive integer or fail closed during service startup."""
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


MAX_CONCURRENT_REQUESTS = _positive_env_int(
    "JARVIS_COSYVOICE_MAX_CONCURRENT_REQUESTS", 2
)


def _authorized(header: str | None) -> bool:
    """Compare credentials without timing-dependent string equality."""
    if not header or not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:], TOKEN)


def _saved_profile_exists(profile_id: str) -> bool:
    """Match the voice manager's accepted saved-profile contract."""
    if not PROFILE_ID.fullmatch(profile_id):
        return False
    root = PROFILE_STATE.parent
    # The backend resolves candidate directories before saved profiles. Refuse
    # an ambiguous id so a preview can never shadow the accepted profile.
    if (root / "candidates" / profile_id).exists():
        return False
    directory = root / "profiles" / profile_id
    try:
        metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    transcript = metadata.get("transcript")
    if not isinstance(transcript, dict):
        transcript = {}
    prompt_text = str(metadata.get("prompt_text") or transcript.get("text") or "").strip()
    return (
        metadata.get("status") == "saved"
        and transcript.get("accepted", transcript.get("verified")) is True
        and (directory / "reference.wav").is_file()
        and bool(prompt_text)
    )


def _selected_profile() -> str | None:
    """Return only an accepted, persisted session/default profile id."""
    try:
        state = json.loads(PROFILE_STATE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    session_id = str(state.get("session_profile") or "")
    default_id = str(state.get("default_profile") or "")
    if (
        session_id
        and state.get("session_candidate") is False
        and _saved_profile_exists(session_id)
    ):
        return session_id
    if default_id and _saved_profile_exists(default_id):
        return default_id
    return None


def _profile_configured() -> bool:
    """Return whether Home Assistant has an eligible saved profile."""
    return _selected_profile() is not None


class Server(ThreadingHTTPServer):
    """Threaded HTTP server with bounded backend-request capacity."""

    daemon_threads = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_slots = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


class Handler(BaseHTTPRequestHandler):
    """Forward only the bounded CosyVoice endpoints."""

    server_version = "JarvisCosyVoiceBridge/1.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        # Never log headers, request bodies, spoken text, or credentials.
        print(f"{self.client_address[0]} {fmt % args}", flush=True)

    def _allowed(self) -> bool:
        return self.client_address[0] in {ALLOWED_CLIENT, LISTEN_HOST, "127.0.0.1"}

    def _authenticate(self) -> bool:
        if not self._allowed():
            self.send_error(403)
            return False
        if not _authorized(self.headers.get("Authorization")):
            self.send_error(401)
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._authenticate():
            return
        if self.path == "/proxy-health":
            self._json(
                200,
                {
                    "status": "ok",
                    "backend": "cosyvoice2",
                    "profile_configured": _profile_configured(),
                    "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
                },
            )
            return
        if self.path == "/health":
            self._forward("GET", "/health", None)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authenticate():
            return
        if self.path not in {"/synthesize", "/synthesize-stream"}:
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if length <= 0 or length > MAX_BODY:
            self.send_error(413)
            return
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            self.send_error(400)
            return
        profile_id = _selected_profile()
        if profile_id is None:
            self._json(
                503,
                {
                    "error": "voice_profile_unavailable",
                    "detail": "No shared CosyVoice profile is configured",
                },
            )
            return
        # Resolve the shared selector here and forward the saved id explicitly.
        # This prevents the backend's one-shot preview selector from affecting HA.
        payload["voice"] = profile_id
        body = json.dumps(payload, separators=(",", ":")).encode()
        if self.path == "/synthesize-stream":
            self._forward_stream("POST", "/synthesize-stream", body)
            return
        self._forward("POST", "/synthesize", body)

    def _forward(self, method: str, path: str, body: bytes | None) -> None:
        if not self.server.proxy_slots.acquire(blocking=False):
            self._json(
                503,
                {
                    "error": "proxy_capacity_exceeded",
                    "detail": "Concurrent proxy request limit reached",
                },
                {"Retry-After": "1"},
            )
            return
        target = urlsplit(BACKEND_URL)
        connection = None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            connection = http.client.HTTPConnection(
                target.hostname, target.port or 80, timeout=BACKEND_TIMEOUT
            )
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            self.send_response(response.status)
            self.send_header(
                "Content-Type", response.getheader("Content-Type", "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (OSError, TimeoutError, http.client.HTTPException) as err:
            self._json(503, {"error": "backend_unavailable", "detail": type(err).__name__})
        finally:
            if connection is not None:
                connection.close()
            self.close_connection = True
            self.server.proxy_slots.release()

    def _forward_stream(self, method: str, path: str, body: bytes) -> None:
        """Pass a backend audio stream through without response buffering.

        ``http.client`` de-chunks an upstream HTTP/1.1 response as it is read.
        Re-chunking the decoded bytes is therefore required for the client-facing
        HTTP/1.1 response. The bounded slot remains held until the reader or
        writer ends, which keeps slow clients from creating unbounded work.
        """
        if not self.server.proxy_slots.acquire(blocking=False):
            self._json(
                503,
                {
                    "error": "proxy_capacity_exceeded",
                    "detail": "Concurrent proxy request limit reached",
                },
                {"Retry-After": "1"},
            )
            return

        target = urlsplit(BACKEND_URL)
        connection = None
        stream_started = False
        try:
            connection = http.client.HTTPConnection(
                target.hostname, target.port or 80, timeout=BACKEND_TIMEOUT
            )
            connection.request(
                method,
                path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                self._forward_stream_error(response)
                return

            self.send_response(200)
            self.send_header(
                "Content-Type", response.getheader("Content-Type", "audio/wav")
            )
            # The stream may end early. HTTP chunk framing makes that explicit
            # without promising a final byte count that the backend has not made.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            stream_started = True

            # ``read1`` returns as soon as the upstream socket has a chunk. A
            # normal ``read(size)`` may wait to fill its requested buffer and
            # would quietly turn a small first PCM chunk into buffered output.
            while data := response.read1(STREAM_READ_SIZE):
                try:
                    self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Closing the upstream connection in finally stops reading
                    # as soon as the backend notices the cancelled request.
                    self.close_connection = True
                    return
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                return
        except (OSError, TimeoutError, http.client.HTTPException) as err:
            # Do not attempt a JSON response once successful stream headers were
            # written: that would corrupt audio already handed to Home Assistant.
            if not stream_started:
                self._json(503, {"error": "backend_unavailable", "detail": type(err).__name__})
        finally:
            if connection is not None:
                connection.close()
            self.close_connection = True
            self.server.proxy_slots.release()

    def _forward_stream_error(self, response: http.client.HTTPResponse) -> None:
        """Return a bounded upstream error before any audio bytes are emitted."""
        response.read(MAX_ERROR_BODY + 1)
        data = json.dumps(
            {
                "error": "backend_rejected_request",
                "detail": f"HTTP_{response.status}",
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(response.status)
        self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(
        self, status: int, payload: dict, extra_headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    """Run the bridge."""
    server = Server((LISTEN_HOST, LISTEN_PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
