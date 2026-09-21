"""Non-GPU tests for the Jarvis CosyVoice HA bridge."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest


class BackendServer(ThreadingHTTPServer):
    """Small local backend used to exercise forwarding without synthesis."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), BackendHandler)
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_requests = False
        self.fail_next = False
        self.active = 0
        self.max_active = 0
        self.payloads: list[dict] = []


class BackendHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _enter(self) -> None:
        with self.server.lock:
            self.server.active += 1
            self.server.max_active = max(self.server.max_active, self.server.active)
            self.server.started.set()

    def _leave(self) -> None:
        with self.server.lock:
            self.server.active -= 1

    def do_GET(self) -> None:  # noqa: N802
        self._enter()
        try:
            if self.server.fail_next:
                self.server.fail_next = False
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if self.server.block_requests:
                self.server.release.wait(3)
            data = b'{"ready":false}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            self._leave()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.server.lock:
            self.server.payloads.append(payload)
        data = b'{"mocked":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["JARVIS_COSYVOICE_PROXY_TOKEN"] = "test-token"
        os.environ["JARVIS_COSYVOICE_MAX_CONCURRENT_REQUESTS"] = "1"
        path = Path(__file__).parents[1] / "jarvis-cosyvoice-ha-proxy.py"
        spec = importlib.util.spec_from_file_location("proxy", path)
        assert spec and spec.loader
        cls.proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proxy)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.profile_root = Path(self.temporary.name) / "voice_profiles"
        self.profile_root.mkdir()
        self.proxy.PROFILE_STATE = self.profile_root / "state.json"

    def _profile(
        self,
        collection: str,
        profile_id: str,
        *,
        status: str = "saved",
        verified: bool = True,
        accepted: bool | None = None,
    ) -> None:
        directory = self.profile_root / collection / profile_id
        directory.mkdir(parents=True)
        (directory / "reference.wav").write_bytes(b"RIFF")
        (directory / "profile.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "prompt_text": "reference words",
                    "transcript": {
                        "text": "reference words",
                        "verified": verified,
                        **({"accepted": accepted} if accepted is not None else {}),
                    },
                }
            ),
            encoding="utf-8",
        )

    def _state(self, **values) -> None:
        self.proxy.PROFILE_STATE.write_text(json.dumps(values), encoding="utf-8")

    def _start_backend(self) -> BackendServer:
        backend = BackendServer()
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()

        def stop() -> None:
            backend.release.set()
            backend.shutdown()
            backend.server_close()
            thread.join(2)

        self.addCleanup(stop)
        self.proxy.BACKEND_URL = f"http://127.0.0.1:{backend.server_port}"
        return backend

    def _start_proxy(self):
        server = self.proxy.Server(("127.0.0.1", 0), self.proxy.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop() -> None:
            server.shutdown()
            server.server_close()
            thread.join(2)

        self.addCleanup(stop)
        return server

    @staticmethod
    def _request(server, method: str, path: str, body: dict | None = None):
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=4
        )
        encoded = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": "Bearer test-token"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.headers), json.loads(data) if data else None
        finally:
            connection.close()

    def test_bearer_auth(self) -> None:
        self.assertTrue(self.proxy._authorized("Bearer test-token"))
        self.assertFalse(self.proxy._authorized("Bearer wrong"))
        self.assertFalse(self.proxy._authorized(None))

    def test_payload_limit_is_bounded(self) -> None:
        self.assertEqual(65536, self.proxy.MAX_BODY)

    def test_default_backend_is_loopback(self) -> None:
        self.assertEqual("http://127.0.0.1:17870", self.proxy.DEFAULT_BACKEND_URL)

    def test_unaccepted_candidate_is_not_eligible(self) -> None:
        backend = self._start_backend()
        server = self._start_proxy()
        self._profile("candidates", "preview-one", status="preview", verified=False)
        self._state(
            session_profile="preview-one",
            session_candidate=True,
            default_profile=None,
        )

        status, _, payload = self._request(
            server, "POST", "/synthesize", {"text": "test"}
        )

        self.assertEqual(503, status)
        self.assertEqual("voice_profile_unavailable", payload["error"])
        self.assertEqual([], backend.payloads)

    def test_unverified_saved_profile_is_not_eligible(self) -> None:
        self._profile("profiles", "unverified", verified=False)
        self._state(
            session_profile=None,
            session_candidate=False,
            default_profile="unverified",
        )
        self.assertIsNone(self.proxy._selected_profile())

    def test_accepted_automatic_transcript_is_eligible(self) -> None:
        self._profile("profiles", "automatic", verified=False, accepted=True)
        self._state(session_profile=None, session_candidate=False, default_profile="automatic")
        self.assertEqual("automatic", self.proxy._selected_profile())

    def test_candidate_is_ignored_and_saved_default_is_forwarded_explicitly(self) -> None:
        backend = self._start_backend()
        server = self._start_proxy()
        self._profile("candidates", "preview-one", status="preview", verified=False)
        self._profile("profiles", "saved-one")
        self._state(
            session_profile="preview-one",
            session_candidate=True,
            default_profile="saved-one",
        )

        status, _, _ = self._request(
            server, "POST", "/synthesize", {"text": "test"}
        )

        self.assertEqual(200, status)
        self.assertEqual("saved-one", backend.payloads[-1]["voice"])

    def test_accepted_session_profile_is_forwarded_explicitly(self) -> None:
        backend = self._start_backend()
        server = self._start_proxy()
        self._profile("profiles", "accepted-session")
        self._profile("profiles", "saved-default")
        self._state(
            session_profile="accepted-session",
            session_candidate=False,
            default_profile="saved-default",
        )

        status, _, _ = self._request(
            server,
            "POST",
            "/synthesize",
            {"text": "test", "voice": "caller-choice"},
        )

        self.assertEqual(200, status)
        self.assertEqual("accepted-session", backend.payloads[-1]["voice"])

    def test_concurrency_saturation_fails_fast_and_releases_capacity(self) -> None:
        backend = self._start_backend()
        backend.block_requests = True
        server = self._start_proxy()
        results: dict[str, tuple[int, dict, dict | None]] = {}
        first = threading.Thread(
            target=lambda: results.setdefault(
                "first", self._request(server, "GET", "/health")
            )
        )
        first.start()
        self.assertTrue(backend.started.wait(1), "first request did not reach backend")

        status, headers, payload = self._request(server, "GET", "/health")
        self.assertEqual(503, status)
        self.assertEqual("1", headers["Retry-After"])
        self.assertEqual("proxy_capacity_exceeded", payload["error"])
        self.assertEqual(1, backend.max_active)

        backend.release.set()
        first.join(2)
        self.assertFalse(first.is_alive())
        status, _, _ = self._request(server, "GET", "/health")
        self.assertEqual(200, status, "capacity was not released")

    def test_capacity_is_released_after_backend_failure(self) -> None:
        backend = self._start_backend()
        backend.fail_next = True
        server = self._start_proxy()

        failed_status, _, failed_payload = self._request(server, "GET", "/health")
        recovered_status, _, _ = self._request(server, "GET", "/health")

        self.assertEqual(503, failed_status)
        self.assertEqual("backend_unavailable", failed_payload["error"])
        self.assertEqual(200, recovered_status)

    def test_server_threads_do_not_block_shutdown(self) -> None:
        self.assertTrue(self.proxy.Server.daemon_threads)


if __name__ == "__main__":
    unittest.main()
