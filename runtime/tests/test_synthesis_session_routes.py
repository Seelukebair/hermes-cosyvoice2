from __future__ import annotations

import asyncio
import io
import os
import sys
import types
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

# The server imports this pinned upstream package at module load. Route tests
# replace model work below, so only its import shape is needed on CPU CI.
if "cosyvoice.cli.cosyvoice" not in sys.modules:
    cosyvoice = types.ModuleType("cosyvoice")
    cli = types.ModuleType("cosyvoice.cli")
    cosyvoice_module = types.ModuleType("cosyvoice.cli.cosyvoice")
    cosyvoice_module.AutoModel = object
    sys.modules.update(
        {
            "cosyvoice": cosyvoice,
            "cosyvoice.cli": cli,
            "cosyvoice.cli.cosyvoice": cosyvoice_module,
        }
    )

import cosyvoice_server as server
from streaming_audio import pcm16_bytes, streaming_wav_header


class FakeSpeech:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.samples


class FakeProfiles:
    def __init__(self, *_):
        self.prompt = SimpleNamespace(profile_id="pinned", instruct="", consume_preview=True)
        self.resolve_calls = 0
        self.consume_calls = 0

    def resolve(self, _voice):
        self.resolve_calls += 1
        return self.prompt

    def resolve_saved_default(self):
        return None

    def consume(self, prompt):
        self.assert_prompt(prompt)
        self.consume_calls += 1

    def assert_prompt(self, prompt):
        if prompt is not self.prompt:
            raise AssertionError("session did not retain the resolved profile")


class SynthesisSessionRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_buffered_wav_includes_configured_leading_silence(self):
        payload = server.wav_bytes(
            np.array([0.5, -0.5], dtype=np.float32),
            24000,
            leading_silence_seconds=0.25,
        )
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
        self.assertEqual(b"\x00\x00" * 6000, pcm[:12000])
        self.assertEqual(
            pcm16_bytes(np.array([0.5, -0.5], dtype=np.float32)), pcm[12000:]
        )

    def make_app(self):
        profiles = FakeProfiles()
        model = SimpleNamespace(sample_rate=24000)

        def conditioned_chunks(_model, text, _conditioning, _speed, *, stream=False):
            self.assertTrue(stream)
            samples = {"first.": [0.0, 0.5], "second.": [-0.5, 1.0]}[text]
            yield {"tts_speech": FakeSpeech(samples)}

        environment = {
            "COSYVOICE_WARMUP_ENABLED": "false",
            "COSYVOICE_PROFILE_WARMER_ENABLED": "false",
            "COSYVOICE_SYNTHESIS_SESSION_MAX_SESSIONS": "1",
            "COSYVOICE_SYNTHESIS_SESSION_MAX_QUEUE_SENTENCES": "2",
            "COSYVOICE_SYNTHESIS_SESSION_MAX_TEXT_BYTES": "64",
            "COSYVOICE_SYNTHESIS_SESSION_IDLE_TIMEOUT_SECONDS": "30",
            "COSYVOICE_SYNTHESIS_SESSION_TOTAL_TIMEOUT_SECONDS": "60",
            "COSYVOICE_OUTPUT_LEADING_SILENCE_SECONDS": "0.25",
        }
        args = SimpleNamespace(
            model_dir=Path("model"),
            profile_root=Path("profiles"),
            prompt_wav=Path("prompt.wav"),
            prompt_text="prompt",
            model_revision="test-model",
            source_revision="test-source",
        )
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, environment, clear=False))
        stack.enter_context(patch.object(server, "load_model", return_value=(model, {})))
        stack.enter_context(patch.object(server, "VoiceProfileRegistry", return_value=profiles))
        stack.enter_context(
            patch.object(server, "_conditioning", return_value=("key", "zero_shot", lambda: {}))
        )
        stack.enter_context(patch.object(server, "_conditioned_chunks", side_effect=conditioned_chunks))
        return stack, server.create_app(args), profiles

    async def test_routes_emit_one_header_and_reuse_the_pinned_profile(self):
        stack, app, profiles = self.make_app()
        with stack:
            async with app.router.lifespan_context(app):
                routes = {route.path: route.endpoint for route in app.routes}
                create = routes["/synthesis-sessions"]
                enqueue = routes["/synthesis-sessions/{session_id}/text"]
                finish = routes["/synthesis-sessions/{session_id}/finish"]
                audio = routes["/synthesis-sessions/{session_id}/audio"]

                created = create(
                    server.SynthesisSessionCreateRequest(text="first.", voice="voice", speed=1.0)
                )
                session_id = created["session_id"]
                self.assertTrue(session_id)
                self.assertEqual({"status": "queued"}, enqueue(session_id, server.SynthesisSessionTextRequest(text="second.")))
                self.assertEqual({"status": "finishing"}, finish(session_id))

                response = audio(session_id)
                payload = b"".join([part async for part in response.body_iterator])
                self.assertEqual(1, payload.count(streaming_wav_header(24000)))
                self.assertEqual(streaming_wav_header(24000), payload[:44])
                self.assertEqual(
                    (b"\x00\x00" * 6000)
                    + pcm16_bytes(
                        np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
                    ),
                    payload[44:],
                )
                self.assertEqual(1, profiles.resolve_calls)
                self.assertEqual(1, profiles.consume_calls)
                self.assertEqual(0, routes["/health"]()["synthesis_sessions"]["active"])

    async def test_disconnect_closes_the_session_without_consuming_preview(self):
        stack, app, profiles = self.make_app()
        with stack:
            async with app.router.lifespan_context(app):
                routes = {route.path: route.endpoint for route in app.routes}
                created = routes["/synthesis-sessions"](
                    server.SynthesisSessionCreateRequest(text="first.")
                )
                response = routes["/synthesis-sessions/{session_id}/audio"](created["session_id"])
                self.assertEqual(streaming_wav_header(24000), await anext(response.body_iterator))
                await response.body_iterator.aclose()
                await asyncio.sleep(0)
                health = routes["/health"]()
                self.assertEqual(0, health["synthesis_sessions"]["active"])
                self.assertEqual(1, health["synthesis_sessions"]["cancelled"])
                self.assertEqual(0, profiles.consume_calls)

    async def test_model_failure_before_audio_emits_no_wav_header(self):
        stack, app, profiles = self.make_app()
        with stack, patch.object(
            server, "_conditioned_chunks", side_effect=RuntimeError("model failed")
        ):
            async with app.router.lifespan_context(app):
                routes = {route.path: route.endpoint for route in app.routes}
                created = routes["/synthesis-sessions"](
                    server.SynthesisSessionCreateRequest(text="first.")
                )
                response = routes["/synthesis-sessions/{session_id}/audio"](
                    created["session_id"]
                )
                with self.assertRaises(RuntimeError):
                    await anext(response.body_iterator)
                await response.body_iterator.aclose()
                self.assertEqual(0, profiles.consume_calls)
                self.assertEqual(
                    0, routes["/health"]()["synthesis_sessions"]["active"]
                )


if __name__ == "__main__":
    unittest.main()
