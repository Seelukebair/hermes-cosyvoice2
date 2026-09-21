"""Contract tests for the Home Assistant streaming TTS provider."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

from aiohttp import ClientError


class HomeAssistantError(Exception):
    """Minimal Home Assistant error stand-in for import-isolated tests."""


class TTSAudioRequest:
    """Minimal HA request model."""

    def __init__(self, language: str, options: dict, message_gen: AsyncGenerator[str]):
        self.language = language
        self.options = options
        self.message_gen = message_gen


class TTSAudioResponse:
    """Minimal HA response model."""

    def __init__(self, extension: str, data_gen: AsyncGenerator[bytes]):
        self.extension = extension
        self.data_gen = data_gen


class FakeContent:
    """Small aiohttp StreamReader substitute with deterministic failure hooks."""

    def __init__(self, data: bytes, *, fail_after_chunks: int | None = None) -> None:
        self._data = data
        self._position = 0
        self._fail_after_chunks = fail_after_chunks

    async def readexactly(self, count: int) -> bytes:
        data = self._data[self._position : self._position + count]
        self._position += len(data)
        if len(data) != count:
            raise asyncio.IncompleteReadError(data, count)
        return data

    async def iter_chunked(self, count: int) -> AsyncGenerator[bytes]:
        yielded = 0
        while self._position < len(self._data):
            if self._fail_after_chunks is not None and yielded >= self._fail_after_chunks:
                raise ClientError("stream disconnected")
            data = self._data[self._position : self._position + count]
            self._position += len(data)
            yielded += 1
            yield data
        if self._fail_after_chunks is not None and yielded >= self._fail_after_chunks:
            raise ClientError("stream disconnected")


class FakeResponse:
    """Response object accepted by the integration's request context."""

    def __init__(
        self,
        data: bytes,
        *,
        status: int = 200,
        text: str = "bridge error",
        fail_after_chunks: int | None = None,
    ):
        self.status = status
        self.content = FakeContent(data, fail_after_chunks=fail_after_chunks)
        self._text = text
        self.closed = False

    async def text(self) -> str:
        return self._text

    def close(self) -> None:
        self.closed = True


class FakeRequestContext:
    """Async request context which records lifecycle cleanup."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.exited = 0

    async def __aenter__(self) -> FakeResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.exited += 1


class FakeSession:
    """Capture outgoing bridge calls without a network dependency."""

    def __init__(self, context: FakeRequestContext) -> None:
        self.context = context
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs) -> FakeRequestContext:
        self.calls.append((url, kwargs))
        return self.context


class SequencedSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.contexts = [FakeRequestContext(response) for response in responses]
        self.calls = []

    def post(self, url: str, **kwargs) -> FakeRequestContext:
        self.calls.append((url, kwargs))
        return self.contexts[len(self.calls) - 1]


class RoutingSession:
    """Route session protocol calls while retaining their exact order and data."""

    def __init__(self, responses: dict[tuple[str, str], list[FakeResponse]]) -> None:
        self._responses = {
            key: [FakeRequestContext(response) for response in value]
            for key, value in responses.items()
        }
        self.calls: list[tuple[str, str, dict]] = []

    def _request(self, method: str, url: str, **kwargs) -> FakeRequestContext:
        self.calls.append((method, url, kwargs))
        return self._responses[(method, url)].pop(0)

    def post(self, url: str, **kwargs) -> FakeRequestContext:
        return self._request("post", url, **kwargs)

    def get(self, url: str, **kwargs) -> FakeRequestContext:
        return self._request("get", url, **kwargs)

    def delete(self, url: str, **kwargs) -> FakeRequestContext:
        return self._request("delete", url, **kwargs)


def _install_homeassistant_stubs() -> None:
    """Install only the imports required to exercise this custom component."""
    homeassistant = ModuleType("homeassistant")
    components = ModuleType("homeassistant.components")
    tts = ModuleType("homeassistant.components.tts")
    tts.ATTR_VOICE = "voice"
    tts.DATA_TTS_MANAGER = "tts_manager"
    tts.TTSAudioRequest = TTSAudioRequest
    tts.TTSAudioResponse = TTSAudioResponse
    tts.TextToSpeechEntity = type("TextToSpeechEntity", (), {})
    tts.TtsAudioType = tuple[str, bytes]
    tts.Voice = lambda voice_id, name: SimpleNamespace(voice_id=voice_id, name=name)
    helper = ModuleType("homeassistant.components.tts.helper")
    helper.get_engine_instance = lambda hass, entity_id: None
    config_entries = ModuleType("homeassistant.config_entries")
    config_entries.ConfigEntry = object
    core = ModuleType("homeassistant.core")
    core.HomeAssistant = object
    core.callback = lambda func: func
    exceptions = ModuleType("homeassistant.exceptions")
    exceptions.HomeAssistantError = HomeAssistantError
    helpers = ModuleType("homeassistant.helpers")
    aiohttp_client = ModuleType("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: hass.session
    entity_platform = ModuleType("homeassistant.helpers.entity_platform")
    entity_platform.AddConfigEntryEntitiesCallback = object
    sys.modules.update(
        {
            "homeassistant": homeassistant,
            "homeassistant.components": components,
            "homeassistant.components.tts": tts,
            "homeassistant.components.tts.helper": helper,
            "homeassistant.config_entries": config_entries,
            "homeassistant.core": core,
            "homeassistant.exceptions": exceptions,
            "homeassistant.helpers": helpers,
            "homeassistant.helpers.aiohttp_client": aiohttp_client,
            "homeassistant.helpers.entity_platform": entity_platform,
        }
    )


def _load_tts_module():
    _install_homeassistant_stubs()
    base = Path(__file__).parents[1] / "custom_components" / "jarvis_cosyvoice_tts"
    package = ModuleType("jarvis_cosyvoice_tts")
    package.__path__ = [str(base)]
    sys.modules[package.__name__] = package
    for name in ("const", "tts"):
        spec = importlib.util.spec_from_file_location(
            f"jarvis_cosyvoice_tts.{name}", base / f"{name}.py"
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["jarvis_cosyvoice_tts.tts"]


TTS = _load_tts_module()
WAV_HEADER = (
    b"RIFF"
    + (36).to_bytes(4, "little")
    + b"WAVEfmt "
    + (16).to_bytes(4, "little")
    + (1).to_bytes(2, "little")
    + (1).to_bytes(2, "little")
    + (24000).to_bytes(4, "little")
    + (48000).to_bytes(4, "little")
    + (2).to_bytes(2, "little")
    + (16).to_bytes(2, "little")
    + b"data"
    + (0).to_bytes(4, "little")
)


async def _message(parts: list[str]) -> AsyncGenerator[str]:
    for part in parts:
        yield part


async def _consume(data_gen: AsyncGenerator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in data_gen])


class TtsStreamTests(unittest.IsolatedAsyncioTestCase):
    """Exercise stream handling and fallback before real HA deployment."""

    def _entity(self, response: FakeResponse):
        entity = TTS.JarvisCosyVoiceTTSEntity.__new__(TTS.JarvisCosyVoiceTTSEntity)
        entity.hass = SimpleNamespace(session=FakeSession(FakeRequestContext(response)))
        entity._base_url = "http://bridge"
        entity._token = "test-token"
        entity._fallback_entity_id = "tts.kokoro"
        entity._request_timeout = 30
        # Most tests below exercise the original buffered or legacy paths.
        # Continuous-session cases enable the production default explicitly.
        entity._continuous_sentence_streaming = False
        return entity

    def test_empty_options_use_global_default_speed(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))

        self.assertEqual(1.2, entity._payload("test", {})["speed"])

    async def test_stream_forwards_wav_and_releases_response(self) -> None:
        response = FakeResponse(WAV_HEADER + b"first pcm" + b"second pcm")
        entity = self._entity(response)

        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {"speed": 1.1}, _message(["Hello", " world"]))
        )

        self.assertEqual("wav", result.extension)
        self.assertEqual(WAV_HEADER + b"first pcmsecond pcm", await _consume(result.data_gen))
        url, kwargs = entity.hass.session.calls[0]
        self.assertEqual("http://bridge/synthesize-stream", url)
        self.assertEqual("Hello world", kwargs["json"]["text"])
        self.assertEqual(1.1, kwargs["json"]["speed"])
        self.assertTrue(response.closed)
        self.assertEqual(1, entity.hass.session.context.exited)

    async def test_completed_sentences_open_sequential_streams(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._multi_sentence_streaming = True
        entity.hass.session = SequencedSession(
            [FakeResponse(WAV_HEADER + b"one"), FakeResponse(WAV_HEADER + b"two")]
        )
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["First sentence. Sec", "ond sentence!"]))
        )

        self.assertEqual(WAV_HEADER + b"onetwo", await _consume(result.data_gen))
        self.assertEqual(
            ["First sentence.", "Second sentence!"],
            [call[1]["json"]["text"] for call in entity.hass.session.calls],
        )

    async def test_multi_sentence_streaming_is_disabled_by_default(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER + b"all"))
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["First sentence. ", "Second sentence."]))
        )

        self.assertEqual(WAV_HEADER + b"all", await _consume(result.data_gen))
        self.assertEqual(1, len(entity.hass.session.calls))
        self.assertEqual(
            "First sentence. Second sentence.",
            entity.hass.session.calls[0][1]["json"]["text"],
        )

    def test_continuous_sentence_streaming_is_enabled_by_default(self) -> None:
        self.assertTrue(TTS.DEFAULT_CONTINUOUS_SENTENCE_STREAMING)

    async def test_invalid_header_uses_fallback_before_audio(self) -> None:
        response = FakeResponse(b"NOTW" + (b"\x00" * 40))
        entity = self._entity(response)

        async def fallback(message: str):
            self.assertEqual("test", message)
            return "mp3", b"fallback"

        entity._async_get_fallback_audio = fallback
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["test"]))
        )

        self.assertEqual("mp3", result.extension)
        self.assertEqual(b"fallback", await _consume(result.data_gen))
        self.assertTrue(response.closed)
        self.assertEqual(1, entity.hass.session.context.exited)

    async def test_error_after_header_ends_without_voice_switch(self) -> None:
        response = FakeResponse(WAV_HEADER + b"pcm", fail_after_chunks=1)
        entity = self._entity(response)

        async def forbidden_fallback(message: str):
            raise AssertionError("fallback must not run after audio starts")

        entity._async_get_fallback_audio = forbidden_fallback
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["test"]))
        )

        self.assertEqual(WAV_HEADER + b"pcm", await _consume(result.data_gen))
        self.assertTrue(response.closed)
        self.assertEqual(1, entity.hass.session.context.exited)

    async def test_message_limit_rejects_before_opening_bridge(self) -> None:
        response = FakeResponse(WAV_HEADER)
        entity = self._entity(response)

        with self.assertRaises(HomeAssistantError):
            await entity.async_stream_tts_audio(
                TTSAudioRequest(
                    "en-US",
                    {},
                    _message(["x" * (TTS.MAX_STREAM_MESSAGE_BYTES + 1)]),
                )
            )
        self.assertEqual([], entity.hass.session.calls)

    async def test_options_cannot_push_stream_request_past_bridge_limit(self) -> None:
        response = FakeResponse(WAV_HEADER)
        entity = self._entity(response)

        with self.assertRaises(HomeAssistantError):
            await entity.async_stream_tts_audio(
                TTSAudioRequest(
                    "en-US",
                    {"instruct": "x" * TTS.MAX_STREAM_REQUEST_BYTES},
                    _message(["short message"]),
                )
            )
        self.assertEqual([], entity.hass.session.calls)

    async def test_continuous_session_uses_one_audio_stream_for_many_sentences(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._continuous_sentence_streaming = True
        session_url = "http://bridge/synthesis-sessions/session-1"
        entity.hass.session = RoutingSession(
            {
                (
                    "post",
                    "http://bridge/synthesis-sessions",
                ): [FakeResponse(b"", status=201, text='{"session_id":"session-1"}')],
                ("get", session_url + "/audio"): [FakeResponse(WAV_HEADER + b"pcm")],
                ("post", session_url + "/text"): [FakeResponse(b"", status=204)],
                ("post", session_url + "/finish"): [FakeResponse(b"", status=204)],
            }
        )

        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["First sentence. Sec", "ond sentence!"]))
        )

        self.assertEqual(WAV_HEADER + b"pcm", await _consume(result.data_gen))
        audio_calls = [call for call in entity.hass.session.calls if call[0] == "get"]
        self.assertEqual(1, len(audio_calls))
        create = entity.hass.session.calls[0]
        self.assertEqual("First sentence.", create[2]["json"]["text"])
        text_calls = [call for call in entity.hass.session.calls if call[1].endswith("/text")]
        self.assertEqual(["Second sentence!"], [call[2]["json"]["text"] for call in text_calls])
        self.assertTrue(any(call[1].endswith("/finish") for call in entity.hass.session.calls))

    async def test_continuous_session_falls_back_before_audio(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._continuous_sentence_streaming = True
        session_url = "http://bridge/synthesis-sessions/session-2"
        entity.hass.session = RoutingSession(
            {
                (
                    "post",
                    "http://bridge/synthesis-sessions",
                ): [FakeResponse(b"", status=201, text='{"session_id":"session-2"}')],
                ("get", session_url + "/audio"): [FakeResponse(b"NOTW" + (b"\x00" * 40))],
                ("post", session_url + "/text"): [FakeResponse(b"", status=204)],
                ("post", session_url + "/finish"): [FakeResponse(b"", status=204)],
                ("delete", session_url): [FakeResponse(b"", status=204)],
            }
        )

        async def fallback(message: str):
            self.assertEqual("First sentence. Second sentence!", message)
            return "mp3", b"fallback"

        entity._async_get_fallback_audio = fallback
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["First sentence. Second sentence!"]))
        )

        self.assertEqual("mp3", result.extension)
        self.assertEqual(b"fallback", await _consume(result.data_gen))
        self.assertTrue(any(call[0] == "delete" for call in entity.hass.session.calls))

    async def test_continuous_session_does_not_fallback_after_header(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._continuous_sentence_streaming = True
        session_url = "http://bridge/synthesis-sessions/session-3"
        entity.hass.session = RoutingSession(
            {
                (
                    "post",
                    "http://bridge/synthesis-sessions",
                ): [FakeResponse(b"", status=201, text='{"session_id":"session-3"}')],
                ("get", session_url + "/audio"): [
                    FakeResponse(WAV_HEADER + b"pcm", fail_after_chunks=1)
                ],
                ("post", session_url + "/finish"): [FakeResponse(b"", status=204)],
                ("delete", session_url): [FakeResponse(b"", status=204)],
            }
        )

        async def forbidden_fallback(message: str):
            raise AssertionError("fallback must not run after audio starts")

        entity._async_get_fallback_audio = forbidden_fallback
        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["Only sentence."]))
        )

        self.assertEqual(WAV_HEADER + b"pcm", await _consume(result.data_gen))
        self.assertTrue(any(call[0] == "delete" for call in entity.hass.session.calls))

    async def test_continuous_session_disconnect_cancels_and_deletes(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._continuous_sentence_streaming = True
        session_url = "http://bridge/synthesis-sessions/session-4"
        entity.hass.session = RoutingSession(
            {
                (
                    "post",
                    "http://bridge/synthesis-sessions",
                ): [FakeResponse(b"", status=201, text='{"session_id":"session-4"}')],
                ("get", session_url + "/audio"): [FakeResponse(WAV_HEADER + b"pcm")],
                ("delete", session_url): [FakeResponse(b"", status=204)],
            }
        )

        result = await entity.async_stream_tts_audio(
            TTSAudioRequest("en-US", {}, _message(["First sentence. Later sentence."]))
        )
        await anext(result.data_gen)
        await result.data_gen.aclose()

        self.assertTrue(any(call[0] == "delete" for call in entity.hass.session.calls))

    async def test_continuous_session_enforces_first_request_bound(self) -> None:
        entity = self._entity(FakeResponse(WAV_HEADER))
        entity._continuous_sentence_streaming = True

        with self.assertRaises(HomeAssistantError):
            await entity.async_stream_tts_audio(
                TTSAudioRequest(
                    "en-US",
                    {"instruct": "x" * TTS.MAX_STREAM_REQUEST_BYTES},
                    _message(["First sentence."]),
                )
            )
        self.assertEqual([], entity.hass.session.calls)


if __name__ == "__main__":
    unittest.main()
