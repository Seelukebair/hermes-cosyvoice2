"""TTS platform for the Jarvis CosyVoice backend."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Callable

from aiohttp import ClientError, ClientTimeout

from homeassistant.components.tts import (
    ATTR_VOICE,
    DATA_TTS_MANAGER,
    TTSAudioRequest,
    TTSAudioResponse,
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.components.tts.helper import get_engine_instance
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    CONF_BASE_URL,
    CONF_BEARER_TOKEN,
    CONF_CONTINUOUS_SENTENCE_STREAMING,
    CONF_FALLBACK_ENTITY_ID,
    CONF_REQUEST_TIMEOUT,
    CONF_MULTI_SENTENCE_STREAMING,
    DEFAULT_LANGUAGE,
    DEFAULT_CONTINUOUS_SENTENCE_STREAMING,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    DEFAULT_MULTI_SENTENCE_STREAMING,
    DOMAIN,
    OPTION_INSTRUCT,
    OPTION_SPEED,
    MAX_STREAM_MESSAGE_BYTES,
    MAX_STREAM_REQUEST_BYTES,
    STREAM_ENDPOINT,
    SYNTHESIS_SESSIONS_ENDPOINT,
    STREAM_READ_SIZE,
)

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_LANGUAGES = ["en-US", "en-GB", "en"]
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"')\]]+)?\s+")
_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the Jarvis CosyVoice TTS entity."""
    async_add_entities([JarvisCosyVoiceTTSEntity(config_entry)])


class JarvisCosyVoiceTTSEntity(TextToSpeechEntity):
    """Route TTS to Jarvis CosyVoice with an existing HA fallback."""

    _attr_name = "Jarvis CosyVoice"
    _attr_translation_key = "jarvis_cosyvoice"
    _attr_has_entity_name = False
    _attr_default_language = DEFAULT_LANGUAGE
    _attr_supported_languages = _SUPPORTED_LANGUAGES
    _attr_supported_options = [ATTR_VOICE, OPTION_SPEED, OPTION_INSTRUCT]
    _attr_default_options = {
        ATTR_VOICE: DEFAULT_VOICE,
        OPTION_SPEED: DEFAULT_SPEED,
        OPTION_INSTRUCT: "",
    }

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the TTS entity."""
        self._attr_unique_id = entry.entry_id
        self._base_url = entry.data[CONF_BASE_URL].rstrip("/")
        self._token = entry.data[CONF_BEARER_TOKEN]
        self._fallback_entity_id = entry.data[CONF_FALLBACK_ENTITY_ID]
        self._request_timeout = int(entry.data[CONF_REQUEST_TIMEOUT])
        self._multi_sentence_streaming = bool(
            entry.data.get(
                CONF_MULTI_SENTENCE_STREAMING, DEFAULT_MULTI_SENTENCE_STREAMING
            )
        )
        self._continuous_sentence_streaming = bool(
            entry.data.get(
                CONF_CONTINUOUS_SENTENCE_STREAMING,
                DEFAULT_CONTINUOUS_SENTENCE_STREAMING,
            )
        )

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice]:
        """Expose the dynamic shared selector, not stale profile ids."""
        return [Voice(DEFAULT_VOICE, "Hermes shared default")]

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Generate audio with CosyVoice, then Kokoro on failure."""
        try:
            return await self._async_get_cosyvoice_audio(message, options)
        except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
            _LOGGER.warning(
                "Jarvis CosyVoice unavailable; trying fallback entity %s: %s",
                self._fallback_entity_id,
                err,
            )
            return await self._async_get_fallback_audio(message)

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Return the bridge's progressive WAV response to Home Assistant.

        The current conversation adapter delivers a completed reply as one text
        item. Collecting that one item here still lets the client play while
        CosyVoice generates, and avoids synthesizing arbitrary LLM token pieces.
        """
        if getattr(
            self,
            "_continuous_sentence_streaming",
            DEFAULT_CONTINUOUS_SENTENCE_STREAMING,
        ):
            return await self._async_stream_continuous_sentences(request)

        if not getattr(
            self, "_multi_sentence_streaming", DEFAULT_MULTI_SENTENCE_STREAMING
        ):
            message = await self._async_collect_stream_message(request.message_gen)
            payload = self._payload(message, request.options)
            self._assert_stream_request_bound(payload)
            try:
                return await self._async_open_cosyvoice_stream(payload)
            except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
                _LOGGER.warning(
                    "Jarvis CosyVoice stream unavailable before audio; trying fallback entity %s: %s",
                    self._fallback_entity_id,
                    err,
                )
                extension, audio = await self._async_get_fallback_audio(message)

                async def buffered_fallback() -> AsyncGenerator[bytes]:
                    yield audio

                return TTSAudioResponse(extension, buffered_fallback())

        sentences = self._async_sentence_messages(request.message_gen)
        try:
            first_sentence = await anext(sentences)
        except StopAsyncIteration as err:
            raise HomeAssistantError("TTS stream input is empty") from err
        payload = self._payload(first_sentence, request.options)
        self._assert_stream_request_bound(payload)
        try:
            first = await self._async_open_cosyvoice_stream(payload)

            async def sentence_audio() -> AsyncGenerator[bytes]:
                async for chunk in first.data_gen:
                    yield chunk
                async for sentence in sentences:
                    following = await self._async_open_cosyvoice_stream(
                        self._payload(sentence, request.options)
                    )
                    skipped_header = False
                    async for chunk in following.data_gen:
                        if not skipped_header:
                            skipped_header = True
                            chunk = chunk[44:]
                        if chunk:
                            yield chunk

            return TTSAudioResponse("wav", sentence_audio())
        except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
            _LOGGER.warning(
                "Jarvis CosyVoice stream unavailable before audio; trying fallback entity %s: %s",
                self._fallback_entity_id,
                err,
            )
            remaining = [first_sentence]
            remaining.extend([sentence async for sentence in sentences])
            extension, audio = await self._async_get_fallback_audio(" ".join(remaining))

            async def fallback_data_gen() -> AsyncGenerator[bytes]:
                yield audio

            return TTSAudioResponse(extension, fallback_data_gen())

    async def _async_stream_continuous_sentences(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Feed completed sentences into one backend-owned WAV session."""
        sentences = self._async_sentence_messages(request.message_gen)
        try:
            first_sentence = await anext(sentences)
        except StopAsyncIteration as err:
            raise HomeAssistantError("TTS stream input is empty") from err

        received_sentences = [first_sentence]
        payload = self._payload(first_sentence, request.options)
        self._assert_stream_request_bound(payload)
        session_id: str | None = None
        producer: asyncio.Task[None] | None = None
        audio_interrupted = False

        def mark_audio_interrupted() -> None:
            nonlocal audio_interrupted
            audio_interrupted = True

        try:
            session_id = await self._async_create_synthesis_session(payload)
            producer = asyncio.create_task(
                self._async_produce_session_text(
                    session_id, sentences, received_sentences
                ),
                name=f"jarvis-cosyvoice-session-{session_id}",
            )
            stream = await self._async_open_session_audio(
                session_id, mark_audio_interrupted
            )
        except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
            if producer is not None:
                await self._async_wait_for_producer(producer)
            else:
                received_sentences.extend(
                    [sentence async for sentence in sentences]
                )
            if session_id is not None:
                await self._async_delete_synthesis_session(session_id, quiet=True)
            _LOGGER.warning(
                "Jarvis CosyVoice continuous stream unavailable before audio; trying fallback entity %s: %s",
                self._fallback_entity_id,
                err,
            )
            extension, audio = await self._async_get_fallback_audio(
                " ".join(received_sentences)
            )

            async def fallback_data_gen() -> AsyncGenerator[bytes]:
                yield audio

            return TTSAudioResponse(extension, fallback_data_gen())

        async def session_audio() -> AsyncGenerator[bytes]:
            completed = False
            try:
                async for chunk in stream.data_gen:
                    yield chunk
                completed = True
            finally:
                if completed and not audio_interrupted:
                    # The producer sends finish even when source parsing or a
                    # later control call fails. Audio has already started, so a
                    # producer failure is logged rather than changing voices.
                    await self._async_wait_for_producer(producer)
                else:
                    await self._async_cancel_session(session_id, producer)

        return TTSAudioResponse("wav", session_audio())

    async def _async_get_cosyvoice_audio(
        self, message: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Request WAV audio from the authenticated Jarvis bridge."""
        payload = self._payload(message, options)
        session = async_get_clientsession(self.hass)
        async with session.post(
            self._base_url + "/synthesize",
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=ClientTimeout(total=self._request_timeout),
        ) as response:
            if response.status != 200:
                detail = (await response.text())[:240]
                raise HomeAssistantError(
                    f"CosyVoice bridge returned HTTP {response.status}: {detail}"
                )
            audio = await response.read()
        if len(audio) < 44 or not audio.startswith(b"RIFF") or audio[8:12] != b"WAVE":
            raise HomeAssistantError("CosyVoice bridge returned invalid WAV audio")
        return "wav", audio

    def _payload(self, message: str, options: dict[str, Any]) -> dict[str, Any]:
        """Build the request shape shared by buffered and streaming calls."""
        return {
            "text": message,
            # Never pin a Home Assistant request to a profile id. The Jarvis
            # bridge resolves this shared selector from current profile state.
            "voice": DEFAULT_VOICE,
            "speed": float(options.get(OPTION_SPEED, DEFAULT_SPEED)),
            "instruct": str(options.get(OPTION_INSTRUCT, "")),
        }

    @staticmethod
    def _assert_stream_request_bound(payload: dict[str, Any]) -> None:
        """Keep every bridge POST within the shared bounded request contract."""
        if (
            len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            > MAX_STREAM_REQUEST_BYTES
        ):
            raise HomeAssistantError("TTS stream request exceeds the bridge limit")

    async def _async_collect_stream_message(
        self, message_gen: AsyncGenerator[str]
    ) -> str:
        """Bound a completed HA assistant response before requesting speech."""
        chunks: list[str] = []
        byte_count = 0
        async for chunk in message_gen:
            if not isinstance(chunk, str):
                raise HomeAssistantError("TTS stream input must contain text")
            byte_count += len(chunk.encode("utf-8"))
            if byte_count > MAX_STREAM_MESSAGE_BYTES:
                raise HomeAssistantError("TTS stream input exceeds the bridge limit")
            chunks.append(chunk)
        message = "".join(chunks).strip()
        if not message:
            raise HomeAssistantError("TTS stream input is empty")
        return message

    async def _async_sentence_messages(
        self, message_gen: AsyncGenerator[str]
    ) -> AsyncGenerator[str]:
        """Yield bounded complete sentences, retaining a final fragment."""
        pending = ""
        byte_count = 0
        async for chunk in message_gen:
            if not isinstance(chunk, str):
                raise HomeAssistantError("TTS stream input must contain text")
            byte_count += len(chunk.encode("utf-8"))
            if byte_count > MAX_STREAM_MESSAGE_BYTES:
                raise HomeAssistantError("TTS stream input exceeds the bridge limit")
            pending += chunk
            while True:
                match = _SENTENCE_END.search(pending)
                if match is None:
                    break
                sentence = pending[: match.start() + 1].strip()
                pending = pending[match.end() :]
                if sentence:
                    yield sentence
        if pending.strip():
            yield pending.strip()

    async def _async_open_cosyvoice_stream(
        self, payload: dict[str, Any]
    ) -> TTSAudioResponse:
        """Open and validate the first WAV bytes before exposing audio to HA."""
        session = async_get_clientsession(self.hass)
        return await self._async_open_audio_stream(
            session.post(
                self._base_url + STREAM_ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=ClientTimeout(total=self._request_timeout),
            )
        )

    async def _async_open_session_audio(
        self, session_id: str, on_interrupted: Callable[[], None]
    ) -> TTSAudioResponse:
        """Open the one WAV stream associated with a synthesis session."""
        session = async_get_clientsession(self.hass)
        return await self._async_open_audio_stream(
            session.get(
                self._session_url(session_id) + "/audio",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=ClientTimeout(total=self._request_timeout),
            ),
            on_interrupted=on_interrupted,
        )

    async def _async_open_audio_stream(
        self,
        request_context: Any,
        on_interrupted: Callable[[], None] | None = None,
    ) -> TTSAudioResponse:
        """Validate a WAV response before returning a generator to HA."""
        response = await request_context.__aenter__()
        try:
            if response.status != 200:
                detail = (await response.text())[:240]
                raise HomeAssistantError(
                    f"CosyVoice bridge returned HTTP {response.status}: {detail}"
                )
            # The native Wyoming provider sends a 44-byte PCM WAV header with
            # nframes=0, followed by PCM. Validate it before HA commits to the
            # stream so a fallback can still be returned safely.
            header = await response.content.readexactly(44)
            if not (
                header.startswith(b"RIFF")
                and header[8:12] == b"WAVE"
                and header[12:16] == b"fmt "
                and header[36:40] == b"data"
                and header[40:44] == b"\x00\x00\x00\x00"
            ):
                raise HomeAssistantError(
                    "CosyVoice bridge returned invalid streaming WAV audio"
                )
        except BaseException:
            response.close()
            await request_context.__aexit__(None, None, None)
            raise

        async def data_gen() -> AsyncGenerator[bytes]:
            try:
                yield header
                async for chunk in response.content.iter_chunked(STREAM_READ_SIZE):
                    if chunk:
                        yield chunk
            except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
                # Audio already reached HA. A fallback would switch voices partway
                # through an utterance, so close this stream and report the fault.
                _LOGGER.warning("Jarvis CosyVoice stream ended after audio: %s", err)
                if on_interrupted is not None:
                    on_interrupted()
                return
            finally:
                response.close()
                await request_context.__aexit__(None, None, None)

        return TTSAudioResponse("wav", data_gen())

    def _session_url(self, session_id: str) -> str:
        """Build a bridge session URL only from an opaque server-issued id."""
        if not _SESSION_ID.fullmatch(session_id):
            raise HomeAssistantError("CosyVoice bridge returned an invalid session id")
        return self._base_url + SYNTHESIS_SESSIONS_ENDPOINT + "/" + session_id

    async def _async_create_synthesis_session(self, payload: dict[str, Any]) -> str:
        """Create a backend session and validate its opaque response id."""
        session = async_get_clientsession(self.hass)
        async with session.post(
            self._base_url + SYNTHESIS_SESSIONS_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=ClientTimeout(total=self._request_timeout),
        ) as response:
            if response.status not in (200, 201):
                detail = (await response.text())[:240]
                raise HomeAssistantError(
                    f"CosyVoice bridge returned HTTP {response.status}: {detail}"
                )
            try:
                result = json.loads(await response.text())
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise HomeAssistantError(
                    "CosyVoice bridge returned an invalid synthesis session"
                ) from err
        session_id = result.get("session_id") if isinstance(result, dict) else None
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise HomeAssistantError("CosyVoice bridge returned an invalid session id")
        return session_id

    async def _async_produce_session_text(
        self,
        session_id: str,
        sentences: AsyncGenerator[str],
        received_sentences: list[str],
    ) -> None:
        """Serially upload later sentences and always signal end of input."""
        failure: BaseException | None = None
        try:
            async for sentence in sentences:
                received_sentences.append(sentence)
                if failure is not None:
                    continue
                try:
                    payload = {"text": sentence}
                    self._assert_stream_request_bound(payload)
                    await self._async_post_session_text(session_id, payload)
                except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
                    # Continue draining the HA generator so pre-audio fallback
                    # still has the complete answer and input stays bounded.
                    failure = err
        except BaseException as err:
            failure = err
        finally:
            try:
                await self._async_finish_synthesis_session(session_id)
            except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
                if failure is None:
                    failure = err
        if failure is not None:
            raise failure

    async def _async_post_session_text(
        self, session_id: str, payload: dict[str, str]
    ) -> None:
        """Forward one bounded completed sentence with natural backpressure."""
        await self._async_session_control_request("post", session_id, "text", payload)

    async def _async_finish_synthesis_session(self, session_id: str) -> None:
        """Tell the backend no additional session text will be supplied."""
        await self._async_session_control_request("post", session_id, "finish")

    async def _async_delete_synthesis_session(
        self, session_id: str, *, quiet: bool = False
    ) -> None:
        """Cancel backend work after a disconnect or failed stream setup."""
        try:
            await self._async_session_control_request("delete", session_id)
        except (ClientError, TimeoutError, HomeAssistantError, ValueError):
            if not quiet:
                _LOGGER.warning("Unable to cancel CosyVoice synthesis session %s", session_id)

    async def _async_session_control_request(
        self,
        method: str,
        session_id: str,
        suffix: str = "",
        payload: dict[str, str] | None = None,
    ) -> None:
        """Perform a short session control request without holding audio state."""
        session = async_get_clientsession(self.hass)
        request = getattr(session, method)
        kwargs: dict[str, Any] = {
            "headers": {"Authorization": f"Bearer {self._token}"},
            "timeout": ClientTimeout(total=self._request_timeout),
        }
        if payload is not None:
            kwargs["json"] = payload
        url = self._session_url(session_id)
        if suffix:
            url += "/" + suffix
        async with request(url, **kwargs) as response:
            if response.status not in (200, 202, 204):
                detail = (await response.text())[:240]
                raise HomeAssistantError(
                    f"CosyVoice bridge returned HTTP {response.status}: {detail}"
                )

    async def _async_wait_for_producer(self, producer: asyncio.Task[None]) -> None:
        """Observe producer failure without ever changing a started voice."""
        try:
            await producer
        except asyncio.CancelledError:
            raise
        except (ClientError, TimeoutError, HomeAssistantError, ValueError) as err:
            _LOGGER.warning("Jarvis CosyVoice sentence producer ended: %s", err)

    async def _async_cancel_session(
        self, session_id: str, producer: asyncio.Task[None]
    ) -> None:
        """Cancel source production before cancelling the backend session."""
        if not producer.done():
            producer.cancel()
        try:
            await producer
        except (
            asyncio.CancelledError,
            ClientError,
            TimeoutError,
            HomeAssistantError,
            ValueError,
        ):
            pass
        await self._async_delete_synthesis_session(session_id, quiet=True)

    async def _async_get_fallback_audio(self, message: str) -> TtsAudioType:
        """Call the existing Home Assistant TTS entity without media playback."""
        fallback = get_engine_instance(self.hass, self._fallback_entity_id)
        if fallback is None:
            raise HomeAssistantError(
                f"Fallback TTS provider {self._fallback_entity_id} is unavailable"
            )
        if fallback is self:
            raise HomeAssistantError("Fallback TTS provider cannot be this entity")
        manager = self.hass.data[DATA_TTS_MANAGER]
        language, options = manager.process_options(fallback, None, {})
        extension, audio = await fallback.async_internal_get_tts_audio(
            message, language, options
        )
        if extension is None or audio is None:
            raise HomeAssistantError(
                f"Fallback TTS provider {self._fallback_entity_id} returned no audio"
            )
        return extension, audio
