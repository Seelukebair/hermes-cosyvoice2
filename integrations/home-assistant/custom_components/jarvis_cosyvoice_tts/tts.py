"""TTS platform for the Jarvis CosyVoice backend."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any

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
    CONF_FALLBACK_ENTITY_ID,
    CONF_REQUEST_TIMEOUT,
    CONF_MULTI_SENTENCE_STREAMING,
    DEFAULT_LANGUAGE,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    DEFAULT_MULTI_SENTENCE_STREAMING,
    DOMAIN,
    OPTION_INSTRUCT,
    OPTION_SPEED,
    MAX_STREAM_MESSAGE_BYTES,
    MAX_STREAM_REQUEST_BYTES,
    STREAM_ENDPOINT,
    STREAM_READ_SIZE,
)

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_LANGUAGES = ["en-US", "en-GB", "en"]
_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"')\]]+)?\s+")


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
        if not getattr(
            self, "_multi_sentence_streaming", DEFAULT_MULTI_SENTENCE_STREAMING
        ):
            message = await self._async_collect_stream_message(request.message_gen)
            payload = self._payload(message, request.options)
            if (
                len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
                > MAX_STREAM_REQUEST_BYTES
            ):
                raise HomeAssistantError("TTS stream request exceeds the bridge limit")
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
        if (
            len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            > MAX_STREAM_REQUEST_BYTES
        ):
            raise HomeAssistantError("TTS stream request exceeds the bridge limit")
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
        request_context = session.post(
            self._base_url + STREAM_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=ClientTimeout(total=self._request_timeout),
        )
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
                return
            finally:
                response.close()
                await request_context.__aexit__(None, None, None)

        return TTSAudioResponse("wav", data_gen())

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
