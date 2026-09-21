"""TTS platform for the Jarvis CosyVoice backend."""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import ClientError, ClientTimeout

from homeassistant.components.tts import (
    ATTR_VOICE,
    DATA_TTS_MANAGER,
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
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    DOMAIN,
    OPTION_INSTRUCT,
    OPTION_SPEED,
)

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_LANGUAGES = ["en-US", "en-GB", "en"]


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
        OPTION_SPEED: 1.0,
        OPTION_INSTRUCT: "",
    }

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the TTS entity."""
        self._attr_unique_id = entry.entry_id
        self._base_url = entry.data[CONF_BASE_URL].rstrip("/")
        self._token = entry.data[CONF_BEARER_TOKEN]
        self._fallback_entity_id = entry.data[CONF_FALLBACK_ENTITY_ID]
        self._request_timeout = int(entry.data[CONF_REQUEST_TIMEOUT])

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

    async def _async_get_cosyvoice_audio(
        self, message: str, options: dict[str, Any]
    ) -> TtsAudioType:
        """Request WAV audio from the authenticated Jarvis bridge."""
        payload = {
            "text": message,
            # Never pin a Home Assistant request to a profile id. The Jarvis
            # bridge resolves this shared selector from current profile state.
            "voice": DEFAULT_VOICE,
            "speed": float(options.get(OPTION_SPEED, 1.0)),
            "instruct": str(options.get(OPTION_INSTRUCT, "")),
        }
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
