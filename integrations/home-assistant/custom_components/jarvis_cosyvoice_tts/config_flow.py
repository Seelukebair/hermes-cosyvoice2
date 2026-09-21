"""Config flow for Jarvis CosyVoice TTS."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientError, ClientResponseError, ClientTimeout
import probatio

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import (
    CONF_BASE_URL,
    CONF_BEARER_TOKEN,
    CONF_FALLBACK_ENTITY_ID,
    CONF_REQUEST_TIMEOUT,
    CONF_MULTI_SENTENCE_STREAMING,
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACK_ENTITY_ID,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_MULTI_SENTENCE_STREAMING,
    DOMAIN,
)


class CannotConnect(Exception):
    """Raised when the bridge cannot be reached."""


class InvalidAuth(Exception):
    """Raised when bridge authentication fails."""


async def _validate_bridge(hass, data: dict[str, Any]) -> None:
    """Validate only the lightweight bridge, never synthesize audio."""
    session = async_get_clientsession(hass)
    url = data[CONF_BASE_URL].rstrip("/") + "/proxy-health"
    headers = {"Authorization": f"Bearer {data[CONF_BEARER_TOKEN]}"}
    try:
        async with session.get(
            url,
            headers=headers,
            timeout=ClientTimeout(total=10),
        ) as response:
            if response.status in (401, 403):
                raise InvalidAuth
            response.raise_for_status()
            payload = await response.json()
    except InvalidAuth:
        raise
    except (ClientError, TimeoutError, ValueError) as err:
        raise CannotConnect from err
    if payload.get("status") != "ok":
        raise CannotConnect


class JarvisCosyVoiceConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the Jarvis CosyVoice TTS config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            user_input[CONF_BASE_URL] = user_input[CONF_BASE_URL].rstrip("/")
            try:
                await _validate_bridge(self.hass, user_input)
            except InvalidAuth:
                errors["base"] = "invalid_auth"
            except CannotConnect:
                errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                errors["base"] = "unknown"
            else:
                await self.async_set_unique_id("jarvis-cosyvoice-shared-default")
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Jarvis CosyVoice",
                    data=user_input,
                )

        schema = probatio.Schema(
            {
                probatio.Required(
                    CONF_BASE_URL,
                    default=(user_input or {}).get(CONF_BASE_URL, DEFAULT_BASE_URL),
                ): str,
                probatio.Required(CONF_BEARER_TOKEN): str,
                probatio.Required(
                    CONF_FALLBACK_ENTITY_ID,
                    default=(user_input or {}).get(
                        CONF_FALLBACK_ENTITY_ID, DEFAULT_FALLBACK_ENTITY_ID
                    ),
                ): str,
                probatio.Required(
                    CONF_REQUEST_TIMEOUT,
                    default=(user_input or {}).get(
                        CONF_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT
                    ),
                ): probatio.All(int, probatio.Range(min=10, max=600)),
                probatio.Optional(
                    CONF_MULTI_SENTENCE_STREAMING,
                    default=(user_input or {}).get(
                        CONF_MULTI_SENTENCE_STREAMING,
                        DEFAULT_MULTI_SENTENCE_STREAMING,
                    ),
                ): bool,
            }
        )
        return self.async_show_form(
            step_id="user", data_schema=schema, errors=errors
        )
