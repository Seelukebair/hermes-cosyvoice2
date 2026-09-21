"""Config flow for Hermes Wyoming Agent."""

from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries

from .const import (
    CONF_NODE_RED_AUTH_SECRET,
    CONF_NODE_RED_URL,
    CONF_STALL_ACK_SECONDS,
    CONF_STALL_ACK_TEXT,
    DEFAULT_NODE_RED_AUTH_SECRET,
    DEFAULT_NODE_RED_URL,
    DEFAULT_STALL_ACK_SECONDS,
    DEFAULT_STALL_ACK_TEXT,
    DOMAIN,
)


def _schema(data: dict | None = None) -> vol.Schema:
    values = data or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NODE_RED_URL,
                default=values.get(CONF_NODE_RED_URL, DEFAULT_NODE_RED_URL),
            ): str,
            vol.Optional(
                CONF_NODE_RED_AUTH_SECRET,
                default=values.get(
                    CONF_NODE_RED_AUTH_SECRET, DEFAULT_NODE_RED_AUTH_SECRET
                ),
            ): str,
            vol.Optional(
                CONF_STALL_ACK_SECONDS,
                default=values.get(CONF_STALL_ACK_SECONDS, DEFAULT_STALL_ACK_SECONDS),
            ): vol.All(vol.Coerce(float), vol.Range(min=0, max=60)),
            vol.Optional(
                CONF_STALL_ACK_TEXT,
                default=values.get(CONF_STALL_ACK_TEXT, DEFAULT_STALL_ACK_TEXT),
            ): str,
        }
    )


class HermesWyomingConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Hermes Wyoming Agent."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            return self.async_create_entry(title="Hermes Agent", data=user_input)
        return self.async_show_form(step_id="user", data_schema=_schema())

    @staticmethod
    def async_get_options_flow(config_entry):
        """Create the options flow."""
        return HermesWyomingOptionsFlow(config_entry)


class HermesWyomingOptionsFlow(config_entries.OptionsFlow):
    """Handle Hermes Wyoming Agent options."""

    def __init__(self, config_entry):
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage integration options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)
        data = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id="init", data_schema=_schema(data))
