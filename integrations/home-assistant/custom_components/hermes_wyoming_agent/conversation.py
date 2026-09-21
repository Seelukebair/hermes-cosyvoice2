"""Conversation support for the Hermes Wyoming Agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
import json
import logging
import time
import uuid

import aiohttp
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util.yaml.loader import load_yaml_dict

from .const import (
    CONF_NODE_RED_AUTH_SECRET,
    CONF_NODE_RED_URL,
    CONF_STALL_ACK_SECONDS,
    CONF_STALL_ACK_TEXT,
    DEFAULT_NODE_RED_AUTH_SECRET,
    DEFAULT_STALL_ACK_SECONDS,
    DEFAULT_STALL_ACK_TEXT,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


def latency_log(request_id: str, event: str, **fields) -> None:
    """Write content-free structured latency events for correlation."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request_id,
        "event": event,
        "source": "ha_custom_component",
        **fields,
    }
    _LOGGER.info("LATENCY %s", json.dumps(payload, sort_keys=True))


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the conversation agent."""
    async_add_entities([HermesWyomingAgent(hass, config_entry)])


class HermesWyomingAgent(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Forward Assist requests to Hermes through the existing Node-RED route."""

    _attr_has_entity_name = True
    _attr_name = "Hermes Proxy Agent"
    _attr_supports_streaming = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._node_red_url = entry.options.get(CONF_NODE_RED_URL) or entry.data.get(
            CONF_NODE_RED_URL
        )
        self._node_red_auth_secret = entry.options.get(
            CONF_NODE_RED_AUTH_SECRET,
            entry.data.get(CONF_NODE_RED_AUTH_SECRET, DEFAULT_NODE_RED_AUTH_SECRET),
        )
        self._node_red_auth: aiohttp.BasicAuth | None = None
        self._stall_ack_seconds = float(
            entry.options.get(
                CONF_STALL_ACK_SECONDS,
                entry.data.get(CONF_STALL_ACK_SECONDS, DEFAULT_STALL_ACK_SECONDS),
            )
            or 0
        )
        self._stall_ack_text = (
            entry.options.get(
                CONF_STALL_ACK_TEXT,
                entry.data.get(CONF_STALL_ACK_TEXT, DEFAULT_STALL_ACK_TEXT),
            )
            or DEFAULT_STALL_ACK_TEXT
        )

    @property
    def supported_languages(self) -> list[str] | str:
        return ["en"]

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        latency_log(
            request_id,
            "ha_received",
            conversation_id=user_input.conversation_id,
            device_id=user_input.device_id or "default_device",
            text_chars=len(user_input.text or ""),
        )
        try:
            forward_task = asyncio.create_task(
                self._forward_to_nodered(
                    user_input.text,
                    user_input.context.user_id,
                    user_input.device_id or "default_device",
                    user_input.conversation_id,
                    request_id,
                )
            )
            response_text = await self._wait_for_voice_response(
                forward_task, request_id, started
            )
            if not response_text:
                response_text = "Hermes did not return a final answer."
        except Exception as error:
            latency_log(
                request_id,
                "ha_error",
                elapsed_ms=round((time.monotonic() - started) * 1000),
                error=type(error).__name__,
            )
            response_text = "Network error reaching Node-RED."

        latency_log(
            request_id,
            "ha_conversation_return",
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )

        async def final_response() -> AsyncGenerator[dict, None]:
            # HA selects streaming TTS for sufficiently long chat-log deltas.
            yield {"role": "assistant", "content": response_text}

        async for _ in chat_log.async_add_delta_content_stream(
            self.entity_id or DOMAIN, final_response()
        ):
            pass
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    async def _get_node_red_auth(self) -> aiohttp.BasicAuth:
        """Resolve Node-RED credentials from HA's protected secrets file."""
        if self._node_red_auth is not None:
            return self._node_red_auth
        secret_name = str(self._node_red_auth_secret or "").strip()
        if not secret_name:
            raise HomeAssistantError("Node-RED auth secret is not configured")

        def load_secret() -> tuple[str, str]:
            values = load_yaml_dict(self.hass.config.path("secrets.yaml"))
            auth = values.get(secret_name)
            if not isinstance(auth, dict):
                raise HomeAssistantError("Node-RED auth secret is missing or invalid")
            username = auth.get("username")
            password = auth.get("password")
            if not isinstance(username, str) or not username:
                raise HomeAssistantError("Node-RED auth username is missing")
            if not isinstance(password, str) or not password:
                raise HomeAssistantError("Node-RED auth password is missing")
            return username, password

        username, password = await self.hass.async_add_executor_job(load_secret)
        self._node_red_auth = aiohttp.BasicAuth(username, password)
        return self._node_red_auth

    async def _wait_for_voice_response(
        self, forward_task: asyncio.Task, request_id: str, started: float
    ) -> str:
        """Wait for the answer or return a configured stall acknowledgement."""
        if self._stall_ack_seconds <= 0:
            return await forward_task
        try:
            return await asyncio.wait_for(
                asyncio.shield(forward_task), timeout=self._stall_ack_seconds
            )
        except asyncio.TimeoutError:
            latency_log(
                request_id,
                "stall_ack_return",
                elapsed_ms=round((time.monotonic() - started) * 1000),
                stall_ack_seconds=self._stall_ack_seconds,
            )
            forward_task.add_done_callback(
                lambda task: self._log_late_voice_response(task, request_id, started)
            )
            return self._stall_ack_text

    def _log_late_voice_response(
        self, task: asyncio.Task, request_id: str, started: float
    ) -> None:
        try:
            late_response = task.result()
            latency_log(
                request_id,
                "late_voice_response",
                elapsed_ms=round((time.monotonic() - started) * 1000),
                response_chars=len(late_response or ""),
            )
        except Exception as error:
            latency_log(
                request_id,
                "late_voice_response_error",
                elapsed_ms=round((time.monotonic() - started) * 1000),
                error=type(error).__name__,
            )

    async def _forward_to_nodered(
        self,
        text: str,
        user_id: str,
        device: str,
        conversation_id: str | None,
        request_id: str,
    ) -> str:
        payload = {
            "text": text,
            "user_id": user_id,
            "device": device,
            "device_id": device,
            "conversation_id": conversation_id,
            "request_id": request_id,
            "source": "home_assistant_assist",
            "expects": {"response_field": "response"},
        }
        auth = await self._get_node_red_auth()
        timeout = aiohttp.ClientTimeout(total=180)
        post_started = time.monotonic()
        latency_log(request_id, "nodered_post_start")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._node_red_url, json=payload, auth=auth) as response:
                latency_log(
                    request_id,
                    "nodered_response_received",
                    status=response.status,
                    elapsed_ms=round((time.monotonic() - post_started) * 1000),
                )
                if response.status != 200:
                    return f"Error HTTP {response.status}"
                raw_text = await response.text()
                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    value = raw_text.strip()
                    return value if value and value != text else "Hermes responded unexpectedly."
                return self._extract_response_text(data, text) or (
                    "Hermes responded, but the Node-RED flow did not return a final answer."
                )

    def _extract_response_text(self, data: dict, original_text: str) -> str | None:
        """Extract the final answer without echoing the user's input."""
        for candidate in (
            data.get("response"),
            data.get("speech"),
            data.get("answer"),
            data.get("message"),
        ):
            if isinstance(candidate, str):
                value = candidate.strip()
                if value and value != original_text:
                    return value
        payload = data.get("payload")
        return self._extract_response_text(payload, original_text) if isinstance(payload, dict) else None
