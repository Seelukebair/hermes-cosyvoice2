"""Conversation support for the Hermes Wyoming Agent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
import json
import logging
import time
import uuid
import re

import aiohttp
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util.yaml.loader import load_yaml_dict

from .const import (
    CONF_NODE_RED_CREDENTIALS,
    CONF_NODE_RED_URL,
    CONF_STALL_ACK_SECONDS,
    CONF_STALL_ACK_TEXT,
    DEFAULT_NODE_RED_CREDENTIALS,
    DEFAULT_STALL_ACK_SECONDS,
    DEFAULT_STALL_ACK_TEXT,
    DOMAIN,
    CONF_HERMES_URL,
    CONF_HERMES_CREDENTIALS,
    DEFAULT_HERMES_URL,
    DEFAULT_HERMES_CREDENTIALS,
)

_LOGGER = logging.getLogger(__name__)
_BACKGROUND_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bresearch\b", r"\binvestigate\b", r"\bdig into\b", r"\bdeep dive\b",
        r"\blook into\b", r"\btroubleshoot\b", r"\bdiagnose\b", r"\banaly[sz]e\b",
        r"\baudit\b", r"\bcheck (the )?logs?\b", r"\bfigure out\b", r"\breport back\b",
        r"\bget back to (me|us)\b", r"\bwhy did\b", r"\bwhat caused\b", r"\broot cause\b",
    )
)


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
        self._node_red_credentials = entry.options.get(
            CONF_NODE_RED_CREDENTIALS,
            entry.data.get(CONF_NODE_RED_CREDENTIALS, DEFAULT_NODE_RED_CREDENTIALS),
        )
        self._node_red_auth: aiohttp.BasicAuth | None = None
        self._hermes_url = entry.options.get(CONF_HERMES_URL, entry.data.get(CONF_HERMES_URL, DEFAULT_HERMES_URL))
        self._hermes_credentials = entry.options.get(
            CONF_HERMES_CREDENTIALS,
            entry.data.get(CONF_HERMES_CREDENTIALS, DEFAULT_HERMES_CREDENTIALS),
        )
        self._hermes_token: str | None = None
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
        use_background = self._is_background_request(user_input.text)
        try:
            if not use_background:
                async def streaming_response() -> AsyncGenerator[dict, None]:
                    emitted = False
                    try:
                        async for delta in self._stream_from_hermes(user_input, request_id):
                            emitted = True
                            yield {"role": "assistant", "content": delta}
                    except Exception:
                        if emitted:
                            raise
                        latency_log(request_id, "hermes_stream_fallback")
                        fallback = await self._forward_to_nodered(
                            user_input.text,
                            user_input.context.user_id,
                            user_input.device_id or "default_device",
                            user_input.conversation_id,
                            request_id,
                        )
                        yield {"role": "assistant", "content": fallback}

                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id or DOMAIN, streaming_response()
                ):
                    pass
                latency_log(request_id, "ha_conversation_return", elapsed_ms=round((time.monotonic() - started) * 1000), mode="stream")
                return conversation.async_get_result_from_chat_log(user_input, chat_log)
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

    @staticmethod
    def _is_background_request(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if re.match(r"^(quick|fast|answer now|right now)\b", normalized):
            return False
        return bool(
            re.match(r"^(background|long task|research task|deep task)\b", normalized)
            or any(pattern.search(normalized) for pattern in _BACKGROUND_PATTERNS)
        )

    async def _get_hermes_token(self) -> str:
        if self._hermes_token is not None:
            return self._hermes_token
        secret_name = str(self._hermes_credentials or "").strip()

        def load_secret() -> str:
            values = load_yaml_dict(self.hass.config.path("secrets.yaml"))
            auth = values.get(secret_name)
            if isinstance(auth, str) and auth:
                return auth
            if isinstance(auth, dict):
                token = auth.get("token") or auth.get("api_key") or auth.get("bearer_token")
                if isinstance(token, str) and token:
                    return token
            raise HomeAssistantError("Hermes API auth secret is missing or invalid")

        self._hermes_token = await self.hass.async_add_executor_job(load_secret)
        return self._hermes_token

    async def _stream_from_hermes(self, user_input, request_id: str) -> AsyncGenerator[str, None]:
        """Yield final-answer text only; Hermes tool events remain unspoken."""
        token = await self._get_hermes_token()
        device = user_input.device_id or "default_device"
        user_id = user_input.context.user_id or "unknown_user"
        conversation_id = user_input.conversation_id or str(uuid.uuid4())
        safe = lambda value: re.sub(r"[^A-Za-z0-9_.:-]", "_", str(value))[:96]
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Hermes-Session-Id": f"ha-conv-{safe(conversation_id)}",
            "X-Hermes-Session-Key": f"ha-user-{safe(user_id)}-{safe(device)}",
        }
        body = {
            "model": "default",
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are Jarvis in a live Home Assistant voice session. Answer naturally and concisely for speech. Use tools when needed, but never narrate tool mechanics or expose tool JSON."},
                {"role": "user", "content": user_input.text},
            ],
        }
        timeout = aiohttp.ClientTimeout(total=180)
        emitted = False
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._hermes_url, json=body, headers=headers) as response:
                if response.status != 200:
                    raise HomeAssistantError(f"Hermes stream returned HTTP {response.status}")
                event_name = "message"
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        raw_line, buffer = buffer.split(b"\n", 1)
                        line = raw_line.decode("utf-8", "replace").strip()
                        if line.startswith("event:"):
                            event_name = line[6:].strip()
                            continue
                        if not line or not line.startswith("data:"):
                            if not line:
                                event_name = "message"
                            continue
                        if event_name != "message":
                            continue
                        raw_data = line[5:].strip()
                        if raw_data == "[DONE]":
                            return
                        try:
                            payload = json.loads(raw_data)
                            choice = (payload.get("choices") or [{}])[0]
                            delta = (choice.get("delta") or {}).get("content")
                        except (json.JSONDecodeError, AttributeError, IndexError):
                            continue
                        if isinstance(delta, str) and delta:
                            emitted = True
                            yield delta
        if not emitted:
            raise HomeAssistantError("Hermes stream returned no answer text")

    async def _get_node_red_auth(self) -> aiohttp.BasicAuth:
        """Resolve Node-RED credentials from HA's protected secrets file."""
        if self._node_red_auth is not None:
            return self._node_red_auth
        secret_name = str(self._node_red_credentials or "").strip()
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
