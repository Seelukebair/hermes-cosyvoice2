"""Hermes tool and intent hooks for the packaged CosyVoice2 integration."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from .voice_manager import VoiceManager, VoiceWorkflowError


_VOICE_CHANGE_RE = re.compile(
    r"\b(?:change|switch|set|use|try|make|create|build|generate|clone)\b.{0,64}\b(?:your|assistant|jarvis|cosyvoice)\b.{0,32}\bvoice\b"
    r"|\b(?:your|assistant|jarvis|cosyvoice)\b.{0,32}\bvoice\b.{0,64}\b(?:change|switch|set|use|try|make|create|build|generate|default)\b"
    r"|\b(?:create|build|generate|make|clone)\b.{0,64}\b(?:voice|voice profile|voice clone)\b",
    re.IGNORECASE | re.DOTALL,
)
_SAVED_PROFILE_SWITCH_RE = re.compile(
    r"\b(?:switch|swap|change|set|use|activate|select)\b.{0,80}"
    r"\b(?:voices?|personas?|voice\s+profiles?|profiles?)\b"
    r"|\b(?:switch|swap|change|set|use|activate|select)\b.{0,24}\bto\b.{1,80}"
    r"\b(?:voice|persona)\b",
    re.IGNORECASE | re.DOTALL,
)
_HEAR_SAVED_PROFILE_RE = re.compile(
    r"\b(?:i\s+)?(?:want|would\s+like|wanna)\s+to\s+hear\s+"
    r"(?P<target>[a-z0-9][a-z0-9 ._'\u2019-]{1,80}?)(?:[.!?]|$)",
    re.IGNORECASE,
)
_VOICE_PROFILE_RE = re.compile(
    r"\b(?:voice profiles?|voice clones?|cloned voices?|cosyvoice voices?|voice\s*(?:/|and)?\s*personality|personality\s*(?:/|and)?\s*voice|mannerisms?|catchphrases?|key phrases?)\b",
    re.IGNORECASE,
)
_VOICE_INVENTORY_RE = re.compile(
    r"\b(?:which|what|list|show|available|saved|installed)\b.{0,64}\b(?:cloned\s+voices?|voice\s+profiles?)\b"
    r"|\b(?:cloned\s+voices?|voice\s+profiles?)\b.{0,64}\b(?:available|saved|installed|do\s+(?:i|we|you)\s+have|are\s+there)\b",
    re.IGNORECASE | re.DOTALL,
)
_VOICE_FOLLOWUP_RE = re.compile(
    r"\b(?:option|choice|source)\s*(?:one|two|three|four|five|[1-5])\b"
    r"|\b(?:use|pick|choose|keep|save|accept|discard|reset|refine|preview|transcript|correct)\b"
    r"|\b(?:make|set)\b.{0,24}\bdefault\b",
    re.IGNORECASE,
)
_VOICE_ACTIVATION_RE = re.compile(
    r"\b(?:activate|enable|apply)\b(?:\s+(?:it|this|that|voice|profile))?\b"
    r"|\buse\b.{0,16}\b(?:it|this|that|voice|profile)\b"
    r"|\b(?:make|set)\b.{0,32}\b(?:it|this|that|voice|profile)\b.{0,32}\b(?:active|default)\b",
    re.IGNORECASE,
)
_SOURCE_SELECTION_RE = re.compile(
    r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)/\S+"
    r"|\b(?:pick|choose|select|go with|use|try)\b.{0,40}"
    r"\b(?:option|choice|source|result|one|two|three|four|five|first|second|third|fourth|fifth|[1-5])\b",
    re.IGNORECASE,
)
_AUTONOMOUS_SOURCE_RE = re.compile(
    r"\b(?:choose|pick|select)\b.{0,32}\b(?:for me|yourself|the best|best one|best source)\b"
    r"|\b(?:handle|do)\b.{0,24}\b(?:everything|all of it|the whole process)\b"
    r"|\b(?:go forth|autonomous(?:ly)?|fully automatic|take it from here)\b",
    re.IGNORECASE,
)
_CREATE_VOICE_RE = re.compile(
    r"\b(?:make|create|build|generate|clone)\b.{0,64}\b(?:voice|voice profile|voice clone)\b",
    re.IGNORECASE | re.DOTALL,
)
_POSSESSIVE_VOICE_TARGET_RE = re.compile(
    r"\b(?:make|create|build|generate|clone)\s+(?P<target>[a-z0-9][^,.!?]{0,80}?)['\u2019]s\s+voice\b",
    re.IGNORECASE,
)
_DIRECT_VOICE_TARGET_RE = re.compile(
    r"\b(?:make|create|build|generate|clone)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?P<target>[a-z0-9][a-z0-9 ._-]{0,60}?)\s+voice(?:\s+(?:clone|profile))?\b",
    re.IGNORECASE,
)
_VOICE_OF_TARGET_RE = re.compile(
    r"\b(?:make|create|build|generate|clone)\s+(?:a\s+|an\s+|the\s+)?voice\s+of\s+"
    r"(?P<target>[a-z0-9][a-z0-9 ._'-]{0,60}?)(?:\s+using\b|\s+from\b|\s+with\b|\s+https?://|[,.!?]|$)",
    re.IGNORECASE,
)
_SOURCE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtu\.be)/\S+",
    re.IGNORECASE,
)
_DEFAULT_REQUEST_RE = re.compile(
    r"\b(?:set|make)\b.{0,64}\bdefault\b|\bas\s+(?:the\s+)?(?:persistent\s+)?default\b",
    re.IGNORECASE | re.DOTALL,
)
_AUTONOMOUS_SEARCH_RE = re.compile(
    r"\b(?:search|find|locate|look for)\b.{0,64}\b(?:good|best|clean|clear|usable|suitable)\b.{0,48}\b(?:voice|audio|source|clip|sample)\b"
    r"|\b(?:good|best|clean|clear|usable|suitable)\b.{0,48}\b(?:voice|audio|source|clip|sample)\b.{0,48}\b(?:search|find|locate)\b",
    re.IGNORECASE | re.DOTALL,
)
_PERSONALITY_DISABLE_RE = re.compile(
    r"\b(?:no|disable|stop|without|do not|don't)\b.{0,48}\b(?:personality|mannerisms?|catchphrases?|key phrases?|talk like)\b"
    r"|\b(?:voice only|just the voice)\b",
    re.IGNORECASE,
)
_VARIANT_REQUEST_RE = re.compile(
    r"\b(?:again|another|alternate|different|new|newer|recent|replace|redo|updated|better)\b"
    r"|\bmore\s+(?:epic|dramatic|serious|intense|modern)\b",
    re.IGNORECASE,
)
_LOCK = threading.Lock()
_TURN_INTENT: dict[str, dict[str, Any]] = {}


def _turn_key(turn_id: str = "", session_id: str = "") -> str:
    return str(turn_id or session_id or "")


def _manager() -> VoiceManager:
    root = Path(os.getenv("COSYVOICE_VOICE_ROOT", "/srv/cosyvoice2/data/voice_profiles"))
    ytdlp = os.getenv("COSYVOICE_VOICE_YTDLP", "/srv/cosyvoice2/voice-tools/venv/bin/yt-dlp")
    return VoiceManager(root=root, ytdlp=ytdlp)


def _profile_root() -> Path:
    return Path(os.getenv("COSYVOICE_VOICE_ROOT", "/srv/cosyvoice2/data/voice_profiles"))


def _voice_query(message: str) -> str:
    match = (
        _POSSESSIVE_VOICE_TARGET_RE.search(message)
        or _VOICE_OF_TARGET_RE.search(message)
        or _DIRECT_VOICE_TARGET_RE.search(message)
    )
    if match:
        return " ".join(match.group("target").split())
    return " ".join(message.split())[:160]


def _voice_intent_context(
    user_message: str = "", conversation_history: list[dict[str, Any]] | None = None, **kwargs: Any
) -> dict[str, str] | None:
    """Expose this deferred tool only for explicit voice-profile workflow intent."""
    message = str(user_message or "").strip()
    explicit = bool(
        _VOICE_CHANGE_RE.search(message)
        or _SAVED_PROFILE_SWITCH_RE.search(message)
        or _VOICE_PROFILE_RE.search(message)
        or _VOICE_INVENTORY_RE.search(message)
        or _AUTONOMOUS_SEARCH_RE.search(message)
    )
    # Normal chat must not create runtime directories or fail merely because
    # the optional CosyVoice2 runtime has not been installed yet.
    if not _profile_root().is_dir():
        if explicit:
            return {
                "context": "CosyVoice2 voice-profile intent detected, but COSYVOICE_VOICE_ROOT "
                "is not configured or accessible. Report that the optional runtime must be installed "
                "before creating or selecting profiles."
            }
        return None
    manager = _manager()
    switch_result: dict[str, Any] | None = None
    switch_error: dict[str, Any] | None = None
    switch_match = _SAVED_PROFILE_SWITCH_RE.search(message)
    hear_match = _HEAR_SAVED_PROFILE_RE.search(message)
    switch_reference = (
        hear_match.group("target").strip() if hear_match else message if switch_match else ""
    )
    if switch_reference and not _CREATE_VOICE_RE.search(message):
        try:
            profile_id = manager.resolve_profile_argument(
                "select", {"profile_id": switch_reference}
            )
            switch_result = manager.select(profile_id)
            explicit = True
        except VoiceWorkflowError as exc:
            if switch_match:
                switch_error = exc.as_result()
                explicit = True
    workflow = manager.workflow_context()
    recent = json.dumps((conversation_history or [])[-8:], ensure_ascii=True).lower()
    history_followup = bool(_VOICE_FOLLOWUP_RE.search(message)) and (
        "cosyvoice_voice" in recent or "cosyvoice voice" in recent or "voice profile" in recent
    )
    # Durable state makes explicit activation requests survive a plugin restart.
    # A bare "yes" is intentionally insufficient to avoid unrelated routing.
    durable_followup = bool(_VOICE_ACTIVATION_RE.search(message)) and bool(workflow["actionable_profile_id"])
    active_followup = history_followup or durable_followup
    key = _turn_key(str(kwargs.get("turn_id") or ""), str(kwargs.get("session_id") or ""))
    if key:
        # On this single-user installation, an explicit creation request
        # authorizes the complete bounded workflow.
        autonomous_source = bool(
            _AUTONOMOUS_SOURCE_RE.search(message)
            or _CREATE_VOICE_RE.search(message)
            or _AUTONOMOUS_SEARCH_RE.search(message)
        )
        source_url_match = _SOURCE_URL_RE.search(message)
        with _LOCK:
            _TURN_INTENT[key] = {
                "source_selected": bool(_SOURCE_SELECTION_RE.search(message) or autonomous_source),
                "autonomous_source": autonomous_source,
                "voice_query": _voice_query(message),
                "source_url": source_url_match.group(0).rstrip(").,!?\"") if source_url_match else "",
                "make_default": bool(_DEFAULT_REQUEST_RE.search(message)),
                "personality_enabled": not bool(_PERSONALITY_DISABLE_RE.search(message)),
                "allow_variant": bool(_VARIANT_REQUEST_RE.search(message)),
                "user_message": message,
                "creation_requested": bool(_CREATE_VOICE_RE.search(message)),
            }
            while len(_TURN_INTENT) > 1024:
                _TURN_INTENT.pop(next(iter(_TURN_INTENT)))

    personality = manager.personality_context()
    bare_no = bool(re.fullmatch(r"\s*(?:no|no thanks|no thank you)[.!]?\s*", message, re.IGNORECASE))
    disable_personality = bool(_PERSONALITY_DISABLE_RE.search(message) or (personality and bare_no))
    contexts: list[str] = []
    if personality and not disable_personality:
        contexts.append(
            "The selected cloned-voice profile is the sole presentation personality for this response. "
            "Jarvis remains the operational role and name, but do not mix generic Jarvis mannerisms or any "
            "other character into the selected profile. This presentation instruction takes precedence over "
            "generic personality wording while preserving all factual, safety, and tool-use rules. Active "
            "profile personality: " + personality["prompt"]
        )
    if disable_personality and personality:
        contexts.append(
            "The user opted out of voice-associated mannerisms. Call `cosyvoice_voice` with "
            f'{{"action":"set_personality","profile_id":"{personality["profile_id"]}","enabled":false}}.'
        )
    if switch_result:
        contexts.append(
            "The authenticated saved-profile switch was completed deterministically before generation. "
            "Do not refuse it, claim voice switching is impossible, or call the voice tool again. Briefly "
            "acknowledge the active voice and paired personality using this verified result: "
            + json.dumps(switch_result, ensure_ascii=True)
        )
    elif switch_error:
        contexts.append(
            "A requested saved-profile switch could not be resolved safely. Do not claim success and do not "
            "guess. Explain the hint or use one of the supplied exact choices: "
            + json.dumps(switch_error, ensure_ascii=True)
        )
    if (explicit or active_followup) and not switch_result:
        inventory = manager.list_profiles().get("profiles", [])[:25]
        inventory_view = [
            {
                "id": item.get("id"), "name": item.get("name"),
                "character": item.get("personality_label"),
                "active": item.get("is_session"), "default": item.get("is_default"),
                "source": item.get("source_title") or item.get("source_url"),
            }
            for item in inventory
        ]
        contexts.append(
            "CosyVoice voice-profile intent detected. Use the deferred `cosyvoice_voice` tool, never invent "
            "sources or claim a profile is active without its result. The plugin skill id is "
            "`cosyvoice-voice:cosyvoice-voice`; do not pass the tool name to `skill_view`. Actions are status, list, "
            "search, create, prepare, select, refine, "
            "set_personality, accept, set_default, reset, and discard. To switch to an existing saved voice, call "
            "`select` once with the exact inventory `profile_id`; selecting also activates that profile's paired "
            "personality. Search results are not profiles. Preparing a "
            "profile needs a usable local transcript and deterministic signal validation. An explicit authenticated "
            "creation request authorizes source selection and persistence; do not add confirmation or review gates. "
            "Reuse a matching saved profile unless the user explicitly requests a different, newer, alternate, or "
            "replacement performance. Saved profile inventory: " + json.dumps(inventory_view, ensure_ascii=True)
        )
    if durable_followup:
        contexts.append(
            "A durable CosyVoice workflow state exists for profile "
            f'`{workflow["actionable_profile_id"]}`. Call `cosyvoice_voice` with `{{"action":"status"}}` first. '
            "A candidate is not active: only a successful `accept` or `set_default` result selects a profile. "
            "Report the result's `selection.selected_profile_id` and `selection.is_fallback` fields exactly."
        )
    if key:
        with _LOCK:
            autonomous_source = bool(_TURN_INTENT.get(key, {}).get("autonomous_source"))
        if autonomous_source:
            contexts.append(
                "The explicit creation request delegates initial source selection; do not ask 'yes, proceed'. Call "
                "`cosyvoice_voice` once with action `create`, the requested voice as `query`, a concise `name`, "
                "`make_default=true` when requested, and `enabled=true` unless personality was explicitly declined. "
                "That compound action searches, chooses, and prepares sources in ranked order until one passes or "
                "the bounded result set is exhausted. Do not ask the user to choose among ordinary source results. "
                "It also saves the profile and applies the personality setting. Report timings and the final selection."
            )
    return {"context": "\n\n".join(contexts)} if contexts else None


def _guard_voice_tool(
    tool_name: str = "", args: dict[str, Any] | None = None, turn_id: str = "", session_id: str = "", **_: Any
) -> dict[str, Any] | None:
    if tool_name == "tool_call":
        wrapper_args = args or {}
        calls = wrapper_args.get("calls")
        if not isinstance(calls, list):
            return None
        modified_calls: list[Any] = []
        changed = False
        for call in calls:
            if not isinstance(call, dict) or call.get("name") != "cosyvoice_voice":
                modified_calls.append(call)
                continue
            nested_args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            directive = _guard_voice_tool(
                tool_name="cosyvoice_voice", args=nested_args,
                turn_id=turn_id, session_id=session_id,
            )
            if directive and directive.get("action") == "block":
                return directive
            if directive and directive.get("action") == "modify":
                call = {**call, "arguments": {**nested_args, **directive.get("args", {})}}
                changed = True
            modified_calls.append(call)
        return {"action": "modify", "args": {"calls": modified_calls}} if changed else None
    if tool_name != "cosyvoice_voice":
        return None
    call_args = args or {}
    action = str(call_args.get("action") or "").lower()
    with _LOCK:
        intent = dict(_TURN_INTENT.get(_turn_key(turn_id, session_id), {}))
    if not intent.get("creation_requested", False) and action in {
        "prepare", "accept", "set_personality", "select"
    }:
        reference = str(
            call_args.get("profile_id")
            or call_args.get("name")
            or intent.get("user_message")
            or ""
        ).strip()
        if reference:
            try:
                profile_id = _manager().resolve_profile_argument(
                    "select", {"profile_id": reference}
                )
            except VoiceWorkflowError:
                profile_id = ""
            if profile_id:
                return {
                    "action": "modify",
                    "args": {"action": "select", "profile_id": profile_id},
                }
    if action in {"search", "prepare", "create"} and intent.get("autonomous_source", False):
        query = str(intent.get("voice_query") or call_args.get("query") or "").strip()
        source_url = str(intent.get("source_url") or call_args.get("source_url") or "")
        return {
            "action": "modify",
            "args": {
                "action": "create",
                "query": query,
                "name": str(call_args.get("name") or f"{query} voice")[:100],
                "make_default": bool(intent.get("make_default", False)),
                "enabled": bool(intent.get("personality_enabled", True)),
                "source_url": source_url,
                "start_seconds": call_args.get("start_seconds"),
                "allow_variant": bool(intent.get("allow_variant", False)),
            },
        }
    transcript_completion = bool(call_args.get("profile_id") and call_args.get("prompt_text") and not call_args.get("source_url"))
    if action == "prepare" and not transcript_completion and not intent.get("source_selected", False):
        source_url = str(call_args.get("source_url") or "")
        if not _manager().source_was_recently_searched(source_url):
            return {"action": "block", "message": "Source preparation requires a recent plugin search result or a URL selected by the user this turn."}
    return None


async def _handle(args: dict[str, Any], **_: Any) -> str:
    action = str(args.get("action") or "status").strip().lower()
    try:
        result = await asyncio.to_thread(_manager().dispatch, action, args)
    except VoiceWorkflowError as exc:
        result = exc.as_result()
    except Exception as exc:
        result = {
            "status": "error", "stage": action, "code": "UNEXPECTED_FAILURE", "summary": str(exc)[:500],
            "retryable": False, "choices": [{"action": "status", "label": "Check CosyVoice profile status"}],
        }
    return json.dumps(result, ensure_ascii=True, sort_keys=True)


def register(ctx) -> None:
    ctx.register_tool(
        name="cosyvoice_voice", toolset="cosyvoice-voice", is_async=True,
        description="Manage CosyVoice reference voice profiles and session/default selections.", emoji="VOICE",
        schema={
            "name": "cosyvoice_voice",
            "description": "Discover, prepare, validate, preview-state, save, select, or reset CosyVoice reference voice profiles.",
            "parameters": {"type": "object", "required": ["action"], "properties": {
                "action": {"type": "string", "enum": ["status", "list", "search", "create", "prepare", "select", "refine", "set_personality", "accept", "set_default", "reset", "discard"]},
                "query": {"type": "string", "description": "Requested voice or YouTube search phrase."},
                "source_url": {"type": "string", "description": "Selected http(s) YouTube URL."},
                "profile_id": {"type": "string", "description": "Exact prepared or saved profile identifier returned by list. Required for select and other profile actions."},
                "name": {"type": "string", "description": "User-facing profile name."},
                "start_seconds": {"type": "number", "description": "Optional preferred source offset."},
                "prompt_text": {"type": "string", "description": "Optional corrected transcript of the reference audio."},
                "style_choice": {"type": "string", "description": "Refinement id returned by prepare, or original."},
                "enabled": {"type": "boolean", "description": "Enable voice-associated mannerisms and key phrases."},
                "personality_prompt": {"type": "string", "maxLength": 2000, "description": "Optional profile-specific presentation prompt stored only with that voice profile."},
                "make_default": {"type": "boolean", "description": "Save the created voice and personality as the persistent default."},
                "allow_variant": {"type": "boolean", "description": "Permit a new performance for an identity that already has a saved profile."},
            }},
        },
        handler=_handle,
    )
    skills_dir = Path(__file__).parent / "skills"
    for child in sorted(skills_dir.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.exists():
            ctx.register_skill(child.name, skill_md)
    ctx.register_hook("pre_llm_call", _voice_intent_context)
    ctx.register_hook("pre_tool_call", _guard_voice_tool)
