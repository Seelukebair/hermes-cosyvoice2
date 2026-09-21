"""Resolve CosyVoice reference profiles without coupling Hermes to model code."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path


PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PROCESS_LOCK = threading.RLock()

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development fallback
    fcntl = None


@dataclass(frozen=True)
class VoicePrompt:
    profile_id: str
    prompt_wav: Path
    prompt_text: str
    instruct: str = ""
    consume_preview: bool = False


class VoiceProfileRegistry:
    """Read plugin-owned profiles and atomically consume one-shot previews."""

    def __init__(self, root: Path, fallback_wav: Path, fallback_text: str) -> None:
        self.root = Path(root)
        self.fallback = VoicePrompt("default", Path(fallback_wav), fallback_text)

    def resolve(self, requested: str) -> VoicePrompt:
        profile_id = str(requested or "default").strip().lower()
        consume_preview = False
        if profile_id == "default":
            state = self._read_json(self.root / "state.json")
            profile_id = str(state.get("session_profile") or state.get("default_profile") or "")
            # Prepared candidates are one-shot previews. An accepted session
            # profile is deliberately persistent until reset or replacement.
            consume_preview = bool(
                state.get("session_profile") and state.get("session_candidate")
            )
            if not profile_id:
                return self.fallback
        if not PROFILE_ID.fullmatch(profile_id):
            raise ValueError("invalid voice profile id")

        return self._resolve_profile(profile_id, ("candidates", "profiles"), consume_preview)

    def resolve_saved_default(self) -> VoicePrompt | None:
        """Return the persisted session/default voice without using previews."""
        state = self._read_json(self.root / "state.json")
        profile_ids = []
        if not state.get("session_candidate"):
            profile_ids.append(state.get("session_profile"))
        profile_ids.append(state.get("default_profile"))
        for raw_profile_id in profile_ids:
            profile_id = str(raw_profile_id or "").strip().lower()
            if not profile_id or not PROFILE_ID.fullmatch(profile_id):
                continue
            try:
                return self._resolve_profile(profile_id, ("profiles",), False)
            except ValueError:
                continue
        return None

    def _resolve_profile(
        self, profile_id: str, collections: tuple[str, ...], consume_preview: bool
    ) -> VoicePrompt:
        for collection in collections:
            directory = self.root / collection / profile_id
            metadata_path = directory / "profile.json"
            prompt_wav = directory / "reference.wav"
            if not metadata_path.is_file():
                continue
            metadata = self._read_json(metadata_path)
            prompt_text = str(
                metadata.get("prompt_text")
                or (metadata.get("transcript") or {}).get("text")
                or ""
            ).strip()
            if not prompt_wav.is_file() or not prompt_text:
                raise ValueError(f"voice profile {profile_id!r} is incomplete")
            return VoicePrompt(
                profile_id=profile_id,
                prompt_wav=prompt_wav,
                prompt_text=prompt_text,
                instruct=str(metadata.get("delivery_prompt") or "").strip(),
                consume_preview=consume_preview,
            )
        raise ValueError(f"voice profile not found: {profile_id}")

    def consume(self, prompt: VoicePrompt) -> None:
        """Clear only the preview that was resolved, preserving newer writes."""
        if not prompt.consume_preview:
            return
        state_path = self.root / "state.json"
        with self._state_lock():
            state = self._read_json(state_path)
            if state.get("session_profile") != prompt.profile_id:
                return
            state["session_profile"] = None
            state["session_candidate"] = False
            self._write_json(state_path, state)

    @contextmanager
    def _state_lock(self):
        lock_path = self.root / ".state.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _PROCESS_LOCK, lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid voice profile metadata: {path.name}") from exc

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
