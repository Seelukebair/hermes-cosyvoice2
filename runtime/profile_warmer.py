"""Warm saved voice conditioning after profile state or reference changes."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable


def prompt_signature(prompt) -> tuple | None:
    """Return a cheap change signature without retaining sensitive content."""
    if prompt is None:
        return None
    wav = Path(prompt.prompt_wav)
    stat = wav.stat()
    return (
        prompt.profile_id,
        stat.st_size,
        stat.st_mtime_ns,
        prompt.prompt_text,
        prompt.instruct,
    )


class ProfileWarmer:
    """Coalesce profile changes and warm the newest persisted selection."""

    def __init__(
        self,
        resolve: Callable,
        warm: Callable,
        *,
        interval_seconds: float = 0.75,
        debounce_seconds: float = 0.5,
        initial_signature: tuple | None = None,
    ) -> None:
        self._resolve = resolve
        self._warm = warm
        self._interval = interval_seconds
        self._debounce = debounce_seconds
        self._last_signature = initial_signature
        self._pending_signature = initial_signature
        self._pending_since = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: dict[str, object] = {
            "enabled": True,
            "status": "watching",
            "profile_id": None,
            "generation": 0,
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="cosyvoice-profile-warmer", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._status)

    def poll_once(self, now: float | None = None) -> bool:
        """Inspect once; return true only when a warm was attempted."""
        now = time.monotonic() if now is None else now
        try:
            prompt = self._resolve()
            signature = prompt_signature(prompt)
        except Exception:
            self._set_status("resolve_error")
            return False

        if signature == self._last_signature:
            self._pending_signature = signature
            return False
        if signature != self._pending_signature:
            self._pending_signature = signature
            self._pending_since = now
            self._set_status("change_detected", prompt)
            return False
        if now - self._pending_since < self._debounce:
            return False
        if prompt is None:
            self._last_signature = signature
            self._set_status("skipped_no_saved_profile")
            return False

        started = time.monotonic()
        try:
            cache_hit = bool(self._warm(prompt))
        except Exception:
            self._set_status("error", prompt, elapsed_ms=started)
        else:
            self._last_signature = signature
            self._set_status(
                "ok", prompt, elapsed_ms=started, cache_hit=cache_hit, increment=True
            )
        return True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll_once()

    def _set_status(
        self,
        status: str,
        prompt=None,
        *,
        elapsed_ms: float | None = None,
        cache_hit: bool | None = None,
        increment: bool = False,
    ) -> None:
        with self._lock:
            generation = int(self._status.get("generation", 0)) + int(increment)
            value: dict[str, object] = {
                "enabled": True,
                "status": status,
                "profile_id": getattr(prompt, "profile_id", None),
                "generation": generation,
            }
            if elapsed_ms is not None:
                value["elapsed_ms"] = round(
                    (time.monotonic() - elapsed_ms) * 1000, 2
                )
            if cache_hit is not None:
                value["cache_hit"] = cache_hit
            self._status = value
