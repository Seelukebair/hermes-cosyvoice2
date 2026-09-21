"""Bounded, in-memory state for continuous sentence synthesis streams."""

from __future__ import annotations

import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator


class SessionError(RuntimeError):
    """Base class for synthesis session lifecycle errors."""


class SessionNotFound(SessionError):
    pass


class SessionConflict(SessionError):
    pass


class SessionLimitExceeded(SessionError):
    pass


class SessionCancelled(SessionError):
    pass


class SessionTimedOut(SessionError):
    pass


@dataclass
class SynthesisSession:
    session_id: str
    prompt: object
    instruct: str
    speed: float
    created_at: float
    last_activity_at: float
    sentences: queue.Queue[str]
    queued_bytes: int
    total_bytes: int
    finished: bool = False
    stream_claimed: bool = False


@dataclass
class SynthesisSessionStore:
    """Coordinate short-lived synthesis sessions without retaining their text."""

    max_sessions: int
    max_queue_sentences: int
    max_text_bytes: int
    idle_timeout_seconds: float
    total_timeout_seconds: float
    clock: Callable[[], float] = time.monotonic
    cancellation_poll_seconds: float = 0.25
    _sessions: dict[str, SynthesisSession] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _janitor: threading.Thread | None = field(default=None, init=False)
    _stats: dict[str, int] = field(
        default_factory=lambda: {
            "created": 0,
            "completed": 0,
            "cancelled": 0,
            "errors": 0,
            "timed_out": 0,
            "rejected_capacity": 0,
            "rejected_queue": 0,
            "rejected_bytes": 0,
        },
        init=False,
    )

    def __post_init__(self) -> None:
        if self.max_sessions < 1 or self.max_queue_sentences < 1 or self.max_text_bytes < 1:
            raise ValueError("synthesis session limits must be positive")
        if self.idle_timeout_seconds <= 0 or self.total_timeout_seconds <= 0:
            raise ValueError("synthesis session timeouts must be positive")
        if self.cancellation_poll_seconds <= 0:
            raise ValueError("synthesis session cancellation poll must be positive")

    def start(self) -> None:
        with self._lock:
            if self._janitor is not None:
                return
            self._stop.clear()
            self._janitor = threading.Thread(
                target=self._run_janitor,
                name="cosyvoice-synthesis-session-janitor",
                daemon=True,
            )
            self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        janitor = self._janitor
        if janitor is not None:
            janitor.join(timeout=2)
        with self._lock:
            self._janitor = None
            for session_id in list(self._sessions):
                self._cleanup_locked(session_id, "cancelled")

    def create(self, text: str, *, prompt: object, instruct: str, speed: float) -> SynthesisSession:
        normalized = self._normalize(text)
        byte_count = len(normalized.encode("utf-8"))
        if byte_count > self.max_text_bytes:
            with self._lock:
                self._stats["rejected_bytes"] += 1
            raise SessionLimitExceeded("sentence exceeds the session text limit")
        with self._lock:
            self._expire_locked()
            if len(self._sessions) >= self.max_sessions:
                self._stats["rejected_capacity"] += 1
                raise SessionLimitExceeded("synthesis session capacity is full")
            now = self.clock()
            session = SynthesisSession(
                # HA and the bridge require an alphanumeric first character.
                # token_urlsafe() may begin with '-' or '_', so give every
                # session a stable route-safe prefix.
                session_id=f"s{secrets.token_urlsafe(24)}",
                prompt=prompt,
                instruct=instruct,
                speed=speed,
                created_at=now,
                last_activity_at=now,
                sentences=queue.Queue(maxsize=self.max_queue_sentences),
                queued_bytes=byte_count,
                total_bytes=byte_count,
            )
            session.sentences.put_nowait(normalized)
            self._sessions[session.session_id] = session
            self._stats["created"] += 1
            return session

    def enqueue(self, session_id: str, text: str) -> None:
        normalized = self._normalize(text)
        byte_count = len(normalized.encode("utf-8"))
        with self._lock:
            session = self._active_locked(session_id)
            if session.finished:
                raise SessionConflict("synthesis session is already finished")
            if session.total_bytes + byte_count > self.max_text_bytes:
                self._stats["rejected_bytes"] += 1
                raise SessionLimitExceeded("synthesis session text limit reached")
            if session.sentences.full():
                self._stats["rejected_queue"] += 1
                raise SessionLimitExceeded("synthesis session queue is full")
            session.sentences.put_nowait(normalized)
            session.queued_bytes += byte_count
            session.total_bytes += byte_count
            session.last_activity_at = self.clock()

    def finish(self, session_id: str) -> None:
        with self._lock:
            session = self._active_locked(session_id)
            session.finished = True
            session.last_activity_at = self.clock()

    def claim_stream(self, session_id: str) -> SynthesisSession:
        with self._lock:
            session = self._active_locked(session_id)
            if session.stream_claimed:
                raise SessionConflict("synthesis session audio is already claimed")
            session.stream_claimed = True
            session.last_activity_at = self.clock()
            return session

    def cancel(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFound("synthesis session was not found")
            self._cleanup_locked(session_id, "cancelled")

    def cleanup(self, session_id: str, outcome: str) -> None:
        with self._lock:
            self._cleanup_locked(session_id, outcome)

    def touch(self, session_id: str) -> None:
        with self._lock:
            session = self._active_locked(session_id)
            session.last_activity_at = self.clock()

    def iter_sentences(self, session_id: str) -> Iterator[str]:
        """Wait for complete sentences until finish, cancellation, or timeout."""
        while True:
            with self._lock:
                session = self._active_locked(session_id)
                if session.finished and session.sentences.empty():
                    return
                wait_seconds = min(
                    self._next_wait_seconds(session), self.cancellation_poll_seconds
                )
            try:
                sentence = session.sentences.get(timeout=wait_seconds)
            except queue.Empty:
                with self._lock:
                    self._active_locked(session_id)
                continue
            byte_count = len(sentence.encode("utf-8"))
            with self._lock:
                session = self._active_locked(session_id)
                session.queued_bytes -= byte_count
                session.last_activity_at = self.clock()
            yield sentence

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            self._expire_locked()
            return {"active": len(self._sessions), **self._stats}

    def _active_locked(self, session_id: str) -> SynthesisSession:
        self._expire_locked()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound("synthesis session was not found")
        return session

    def _expire_locked(self) -> None:
        now = self.clock()
        for session_id, session in list(self._sessions.items()):
            if now - session.created_at >= self.total_timeout_seconds or (
                now - session.last_activity_at >= self.idle_timeout_seconds
            ):
                self._cleanup_locked(session_id, "timed_out")

    def _next_wait_seconds(self, session: SynthesisSession) -> float:
        now = self.clock()
        idle_remaining = self.idle_timeout_seconds - (now - session.last_activity_at)
        total_remaining = self.total_timeout_seconds - (now - session.created_at)
        return max(0.001, min(idle_remaining, total_remaining))

    def _cleanup_locked(self, session_id: str, outcome: str) -> None:
        if self._sessions.pop(session_id, None) is None:
            return
        if outcome in self._stats:
            self._stats[outcome] += 1

    def _run_janitor(self) -> None:
        interval = min(1.0, self.idle_timeout_seconds, self.total_timeout_seconds)
        while not self._stop.wait(interval):
            with self._lock:
                self._expire_locked()

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("text is empty")
        return normalized
