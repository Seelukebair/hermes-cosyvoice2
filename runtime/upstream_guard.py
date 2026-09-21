"""Bound failures from CosyVoice's internal generation thread."""

from __future__ import annotations

import threading
import types


class BackgroundGenerationGuard:
    """Make background LLM failures visible to the response generator."""

    _STATE_DICTIONARIES = (
        "tts_speech_token_dict",
        "llm_end_dict",
        "mel_overlap_dict",
        "hift_cache_dict",
        "flow_cache_dict",
    )

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self._errors: dict[str, BaseException] = {}
        self._errors_lock = threading.Lock()

    def record(self, request_id: str, error: BaseException) -> None:
        with self._errors_lock:
            self._errors[request_id] = error
        # Upstream waits on this flag and otherwise spins forever when its LLM
        # thread exits with an exception.
        with self.runtime.lock:
            if request_id in self.runtime.llm_end_dict:
                self.runtime.llm_end_dict[request_id] = True

    def pop_and_raise(self) -> None:
        with self._errors_lock:
            if not self._errors:
                return
            request_id, error = self._errors.popitem()
        with self.runtime.lock:
            for name in self._STATE_DICTIONARIES:
                values = getattr(self.runtime, name, None)
                if isinstance(values, dict):
                    values.pop(request_id, None)
        raise RuntimeError("CosyVoice background generation failed") from error


def install_background_generation_guard(runtime) -> BackgroundGenerationGuard:
    """Wrap one runtime instance without modifying the pinned upstream tree."""
    guard = BackgroundGenerationGuard(runtime)
    original_llm_job = runtime.llm_job

    def guarded_llm_job(self, *args, **kwargs):
        request_id = str(args[-1] if args else kwargs["uuid"])
        try:
            return original_llm_job(*args, **kwargs)
        except BaseException as error:
            guard.record(request_id, error)
            return None

    runtime.llm_job = types.MethodType(guarded_llm_job, runtime)
    runtime._jarvis_generation_guard = guard
    return guard
