from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from streaming_audio import iter_streaming_wav, pcm16_bytes, streaming_wav_header
from synthesis_sessions import (
    SessionConflict,
    SessionLimitExceeded,
    SessionNotFound,
    SynthesisSessionStore,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class SynthesisSessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()

    def store(self, **overrides) -> SynthesisSessionStore:
        options = {
            "max_sessions": 2,
            "max_queue_sentences": 3,
            "max_text_bytes": 32,
            "idle_timeout_seconds": 10.0,
            "total_timeout_seconds": 30.0,
            "clock": self.clock,
        }
        options.update(overrides)
        return SynthesisSessionStore(**options)

    def test_one_wav_header_carries_pcm_for_multiple_finished_sentences(self):
        store = self.store()
        session = store.create("first.", prompt=object(), instruct="", speed=1.0)
        store.enqueue(session.session_id, "second.")
        store.finish(session.session_id)

        rendered = {
            "first.": np.array([0.0, 0.5], dtype=np.float32),
            "second.": np.array([-0.5, 1.0], dtype=np.float32),
        }

        def sentence_audio():
            for sentence in store.iter_sentences(session.session_id):
                yield rendered[sentence]

        output = list(iter_streaming_wav(sentence_audio(), 24000, 1.0))
        self.assertEqual(streaming_wav_header(24000), output[0])
        self.assertEqual(
            pcm16_bytes(np.concatenate([rendered["first."], rendered["second."]])),
            b"".join(output[1:]),
        )
        store.cleanup(session.session_id, "completed")
        self.assertEqual(1, store.snapshot()["completed"])

    def test_capacity_queue_and_total_byte_limits_are_enforced(self):
        store = self.store(max_sessions=1, max_queue_sentences=1)
        session = store.create("one", prompt=object(), instruct="", speed=1.0)
        with self.assertRaises(SessionLimitExceeded):
            store.create("two", prompt=object(), instruct="", speed=1.0)
        with self.assertRaises(SessionLimitExceeded):
            store.enqueue(session.session_id, "two")

        store.cleanup(session.session_id, "cancelled")
        byte_limited = self.store(max_text_bytes=5)
        session = byte_limited.create("one", prompt=object(), instruct="", speed=1.0)
        with self.assertRaises(SessionLimitExceeded):
            byte_limited.enqueue(session.session_id, "two")
        self.assertEqual(1, byte_limited.snapshot()["rejected_bytes"])

    def test_finish_rejects_future_text_after_draining_queued_sentences(self):
        store = self.store()
        session = store.create("first", prompt=object(), instruct="", speed=1.0)
        store.enqueue(session.session_id, "second")
        store.finish(session.session_id)
        self.assertEqual(["first", "second"], list(store.iter_sentences(session.session_id)))
        with self.assertRaises(SessionConflict):
            store.enqueue(session.session_id, "third")

    def test_cancellation_and_disconnect_cleanup_release_the_session(self):
        store = self.store(max_sessions=1)
        session = store.create("first", prompt=object(), instruct="", speed=1.0)
        store.claim_stream(session.session_id)
        source = store.iter_sentences(session.session_id)
        self.assertEqual("first", next(source))
        # The response wrapper performs this cleanup when the client closes.
        source.close()
        store.cleanup(session.session_id, "cancelled")
        self.assertEqual(0, store.snapshot()["active"])
        self.assertEqual(1, store.snapshot()["cancelled"])

        replacement = store.create("next", prompt=object(), instruct="", speed=1.0)
        store.cancel(replacement.session_id)
        with self.assertRaises(SessionNotFound):
            store.claim_stream(replacement.session_id)

    def test_cancellation_wakes_a_waiting_sentence_stream_promptly(self):
        store = self.store(
            idle_timeout_seconds=30.0,
            total_timeout_seconds=60.0,
            clock=time.monotonic,
            cancellation_poll_seconds=0.02,
        )
        session = store.create("first", prompt=object(), instruct="", speed=1.0)
        source = store.iter_sentences(session.session_id)
        self.assertEqual("first", next(source))
        stopped = threading.Event()

        def wait_for_more() -> None:
            try:
                next(source)
            except (SessionNotFound, StopIteration):
                stopped.set()

        waiter = threading.Thread(target=wait_for_more)
        waiter.start()
        store.cancel(session.session_id)
        self.assertTrue(stopped.wait(0.5))
        waiter.join(timeout=0.5)
        self.assertFalse(waiter.is_alive())

    def test_idle_and_total_timeouts_release_capacity_for_the_next_request(self):
        store = self.store(max_sessions=1, idle_timeout_seconds=2.0, total_timeout_seconds=5.0)
        idle = store.create("idle", prompt=object(), instruct="", speed=1.0)
        self.clock.now = 2.1
        with self.assertRaises(SessionNotFound):
            store.claim_stream(idle.session_id)
        self.assertEqual(1, store.snapshot()["timed_out"])

        total_store = self.store(
            max_sessions=1, idle_timeout_seconds=50.0, total_timeout_seconds=5.0
        )
        total = total_store.create("total", prompt=object(), instruct="", speed=1.0)
        self.clock.now = 7.2
        with self.assertRaises(SessionNotFound):
            total_store.finish(total.session_id)
        self.assertEqual(1, total_store.snapshot()["timed_out"])
        self.assertIsNotNone(
            total_store.create("recovered", prompt=object(), instruct="", speed=1.0)
        )


if __name__ == "__main__":
    unittest.main()
