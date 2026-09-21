import threading
import unittest

from upstream_guard import install_background_generation_guard


class FakeRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.tts_speech_token_dict = {"request-1": [1]}
        self.llm_end_dict = {"request-1": False}
        self.mel_overlap_dict = {"request-1": object()}
        self.hift_cache_dict = {"request-1": object()}
        self.flow_cache_dict = {"request-1": object()}

    def llm_job(self, *args):
        raise ValueError("sampling failed")


class BackgroundGenerationGuardTests(unittest.TestCase):
    def test_failure_signals_completion_and_cleans_request_state(self):
        runtime = FakeRuntime()
        guard = install_background_generation_guard(runtime)

        runtime.llm_job(None, "request-1")

        self.assertTrue(runtime.llm_end_dict["request-1"])
        with self.assertRaisesRegex(RuntimeError, "background generation failed"):
            guard.pop_and_raise()
        for name in guard._STATE_DICTIONARIES:
            self.assertNotIn("request-1", getattr(runtime, name))

    def test_success_preserves_upstream_behavior(self):
        runtime = FakeRuntime()
        runtime.llm_job = lambda *args: "ok"
        guard = install_background_generation_guard(runtime)

        self.assertEqual("ok", runtime.llm_job(None, "request-1"))
        guard.pop_and_raise()


if __name__ == "__main__":
    unittest.main()
