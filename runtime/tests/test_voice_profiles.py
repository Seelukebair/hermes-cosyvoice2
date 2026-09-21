import json
import tempfile
import unittest
from pathlib import Path

from voice_profiles import VoiceProfileRegistry


class VoiceProfileRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fallback = self.root / "fallback.wav"
        self.fallback.write_bytes(b"RIFF")
        self.registry = VoiceProfileRegistry(self.root, self.fallback, "fallback text")

    def tearDown(self):
        self.temporary.cleanup()

    def profile(self, collection, profile_id, prompt_text="reference words"):
        directory = self.root / collection / profile_id
        directory.mkdir(parents=True)
        (directory / "reference.wav").write_bytes(b"RIFF")
        (directory / "profile.json").write_text(
            json.dumps({"prompt_text": prompt_text, "delivery_prompt": "Speak warmly."}),
            encoding="utf-8",
        )

    def state(self, **values):
        (self.root / "state.json").write_text(json.dumps(values), encoding="utf-8")

    def test_fallback_without_selection(self):
        prompt = self.registry.resolve("default")
        self.assertEqual("default", prompt.profile_id)
        self.assertEqual("fallback text", prompt.prompt_text)

    def test_preview_precedes_default_and_is_consumed_once(self):
        self.profile("candidates", "preview-one")
        self.profile("profiles", "saved-one", "saved words")
        self.state(session_profile="preview-one", session_candidate=True, default_profile="saved-one")
        prompt = self.registry.resolve("default")
        self.assertEqual("preview-one", prompt.profile_id)
        self.assertTrue(prompt.consume_preview)
        self.registry.consume(prompt)
        self.assertEqual("saved-one", self.registry.resolve("default").profile_id)

    def test_explicit_profile_does_not_consume_preview(self):
        self.profile("candidates", "preview-one")
        self.state(session_profile="preview-one", session_candidate=True)
        prompt = self.registry.resolve("preview-one")
        self.assertFalse(prompt.consume_preview)
        self.registry.consume(prompt)
        self.assertEqual("preview-one", self.registry.resolve("default").profile_id)

    def test_rejects_traversal_and_incomplete_profiles(self):
        with self.assertRaisesRegex(ValueError, "invalid voice profile id"):
            self.registry.resolve("../secret")
        self.profile("profiles", "missing-text", "")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.registry.resolve("missing-text")


if __name__ == "__main__":
    unittest.main()
