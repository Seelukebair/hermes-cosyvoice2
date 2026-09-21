"""Tests for persisted-profile change warming."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from profile_warmer import ProfileWarmer, prompt_signature


class ProfileWarmerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.wav = Path(self.temp.name) / "reference.wav"
        self.wav.write_bytes(b"first")
        self.prompt = SimpleNamespace(
            profile_id="jarvis",
            prompt_wav=self.wav,
            prompt_text="reference words",
            instruct="",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_change_is_debounced_then_warmed_once(self) -> None:
        warmed = []
        warmer = ProfileWarmer(
            lambda: self.prompt,
            lambda prompt: warmed.append(prompt.profile_id) or False,
            interval_seconds=10,
            debounce_seconds=0.5,
        )

        self.assertFalse(warmer.poll_once(now=1.0))
        self.assertFalse(warmer.poll_once(now=1.4))
        self.assertTrue(warmer.poll_once(now=1.5))
        self.assertFalse(warmer.poll_once(now=2.0))
        self.assertEqual(["jarvis"], warmed)
        self.assertEqual("ok", warmer.snapshot()["status"])

    def test_same_id_reference_replacement_changes_signature(self) -> None:
        before = prompt_signature(self.prompt)
        self.wav.write_bytes(b"replacement reference")
        after = prompt_signature(self.prompt)
        self.assertNotEqual(before, after)

    def test_none_selection_is_not_warmed(self) -> None:
        warmer = ProfileWarmer(lambda: None, lambda _: self.fail("unexpected warm"))
        self.assertFalse(warmer.poll_once(now=1.0))
        self.assertEqual("watching", warmer.snapshot()["status"])


if __name__ == "__main__":
    unittest.main()
