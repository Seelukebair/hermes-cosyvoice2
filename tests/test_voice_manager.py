from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import types
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

from voice_manager import VoiceManager, _character_label


class VoiceManagerTests(unittest.TestCase):
    def test_reference_window_uses_quality_oriented_duration(self) -> None:
        self.assertEqual(12, VoiceManager.TARGET_SECONDS)
        self.assertEqual(10, VoiceManager.MIN_REFERENCE_SECONDS)
        self.assertEqual(15, VoiceManager.MAX_REFERENCE_SECONDS)
        self.assertEqual(86400, VoiceManager.CANDIDATE_MAX_AGE_SECONDS)

    def test_stale_candidates_are_pruned_but_active_candidate_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = VoiceManager(root, "/missing/yt-dlp")
            stale_id = self._candidate(manager)
            stale = manager.candidates / stale_id / "profile.json"
            data = manager._read_json(stale)
            data["created_at"] = int(time.time()) - manager.CANDIDATE_MAX_AGE_SECONDS - 1
            manager._write_json(stale, data)
            VoiceManager(root, "/missing/yt-dlp")
            self.assertFalse(stale.parent.exists())

            active_id = self._candidate(manager)
            active = manager.candidates / active_id / "profile.json"
            data = manager._read_json(active)
            data["created_at"] = int(time.time()) - manager.CANDIDATE_MAX_AGE_SECONDS - 1
            manager._write_json(active, data)
            manager._set_state(session_profile=active_id, candidate=True)
            VoiceManager(root, "/missing/yt-dlp")
            self.assertTrue(active.parent.exists())

    def test_generated_personality_is_paired_character_not_jarvis_blend(self) -> None:
        prompt = VoiceManager._personality_prompt("Optimus Prime")
        self.assertIn("Use Optimus Prime as the sole presentation persona", prompt)
        self.assertIn("answering the user directly", prompt)
        self.assertIn("never begin with phrases such as 'As Optimus Prime'", prompt)
        self.assertIn("Do not turn routine answers into speeches", prompt)
        self.assertIn("Use recognizable catchphrases sparingly", prompt)
        self.assertNotIn("force catchphrases", prompt)
        self.assertIn("do not blend in another assistant or character persona", prompt)
        self.assertIn("Jarvis remains the operational role and name", prompt)

    def test_character_label_rejects_source_urls_and_variant_suffixes(self) -> None:
        self.assertEqual(
            "Samuel L. Jackson",
            _character_label("https://www.youtube.com/watch?v=example", "Samuel L. Jackson (Pulp)"),
        )
        self.assertEqual("Optimus Prime", _character_label("Optimus Prime voice", "Ignored"))

    def test_reference_validation_enforces_quality_duration_range(self) -> None:
        valid = {
            "duration_seconds": 10.0, "rms": 900.0,
            "voiced_ratio": 0.5, "clipping_ratio": 0.0,
        }
        VoiceManager._validate_metrics(valid)
        with self.assertRaisesRegex(Exception, "failed deterministic signal checks"):
            VoiceManager._validate_metrics({**valid, "duration_seconds": 9.99})
        with self.assertRaisesRegex(Exception, "failed deterministic signal checks"):
            VoiceManager._validate_metrics({**valid, "duration_seconds": 15.01})

    def test_reference_cleaner_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp) / "profiles", "/missing/yt-dlp")
            source = Path(tmp) / "raw.wav"
            output = Path(tmp) / "clean.wav"
            self._speech_like_wav(source)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("COSYVOICE_CLEAN_REFERENCE_COMMAND", None)
                result = manager._clean_reference(source, output)
            self.assertEqual("not_configured", result["status"])
            self.assertTrue(output.is_file())
            self.assertFalse(source.exists())

    def test_reference_cleaner_falls_back_when_command_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp) / "profiles", "/missing/yt-dlp")
            source = Path(tmp) / "raw.wav"
            output = Path(tmp) / "clean.wav"
            self._speech_like_wav(source)
            with mock.patch.dict(
                os.environ,
                {"COSYVOICE_CLEAN_REFERENCE_COMMAND": "missing-cleaner {input_path} {output_path}"},
            ):
                result = manager._clean_reference(source, output)
            self.assertEqual("fallback_raw", result["status"])
            self.assertTrue(output.is_file())

    def test_reference_cleaner_accepts_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp) / "profiles", "/missing/yt-dlp")
            source = Path(tmp) / "raw.wav"
            output = Path(tmp) / "clean.wav"
            helper = Path(tmp) / "copy_cleaner.py"
            self._speech_like_wav(source)
            helper.write_text(
                "import shutil,sys\nshutil.copy2(sys.argv[1], sys.argv[2])\n",
                encoding="utf-8",
            )
            command = f'"{sys.executable}" "{helper}" {{input_path}} {{output_path}}'
            with mock.patch.dict(os.environ, {"COSYVOICE_CLEAN_REFERENCE_COMMAND": command}):
                result = manager._clean_reference(source, output)
            self.assertEqual("applied", result["status"])
            self.assertTrue(output.is_file())
            self.assertFalse(source.exists())

    def _speech_like_wav(self, path: Path, rate: int = 24000) -> None:
        samples = array("h")
        for index in range(rate * 10):
            envelope = 0.45 + 0.25 * math.sin(2 * math.pi * 2.2 * index / rate)
            samples.append(int(9000 * envelope * math.sin(2 * math.pi * 180 * index / rate)))
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(samples.tobytes())

    def test_segment_scan_handles_quiet_stereo_speech_in_frame_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            rate = 24000
            interleaved = array("h")
            for index in range(rate * 18):
                sample = int(900 * math.sin(2 * math.pi * 180 * index / rate))
                interleaved.extend((sample, sample))
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(interleaved.tobytes())

            segment = VoiceManager(Path(tmp), "/missing/yt-dlp")._choose_segment(path)
            self.assertLess(segment["score"], 0.12)
            self.assertGreater(segment["metrics"]["rms"], 250)
            self.assertLessEqual(segment["start_seconds"], 6)

    def _candidate(self, manager: VoiceManager, prompt_text: str = "exact words") -> str:
        profile = manager.candidates / "narrator"
        profile.mkdir(parents=True)
        self._speech_like_wav(profile / "reference.wav")
        manager._write_json(profile / "profile.json", {
            "id": "narrator", "name": "Narrator", "status": "preview", "selected_style": "original",
            "reference_wav": "reference.wav", "prompt_text": prompt_text,
            "transcript": {"text": prompt_text, "verified": bool(prompt_text), "source": "user", "error": None},
            "cosyvoice": {"reference_wav": "reference.wav", "sample_rate": 24000},
            "refinements": [{"id": "calmer", "label": "Calmer", "prompt": "Speak calmly."}],
            "personality": {"enabled": True, "label": "Narrator", "prompt": "Use narrator mannerisms sparingly."},
        })
        return "narrator"

    def test_atomic_state_and_metadata_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager._set_state(session_profile=profile_id, candidate=True)
            data = manager._read_json(manager.candidates / profile_id / "profile.json")
            self.assertEqual("reference.wav", data["cosyvoice"]["reference_wav"])
            self.assertEqual("exact words", data["prompt_text"])
            self.assertTrue(data["transcript"]["verified"])
            self.assertFalse(list(manager.root.rglob(".*.json.*")))

    def test_explicit_request_saves_default_without_secondary_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager._set_state(session_profile=profile_id, candidate=True)
            saved = manager.accept(profile_id, "original", False, True)
            self.assertEqual("saved", saved["status"])
            self.assertEqual(profile_id, saved["selection"]["selected_profile_id"])
            self.assertEqual("session+default", saved["selection"]["scope"])
            self.assertTrue(saved["persistence_verified"])
            self.assertFalse(saved["selection"]["is_fallback"])
            self.assertEqual(profile_id, manager.status()["state"]["default_profile"])
            manager.reset()
            state = manager.status()["state"]
            self.assertIsNone(state["session_profile"])
            self.assertEqual(profile_id, state["default_profile"])
            self.assertEqual(profile_id, manager.status()["selection"]["selected_profile_id"])

    def test_saved_selection_and_personality_survive_manager_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager.accept(profile_id, "original", True, True)
            reloaded = VoiceManager(Path(tmp), "/missing/yt-dlp")
            self.assertEqual(profile_id, reloaded.workflow_context()["actionable_profile_id"])
            self.assertEqual(profile_id, reloaded.status()["selection"]["selected_profile_id"])
            self.assertFalse(reloaded.status()["selection"]["is_fallback"])
            self.assertEqual(profile_id, reloaded.personality_context()["profile_id"])

    def test_create_reuses_saved_identity_instead_of_duplicating_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager.accept(profile_id, "original", True, False)
            manager.search = lambda _: self.fail("a matching saved identity must be reused")
            result = manager.create("Narrator", "Narrator")
            self.assertTrue(result["reused_existing"])
            self.assertEqual("same_identity", result["duplicate_reason"])
            self.assertEqual(profile_id, result["selection"]["selected_profile_id"])

    def test_create_reuses_same_youtube_source_even_when_variant_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            source = manager.candidates / profile_id / "profile.json"
            data = manager._read_json(source)
            data["source"] = {"url": "https://youtu.be/source123"}
            manager._write_json(source, data)
            manager.accept(profile_id, "original", True, False)
            manager.prepare = lambda *args, **kwargs: self.fail("an identical source must not be downloaded twice")
            result = manager.create(
                "New Narrator", "New Narrator", source_url="https://www.youtube.com/watch?v=source123",
                allow_variant=True,
            )
            self.assertTrue(result["reused_existing"])
            self.assertEqual("same_source", result["duplicate_reason"])

    def test_supplied_source_cannot_create_duplicate_persona(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager.accept(profile_id, "original", True, False)
            manager.prepare = lambda *args, **kwargs: self.fail("duplicate persona must be reused")
            result = manager.create(
                "Narrator", "Narrator", source_url="https://youtu.be/a-different-source",
                allow_variant=True,
            )
            self.assertTrue(result["reused_existing"])
            self.assertEqual("same_identity", result["duplicate_reason"])

    def test_profile_inventory_exposes_identity_source_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            source = manager.candidates / profile_id / "profile.json"
            data = manager._read_json(source)
            data["source"] = {"url": "https://youtu.be/source123", "title": "Recent movie speech"}
            data["personality"].update({"mode": "character", "paired_with_voice": True})
            manager._write_json(source, data)
            manager.accept(profile_id, "original", True, True)
            profile = manager.list_profiles()["profiles"][0]
            self.assertEqual("Narrator", profile["personality_label"])
            self.assertEqual("Recent movie speech", profile["source_title"])
            self.assertTrue(profile["paired_with_voice"])
            self.assertTrue(profile["is_session"])
            self.assertTrue(profile["is_default"])

    def test_missing_exact_transcript_cannot_be_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager, prompt_text="")
            result = manager.accept(profile_id, "original", True, False)
            self.assertEqual("needs_choice", result["status"])
            self.assertEqual("prompt_transcript", result["stage"])

    def test_verified_transcript_promotes_candidate_to_preview_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager, prompt_text="")
            result = manager.prepare("", "", "", prompt_text="Exact spoken words.", profile_id=profile_id)
            self.assertEqual("preview_ready", result["status"])
            self.assertEqual("Exact spoken words.", result["profile"]["prompt_text"])
            self.assertTrue(result["profile"]["transcript"]["verified"])

    def test_accepted_automatic_transcript_can_be_saved_without_false_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            path = manager.candidates / profile_id / "profile.json"
            data = manager._read_json(path)
            data["transcript"].update({"accepted": True, "verified": False, "source": "automatic_asr"})
            manager._write_json(path, data)
            result = manager.accept(profile_id, "original", True, False)
            self.assertEqual("saved", result["status"])
            self.assertFalse(result["profile"]["transcript"]["verified"])

    def test_empty_transcript_update_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager, prompt_text="")
            with self.assertRaisesRegex(Exception, "exact spoken transcript") as raised:
                manager.prepare("", "", "", profile_id=profile_id)
            self.assertEqual("TRANSCRIPT_REQUIRED", raised.exception.code)

    def test_profile_id_cannot_escape_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            with self.assertRaisesRegex(Exception, "profile id is invalid") as raised:
                manager.refine("../outside", "original")
            self.assertEqual("INVALID_PROFILE_ID", raised.exception.code)

    def test_local_transcription_adapter_preserves_success_and_failure_detail(self) -> None:
        package = types.ModuleType("tools")
        package.__path__ = []
        module = types.ModuleType("tools.transcription_tools")
        module.transcribe_audio_local_fallback = lambda _: {"text": " Draft transcript. "}
        old_package, old_module = sys.modules.get("tools"), sys.modules.get("tools.transcription_tools")
        sys.modules["tools"], sys.modules["tools.transcription_tools"] = package, module
        try:
            text, error = VoiceManager._draft_transcript(Path("reference.wav"))
            self.assertEqual("Draft transcript.", text)
            self.assertIsNone(error)
            module.transcribe_audio_local_fallback = lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable"))
            text, error = VoiceManager._draft_transcript(Path("reference.wav"))
            self.assertEqual("", text)
            self.assertIn("model unavailable", error)
        finally:
            if old_package is None:
                sys.modules.pop("tools", None)
            else:
                sys.modules["tools"] = old_package
            if old_module is None:
                sys.modules.pop("tools.transcription_tools", None)
            else:
                sys.modules["tools.transcription_tools"] = old_module

    def test_configured_transcription_command_is_public_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "transcribe.py"
            script.write_text('print("{\\"text\\": \\" Adapter transcript. \\"}")\n', encoding="utf-8")
            old = os.environ.get("COSYVOICE_TRANSCRIBE_COMMAND")
            os.environ["COSYVOICE_TRANSCRIBE_COMMAND"] = (
                f'"{Path(sys.executable).as_posix()}" "{script.as_posix()}" --input {{input_path}}'
            )
            try:
                text, error = VoiceManager._draft_transcript(Path(tmp) / "reference.wav")
            finally:
                if old is None:
                    os.environ.pop("COSYVOICE_TRANSCRIBE_COMMAND", None)
                else:
                    os.environ["COSYVOICE_TRANSCRIBE_COMMAND"] = old
            self.assertEqual("Adapter transcript.", text)
            self.assertIsNone(error)

    def test_refine_personality_and_discard_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            manager._set_state(session_profile=profile_id, candidate=True)
            result = manager.refine(profile_id, "calmer")
            self.assertIn("Speak calmly.", result["profile"]["style_prompt"])
            result = manager.set_personality(profile_id, False)
            self.assertEqual("Speak calmly.", result["profile"]["style_prompt"])
            manager.discard(profile_id)
            self.assertIsNone(manager.status()["state"]["session_profile"])

    def test_custom_personality_prompt_is_profile_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            result = manager.set_personality(
                profile_id,
                True,
                "Use a serious theatrical delivery.  Say motherfucker naturally.",
            )
            personality = result["profile"]["personality"]
            self.assertEqual("custom", personality["prompt_source"])
            self.assertIn("motherfucker", personality["prompt"])
            self.assertEqual(personality["prompt"], result["profile"]["style_prompt"])

    def test_search_dependency_and_error_choices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            with self.assertRaisesRegex(Exception, "yt-dlp") as raised:
                manager.search("example narrator")
            self.assertEqual("provide_url", raised.exception.choices[0]["action"])

    def test_recent_search_source_is_durable_bounded_and_reject_aware(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            manager._remember_recent_sources([
                {"id": "good", "source_url": "https://youtu.be/good", "suitability": "review"},
                {"id": "bad", "source_url": "https://youtu.be/bad", "suitability": "reject"},
            ], "example voice")

            reloaded = VoiceManager(Path(tmp), "/missing/yt-dlp")
            self.assertTrue(reloaded.source_was_recently_searched("https://youtu.be/good"))
            self.assertFalse(reloaded.source_was_recently_searched("https://youtu.be/bad"))
            self.assertFalse(reloaded.source_was_recently_searched("https://youtu.be/unknown"))

            record = reloaded._read_json(reloaded.recent_sources_file)
            record["created_at"] = int(time.time()) - reloaded.RECENT_SOURCE_MAX_AGE_SECONDS - 1
            reloaded._write_json(reloaded.recent_sources_file, record)
            self.assertFalse(reloaded.source_was_recently_searched("https://youtu.be/good"))

    def test_create_selects_best_unflagged_source_and_prepares_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            manager.search = lambda query: {
                "choices": [
                    {"id": "bad", "source_url": "https://youtu.be/bad", "confidence": 0.95, "suitability": "reject"},
                    {"id": "best", "source_url": "https://youtu.be/best", "confidence": 0.81, "suitability": "review"},
                ],
                "timings_ms": {"search_metadata": 123},
            }
            captured = {}
            profile_id = self._candidate(manager)
            manager.prepare = lambda source_url, query, name: captured.update(
                source_url=source_url, query=query, name=name
            ) or {"status": "preview_ready", "profile": {"id": profile_id}, "timings_ms": {"prepare_total": 456}}
            manager.accept = lambda profile_id, style, authorized, make_default: {
                "status": "saved", "profile": {"id": profile_id, "refinements": []},
                "selection": {"selected_profile_id": profile_id}, "default": make_default,
            }
            result = manager.create("Jarvis voice", "Jarvis", make_default=True)
            self.assertEqual("https://youtu.be/best", captured["source_url"])
            self.assertEqual("automatic", result["source_selection"]["mode"])
            self.assertEqual(123, result["timings_ms"]["search_metadata"])
            self.assertTrue(result["default"])

    def test_create_uses_provided_url_without_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            profile_id = self._candidate(manager)
            captured = {}
            manager.search = lambda _: self.fail("provided URL must not invoke search")
            manager.prepare = lambda source_url, query, name, start_seconds=None: captured.update(
                source_url=source_url, query=query, name=name, start_seconds=start_seconds
            ) or {"status": "preview_ready", "profile": {"id": profile_id}, "timings_ms": {"prepare_total": 456}}
            result = manager.create(
                "Optimus Prime", "Optimus Prime", make_default=True,
                source_url="https://youtu.be/known", start_seconds=42,
            )
            self.assertEqual("https://youtu.be/known", captured["source_url"])
            self.assertEqual(42, captured["start_seconds"])
            self.assertEqual("provided_url", result["source_selection"]["mode"])
            self.assertTrue(result["default"])

    def test_create_enriches_search_for_voice_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceManager(Path(tmp), "/missing/yt-dlp")
            observed = {}
            manager.search = lambda query: observed.update(query=query) or {"choices": []}
            with self.assertRaises(Exception):
                manager.create("Optimus Prime", "Optimus Prime", make_default=True)
            self.assertEqual("Optimus Prime voice", observed["query"])

    def test_objectionable_and_compilation_sources_are_flagged(self) -> None:
        self.assertIn("objectionable_source", VoiceManager._source_flags({"title": "Racist commercial"}))
        self.assertIn("compilation_or_music", VoiceManager._source_flags({"title": "Vintage radio playlist"}))
        self.assertIn("commentary_or_review", VoiceManager._source_flags({"title": "My Serious PROBLEM With Optimus Prime"}))

    def test_theatrical_character_sources_outrank_commentary(self) -> None:
        theatrical = VoiceManager._source_confidence(
            {"title": "JARVIS movie scene voice lines", "duration": 90},
            "JARVIS voice",
            0,
        )
        commentary = VoiceManager._source_confidence(
            {"title": "JARVIS movie reaction and gameplay", "duration": 90},
            "JARVIS voice",
            0,
        )
        self.assertGreater(theatrical, commentary)


if __name__ == "__main__":
    unittest.main()
