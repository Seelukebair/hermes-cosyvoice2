from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


def _load_plugin():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("cosyvoice_voice", root / "__init__.py", submodule_search_locations=[str(root)])
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IntentHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.old_root = os.environ.get("COSYVOICE_VOICE_ROOT")
        os.environ["COSYVOICE_VOICE_ROOT"] = self.temporary.name
        self.addCleanup(self._restore_root)

    def _restore_root(self) -> None:
        if self.old_root is None:
            os.environ.pop("COSYVOICE_VOICE_ROOT", None)
        else:
            os.environ["COSYVOICE_VOICE_ROOT"] = self.old_root

    def test_unconfigured_runtime_does_not_create_storage_for_normal_chat(self) -> None:
        missing = Path(self.temporary.name) / "missing"
        os.environ["COSYVOICE_VOICE_ROOT"] = str(missing)
        hook = _load_plugin()._voice_intent_context
        self.assertIsNone(hook(user_message="What is the weather?"))
        self.assertFalse(missing.exists())
        result = hook(user_message="Clone a new voice.")
        self.assertIn("runtime must be installed", result["context"])
        self.assertFalse(missing.exists())

    def test_explicit_intent_is_routed_and_unrelated_tts_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("COSYVOICE_VOICE_ROOT")
            os.environ["COSYVOICE_VOICE_ROOT"] = tmp
            try:
                hook = _load_plugin()._voice_intent_context
                self.assertIn("cosyvoice_voice", hook(user_message="Change your Jarvis voice to a radio announcer.")["context"])
                self.assertIn("cosyvoice_voice", hook(user_message="Create the voice/personality of a classic radio host.")["context"])
                self.assertIn("cosyvoice_voice", hook(user_message="Build a Jarvis voice profile with mannerisms.")["context"])
                self.assertIn("cosyvoice_voice", hook(user_message="Clone Optimus Prime's voice again.")["context"])
                self.assertIsNone(hook(user_message="What TTS providers are available?"))
            finally:
                if old is None:
                    os.environ.pop("COSYVOICE_VOICE_ROOT", None)
                else:
                    os.environ["COSYVOICE_VOICE_ROOT"] = old

    def test_creation_request_delegates_source_without_proceed_prompt(self) -> None:
        plugin = _load_plugin()
        result = plugin._voice_intent_context(
            user_message="Make a Jarvis voice clone.", turn_id="create-voice"
        )
        self.assertIn("action `create`", result["context"])
        self.assertIn("do not ask 'yes, proceed'", result["context"])
        self.assertIn("make_default=true", result["context"])
        directive = plugin._guard_voice_tool(
            tool_name="cosyvoice_voice",
            args={"action": "create", "query": "Jarvis voice"},
            turn_id="create-voice",
        )
        self.assertEqual("modify", directive["action"])
        self.assertEqual("Jarvis", directive["args"]["query"])

    def test_clean_source_search_uses_compound_autonomous_create(self) -> None:
        plugin = _load_plugin()
        result = plugin._voice_intent_context(
            user_message="Yes, search for a good clean Jarvis audio source.",
            turn_id="search-clean-source",
        )
        self.assertIn("action `create`", result["context"])
        self.assertIn("bounded result set is exhausted", result["context"])
        self.assertIn("Do not ask the user to choose", result["context"])

    def test_variant_language_is_forwarded_to_create(self) -> None:
        plugin = _load_plugin()
        plugin._voice_intent_context(
            user_message="Clone Optimus Prime's voice again from a recent movie with a more epic performance.",
            turn_id="optimus-variant",
        )
        directive = plugin._guard_voice_tool(
            tool_name="cosyvoice_voice", args={"action": "search"}, turn_id="optimus-variant"
        )
        self.assertTrue(directive["args"]["allow_variant"])

    def test_fragmented_tool_call_is_normalized_to_complete_create(self) -> None:
        plugin = _load_plugin()
        plugin._voice_intent_context(
            user_message=(
                "Clone Optimus Prime's voice, enable the matching personality and mannerisms, "
                "and set both as the persistent default."
            ),
            turn_id="optimus",
        )
        directive = plugin._guard_voice_tool(
            tool_name="cosyvoice_voice", args={"action": "search"}, turn_id="optimus"
        )
        self.assertEqual("modify", directive["action"])
        self.assertEqual("create", directive["args"]["action"])
        self.assertEqual("Optimus Prime", directive["args"]["query"])
        self.assertTrue(directive["args"]["make_default"])
        self.assertTrue(directive["args"]["enabled"])

    def test_user_supplied_link_is_preserved_in_compound_create(self) -> None:
        plugin = _load_plugin()
        source_url = "https://www.youtube.com/watch?v=zryfjSaxXLo"
        plugin._voice_intent_context(
            user_message=f"Clone the voice of Optimus Prime using this link: {source_url}",
            turn_id="linked-optimus",
        )
        directive = plugin._guard_voice_tool(
            tool_name="cosyvoice_voice",
            args={"action": "prepare", "source_url": source_url, "start_seconds": 240},
            turn_id="linked-optimus",
        )
        self.assertEqual("modify", directive["action"])
        self.assertEqual("create", directive["args"]["action"])
        self.assertEqual("Optimus Prime", directive["args"]["query"])
        self.assertEqual(source_url, directive["args"]["source_url"])
        self.assertEqual(240, directive["args"]["start_seconds"])

    def test_deferred_tool_wrapper_is_normalized_before_nested_dispatch(self) -> None:
        plugin = _load_plugin()
        plugin._voice_intent_context(
            user_message="Clone Optimus Prime's voice and set it as the persistent default.",
            turn_id="wrapped",
        )
        directive = plugin._guard_voice_tool(
            tool_name="tool_call",
            args={"calls": [{"name": "cosyvoice_voice", "arguments": {"action": "search"}}]},
            turn_id="wrapped",
        )
        nested = directive["args"]["calls"][0]["arguments"]
        self.assertEqual("create", nested["action"])
        self.assertEqual("Optimus Prime", nested["query"])
        self.assertTrue(nested["make_default"])

    def test_source_guard_remains_but_persistence_has_no_secondary_gate(self) -> None:
        plugin = _load_plugin()
        plugin._voice_intent_context(user_message="Set this as default", conversation_history=[{"content": "cosyvoice_voice choices"}], turn_id="no-source")
        self.assertEqual("block", plugin._guard_voice_tool(tool_name="cosyvoice_voice", args={"action": "prepare"}, turn_id="no-source")["action"])
        plugin._voice_intent_context(user_message="Pick option 2", conversation_history=[{"content": "cosyvoice_voice choices"}], turn_id="selected")
        self.assertIsNone(plugin._guard_voice_tool(tool_name="cosyvoice_voice", args={"action": "prepare"}, turn_id="selected"))
        self.assertIsNone(plugin._guard_voice_tool(
            tool_name="cosyvoice_voice", args={"action": "set_default"}, turn_id="selected"
        ))
        self.assertIsNone(plugin._guard_voice_tool(
            tool_name="cosyvoice_voice",
            args={"action": "prepare", "profile_id": "candidate", "prompt_text": "Exact words."},
            turn_id="no-source",
        ))

    def test_recent_plugin_search_result_can_be_prepared_on_later_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("COSYVOICE_VOICE_ROOT")
            os.environ["COSYVOICE_VOICE_ROOT"] = tmp
            try:
                plugin = _load_plugin()
                plugin._manager()._remember_recent_sources([{
                    "id": "clean",
                    "source_url": "https://www.youtube.com/watch?v=clean",
                    "suitability": "review",
                }], "clean voice")
                self.assertIsNone(plugin._guard_voice_tool(
                    tool_name="cosyvoice_voice",
                    args={
                        "action": "prepare",
                        "source_url": "https://www.youtube.com/watch?v=clean",
                    },
                    turn_id="later-turn",
                ))
                blocked = plugin._guard_voice_tool(
                    tool_name="cosyvoice_voice",
                    args={
                        "action": "prepare",
                        "source_url": "https://www.youtube.com/watch?v=unseen",
                    },
                    turn_id="later-turn",
                )
                self.assertEqual("block", blocked["action"])
            finally:
                if old is None:
                    os.environ.pop("COSYVOICE_VOICE_ROOT", None)
                else:
                    os.environ["COSYVOICE_VOICE_ROOT"] = old

    def test_durable_selection_survives_plugin_reload_and_candidate_does_not_activate(self) -> None:
        plugin = _load_plugin()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "candidates" / "sample"
            profile.mkdir(parents=True)
            (root / "state.json").write_text(json.dumps({
                "session_profile": "sample", "session_candidate": True, "default_profile": None,
            }), encoding="utf-8")
            (profile / "reference.wav").write_bytes(b"RIFF")
            (profile / "profile.json").write_text(json.dumps({
                "id": "sample",
                "prompt_text": "Sample reference words.",
                "personality": {"enabled": True, "prompt": "Use sample mannerisms sparingly."},
            }), encoding="utf-8")
            old = os.environ.get("COSYVOICE_VOICE_ROOT")
            os.environ["COSYVOICE_VOICE_ROOT"] = str(root)
            try:
                # Loading a fresh module simulates a plugin or gateway restart.
                reloaded = _load_plugin()
                followup = reloaded._voice_intent_context(user_message="Yes, activate it.", session_id="session", turn_id="activate")
                self.assertIn("durable CosyVoice workflow state", followup["context"])
                self.assertNotIn("sample mannerisms", followup["context"])
                disabled = reloaded._voice_intent_context(user_message="Voice only", session_id="session", turn_id="off")
                self.assertIsNone(disabled)
            finally:
                if old is None:
                    os.environ.pop("COSYVOICE_VOICE_ROOT", None)
                else:
                    os.environ["COSYVOICE_VOICE_ROOT"] = old

    def test_active_personality_is_not_described_as_subordinate_jarvis_blend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profiles" / "optimus"
            profile.mkdir(parents=True)
            (profile / "reference.wav").write_bytes(b"RIFF")
            (root / "state.json").write_text(json.dumps({
                "session_profile": "optimus", "session_candidate": False,
                "default_profile": "optimus",
            }), encoding="utf-8")
            (profile / "profile.json").write_text(json.dumps({
                "id": "optimus", "prompt_text": "Reference words.",
                "personality": {"enabled": True, "prompt": "Adopt Optimus Prime consistently."},
            }), encoding="utf-8")
            old = os.environ.get("COSYVOICE_VOICE_ROOT")
            os.environ["COSYVOICE_VOICE_ROOT"] = str(root)
            try:
                plugin = _load_plugin()
                context = plugin._voice_intent_context(user_message="Hello")
                self.assertIn("paired with the selected cloned voice", context["context"])
                self.assertNotIn("subordinate to Jarvis identity", context["context"])
                creation = plugin._voice_intent_context(user_message="Create an Optimus Prime voice profile.")
                self.assertIn('"id": "optimus"', creation["context"])
                self.assertIn("Reuse a matching saved profile", creation["context"])
            finally:
                if old is None:
                    os.environ.pop("COSYVOICE_VOICE_ROOT", None)
                else:
                    os.environ["COSYVOICE_VOICE_ROOT"] = old

    def test_register_exposes_tool_skill_and_hooks(self) -> None:
        plugin = _load_plugin()

        class Context:
            def __init__(self):
                self.tools, self.skills, self.hooks = [], [], []

            def register_tool(self, **kwargs):
                self.tools.append(kwargs)

            def register_skill(self, *args):
                self.skills.append(args)

            def register_hook(self, *args):
                self.hooks.append(args)

        context = Context()
        plugin.register(context)
        self.assertEqual("cosyvoice_voice", context.tools[0]["name"])
        self.assertIn("cosyvoice-voice", [item[0] for item in context.skills])
        self.assertEqual({"pre_llm_call", "pre_tool_call"}, {item[0] for item in context.hooks})


if __name__ == "__main__":
    unittest.main()
