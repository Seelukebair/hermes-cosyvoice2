from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_migration():
    path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_generated_personalities.py"
    spec = importlib.util.spec_from_file_location("migrate_generated_personalities", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PersonalityMigrationTests(unittest.TestCase):
    def test_url_label_is_repaired_from_friendly_name(self) -> None:
        migration = _load_migration()
        bad_label = "https://www.youtube.com/watch?v=example"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps({
                "name": "Samuel L. Jackson (Pulp)",
                "delivery_prompt": "",
                "personality": {
                    "enabled": True,
                    "label": bad_label,
                    "mode": "character",
                    "paired_with_voice": True,
                    "prompt": migration.OLD_TEMPLATES[-1].format(label=bad_label),
                },
            }), encoding="utf-8")
            self.assertTrue(migration.migrate(path, True))
            result = json.loads(path.read_text(encoding="utf-8"))
            personality = result["personality"]
            self.assertEqual("Samuel L. Jackson", personality["label"])
            self.assertIn("sole presentation persona", personality["prompt"])
            self.assertNotIn("youtube", personality["prompt"])

    def test_custom_prompt_is_not_rewritten(self) -> None:
        migration = _load_migration()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            payload = {
                "name": "Custom",
                "personality": {"enabled": True, "label": "Custom", "prompt": "My hand-written prompt."},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(migration.migrate(path, True))
            self.assertEqual(payload, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
