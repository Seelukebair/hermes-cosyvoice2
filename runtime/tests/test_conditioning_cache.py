import tempfile
import unittest
from pathlib import Path
import sys

RUNTIME_DIR = Path(__file__).resolve().parents[1]
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from conditioning_cache import ConditioningCache, ConditioningKey, fingerprint_file, hash_text


class CloneBox:
    def __init__(self, value, nbytes=8):
        self.value = value
        self.nbytes = nbytes

    def clone(self):
        return CloneBox(self.value, self.nbytes)


def key(*, mode="zero_shot", text="reference"):
    return ConditioningKey(
        source_revision="source-a",
        model_revision="model-a",
        profile_id="voice-a",
        reference_fingerprint="wav-a",
        mode=mode,
        conditioning_text_hash=hash_text(text),
    )


class ConditioningCacheTests(unittest.TestCase):
    def test_hit_reuses_entry_and_returns_an_independent_copy(self):
        cache = ConditioningCache(max_entries=2, max_bytes=64)
        calls = 0

        def build():
            nonlocal calls
            calls += 1
            return {"conditioning": CloneBox("stable", 16)}

        first = cache.get_or_create(key(), build)
        first.value["conditioning"].value = "changed by inference"
        second = cache.get_or_create(key(), build)

        self.assertFalse(first.hit)
        self.assertTrue(second.hit)
        self.assertEqual(1, calls)
        self.assertEqual("stable", second.value["conditioning"].value)
        self.assertEqual(1, cache.snapshot()["hits"])

    def test_instruct_and_zero_shot_never_share_conditioning(self):
        cache = ConditioningCache(max_entries=8, max_bytes=64)
        zero = cache.get_or_create(key(mode="zero_shot", text="reference transcript"), lambda: CloneBox("zero"))
        instruct = cache.get_or_create(key(mode="instruct2", text="speak briskly"), lambda: CloneBox("instruct"))

        self.assertFalse(zero.hit)
        self.assertFalse(instruct.hit)
        self.assertEqual(2, cache.snapshot()["entries"])

    def test_profile_content_or_conditioning_change_invalidates(self):
        cache = ConditioningCache(max_entries=8, max_bytes=64)
        original = key(text="original")
        cache.get_or_create(original, lambda: CloneBox("original"))
        changed_text = cache.get_or_create(key(text="changed"), lambda: CloneBox("changed"))
        changed_wav_key = ConditioningKey(
            **{**original.__dict__, "reference_fingerprint": "wav-b"}
        )
        changed_wav = cache.get_or_create(changed_wav_key, lambda: CloneBox("new-wav"))
        changed_profile_key = ConditioningKey(**{**original.__dict__, "profile_id": "voice-b"})
        changed_profile = cache.get_or_create(changed_profile_key, lambda: CloneBox("new-profile"))
        changed_revision_key = ConditioningKey(
            **{**original.__dict__, "source_revision": "source-b", "model_revision": "model-b"}
        )
        changed_revision = cache.get_or_create(changed_revision_key, lambda: CloneBox("new-revision"))

        self.assertFalse(changed_text.hit)
        self.assertFalse(changed_wav.hit)
        self.assertFalse(changed_profile.hit)
        self.assertFalse(changed_revision.hit)
        self.assertEqual(5, cache.snapshot()["entries"])

    def test_lru_and_byte_limits_are_enforced(self):
        cache = ConditioningCache(max_entries=2, max_bytes=20)
        cache.get_or_create(key(text="one"), lambda: CloneBox("one", 10))
        cache.get_or_create(key(text="two"), lambda: CloneBox("two", 10))
        cache.get_or_create(key(text="one"), lambda: CloneBox("one", 10))
        cache.get_or_create(key(text="three"), lambda: CloneBox("three", 10))

        missing = cache.get_or_create(key(text="two"), lambda: CloneBox("two", 10))
        snapshot = cache.snapshot()
        self.assertFalse(missing.hit)
        self.assertLessEqual(snapshot["entries"], 2)
        self.assertLessEqual(snapshot["bytes"], 20)
        self.assertGreaterEqual(snapshot["evictions"], 1)

    def test_disabled_cache_returns_factory_value_without_retaining_it(self):
        cache = ConditioningCache(max_entries=0, max_bytes=64)
        calls = 0

        def build():
            nonlocal calls
            calls += 1
            return CloneBox("uncached", 8)

        first = cache.get_or_create(key(), build)
        second = cache.get_or_create(key(), build)

        self.assertFalse(first.hit)
        self.assertFalse(second.hit)
        self.assertEqual(2, calls)
        self.assertEqual(0, cache.snapshot()["entries"])
        self.assertFalse(cache.snapshot()["enabled"])

    def test_oversized_entry_is_not_retained(self):
        cache = ConditioningCache(max_entries=2, max_bytes=8)
        calls = 0

        def build():
            nonlocal calls
            calls += 1
            return CloneBox("too-large", 16)

        first = cache.get_or_create(key(), build)
        second = cache.get_or_create(key(), build)

        self.assertFalse(first.hit)
        self.assertFalse(second.hit)
        self.assertEqual(2, calls)
        self.assertEqual(0, cache.snapshot()["entries"])
        self.assertEqual(0, cache.snapshot()["bytes"])

    def test_file_fingerprint_tracks_content_not_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(b"same audio")
            second.write_bytes(b"same audio")
            self.assertEqual(fingerprint_file(first), fingerprint_file(second))
            second.write_bytes(b"replaced audio")
            self.assertNotEqual(fingerprint_file(first), fingerprint_file(second))


if __name__ == "__main__":
    unittest.main()
