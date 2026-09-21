from __future__ import annotations

import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from clean_reference import BOUNDARY_SILENCE_SECONDS, pad_boundaries


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
class CleanReferenceTests(unittest.TestCase):
    def test_boundary_padding_adds_silence_without_altering_inner_audio(self):
        sample_rate = 24000
        source_samples = np.full(sample_rate, 12000, dtype="<i2")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            output = root / "output.wav"
            with wave.open(str(source), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(source_samples.tobytes())

            pad_boundaries(source, output)

            with wave.open(str(output), "rb") as wav_file:
                rendered = np.frombuffer(
                    wav_file.readframes(wav_file.getnframes()), dtype="<i2"
                )
                self.assertEqual(sample_rate, wav_file.getframerate())
            padding = round(sample_rate * BOUNDARY_SILENCE_SECONDS)
            self.assertEqual(source_samples.size + 2 * padding, rendered.size)
            np.testing.assert_array_equal(0, rendered[:padding])
            np.testing.assert_array_equal(source_samples, rendered[padding:-padding])
            np.testing.assert_array_equal(0, rendered[-padding:])


if __name__ == "__main__":
    unittest.main()
