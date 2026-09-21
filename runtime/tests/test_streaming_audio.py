import io
import shutil
import unittest
import wave

import numpy as np

from streaming_audio import iter_streaming_wav, pcm16_bytes, streaming_wav_header


class StreamingAudioTests(unittest.TestCase):
    def test_header_is_valid_zero_frame_mono_pcm(self):
        header = streaming_wav_header(24000)
        with wave.open(io.BytesIO(header), "rb") as wav_file:
            self.assertEqual(1, wav_file.getnchannels())
            self.assertEqual(2, wav_file.getsampwidth())
            self.assertEqual(24000, wav_file.getframerate())
            self.assertEqual(0, wav_file.getnframes())

    def test_pcm_conversion_clips_and_uses_little_endian_int16(self):
        converted = np.frombuffer(
            pcm16_bytes(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)),
            dtype="<i2",
        )
        np.testing.assert_array_equal(converted, [-32767, -32767, 0, 32767, 32767])

    def test_identity_stream_preserves_all_chunks(self):
        chunks = [np.array([0.0, 0.5]), np.array([-0.5, 1.0])]
        output = list(iter_streaming_wav(chunks, 24000, 1.0))
        self.assertEqual(streaming_wav_header(24000), output[0])
        self.assertEqual(pcm16_bytes(np.concatenate(chunks)), b"".join(output[1:]))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    def test_tempo_filter_emits_decodable_pcm(self):
        sample_rate = 24000
        seconds = 0.5
        samples = np.sin(
            2 * np.pi * 220 * np.arange(int(sample_rate * seconds)) / sample_rate
        ).astype(np.float32)
        output = b"".join(iter_streaming_wav([samples[:6000], samples[6000:]], sample_rate, 1.1))
        self.assertTrue(output.startswith(b"RIFF"))
        pcm_samples = (len(output) - 44) // 2
        self.assertGreater(pcm_samples, int(sample_rate * 0.40))
        self.assertLess(pcm_samples, int(sample_rate * 0.48))


if __name__ == "__main__":
    unittest.main()
