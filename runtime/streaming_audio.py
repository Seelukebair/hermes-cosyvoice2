"""Helpers for bounded, continuous PCM streaming."""

from __future__ import annotations

import io
import queue
import shutil
import subprocess
import threading
import wave
from collections.abc import Iterable, Iterator

import numpy as np


_READ_SIZE = 16 * 1024


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert floating point mono audio to little-endian signed PCM16."""
    samples = np.clip(audio, -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def silence_pcm16(sample_rate: int, seconds: float) -> bytes:
    """Return an exact duration of mono PCM16 digital silence."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if seconds < 0:
        raise ValueError("silence duration cannot be negative")
    return b"\x00\x00" * round(sample_rate * seconds)


def streaming_wav_header(sample_rate: int) -> bytes:
    """Return Home Assistant's unknown-length WAV header (zero frames)."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
    return output.getvalue()


def _tempo_filters(speed: float) -> str:
    """Build a valid FFmpeg atempo chain for the accepted 0.5-2.0 range."""
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    return f"atempo={speed:.6f}"


def iter_streaming_wav(
    audio_chunks: Iterable[np.ndarray],
    sample_rate: int,
    speed: float,
    *,
    ffmpeg_path: str = "ffmpeg",
    leading_silence_seconds: float = 0.0,
) -> Iterator[bytes]:
    """Yield a streaming WAV header and continuously tempo-adjusted PCM.

    FFmpeg receives one continuous raw stream, so its filter state spans model
    chunk boundaries. If the consumer disconnects, the model iterator is drained
    in its producer thread so the pinned CosyVoice runtime can finish cleanup.
    """
    yield streaming_wav_header(sample_rate)
    leading_silence = silence_pcm16(sample_rate, leading_silence_seconds)
    if leading_silence:
        yield leading_silence

    if speed == 1.0:
        for audio in audio_chunks:
            pcm = pcm16_bytes(audio)
            if pcm:
                yield pcm
        return

    executable = shutil.which(ffmpeg_path)
    if executable is None:
        raise RuntimeError("ffmpeg is required for streaming speed adjustment")

    process = subprocess.Popen(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-filter:a",
            _tempo_filters(speed),
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    stopped = threading.Event()
    failures: queue.SimpleQueue[BaseException] = queue.SimpleQueue()

    def produce() -> None:
        try:
            for audio in audio_chunks:
                if stopped.is_set():
                    continue
                try:
                    process.stdin.write(pcm16_bytes(audio))
                except (BrokenPipeError, OSError, ValueError):
                    stopped.set()
        except BaseException as exc:  # passed back to response iterator
            failures.put(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    producer = threading.Thread(target=produce, name="cosyvoice-stream-producer", daemon=True)
    producer.start()
    try:
        while data := process.stdout.read(_READ_SIZE):
            yield data
        producer.join(timeout=5)
        return_code = process.wait(timeout=5)
        if not failures.empty():
            raise failures.get()
        if return_code != 0 and not stopped.is_set():
            detail = process.stderr.read(512).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg streaming conversion failed: {detail or return_code}")
    finally:
        stopped.set()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        producer.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
