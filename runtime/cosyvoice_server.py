#!/usr/bin/env python3
"""Local-only CosyVoice2 synthesis service for the Hermes TTS adapter."""

from __future__ import annotations

import argparse
import io
import itertools
import os
import threading
import time
import types
import wave
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path


def _boolean_env(name: str, default: bool) -> bool:
    value = os.environ.get(name, "true" if default else "false").strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise RuntimeError(f"{name} must be a boolean")
    return value in {"true", "1", "yes"}


_legacy_device = os.environ.get("COSYVOICE_DEVICE", "cpu").lower()
PLACEMENT = os.environ.get(
    "COSYVOICE_PLACEMENT", "all-gpu" if _legacy_device == "cuda" else "cpu"
).lower()
VALID_PLACEMENTS = {
    "cpu",
    "llm-gpu",
    "acoustic-gpu",
    "llm-flow-gpu",
    "llm-hift-gpu",
    "all-gpu",
}
GPU_WEIGHT_DTYPE = os.environ.get("COSYVOICE_GPU_WEIGHT_DTYPE", "fp32").lower()
LOAD_JIT = _boolean_env("COSYVOICE_LOAD_JIT", False)
LOAD_TRT = _boolean_env("COSYVOICE_LOAD_TRT", False)
try:
    TRT_CONCURRENT = int(os.environ.get("COSYVOICE_TRT_CONCURRENT", "1"))
except ValueError as exc:
    raise RuntimeError("COSYVOICE_TRT_CONCURRENT must be a positive integer") from exc
if TRT_CONCURRENT < 1:
    raise RuntimeError("COSYVOICE_TRT_CONCURRENT must be a positive integer")
if GPU_WEIGHT_DTYPE not in {"fp32", "fp16"}:
    raise RuntimeError("COSYVOICE_GPU_WEIGHT_DTYPE must be fp32 or fp16")
if PLACEMENT not in VALID_PLACEMENTS:
    raise RuntimeError(
        f"invalid COSYVOICE_PLACEMENT={PLACEMENT!r}; expected one of {sorted(VALID_PLACEMENTS)}"
    )
if PLACEMENT == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from cosyvoice.cli.cosyvoice import AutoModel
from conditioning_cache import ConditioningCache, ConditioningKey, fingerprint_file, hash_text
from profile_warmer import ProfileWarmer, prompt_signature
from streaming_audio import iter_streaming_wav
from synthesis_sessions import (
    SessionCancelled,
    SessionConflict,
    SessionError,
    SessionLimitExceeded,
    SessionNotFound,
    SessionTimedOut,
    SynthesisSessionStore,
)
from upstream_guard import install_background_generation_guard
from voice_profiles import VoiceProfileRegistry


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instruct: str = Field(default="", max_length=500)
    voice: str = Field(default="default", max_length=64)


class SynthesisSessionCreateRequest(SynthesisRequest):
    pass


class SynthesisSessionTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def _cache_limit(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be a non-negative integer")
    return parsed


def _enabled(name: str, default: bool) -> bool:
    return _boolean_env(name, default)


def _positive_env(name: str, default: int | float) -> int | float:
    value = os.environ.get(name, str(default))
    try:
        parsed = type(default)(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be positive") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _conditioning_key(args: argparse.Namespace, prompt, mode: str, conditioning_text: str) -> ConditioningKey:
    return ConditioningKey(
        source_revision=args.source_revision,
        model_revision=args.model_revision,
        profile_id=prompt.profile_id,
        reference_fingerprint=fingerprint_file(prompt.prompt_wav),
        mode=mode,
        conditioning_text_hash=hash_text(conditioning_text),
    )


def _cpu_conditioning(value):
    """Keep cached reference tensors out of scarce production VRAM."""
    if isinstance(value, dict):
        return {key: _cpu_conditioning(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_conditioning(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_conditioning(item) for item in value)
    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        return detach().cpu()
    return value


def _conditioned_chunks(
    model, text: str, conditioning: dict, speed: float, *, stream: bool = False
):
    """Rebuild request text while reusing only reference-derived frontend inputs."""
    for segment in model.frontend.text_normalize(text, split=True, text_frontend=True):
        request_input = dict(conditioning)
        text_tokens, text_token_lengths = model.frontend._extract_text_token(segment)
        request_input["text"] = text_tokens
        request_input["text_len"] = text_token_lengths
        if stream:
            # Upstream grows this mutable value while streaming and does not
            # reset it for the next request. Keep first-audio behavior stable.
            model.model.token_hop_len = 25
        try:
            yield from model.model.tts(
                **request_input,
                stream=stream,
                speed=1.0 if stream else speed,
            )
        finally:
            if stream:
                model.model.token_hop_len = 25
            guard = getattr(model.model, "_jarvis_generation_guard", None)
            if guard is not None:
                guard.pop_and_raise()


def _configure_frontend_onnx(model, model_dir: Path) -> dict[str, object]:
    """Optionally move reference speech-token extraction to ONNX CUDA."""
    requested = os.environ.get("COSYVOICE_FRONTEND_ONNX_PROVIDER", "cpu").lower()
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError("COSYVOICE_FRONTEND_ONNX_PROVIDER must be cpu or cuda")
    if requested == "cuda":
        import onnxruntime

        available = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("ONNX CUDA provider requested but unavailable")
        tokenizer_path = model_dir / "speech_tokenizer_v2.onnx"
        model.frontend.speech_tokenizer_session = onnxruntime.InferenceSession(
            str(tokenizer_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return {
        "requested": requested,
        "speech_tokenizer": model.frontend.speech_tokenizer_session.get_providers(),
        "speaker_embedding": model.frontend.campplus_session.get_providers(),
    }


def _conditioning(model, args, prompt, instruct: str = ""):
    effective_instruct = instruct.strip() or prompt.instruct
    mode = "instruct2" if effective_instruct else "zero_shot"
    conditioning_text = (
        effective_instruct
        if effective_instruct
        else model.frontend.text_normalize(
            prompt.prompt_text, split=False, text_frontend=True
        )
    )
    key = _conditioning_key(args, prompt, mode, conditioning_text)

    def build() -> dict:
        if mode == "instruct2":
            model_input = model.frontend.frontend_instruct2(
                "", conditioning_text, str(prompt.prompt_wav), model.sample_rate, ""
            )
        else:
            model_input = model.frontend.frontend_zero_shot(
                "", conditioning_text, str(prompt.prompt_wav), model.sample_rate, ""
            )
        model_input.pop("text", None)
        model_input.pop("text_len", None)
        return _cpu_conditioning(model_input)

    return key, mode, build


def load_model(model_dir: Path):
    """Load on CPU first, then place stable pipeline boundaries explicitly.

    CosyVoice2 normally puts its LLM, flow decoder, and vocoder on one device.
    Hermes uses non-streaming synthesis, so the LLM phase completes before the
    acoustic phase. That makes the LLM/acoustic boundary a safe, useful place
    to trade latency for VRAM without patching the pinned upstream checkout.
    """
    wants_gpu = PLACEMENT != "cpu"
    if wants_gpu and not torch.cuda.is_available():
        raise RuntimeError(f"{PLACEMENT} requested but CUDA is unavailable")

    real_cuda_available = torch.cuda.is_available
    try:
        # Upstream chooses one global device while loading. Force a CPU load so
        # this wrapper can move only the requested production components.
        torch.cuda.is_available = lambda: False
        model = AutoModel(
            model_dir=str(model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=False,
        )
    finally:
        torch.cuda.is_available = real_cuda_available

    runtime = model.model
    llm_device = torch.device(
        "cuda" if PLACEMENT in {"llm-gpu", "llm-flow-gpu", "llm-hift-gpu", "all-gpu"} else "cpu"
    )
    flow_device = torch.device(
        "cuda" if PLACEMENT in {"acoustic-gpu", "llm-flow-gpu", "all-gpu"} else "cpu"
    )
    hift_device = torch.device(
        "cuda" if PLACEMENT in {"acoustic-gpu", "llm-hift-gpu", "all-gpu"} else "cpu"
    )
    def place(module, device, *, allow_fp16=True):
        dtype = (
            torch.float16
            if device.type == "cuda" and GPU_WEIGHT_DTYPE == "fp16" and allow_fp16
            else torch.float32
        )
        # Convert device and dtype together so production startup never stages
        # a complete FP32 copy on a GPU already shared with Qwen.
        module.to(device=device, dtype=dtype)
        return module.eval()

    place(runtime.llm, llm_device)
    place(runtime.flow, flow_device)
    # HiFT constructs float32 oscillator tensors internally, so retaining its
    # small weights in FP32 avoids mixed-dtype failures with negligible VRAM.
    place(runtime.hift, hift_device, allow_fp16=False)
    runtime.fp16 = wants_gpu and GPU_WEIGHT_DTYPE == "fp16"

    acceleration = {
        "jit_flow_encoder": False,
        "tensorrt_flow_decoder": False,
        "tensorrt_contexts": 0,
    }
    if LOAD_JIT:
        if flow_device.type != "cuda":
            raise RuntimeError("COSYVOICE_LOAD_JIT requires the flow component on CUDA")
        jit_path = Path(
            os.environ.get(
                "COSYVOICE_JIT_FLOW_ENCODER",
                str(model_dir / f"flow.encoder.{GPU_WEIGHT_DTYPE}.zip"),
            )
        )
        if not jit_path.is_file():
            raise RuntimeError(f"CosyVoice JIT flow encoder not found: {jit_path}")
        runtime.device = flow_device
        runtime.load_jit(str(jit_path))
        place(runtime.flow.encoder, flow_device)
        acceleration["jit_flow_encoder"] = True

    if LOAD_TRT:
        if flow_device.type != "cuda":
            raise RuntimeError("COSYVOICE_LOAD_TRT requires the flow component on CUDA")
        engine_path = Path(
            os.environ.get(
                "COSYVOICE_TRT_ENGINE",
                str(model_dir / f"flow.decoder.estimator.{GPU_WEIGHT_DTYPE}.mygpu.plan"),
            )
        )
        onnx_path = Path(
            os.environ.get(
                "COSYVOICE_TRT_ONNX",
                str(model_dir / "flow.decoder.estimator.fp32.onnx"),
            )
        )
        if not onnx_path.is_file():
            raise RuntimeError(f"CosyVoice TensorRT ONNX model not found: {onnx_path}")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        runtime.device = flow_device
        runtime.load_trt(
            str(engine_path),
            str(onnx_path),
            TRT_CONCURRENT,
            runtime.fp16,
        )
        acceleration.update(
            tensorrt_flow_decoder=True,
            tensorrt_contexts=TRT_CONCURRENT,
        )

    original_llm_job = runtime.llm_job
    original_token2wav = runtime.token2wav
    original_hift_inference = runtime.hift.inference
    runtime.llm_context = (
        torch.cuda.stream(torch.cuda.Stream(device=llm_device))
        if llm_device.type == "cuda"
        else nullcontext()
    )

    def llm_job(self, *args, **kwargs):
        self.device = llm_device
        return original_llm_job(*args, **kwargs)

    def token2wav(self, *args, **kwargs):
        self.device = flow_device
        return original_token2wav(*args, **kwargs)

    def hift_inference(*args, **kwargs):
        hift_dtype = next(runtime.hift.parameters()).dtype
        if "speech_feat" in kwargs:
            kwargs["speech_feat"] = kwargs["speech_feat"].to(
                device=hift_device, dtype=hift_dtype
            )
        if "cache_source" in kwargs:
            kwargs["cache_source"] = kwargs["cache_source"].to(
                device=hift_device, dtype=hift_dtype
            )
        return original_hift_inference(*args, **kwargs)

    runtime.device = llm_device
    runtime.llm_job = types.MethodType(llm_job, runtime)
    install_background_generation_guard(runtime)
    runtime.token2wav = types.MethodType(token2wav, runtime)
    runtime.hift.inference = hift_inference
    frontend_onnx = _configure_frontend_onnx(model, model_dir)
    return model, {
        "llm": str(llm_device),
        "flow": str(flow_device),
        "hift": str(hift_device),
        "gpu_weight_dtype": GPU_WEIGHT_DTYPE,
        "hift_weight_dtype": "fp32",
        "cuda_autocast": runtime.fp16,
        "acceleration": acceleration,
        "frontend_onnx": frontend_onnx,
    }


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    samples = np.clip(audio, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def create_app(args: argparse.Namespace) -> FastAPI:
    lock = threading.Lock()
    metrics_lock = threading.Lock()
    state: dict[str, object] = {
        "ready": False,
        "conditioning_cache": ConditioningCache(
            max_entries=_cache_limit("COSYVOICE_CONDITIONING_CACHE_ENTRIES", 8),
            max_bytes=_cache_limit("COSYVOICE_CONDITIONING_CACHE_MAX_BYTES", 256 * 1024 * 1024),
        ),
        "synthesis_sessions": SynthesisSessionStore(
            max_sessions=int(_positive_env("COSYVOICE_SYNTHESIS_SESSION_MAX_SESSIONS", 8)),
            max_queue_sentences=int(
                _positive_env("COSYVOICE_SYNTHESIS_SESSION_MAX_QUEUE_SENTENCES", 8)
            ),
            max_text_bytes=int(
                _positive_env("COSYVOICE_SYNTHESIS_SESSION_MAX_TEXT_BYTES", 16 * 1024)
            ),
            idle_timeout_seconds=float(
                _positive_env("COSYVOICE_SYNTHESIS_SESSION_IDLE_TIMEOUT_SECONDS", 30.0)
            ),
            total_timeout_seconds=float(
                _positive_env("COSYVOICE_SYNTHESIS_SESSION_TOTAL_TIMEOUT_SECONDS", 300.0)
            ),
        ),
        "last_synthesis": None,
    }

    def record_last_synthesis(metadata: dict[str, object]) -> None:
        # This endpoint is intentionally safe to expose on the LAN: never keep
        # request text, instruction text, paths, hashes, or exception details.
        with metrics_lock:
            state["last_synthesis"] = metadata

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        model, component_devices = load_model(args.model_dir)
        profiles = VoiceProfileRegistry(args.profile_root, args.prompt_wav, args.prompt_text)
        state.update(
            model=model,
            profiles=profiles,
            component_devices=component_devices,
        )
        warmup = {"enabled": _enabled("COSYVOICE_WARMUP_ENABLED", True), "status": "disabled"}
        if warmup["enabled"]:
            warmup_started_at = time.monotonic()
            prompt = profiles.resolve_saved_default()
            if prompt is None:
                warmup["status"] = "skipped_no_saved_profile"
            else:
                try:
                    cache_key, mode, build_conditioning = _conditioning(model, args, prompt)
                    cached = state["conditioning_cache"].get_or_create(cache_key, build_conditioning)
                    with torch.inference_mode():
                        for _chunk in _conditioned_chunks(
                            model, "Ready.", cached.value, 1.0, stream=False
                        ):
                            pass
                    warmup.update(status="ok", profile_id=prompt.profile_id, mode=mode)
                except Exception:
                    # Startup remains available for explicit requests; health
                    # reports the failed warm-up without exposing private data.
                    warmup.update(status="error", profile_id=prompt.profile_id)
            warmup["elapsed_ms"] = round((time.monotonic() - warmup_started_at) * 1000, 2)
        def warm_profile(prompt) -> bool:
            with lock, torch.inference_mode():
                key, _mode, build = _conditioning(model, args, prompt)
                return state["conditioning_cache"].get_or_create(key, build).hit

        watcher = ProfileWarmer(
            profiles.resolve_saved_default,
            warm_profile,
            initial_signature=prompt_signature(profiles.resolve_saved_default()),
        )
        if _enabled("COSYVOICE_PROFILE_WARMER_ENABLED", True):
            watcher.start()
        state["synthesis_sessions"].start()
        state.update(warmup=warmup, profile_warmer=watcher, ready=True)
        yield
        state["ready"] = False
        state["synthesis_sessions"].stop()
        watcher.stop()

    app = FastAPI(title="Jarvis CosyVoice2", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        with metrics_lock:
            last_synthesis = state["last_synthesis"]
        return {
            "ready": state["ready"],
            "backend": "cosyvoice2",
            "placement": PLACEMENT,
            "component_devices": state.get("component_devices", {}),
            "model_revision": args.model_revision,
            "source_revision": args.source_revision,
            "conditioning_cache": state["conditioning_cache"].snapshot(),
            "warmup": state.get("warmup", {}),
            "profile_warmer": state["profile_warmer"].snapshot() if state.get("profile_warmer") else {},
            "synthesis_sessions": state["synthesis_sessions"].snapshot(),
            "last_synthesis": last_synthesis,
        }

    @app.post("/synthesize")
    def synthesize(request: SynthesisRequest) -> Response:
        started_at = time.monotonic()
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model is not ready")
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        model = state["model"]
        try:
            prompt = state["profiles"].resolve(request.voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        queue_started_at = time.monotonic()
        with lock, torch.inference_mode():
            queue_wait_ms = round((time.monotonic() - queue_started_at) * 1000, 2)
            instruct = request.instruct.strip() or prompt.instruct
            mode = "instruct2" if instruct else "zero_shot"
            # Upstream normalizes the zero-shot transcript before extracting
            # prompt tokens. Instruct2 uses its effective instruction directly.
            conditioning_text = (
                instruct
                if instruct
                else model.frontend.text_normalize(prompt.prompt_text, split=False, text_frontend=True)
            )
            cache_started_at = time.monotonic()
            cache_key = _conditioning_key(args, prompt, mode, conditioning_text)

            def build_conditioning() -> dict:
                if mode == "instruct2":
                    model_input = model.frontend.frontend_instruct2(
                        "", conditioning_text, str(prompt.prompt_wav), model.sample_rate, ""
                    )
                else:
                    model_input = model.frontend.frontend_zero_shot(
                        "", conditioning_text, str(prompt.prompt_wav), model.sample_rate, ""
                    )
                model_input.pop("text", None)
                model_input.pop("text_len", None)
                return _cpu_conditioning(model_input)

            try:
                cached = state["conditioning_cache"].get_or_create(cache_key, build_conditioning)
            except Exception:
                record_last_synthesis(
                    {
                        "outcome": "error",
                        "phase": "conditioning",
                        "profile_id": prompt.profile_id,
                        "mode": mode,
                        "queue_wait_ms": queue_wait_ms,
                        "total_ms": round((time.monotonic() - started_at) * 1000, 2),
                    }
                )
                raise
            cache_lookup_ms = round((time.monotonic() - cache_started_at) * 1000, 2)
            synthesis_started_at = time.monotonic()
            try:
                audio = [
                    chunk["tts_speech"].detach().cpu().float().numpy().reshape(-1)
                    for chunk in _conditioned_chunks(model, text, cached.value, request.speed)
                ]
            except Exception:
                record_last_synthesis(
                    {
                        "outcome": "error",
                        "profile_id": prompt.profile_id,
                        "mode": mode,
                        "cache_hit": cached.hit,
                        "queue_wait_ms": queue_wait_ms,
                        "cache_lookup_ms": cache_lookup_ms,
                        "total_ms": round((time.monotonic() - started_at) * 1000, 2),
                    }
                )
                raise
        if not audio:
            record_last_synthesis(
                {
                    "outcome": "empty_audio",
                    "profile_id": prompt.profile_id,
                    "mode": mode,
                    "cache_hit": cached.hit,
                    "queue_wait_ms": queue_wait_ms,
                    "cache_lookup_ms": cache_lookup_ms,
                    "total_ms": round((time.monotonic() - started_at) * 1000, 2),
                }
            )
            raise HTTPException(status_code=500, detail="model produced no audio")
        state["profiles"].consume(prompt)
        combined = np.concatenate(audio)
        record_last_synthesis(
            {
                "outcome": "ok",
                "profile_id": prompt.profile_id,
                "mode": mode,
                "cache_hit": cached.hit,
                "queue_wait_ms": queue_wait_ms,
                "cache_lookup_ms": cache_lookup_ms,
                "synthesis_ms": round((time.monotonic() - synthesis_started_at) * 1000, 2),
                "total_ms": round((time.monotonic() - started_at) * 1000, 2),
                "audio_duration_ms": round((combined.size / model.sample_rate) * 1000, 2),
            }
        )
        return Response(wav_bytes(combined, model.sample_rate), media_type="audio/wav")

    @app.post("/synthesize-stream")
    def synthesize_stream(request: SynthesisRequest) -> StreamingResponse:
        """Stream one continuous WAV while retaining the buffered API."""
        started_at = time.monotonic()
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model is not ready")
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is empty")
        model = state["model"]
        try:
            prompt = state["profiles"].resolve(request.voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def model_audio():
            queue_started_at = time.monotonic()
            completed = False
            with lock, torch.inference_mode():
                queue_wait_ms = round((time.monotonic() - queue_started_at) * 1000, 2)
                instruct = request.instruct.strip() or prompt.instruct
                mode = "instruct2" if instruct else "zero_shot"
                conditioning_text = (
                    instruct
                    if instruct
                    else model.frontend.text_normalize(
                        prompt.prompt_text, split=False, text_frontend=True
                    )
                )
                cache_started_at = time.monotonic()
                cache_key = _conditioning_key(args, prompt, mode, conditioning_text)

                def build_conditioning() -> dict:
                    if mode == "instruct2":
                        model_input = model.frontend.frontend_instruct2(
                            "",
                            conditioning_text,
                            str(prompt.prompt_wav),
                            model.sample_rate,
                            "",
                        )
                    else:
                        model_input = model.frontend.frontend_zero_shot(
                            "",
                            conditioning_text,
                            str(prompt.prompt_wav),
                            model.sample_rate,
                            "",
                        )
                    model_input.pop("text", None)
                    model_input.pop("text_len", None)
                    return _cpu_conditioning(model_input)

                cached = state["conditioning_cache"].get_or_create(
                    cache_key, build_conditioning
                )
                cache_lookup_ms = round((time.monotonic() - cache_started_at) * 1000, 2)
                synthesis_started_at = time.monotonic()
                audio_samples = 0
                first_audio_ms = None
                try:
                    for chunk in _conditioned_chunks(
                        model, text, cached.value, request.speed, stream=True
                    ):
                        audio = chunk["tts_speech"].detach().cpu().float().numpy().reshape(-1)
                        if first_audio_ms is None:
                            first_audio_ms = round(
                                (time.monotonic() - started_at) * 1000, 2
                            )
                        audio_samples += audio.size
                        yield audio
                    completed = True
                finally:
                    outcome = "ok" if completed else "cancelled_or_error"
                    record_last_synthesis(
                        {
                            "outcome": outcome,
                            "profile_id": prompt.profile_id,
                            "mode": mode,
                            "streaming": True,
                            "cache_hit": cached.hit,
                            "queue_wait_ms": queue_wait_ms,
                            "cache_lookup_ms": cache_lookup_ms,
                            "first_audio_ms": first_audio_ms,
                            "synthesis_ms": round(
                                (time.monotonic() - synthesis_started_at) * 1000, 2
                            ),
                            "total_ms": round(
                                (time.monotonic() - started_at) * 1000, 2
                            ),
                            "audio_duration_ms": round(
                                (audio_samples / model.sample_rate) * 1000, 2
                            ),
                        }
                    )
            if completed:
                state["profiles"].consume(prompt)

        return StreamingResponse(
            iter_streaming_wav(model_audio(), model.sample_rate, request.speed),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-CosyVoice-Stream": "pcm16",
            },
        )

    def session_http_error(error: Exception) -> HTTPException:
        if isinstance(error, SessionNotFound):
            return HTTPException(status_code=404, detail="synthesis session was not found")
        if isinstance(error, SessionConflict):
            return HTTPException(status_code=409, detail=str(error))
        if isinstance(error, SessionLimitExceeded):
            return HTTPException(status_code=429, detail=str(error))
        if isinstance(error, ValueError):
            return HTTPException(status_code=400, detail="text is empty")
        raise error

    @app.post("/synthesis-sessions")
    def create_synthesis_session(request: SynthesisSessionCreateRequest) -> dict[str, str]:
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model is not ready")
        try:
            prompt = state["profiles"].resolve(request.voice)
            session = state["synthesis_sessions"].create(
                request.text,
                prompt=prompt,
                instruct=request.instruct.strip() or prompt.instruct,
                speed=request.speed,
            )
        except (SessionError, ValueError) as exc:
            raise session_http_error(exc) from exc
        return {"session_id": session.session_id}

    @app.post("/synthesis-sessions/{session_id}/text")
    def enqueue_synthesis_session_text(
        session_id: str, request: SynthesisSessionTextRequest
    ) -> dict[str, str]:
        try:
            state["synthesis_sessions"].enqueue(session_id, request.text)
        except (SessionError, ValueError) as exc:
            raise session_http_error(exc) from exc
        return {"status": "queued"}

    @app.post("/synthesis-sessions/{session_id}/finish")
    def finish_synthesis_session(session_id: str) -> dict[str, str]:
        try:
            state["synthesis_sessions"].finish(session_id)
        except SessionError as exc:
            raise session_http_error(exc) from exc
        return {"status": "finishing"}

    @app.delete("/synthesis-sessions/{session_id}", status_code=204)
    def cancel_synthesis_session(session_id: str) -> Response:
        try:
            state["synthesis_sessions"].cancel(session_id)
        except SessionError as exc:
            raise session_http_error(exc) from exc
        return Response(status_code=204)

    @app.get("/synthesis-sessions/{session_id}/audio")
    def stream_synthesis_session_audio(session_id: str) -> StreamingResponse:
        if not state["ready"]:
            raise HTTPException(status_code=503, detail="model is not ready")
        sessions = state["synthesis_sessions"]
        try:
            session = sessions.claim_stream(session_id)
        except SessionError as exc:
            raise session_http_error(exc) from exc
        model = state["model"]
        started_at = time.monotonic()
        prompt = session.prompt
        mode = "instruct2" if session.instruct else "zero_shot"
        cache_hit = None
        cache_lookup_ms = None
        queue_wait_ms = 0.0
        synthesis_started_at = None
        first_audio_ms = None
        audio_samples = 0
        sentence_count = 0
        model_completed = False

        def model_audio():
            nonlocal audio_samples, cache_hit, cache_lookup_ms, first_audio_ms
            nonlocal model_completed, queue_wait_ms, sentence_count, synthesis_started_at
            queue_started_at = time.monotonic()
            with lock, torch.inference_mode():
                queue_wait_ms += round((time.monotonic() - queue_started_at) * 1000, 2)
                sessions.touch(session_id)
                cache_started_at = time.monotonic()
                cache_key, _mode, build_conditioning = _conditioning(
                    model, args, prompt, session.instruct
                )
                cached = state["conditioning_cache"].get_or_create(
                    cache_key, build_conditioning
                )
                cache_hit = cached.hit
                cache_lookup_ms = round((time.monotonic() - cache_started_at) * 1000, 2)

            synthesis_started_at = time.monotonic()
            for sentence in sessions.iter_sentences(session_id):
                sessions.touch(session_id)
                queue_started_at = time.monotonic()
                with lock, torch.inference_mode():
                    queue_wait_ms += round((time.monotonic() - queue_started_at) * 1000, 2)
                    sessions.touch(session_id)
                    for chunk in _conditioned_chunks(
                        model, sentence, cached.value, session.speed, stream=True
                    ):
                        sessions.touch(session_id)
                        audio = chunk["tts_speech"].detach().cpu().float().numpy().reshape(-1)
                        if first_audio_ms is None:
                            first_audio_ms = round((time.monotonic() - started_at) * 1000, 2)
                        audio_samples += audio.size
                        yield audio
                sentence_count += 1
                sessions.touch(session_id)
            if audio_samples == 0:
                raise RuntimeError("model produced no audio")
            model_completed = True

        def session_wav():
            outcome = "cancelled"
            audio_stream = None
            try:
                model_stream = iter(model_audio())
                # Preserve Home Assistant's fallback until synthesis has yielded
                # real audio instead of committing the client on a header alone.
                first_audio = next(model_stream)
                audio_stream = iter_streaming_wav(
                    itertools.chain((first_audio,), model_stream),
                    model.sample_rate,
                    session.speed,
                )
                for payload in audio_stream:
                    yield payload
                if not model_completed:
                    raise RuntimeError("synthesis session ended before completion")
                state["profiles"].consume(prompt)
                outcome = "completed"
            except (SessionCancelled, SessionTimedOut, SessionNotFound):
                outcome = "cancelled"
                raise
            except GeneratorExit:
                outcome = "cancelled"
                raise
            except BaseException:
                outcome = "errors"
                raise
            finally:
                sessions.cleanup(session_id, outcome)
                try:
                    if audio_stream is not None:
                        audio_stream.close()
                finally:
                    record_last_synthesis(
                        {
                            "outcome": "ok" if outcome == "completed" else outcome,
                            "streaming": True,
                            "session": True,
                            "profile_id": prompt.profile_id,
                            "mode": mode,
                            "cache_hit": cache_hit,
                            "queue_wait_ms": queue_wait_ms,
                            "cache_lookup_ms": cache_lookup_ms,
                            "first_audio_ms": first_audio_ms,
                            "synthesis_ms": round(
                                (time.monotonic() - synthesis_started_at) * 1000, 2
                            )
                            if synthesis_started_at is not None
                            else None,
                            "total_ms": round((time.monotonic() - started_at) * 1000, 2),
                            "audio_duration_ms": round(
                                (audio_samples / model.sample_rate) * 1000, 2
                            ),
                            "sentences": sentence_count,
                        }
                    )

        def next_payload(iterator):
            try:
                return next(iterator)
            except StopIteration:
                return None

        async def async_session_wav():
            iterator = iter(session_wav())
            try:
                while True:
                    payload = await run_in_threadpool(next_payload, iterator)
                    if payload is None:
                        return
                    yield payload
            finally:
                await run_in_threadpool(iterator.close)

        return StreamingResponse(
            async_session_wav(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-CosyVoice-Stream": "pcm16",
            },
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17870)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-wav", type=Path, required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("/srv/cosyvoice2/data/voice_profiles"))
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    options = parse_args()
    uvicorn.run(create_app(options), host=options.host, port=options.port, log_level="info")
