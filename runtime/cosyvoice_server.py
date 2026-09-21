#!/usr/bin/env python3
"""Local-only CosyVoice2 synthesis service for the Hermes TTS adapter."""

from __future__ import annotations

import argparse
import io
import os
import threading
import types
import wave
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

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
from fastapi.responses import Response
from pydantic import BaseModel, Field

from cosyvoice.cli.cosyvoice import AutoModel
from voice_profiles import VoiceProfileRegistry


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    instruct: str = Field(default="", max_length=500)
    voice: str = Field(default="default", max_length=64)


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

    runtime.llm_job = types.MethodType(llm_job, runtime)
    runtime.token2wav = types.MethodType(token2wav, runtime)
    runtime.hift.inference = hift_inference
    runtime.device = llm_device
    return model, {
        "llm": str(llm_device),
        "flow": str(flow_device),
        "hift": str(hift_device),
        "gpu_weight_dtype": GPU_WEIGHT_DTYPE,
        "hift_weight_dtype": "fp32",
        "cuda_autocast": runtime.fp16,
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
    state: dict[str, object] = {"ready": False}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        model, component_devices = load_model(args.model_dir)
        # Current CosyVoice frontend methods load and resample the prompt path.
        state.update(
            model=model,
            profiles=VoiceProfileRegistry(args.profile_root, args.prompt_wav, args.prompt_text),
            component_devices=component_devices,
            ready=True,
        )
        yield
        state["ready"] = False

    app = FastAPI(title="Jarvis CosyVoice2", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": state["ready"],
            "backend": "cosyvoice2",
            "placement": PLACEMENT,
            "component_devices": state.get("component_devices", {}),
            "model_revision": args.model_revision,
            "source_revision": args.source_revision,
        }

    @app.post("/synthesize")
    def synthesize(request: SynthesisRequest) -> Response:
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
        with lock, torch.inference_mode():
            instruct = request.instruct.strip() or prompt.instruct
            if instruct:
                chunks = model.inference_instruct2(
                    text,
                    instruct,
                    str(prompt.prompt_wav),
                    stream=False,
                    speed=request.speed,
                )
            else:
                chunks = model.inference_zero_shot(
                    text,
                    prompt.prompt_text,
                    str(prompt.prompt_wav),
                    stream=False,
                    speed=request.speed,
                )
            audio = [chunk["tts_speech"].detach().cpu().float().numpy().reshape(-1) for chunk in chunks]
        if not audio:
            raise HTTPException(status_code=500, detail="model produced no audio")
        state["profiles"].consume(prompt)
        return Response(wav_bytes(np.concatenate(audio), model.sample_rate), media_type="audio/wav")

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
