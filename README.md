# Hermes CosyVoice2

A complete local CosyVoice2 integration for Hermes Agent. The repository keeps
four boundaries explicit:

| Component | Purpose |
|---|---|
| Hermes plugin | Discovers, validates, saves, selects, and injects paired voice/personality profiles. |
| CosyVoice2 runtime | Loopback FastAPI synthesis service with configurable CPU/GPU component placement. |
| Hermes TTS adapter | Registers the runtime as a normal command-based Hermes TTS provider. |
| Home Assistant integration | Optional authenticated LAN bridge and config-entry TTS provider with fallback. |

Model weights, generated voices, transcripts, credentials, and runtime state are
never stored in Git.

## Tested pins

The reviewed [voice latency deployment plan](docs/voice-latency-deployment-plan.md)
records the deployed reference cache, startup hydration, progressive Home
Assistant streaming, measurements, and rollback. The buffered command endpoint
remains available for Hermes and compatibility clients. The RTX 3090 reference
deployment uses a tested FP16 TensorRT flow-decoder engine. JIT remains a
separate test-and-promote experiment.

- CosyVoice source: `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`
- `FunAudioLLM/CosyVoice2-0.5B`: `eec1ae6c79877dbd9379285cf8789c9e0879293d`
- Python 3.10, PyTorch 2.3.1/CUDA 12.1 wheels
- Hermes plugin API validated with `hermes plugins doctor`

The runtime follows the official [QwenAudio CosyVoice](https://github.com/QwenAudio/CosyVoice)
zero-shot interface but pins known-working revisions instead of following a
moving branch.

## Installation

### 1. Runtime

Ubuntu prerequisites are Python 3.10 with `venv`, Git, FFmpeg, SoX, a working
NVIDIA driver for GPU placement, and enough disk for roughly 11 GB of runtime,
source, and model files.

```bash
git clone https://github.com/Seelukebair/hermes-cosyvoice2.git
cd hermes-cosyvoice2
./scripts/stage-runtime.sh --prefix /srv/cosyvoice2 --download-model
```

The staging script is intentionally conservative. It refuses to replace a source
checkout at another revision and backs up the deployed adapter before copying a
new one. It writes only below the selected prefix and renders a service unit at
`<prefix>/cosyvoice2.service`; review and install that unit through the host's
service manager. The script does not modify system service or Hermes configuration.

Component placement lives in `/etc/default/cosyvoice2`:

```ini
COSYVOICE_PLACEMENT=all-gpu
COSYVOICE_GPU_WEIGHT_DTYPE=fp16
```

Supported placements are `cpu`, `llm-gpu`, `acoustic-gpu`, `llm-flow-gpu`,
`llm-hift-gpu`, and `all-gpu`. HiFT remains FP32 because its oscillator path
constructs FP32 tensors internally.

### 2. Hermes provider and plugin

Register the command provider with an atomic config backup under
`${JARVIS_BACKUP_ROOT:-/srv/jarvis-backups}/YYYY-MM-DD/cosyvoice2/`:

```bash
python scripts/register_hermes_provider.py --activate --speed 1.20
hermes gateway restart
```

Install this repository through Hermes and enable it:

```bash
hermes plugins install Seelukebair/hermes-cosyvoice2 --enable
hermes plugins doctor cosyvoice-voice
```

For reproducible installations, pin a full commit SHA with `--ref`. Hermes keeps
Git-installed plugin provenance, so later `hermes plugins update cosyvoice-voice`
can use the normal update path. Pinned installations require an explicit new
commit when upgrading. See the current [Hermes plugin documentation](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/plugins.md).

### 3. Verify

```bash
python scripts/check_runtime.py \
  --expect-source 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc \
  --expect-model eec1ae6c79877dbd9379285cf8789c9e0879293d \
  --expect-trt
python -m unittest discover -s tests -v
PYTHONPATH=runtime python -m unittest discover -s runtime/tests -v
```

The tested TensorRT settings are in `deploy/cosyvoice2-tensorrt.env`; the
matching systemd Python-path overlay is `deploy/cosyvoice2-tensorrt.conf`.
TensorRT is deliberately installed outside the production virtualenv and the
GPU-specific engine remains outside the model directory. The live engine
manifest records its source ONNX hash, engine hash, GPU compute capability,
driver, precision, revisions, and context count. Never copy a plan file to a
different GPU/runtime combination without rebuilding and validating it.

The read-only health check confirms readiness, placement, and both pinned
revisions. A production acceptance test should additionally synthesize a short
line through Hermes and decode the resulting WAV.

`GET /health` also reports the bounded conditioning-cache state, persisted-voice
warm-up result, and content-free metadata for the last synthesis. Progressive
clients use `POST /synthesize-stream`; the existing `POST /synthesize` contract
is unchanged.

## Voice profiles

The default profile root is `/srv/cosyvoice2/data/voice_profiles`. It remains
outside the plugin checkout so plugin updates cannot erase voices.

Each saved profile contains:

- `reference.wav`: mono, 24 kHz, 16-bit PCM reference audio.
- `profile.json`: source provenance, signal metrics, conditioning transcript,
  delivery settings, and paired character personality.
- `state.json`: accepted session selection and persistent default.

Voice creation can run an isolated reference cleaner before transcription by
setting `COSYVOICE_CLEAN_REFERENCE_COMMAND`. The command receives
`{input_path}` and `{output_path}` placeholders. `runtime/clean_reference.py`
provides the production implementation: BS-RoFormer dialogue isolation followed
by conservative removal of pauses longer than 350 ms. Cleaned output must still
pass the normal 10-15 second signal contract; otherwise creation retains the raw
reference and records `fallback_raw` in profile metadata. Use
`deploy/install-reference-cleaner.sh` to create the isolated runtime and
`deploy/hermes-gateway-reference-cleaner.conf` to activate it for Hermes.
- `recent_sources.json`: bounded source authorization used by the acquisition workflow.

The plugin exposes `status`, `list`, `search`, `create`, `prepare`, `refine`,
`set_personality`, `accept`, `set_default`, `reset`, and `discard` through the
`cosyvoice_voice` tool. Explicit creation requests may complete acquisition,
validation, saving, and selection in one call.

Automatic ASR transcripts are marked `accepted: true, verified: false` because
they are usable conditioning text but are not human-verified. A corrected user
transcript is marked verified. `COSYVOICE_TRANSCRIBE_COMMAND` provides a public
adapter contract; the packaged default invokes `runtime/transcribe.py`.

Voice and personality stay paired. Personality prompts favor natural cadence,
temperament, vocabulary, and values; they avoid announcing the role or turning
routine answers into speeches while still allowing occasional catchphrases when
they fit naturally.

## Safety and operating model

- The plugin is currently **single-user**. Session/default selectors are global
  to one Hermes installation. Do not expose one instance to mutually untrusted users.
- Source acquisition invokes yt-dlp and FFmpeg with argument arrays, not shell interpolation.
- JSON state writes use same-directory atomic replacement and a shared file lock.
- The synthesis API binds to loopback by default and serializes GPU inference.
- Review source rights and obtain permission before cloning a person's voice.
- Keep profile WAVs and transcripts in protected local storage and backups.
- Keep Hermes install-time plugin scanning enabled.

## Home Assistant

The optional package under [`integrations/home-assistant`](integrations/home-assistant)
contains a config-entry TTS entity and an authenticated bridge. The bridge reads
the same accepted session/default selector as Hermes, rejects preview candidates,
limits concurrent backend work, and can fall back to another HA TTS entity.

It is deliberately separate from the core plugin because deploying it requires
Home Assistant-specific configuration, a LAN address allow-list, and a dedicated
secret. No secret is included in the repository.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
PYTHONPATH=runtime python -m unittest discover -s runtime/tests -v
python -m unittest discover -s integrations/home-assistant/tests -v
```

Runtime and profile behavior are covered by CPU-only tests. GPU synthesis,
speaker similarity, and end-to-end audio delivery remain deployment acceptance
tests because they require the pinned model and real hardware.

## License

Integration code is Apache-2.0. CosyVoice, model weights, and external tools
retain their own licenses and notices; see [NOTICE](NOTICE).
