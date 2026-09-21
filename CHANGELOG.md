# Changelog

## 1.12.0

- Warm saved voice conditioning after profile switches or reference replacement.
- Add test-and-promote CUDA ONNX speech-token preprocessing with health telemetry.
- Stream ordinary Home Assistant conversations directly from Hermes as safe answer
  deltas and synthesize completed sentences while later text is still generated.
- Preserve the Node-RED background route and pre-output fallback behavior.

## 1.11.4

- Resolve concise saved-profile requests such as `Switch to Jarvis` before
  generation, without requiring the words voice, persona, or profile.
- Leave unrelated commands such as `Switch to dark mode` untouched unless the
  target resolves uniquely to a saved voice profile.

## 1.11.3

- Recognize natural saved-profile switch requests including plural voice and
  persona wording, possessive phrasing, and requests to hear a saved identity.
- Resolve minor profile-name misspellings only when one saved profile is a
  clear fuzzy match.
- Apply unambiguous authenticated switches before generation so the outgoing
  personality cannot refuse or override the requested profile change.

## 1.11.2

- Add an explicit `select` action for switching to an existing saved voice and
  its paired personality without changing the persistent default.
- Repair common saved-profile calls that put an ID in `name` or incorrectly use
  `set_personality`, and resolve unambiguous friendly names for compatibility.
- Return explicit recovery hints and copyable profile choices when a profile is
  missing, unknown, or ambiguous.

## 1.11.1

- Align the Hermes command provider and Home Assistant TTS defaults at 1.20x
  delivery speed, with a contract test for requests that omit speed options.

## 1.11.0

- Add explicit post-placement JIT and TensorRT flow acceleration controls while
  keeping both disabled by default.
- Expose effective accelerator state and TensorRT context count through health.
- Add a tested RTX 3090 FP16 TensorRT deployment profile with an isolated
  Python-package overlay and external engine path.
- Extend runtime verification with `--expect-jit` and `--expect-trt` assertions.

## 1.10.0

- Add bounded CPU reference-conditioning caching and persisted-profile startup
  hydration while explicitly excluding one-shot preview candidates.
- Add progressive PCM/WAV synthesis with continuous pitch-preserving 1.10x
  tempo conversion, bounded disconnect draining, and background-thread failure
  recovery without modifying pinned CosyVoice source.
- Stream authenticated audio through the Home Assistant bridge and implement
  Home Assistant's native streaming TTS interface with pre-audio Kokoro fallback.
- Package the companion Hermes Assist conversation integration, publish final
  answers through HA's supported chat-log delta API, and move Node-RED auth to a
  named Home Assistant secret.
- Add content-free health timing/cache metrics and a reproducible latency
  benchmark utility.

## 1.9.3

- Store operator-created runtime and provider-registration rollbacks beneath the
  centralized dated Jarvis backup root instead of beside live files.

## 1.9.2

- Prefer canonical theatrical character performances over actor interviews or
  commentary while retaining interview preference for real-person profiles.
- Preserve character-defining cinematic vocal processing during cleanup.

## 1.9.1

- Enforce character-only profile display names and one saved profile per
  normalized persona, even when a different source URL is supplied.
- Automatically remove abandoned candidates after 24 hours while preserving
  the active preview candidate.

## 1.9.0

- Clean new voice references with an isolated BS-RoFormer runtime before local
  transcription and CosyVoice conditioning.
- Conservatively compact pauses longer than 350 ms while retaining natural
  150 ms joins and enforcing the existing 10-15 second reference contract.
- Fall back to the raw extraction when cleanup is unavailable or invalid, and
  record the cleanup result in profile metadata.

## 1.8.1

- Recognize natural plural inventory questions such as "which cloned voices are
  available" and expose the saved profile list to Hermes.
- Make the selected profile the sole presentation persona while retaining
  Jarvis as the operational role.
- Reject URL-shaped personality labels and repair malformed generated profile
  metadata from friendly profile names.
- Support bounded profile-specific personality prompts for intentionally heavy
  or theatrical variants without changing other voices.

## 1.8.0

- Package the tested CosyVoice2 server, Hermes command adapter, profile resolver,
  runtime checks, systemd template, and optional Home Assistant bridge.
- Add a public transcription-command contract with Hermes fallback support.
- Prevent ordinary chat from creating storage on unconfigured installations.
- Distinguish accepted automatic transcripts from human-verified transcripts.
- Serialize shared profile-state updates across Hermes and the TTS runtime.
- Preserve mannerism-first personalities with occasional natural catchphrases.
