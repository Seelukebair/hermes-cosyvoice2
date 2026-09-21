# CosyVoice2 voice latency deployment plan

Status: CVL-01 through CVL-04 deployed and automatically validated on
2026-09-21. Physical mobile playback/listening approval remains operator-owned.
CVL-05 is deferred as an optional test-and-promote experiment.

## Deployment result

- Runtime, authenticated bridge, HA native streaming TTS, and companion Assist
  delta handoff are live. The legacy buffered synthesis endpoint remains active.
- Five warm direct streaming repetitions measured median first PCM at 3.14 s
  (short), 4.75 s (medium), and 4.59 s (long). The long fixture completed in a
  median 16.76 s, so playback can begin roughly 12 seconds before completion.
- Persisted-profile startup hydration is enabled. The deployed restart warmed
  the saved Samuel profile and its CPU conditioning cache in 3.77 s without
  selecting or consuming a preview candidate.
- An authenticated bridge disconnect test delivered first PCM, disconnected,
  drained the in-flight native job, and completed the immediate recovery request.
  A discovered upstream background-thread failure hang was corrected with a
  local wrapper that signals completion and cleans per-request dictionaries.
- HA API read-back confirmed `conversation.hermes_proxy_agent` and
  `tts.jarvis_cosyvoice`; buffered compatibility produced a decodable 2.93 s MP3.
  The protected conversation adapter returned a 108-character answer without
  logging content or credentials.
- With Gemma and CosyVoice resident after acceptance, the GPU reported 20,931
  MiB used and 3,194 MiB free. JIT/TensorRT was not promoted because streaming
  solved the delivery bottleneck without consuming the remaining shared-GPU
  margin.
- Rollback snapshot:
  `/srv/jarvis-backups/2026-09-20/cosyvoice-latency/20260921T073603Z-streaming-deployment`.
Review date: 2026-09-20 America/Anchorage.
Repository baseline: `42ec691b0d47533c903f9e4a08ffc6377339487d`.

## Outcome and boundaries

Make Home Assistant voice replies start sooner while preserving the selected
CosyVoice voice, its delivery instructions, the paired Hermes personality,
1.10x speaking rate, and the existing text response behavior. Keep the standard
Hermes command-provider WAV endpoint working. Use supported HA APIs and adapters
in this repository; do not edit Hermes, HA Core, or pinned CosyVoice sources.

Only one TTS GPU runtime should be loaded during tests. Keep Gemma resident for
acceptance measurements. A faster result obtained by unloading Gemma is a lab
result and cannot qualify as the production result.

## Review findings

| ID | Severity | Verified finding | Deployment implication |
|---|---|---|---|
| R01 | High | Runtime uses `stream=False`, collects all chunks, and returns a complete WAV. Proxy calls `response.read()`; HA calls `await response.read()`. | All three layers must support incremental audio. Buffering at any one defeats early playback. These are overlapping waits for the same utterance, not three complete synthesis operations. |
| R02 | High | Installed HA is 2026.9.2 and has `async_stream_tts_audio`. Current `hermes_wyoming_agent` returns a completed `IntentResponse` after a Node-RED JSON response; it does not publish text deltas. Independent review of the installed Assist pipeline confirms this takes the buffered TTS path. | A conversation-adapter change is required for HA Assist streaming. Distinguish streaming audio after final text from synthesis during LLM text generation. |
| R03 | High | CosyVoice2's streaming branch never passes `speed` to `token2wav`. Its non-streaming speed interpolation cannot be used with a populated HiFT stream cache. | Simply enabling streaming silently loses 1.10x. Add one continuous pitch-preserving tempo stage or keep buffered mode until that works. |
| R04 | High | `frontend_instruct2` obtains cached `prompt_text` through `frontend_zero_shot`; a cached transcript can replace the requested instruction. | Cache zero-shot and instruct2 separately, keyed by effective conditioning text. Test instruction changes and profile replacement under the same ID. |
| R05 | High | Pinned streaming code grows `self.token_hop_len` from 25 toward 100 across yields and does not reset it per request. Cleanup follows normal generator completion; LLM generation uses a background thread. | Keep serialized synthesis, restore initial chunk state per request, and explicitly implement disconnect/error cleanup. An abandoned iterator must not leave a worker or request dictionaries active. |
| R06 | Medium | Wrapper forces CUDA unavailable during initial model construction, then moves LLM/flow/HiFT to GPU. Frontend ONNX sessions are constructed for CPU. | 'All-GPU' describes synthesis components, not reference preprocessing. Cache features before spending VRAM on frontend acceleration. |
| R07 | Medium | Wrapper disables JIT/TRT/vLLM and loads initially with `fp16=False`; upstream also disables accelerator flags when CUDA is unavailable. | Accelerator flags alone are insufficient. Load compatible artifacts after placement or provide a verified all-GPU loader path; health must expose effective acceleration. |
| R08 | Medium | Live proxy requires `transcript.verified`; Git accepts `transcript.accepted` with a legacy fallback. Runtime file hash matches Git, proxy differs. | Reconcile this specific drift so a normally accepted voice works equally in Hermes and HA. Preserve saved-profile and preview-isolation checks. |
| R09 | Medium | Live conversation adapter contains an inline credential fallback and debug logging of full inputs/responses. | If that adapter is touched, move auth to protected configuration and use metadata-only diagnostics; never copy the credential into this public repository. |

Observed synthesis log RTF is commonly 0.85-0.95 on recent warm requests, with
slower examples too. This excludes some frontend/queue/delivery work and is not
a controlled benchmark. There is no measured phone playback latency baseline yet.
The previous attempted synthetic benchmark did not run due to orchestration
errors; do not treat it as evidence. Published 50%, 2-3x, and sub-250ms claims
are not acceptance promises for our installation.

Current pins: source `074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`,
model `eec1ae6c79877dbd9379285cf8789c9e0879293d`. Repository requirements pin
PyTorch 2.3.1; inventory effective installed packages before building accelerators.
Synthesis LLM/flow use FP16; HiFT uses FP32. Last idle GPU snapshot was 20,915 MiB
used and 3,210 MiB free on the 24,576 MiB GPU. This is not peak headroom.

## Phased work queue

Each phase produces a bounded commit, benchmark comparison, and independently
reversible switch. Status values: planned, implementing, validated, deployed,
rejected. Do not mark deployed based only on a successful HTTP response.

### CVL-01: Baseline, observability, and deployment parity

Status: deployed. Dependencies: none.

1. Inventory live source hashes, service/environment settings with values
   redacted where sensitive, effective Python/CUDA/ONNX packages, HA version,
   selected profile revision, and Gemma configuration. Capture without private
   transcripts or credential files in Git.
2. Reconcile proxy acceptance behavior with the maintained Git contract; test
   accepted-only, verified legacy, rejected, missing, and preview profiles.
3. Add request IDs and monotonic timings for queue wait, reference preprocessing,
   first generated audio, synthesis completion, first forwarded audio, HA first
   consumed bytes, and completion. Record audio duration, cache hit/miss, effective
   mode, rate, and accelerator flags. Do not log spoken text or tokens.
4. Add a reproducible benchmark script with fixed synthetic text fixtures,
   fixed seeds where supported, named configuration snapshots, JSON results,
   and peak GPU/RSS sampling. Seed control reduces variation; it does not promise
   identical waveforms across different kernels.
5. Measure direct backend, authenticated proxy, HA TTS API, and actual Assist
   playback separately. HTTP first bytes are not proof of audible playback.

Acceptance: repeatable baseline across at least five warm repetitions of short,
medium, and long fixtures, plus first request after service restart. Report
median/range; larger soak samples may report p95. No synthesis invoked during
routine readiness probes. No model unloading during production benchmarks.

### CVL-02: Cache reference conditioning and warm the selected profile

Status: deployed. Dependencies: CVL-01.

Files: `runtime/cosyvoice_server.py`, a small cache helper if needed,
`runtime/tests/`, `runtime/cosyvoice2.env.example`.

1. Use the existing upstream frontend speaker-cache interface through a narrow
   adapter. Cache by source/model revision, profile identity, reference content
   hash, inference mode, and effective conditioning text hash.
2. For zero-shot, condition on the reference transcript. For instruct2, build
   with effective instructions and preserve upstream's removal of LLM prompt
   speech tokens. Do not reuse transcript-conditioned cache entries for instruct2.
3. Start with a bounded in-process CPU cache (e.g. eight entries and a byte cap).
   No persistent pickles or writes to the model's bundled `spk2info.pt`. Reference
   replacement, metadata edits, deletion, and instruction changes invalidate it.
4. Pin the resolved profile/revision for each request. Concurrent profile edits
   must either use a consistent old snapshot or the new one, never a mixture.
5. Precompute the accepted default after startup and run a short internal warm-up
   with an explicit saved ID. Warm-up must never consume a one-shot preview,
   play audio, or change voice/personality selectors. Expose warming versus ready.

Acceptance: cache hits skip reference extraction, cached/uncached controlled
tests retain conditioning, all three retained voices work, edits invalidate,
warm-up leaves state unchanged, and memory use remains bounded. Keep only if
measured latency improves without a voice regression.

### CVL-03: Runtime streaming and lifecycle correctness

Status: deployed. Dependencies: CVL-02.

1. Add an opt-in streaming endpoint beside `/synthesize`; keep the existing WAV
   response contract for Hermes command clients and rollback.
2. Use the pinned native audio chunk generator with a serialized worker and a
   bounded producer/consumer queue. Start from the trained chunk size of 25;
   reset request state rather than shrinking arbitrary model chunk parameters.
3. Reuse cached conditioning. Validate profile, request size, and admission
   before sending success headers. Apply queue, first-chunk, idle, and total
   deadlines. Do not hold the event loop during inference.
4. Preserve 1.10x via one continuous FFmpeg `atempo=1.10` stage for the whole
   utterance, flushing once at the end. Feed CosyVoice streaming at native speed.
   Test pitch, boundaries, drift, process cleanup, and added first-audio delay.
   Do not stretch chunks independently or quietly revert to 1.0.
5. Start with HA's native Wyoming streaming WAV contract: its unknown-length
   header followed by raw PCM, verified against the installed implementation.
   Confirm progressive decoding on the actual app; use a continuous FLAC stream
   only if required and supported. Never concatenate complete WAV files or
   advertise a fabricated final data length. Keep native sample rate and mono.
6. Own worker/generator cleanup explicitly. Upstream has no proven immediate
   LLM cancellation here: first version may stop delivery and drain a bounded
   inference job before releasing the slot. Document that behavior. If workers
   cannot be bounded and recovered, streaming is not ready to promote.
7. After failure/cancel, next request must succeed with fresh chunk state and
   no retained request dictionaries, orphaned threads, or encoder subprocesses.

Acceptance: first decodable audio arrives before synthesis completion, three
consecutive requests have stable startup behavior, long output has no missing
tail/duplicate chunks, 1.10x remains effective, and cancellation releases bounded
resources. Voice quality checked against buffered baseline on identical text.

### CVL-04: Bridge and Home Assistant progressive playback

Status: deployed; physical mobile playback remains pending operator listening.
Dependencies: CVL-03; conversation adapter is in required scope.

Files: `integrations/home-assistant/jarvis-cosyvoice-ha-proxy.py`, its tests,
`custom_components/jarvis_cosyvoice_tts/tts.py`, constants/config flow as needed.
Companion component: live `/config/custom_components/hermes_wyoming_agent/`;
verify its maintained source under `_Jarvis_vscode/hermes_wyoming_agent` and
deploy its changes as a separately versioned integration, not an HA Core patch.

1. Forward streaming bytes with backpressure; do not call unbounded `read()`.
   Use valid HTTP framing and no invented Content-Length. Hold admission capacity
   for the stream lifetime and close upstream on client disconnect.
2. Preserve existing LAN binding, authentication, accepted-profile selection,
   request limits, and preview isolation. Resolve the voice once per utterance.
3. Implement HA's native `async_stream_tts_audio(TTSAudioRequest)` and
   `TTSAudioResponse`, while retaining the mandatory one-shot method. Bound text
   buffering and segment on sensible sentence boundaries; never synthesize each
   individual LLM token. Initial scope can accept one complete text message.
4. Installed Assist requires chat-log deltas. Make the smallest native
   conversation-adapter change to publish the completed answer through its
   supported chat-log delta API without duplicate final speech. Verify the
   installed streaming character threshold and behavior for short answers;
   short replies may legitimately stay buffered. Do not manually call internal
   pipeline listeners or emit dummy text to bypass the threshold. Keep that
   change separately scoped and versioned, and remove inline auth from source.
5. True overlap with Gemma generation is optional follow-up: it also needs
   authenticated SSE through Hermes/Node-RED and native HA chat-log deltas.
   Do not claim token-to-audio streaming from a final-text-only route. Never
   speak reasoning, tool JSON, interim tool results, or a stale acknowledgement.
6. Before first emitted audio, allow buffered CosyVoice or existing Kokoro
   fallback as appropriate. After any emitted audio, terminate cleanly on error;
   do not replay the whole answer or change voices midway. Report failure.
7. Inspect HA generated-audio cache identity. Identical text after a voice switch
   must use the new profile. Include a profile revision in the effective cache
   identity via supported integration mechanisms, or explicitly avoid caching
   dynamic-default requests where HA permits it. Verify rather than assuming.

Acceptance: mobile app plays while backend generation is still running, correct
voice follows a shared profile switch, no duplicate speech, stream cancellation
works, buffered endpoint still passes, and fallback before audio is observable.
If app decoding buffers despite server streaming, report that as a blocker.

### CVL-05: Optional JIT and TensorRT acceleration

Status: deferred. Dependencies: CVL-01; test against accepted streaming/cache build.

1. Inventory exact artifacts, versions, and kernels. Stage immutable candidate
   artifacts outside model/profile data; never upgrade the production venv in place.
2. JIT: load matching FP16 flow-encoder artifact after correct device placement.
   Confirm effective activation, numerical compatibility, and startup memory.
3. TensorRT: build a compatible flow-decoder engine for the actual GPU, driver,
   CUDA, precision, and supported input shapes. Record hashes and a manifest.
   Keep one execution context initially and bound workspace memory.
4. Separate build-time resource use from inference. If build needs more memory,
   use the workstation where compatible and validate the resulting engine on
   Jarvis; engine portability is not assumed. Do not unload Gemma secretly.
5. Test cache-only, cache+JIT, and cache+JIT+TRT separately. Promote a variant only
   if latency improves meaningfully (target >=10% median improvement), p95 and
   stability do not regress, and voice quality stays acceptable.

Deferred: vLLM, Triton/TensorRT-LLM, LightTTS, alternative model versions, changing
diffusion step counts, removing allocator synchronization, and quantizing HiFT.
Pinned vLLM helper requests `gpu_memory_utilization=0.2`, approximately 4.8 GiB
of a 24 GiB GPU, and introduces incompatible-version risk. It needs its own
capacity study; it is not a free toggle within the current ~3.2 GiB idle margin.

## Acceptance matrix and target measurements

| Test | Required result |
|---|---|
| Saved voices | Samuel, Optimus, and Jarvis retain reference, instructions, and paired personality; no selector changes from benchmarks. |
| Modes | Zero-shot and instruct2, cold miss and warm hit, instruction override and metadata replacement all work. |
| Lengths | Short confirmation, 30-60-word reply, 200-300-word reply, maximum accepted input, and over-limit rejection. |
| Content | Numbers, punctuation, quoted phrases, abbreviations, and existing expressive delivery remain intelligible. |
| Repetition | Multiple sequential streams do not inherit increased startup chunk sizes or grow memory indefinitely. |
| Shared GPU | Test typical Gemma load plus configured maximum concurrency; record context and measured GPU/RAM peaks. No OOM/restart or CPU synthesis fallback. |
| Overload | One active synthesis worker initially, bounded queue, honest overload error, no unbounded memory growth. |
| Faults | Backend down, timeout before first chunk, disconnect midstream, bad metadata, slow reader, and decoder failure produce bounded cleanup. |
| Profile changes | Same text after switch speaks new profile; an in-flight utterance does not switch voices partway through. |
| Clients | HA app progressive playback and legacy Hermes WAV use both pass; HA API success alone is insufficient. |

Provisional performance target: warm first playable audio after text submission
<=2 seconds median for a medium fixture, or at least 50% improvement over measured
baseline; report both absolute and relative results. Target sustained RTF <=0.7
to leave room for shared-GPU variation. These are test targets, not predictions.
The final promotion decision also requires no underruns in long mobile playback.
Sample GPU memory during generation, not only before/after. Aim to preserve at
least ~800 MiB under representative peak load; if missed, report actual margin
and adjust the candidate before changing Gemma's tested production profile.

## Deployment and rollback

1. Build in this Git repository, with bounded delegates for cache, HA/proxy, and
   tests. Primary agent owns runtime lifecycle integration and independent QC.
2. Capture only scoped scripts/configuration and a manifest under
   `/srv/jarvis-backups/YYYY-MM-DD/cosyvoice-latency/<timestamp>-<operation>/`.
   Keep protected HA/Jarvis configuration private. Follow existing seven-day
   backup retention; do not duplicate model weights or entire old runtimes.
3. Stage a candidate release with exact hashes and explicit switches for cache,
   streaming, and acceleration. New code starts with streaming/accelerators off.
4. Deploy compatible runtime, then bridge, then HA integration. Reload the HA
   config entry if supported; otherwise restart HA Core only. Do not reboot the
   host, restart Hermes, or recreate models for a TTS-only adapter change.
5. Enable and validate cache, then streaming, then each acceleration candidate.
   Keep a completed benchmark/result row for every promoted or rejected variant.
6. Soak representative requests with Gemma active, then verify actual phone
   playback. Mark mobile auditory approval pending when operator testing is
   unavailable, even if automated audio/transport tests pass.
7. Update README, CHANGELOG, HA README, jarvis1.txt deployment inventory without
   secrets in Git, ops-playbook, and Jarvis voice health checks. Health exposes
   effective mode, cache/warm status, last synthesis result, and queue condition;
   routine polling does not generate audio or trigger fallback.
8. Roll back the failing feature switch first. For code/runtime failure restore
   the preceding scripts/venv target and restart only affected services. Keep
   profiles and their defaults intact. Verify buffered CosyVoice and HA speech.
   Do not remove the working HA integration as the normal rollback method.

Completion report: component/version matrix, exact promoted flags, first-audio
and total-time comparisons, RTF, peak VRAM/RAM, quality observations, mobile
playback evidence, cancellation/fallback results, rollback location, and any
deferred work. The report must distinguish measured facts from target numbers.

## References

- [HA native TTS interface](https://developers.home-assistant.io/docs/core/entity/tts/).
- [HA native Wyoming streaming implementation](https://github.com/home-assistant/core/blob/2026.9.2/homeassistant/components/wyoming/tts.py).
- [CosyVoice frontend and caching](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/frontend.py).
- [Pinned generation and streaming](https://github.com/QwenAudio/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/cosyvoice/cli/model.py).
- [Optional official accelerated runtime](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.Unet.md).
- [Community async runtime](https://github.com/qi-hua/async_cosyvoice): author-reported measurements on a different stack, not Jarvis benchmarks.

Independent read-only review by a Terra agent confirmed HA's buffered fallback
and the missing chat-log delta handoff. Primary review verified the pinned
CosyVoice instruction cache, streaming speed/state behavior, deployment drift,
and shared-GPU constraints. No new audio benchmark or deployment was performed
during this planning review.
