---
name: cosyvoice-voice
description: Manage CosyVoice reference-clip discovery, session previews, saved defaults, and voice-associated presentation settings.
---

# CosyVoice Voice Workflow

Use `cosyvoice_voice`; do not invoke download or conversion commands directly.

When the user asks which cloned voices or voice profiles are available, call
`cosyvoice_voice` with `action=list` and report friendly names. Mark the active
session voice and persistent default; do not answer from memory.

1. An authenticated explicit make/build/create/clone request authorizes the
   whole workflow. Use `create` with the requested voice as `query`, a concise
   profile `name`, `make_default=true` when requested, and `enabled=true` unless
   personality was declined. Do not ask for source, transcript, rights, preview,
   or save confirmation.
   The profile name must be only the canonical character or person name, such
   as `JARVIS`, `Optimus Prime`, or `Samuel L. Jackson`. Store era, movie,
   intensity, source, and performance details in profile metadata, never in the
   display name.
   When the user supplies a YouTube URL, pass it as `source_url` to `create` and
   preserve an explicitly requested `start_seconds`; do not replace it with a search.
   Check `list` first. One saved profile is allowed per normalized persona.
   Always reuse the matching saved profile; a different URL or requested style
   must not create a duplicate persona. Exact source URLs are also deduplicated.
   Prepared candidates are temporary and expire after 24 hours unless they are
   the currently active candidate.
2. `create` uses the supplied URL or searches ranked unflagged sources, extracts
   a 12-second reference and enforces the 10-15 second reference contract,
   runs deterministic signal checks, accepts the local ASR transcript as
   conditioning text without labeling it human-verified, tries bounded
   alternate sources on failure, saves the profile, and applies the personality.
   Use `search` only when the user explicitly asks to compare sources.
   Reference quality is not a longest-clip contest. Prefer 10-15 seconds of
   dense, continuous, single-speaker audio with no long dramatic pauses,
   soundtrack, overlapping dialogue, or strong effects. A shorter clean sample
   is better than a 15-second cinematic sample. If a supplied movie source has
   music, use an externally validated vocal-isolated reference or return a clear
   source-quality limitation; do not silently save the contaminated audio.
3. Report the selected source, timings, saved profile id, default/session scope,
   persistence verification, and any returned refinements. Never claim success
   unless `selection.selected_profile_id` matches the created profile and
   `persistence_verified` is true. Run `status` afterward when the request asks
   for explicit verification.
   Then offer two concise listening lines suited to the requested character:
   one short recognizable quote only when you are confident it is accurate,
   and one original task-relevant line that tests the same cadence. Keep each
   line brief enough for a quick voice preview and label the original clearly.
4. The matching character personality is paired with the cloned voice and is
   enabled by default. Express the character through natural temperament, cadence,
   vocabulary, values, and restrained occasional signature phrasing. Answer the
   user directly: never announce or explain the persona, say "As <character>",
   or turn routine answers into theatrical speeches. Use recognizable
   catchphrases sparingly when they fit naturally. Do not
   blend it with a different assistant personality. The selected profile is the
   sole presentation persona; Jarvis remains only the operational role/name.
   Factual accuracy, tool
   discipline, and safety behavior remain unchanged underneath presentation. On `no`, `voice
   only`, or a similar opt-out, call `set_personality` with `enabled=false`.
   When the user requests a materially different presentation style, update the
   existing persona's profile-scoped prompt; do not create another profile for
   the same character or alter the shared template. Profile-specific
   instructions may explicitly allow theatrical delivery when requested.
   Do not alter `SOUL.md`; presentation stays profile-scoped.
5. Read tool errors and present their structured `choices`; do not invent a
   recovery path.

The plugin stores profile selections only. A prepared selection is a one-shot
preview lease for the next default TTS call once an adapter is installed. That
adapter must apply `reference.wav`, top-level `prompt_text`, and optional
delivery prompt to the actual TTS request.
