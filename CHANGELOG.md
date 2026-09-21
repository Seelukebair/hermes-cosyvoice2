# Changelog

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
