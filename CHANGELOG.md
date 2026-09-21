# Changelog

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
