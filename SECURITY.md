# Security

Report vulnerabilities privately through GitHub's security-advisory interface.
Do not include credentials, private voice samples, or generated profile data in
an issue.

The plugin executes `yt-dlp`, FFmpeg, and an optional transcription command.
Install only reviewed revisions, keep Hermes plugin scanning enabled, and do
not point `COSYVOICE_TRANSCRIBE_COMMAND` at an untrusted executable. The TTS
server binds to loopback by default. Any network bridge must authenticate
requests and restrict source addresses.

Voice cloning requires permission to use the source audio and the resulting
voice. Profile WAV files and transcripts are sensitive local data and are not
part of this repository.

## Compatibility-pinned runtime

CosyVoice2 currently requires an older, mutually compatible ML dependency
stack. Automated scanners report known advisories against several pinned
packages, including Torch, ONNX, Transformers, and Lightning. Do not expose the
runtime API directly to an untrusted network, load untrusted model or pickle
artifacts, or grant untrusted users filesystem access to its model tree.

Dependency upgrades must be tested with model loading, profile synthesis, WAV
validation, and GPU placement before promotion. Treat a scanner-only version
bump as unsafe until that acceptance suite passes. The command provider and
Home Assistant bridge should remain the authenticated boundary around the
loopback synthesis service.
