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

