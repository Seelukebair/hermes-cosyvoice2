# Contributing

Keep the Hermes plugin, runtime, and optional integrations separated. Runtime
state and credentials must remain outside the repository. Changes to profile
state require atomic writes and shared locking; changes to the runtime require
CPU tests plus a documented GPU acceptance test.

Before opening a pull request, run the three unit-test commands from the README,
`python -m compileall`, and `bash -n scripts/stage-runtime.sh`. Do not include
voice samples, transcripts, model files, provider keys, or local configuration.
