#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HOME}/.local/share/hermes-cosyvoice2"
SOURCE_REVISION=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
MODEL_REVISION=eec1ae6c79877dbd9379285cf8789c9e0879293d
MODEL_ID=FunAudioLLM/CosyVoice2-0.5B
YT_DLP_VERSION=2026.8.19
DOWNLOAD_MODEL=0

usage() {
  echo "usage: $0 [--prefix PATH] [--download-model]"
}

while (($#)); do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --download-model) DOWNLOAD_MODEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

for command in git ffmpeg sox python3.10; do
  command -v "$command" >/dev/null || { echo "missing prerequisite: $command" >&2; exit 2; }
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="${JARVIS_BACKUP_ROOT:-/srv/jarvis-backups}"
BACKUP_DIR="$BACKUP_ROOT/$(date +%F)/cosyvoice2/${STAMP}-runtime-stage"
mkdir -p "$PREFIX/data/voice_profiles"

if [[ -d "$PREFIX/source/.git" ]]; then
  current="$(git -C "$PREFIX/source" rev-parse HEAD)"
  [[ "$current" == "$SOURCE_REVISION" ]] || {
    echo "existing source is $current, expected $SOURCE_REVISION; stage upgrades separately" >&2
    exit 3
  }
else
  git clone --recursive https://github.com/QwenAudio/CosyVoice.git "$PREFIX/source"
  git -C "$PREFIX/source" checkout --detach "$SOURCE_REVISION"
  git -C "$PREFIX/source" submodule update --init --recursive
fi

if [[ ! -x "$PREFIX/venv/bin/python" ]]; then
  python3.10 -m venv "$PREFIX/venv"
fi
"$PREFIX/venv/bin/python" -m pip install --upgrade pip
"$PREFIX/venv/bin/pip" install -r "$REPO_ROOT/runtime/requirements.txt"

if [[ ! -x "$PREFIX/voice-tools/venv/bin/python" ]]; then
  mkdir -p "$PREFIX/voice-tools"
  python3.10 -m venv "$PREFIX/voice-tools/venv"
fi
"$PREFIX/voice-tools/venv/bin/pip" install "yt-dlp==$YT_DLP_VERSION"

if [[ $DOWNLOAD_MODEL -eq 1 && ! -f "$PREFIX/model/cosyvoice2.yaml" ]]; then
  "$PREFIX/venv/bin/python" - "$MODEL_ID" "$MODEL_REVISION" "$PREFIX/model" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
fi
[[ -f "$PREFIX/model/cosyvoice2.yaml" ]] || {
  echo "model missing; rerun with --download-model or place the pinned snapshot at $PREFIX/model" >&2
  exit 4
}

if [[ -d "$PREFIX/app" ]]; then
  mkdir -p "$BACKUP_DIR"
  chmod 700 "$BACKUP_DIR"
  cp -a "$PREFIX/app" "$BACKUP_DIR/app"
fi
mkdir -p "$PREFIX/app"
cp "$REPO_ROOT"/runtime/*.py "$PREFIX/app/"
cp "$REPO_ROOT/runtime/requirements.txt" "$PREFIX/app/requirements-runtime.txt"

sed \
  -e "s|@SERVICE_USER@|$(id -un)|g" \
  -e "s|@SERVICE_GROUP@|$(id -gn)|g" \
  -e "s|@PREFIX@|$PREFIX|g" \
  -e "s|@SOURCE_REVISION@|$SOURCE_REVISION|g" \
  -e "s|@MODEL_REVISION@|$MODEL_REVISION|g" \
  "$REPO_ROOT/deploy/systemd/cosyvoice2.service" >"$PREFIX/cosyvoice2.service"

echo "runtime staged at $PREFIX"
echo "rendered service unit: $PREFIX/cosyvoice2.service"
echo "review and install the unit with your operating system service manager"
