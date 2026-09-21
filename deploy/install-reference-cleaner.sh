#!/usr/bin/env bash
set -euo pipefail

COSYVOICE_ROOT="${COSYVOICE_ROOT:-/srv/cosyvoice2}"
COSYVOICE_PYTHON="${COSYVOICE_PYTHON:-${COSYVOICE_ROOT}/venv/bin/python}"
CLEANER_VENV="${CLEANER_VENV:-${COSYVOICE_ROOT}/cleaner-venv}"

"${COSYVOICE_PYTHON}" -m venv "${CLEANER_VENV}"
python_version="$("${CLEANER_VENV}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
cosy_site="${COSYVOICE_ROOT}/venv/lib/python${python_version}/site-packages"
printf '%s\n' "${cosy_site}" > "${CLEANER_VENV}/lib/python${python_version}/site-packages/cosyvoice-torch.pth"

pip="${CLEANER_VENV}/bin/pip"
"${pip}" install --upgrade pip
"${pip}" install --no-deps --ignore-installed audio-separator==0.47.0
"${pip}" install --no-deps --ignore-installed \
  'beartype==0.18.5' \
  'diffq==0.2.4' \
  'einops==0.8.2' \
  'julius==0.2.8' \
  'librosa==0.10.2' \
  'ml-collections==1.1.0' \
  'numpy==2.2.6' \
  'onnxruntime==1.23.2' \
  'packaging==24.2' \
  'pydub==0.25.1' \
  'pyyaml==6.0.3' \
  'requests==2.34.2' \
  'resampy==0.4.3' \
  'rotary-embedding-torch==0.6.5' \
  'samplerate==0.1.0' \
  'scipy==1.15.3' \
  'six==1.17.0' \
  'soundfile==0.12.1' \
  'tqdm==4.70.0' \
  'audioread==3.1.0' \
  'absl-py==2.3.1'

"${CLEANER_VENV}/bin/python" - <<'PY'
import torch
from audio_separator.separator import Separator

assert torch.cuda.is_available(), "CUDA is unavailable to the reference cleaner"
print(f"reference cleaner ready: torch={torch.__version__}")
PY
