#!/bin/bash
# One-shot runner: venv -> deps -> configure qai-hub -> run pipeline. Log goes to run.log.
set -uo pipefail
cd "$(dirname "$0")"
rm -f DONE FAILED
exec > >(tee -a run.log) 2>&1
echo "=== START $(date)"
if [ ! -d .venv ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv || uv venv --python 3.11 .venv || uv venv .venv
  else
    PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
    echo "using $PY ($($PY --version))"
    "$PY" -m venv .venv
  fi
fi
source .venv/bin/activate
python --version
if command -v uv >/dev/null 2>&1; then
  uv pip install qai-hub torch torchvision onnx pillow numpy
else
  pip install -U pip && pip install qai-hub torch torchvision onnx pillow numpy
fi
qai-hub configure --api_token "$(cat .qai_token)" >/dev/null && echo "configured qai-hub"
echo "--- device check:"; qai-hub list-devices 2>&1 | grep -i "S24" | head -5
python aihub_mobilenet_int8.py
RC=$?
echo "=== END rc=$RC $(date)"
if [ $RC -eq 0 ]; then touch DONE; else touch FAILED; fi
