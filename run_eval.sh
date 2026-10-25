#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
rm -f EVAL_DONE EVAL_FAILED
exec > >(tee -a eval.log) 2>&1
echo "=== EVAL START $(date)"
source .venv/bin/activate
python -c "import ai_edge_litert" 2>/dev/null || uv pip install ai-edge-litert || uv pip install tensorflow
python eval_accuracy.py
RC=$?
echo "=== EVAL END rc=$RC $(date)"
if [ $RC -eq 0 ]; then touch EVAL_DONE; else touch EVAL_FAILED; fi
