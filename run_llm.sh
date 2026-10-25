#!/bin/bash
# Qwen3-0.6B W4A16 -> QNN context binaries -> profile on Samsung Galaxy S25 (Snapdragon 8 Elite) via AI Hub
set -uo pipefail
cd "$(dirname "$0")"
rm -f LLM_DONE LLM_FAILED
exec > >(tee -a llm_export.log) 2>&1
echo "=== LLM START $(date)"; date -u +%Y-%m-%dT%H:%M:%SZ > llm_start_utc.txt
sysctl -n hw.memsize 2>/dev/null | awk '{printf "host RAM: %.0f GB\n", $1/1e9}'; df -h . | tail -1
if [ ! -d .venv_llm ]; then uv venv --python 3.12 .venv_llm || python3 -m venv .venv_llm; fi
source .venv_llm/bin/activate
uv pip install "qai-hub-models==0.65.0" || pip install "qai-hub-models==0.65.0"
qai-hub-models install qwen3_0_6b -y
qai-hub list-devices 2>/dev/null | grep -i "Galaxy S25" | head -3
export TOKENIZERS_PARALLELISM=false
qai-hub-models export qwen3_0_6b \
  --checkpoint DEFAULT_W4A16 \
  --runtime geniex_qairt \
  --device "Samsung Galaxy S25 (Family)" \
  --context-lengths 1024,4096 \
  --output-dir llm_export
RC=$?
echo "=== EXPORT END rc=$RC $(date)"
python llm_metrics.py 6
RC2=$?
echo "=== LLM END rc=$RC/$RC2 $(date)"
if [ $RC -eq 0 ] && [ $RC2 -eq 0 ]; then touch LLM_DONE; else touch LLM_FAILED; fi
