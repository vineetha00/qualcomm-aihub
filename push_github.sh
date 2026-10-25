#!/bin/bash
# Creates the public GitHub repo (under the gh-authenticated account) and pushes main.
set -euo pipefail
cd "$(dirname "$0")"
command -v gh >/dev/null || { echo "gh CLI not found: brew install gh && gh auth login"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "run: gh auth login"; exit 1; }
git branch -M main
gh repo create qualcomm-aihub --public --source=. --remote=origin --push \
  --description "MobileNetV2 INT8 + Qwen3-0.6B W4A16 on Snapdragon NPUs via Qualcomm AI Hub (QNN/TFLite), with latency, memory and accuracy results"
gh repo view --web
