# On-device ML on Snapdragon with Qualcomm AI Hub

Reproducible experiments that take models from PyTorch to Qualcomm's Hexagon NPU through
[Qualcomm AI Hub](https://aihub.qualcomm.com) (compile / quantize / profile jobs on hosted
Snapdragon devices), with every number traceable to an AI Hub job URL.

## 1. MobileNetV2: INT8 post-training quantization on Snapdragon 8 Gen 3

**Target:** Samsung Galaxy S24 (Snapdragon 8 Gen 3, Hexagon v75 NPU), TFLite runtime.
**Pipeline** (`aihub_mobilenet_int8.py`): `torch.export` → AI Hub compile to FP32 ONNX →
AI Hub quantize job (INT8 weights + INT8 activations, 100 Imagenette calibration images) →
compile to INT8 TFLite with `--quantize_io` → profile on device. FP32 compiled and profiled
the same way for comparison.

| | FP32 TFLite | INT8 TFLite |
|---|---|---|
| Inference latency (NPU) | 0.37 ms | **0.28 ms** (1.31× faster) |
| Peak inference memory | 210.5 MB | **149.3 MB** (−29%) |
| Layers on NPU | 70 / 70 (100%) | 105 / 105 (100%) |
| Top-1, Imagenette val (3,925 imgs, 1000-way) | 79.03% | **78.85%** (−0.18 pp) |
| Top-5 | 95.69% | 95.31% |
| Top-1 agreement INT8 vs FP32 | | 94.4% |

Accuracy was measured on the exact compiled `.tflite` artifacts with the TFLite interpreter
(`eval_accuracy.py`); the FP32 TFLite artifact matches the original PyTorch model image-for-image,
so the compile step is lossless and the −0.18 pp is purely quantization error.

AI Hub jobs: [FP32 ONNX compile](https://workbench.aihub.qualcomm.com/jobs/jp0m3mn2g/) ·
[INT8 quantize](https://workbench.aihub.qualcomm.com/jobs/jp8e0elzp/) ·
[INT8 TFLite compile](https://workbench.aihub.qualcomm.com/jobs/jp3zvzyx5/) ·
[FP32 TFLite compile](https://workbench.aihub.qualcomm.com/jobs/jgolklj4g/) ·
[INT8 profile](https://workbench.aihub.qualcomm.com/jobs/jgjrzrj7p/) ·
[FP32 profile](https://workbench.aihub.qualcomm.com/jobs/jpe7e7j75/)

Raw outputs: `results.json`, `profile_int8.json`, `profile_fp32.json`, `accuracy_results.json`, `run.log`, `eval.log`.

## 2. Qwen3-0.6B W4A16 LLM on Snapdragon 8 Elite via QNN (in progress)

**Target:** Samsung Galaxy S25 (Snapdragon 8 Elite). **Pipeline** (`run_llm.sh`):
`qai-hub-models export qwen3_0_6b --checkpoint DEFAULT_W4A16 --runtime geniex_qairt` —
int4 weights / int16 activations / int8 KV cache (SpinQuant + AdaScale recipe), split into
prompt-processor and token-generator graphs at 1024 and 4096 context, compiled and linked
into QNN (QAIRT) context binaries, each graph profiled on device. `llm_metrics.py` derives
decode tok/s, prefill tok/s, TTFT, peak memory and NPU residency from the per-graph profiles →
`llm_results.json`. Results will be added here when the run completes.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install qai-hub torch torchvision onnx pillow numpy ai-edge-litert
qai-hub configure --api_token <your AI Hub token>
python aihub_mobilenet_int8.py      # ~10 min: quantize + compile + profile
bash run_eval.sh                    # ~4 min: top-1 accuracy FP32 vs INT8
bash run_llm.sh                     # ~1 h: Qwen3-0.6B W4A16 → QNN → profile on Galaxy S25
```
