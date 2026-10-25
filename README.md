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

## 2. Qwen3-0.6B W4A16 LLM on Snapdragon 8 Elite via QNN

**Target:** Samsung Galaxy S25 (Snapdragon 8 Elite for Galaxy, Hexagon NPU), QNN/QAIRT context binaries
(`geniex_qairt` runtime). **Pipeline** (`run_llm.sh`, 77 min end-to-end on a MacBook Air + AI Hub):
`qai-hub-models export qwen3_0_6b --checkpoint DEFAULT_W4A16 --runtime geniex_qairt --context-lengths 1024,4096`
— Qualcomm's pre-quantized checkpoint (int4 per-channel weights, int16 activations, int8 KV cache,
int8 LM head; SpinQuant R2+R3 → AdaScale → calibration recipe), split into 2 parts (embedding / 28 layers +
LM head) × {prompt processor AR=128, token generator AR=1} × {1024, 4096 context} = 8 graphs, compiled and
linked into QNN context binaries, each graph profiled on the device. `llm_metrics.py` turns the per-graph
profiles into end-to-end figures → `llm_results.json`.

| Context length | 1024 | 4096 |
|---|---|---|
| Decode step (sum of parts) | 10.30 ms | 24.81 ms |
| **Decode throughput** | **97.1 tok/s** | **40.3 tok/s** |
| Prefill (128-token chunk) | 18.6 ms → 6,890 tok/s | 40.3 ms → 3,180 tok/s |
| Time-to-first-token (prompt ≤ 128 tok) | 29 ms | 65 ms |
| Peak memory, decode (sum of parts) | 319 MB | 659 MB |
| Peak memory, largest graph | 236 MB | 555 MB |
| Ops on NPU / GPU / CPU | 27,627 / 0 / 0 (**100% NPU**) | 27,627 / 0 / 0 (**100% NPU**) |
| Context-binary size on disk | 714 MB (both parts, shared) | |

Reference: Qualcomm's published `geniex_qairt` numbers for the same model on Galaxy S25 are 79.9 tok/s
(cl 1024) and 48.7 tok/s (cl 4096). Ours are derived from on-device NPU graph latency via profile jobs,
so they exclude runtime overhead (sampling, tokenizer, IPC) and assume a full-length KV cache; expect
end-to-end Genie/GenieX throughput to land a bit lower at short context and similar at long context.
The 2.4× decode slowdown from 1024 → 4096 context is the KV-cache attention cost at this model size.

AI Hub jobs (Galaxy S25): token generator cl1024 [part 1](https://workbench.aihub.qualcomm.com/jobs/j5wz2v465/) ·
[part 2](https://workbench.aihub.qualcomm.com/jobs/jp44ko3vp/); token generator cl4096
[part 1](https://workbench.aihub.qualcomm.com/jobs/jg92j1dlp/) · [part 2](https://workbench.aihub.qualcomm.com/jobs/jpxrnjx1p/);
prompt processor cl1024 [part 1](https://workbench.aihub.qualcomm.com/jobs/jgjyzmwxg/) ·
[part 2](https://workbench.aihub.qualcomm.com/jobs/jgd4e92eg/); prompt processor cl4096
[part 1](https://workbench.aihub.qualcomm.com/jobs/jgzyo94kp/) · [part 2](https://workbench.aihub.qualcomm.com/jobs/j57n0w9lp/);
link jobs [part 1](https://workbench.aihub.qualcomm.com/jobs/jp11y072p/) · [part 2](https://workbench.aihub.qualcomm.com/jobs/jgd4ew8eg/).

Raw outputs: `llm_results.json`, `llm_export.log`. The exported Genie bundle (`llm_export/…/part{1,2}_of_2.bin`,
`genie_config.json`, tokenizer) is not committed (714 MB) but is what `genie-t2t-run` / GenieX load on a phone.

### Why not Llama 3.2 1B, and why not the Galaxy S24?
Qualcomm's LLM recipes in `qai-hub-models` 0.65 target Snapdragon 8 Elite and newer only (Hexagon v79+),
so the 8 Gen 3 used in experiment 1 is not a supported LLM target. Llama 3.2 1B is also available but is
license-gated on Hugging Face and its export recipe wants ~50 GB host RAM (3 splits); Qwen3-0.6B needs 8 GB
and is Apache-2.0, so it is the one that runs on a laptop.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install qai-hub torch torchvision onnx pillow numpy ai-edge-litert
qai-hub configure --api_token <your AI Hub token>
python aihub_mobilenet_int8.py      # ~10 min: quantize + compile + profile
bash run_eval.sh                    # ~4 min: top-1 accuracy FP32 vs INT8
bash run_llm.sh                     # ~75 min: Qwen3-0.6B W4A16 → QNN → profile on Galaxy S25
```
