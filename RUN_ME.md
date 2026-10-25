# Qualcomm AI Hub: MobileNetV2 -> INT8 -> latency on Galaxy S24

Run these in Terminal on your Mac (about 2 min of setup, then ~10-15 min of hosted-device time).

## 1. Get an API token (one time)
- Sign in / sign up at https://aihub.qualcomm.com (free, no approval wait)
- Avatar (top right) -> Settings -> copy the API Token

## 2. Set up a Python env (3.10 - 3.12)
```bash
mkdir -p ~/projects/qualcomm-aihub && cd ~/projects/qualcomm-aihub
# copy aihub_mobilenet_int8.py into this folder, then:
python3 -m venv .venv && source .venv/bin/activate     # python3.12 -m venv .venv if python3 is 3.9
pip install -U pip
pip install qai-hub torch torchvision onnx pillow numpy
qai-hub configure --api_token PASTE_YOUR_TOKEN_HERE
qai-hub list-devices | grep -i "galaxy s24"           # sanity check: auth works, device exists
```

## 3. Run it
```bash
python aihub_mobilenet_int8.py
```
It prints a job URL for each of the 7 jobs as they're submitted (compile->ONNX, quantize,
compile->TFLite x2, profile x2), waits on each, then prints a SUMMARY block with
FP32 vs INT8 latency in ms and the INT8 speedup. Full profiles land in
profile_int8.json / profile_fp32.json and results.json next to the script.

Optional knobs (env vars):
- AIHUB_DEVICE="Samsung Galaxy S25 (Family)"   # Snapdragon 8 Elite
- AIHUB_DEVICE="Snapdragon X Elite CRD"        # Windows-on-ARM laptop
- N_CALIB=500                                  # more calibration images (default 100)

## What each step is doing
1. torch.export traces MobileNetV2 into an ExportedProgram.
2. Compile job (--target_runtime onnx) turns it into FP32 ONNX on AI Hub's servers.
3. Quantize job does post-training INT8 quantization (weights + activations) using
   100 Imagenette images as calibration data -> INT8 ONNX with QDQ nodes.
4. Compile job (--target_runtime tflite --quantize_io) produces an INT8 TFLite model
   with int8 input/output so the NPU never has to do float<->int conversion.
5. Profile job runs the TFLite model on a real Galaxy S24 in Qualcomm's device farm
   and reports inference time, load time, peak memory, and per-layer compute unit
   (NPU / GPU / CPU). The FP32 model is profiled the same way for comparison.

## Reading the result
- `estimated_inference_time` is in microseconds; the script converts to ms.
- Expect INT8 MobileNetV2 on Snapdragon 8 Gen 3 to land around ~0.5-1 ms on the NPU
  (HTP), with FP32 noticeably slower and some layers possibly falling back to GPU/CPU.
- Each job URL opens a page on aihub.qualcomm.com with the full per-op breakdown --
  that's the artifact to screenshot/link if this is for a checklist or write-up.

## If something fails
- `HTTP 401` on configure -> token pasted wrong.
- Quantize job FAILED -> open the job URL, the log usually names the unsupported op.
- Profile stuck in "queued" -> the device family is busy; try AIHUB_DEVICE="Samsung Galaxy S23 (Family)".
