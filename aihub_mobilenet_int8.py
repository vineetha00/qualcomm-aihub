"""
Qualcomm AI Hub end-to-end: MobileNetV2 -> ONNX -> INT8 quantize -> TFLite -> profile on Galaxy S24.

Also profiles the FP32 version so we can compare latency.
Writes results to results.json.
"""
import io
import json
import os
import sys
import time
import zipfile
import urllib.request

import numpy as np
import torch
import torchvision
from PIL import Image
import qai_hub as hub

DEVICE_NAME = os.environ.get("AIHUB_DEVICE", "Samsung Galaxy S24 (Family)")
INPUT_SHAPE = (1, 3, 224, 224)
N_CALIB = int(os.environ.get("N_CALIB", "100"))
CALIB_URL = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenette_samples.zip"
SCRATCH = os.environ.get("SCRATCH", ".")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def wait(job, label):
    log(f"{label}: submitted -> {job.url}")
    status = job.wait()
    log(f"{label}: {status}")
    if not status.success:
        try:
            print(job.download_job_logs(os.path.join(SCRATCH, f"{label}_logs.txt")))
        except Exception as e:
            print("could not fetch logs:", e)
        sys.exit(f"{label} failed")
    return job

results = {"device": DEVICE_NAME, "model": "torchvision mobilenet_v2 IMAGENET1K_V1"}
device = hub.Device(DEVICE_NAME)

# ---- 1. Export PyTorch model
log("exporting MobileNetV2 with torch.export")
torch_model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
with torch.no_grad():
    pt2_model = torch.export.export(torch_model, (torch.rand(INPUT_SHAPE),))

# ---- 2. Compile to ONNX (FP32) on AI Hub
onnx_job = wait(hub.submit_compile_job(
    model=pt2_model, device=device,
    input_specs=dict(image_tensor=INPUT_SHAPE),
    options="--target_runtime onnx",
    name="mnv2 fp32 -> onnx",
), "compile_onnx_fp32")
fp32_onnx = onnx_job.get_target_model()
results["job_compile_onnx_fp32"] = onnx_job.url

# ---- 3. Calibration data (real images if downloadable, else random)
calib = []
try:
    log("downloading imagenette calibration samples")
    zbytes = urllib.request.urlopen(CALIB_URL, timeout=60).read()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        for n in names[:N_CALIB]:
            img = Image.open(io.BytesIO(zf.read(n))).convert("RGB").resize(INPUT_SHAPE[2:])
            x = np.array(img).astype(np.float32) / 255.0
            x = np.expand_dims(np.transpose(x, (2, 0, 1)), 0)
            calib.append(((x - mean) / std).astype(np.float32))
    results["calibration"] = f"{len(calib)} imagenette images"
except Exception as e:
    log(f"calibration download failed ({e}); using random data")
    calib = [np.random.randn(*INPUT_SHAPE).astype(np.float32) for _ in range(N_CALIB)]
    results["calibration"] = f"{len(calib)} random tensors"
log(f"calibration set: {results['calibration']}")

# ---- 4. Quantize INT8 weights + INT8 activations
quant_job = wait(hub.submit_quantize_job(
    model=fp32_onnx,
    calibration_data=dict(image_tensor=calib),
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
    name="mnv2 int8 quantize",
), "quantize_int8")
int8_onnx = quant_job.get_target_model()
results["job_quantize"] = quant_job.url

# ---- 5. Compile INT8 -> TFLite, and FP32 -> TFLite for comparison
int8_compile = hub.submit_compile_job(
    model=int8_onnx, device=device,
    options="--target_runtime tflite --quantize_io",
    name="mnv2 int8 -> tflite",
)
fp32_compile = hub.submit_compile_job(
    model=fp32_onnx, device=device,
    options="--target_runtime tflite",
    name="mnv2 fp32 -> tflite",
)
wait(int8_compile, "compile_tflite_int8")
wait(fp32_compile, "compile_tflite_fp32")
results["job_compile_tflite_int8"] = int8_compile.url
results["job_compile_tflite_fp32"] = fp32_compile.url

# ---- 6. Profile both on the device
int8_prof = hub.submit_profile_job(model=int8_compile.get_target_model(), device=device, name="mnv2 int8 profile")
fp32_prof = hub.submit_profile_job(model=fp32_compile.get_target_model(), device=device, name="mnv2 fp32 profile")
wait(int8_prof, "profile_int8")
wait(fp32_prof, "profile_fp32")

def summarize(job, tag):
    prof = job.download_profile()
    with open(os.path.join(SCRATCH, f"profile_{tag}.json"), "w") as f:
        json.dump(prof, f, indent=1, default=str)
    es = prof.get("execution_summary", {})
    out = {"job": job.url, "device_actual": str(job.device)}
    for k, v in es.items():
        if isinstance(v, (int, float, str)):
            out[k] = v
    # compute-unit breakdown from per-layer details
    units = {}
    for layer in prof.get("execution_detail", []):
        cu = layer.get("compute_unit", "?")
        units[cu] = units.get(cu, 0) + 1
    out["layers_by_compute_unit"] = units
    return out

results["int8"] = summarize(int8_prof, "int8")
results["fp32"] = summarize(fp32_prof, "fp32")

with open(os.path.join(SCRATCH, "results.json"), "w") as f:
    json.dump(results, f, indent=1)
print(json.dumps(results, indent=1))

# ---- 7. Human-readable summary (AI Hub reports times in microseconds)
def ms(v):
    return f"{v/1000:.2f} ms" if isinstance(v, (int, float)) else str(v)

print("\n==================== SUMMARY ====================")
print(f"Device : {DEVICE_NAME}")
print(f"Model  : MobileNetV2 (torchvision, 1x3x224x224)")
for tag in ("fp32", "int8"):
    r = results[tag]
    print(f"\n{tag.upper()} TFLite  ->  {r['job']}")
    for key in ("estimated_inference_time", "first_load_time", "warm_load_time",
                "estimated_inference_peak_memory"):
        if key in r:
            val = r[key]
            print(f"  {key:32s} {ms(val) if 'time' in key else f'{val/1e6:.1f} MB'}")
    print(f"  {'layers by compute unit':32s} {r['layers_by_compute_unit']}")
f, i = results["fp32"].get("estimated_inference_time"), results["int8"].get("estimated_inference_time")
if isinstance(f, (int, float)) and isinstance(i, (int, float)) and i > 0:
    print(f"\nINT8 speedup vs FP32: {f/i:.2f}x")
print("=================================================")
