"""
Top-1 accuracy on the Imagenette validation split for the exact TFLite artifacts that were
profiled on the Galaxy S24 (FP32 and INT8), plus the original PyTorch FP32 model as a reference.
Runs on the Mac CPU via the TFLite interpreter (same numerics as the compiled artifact).
"""
import io, json, os, sys, tarfile, time, urllib.request
import numpy as np
from PIL import Image
import qai_hub as hub

JOB_INT8 = "jp3zvzyx5"   # compile job: mnv2 int8 -> tflite (--quantize_io)
JOB_FP32 = "jgolklj4g"   # compile job: mnv2 fp32 -> tflite
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = full val split (3925 images)

# Imagenette wnid -> ImageNet-1k class index
WNID_TO_IDX = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
               "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
               "n03445777": 574, "n03888257": 701}
IMAGENETTE_IDX = np.array(sorted(WNID_TO_IDX.values()))

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- TFLite interpreter (litert preferred, tensorflow fallback)
try:
    from ai_edge_litert.interpreter import Interpreter
    RUNTIME = "ai-edge-litert"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = f"tensorflow {tf.__version__}"
log(f"tflite runtime: {RUNTIME}")

# ---- 1. Download compiled models from AI Hub
for jid, fn in ((JOB_INT8, "mnv2_int8.tflite"), (JOB_FP32, "mnv2_fp32.tflite")):
    if not os.path.exists(fn):
        log(f"downloading {fn} from job {jid}")
        hub.get_job(jid).get_target_model().download(fn)
    log(f"{fn}: {os.path.getsize(fn)/1e6:.2f} MB")

# ---- 2. Imagenette val split
if not os.path.isdir("imagenette2-320/val"):
    log("downloading imagenette2-320.tgz (~340 MB)")
    urllib.request.urlretrieve(IMAGENETTE_URL, "imagenette2-320.tgz")
    log("extracting")
    with tarfile.open("imagenette2-320.tgz") as t:
        t.extractall(".")
samples = []
for wnid, idx in WNID_TO_IDX.items():
    d = os.path.join("imagenette2-320/val", wnid)
    for f in sorted(os.listdir(d)):
        samples.append((os.path.join(d, f), idx))
samples.sort()
if LIMIT: samples = samples[:LIMIT]
log(f"{len(samples)} validation images")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
def preprocess(path):
    """Standard ImageNet eval: resize short side to 256, center-crop 224, normalize. Returns CHW float32."""
    img = Image.open(path).convert("RGB")
    w, h = img.size; s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size; l, t = (w - 224) // 2, (h - 224) // 2
    img = img.crop((l, t, l + 224, t + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return (np.transpose(x, (2, 0, 1)) - MEAN) / STD

class TFLiteModel:
    def __init__(self, path):
        self.interp = Interpreter(model_path=path, num_threads=os.cpu_count() or 4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.shape = tuple(self.inp["shape"]); self.dtype = self.inp["dtype"]
        self.scale, self.zp = self.inp["quantization"]
        self.nhwc = self.shape[-1] == 3
        log(f"{path}: input {self.shape} {np.dtype(self.dtype).name} q=({self.scale}, {self.zp}) "
            f"layout={'NHWC' if self.nhwc else 'NCHW'}; output {tuple(self.out['shape'])} {np.dtype(self.out['dtype']).name}")
    def __call__(self, chw):
        x = chw[None]
        if self.nhwc: x = np.transpose(x, (0, 2, 3, 1))
        if self.dtype in (np.int8, np.uint8):
            x = np.round(x / self.scale + self.zp)
            lo, hi = (-128, 127) if self.dtype == np.int8 else (0, 255)
            x = np.clip(x, lo, hi).astype(self.dtype)
        else:
            x = x.astype(np.float32)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out["index"])[0].astype(np.float32)

models = {"fp32_tflite": TFLiteModel("mnv2_fp32.tflite"), "int8_tflite": TFLiteModel("mnv2_int8.tflite")}

# PyTorch reference (the model everything was exported from)
try:
    import torch, torchvision
    tm = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
    @torch.no_grad()
    def torch_fwd(chw): return tm(torch.from_numpy(chw[None])).numpy()[0]
    models["pytorch_fp32_ref"] = torch_fwd
except Exception as e:
    log(f"skipping pytorch reference: {e}")

# ---- 3. Evaluate
preds = {k: [] for k in models}; labels = []; t0 = time.time()
for i, (path, label) in enumerate(samples):
    x = preprocess(path); labels.append(label)
    for k, m in models.items():
        preds[k].append(m(x))
    if (i + 1) % 500 == 0: log(f"{i+1}/{len(samples)} ({time.time()-t0:.0f}s)")
labels = np.array(labels)
logits = {k: np.stack(v) for k, v in preds.items()}

res = {"n_images": int(len(labels)), "runtime": RUNTIME, "split": "imagenette2-320/val",
       "preprocess": "resize256 + centercrop224 + imagenet norm", "models": {}}
top1 = {}
for k, L in logits.items():
    p1000 = L.argmax(1)
    p10 = IMAGENETTE_IDX[L[:, IMAGENETTE_IDX].argmax(1)]
    top5 = np.argsort(-L, 1)[:, :5]
    top1[k] = p1000
    res["models"][k] = {
        "top1_1000class": float((p1000 == labels).mean()),
        "top5_1000class": float((top5 == labels[:, None]).any(1).mean()),
        "top1_10class_restricted": float((p10 == labels).mean()),
    }
if "fp32_tflite" in top1 and "int8_tflite" in top1:
    res["int8_vs_fp32_prediction_agreement"] = float((top1["int8_tflite"] == top1["fp32_tflite"]).mean())
    res["top1_drop_int8_vs_fp32_pct_points"] = 100 * (res["models"]["fp32_tflite"]["top1_1000class"]
                                                     - res["models"]["int8_tflite"]["top1_1000class"])
json.dump(res, open("accuracy_results.json", "w"), indent=1)

print("\n==================== ACCURACY ====================")
print(f"Imagenette val, {len(labels)} images, {RUNTIME}")
print(f"{'model':20s} {'top1(1000)':>11s} {'top5(1000)':>11s} {'top1(10cls)':>12s}")
for k, r in res["models"].items():
    print(f"{k:20s} {100*r['top1_1000class']:10.2f}% {100*r['top5_1000class']:10.2f}% {100*r['top1_10class_restricted']:11.2f}%")
if "int8_vs_fp32_prediction_agreement" in res:
    print(f"\nINT8 vs FP32 top-1 agreement : {100*res['int8_vs_fp32_prediction_agreement']:.2f}%")
    print(f"INT8 top-1 drop vs FP32      : {res['top1_drop_int8_vs_fp32_pct_points']:+.2f} pct points")
print("==================================================")
