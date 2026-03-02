"""
Quantisation and performance analysis for HalfUNet 300x480 (grayscale)
Target: rk3566

Usage:
    cd export/hal_300x480_B8_E77_T2026-02-28_06-14-30
    python quantize_and_analyze.py

Output:
    hal_300x480.rknn       - quantised model ready for the device
    snapshot/              - layer-by-layer accuracy analysis results
"""

import numpy as np
import cv2
from rknn.api import RKNN

MODEL_PT   = './hal_300x480_B8_E77_T2026-02-28_06-14-30.pt'
RKNN_OUT   = './hal_300x480.rknn'
DATASET    = './dataset.txt'
INPUT_SIZE = [1, 1, 300, 480]   # B x C x H x W  (grayscale: C=1)

# Take the first image from the dataset as the analysis sample
with open(DATASET) as f:
    SAMPLE_IMG = f.readline().strip()


def load_grayscale(path, h=300, w=480):
    """Load a grayscale image as (1, H, W, 1) uint8 array — NHWC format."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h))
    img = np.expand_dims(img, axis=-1)   # (H, W, 1)  — channel last (HWC)
    return np.expand_dims(img, axis=0)   # (1, H, W, 1) — add batch dim


# ──────────────────────────────────────────────
# 1. CONFIGURATION AND BUILD
# ──────────────────────────────────────────────
rknn = RKNN(verbose=True)

print('\n[1/5] Config')
rknn.config(
    mean_values      = [[0]],       # replicates ToTensor() /255: (pixel - 0) / 255
    std_values       = [[255]],
    target_platform  = 'rk3566',
    quantized_dtype  = 'w8a8',
    optimization_level = 3,
)

print('\n[2/5] Load PyTorch model')
ret = rknn.load_pytorch(model=MODEL_PT, input_size_list=[INPUT_SIZE])
if ret != 0:
    print('ERROR: load_pytorch failed'); exit(ret)

print('\n[3/5] Build (INT8 quantisation)')
ret = rknn.build(do_quantization=True, dataset=DATASET)
if ret != 0:
    print('ERROR: build failed'); exit(ret)

print('\n[4/5] Export .rknn')
ret = rknn.export_rknn(RKNN_OUT)
if ret != 0:
    print('ERROR: export_rknn failed'); exit(ret)
print(f'Model saved: {RKNN_OUT}')


# ──────────────────────────────────────────────
# 2. ANALYSIS WITHOUT DEVICE (simulator)
# ──────────────────────────────────────────────
print('\n[5/5] Analysis in simulator mode (no device required)')
print('NOTE v2.2.0: eval_perf() requires a real device — only accuracy_analysis here.')

ret = rknn.init_runtime(perf_debug=True, eval_mem=True)
if ret != 0:
    print('ERROR: init_runtime failed'); exit(ret)

print('\n' + '='*60)
print('NPU MEMORY ESTIMATE')
print('='*60)
try:
    rknn.eval_memory()
except Exception as e:
    print(f'eval_memory not available in simulator ({e})')
    print('→ run with target=rk3566 (device connected) for the real estimate')

print('\n' + '='*60)
print('ACCURACY ANALYSIS (layer-by-layer vs float32)')
print('Legend: cos->1.0 and euc->0 is good')
print('  entire: cumulative error from the first layer')
print('  single: error of the individual layer (shows where accuracy is lost)')
print('='*60)
ret = rknn.accuracy_analysis(inputs=[SAMPLE_IMG], output_dir='./snapshot')
if ret != 0:
    print('ERROR: accuracy_analysis failed'); exit(ret)
print('Results saved in ./snapshot/')

# Test inference to verify the model produces sensible output
print('\n' + '='*60)
print('TEST INFERENCE (simulator)')
print('='*60)
img = load_grayscale(SAMPLE_IMG)   # → (1, 300, 480, 1) NHWC
outputs = rknn.inference(inputs=[img], data_format='nhwc')
print(f'Output shape: {outputs[0].shape}')
print(f'Output min/max: {outputs[0].min():.4f} / {outputs[0].max():.4f}')

rknn.release()
print('\nDone.')
