"""
test_multipass.py — Multi-pass inference on HalfUNet (RKNN)

Usage:
    cd export/hal_300x480_B8_E77_T2026-02-28_06-14-30
    python test_multipass.py <image.jpg>              # display result
    python test_multipass.py <image.jpg> --save out.jpg

Pipeline:
  1. Load original image (ideally 1920x1200, grayscale)
  2. Pass 1  – rescale to 480x300, global inference
  3. Find the 2 largest blobs in the output mask
  4. Pass 2/3 – extract 2 ROIs from the original image centred on the blobs,
                resize to 480x300, detail inference
  5. Project all 3 masks onto the original image and display
"""

import sys
import argparse
import numpy as np
import cv2

# ── Runtime: use RKNNLite on device, RKNN simulator on PC ────────────────────
try:
    from rknnlite.api import RKNNLite as _Backend
    _USE_LITE = True
except ImportError:
    from rknn.api import RKNN as _Backend
    _USE_LITE = False

# ── Model constants ───────────────────────────────────────────────────────────
RKNN_MODEL = 'hal_300x480.rknn'
MODEL_H, MODEL_W = 300, 480      # HxW expected by the model (grayscale)

# BGR colours: pass 1 global, pass 2 ROI-1, pass 3 ROI-2
PASS_COLORS = [
    (0,   220, 220),   # light cyan    – global
    (0,   140, 255),   # orange        – blob 1
    (200,  50, 255),   # violet-magenta – blob 2
]

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f'[ERROR] Cannot load image: {path}')
    return img


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """Resize to MODEL_HxMODEL_W and return (1, H, W, 1) uint8 NHWC."""
    resized = cv2.resize(gray, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    return resized[np.newaxis, ..., np.newaxis]   # (1, H, W, 1)


def run_inference(rknn, gray: np.ndarray) -> np.ndarray:
    """
    Run one inference pass.
    Returns the raw float mask with shape (MODEL_H, MODEL_W).

    RKNN output format handling:
      NCHW (C,H,W) → mask[0]       if shape[0] is the channel dim (small)
      NHWC (H,W,C) → mask[..., 0]  if shape[-1] is the channel dim (small)
    """
    inp     = to_model_input(gray)
    outputs = rknn.inference(inputs=[inp], data_format='nhwc')
    mask    = outputs[0][0]      # remove batch dim → (C,H,W) or (H,W,C) or (H,W)

    if mask.ndim == 3:
        if mask.shape[0] == 1:          # NCHW: (1, H, W)
            mask = mask[0]
        elif mask.shape[-1] == 1:       # NHWC: (H, W, 1)
            mask = mask[..., 0]
        else:
            mask = mask[0]              # fallback: take first plane

    print(f'         [dbg] raw output: {outputs[0].shape} → mask: {mask.shape} '
          f'min={mask.min():.3f} max={mask.max():.3f}')
    return mask                         # float (H, W), logit or [0,1]


def binarize(mask: np.ndarray, threshold: float) -> np.ndarray:
    """Float → uint8 {0,255}: apply sigmoid first if values look like logits."""
    if mask.max() > 1.0 or mask.min() < 0.0:
        mask = 1.0 / (1.0 + np.exp(-mask.astype(np.float64)))
    return (mask >= threshold).astype(np.uint8) * 255


def find_top_blobs(bin_mask: np.ndarray, n: int, min_area: int) -> list:
    """
    Connected-components analysis: return the top-n blobs by area.
    Each blob: {'area', 'bbox':(x,y,w,h), 'centroid':(cx,cy)}.
    """
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask, connectivity=8
    )
    blobs = []
    for i in range(1, n_labels):          # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        blobs.append({
            'area':     area,
            'bbox':     (int(stats[i, cv2.CC_STAT_LEFT]),
                         int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]),
                         int(stats[i, cv2.CC_STAT_HEIGHT])),
            'centroid': (float(centroids[i][0]), float(centroids[i][1])),
        })
    blobs.sort(key=lambda b: b['area'], reverse=True)
    return blobs[:n]


def blob_to_roi(blob: dict, img_h: int, img_w: int, margin: float) -> tuple:
    """
    Project blob coordinates (480x300 space) onto the original image and
    compute an ROI rectangle centred on the blob centroid, with:
      - size proportional to blob bbox × margin
      - aspect ratio forced to MODEL_W : MODEL_H (480:300)
      - minimum size MODEL_W x MODEL_H
      - clamped to image boundaries

    Returns (x1, y1, x2, y2) in original image coordinates.
    """
    sx = img_w / MODEL_W    # scale factor X (4.0 if 1920→480)
    sy = img_h / MODEL_H    # scale factor Y (4.0 if 1200→300)

    cx_o = blob['centroid'][0] * sx
    cy_o = blob['centroid'][1] * sy

    bw, bh = blob['bbox'][2], blob['bbox'][3]
    w_o = max(int(bw * sx * margin), MODEL_W)
    h_o = max(int(bh * sy * margin), MODEL_H)

    # Force aspect ratio MODEL_W:MODEL_H (e.g. 16:10)
    if w_o * MODEL_H > h_o * MODEL_W:
        h_o = round(w_o * MODEL_H / MODEL_W)
    else:
        w_o = round(h_o * MODEL_W / MODEL_H)

    # Centre on blob centroid
    x1 = int(round(cx_o - w_o / 2))
    y1 = int(round(cy_o - h_o / 2))
    x2 = x1 + w_o
    y2 = y1 + h_o

    # Clamp: shift window if it overflows, then truncate if larger than image
    if x1 < 0:
        x2 -= x1
        x1  = 0
    if y1 < 0:
        y2 -= y1
        y1  = 0
    if x2 > img_w:
        x1 = max(0, x1 - (x2 - img_w))
        x2 = img_w
    if y2 > img_h:
        y1 = max(0, y1 - (y2 - img_h))
        y2 = img_h

    return (x1, y1, x2, y2)


def overlay_mask(canvas: np.ndarray, bin_mask: np.ndarray,
                 roi_rect: tuple, color: tuple,
                 alpha: float = 0.45, border: int = 0) -> None:
    """
    Blend bin_mask (MODEL_HxMODEL_W, uint8) into roi_rect on the BGR canvas
    with transparency alpha. Draw the rectangle border if border > 0.
    """
    x1, y1, x2, y2 = roi_rect
    rw, rh = x2 - x1, y2 - y1
    if rw <= 0 or rh <= 0:
        return
    mask_rs  = cv2.resize(bin_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    roi      = canvas[y1:y2, x1:x2]
    overlay  = roi.copy()
    overlay[mask_rs > 0] = color
    canvas[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0)
    if border > 0:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, border)


def draw_label(canvas: np.ndarray, text: str,
               roi_rect: tuple, color: tuple) -> None:
    x1, y1 = roi_rect[0], roi_rect[1]
    y_text  = max(y1 - 8, 24)
    # Dark background for readability
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.rectangle(canvas, (x1, y_text - th - 4), (x1 + tw + 6, y_text + 4),
                  (0, 0, 0), -1)
    cv2.putText(canvas, text, (x1 + 3, y_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def draw_legend(canvas: np.ndarray, items: list) -> None:
    """items: list of (color, label_str)"""
    for i, (color, label) in enumerate(items):
        y = 30 + i * 38
        cv2.rectangle(canvas, (12, y - 18), (40, y + 8), color, -1)
        cv2.putText(canvas, label, (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


# ── Runtime init ──────────────────────────────────────────────────────────────

def init_rknn(rknn_path: str, target: str = None, device_id: str = None,
              pt_path: str = None):
    """
    Initialise the RKNN runtime in one of three modes:

    1. RKNNLite  – on device (rknn_toolkit_lite2 available)
    2. RKNN + target – on RK device via ADB (--target rk3566)
    3. RKNN simulator – build from .pt model (--sim / --pt-model)
       Required because Toolkit2 >= 2.x does not support simulator with load_rknn.
    """
    rknn = _Backend(verbose=False)

    if _USE_LITE:
        print('[INFO] Using RKNNLite (on-device)')
        ret = rknn.load_rknn(rknn_path)
        if ret != 0: sys.exit(f'[ERROR] load_rknn failed: {ret}')
        ret = rknn.init_runtime()

    elif target:
        print(f'[INFO] RKNN Toolkit — target device: {target}'
              + (f'  id={device_id}' if device_id else ''))
        ret = rknn.load_rknn(rknn_path)
        if ret != 0: sys.exit(f'[ERROR] load_rknn failed: {ret}')
        kwargs = {'target': target}
        if device_id:
            kwargs['device_id'] = device_id
        ret = rknn.init_runtime(**kwargs)

    elif pt_path:
        import os
        if not os.path.isfile(pt_path):
            sys.exit(f'[ERROR] .pt file not found: {pt_path}')
        print(f'[INFO] Simulator — building from: {pt_path}  (float, no quant)')
        rknn.config(
            mean_values     = [[0]],
            std_values      = [[255]],
            target_platform = 'rk3566',
        )
        ret = rknn.load_pytorch(model=pt_path,
                                input_size_list=[[1, 1, MODEL_H, MODEL_W]])
        if ret != 0: sys.exit(f'[ERROR] load_pytorch failed: {ret}')
        ret = rknn.build(do_quantization=False)
        if ret != 0: sys.exit(f'[ERROR] build failed: {ret}')
        ret = rknn.init_runtime()

    else:
        sys.exit(
            '[ERROR] No runtime mode selected.\n'
            '  • Device via ADB  →  add  --target rk3566\n'
            '  • PC simulator    →  add  --sim\n'
            '                       (requires the .pt file in the same folder)'
        )

    if ret != 0:
        sys.exit(f'[ERROR] init_runtime failed: {ret}')
    return rknn


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='HalfUNet multi-pass inference on a 1920x1200 image')
    p.add_argument('image',
                   help='Input image (1920x1200 grayscale recommended)')
    p.add_argument('--model',     default=RKNN_MODEL,
                   help=f'.rknn model file (default: {RKNN_MODEL})')
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Mask binarisation threshold (default: 0.5)')
    p.add_argument('--margin',    type=float, default=1.5,
                   help='Blob bbox expansion factor for ROI (default: 1.5)')
    p.add_argument('--min-area',  type=int,   default=50,
                   help='Minimum blob area in px (480x300 space, default: 50)')
    p.add_argument('--display-scale', type=float, default=0.6,
                   help='Scale factor for display window (default: 0.6 → ~1152x720)')
    p.add_argument('--target', metavar='PLATFORM',
                   help='NPU target via ADB, e.g. rk3566 (device must be connected)')
    p.add_argument('--device-id', metavar='ID',
                   help='ADB device ID when multiple devices are connected')
    p.add_argument('--sim', action='store_true',
                   help='Use RKNN simulator (builds from .pt, no device required)')
    p.add_argument('--pt-model', metavar='FILE',
                   help='Explicit path to .pt file for --sim (auto-detected if omitted)')
    p.add_argument('--save', metavar='FILE',
                   help='Save result to FILE instead of displaying it')
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve .pt path for simulator mode
    pt_path = None
    if args.sim:
        if args.pt_model:
            pt_path = args.pt_model
        else:
            import glob, os
            model_dir  = os.path.dirname(os.path.abspath(args.model))
            candidates = glob.glob(os.path.join(model_dir, '*.pt'))
            if not candidates:
                sys.exit(f'[ERROR] No .pt file found in {model_dir}\n'
                         '        Specify the path with --pt-model')
            pt_path = candidates[0]
            print(f'[INFO] Auto-detected .pt: {pt_path}')

    rknn = init_rknn(args.model, target=args.target,
                     device_id=args.device_id, pt_path=pt_path)

    # ── Load original image ──────────────────────────────────────────────────
    orig_gray = load_grayscale(args.image)
    img_h, img_w = orig_gray.shape
    print(f'[INFO] Image: {img_w}×{img_h} px (WxH), grayscale')

    # BGR canvas for final visualisation
    canvas = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)

    # ── PASS 1: global inference on downscaled image ─────────────────────────
    print('\n[PASS 1] Global inference ...')
    mask1_f = run_inference(rknn, orig_gray)
    mask1_b = binarize(mask1_f, args.threshold)
    print(f'         Output shape: {mask1_f.shape}  '
          f'min={mask1_f.min():.3f}  max={mask1_f.max():.3f}')
    print(f'         Active pixels: {(mask1_b > 0).sum()} / {mask1_b.size}')

    # Overlay global mask with low opacity (background layer)
    overlay_mask(canvas, mask1_b, (0, 0, img_w, img_h),
                 PASS_COLORS[0], alpha=0.25, border=0)

    # Find the 2 largest blobs
    blobs = find_top_blobs(mask1_b, n=2, min_area=args.min_area)
    print(f'         Blobs found: {len(blobs)} '
          f'(min_area={args.min_area}px)')

    if not blobs:
        print('[WARN] No blobs found in pass 1. Check threshold or input image.')

    # ── PASS 2 / 3: ROIs centred on blobs ────────────────────────────────────
    legend_items = [(PASS_COLORS[0], 'Pass 1 - global (480x300)')]

    for idx, blob in enumerate(blobs):
        pass_num = idx + 2
        color    = PASS_COLORS[idx + 1]

        print(f'\n[PASS {pass_num}] Blob #{idx + 1}: '
              f'area={blob["area"]} px  '
              f'centroid=({blob["centroid"][0]:.1f}, {blob["centroid"][1]:.1f})')

        roi_rect = blob_to_roi(blob, img_h, img_w, args.margin)
        x1, y1, x2, y2 = roi_rect
        print(f'         ROI in original: ({x1},{y1})->({x2},{y2}) '
              f'= {x2 - x1}x{y2 - y1} px')

        # Extract crop and infer (to_model_input resizes to 480x300 internally)
        crop_gray = orig_gray[y1:y2, x1:x2]
        mask_f    = run_inference(rknn, crop_gray)
        mask_b    = binarize(mask_f, args.threshold)
        print(f'         Active pixels (ROI): {(mask_b > 0).sum()} / {mask_b.size}')

        overlay_mask(canvas, mask_b, roi_rect, color, alpha=0.45, border=3)
        draw_label(canvas,
                   f'Pass {pass_num} | blob area={blob["area"]}px',
                   roi_rect, color)

        legend_items.append(
            (color, f'Pass {pass_num} - ROI blob {idx + 1} '
                    f'({x2-x1}x{y2-y1}->480x300)'))

    # ── Legend ────────────────────────────────────────────────────────────────
    draw_legend(canvas, legend_items)

    rknn.release()

    # ── Output ────────────────────────────────────────────────────────────────
    if args.save:
        cv2.imwrite(args.save, canvas)
        print(f'\n[INFO] Result saved to: {args.save}')
    else:
        scale = args.display_scale
        disp  = cv2.resize(canvas, (int(img_w * scale), int(img_h * scale)))
        title = 'HalfUNet Multi-Pass Inference — press any key to close'
        cv2.imshow(title, disp)
        print('\n[INFO] Press any key in the window to close ...')
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
