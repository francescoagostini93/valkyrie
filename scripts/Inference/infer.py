"""
infer.py — Two-pass pupil segmentation inference with HalfUNet.

Pipeline
--------
1. Coarse pass  : resize input (1920×1200) to 480×300 → run model
                  → detect the two pupil blobs in the coarse mask
                  → compute their centres in original-image coordinates

2. Fine pass    : extract two 480×300 crops from the original image,
                  one centred on each blob → run model on each crop
                  → place the fine masks back onto a 1920×1200 canvas

Usage
-----
    python infer.py <image> <checkpoint.pth> [--output mask.png] [--device auto]

    image        : grayscale input image (any resolution; 1920×1200 expected)
    checkpoint   : .pth file produced by train.py
    --output     : output mask path           (default: output_mask.png)
    --device     : cuda | cpu | auto          (default: auto)
    --threshold  : sigmoid threshold [0,1]    (default: 0.5)
    --min-area   : minimum blob area (px) in coarse mask to be considered a pupil
                   (default: 50)
    --save-coarse: also save the coarse mask as <output>_coarse.png
    --display    : show segmentation overlay (saved as <output>_display.png;
                   also opens a window if a display is available)
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Import model from src/ ────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(_SRC))
from new_models import HalfUNet  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_W = 480   # network input width
MODEL_H = 300   # network input height


# ── Model helpers ─────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = HalfUNet(in_channels=1, out_channels=1, features=64, num_levels=5)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def infer(model: torch.nn.Module,
          img: Image.Image,
          device: torch.device,
          threshold: float = 0.5) -> np.ndarray:
    """Run the model on a PIL Image (mode L, any size).
    Returns a binary (H, W) uint8 numpy array."""
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        prob = torch.sigmoid(model(t))
    return (prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)


# ── BFS connected-component labeling ─────────────────────────────────────────

def _bfs_label(binary: np.ndarray):
    """Returns (labeled_array, n_components). Pure numpy/stdlib, no scipy."""
    H, W = binary.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    n = 0
    for sy, sx in zip(*np.where(binary)):
        if labeled[sy, sx]:
            continue
        n += 1
        q = deque([(sy, sx)])
        labeled[sy, sx] = n
        while q:
            y, x = q.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not labeled[ny, nx]:
                    labeled[ny, nx] = n
                    q.append((ny, nx))
    return labeled, n


def find_blob_centers(mask: np.ndarray, min_area: int = 50):
    """
    Find the centroids of the two largest blobs in the binary mask.
    Returns [(cx_left, cy_left), (cx_right, cy_right)] in pixel coordinates.
    Raises ValueError if fewer than 2 valid blobs are found.
    """
    labeled, n = _bfs_label(mask)
    blobs = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area = len(ys)
        if area < min_area:
            continue
        blobs.append((area, float(xs.mean()), float(ys.mean())))

    if len(blobs) < 2:
        raise ValueError(
            f"Expected 2 blobs in coarse mask, found {len(blobs)} "
            f"(min_area={min_area}). "
            "Try lowering --min-area or checking the checkpoint."
        )

    # Keep the two largest, then sort left→right by cx
    blobs.sort(key=lambda b: b[0], reverse=True)
    blobs = blobs[:2]
    blobs.sort(key=lambda b: b[1])
    return [(cx, cy) for _, cx, cy in blobs]


# ── ROI extraction / placement ────────────────────────────────────────────────

def extract_roi(image_np: np.ndarray,
                cx: float, cy: float,
                w: int = MODEL_W,
                h: int = MODEL_H) -> tuple[np.ndarray, int, int]:
    """
    Extract a w×h crop centred on (cx, cy) from image_np.
    Out-of-bounds regions are reflect-padded.
    Returns (crop, x0, y0) where (x0, y0) is the top-left corner
    of the crop window in original-image coordinates (may be negative).
    """
    H, W = image_np.shape

    x0 = int(round(cx - w / 2))
    y0 = int(round(cy - h / 2))
    x1, y1 = x0 + w, y0 + h

    pad_l = max(0, -x0);  pad_r = max(0, x1 - W)
    pad_t = max(0, -y0);  pad_b = max(0, y1 - H)

    crop = image_np[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]

    if pad_l or pad_r or pad_t or pad_b:
        crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r)), mode='reflect')

    return crop, x0, y0


def place_roi(canvas: np.ndarray,
              roi_mask: np.ndarray,
              x0: int, y0: int) -> None:
    """
    OR-paste roi_mask into canvas at position (x0, y0).
    Clips gracefully at image borders.
    """
    H_c, W_c = canvas.shape
    H_r, W_r = roi_mask.shape

    src_x0 = max(0, -x0);       src_y0 = max(0, -y0)
    src_x1 = W_r - max(0, x0 + W_r - W_c)
    src_y1 = H_r - max(0, y0 + H_r - H_c)

    dst_x0 = max(0, x0);        dst_y0 = max(0, y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    canvas[dst_y0:dst_y1, dst_x0:dst_x1] |= roi_mask[src_y0:src_y1, src_x0:src_x1]


# ── Display ───────────────────────────────────────────────────────────────────

def _mask_contour(mask: np.ndarray) -> np.ndarray:
    """Return a boolean array that is True only on the border pixels of the mask.
    A pixel is a border if it is foreground and has at least one background neighbour.
    Pure numpy, no OpenCV/scipy needed."""
    fg = mask.astype(bool)
    # erode with a 3×3 cross: a pixel survives only if all 4 neighbours are also fg
    inner = (
        fg
        & np.roll(fg,  1, axis=0)
        & np.roll(fg, -1, axis=0)
        & np.roll(fg,  1, axis=1)
        & np.roll(fg, -1, axis=1)
    )
    return fg & ~inner   # border = foreground minus interior


def display_result(orig_np: np.ndarray,
                   final_mask: np.ndarray,
                   centers_orig: list,
                   fine_crops: list,          # [(crop_np, fine_mask, label), ...]
                   output_path: str) -> None:
    """
    Build and save a 3-panel figure:
      Left   : full 1920×1200 image with green overlay + contour + pupil centres
      Centre : left-pupil ROI crop with fine mask overlay
      Right  : right-pupil ROI crop with fine mask overlay
    Also attempts plt.show() — works on desktop; silently skipped on headless/Colab.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # ── colour helpers ────────────────────────────────────────────────────────
    OVERLAY_COLOR  = np.array([0.0, 1.0, 0.2])   # bright green
    OVERLAY_ALPHA  = 0.35
    CONTOUR_COLOR  = np.array([0.0, 1.0, 0.0])   # pure green
    CENTER_COLORS  = ['#ff4444', '#4488ff']        # L=red, R=blue

    def _apply_overlay(gray_np, bin_mask):
        """Return an RGB float32 image with the mask blended in."""
        rgb = np.stack([gray_np / 255.0] * 3, axis=-1)
        contour = _mask_contour(bin_mask)
        # semi-transparent fill
        rgb[bin_mask == 1] = (
            rgb[bin_mask == 1] * (1 - OVERLAY_ALPHA)
            + OVERLAY_COLOR * OVERLAY_ALPHA
        )
        # solid contour
        rgb[contour] = CONTOUR_COLOR
        return rgb

    # ── figure layout — full image top, two crops bottom ─────────────────────
    fig = plt.figure(figsize=(20, 11), facecolor='#1a1a1a')
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.01, right=0.99, top=0.96, bottom=0.02,
        height_ratios=[1.6, 1],
    )

    # Row 0 — full image spanning both columns
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(_apply_overlay(orig_np, final_mask))
    ax0.set_title('Full image — segmentation overlay', color='white', fontsize=12)
    ax0.axis('off')

    # Row 1 — per-pupil fine crops side by side
    for panel_idx, (crop_np, fine_mask, label) in enumerate(fine_crops):
        ax = fig.add_subplot(gs[1, panel_idx])
        ax.imshow(_apply_overlay(crop_np, fine_mask))
        ax.set_title(f'{label.capitalize()} pupil — fine segmentation',
                     color='white', fontsize=11)
        ax.axis('off')
        ax.text(0.02, 0.97, f'{fine_mask.sum()} px',
                transform=ax.transAxes, color='white', fontsize=9,
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.5, pad=3, edgecolor='none'))

    # ── save & show ───────────────────────────────────────────────────────────
    p = Path(output_path)
    disp_path = p.with_name(p.stem + '_display.png')
    fig.savefig(disp_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Display    : {disp_path}")

    try:
        plt.show()
    except Exception:
        pass   # headless / Colab — image already saved
    plt.close(fig)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(image_path: str,
        checkpoint: str,
        output_path: str,
        device_name: str = 'auto',
        threshold: float = 0.5,
        min_area: int = 50,
        save_coarse: bool = False,
        display: bool = False) -> np.ndarray:

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'auto' else device_name
    )
    print(f"Device     : {device}")

    model = load_model(checkpoint, device)
    print(f"Checkpoint : {checkpoint}")

    # Load original image, keep it as numpy for ROI extraction
    orig_pil = Image.open(image_path).convert('L')
    W_img, H_img = orig_pil.size          # PIL: (width, height)
    orig_np = np.array(orig_pil, dtype=np.uint8)
    print(f"Image      : {image_path}  ({W_img}×{H_img})")

    # ── Pass 1: coarse segmentation ───────────────────────────────────────────
    coarse_pil  = orig_pil.resize((MODEL_W, MODEL_H), Image.BILINEAR)
    coarse_mask = infer(model, coarse_pil, device, threshold)   # (MODEL_H, MODEL_W)
    print(f"Coarse mask: {coarse_mask.sum()} active pixels")

    if save_coarse:
        p = Path(output_path)
        coarse_out = p.with_name(p.stem + '_coarse' + p.suffix)
        Image.fromarray(coarse_mask * 255, mode='L').save(coarse_out)
        print(f"Coarse mask saved: {coarse_out}")

    # Find blob centres in coarse space, scale to original resolution
    centers_coarse = find_blob_centers(coarse_mask, min_area)
    scale_x = W_img / MODEL_W
    scale_y = H_img / MODEL_H
    centers_orig = [(cx * scale_x, cy * scale_y) for cx, cy in centers_coarse]
    print(f"Pupil centres (original px): "
          f"L=({round(centers_orig[0][0])}, {round(centers_orig[0][1])})  "
          f"R=({round(centers_orig[1][0])}, {round(centers_orig[1][1])})")

    # ── Pass 2: fine segmentation on full-res ROI crops ───────────────────────
    final_mask = np.zeros((H_img, W_img), dtype=np.uint8)
    fine_crops = []   # [(crop_np, fine_mask, label), ...] kept for display

    for label, (cx, cy) in zip(('left', 'right'), centers_orig):
        crop_np, x0, y0 = extract_roi(orig_np, cx, cy, MODEL_W, MODEL_H)
        crop_pil  = Image.fromarray(crop_np, mode='L')
        fine_mask = infer(model, crop_pil, device, threshold)   # (MODEL_H, MODEL_W)
        place_roi(final_mask, fine_mask, x0, y0)
        fine_crops.append((crop_np, fine_mask, label))
        print(f"  [{label}] ROI top-left=({x0}, {y0})  "
              f"active pixels={fine_mask.sum()}")

    # ── Save mask output ──────────────────────────────────────────────────────
    Image.fromarray(final_mask * 255, mode='L').save(output_path)
    print(f"Output     : {output_path}")

    # ── Display ───────────────────────────────────────────────────────────────
    if display:
        display_result(orig_np, final_mask, centers_orig, fine_crops, output_path)

    return final_mask


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Two-pass pupil segmentation: coarse (480×300) + fine ROI crops.'
    )
    parser.add_argument('image',
                        help='Input image (grayscale, expected 1920×1200)')
    parser.add_argument('checkpoint',
                        help='HalfUNet checkpoint (.pth) produced by train.py')
    parser.add_argument('--output',      default='output_mask.png',
                        help='Output mask path (default: output_mask.png)')
    parser.add_argument('--device',      default='auto',
                        help='cuda | cpu | auto  (default: auto)')
    parser.add_argument('--threshold',   type=float, default=0.5,
                        help='Sigmoid threshold [0,1]  (default: 0.5)')
    parser.add_argument('--min-area',    type=int,   default=50,
                        help='Min blob area (px) in coarse mask  (default: 50)')
    parser.add_argument('--save-coarse', action='store_true',
                        help='Also save the coarse mask as <output>_coarse.png')
    parser.add_argument('--display',     action='store_true',
                        help='Show segmentation overlay (saved as <output>_display.png; '
                             'also opens a window if a display is available)')
    args = parser.parse_args()

    run(
        image_path  = args.image,
        checkpoint  = args.checkpoint,
        output_path = args.output,
        device_name = args.device,
        threshold   = args.threshold,
        min_area    = args.min_area,
        save_coarse = args.save_coarse,
        display     = args.display,
    )
