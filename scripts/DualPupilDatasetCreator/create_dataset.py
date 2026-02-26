"""
create_dataset.py — Build a DualPupilNet training dataset from a high-resolution
segmentation dataset.

For each image-mask pair the script produces three output pairs:
  {name}.png     full image scaled to output resolution
  {name}_l.png   left-pupil  crop at output resolution (extracted at original resolution)
  {name}_r.png   right-pupil crop at output resolution (extracted at original resolution)

Left/right is determined by the x-coordinate of each pupil's centroid in the frame
(lower x = left in the frame).

Usage:
  python create_dataset.py \\
      --input_res  1920 1200 \\
      --output_res  480  300 \\
      --input_path  data/dataset_1920_1200 \\
      --output_path data/dataset_dual_480_300
"""

import argparse
import sys
from collections import deque
import numpy as np
from pathlib import Path
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _label(binary: np.ndarray):
    """
    Connected-component labeling for a 2D binary array (4-connectivity).
    Pure numpy/stdlib implementation — no scipy required.

    Returns (labeled, n_labels) with the same interface as scipy.ndimage.label.
    Only nonzero pixels are visited, so performance scales with mask density
    rather than full image size.
    """
    H, W = binary.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    n = 0
    for start_y, start_x in zip(*np.where(binary)):
        if labeled[start_y, start_x]:
            continue
        n += 1
        queue = deque([(start_y, start_x)])
        labeled[start_y, start_x] = n
        while queue:
            y, x = queue.popleft()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and binary[ny, nx] and not labeled[ny, nx]:
                    labeled[ny, nx] = n
                    queue.append((ny, nx))
    return labeled, n


def find_pupil_centers(mask_np: np.ndarray, min_area: int = 50):
    """
    Find the centroids of the two largest connected components in mask_np.

    Components smaller than min_area pixels are treated as noise and ignored.
    The two largest surviving components are returned sorted left→right
    by their x centroid.

    Returns: [(cx_l, cy_l), (cx_r, cy_r)]
    Raises:  ValueError if fewer than 2 valid components are found.
    """
    binary = (mask_np > 0).astype(np.uint8)
    labeled, n = _label(binary)

    if n == 0:
        raise ValueError("No segmented objects found in mask.")

    # Collect (cx, cy, area) for each component above the minimum area threshold
    components = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)  # rows = (y, x)
        area = len(coords)
        if area >= min_area:
            cy = int(np.round(coords[:, 0].mean()))
            cx = int(np.round(coords[:, 1].mean()))
            components.append((cx, cy, area))

    if len(components) < 2:
        raise ValueError(
            f"Only {len(components)} component(s) with area >= {min_area} px found "
            f"(need exactly 2)."
        )

    # Keep the 2 largest components (handles spurious tiny blobs)
    components.sort(key=lambda c: c[2], reverse=True)
    components = components[:2]

    # Sort left → right by x centroid
    components.sort(key=lambda c: c[0])

    return [(c[0], c[1]) for c in components]


def extract_crop(
    img_np: np.ndarray,
    cx: int,
    cy: int,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Extract a w×h crop centred on pixel (cx, cy) from img_np.

    If the crop window extends beyond the image border, the missing area is
    filled using reflect padding.

    img_np: (H, W) or (H, W, C)
    Returns: array of shape (h, w) or (h, w, C)
    """
    H, W = img_np.shape[:2]
    half_w = w // 2
    half_h = h // 2

    x0 = cx - half_w
    x1 = x0 + w
    y0 = cy - half_h
    y1 = y0 + h

    pad_l = max(0, -x0)
    pad_r = max(0, x1 - W)
    pad_t = max(0, -y0)
    pad_b = max(0, y1 - H)

    if pad_l or pad_r or pad_t or pad_b:
        pad_width = (
            ((pad_t, pad_b), (pad_l, pad_r))
            if img_np.ndim == 2
            else ((pad_t, pad_b), (pad_l, pad_r), (0, 0))
        )
        img_np = np.pad(img_np, pad_width, mode='reflect')
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t

    return img_np[y0:y1, x0:x1]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a DualPupilNet training dataset from a high-resolution "
            "segmentation dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_res", nargs=2, type=int, metavar=("W", "H"),
        help=(
            "Expected input resolution (width height). "
            "Images that do not match are skipped with a warning. "
            "Omit to skip the resolution check."
        ),
    )
    parser.add_argument(
        "--output_res", nargs=2, type=int, metavar=("W", "H"), required=True,
        help=(
            "Output resolution (width height). "
            "The full image is scaled to this size. "
            "Pupil crops are extracted at this pixel size from the original image "
            "(no scaling is applied to the crops)."
        ),
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to the input dataset folder (must contain images/ and masks/ subfolders).",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to the output dataset folder (created if it does not exist).",
    )
    parser.add_argument(
        "--min_area", type=int, default=50,
        help="Minimum pixel area for a connected component to count as a pupil (default: 50).",
    )
    args = parser.parse_args()

    out_w, out_h   = args.output_res
    input_path     = Path(args.input_path)
    output_path    = Path(args.output_path)
    images_in      = input_path  / "images"
    masks_in       = input_path  / "masks"
    images_out     = output_path / "images"
    masks_out      = output_path / "masks"

    # Validate input paths
    for p in (images_in, masks_in):
        if not p.exists():
            print(f"Error: {p} does not exist.", file=sys.stderr)
            sys.exit(1)

    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    # Collect image files that have a corresponding mask
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_files = sorted(
        f for f in images_in.iterdir()
        if f.suffix.lower() in extensions
        and (masks_in / f.name).exists()
    )

    if not image_files:
        print("No valid image-mask pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} image-mask pairs.")
    print(f"Output resolution : {out_w}×{out_h}")
    if args.input_res:
        print(f"Expected input res : {args.input_res[0]}×{args.input_res[1]}")
    print(f"Output path       : {output_path}\n")

    processed = 0
    skipped   = 0

    total = len(image_files)
    for idx, img_path in enumerate(image_files, 1):
        mask_path = masks_in / img_path.name
        stem      = img_path.stem

        print(f"  [{idx}/{total}] {img_path.name}", end="\r", flush=True)

        # Load
        try:
            image = Image.open(img_path).convert("L")
            mask  = Image.open(mask_path).convert("L")
        except Exception as e:
            print(f"\n  [SKIP] {img_path.name}: load error — {e}", file=sys.stderr)
            skipped += 1
            continue

        # Optional resolution check
        if args.input_res:
            exp_w, exp_h = args.input_res
            if image.size != (exp_w, exp_h):
                print(
                    f"\n  [SKIP] {img_path.name}: "
                    f"expected {exp_w}×{exp_h}, got {image.size[0]}×{image.size[1]}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

        image_np = np.array(image)
        mask_np  = np.array(mask)

        # Step 1 — find the two pupil centres
        try:
            (cx_l, cy_l), (cx_r, cy_r) = find_pupil_centers(mask_np, args.min_area)
        except ValueError as e:
            print(f"\n  [SKIP] {img_path.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Step 2 — extract crops at output resolution from the original image
        crop_img_l  = extract_crop(image_np, cx_l, cy_l, out_w, out_h)
        crop_mask_l = extract_crop(mask_np,  cx_l, cy_l, out_w, out_h)
        crop_img_r  = extract_crop(image_np, cx_r, cy_r, out_w, out_h)
        crop_mask_r = extract_crop(mask_np,  cx_r, cy_r, out_w, out_h)

        # Step 3 — scale full image and mask to output resolution
        scaled_image = image.resize((out_w, out_h), Image.BILINEAR)
        scaled_mask  = mask.resize((out_w, out_h),  Image.NEAREST)

        # Save
        scaled_image.save(images_out / f"{stem}.png")
        scaled_mask .save(masks_out  / f"{stem}.png")

        Image.fromarray(crop_img_l) .save(images_out / f"{stem}_l.png")
        Image.fromarray(crop_mask_l).save(masks_out  / f"{stem}_l.png")

        Image.fromarray(crop_img_r) .save(images_out / f"{stem}_r.png")
        Image.fromarray(crop_mask_r).save(masks_out  / f"{stem}_r.png")

        processed += 1

    print(f"\nDone.  Processed: {processed}  |  Skipped: {skipped}")
    if skipped:
        print("  Check the warnings above for details on skipped images.")


if __name__ == "__main__":
    main()
