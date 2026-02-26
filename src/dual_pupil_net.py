"""
dual_pupil_net.py — Single-model coarse+fine pupil segmentation for RK3566 NPU (0.9 TOPS).

Steady-state pipeline (frame N ≥ 1):
  1. CPU  downscale img_full[N] to 480×300                          → x_coarse   [~0.5ms]
  2. CPU  for each valid pupil i: crop img_full[N] from roi[i]      → x_fine     [~0.1ms]
           if pupil i absent in last frame: x_fine[i] = last valid crop (or zeros)
  3. NPU  rknn_run(x_coarse, x_fine)  → y_heatmap, y_conf, y_fine  [~10ms]
  4. CPU  for each pupil i:
           if sigmoid(y_conf[0,i]) > CONF_THR:
               roi[i] = soft_argmax(y_heatmap[0,i]) * 16    ← update for frame N+1
               mask_i  = sigmoid(y_fine[i]) > SEG_THR       ← valid mask
           else:
               roi[i] unchanged (use last valid centre for next frame crop)
               mask_i  = None / invalid mask

Frame 0 (init):
  x_fine = zeros → rknn_run → use y_heatmap/y_conf to initialise roi,
                               discard y_fine.

L/R Convention:
  Channel 0 = LEFT pupil in the camera frame (= patient's RIGHT eye, OD).
  Channel 1 = RIGHT pupil in the camera frame (= patient's LEFT eye, OS).
  Dataset annotations must follow this convention.

RKNN export:
  model.eval()
  tc, tf = torch.zeros(1,1,300,480), torch.zeros(2,1,160,160)
  torch.onnx.export(model, (tc, tf), 'dual_pupil_net.onnx',
                    input_names=['x_coarse', 'x_fine'],
                    output_names=['y_heatmap', 'y_conf', 'y_fine'],
                    opset_version=12, do_constant_folding=True)

  # rknn-toolkit2 (Python):
  # rknn.config(mean_values=[[128],[128]], std_values=[[128],[128]],
  #             target_platform='rk3566')
  # rknn.load_onnx('dual_pupil_net.onnx')
  # rknn.build(do_quantization=True, dataset='calibration.txt')  # ~200 img
  # rknn.export_rknn('dual_pupil_net.rknn')

RKNN notes:
  - ReLU6 preferred over ReLU: limited range → less INT8 quantisation error.
  - sigmoid NOT included in model outputs: apply on CPU, or threshold 0 on logits.
  - F.interpolate(mode='bilinear') supported by RKNN; if issues → mode='nearest'.
  - fine branch batch=2 must be fixed at compile-time:
    specify input_size_list=[[1,1,300,480],[2,1,160,160]] in the config.
  - y_conf has shape [1,2,1,1] (Conv2d avoids Reshape/Flatten which are problematic on NPU);
    on CPU: conf = sigmoid(y_conf[0,:,0,0])  →  tensor([conf_L, conf_R]).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class GhostModule(nn.Module):
    """
    Ghost module (Han et al., GhostNet CVPR 2020).
    Generates out_ch features at ~50% of the cost of a standard conv:
      half channels via regular conv (primary), half via depthwise cheap op.
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, ratio: int = 2):
        super().__init__()
        init_ch  = math.ceil(out_ch / ratio)
        cheap_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, k, padding=k // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU6(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, k, padding=k // 2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.primary(x)
        return torch.cat([p, self.cheap(p)], dim=1)


class DoubleGhost(nn.Module):
    """Two GhostModules in sequence."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            GhostModule(in_ch, out_ch),
            GhostModule(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two 3×3 convolutions + BN + ReLU6."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Coarse Branch ──────────────────────────────────────────────────────────────

class CoarseBranch(nn.Module):
    """
    Encoder-only branch for pupil localisation on a 1/4 scale image.

    Input : [1, 1, 300, 480]
    Output: heatmap [1, 2, 75, 120]  — pupil position (logit)
            conf    [1, 2,  1,  1]  — pupil existence (logit)

    Each heatmap pixel corresponds to a 4×4 px block in 480×300
    and to 16×16 px in the original 1920×1200.

    Architecture:
      Conv2d(1→base_ch, 3×3, stride=2)  @ 300×480  → 150×240   ← lightweight stem
      DoubleGhost(base_ch → base_ch*2)  @ 150×240
      MaxPool ────────────────────────────────────── 75×120
      DoubleGhost(base_ch*2 → base_ch*2) @ 75×120
        ├─ Conv1×1(base_ch*2 → 2)  @ 75×120 → heatmap  [1, 2, 75, 120]
        └─ AvgPool(75×120) → Conv1×1(base_ch*2 → 2)  → conf  [1, 2, 1, 1]

    With base_ch=16:
      MACs stem:  1×16×9×150×240  =   5 M   (vs 197 M for full-res DoubleGhost)
      MACs enc1:  DoubleGhost     = 260 M
      MACs enc2:  DoubleGhost     =  86 M
      Total CoarseBranch ≈ 351 M  (-35% vs full-res version)

    The coarse task is to find ~25 px blobs: a stride-2 conv + 2 Ghost stages
    is more than sufficient; full-resolution features are not needed.

    Training:
      Heatmap — for each pupil i:
        PRESENT: target = Gaussian σ≈2px centred at (cx_full/16, cy_full/16).
        ABSENT:  target = zero map.
        Loss: BCEWithLogitsLoss (use pos_weight to balance background).
      Conf — for each pupil i:
        PRESENT: target = 1.  ABSENT: target = 0.
        Loss: BCEWithLogitsLoss.

    Inference:
      conf = torch.sigmoid(y_conf[0, :, 0, 0])         # [2]: conf_L, conf_R
      for i in {0, 1}:
          if conf[i] > CONF_THR:                        # e.g. 0.5
              c = soft_argmax_2d(y_heatmap[:, i:i+1])  # [1, 1, 2] normalised
              roi[i] = c[0, 0] * tensor([1920., 1200.]) # px in 1920×1200

    Note: AvgPool2d with fixed kernel (75, 120) instead of AdaptiveAvgPool2d
    for guaranteed compatibility with all RKNN-Toolkit2 backends.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        # Stride-2 stem: downsampling 300×480 → 150×240 with a single conv.
        # Avoids full-resolution DoubleGhost (197M MACs → 5M MACs).
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DoubleGhost(base_ch,      base_ch * 2)   # @ 150×240
        self.enc2 = DoubleGhost(base_ch * 2,  base_ch * 2)   # @  75×120
        self.heatmap_head = nn.Conv2d(base_ch * 2, 2, 1)
        # Fixed AvgPool on the 75×120 bottleneck → avoids AdaptiveAvgPool which
        # is not always accelerated on NPU; after pool spatial dim is [1,1] → Conv2d as FC.
        self.conf_pool = nn.AvgPool2d(kernel_size=(75, 120))
        self.conf_head = nn.Conv2d(base_ch * 2, 2, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)                               # [1, 16, 150, 240]
        x = self.enc1(x)                               # [1, 32, 150, 240]
        x = self.pool(x)                               # [1, 32,  75, 120]
        x = self.enc2(x)                               # [1, 32,  75, 120]
        heatmap = self.heatmap_head(x)                 # [1,  2,  75, 120]
        conf    = self.conf_head(self.conf_pool(x))    # [1,  2,   1,   1]
        return heatmap, conf


# ── Fine Branch ────────────────────────────────────────────────────────────────

class FineBranch(nn.Module):
    """
    Small U-Net for precise segmentation on full-res crops.

    Input : [2, 1, 160, 160]  — batch of 2 pupils (idx 0=left, idx 1=right).
                                 Shared weights: equivalent to processing them separately
                                 with the same network, but in a single NPU operation.
    Output: [2, 1, 160, 160]  — binary masks (logit, no sigmoid).

    U-Net architecture with features=[16,32,64], 128ch bottleneck:
      Encoder:
        DoubleConv(1→16)   @ 160×160
        MaxPool             → 80×80
        DoubleConv(16→32)  @ 80×80
        MaxPool             → 40×40
        DoubleConv(32→64)  @ 40×40
        MaxPool             → 20×20
      Bottleneck:
        DoubleConv(64→128) @ 20×20
      Decoder (bilinear up + 1×1 + skip cat + DoubleConv):
        →64ch @ 40×40
        →32ch @ 80×80
        →16ch @ 160×160
      Head: Conv1×1(16→1)

    use_ghost : bool
        False (default) = standard DoubleConv, maximum quality.
        True            = DoubleGhost (~35% fewer MACs, ~1,200M vs 1,864M).
                          Recommended if the actual framerate is insufficient.
        Estimated MACs:
          use_ghost=False → ~1,864 M  →  total model ~2,215 M
          use_ghost=True  → ~1,200 M  →  total model ~1,551 M

    RKNN: compile with input_size_list [[2,1,160,160]].
    """

    def __init__(self, features: list = None, use_ghost: bool = False):
        super().__init__()
        if features is None:
            features = [16, 32, 64]

        ConvBlock = DoubleGhost if use_ghost else DoubleConv

        self.pool      = nn.MaxPool2d(2, 2)
        self.encoders  = nn.ModuleList()
        self.up_convs  = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        ch = 1
        for f in features:
            self.encoders.append(ConvBlock(ch, f))
            ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for f in reversed(features):
            # 1×1 conv: halves channels before cat with the skip
            self.up_convs.append(nn.Conv2d(f * 2, f, 1, bias=False))
            self.dec_convs.append(ConvBlock(f * 2, f))

        self.head = nn.Conv2d(features[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up, dec, skip in zip(self.up_convs, self.dec_convs, skips):
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.head(x)


# ── DualPupilNet ───────────────────────────────────────────────────────────────

class DualPupilNet(nn.Module):
    """
    Unified coarse+fine model for IR pupil segmentation.
    A single RKNN model loaded in NPU, called with rknn_run() on every frame.

    ┌─ Inputs ────────────────────────────────────────────────────────────────┐
    │ x_coarse  [1, 1, 300, 480]  full image downscaled 4× (grayscale)       │
    │ x_fine    [2, 1, 160, 160]  full-res crops centred on the pupils        │
    │                              idx 0 = left pupil, idx 1 = right pupil   │
    │                              First frame: zero tensor.                  │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ Outputs ────────────────────────────────────────────────────────────────┐
    │ y_heatmap [1, 2, 75, 120]   pupil position (logit) → soft_argmax        │
    │ y_conf    [1, 2,  1,  1]   pupil existence (logit) → sigmoid > thr      │
    │ y_fine    [2, 1, 160, 160]  precise masks (logit) → sigmoid > thr       │
    └──────────────────────────────────────────────────────────────────────────┘

    Total parameters: ~475K (INT8 quantized ≈ 475KB of weights).

    If a pupil is absent (conf[i] < threshold):
      - roi[i] is NOT updated (keeps the last valid centre)
      - y_fine[i] is discarded
      - x_fine[i] for the next frame = last valid crop (temporal tracking)
        or zeros if never detected (first occurrence).
    """

    def __init__(
        self,
        coarse_base_ch: int = 16,
        fine_features: list = None,
        fine_use_ghost: bool = False,
    ):
        super().__init__()
        self.coarse = CoarseBranch(base_ch=coarse_base_ch)
        self.fine   = FineBranch(features=fine_features, use_ghost=fine_use_ghost)

    def forward(
        self,
        x_coarse: torch.Tensor,   # [1, 1, 300, 480]
        x_fine:   torch.Tensor,   # [2, 1, 160, 160]
    ):
        y_heatmap, y_conf = self.coarse(x_coarse)   # [1,2,75,120], [1,2,1,1]
        y_fine            = self.fine(x_fine)         # [2, 1, 160, 160]
        return y_heatmap, y_conf, y_fine


# ── CPU utilities (not exported to NPU) ────────────────────────────────────────

def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax on a 2D heatmap.
    More accurate than discrete argmax: estimates subpixel peak position.

    Input : [B, C, H, W]
    Output: [B, C, 2]  — (x_norm, y_norm) ∈ [0,1]²

    Example:
        centres_norm = soft_argmax_2d(y_heatmap)           # [1, 2, 2]
        centres_480  = centres_norm[0] * torch.tensor([480., 300.])  # px in 480×300
        centres_full = centres_480 * 4                     # px in 1920×1200
    """
    B, C, H, W = heatmap.shape
    hm = torch.softmax(heatmap.reshape(B, C, -1), dim=-1).reshape(B, C, H, W)

    xs = torch.linspace(0, 1, W, device=heatmap.device)
    ys = torch.linspace(0, 1, H, device=heatmap.device)
    grid_x = xs.view(1, 1, 1, W).expand(B, C, H, W)
    grid_y = ys.view(1, 1, H, 1).expand(B, C, H, W)

    cx = (hm * grid_x).sum(dim=(2, 3))   # [B, C]
    cy = (hm * grid_y).sum(dim=(2, 3))   # [B, C]
    return torch.stack([cx, cy], dim=-1)  # [B, C, 2]


def extract_crops(
    img_full:   torch.Tensor,         # [1, 1, 1200, 1920]
    centres:    torch.Tensor,         # [2, 2]   (cx, cy) in px 1920×1200, for pupils L and R
    valid:      torch.Tensor,         # [2]      bool — True if the pupil was detected
    prev_crops: torch.Tensor | None,  # [2, 1, crop_size, crop_size] or None (first frame)
    crop_size:  int = 160,
) -> torch.Tensor:
    """
    Crops two patches from the full-res image for the fine branch.

    For each pupil i:
      - If valid[i]: crop centred on centres[i] from the current image.
      - If not valid[i] and prev_crops available: reuse prev_crops[i]
        (temporal tracking — the pupil was visible in the previous frame).
      - If not valid[i] and prev_crops is None: zero patch
        (first frame, pupil never detected).

    Borders are handled with reflect padding (rare case, not optimised).

    Output: [2, 1, crop_size, crop_size]
    """
    H, W = img_full.shape[2], img_full.shape[3]
    half  = crop_size // 2
    crops = []

    for i, (cx, cy) in enumerate(centres.long()):
        if not valid[i]:
            if prev_crops is not None:
                crops.append(prev_crops[i:i+1])
            else:
                crops.append(torch.zeros(1, 1, crop_size, crop_size,
                                         device=img_full.device,
                                         dtype=img_full.dtype))
            continue

        cx, cy = cx.item(), cy.item()
        pad_l = max(0, half - cx)
        pad_r = max(0, cx + half - W)
        pad_t = max(0, half - cy)
        pad_b = max(0, cy + half - H)
        x0 = max(0, cx - half);  x1 = min(W, cx + half)
        y0 = max(0, cy - half);  y1 = min(H, cy + half)
        patch = img_full[:, :, y0:y1, x0:x1]
        if pad_l or pad_r or pad_t or pad_b:
            patch = F.pad(patch, (pad_l, pad_r, pad_t, pad_b), mode='reflect')
        crops.append(patch)

    return torch.cat(crops, dim=0)   # [2, 1, crop_size, crop_size]


def make_gaussian_heatmap(
    centre: tuple,    # (cx, cy) in pixels in 1920×1200
    out_h: int = 75,
    out_w: int = 120,
    full_h: int = 1200,
    full_w: int = 1920,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Generates a 2D Gaussian heatmap for CoarseBranch training.

    The centre is projected from 1920×1200 coordinates to out_h×out_w
    (divided by the scale factor = full_h/out_h = 16).

    Output: [out_h, out_w] — values in [0,1]
    """
    scale_y = full_h / out_h
    scale_x = full_w / out_w
    cy_hm = centre[1] / scale_y
    cx_hm = centre[0] / scale_x

    ys = torch.arange(out_h, dtype=torch.float32)
    xs = torch.arange(out_w, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    hm = torch.exp(-((grid_x - cx_hm)**2 + (grid_y - cy_hm)**2) / (2 * sigma**2))
    return hm   # [out_h, out_w]
