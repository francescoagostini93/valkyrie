"""
new_models.py — Architectures for binary segmentation on 1920x1200 grayscale images.

Implemented models:
  - HalfUNet       : half-U encoder with full-scale feature fusion and Ghost modules
                     (Lu et al., Front. Neuroinform. 2022, doi:10.3389/fninf.2022.911679)
  - UNet           : classic encoder-decoder with skip connections
  - AttentionUNet  : UNet with attention gates on skip connections

All three networks share:
  - in_channels=1  (8-bit grayscale)
  - out_channels=1 (binary mask, sigmoid applied by the loss)
  - arbitrary input: 1920x1200 dimensions divide exactly by 2^4,
    so 4 levels of MaxPool2d(2) produce no fractional dimensions.

Memory notes (batch_size=2, float32):
  - The heaviest activation is always the first encoder level
    (1920x1200x64 channels ≈ 590 MB/sample).
  - HalfUNet keeps 64 channels at all levels → minimal decoder memory.
  - UNet / AttentionUNet reach 1024 channels at the bottleneck → heavier.
  - For GPUs with limited VRAM: reduce features or use patch training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared base blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two 3x3 convolutions + BN + ReLU in sequence (basic U-Net block)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Ghost Module (per HalfUNet)
# ---------------------------------------------------------------------------

class GhostModule(nn.Module):
    """
    Ghost module — Han et al., GhostNet, CVPR 2020.

    Generates `out_channels` feature maps at ~50% of the cost of a standard conv:
      - half of the channels: regular 3x3 conv  (primary conv)
      - half of the channels: depthwise 3x3 conv applied to the primary output (cheap op)
    The two groups are concatenated along the channel axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio: int = 2,
        dw_kernel_size: int = 3,
    ):
        super().__init__()
        init_channels  = math.ceil(out_channels / ratio)   # primary channels
        cheap_channels = out_channels - init_channels       # cheap channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, init_channels, kernel_size,
                padding=kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels, cheap_channels, dw_kernel_size,
                padding=dw_kernel_size // 2, groups=init_channels, bias=False,
            ),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class DoubleGhostConv(nn.Module):
    """Two GhostModules in sequence (analogous to DoubleConv for HalfUNet)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            GhostModule(in_channels, out_channels),
            GhostModule(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# HalfUNet
# ---------------------------------------------------------------------------

class HalfUNet(nn.Module):
    """
    Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation.
    Lu et al., Frontiers in Neuroinformatics, 2022.
    doi: 10.3389/fninf.2022.911679

    Key ideas:
      1. Unified channels — all encoder stages produce `features` channels
         (no doubling), eliminating the overhead of a classic decoder.
      2. Full-scale feature fusion — each encoder output is resampled (bilinear)
         to the original resolution and SUMMED
         (addition, not concatenation → no additional parameters).
      3. Ghost modules — replace standard convolutions in the encoder,
         halving parameters and FLOPs at the same channel count.

    Recommended configuration for 1920x1200 grayscale:
        HalfUNet(in_channels=1, out_channels=1, features=64, num_levels=5)
        → maximum downsampling x16 (bottleneck at 120x75)
        → ~0.21 M parameters with Ghost, ~0.41 M without Ghost

    Parameters
    ----------
    in_channels : int   — input channels (1 for grayscale)
    out_channels : int  — output channels (1 for binary mask)
    features : int      — uniform channels across all encoder levels (default 64)
    num_levels : int    — number of encoder levels (default 5 → downsampling x16)
    use_ghost : bool    — True = Ghost modules (Half-UNet),
                          False = standard conv (Half-UNet†, more parameters)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        num_levels: int = 5,
        use_ghost: bool = True,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ConvBlock = DoubleGhostConv if use_ghost else DoubleConv

        # Encoder: num_levels stages, all with `features` channels
        # Level 0 (C1): original resolution, no pooling
        # Level i>0   : MaxPool2d before the conv block
        self.encoders = nn.ModuleList()
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else features
            self.encoders.append(ConvBlock(in_ch, features))

        # Decoder: one conv block after full-scale feature fusion
        self.decoder_conv = ConvBlock(features, features)

        # Final 1x1 classifier
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Encoder pass
        encoder_outs = []
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = encoder(x)
            encoder_outs.append(x)

        # Full-scale feature fusion:
        # C1 is already at (H, W); C2..Cn are upsampled and summed
        fused = encoder_outs[0]
        for i in range(1, self.num_levels):
            up = F.interpolate(
                encoder_outs[i], size=(H, W), mode='bilinear', align_corners=False
            )
            fused = fused + up

        out = self.decoder_conv(fused)
        return self.final_conv(out)


# ---------------------------------------------------------------------------
# MiniHalfUNet — NPU-optimised lightweight variant
# ---------------------------------------------------------------------------

class MiniHalfUNet(nn.Module):
    """
    MiniHalfUNet — compact, NPU-optimised variant of HalfUNet for RK3566.

    Differences from HalfUNet:
      - Default features=32  (vs 64) — halves channel width throughout the network.
      - Default num_levels=4         — bottleneck at ~17×30 for 140×240 input.
      - Hardcoded target resolution (img_h, img_w) stored at init time.
        The full-scale fusion uses  F.interpolate(..., size=(img_h, img_w))
        with Python-integer constants, NOT tensor.shape[2:].
        This guarantees that the exported ONNX graph contains no Shape/Slice
        nodes, so all Resize ops are assigned to the NPU without CPU fallback.
      - mode='nearest' in all Resize nodes — the only mode accelerated on NPU
        driver 0.9.8 for scale factors > 2×.

    NPU performance target (RK3566, INT8, 140×240):
        ~25–50 ms / 20–40 FPS  (vs 203 ms for HalfUNet at 300×480).

    Training note:
        Prepare the dataset at 240×140 px (width×height).
        The ground-truth masks must also be at 140×240 — the model output
        resolution equals the input resolution, so no post-processing resize
        is needed during training.

    Parameters
    ----------
    in_channels  : int  — input channels (1 for grayscale)
    out_channels : int  — output channels (1 for binary mask)
    features     : int  — uniform channel count across all encoder levels (default 32)
    num_levels   : int  — encoder depth (default 4 → bottleneck at ~17×30 for 140×240)
    img_h        : int  — input height in pixels (default 140)
    img_w        : int  — input width  in pixels (default 240)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 32,
        num_levels: int = 4,
        img_h: int = 140,
        img_w: int = 240,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.img_h = img_h
        self.img_w = img_w
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: num_levels stages, all with `features` channels
        self.encoders = nn.ModuleList()
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else features
            self.encoders.append(DoubleGhostConv(in_ch, features))

        # Decoder: single conv block after full-scale fusion
        self.decoder_conv = DoubleGhostConv(features, features)

        # Final 1×1 classifier
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder pass
        encoder_outs = []
        for i, encoder in enumerate(self.encoders):
            if i > 0:
                x = self.pool(x)
            x = encoder(x)
            encoder_outs.append(x)

        # Full-scale feature fusion with NPU-friendly constants:
        #   size=(self.img_h, self.img_w) are Python ints → constant in ONNX,
        #   no Shape/Slice nodes generated → all Resize ops stay on NPU.
        fused = encoder_outs[0]
        for i in range(1, self.num_levels):
            up = F.interpolate(
                encoder_outs[i],
                size=(self.img_h, self.img_w),
                mode='nearest',
            )
            fused = fused + up

        out = self.decoder_conv(fused)
        return self.final_conv(out)


# ---------------------------------------------------------------------------
# Standard UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Classic U-Net (Ronneberger et al., MICCAI 2015).

    Symmetric encoder-decoder with skip connections for concatenation.
    The decoder uses bilinear upsampling + conv (more stable than ConvTranspose2d
    for inputs whose dimensions are not strictly powers of 2).

    Spatial dimension alignment during upsampling is handled with
    F.interpolate for robustness to any input resolution.

    Recommended configuration for 1920x1200 grayscale:
        UNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
        → bottleneck at 120x75 with 1024 channels
        → ~31 M parameters

    Parameters
    ----------
    in_channels : int        — input channels (1 for grayscale)
    out_channels : int       — output channels (1 for binary mask)
    features : list[int]     — channels per encoder level
                               (default [64, 128, 256, 512])
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: for each level a 1x1 upsampling conv + DoubleConv
        # Input channels to DoubleConv: (feature*2 from below) + (feature from skip)
        self.up_convs   = nn.ModuleList()  # reduce channels before cat
        self.dec_convs  = nn.ModuleList()  # process the concatenated feature
        for f in reversed(features):
            # 1x1 conv that brings channels from f*2 to f (then cat with skip of f channels → 2f)
            self.up_convs.append(nn.Conv2d(f * 2, f, kernel_size=1, bias=False))
            self.dec_convs.append(DoubleConv(f * 2, f))

        # Final classifier
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]  # reverse: deepest skip first

        # Decoder
        for up_conv, dec_conv, skip in zip(self.up_convs, self.dec_convs, skips):
            # Bilinear upsampling to the corresponding skip dimension
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = up_conv(x)                          # f*2 → f channels
            x = torch.cat([skip, x], dim=1)         # cat: f + f = 2f channels
            x = dec_conv(x)                         # 2f → f channels

        return self.final_conv(x)


# ---------------------------------------------------------------------------
# AttentionUNet
# ---------------------------------------------------------------------------

class AttentionGate(nn.Module):
    """
    Attention gate (Oktay et al., Attention U-Net, MIDL 2018).

    Selects relevant skip connection features using the output of the
    underlying decoder level as a gating signal.

    Parameters
    ----------
    F_g : int   — channels of the gating signal (from the deeper decoder level)
    F_l : int   — channels of the skip features (from the encoder)
    F_int : int — channels of the intermediate attention space (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : gating signal  (from the decoder, lower resolution)
        x : skip features  (from the encoder, same resolution as the current decoder level)
        """
        # Align g to x's resolution if necessary
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)        # scalar attention map [B, 1, H, W]
        return x * psi             # rescaled features


class AttentionUNet(nn.Module):
    """
    Attention U-Net (Oktay et al., MIDL 2018).

    Identical to UNet but with an AttentionGate on every skip connection:
    before concatenating the encoder tensor with the decoder one, the encoder
    features are rescaled by an attention map guided by the gating signal from
    the underlying decoder level. This suppresses features irrelevant to
    segmentation, improving precision without significantly increasing
    the number of parameters.

    Recommended configuration for 1920x1200 grayscale:
        AttentionUNet(in_channels=1, out_channels=1, features=[64, 128, 256, 512])
        → ~31 M parameters (slightly more than UNet due to gates)

    Parameters
    ----------
    in_channels : int        — input channels (1 for grayscale)
    out_channels : int       — output channels (1 for binary mask)
    features : list[int]     — channels per encoder level
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = None,
    ):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder + Attention gates
        self.up_convs   = nn.ModuleList()
        self.att_gates  = nn.ModuleList()
        self.dec_convs  = nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.Conv2d(f * 2, f, kernel_size=1, bias=False))
            self.att_gates.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            self.dec_convs.append(DoubleConv(f * 2, f))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        # Decoder with attention
        for up_conv, att_gate, dec_conv, skip in zip(
            self.up_convs, self.att_gates, self.dec_convs, skips
        ):
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = up_conv(x)                          # f*2 → f channels
            skip = att_gate(g=x, x=skip)            # gate: rescale the skip
            x = torch.cat([skip, x], dim=1)         # cat: f + f = 2f channels
            x = dec_conv(x)                         # 2f → f channels

        return self.final_conv(x)
