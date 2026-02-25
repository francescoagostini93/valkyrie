# Valkyrie — Pupil Segmentation with U-Net Variants

Binary segmentation of pupil/iris regions in grayscale eye images using PyTorch. Developed as part of a university thesis project.

## Overview

The project trains and evaluates several U-Net variants for binary segmentation on high-resolution grayscale images (1920×1200). The goal is to accurately segment the pupil region from images acquired by ophthalmological devices.

## Models

| Model | Description |
|---|---|
| `UNet` | Classic encoder-decoder U-Net (Ronneberger et al., MICCAI 2015) with skip connections |
| `SimpleUNET` | Lightweight U-Net variant using transposed convolutions for upsampling |
| `AttentionUNet` | U-Net with attention gates on skip connections (Oktay et al., MIDL 2018) |
| `HalfUNet` | Simplified U-Net with unified channel width, full-scale feature fusion and Ghost modules (Lu et al., Front. Neuroinform. 2022) |

All models share:
- `in_channels=1` (8-bit grayscale input)
- `out_channels=1` (binary mask, sigmoid applied by the loss)

## Project Structure

```
valkyrie/
├── config/             # Training configuration (YAML)
├── data/               # Datasets — contents ignored by git
├── docs/               # Documentation and references — contents ignored by git
├── experiments/        # Per-run configuration snapshots
├── models/             # Saved model weights — contents ignored by git
├── notebooks/          # Jupyter notebooks for exploration
├── runs/               # TensorBoard event files — contents ignored by git
├── scripts/            # Auxiliary scripts (e.g. dataset visualizer)
├── src/                # Source code
│   ├── dataset.py      # Dataset class and train/val split
│   ├── evaluate.py     # Evaluation script
│   ├── loss.py         # Dice loss
│   ├── model.py        # Legacy model definitions
│   ├── new_models.py   # Current model architectures (HalfUNet, UNet, AttentionUNet)
│   ├── predict.py      # Inference script
│   └── train.py        # Training loop with TensorBoard logging
└── tests/              # Unit tests
```

## Requirements

```
torch
torchvision
Pillow
tensorboard
matplotlib
numpy
pyyaml
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
cd src
python train.py
```

Training parameters (dataset paths, batch size, epochs, model selection) are configured directly at the top of `train.py`.

TensorBoard logs are written to `runs/`. To monitor training:

```bash
tensorboard --logdir runs
```

### Inference

```bash
cd src
python predict.py
```

Loads `models/unet.pth` and runs inference on images in `data/test_images/images/`, saving predicted masks to `data/test_images/masks/`.

### Evaluation

```bash
cd src
python evaluate.py
```

### Configuration

`config/config.yaml` holds dataset paths, model type, and hyperparameters for reference.

## Memory Notes

With `batch_size=2`, `float32`:
- The heaviest activation is always the first encoder level (`1920×1200×64 channels ≈ 590 MB/sample`)
- `HalfUNet` keeps 64 channels at all levels → minimal decoder memory (~0.21 M parameters with Ghost modules)
- `UNet` / `AttentionUNet` reach 1024 channels at the bottleneck → heavier (~31 M parameters)
- For GPUs with limited VRAM: reduce `features` or use patch-based training

## Loss

Dice Loss is used for training, optimized to handle class imbalance between background and small pupil regions.

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018
- Han et al., *GhostNet: More Features from Cheap Operations*, CVPR 2020
- Lu et al., *Half-UNet: A Simplified U-Net Architecture for Medical Image Segmentation*, Front. Neuroinform. 2022, doi:10.3389/fninf.2022.911679
