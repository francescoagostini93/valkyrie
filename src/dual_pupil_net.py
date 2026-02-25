"""
dual_pupil_net.py — Single-model coarse+fine pupil segmentation for RK3566 NPU (0.9 TOPS).

Pipeline a regime (frame N ≥ 1):
  1. CPU  downscale img_full[N] a 480×300                      → x_coarse   [~0.5ms]
  2. CPU  per ogni pupilla i valida: crop img_full[N] da roi[i] → x_fine     [~0.1ms]
           se pupilla i assente nell'ultimo frame: x_fine[i] = ultimo crop valido (o zeros)
  3. NPU  rknn_run(x_coarse, x_fine)  → y_heatmap, y_conf, y_fine  [~10ms]
  4. CPU  per ogni pupilla i:
           if sigmoid(y_conf[0,i]) > CONF_THR:
               roi[i] = soft_argmax(y_heatmap[0,i]) * 16    ← aggiorna per frame N+1
               mask_i  = sigmoid(y_fine[i]) > SEG_THR       ← maschera valida
           else:
               roi[i] invariato (usa ultimo centro valido per crop del prossimo frame)
               mask_i  = None / maschera invalida

Frame 0 (init):
  x_fine = zeros → rknn_run → usa y_heatmap/y_conf per inizializzare roi,
                               ignora y_fine.

Convenzione L/R:
  Canale 0 = pupilla SINISTRA nel frame camera (= occhio DESTRO del paziente, OD).
  Canale 1 = pupilla DESTRA  nel frame camera (= occhio SINISTRO del paziente, OS).
  Le annotazioni del dataset devono rispettare questa convenzione.

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

Note RKNN:
  - ReLU6 preferito a ReLU: range limitato → meno errore quantizzazione INT8.
  - sigmoid NON inclusa negli output del modello: applica su CPU, oppure soglia 0 sui logit.
  - F.interpolate(mode='bilinear') supportato da RKNN; se problemi → mode='nearest'.
  - batch=2 del fine branch deve essere fisso a compile-time:
    specificare input_size_list=[[1,1,300,480],[2,1,160,160]] nella config.
  - y_conf ha shape [1,2,1,1] (Conv2d evita Reshape/Flatten problematici su NPU);
    su CPU: conf = sigmoid(y_conf[0,:,0,0])  →  tensor([conf_L, conf_R]).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ────────────────────────────────────────────────────────────

class GhostModule(nn.Module):
    """
    Ghost module (Han et al., GhostNet CVPR 2020).
    Genera out_ch feature con ~50% del costo di una conv standard:
      metà canali via conv regolare (primary), metà via depthwise cheap op.
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
    """Due GhostModule in sequenza."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            GhostModule(in_ch, out_ch),
            GhostModule(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Due conv 3×3 + BN + ReLU6."""
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
    Encoder-only per localizzazione pupille su immagine 1/4 scala.

    Input : [1, 1, 300, 480]
    Output: heatmap [1, 2, 75, 120]  — posizione pupille (logit)
            conf    [1, 2,  1,  1]  — esistenza pupille (logit)

    Ogni pixel dell'heatmap corrisponde a un blocco 4×4 px nel 480×300
    e a 16×16 px nel 1920×1200 originale.

    Architettura:
      Conv2d(1→base_ch, 3×3, stride=2)  @ 300×480  → 150×240   ← stem leggero
      DoubleGhost(base_ch → base_ch*2)  @ 150×240
      MaxPool ────────────────────────────────────── 75×120
      DoubleGhost(base_ch*2 → base_ch*2) @ 75×120
        ├─ Conv1×1(base_ch*2 → 2)  @ 75×120 → heatmap  [1, 2, 75, 120]
        └─ AvgPool(75×120) → Conv1×1(base_ch*2 → 2)  → conf  [1, 2, 1, 1]

    Con base_ch=16:
      MACs stem:  1×16×9×150×240  =   5 M   (vs 197 M del DoubleGhost full-res)
      MACs enc1:  DoubleGhost     = 260 M
      MACs enc2:  DoubleGhost     =  86 M
      Totale CoarseBranch ≈ 351 M  (-35% rispetto alla versione full-res)

    Il compito coarse è trovare blob da ~25 px: una conv stride-2 + 2 Ghost stages
    è ampiamente sufficiente; non servono feature a piena risoluzione.

    Training:
      Heatmap — per ogni pupilla i:
        PRESENTE: target = gaussiana σ≈2px centrata su (cx_full/16, cy_full/16).
        ASSENTE:  target = mappa di zeri.
        Loss: BCEWithLogitsLoss (usare pos_weight per bilanciare lo sfondo).
      Conf — per ogni pupilla i:
        PRESENTE: target = 1.  ASSENTE: target = 0.
        Loss: BCEWithLogitsLoss.

    Inference:
      conf = torch.sigmoid(y_conf[0, :, 0, 0])         # [2]: conf_L, conf_R
      for i in {0, 1}:
          if conf[i] > CONF_THR:                        # es. 0.5
              c = soft_argmax_2d(y_heatmap[:, i:i+1])  # [1, 1, 2] normalizzato
              roi[i] = c[0, 0] * tensor([1920., 1200.]) # px in 1920×1200

    Note: AvgPool2d con kernel fisso (75, 120) invece di AdaptiveAvgPool2d
    per compatibilità garantita con tutti i backend RKNN-Toolkit2.
    """

    def __init__(self, base_ch: int = 16):
        super().__init__()
        # Stride-2 stem: downsampling 300×480 → 150×240 con una singola conv.
        # Evita il DoubleGhost a piena risoluzione (197M MACs → 5M MACs).
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.enc1 = DoubleGhost(base_ch,      base_ch * 2)   # @ 150×240
        self.enc2 = DoubleGhost(base_ch * 2,  base_ch * 2)   # @  75×120
        self.heatmap_head = nn.Conv2d(base_ch * 2, 2, 1)
        # AvgPool fisso sul bottleneck 75×120 → evita AdaptiveAvgPool non sempre
        # accelerato su NPU; dopo pool la spatial dim è [1,1] → Conv2d come FC.
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
    Small U-Net per segmentazione precisa su crop full-res.

    Input : [2, 1, 160, 160]  — batch di 2 pupille (idx 0=sx, idx 1=dx).
                                 Pesi condivisi: equivalente a processarle separatamente
                                 con la stessa rete, ma in un'unica operazione NPU.
    Output: [2, 1, 160, 160]  — maschere binarie (logit, nessuna sigmoid).

    Architettura U-Net con features=[16,32,64], bottleneck 128ch:
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
        False (default) = DoubleConv standard, qualità massima.
        True            = DoubleGhost (~35% meno MACs, ~1,200M vs 1,864M).
                          Raccomandato se il framerate reale è insufficiente.
        MACs stimati:
          use_ghost=False → ~1,864 M  →  modello totale ~2,215 M
          use_ghost=True  → ~1,200 M  →  modello totale ~1,551 M

    RKNN: compilare con input_size_list [[2,1,160,160]].
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
            # 1×1 conv: dimezza i canali prima del cat con lo skip
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
    Modello unificato coarse+fine per segmentazione pupille IR.
    Un solo modello RKNN caricato in NPU, chiamato con rknn_run() ad ogni frame.

    ┌─ Inputs ────────────────────────────────────────────────────────────────┐
    │ x_coarse  [1, 1, 300, 480]  immagine full downscalata 4× (grayscale)   │
    │ x_fine    [2, 1, 160, 160]  crop full-res centrati sulle pupille        │
    │                              idx 0 = pupilla sx, idx 1 = pupilla dx     │
    │                              Primo frame: tensor di zeri.               │
    └─────────────────────────────────────────────────────────────────────────┘
    ┌─ Outputs ────────────────────────────────────────────────────────────────┐
    │ y_heatmap [1, 2, 75, 120]   posizione pupille (logit) → soft_argmax     │
    │ y_conf    [1, 2,  1,  1]   esistenza pupille (logit) → sigmoid > thr    │
    │ y_fine    [2, 1, 160, 160]  maschere precise (logit) → sigmoid > thr    │
    └──────────────────────────────────────────────────────────────────────────┘

    Parametri totali: ~475K (INT8 quantized ≈ 475KB di pesi).

    Se una pupilla è assente (conf[i] < soglia):
      - roi[i] NON viene aggiornata (tiene l'ultimo centro valido)
      - y_fine[i] viene ignorata
      - x_fine[i] al frame successivo = ultimo crop valido (tracking temporale)
        oppure zeros se mai rilevata (prima occorrenza).
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


# ── CPU utilities (non esportate su NPU) ───────────────────────────────────────

def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Soft-argmax differenziabile su heatmap 2D.
    Più accurato dell'argmax discreto: stima subpixel della posizione del picco.

    Input : [B, C, H, W]
    Output: [B, C, 2]  — (x_norm, y_norm) ∈ [0,1]²

    Esempio:
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
    centres:    torch.Tensor,         # [2, 2]   (cx, cy) in px 1920×1200, per pupilla L e R
    valid:      torch.Tensor,         # [2]      bool — True se la pupilla è stata rilevata
    prev_crops: torch.Tensor | None,  # [2, 1, crop_size, crop_size] o None (primo frame)
    crop_size:  int = 160,
) -> torch.Tensor:
    """
    Ritaglia due patch dall'immagine full-res per il fine branch.

    Per ogni pupilla i:
      - Se valid[i]: crop centrato su centres[i] dall'immagine corrente.
      - Se not valid[i] e prev_crops disponibile: riusa prev_crops[i]
        (tracking temporale — la pupilla era visibile al frame precedente).
      - Se not valid[i] e prev_crops è None: patch di zeri
        (primo frame, pupilla mai rilevata).

    I bordi vengono gestiti con padding reflect (caso raro, non ottimizzato).

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
    centre: tuple,    # (cx, cy) in pixel nel 1920×1200
    out_h: int = 75,
    out_w: int = 120,
    full_h: int = 1200,
    full_w: int = 1920,
    sigma: float = 2.0,
) -> torch.Tensor:
    """
    Genera un heatmap gaussiano 2D per il training del CoarseBranch.

    Il centro viene proiettato da coordinate 1920×1200 a out_h×out_w
    (divisione per il fattore di scala = full_h/out_h = 16).

    Output: [out_h, out_w] — valori in [0,1]
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
