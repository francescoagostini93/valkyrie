from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import SimpleUNET
from new_models import HalfUNet, UNet, AttentionUNet
from loss import DiceLoss
from dataset import get_train_val_datasets
import torchvision
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
#
# DATASET PARAMETERS
#   dataset_name   : subfolder tag used to locate the dataset directory.
#                    Paths are resolved as:
#                      data/dataset_{name}_{width}_{height}/images
#                      data/dataset_{name}_{width}_{height}/masks
#                    'dual' → output of scripts/DualPupilDatasetCreator
#   dataset_height : image height in pixels (must match the saved dataset)
#   dataset_width  : image width  in pixels (must match the saved dataset)
#   dataset_depth  : bit depth — informational only, embedded in the checkpoint
#                    filename for traceability.
#
# TRAINING PARAMETERS
#   batch_size  : number of samples per gradient update.
#                 Reduce to 1 if you get CUDA out-of-memory errors.
#   num_epochs  : maximum number of training epochs.
#   patience    : early-stopping patience — training stops after this many
#                 consecutive epochs without improvement in validation loss.
#
# MODEL SELECTION  →  set model_name to one of the strings in the table below.
#
#   Key        Class            Module           Params   Notes
#   ─────────────────────────────────────────────────────────────────────────
#   'simple'   SimpleUNET       model.py         ~31 M    Classic U-Net with
#                                                         ConvTranspose2d decoder.
#                                                         No BN in decoder.
#                                                         Good as a quick baseline.
#
#   'unet'     UNet             new_models.py    ~31 M    Improved U-Net with bilinear
#                                                         upsampling + BN throughout.
#                                                         More stable training than
#                                                         SimpleUNET.
#
#   'attn'     AttentionUNet    new_models.py    ~31 M    UNet + soft attention gates
#                                                         on every skip connection.
#                                                         Suppresses background noise;
#                                                         better precision on small
#                                                         objects (pupils).
#
#   'hal'      HalfUNet         new_models.py    ~0.21 M  Half-U-Net with Ghost modules
#                                                         + full-scale feature fusion.
#                                                         Lu et al., Front. Neuroinform.
#                                                         2022. Uniform 64-ch encoder,
#                                                         no decoder memory overhead.
#                                                         Fastest; recommended for
#                                                         embedded (RKNN/NPU) deployment.
#   ─────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

dataset_name   = 'dual'
dataset_height = 300
dataset_width  = 480
dataset_depth  = 8
model_name     = 'hal'   # see model selection table above

batch_size     = 8    # 8 on local GPU / 16 on Colab T4 — HalfUNet is VRAM-light
num_epochs     = 100  # let early stopping decide the actual stopping point
patience       = 10   # epochs without improvement before stopping

# Prepare tensorboard writer
writer = SummaryWriter()

# Paths for images and masks
image_dir = f'data/dataset_{dataset_name}_{dataset_width}_{dataset_height}/images'
mask_dir  = f'data/dataset_{dataset_name}_{dataset_width}_{dataset_height}/masks'

# Set up the transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Get the train and validation datasets
train_dataset, val_dataset = get_train_val_datasets(image_dir, mask_dir, transform)

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# Show sample data on tensorboard
dataiter = iter(train_loader)
images, labels = next(dataiter)

# create grid of images
img_grid    = torchvision.utils.make_grid(images)
labels_grid = torchvision.utils.make_grid(labels)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# show images
matplotlib_imshow(img_grid,    one_channel=True)
matplotlib_imshow(labels_grid, one_channel=True)

# write to tensorboard
writer.add_image('sample dataset images', img_grid)
writer.add_image('sample dataset labels', labels_grid)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model weight initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def log_predictions(model, dataloader, writer, epoch, device, num_images=4):
    model.eval()
    print('Logging predictions...')
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            break  # We only need one batch

    # Convert to binary predictions
    pred_masks = (torch.sigmoid(outputs) > 0.5).float()

    # Create a grid of images
    img_grid  = torchvision.utils.make_grid(images[:num_images])
    mask_grid = torchvision.utils.make_grid(masks[:num_images])
    pred_grid = torchvision.utils.make_grid(pred_masks[:num_images])

    # Log to tensorboard
    writer.add_image('Images',           img_grid,  epoch)
    writer.add_image('True Masks',       mask_grid, epoch)
    writer.add_image('Predicted Masks',  pred_grid, epoch)

    model.train()

# ── Model instantiation ───────────────────────────────────────────────────────
_MODELS = {
    'simple': lambda: SimpleUNET(in_channels=1, out_channels=1),
    'unet':   lambda: UNet(in_channels=1, out_channels=1),
    'attn':   lambda: AttentionUNet(in_channels=1, out_channels=1),
    'hal':    lambda: HalfUNet(in_channels=1, out_channels=1, features=64, num_levels=5),
}

if model_name not in _MODELS:
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Choose from: {list(_MODELS.keys())}"
    )

model = _MODELS[model_name]().to(device)
model.apply(weights_init)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5*1e-4)

dummy_input = torch.randn(1, 1, dataset_height, dataset_width).to(device)
writer.add_graph(model, dummy_input)
writer.flush()
_initial_patience = patience
losses = {'train': [], 'val': [], 'best_val': float('inf')}

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model after each epoch encoding timestamp, dataset size and epoch number on the filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save(model.state_dict(), f'models/{model_name}_{dataset_height}x{dataset_width}_B{dataset_depth}_E{epoch+1}_T{timestamp}.pth')

    avg_train_loss = total_loss / len(train_loader)
    losses['train'].append(avg_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}, Dice Score: {1-avg_train_loss:.4f}')

    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, masks)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    losses['val'].append(avg_val_loss)
    print(f'Validation Loss: {avg_val_loss:.4f}, Validation Dice Score: {1-avg_val_loss:.4f}')

    # Tensorboard logging
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Loss/val',   avg_val_loss,   epoch)
    writer.add_scalar('Dice/train', 1 - avg_train_loss, epoch)
    writer.add_scalar('Dice/val',   1 - avg_val_loss,   epoch)

    # Log learning rate
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    # Log model weights and gradients
    for name, param in model.named_parameters():
        writer.add_histogram(f'Parameters/{name}', param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    # Log example predictions
    if epoch % 5 == 0:
        log_predictions(model, val_loader, writer, epoch, device)

    writer.flush()

    # Early stopping — stop when validation loss stops improving
    if avg_val_loss < losses['best_val']:
        losses['best_val'] = avg_val_loss
        patience = _initial_patience  # reset on improvement
    else:
        patience -= 1
        if patience == 0:
            print('Early stopping...')
            break
        print(f"No improvement in validation loss. Patience: {patience}/{_initial_patience}")

# Save the trained model
torch.save(model.state_dict(), 'models/unet.pth')
