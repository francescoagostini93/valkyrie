import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import PupilDataset
from torchvision import transforms

# Set up the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize the validation dataset
val_dataset = PupilDataset(image_dir='data/val_images', mask_dir='data/val_masks', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('models/unet.pth'))
model.eval()

# Evaluation
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)

        # Add metrics here to evaluate performance
        # e.g. compute the Dice coefficient between outputs and masks
