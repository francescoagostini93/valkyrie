import os
from torch.utils.data import Dataset, random_split
from PIL import Image

class PupilDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def get_train_val_datasets(image_dir, mask_dir, transform=None, val_split=0.2):
    dataset = PupilDataset(image_dir, mask_dir, transform)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])
