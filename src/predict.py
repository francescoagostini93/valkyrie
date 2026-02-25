import os
import torch
from model import SimpleUNET
from PIL import Image
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = SimpleUNET(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load('models/unet.pth'))
model.eval()

# Transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load image and run inference
def predict(image_path, save_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_mask = (output > 0.5).float()  # Threshold to create binary mask

    # Save the predicted mask
    predicted_mask = predicted_mask.squeeze().cpu().numpy()
    result = Image.fromarray((predicted_mask * 255).astype('uint8'))
    result.save(save_path)

# Load each image from data/test_images/images and predict the mask
# Save the predicted mask in data/test_images/masks

for image in os.listdir('data/test_images/images'):
    image_path = os.path.join('data','test_images','images', image)
    save_path = os.path.join('data','test_images','masks', image)
    print(f'Predicting mask for {image_path}')
    predict(image_path, save_path)
