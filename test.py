# PIL, or Python Image Library, helps us with processing the image file
from PIL import Image

# torch is used for tensors and machine learning
import torch
# torchvision.transforms, or torchvision.trasnforms.v2, allow us to transform each image, for example
# resize, crop, rotate, convert to tensor, blur, etc.
from torchvision.transforms import v2 as transforms

# torch Dataset and Dataloader allow us to effiecently process our dataset.
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pandas as pd

import os


dataset = pd.read_csv("_classes.csv")

print(dataset.info())

image_folder = "train"

images = []

for filename in dataset['filename']:
    # Create the full path to the image
    img_path = os.path.join(image_folder, filename.strip())
    
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path)
            images.append(image)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    else:
        print(f"Image not found: {img_path}")

# Check the number of loaded images
print(f"Loaded {len(images)} images.")