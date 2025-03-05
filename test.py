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

"Import CSV and Image Folder"

dataset = pd.read_csv("_classes.csv")

print(dataset.info()) #Imported CSV with labels and filenames of images

image_folder = "train"

images = []

for filename in dataset['filename']: #for each file name in the dataset in column filename
    img_path = os.path.join(image_folder, filename.strip()) #join the path for the image and the filename together
    
    if os.path.exists(img_path): #if the path exists, do try except to open image and append image to list of images
        try:
            images.append(img_path)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    else:
        print(f"Image not found: {img_path}")

# Check the number of loaded images
print(f"Loaded {len(images)} images.") #Loaded all 2000 images!

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        # here, you should set any important vairables you want to use in __len__ or __getitem__
        self.image_paths = image_paths

        self.transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor()
          # put other transforms here! https://pytorch.org/vision/stable/transforms.html#v2-api-ref
          # some useful ones might be resizing or cropping, consider also doing random augmentations
          # to make your model more robust.
        ])


    # this should just return the length of the dataset, without processing or opening any images
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # we get the image path from the index
        image_path = self.image_paths[idx]

        # use the PIL class to get the image
        image = Image.open(image_path).convert('RGB')

        # run transformations on image
        image = self.transform(image)

        return image
    
image_paths = images

# make or dataset class
dataset = ImageFolderDataset(image_paths)

# Create a dataloader, we can make a batch size and shuffle the image order
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Process the images
for batch in dataloader:
    # 'batch' is now a tensor of shape (batch_size, 3, 224, 224)
    # You can perform further operations on this batch of tensors
    print(f"Processed batch shape: {batch.shape}")

print(f"Total number of images processed: {len(dataset)}")