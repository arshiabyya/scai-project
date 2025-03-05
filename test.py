from PIL import Image
import torch
from torchvision.transforms import v2 as transforms
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

"""Create Dataloader"""

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

        self.transform = transforms.Compose([
          transforms.Resize((224, 224)), #resize for torch.tensor shape not being different
          transforms.ToTensor()
          # other transforms: https://pytorch.org/vision/stable/transforms.html#v2-api-ref
        ])

    def __len__(self):
        return len(self.image_paths) #return dataset length

    def __getitem__(self, idx):

        image_path = self.image_paths[idx] #get image path from given index

        image = Image.open(image_path).convert('RGB') #open image with standard RGB format for standard processing

        image = self.transform(image) #transform image

        return image
    
image_paths = images

dataset = ImageFolderDataset(image_paths) #initialize class object with each image

dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# loop through each batch in dataloader
for batch in dataloader:
    print(f"Processed batch shape: {batch.shape}")

print(f"Total number of images processed: {len(dataset)}")

"""Testing and Training Loop"""
