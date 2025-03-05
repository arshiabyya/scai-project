from PIL import Image
import torch
from torchvision.transforms import v2 as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim


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
          transforms.Resize((128, 128)), #resize for torch.tensor shape not being different
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

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# loop through each batch in dataloader
##for batch in dataloader:
    ##print(f"Processed batch shape: {batch.shape}")

##print(f"Total number of images processed: {len(dataset)}")

"""Testing and Training Loop"""
class ConvCAT (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size= 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size= 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size= 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size= 3, padding = 1)

        self.relu = nn.ReLU ()
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(64 * 32 * 32, 1028)
        self.linear2 = nn.Linear(1028 , 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.flatten(start_dim = 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = ConvCAT()
for batch in dataloader:
    output = model(batch)  # Pass a batch through the model

# Training Loop with simplified loss and accuracy calculation
criterion = nn.CrossEntropyLoss()  # For classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    batch_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Loop through batches in the dataloader
    for batch in dataloader:
        # Replace this with actual labels from your dataset
        labels = torch.randint(0, 10, (batch.size(0),))  # Dummy labels

        # Forward pass
        outputs = model(batch)

        # Loss calculation
        loss = criterion(outputs, labels)
        batch_loss += loss.item()

        # Accuracy calculation
        correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    # Print statistics after each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {batch_loss/len(dataloader):.4f}, Accuracy: {100 * correct_predictions/total_samples:.2f}%")
