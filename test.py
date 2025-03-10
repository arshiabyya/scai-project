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

print(dataset.info()) #print original dataset information

image_folder = "train"

images = []

for filename in dataset['filename']:
    img_path = os.path.join(image_folder, filename.strip())
    
    if os.path.exists(img_path):
        images.append(img_path)
    else:
        print(f"Image not found: {img_path}")

print(f"Loaded {len(images)} images.") #Loaded all 2000 images!

"""Clean images"""

dataset.columns = dataset.columns.str.strip()

filtered_dataset = dataset[
    (dataset['Unlabeled'] == 0) & 
    (dataset['angry'] == 0) & 
    (dataset['no clear emotion recognizable'] == 0) & 
    (dataset['sad'] == 0)
].reset_index(drop=True)

labels = filtered_dataset[['attentive', 'relaxed', 'uncomfortable']].values

"""Debugging to check error of balance of values"""

# print(filtered_dataset['attentive'].value_counts())
# print(filtered_dataset['relaxed'].value_counts())
# print(filtered_dataset['uncomfortable'].value_counts())

"""Create Dataloader"""

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

        print(f"Number of images: {len(self.image_paths)}")
        print(f"Number of labels: {len(self.labels)}")

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), #resize to fix tensor shape error
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
          # other transforms: https://pytorch.org/vision/stable/transforms.html#v2-api-ref
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label #return image and corresponding label for each image in the dataset

image_paths = [os.path.join(image_folder, filename.strip()) for filename in filtered_dataset['filename']]

filtered_dataset = ImageFolderDataset(image_paths, labels)

dataloader = DataLoader(filtered_dataset, batch_size=8, shuffle=True)

"""Testing and Training Loop"""
class ConvCAT (nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(32, 512)
        self.linear2 = nn.Linear(512, labels.shape[1])

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

model = ConvCAT()
for images, labels in dataloader: #loop through each image and labels
    outputs = model(images)  

"""Training Loop"""

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 10 #change for time constraints and increase to improve accuracy
for epoch in range(num_epochs):
    model.train()
    batch_loss = 0.0

    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        batch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {batch_loss/len(dataloader):.4f}")


"""Testing Method"""

def predict_single_image(image_path):
    image = Image.open(image_path).convert('RGB') #load in 1 test image
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)), #solve tensor shape error for different sizes
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    image = image.unsqueeze(0) #for one image, make sure it is only 1 dimension
    
    with torch.no_grad():
        model.eval()
        output = model(image)

    predicted_label = torch.sigmoid(output)
    print(f"Predicted probabilities: {predicted_label}") #show the probabilities for each lable
    predicted_label = predicted_label > 0.5  # Apply threshold
    
    label_names = ['Attentive', 'Relaxed', 'Uncomfortable']
    predicted_classes = [label_names[i] for i, value in enumerate(predicted_label[0]) if value.item()]
    
    print(f"Predicted labels: {predicted_classes}") #output prediction of labels with the highest label it predicts
    return predicted_classes

# Test with an image
test_image_path = 'test_cat_attentive.jpg' #testing cat image with attentive label
predict_single_image(test_image_path)

test_image_path = 'test_cat_relaxed.jpg' #testing cat image with relaxed label
predict_single_image(test_image_path)

test_image_path = 'test_cat_uncomfortable.jpg' #testing cat image with uncomfortable label
predict_single_image(test_image_path)