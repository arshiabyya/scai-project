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

transform = transforms.Compose([
        transforms.ToTensor(),
        # put other transforms here! https://pytorch.org/vision/stable/transforms.html#v2-api-ref
        # some useful ones might be resizing or cropping, consider also doing random augmentations
        # to make your model more robust.
    ])

dataset_path = "./train"

dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

print(len(dataset))

class ImageFolderDataset(Dataset):
    def __init__(self, image_paths):
        # here, you should set any important vairables you want to use in __len__ or __getitem__
        self.image_paths = image_paths

        




    # # this should just return the length of the dataset, without processing or opening any images
    # def __len__(self):
    #     return len(image_paths)

    # def __getitem__(self, idx):

    #     # we get the image path from the index
    #     image_path = image_paths[idx]

    #     # use the PIL class to get the image
    #     image = Image.open(image_path).convert('RGB')

    #     # run transformations on image
    #     image = self.transform(image)

    #     return image
    
# image_paths = ["./test.png"]

# # make or dataset class
# dataset = ImageFolderDataset(image_paths)

# # Create a dataloader, we can make a batch size and shuffle the image order
# dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# # Process the images
# for batch in dataloader:
#     # 'batch' is now a tensor of shape (batch_size, 3, 224, 224)
#     # You can perform further operations on this batch of tensors
#     print(f"Processed batch shape: {batch.shape}")

# print(f"Total number of images processed: {len(dataset)}")