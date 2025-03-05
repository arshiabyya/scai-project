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


dataset = pd.read_csv("_classes.csv")

print(dataset.info())