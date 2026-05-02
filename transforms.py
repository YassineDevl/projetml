import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def calculer_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, 
                       shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n_batches = 0

    for images, _ in loader:
        mean += images.mean(dim=[0, 2, 3])
        std  += images.std(dim=[0, 2, 3])
        n_batches += 1

    mean /= n_batches
    std  /= n_batches
    return mean.tolist(), std.tolist()
# Ces variables seront remplies après calcul dans main.py
MEAN = None
STD  = None

def get_transform_normalise(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
def get_train_transform_aug(mean, std):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])