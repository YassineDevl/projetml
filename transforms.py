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