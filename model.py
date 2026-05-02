

import torch.nn as nn

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classificateur = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        x = self.bloc3(x)
        x = self.classificateur(x)
        return x
import torch.nn as nn
from torchvision import models

def charger_resnet_gele(num_classes):
    # Charger ResNet18 pré-entraîné sur ImageNet
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Geler TOUS les paramètres du backbone
    for param in resnet.parameters():
        param.requires_grad = False
    
    # Remplacer la tête : Linear(512 → 1000) par Linear(512 → num_classes)
    in_features = resnet.fc.in_features  # = 512
    resnet.fc   = nn.Linear(in_features, num_classes)
    
    return resnet
