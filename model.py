import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__() # Indispensable en PyTorch
        
        # --- FEATURE EXTRACTOR ---
        # Couche 1 : Entrée 3 canaux (RGB), Sortie 16 canaux
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Couche 2 : Entrée 16, Sortie 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Couche 3 : Entrée 32, Sortie 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Max Pooling (divise la taille spatiale par 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- CLASSIFIER ---
        # Après 3 poolings, l'image 224x224 devient 28x28. 
        # On a 64 canaux, donc 64 * 28 * 28 = 50176 valeurs à aplatir.
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes) # Sortie : 2 valeurs (Benign, Malignant)

    def forward(self, x):
        # Passage dans les blocs convolutifs + activation ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Aplatissement (Flatten) : on garde la dimension du batch (x.size(0))
        # et on met tout le reste dans une seule dimension
        x = x.view(x.size(0), -1) 
        
        # Passage dans le classifieur
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # PAS DE SOFTMAX ICI !
        
        return x