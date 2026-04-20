import os
import cv2
import torch
from torch.utils.data import Dataset

class MelanomaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Chemin vers le dossier 'train' ou 'test'
        transform: Les transformations PyTorch à appliquer
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 1. On récupère les noms des dossiers (classes) en ignorant les fichiers cachés
        self.classes = sorted([d for d in os.listdir(root_dir) if not d.startswith('.')])
        
        # 2. On crée un dictionnaire pour transformer le nom en chiffre : {'Benign': 0, 'Malignant': 1}
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # 3. On parcourt les dossiers pour lister toutes les images
        for s_class in self.classes:
            class_path = os.path.join(root_dir, s_class)
            for img_name in os.listdir(class_path):
                if not img_name.startswith('.'): # Ignorer .DS_Store
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[s_class])

    def __len__(self):
        # Retourne le nombre total d'images
        return len(self.image_paths)

    def __getitem__(self, index):
        # Charger l'image
        img_path = self.image_paths[index]
        image = cv2.imread(img_path)
        
        # Conversion BGR (OpenCV) -> RGB (PyTorch/Standard)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Récupérer le label
        label = self.labels[index]
        
        # Appliquer les transformations (ex: redimensionner et convertir en Tenseur)
        if self.transform:
            image = self.transform(image)
            
        return image, label