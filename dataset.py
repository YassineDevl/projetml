import os
from PIL import Image
from torch.utils.data import Dataset

class MelanomaDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Étape 1 : lister les classes (exclut .DS_Store)
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        # Étape 2 : associer chaque classe à un entier
        # {"Benign": 0, "Malignant": 1}
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Étape 3 : construire la liste de tous les (chemin, label)
        for classe in self.classes:
            label = self.class_to_idx[classe]
            dossier = os.path.join(data_dir, classe)
            for nom_fichier in os.listdir(dossier):
                if nom_fichier.endswith(".jpg"):
                    chemin = os.path.join(dossier, nom_fichier)
                    self.samples.append((chemin, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chemin, label = self.samples[idx]
        image = Image.open(chemin).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label