import os
import matplotlib.pyplot as plt
from PIL import Image

# On suppose que TRAIN_DIR et classes sont déjà définis
TRAIN_DIR = "train"
classes = ['Benign', 'Malignant']

plt.figure(figsize=(12, 8))

# On va parcourir les classes (i=0 pour Benign, i=1 pour Malignant)
for i, classe in enumerate(classes):
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    
    # On liste les fichiers et on prend les 2 premiers (en ignorant les fichiers cachés comme .DS_Store)
    images_noms = [n for n in os.listdir(chemin_classe) if not n.startswith('.')][:2]
    
    for j, image_nom in enumerate(images_noms):
        # Création de la grille : 2 lignes (une par classe), 2 colonnes (deux échantillons)
        # L'index du subplot commence à 1
        plt.subplot(2, 2, i * 2 + j + 1)
        
        # 1. Construction du chemin complet de manière robuste
        img_path = os.path.join(chemin_classe, image_nom)
        
        # 2. Chargement avec PIL (contrairement à cv2, PIL charge directement en RGB)
        img = Image.open(img_path)
        
        # 3. Affichage
        plt.imshow(img)
        plt.title(f"Classe : {classe}")
        plt.axis('off') # On cache les axes pour une meilleure lisibilité

plt.tight_layout()
plt.show()
counts = {}
for classe in classes :
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len([n for n in os.listdir(chemin_classe) if not n.startswith('.')])
    counts[classe] = nb_images

print(f"Distribution : {counts}")

plt.figure(figsize= (8, 5))
plt.bar(counts.keys(), counts.values(), color=['#3498db', '#e74c3c'])

plt.xlabel("Etat de la tumeur")
plt.ylabel("Nombre d'images")
plt.title("Distribution des classes (benign vs Malignant)")

for i, v in enumerate(counts.values()):
    plt.text(i, v+5, str(v), ha='center', fontweight='bold')

plt.show()
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import MelanomaDataset # On importe la classe que tu as créée !

print("\n--- Préparation des données pour PyTorch ---")

# 1. Définition des transformations (Leçon 3)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # On force une taille carrée pour le CNN
    transforms.ToTensor(),         # Conversion en Tenseur + Normalisation [0, 1]
])

# 2. Création du Dataset global
# Attention : Assure-toi que "TRAIN_DIR" est bien le chemin vers ton dossier
dataset_complet = MelanomaDataset(root_dir=TRAIN_DIR, transform=transform)
print(f"Taille du dataset complet : {len(dataset_complet)} images")

# 3. Séparation en Train / Val / Test (Leçon 4)
# On va utiliser une répartition classique : 80% Train, 10% Val, 10% Test
taille_totale = len(dataset_complet)
train_size = int(0.8 * taille_totale)
val_size = int(0.1 * taille_totale)
test_size = taille_totale - train_size - val_size 

# Le random_split mélange tout avant de couper, c'est indispensable !
train_dataset, val_dataset, test_dataset = random_split(
    dataset_complet, 
    [train_size, val_size, test_size]
)

print(f"Répartition -> Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# 4. Création des DataLoaders (La pompe à données)
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("✅ DataLoaders prêts !")