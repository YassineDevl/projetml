import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# ==========================================
# PARTIE 1 - EXPLORER LE DATASET
# ==========================================

# Lister les classes (on exclut .DS_Store et autres fichiers Mac)
classes = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
print("Classes :", classes)
print("Nombre de classes :", len(classes))

for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len([f for f in os.listdir(chemin_classe) if f.endswith(".jpg")])
    print(f"  {classe} : {nb_images} images")

# Afficher des exemples d'images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

i = 0
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    fichiers = [f for f in os.listdir(chemin_classe) if f.endswith(".jpg")][:4]
    
    for nom_fichier in fichiers:
        chemin_image = os.path.join(chemin_classe, nom_fichier)
        img = Image.open(chemin_image).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(classe)
        axes[i].axis("off")
        i += 1

plt.suptitle("Exemples d'images par classe", fontsize=14)
plt.tight_layout()
plt.show()

# Diagramme de distribution
counts = {}
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    counts[classe] = len([f for f in os.listdir(chemin_classe) if f.endswith(".jpg")])

plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=['steelblue', 'tomato'])
plt.title("Distribution des classes")
plt.xlabel("Classe")
plt.ylabel("Nombre d'images")
plt.show()

# ==========================================
# PARTIE 2 - UNE IMAGE POUR PYTORCH
# ==========================================

# Charger une image
dossier_benign = os.path.join(TRAIN_DIR, "Benign")
premier_fichier = [f for f in os.listdir(dossier_benign) if f.endswith(".jpg")][0]
chemin = os.path.join(dossier_benign, premier_fichier)

img_pil = Image.open(chemin).convert("RGB")
print(f"\nTaille PIL : {img_pil.size}")
print(f"Type pixel PIL : {type(img_pil.getpixel((0,0)))}")

# Convertir en tenseur
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_pil)

print(f"Forme tenseur : {img_tensor.shape}")
print(f"Min : {img_tensor.min():.4f}")
print(f"Max : {img_tensor.max():.4f}")

# Visualiser les 3 canaux RGB
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes = axes.flatten()

# Image originale
axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
axes[0].set_title("Image originale")
axes[0].axis("off")

# Canal Rouge
axes[1].imshow(img_tensor[0].numpy(), cmap='Reds')
axes[1].set_title("Canal Rouge")
axes[1].axis("off")

# Canal Vert
axes[2].imshow(img_tensor[1].numpy(), cmap='Greens')
axes[2].set_title("Canal Vert")
axes[2].axis("off")

# Canal Bleu
axes[3].imshow(img_tensor[2].numpy(), cmap='Blues')
axes[3].set_title("Canal Bleu")
axes[3].axis("off")

plt.suptitle(f"Forme du tenseur : {img_tensor.shape}", fontsize=13)
plt.tight_layout()
plt.show()