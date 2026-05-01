import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import MelanomaDataset
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
transform_base = transforms.Compose([
    transforms.ToTensor()
])
train_dataset= MelanomaDataset(TRAIN_DIR, transform = transform_base)
val_dataset = MelanomaDataset(TEST_DIR, transform = transform_base)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)
images_batch, labels_batch = next(iter(train_loader))
print(f"Forme du batch : {images_batch.shape}")
print(f"Forme des labels : {labels_batch.shape}")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(8):
    img = images_batch[i].permute(1, 2, 0).numpy()
    label = train_dataset.classes[labels_batch[i].item()]
    axes[i].imshow(img)
    axes[i].set_title(label)
    axes[i].axis("off")

plt.suptitle("Un batch du train loader")
plt.tight_layout()
plt.show()
from model import SimpleCNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleCNN(num_classes=2).to(device)
print(model)
def compter_parametres(model):
    total = sum(p.numel() for p in model.parameters())
    entrainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres total : {total:,}")
    print(f"Paramètres entraînables : {entrainables:,}")

compter_parametres(model)
import torch.nn as nn
import torch.optim as optim
import time
from train import train_one_epoch, evaluate

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

NUM_EPOCHS = 20
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("Lancement de l'entraînement...")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    duree = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
          f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
          f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
          f"{duree:.1f}s")
# ==========================================
# COURBES D'APPRENTISSAGE
# ==========================================
epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Courbe de la Loss
ax1.plot(epochs, history["train_loss"], label="Train", color='steelblue')
ax1.plot(epochs, history["val_loss"], label="Validation", color='tomato')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# Courbe de l'Accuracy
ax2.plot(epochs, history["train_acc"], label="Train", color='steelblue')
ax2.plot(epochs, history["val_acc"], label="Validation", color='tomato')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

from transforms import calculer_mean_std
from dataset import MelanomaDataset

# Calcul des stats sur le train set uniquement
dataset_stats = MelanomaDataset(TRAIN_DIR, 
                    transform=transforms.ToTensor())
MEAN, STD = calculer_mean_std(dataset_stats)
print(f"Mean : {MEAN}")
print(f"Std  : {STD}")