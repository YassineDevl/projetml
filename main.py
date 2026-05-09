import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import MelanomaDataset
from model import SimpleCNN, charger_resnet_gele
from train import train_one_epoch, evaluate
from transforms import (calculer_mean_std, get_transform_normalise,
                        get_train_transform_aug, get_val_transform)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
TEST_DIR   = os.path.join(BASE_DIR, "test")
device     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NUM_EPOCHS = 20

# FLAGS — mets True uniquement la partie que tu veux lancer
ENTRAINER_PARTIE5  = False
ENTRAINER_PARTIE6  = False
ENTRAINER_PARTIE7  = False
ENTRAINER_PARTIE8  = True
ENTRAINER_PARTIE9  = False
ENTRAINER_PARTIE11 = True
ENTRAINER_PARTIE12 = True


print(f"Device : {device}")

# ==========================================
# PARTIE 1 - EXPLORER LE DATASET
# ==========================================
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
dossier_benign  = os.path.join(TRAIN_DIR, "Benign")
premier_fichier = [f for f in os.listdir(dossier_benign) if f.endswith(".jpg")][0]
chemin          = os.path.join(dossier_benign, premier_fichier)

img_pil    = Image.open(chemin).convert("RGB")
to_tensor  = transforms.ToTensor()
img_tensor = to_tensor(img_pil)

print(f"\nTaille PIL : {img_pil.size}")
print(f"Type pixel PIL : {type(img_pil.getpixel((0,0)))}")
print(f"Forme tenseur : {img_tensor.shape}")
print(f"Min : {img_tensor.min():.4f}  Max : {img_tensor.max():.4f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img_tensor.permute(1, 2, 0).numpy())
axes[0].set_title("Image originale"); axes[0].axis("off")
axes[1].imshow(img_tensor[0].numpy(), cmap='Reds')
axes[1].set_title("Canal Rouge"); axes[1].axis("off")
axes[2].imshow(img_tensor[1].numpy(), cmap='Greens')
axes[2].set_title("Canal Vert"); axes[2].axis("off")
axes[3].imshow(img_tensor[2].numpy(), cmap='Blues')
axes[3].set_title("Canal Bleu"); axes[3].axis("off")
plt.suptitle(f"Forme du tenseur : {img_tensor.shape}", fontsize=13)
plt.tight_layout()
plt.show()

# ==========================================
# PARTIE 3 - DATASET ET DATALOADER
# ==========================================
transform_base = transforms.Compose([transforms.ToTensor()])
train_dataset  = MelanomaDataset(TRAIN_DIR, transform=transform_base)
val_dataset    = MelanomaDataset(TEST_DIR,  transform=transform_base)
train_loader   = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader     = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

images_batch, labels_batch = next(iter(train_loader))
print(f"\nForme du batch  : {images_batch.shape}")
print(f"Forme des labels: {labels_batch.shape}")

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()
for i in range(8):
    img   = images_batch[i].permute(1, 2, 0).numpy()
    label = train_dataset.classes[labels_batch[i].item()]
    axes[i].imshow(img); axes[i].set_title(label); axes[i].axis("off")
plt.suptitle("Un batch du train loader")
plt.tight_layout()
plt.show()

# ==========================================
# PARTIE 5 - CNN SIMPLE SANS NORMALISATION
# ==========================================
model = SimpleCNN(num_classes=2).to(device)
print(model)

def compter_parametres(m):
    total        = sum(p.numel() for p in m.parameters())
    entrainables = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Paramètres total       : {total:,}")
    print(f"Paramètres entraînables: {entrainables:,}")
    print(f"Paramètres gelés       : {total - entrainables:,}")

compter_parametres(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if ENTRAINER_PARTIE5:
    print("\nEntraînement SANS normalisation...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        duree = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train",      color='steelblue')
    ax1.plot(epochs, history["val_loss"],   label="Validation", color='tomato')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss — Sans normalisation")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train",      color='steelblue')
    ax2.plot(epochs, history["val_acc"],   label="Validation", color='tomato')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy — Sans normalisation")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_sans_norm.png", dpi=150)
    plt.show()

# ==========================================
# PARTIE 6 - NORMALISATION
# ==========================================
dataset_stats       = MelanomaDataset(TRAIN_DIR, transform=transforms.ToTensor())
MEAN, STD           = calculer_mean_std(dataset_stats)
print(f"\nMean : {MEAN}")
print(f"Std  : {STD}")

transform_normalise = get_transform_normalise(MEAN, STD)
train_dataset_norm  = MelanomaDataset(TRAIN_DIR, transform=transform_normalise)
val_dataset_norm    = MelanomaDataset(TEST_DIR,  transform=transform_normalise)
train_loader_norm   = DataLoader(train_dataset_norm, batch_size=32,
                                 shuffle=True,  num_workers=0)
val_loader_norm     = DataLoader(val_dataset_norm,   batch_size=32,
                                 shuffle=False, num_workers=0)

img_brute, _       = train_dataset[0]
img_norm, _        = train_dataset_norm[0]
img_norm_affichage = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_brute.permute(1, 2, 0).numpy())
axes[0].set_title(f"Sans normalisation\n"
                  f"Min={img_brute.min():.2f}  Max={img_brute.max():.2f}")
axes[0].axis("off")
axes[1].imshow(img_norm_affichage.permute(1, 2, 0).numpy())
axes[1].set_title(f"Avec normalisation\n"
                  f"Min={img_norm.min():.2f}  Max={img_norm.max():.2f}")
axes[1].axis("off")
plt.suptitle("Effet de la normalisation")
plt.tight_layout()
plt.show()

history_norm = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if ENTRAINER_PARTIE6:
    model_norm     = SimpleCNN(num_classes=2).to(device)
    optimizer_norm = optim.Adam(model_norm.parameters(), lr=1e-3)

    print("\nEntraînement AVEC normalisation...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model_norm, train_loader_norm, criterion, optimizer_norm, device)
        val_loss, val_acc = evaluate(
            model_norm, val_loader_norm, criterion, device)
        duree = time.time() - t0
        history_norm["train_loss"].append(train_loss)
        history_norm["val_loss"].append(val_loss)
        history_norm["train_acc"].append(train_acc)
        history_norm["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["val_loss"],      label="Sans norm", color='tomato')
    ax1.plot(epochs, history_norm["val_loss"], label="Avec norm", color='steelblue')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Comparaison Val Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["val_acc"],       label="Sans norm", color='tomato')
    ax2.plot(epochs, history_norm["val_acc"],  label="Avec norm", color='steelblue')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Comparaison Val Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_comparaison_norm.png", dpi=150)
    plt.show()

# ==========================================
# PARTIE 7 - DATA AUGMENTATION
# ==========================================
train_transform_aug = get_train_transform_aug(MEAN, STD)
val_transform       = get_val_transform(MEAN, STD)

train_dataset_aug = MelanomaDataset(TRAIN_DIR, transform=train_transform_aug)
val_dataset_aug   = MelanomaDataset(TEST_DIR,  transform=val_transform)
train_loader_aug  = DataLoader(train_dataset_aug, batch_size=32,
                               shuffle=True,  num_workers=0)
val_loader_aug    = DataLoader(val_dataset_aug,   batch_size=32,
                               shuffle=False, num_workers=0)

chemin_exemple  = os.path.join(TRAIN_DIR, "Benign",
    [f for f in os.listdir(os.path.join(TRAIN_DIR, "Benign"))
     if f.endswith(".jpg")][0])
img_pil_exemple = Image.open(chemin_exemple).convert("RGB")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
img_originale = transforms.ToTensor()(img_pil_exemple)
axes[0].imshow(img_originale.permute(1, 2, 0).numpy())
axes[0].set_title("Originale"); axes[0].axis("off")
for i in range(1, 8):
    img_aug       = train_transform_aug(img_pil_exemple)
    img_affichage = (img_aug - img_aug.min()) / (img_aug.max() - img_aug.min())
    axes[i].imshow(img_affichage.permute(1, 2, 0).numpy())
    axes[i].set_title(f"Augmentation {i}"); axes[i].axis("off")
plt.suptitle("Effet de la data augmentation — même image, 7 versions")
plt.tight_layout()
plt.show()

history_aug = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if ENTRAINER_PARTIE7:
    model_aug     = SimpleCNN(num_classes=2).to(device)
    optimizer_aug = optim.Adam(model_aug.parameters(), lr=1e-3)

    print("\nEntraînement AVEC augmentation...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model_aug, train_loader_aug, criterion, optimizer_aug, device)
        val_loss, val_acc = evaluate(
            model_aug, val_loader_aug, criterion, device)
        duree = time.time() - t0
        history_aug["train_loss"].append(train_loss)
        history_aug["val_loss"].append(val_loss)
        history_aug["train_acc"].append(train_acc)
        history_aug["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history_norm["val_loss"], label="Sans aug", color='tomato')
    ax1.plot(epochs, history_aug["val_loss"],  label="Avec aug", color='steelblue')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Comparaison Val Loss — Augmentation")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history_norm["val_acc"],  label="Sans aug", color='tomato')
    ax2.plot(epochs, history_aug["val_acc"],   label="Avec aug", color='steelblue')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Comparaison Val Accuracy — Augmentation")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_comparaison_aug.png", dpi=150)
    plt.show()

# ==========================================
# PARTIE 8 - COMPARAISON OPTIMISEURS
# ==========================================
configs = [
    {"nom": "Adam lr=1e-3",        "optim": "adam", "lr": 1e-3, "momentum": 0.0},
    {"nom": "Adam lr=1e-4",        "optim": "adam", "lr": 1e-4, "momentum": 0.0},
    {"nom": "SGD lr=1e-2 mom=0.9", "optim": "sgd",  "lr": 1e-2, "momentum": 0.9},
]
resultats = {}

if ENTRAINER_PARTIE8:
    for config in configs:
        print(f"\nEntraînement : {config['nom']}")
        m = SimpleCNN(num_classes=2).to(device)
        if config["optim"] == "adam":
            opt = optim.Adam(m.parameters(), lr=config["lr"])
        else:
            opt = optim.SGD(m.parameters(), lr=config["lr"],
                            momentum=config["momentum"])
        hist = {"val_acc": [], "val_loss": []}
        for epoch in range(1, NUM_EPOCHS + 1):
            train_one_epoch(m, train_loader_aug, criterion, opt, device)
            val_loss, val_acc = evaluate(m, val_loader_aug, criterion, device)
            hist["val_acc"].append(val_acc)
            hist["val_loss"].append(val_loss)
            print(f"  Epoch {epoch:2d} | Val acc {val_acc:.3f}")
        resultats[config["nom"]] = hist

    epochs   = range(1, NUM_EPOCHS + 1)
    couleurs = ['steelblue', 'tomato', 'green']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for (nom, hist), couleur in zip(resultats.items(), couleurs):
        ax1.plot(epochs, hist["val_loss"], label=nom, color=couleur)
        ax2.plot(epochs, hist["val_acc"],  label=nom, color=couleur)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Comparaison Val Loss — Optimiseurs")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Comparaison Val Accuracy — Optimiseurs")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_optimiseurs.png", dpi=150)
    plt.show()

    print("\n=== TABLEAU RÉCAPITULATIF ===")
    print(f"{'Configuration':<25} {'Meilleure Val Acc':>18}")
    print("-" * 45)
    for nom, hist in resultats.items():
        print(f"{nom:<25} {max(hist['val_acc']):>18.3f}")

# ==========================================
# PARTIE 9 - LEARNING RATE SCHEDULER
# ==========================================
history_sched = {"train_loss": [], "val_loss": [],
                 "train_acc": [],  "val_acc": []}
lrs = []

if ENTRAINER_PARTIE9:
    model_sched     = SimpleCNN(num_classes=2).to(device)
    optimizer_sched = optim.Adam(model_sched.parameters(), lr=1e-3)
    scheduler       = torch.optim.lr_scheduler.StepLR(
                        optimizer_sched, step_size=7, gamma=0.1)

    print("\nEntraînement AVEC scheduler...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model_sched, train_loader_aug, criterion, optimizer_sched, device)
        val_loss, val_acc = evaluate(
            model_sched, val_loader_aug, criterion, device)
        duree = time.time() - t0
        lrs.append(optimizer_sched.param_groups[0]['lr'])
        scheduler.step()
        history_sched["train_loss"].append(train_loss)
        history_sched["val_loss"].append(val_loss)
        history_sched["train_acc"].append(train_acc)
        history_sched["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | "
              f"LR {lrs[-1]:.2e} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history_sched["train_acc"], label="Train",      color='steelblue')
    ax1.plot(epochs, history_sched["val_acc"],   label="Validation", color='tomato')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy — Avec scheduler")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1)
    ax2.plot(epochs, lrs, color='green', marker='o', markersize=4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
    ax2.set_title("Évolution du Learning Rate")
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("courbes_scheduler.png", dpi=150)
    plt.show()

# ==========================================
# PARTIE 11 - TRANSFER LEARNING RESNET18
# ==========================================

# Stats ImageNet — obligatoires pour ResNet pré-entraîné
MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET  = [0.229, 0.224, 0.225]

train_loader_resnet = DataLoader(
    MelanomaDataset(TRAIN_DIR,
                    transform=get_train_transform_aug(MEAN_IMAGENET, STD_IMAGENET)),
    batch_size=32, shuffle=True, num_workers=0)
val_loader_resnet = DataLoader(
    MelanomaDataset(TEST_DIR,
                    transform=get_val_transform(MEAN_IMAGENET, STD_IMAGENET)),
    batch_size=32, shuffle=False, num_workers=0)

history_resnet = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if ENTRAINER_PARTIE11:
    resnet = charger_resnet_gele(num_classes=2).to(device)

    print("\n--- ResNet18 gelé ---")
    compter_parametres(resnet)

    # On entraîne UNIQUEMENT la tête fc
    optimizer_resnet = optim.Adam(resnet.fc.parameters(), lr=1e-3)

    print("\nEntraînement ResNet18 (backbone gelé)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            resnet, train_loader_resnet, criterion, optimizer_resnet, device)
        val_loss, val_acc = evaluate(
            resnet, val_loader_resnet, criterion, device)
        duree = time.time() - t0
        history_resnet["train_loss"].append(train_loss)
        history_resnet["val_loss"].append(val_loss)
        history_resnet["train_acc"].append(train_acc)
        history_resnet["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history_resnet["train_loss"], label="Train",      color='steelblue')
    ax1.plot(epochs, history_resnet["val_loss"],   label="Validation", color='tomato')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss — ResNet18 gelé")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history_resnet["train_acc"], label="Train",      color='steelblue')
    ax2.plot(epochs, history_resnet["val_acc"],   label="Validation", color='tomato')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy — ResNet18 gelé")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_resnet_gele.png", dpi=150)
    plt.show()

# ==========================================
# PARTIE 12 - FINE-TUNING PARTIEL
# ==========================================
history_ft = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

if ENTRAINER_PARTIE12:
    from torchvision import models

    resnet_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in resnet_ft.parameters():
        param.requires_grad = False

    resnet_ft.fc = nn.Linear(resnet_ft.fc.in_features, 2)
    resnet_ft = resnet_ft.to(device)

    # Dégeler layer4 + fc
    for param in resnet_ft.layer4.parameters():
        param.requires_grad = True

    print("\n--- ResNet18 fine-tuning (layer4 + fc) ---")
    compter_parametres(resnet_ft)

    # LR différent pour layer4 et fc
    optimizer_ft = optim.Adam([
        {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
        {"params": resnet_ft.fc.parameters(),     "lr": 1e-3},
    ])

    print("\nEntraînement ResNet18 fine-tuning...")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            resnet_ft, train_loader_resnet, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate(
            resnet_ft, val_loader_resnet, criterion, device)
        duree = time.time() - t0
        history_ft["train_loss"].append(train_loss)
        history_ft["val_loss"].append(val_loss)
        history_ft["train_acc"].append(train_acc)
        history_ft["val_acc"].append(val_acc)
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss train {train_loss:.4f} | Loss val {val_loss:.4f} | "
              f"Acc train {train_acc:.3f} | Acc val {val_acc:.3f} | {duree:.1f}s")

    epochs = range(1, NUM_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history_resnet["val_loss"], label="ResNet gelé",    color='tomato')
    ax1.plot(epochs, history_ft["val_loss"],     label="Fine-tuning",    color='steelblue')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Loss")
    ax1.set_title("Comparaison Val Loss — Fine-tuning")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history_resnet["val_acc"],  label="ResNet gelé",    color='tomato')
    ax2.plot(epochs, history_ft["val_acc"],      label="Fine-tuning",    color='steelblue')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Comparaison Val Accuracy — Fine-tuning")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("courbes_finetuning.png", dpi=150)
    plt.show()
    # ==========================================
# PARTIE 13 - MATRICE DE CONFUSION
# ==========================================
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

ANALYSER_ERREURS = False

if ANALYSER_ERREURS:
    # Utiliser le meilleur modèle = resnet_ft
    # Si tu n'as pas resnet_ft en mémoire, relance avec ENTRAINER_PARTIE12=True

    # 1. Collecter toutes les prédictions sur la validation
    resnet_ft.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader_resnet:
            images = images.to(device)
            outputs = resnet_ft(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 2. Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes))

    # Affichage visuel
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(train_dataset.classes)
    ax.set_yticklabels(train_dataset.classes)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion — ResNet fine-tuné")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=16)
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=150)
    plt.show()

    # 3. Afficher 8 exemples mal classés
    erreurs_images  = []
    erreurs_vrais   = []
    erreurs_predits = []

    resnet_ft.eval()
    with torch.no_grad():
        for idx in range(len(val_dataset_aug)):
            img, label = val_dataset_aug[idx]
            output = resnet_ft(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).item()
            if pred != label and len(erreurs_images) < 8:
                erreurs_images.append(img)
                erreurs_vrais.append(label)
                erreurs_predits.append(pred)
            if len(erreurs_images) == 8:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(8):
        img_aff = (erreurs_images[i] - erreurs_images[i].min()) / \
                  (erreurs_images[i].max() - erreurs_images[i].min())
        axes[i].imshow(img_aff.permute(1, 2, 0).numpy())
        vrai   = train_dataset.classes[erreurs_vrais[i]]
        predit = train_dataset.classes[erreurs_predits[i]]
        axes[i].set_title(f"Vrai : {vrai}\nPrédit : {predit}", color='red')
        axes[i].axis("off")
    plt.suptitle("8 exemples mal classés")
    plt.tight_layout()
    plt.savefig("erreurs.png", dpi=150)
    plt.show()

    # 4. Afficher 8 exemples bien classés
    bons_images  = []
    bons_labels  = []

    resnet_ft.eval()
    with torch.no_grad():
        for idx in range(len(val_dataset_aug)):
            img, label = val_dataset_aug[idx]
            output = resnet_ft(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).item()
            if pred == label and len(bons_images) < 8:
                bons_images.append(img)
                bons_labels.append(label)
            if len(bons_images) == 8:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(8):
        img_aff = (bons_images[i] - bons_images[i].min()) / \
                  (bons_images[i].max() - bons_images[i].min())
        axes[i].imshow(img_aff.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Vrai : {train_dataset.classes[bons_labels[i]]}",
                          color='green')
        axes[i].axis("off")
    plt.suptitle("8 exemples bien classés")
    plt.tight_layout()
    plt.savefig("bons_exemples.png", dpi=150)
    plt.show()
# ==========================================
# PARTIE 13 - MATRICE DE CONFUSION
# ==========================================
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

ANALYSER_ERREURS = True

if ANALYSER_ERREURS:
    # Utiliser le meilleur modèle = resnet_ft
    # Si tu n'as pas resnet_ft en mémoire, relance avec ENTRAINER_PARTIE12=True

    # 1. Collecter toutes les prédictions sur la validation
    resnet_ft.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader_resnet:
            images = images.to(device)
            outputs = resnet_ft(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 2. Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatrice de confusion :")
    print(cm)
    print("\nRapport de classification :")
    print(classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes))

    # Affichage visuel
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(train_dataset.classes)
    ax.set_yticklabels(train_dataset.classes)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion — ResNet fine-tuné")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=16)
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=150)
    plt.show()

    # 3. Afficher 8 exemples mal classés
    erreurs_images  = []
    erreurs_vrais   = []
    erreurs_predits = []

    resnet_ft.eval()
    with torch.no_grad():
        for idx in range(len(val_dataset_aug)):
            img, label = val_dataset_aug[idx]
            output = resnet_ft(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).item()
            if pred != label and len(erreurs_images) < 8:
                erreurs_images.append(img)
                erreurs_vrais.append(label)
                erreurs_predits.append(pred)
            if len(erreurs_images) == 8:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(8):
        img_aff = (erreurs_images[i] - erreurs_images[i].min()) / \
                  (erreurs_images[i].max() - erreurs_images[i].min())
        axes[i].imshow(img_aff.permute(1, 2, 0).numpy())
        vrai   = train_dataset.classes[erreurs_vrais[i]]
        predit = train_dataset.classes[erreurs_predits[i]]
        axes[i].set_title(f"Vrai : {vrai}\nPrédit : {predit}", color='red')
        axes[i].axis("off")
    plt.suptitle("8 exemples mal classés")
    plt.tight_layout()
    plt.savefig("erreurs.png", dpi=150)
    plt.show()

    # 4. Afficher 8 exemples bien classés
    bons_images  = []
    bons_labels  = []

    resnet_ft.eval()
    with torch.no_grad():
        for idx in range(len(val_dataset_aug)):
            img, label = val_dataset_aug[idx]
            output = resnet_ft(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).item()
            if pred == label and len(bons_images) < 8:
                bons_images.append(img)
                bons_labels.append(label)
            if len(bons_images) == 8:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i in range(8):
        img_aff = (bons_images[i] - bons_images[i].min()) / \
                  (bons_images[i].max() - bons_images[i].min())
        axes[i].imshow(img_aff.permute(1, 2, 0).numpy())
        axes[i].set_title(f"Vrai : {train_dataset.classes[bons_labels[i]]}",
                          color='green')
        axes[i].axis("off")
    plt.suptitle("8 exemples bien classés")
    plt.tight_layout()
    plt.savefig("bons_exemples.png", dpi=150)
    plt.show()