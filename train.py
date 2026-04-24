import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Import de tes classes personnalisées
from dataset import MelanomaDataset
from model import SimpleCNN

# --- 1. CONFIGURATION DU MATÉRIEL ---
# On utilise la puce Apple Silicon (MPS) pour la rapidité
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Moteur de calcul : {device}")

# --- 2. PIPELINE DE DONNÉES (DATA PIPELINE) ---
TRAIN_DIR = "train"

# Transformations : Redimensionnement et conversion en Tenseur (Normalisation 0-1)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Chargement initial
dataset_complet = MelanomaDataset(root_dir=TRAIN_DIR, transform=transform)

# Division : 80% Train, 10% Validation, 10% Test
train_size = int(0.8 * len(dataset_complet))
val_size = int(0.1 * len(dataset_complet))
test_size = len(dataset_complet) - train_size - val_size 

# Le seed 42 permet d'avoir toujours la même division si on relance le code
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset_complet, [train_size, val_size, test_size], generator=generator
)

# Création des DataLoaders
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Dataset divisé en : Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")

# --- 3. INITIALISATION DU SYSTÈME D'APPRENTISSAGE ---
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss() # Le "Juge"
optimizer = optim.Adam(model.parameters(), lr=0.001) # Le "Coach"

# PARAMÈTRE DEMANDÉ : 20 EPOCHS
num_epochs = 20

# Listes pour stocker l'historique (pour les graphiques finaux)
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': []
}

print("🔥 Lancement de l'apprentissage...")

# --- 4. LA BOUCLE D'APPRENTISSAGE (TRAINING LOOP) ---
for epoch in range(num_epochs):
    
    # --- PHASE A : ENTRAÎNEMENT ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Le cycle de l'optimisation (Lecture 3 & 4)
        optimizer.zero_grad()               # Reset des gradients
        outputs = model(images)             # Forward pass
        loss = criterion(outputs, labels)   # Calcul de l'erreur
        loss.backward()                     # Rétropropagation (Gradients)
        optimizer.step()                    # Mise à jour des poids
        
        # Calcul des stats de l'epoch
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct / total

    # --- PHASE B : VALIDATION (L'EXAMEN BLANC) ---
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad(): # On économise la mémoire en ne calculant pas de gradients
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    epoch_val_loss = val_running_loss / len(val_dataset)
    epoch_val_acc = val_correct / val_total
    
    # Enregistrement de l'historique
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] -> "
          f"Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} || "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

# --- 5. VISUALISATION DES PERFORMANCES ---
print("📉 Génération des graphiques de performance...")

plt.figure(figsize=(12, 5))

# Graphique de la Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='blue', lw=2)
plt.plot(history['val_loss'], label='Val Loss', color='red', linestyle='--')
plt.title('Évolution de l\'Erreur (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Graphique de l'Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='green', lw=2)
plt.plot(history['val_acc'], label='Val Acc', color='orange', linestyle='--')
plt.title('Évolution de la Précision (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. SAUVEGARDE DU MODÈLE ---
torch.save(model.state_dict(), 'melanoma_model_v1.pth')
print("💾 Modèle sauvegardé sous 'melanoma_model_v1.pth'")