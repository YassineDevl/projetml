import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

# Import de tes classes personnalisées (doivent être dans le même dossier)
from dataset import MelanomaDataset
from model import SimpleCNN

# ==========================================
# 1. CONFIGURATION DU MATÉRIEL (HARDWARE)
# ==========================================
# On utilise la puce Apple Silicon (MPS) ou le CPU par défaut
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Moteur de calcul activé : {device}")

# ==========================================
# 2. PIPELINE DE DONNÉES (DATA PREPARATION)
# ==========================================
TRAIN_DIR = "train"

# Standardisation des images : même taille et valeurs entre 0 et 1
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Moyenne pour R, G, B
                         std=[0.229, 0.224, 0.225])   # Écart-type pour R, G, B
])


# Chargement du dataset via ta classe personnalisée
dataset_complet = MelanomaDataset(root_dir=TRAIN_DIR, transform=transform)

# Découpage : 80% Entraînement, 10% Validation, 10% Test
train_size = int(0.8 * len(dataset_complet))
val_size = int(0.1 * len(dataset_complet))
test_size = len(dataset_complet) - train_size - val_size 

# Le seed 42 garantit que le mélange est identique à chaque lancement
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    dataset_complet, [train_size, val_size, test_size], generator=generator
)

# Création des Loaders (les pompes qui envoient les images par paquets de 32)
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📊 Données prêtes : Train({len(train_dataset)}), Val({len(val_dataset)}), Test({len(test_dataset)})")

# ==========================================
# 3. INITIALISATION DU "CERVEAU"
# ==========================================
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss() # Le juge (calcule l'erreur)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Le coach (corrige les poids)

# On lance l'apprentissage sur 20 cycles complets (Epochs)
num_epochs = 20
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# ==========================================
# 4. LA BOUCLE D'APPRENTISSAGE (TRAINING LOOP)
# ==========================================
print("🔥 Lancement de l'entraînement...")

for epoch in range(num_epochs):
    
    # --- PHASE D'ENTRAÎNEMENT ---
    model.train() # Mode "Apprentissage"
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        # Envoi des données sur la puce MPS du Mac
        images, labels = images.to(device), labels.to(device)
        
        # Le cycle sacré du Deep Learning
        optimizer.zero_grad()               # 1. Effacer les erreurs passées
        outputs = model(images)             # 2. Le modèle devine (Forward)
        loss = criterion(outputs, labels)   # 3. Calculer la punition (Loss)
        loss.backward()                     # 4. Chercher les responsables (Backprop)
        optimizer.step()                    # 5. Corriger les neurones (Step)
        
        # Stats
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct / total

    # --- PHASE DE VALIDATION ---
    model.eval() # Mode "Examen blanc"
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    
    with torch.no_grad(): # Désactive les calculs mathématiques lourds
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
    
    # Sauvegarde des scores pour les graphiques
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

# ==========================================
# 5. VISUALISATION DES RÉSULTATS
# ==========================================
print("📉 Création des courbes de performance...")
plt.figure(figsize=(12, 5))

# Graphique de la Loss
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Erreur (Loss)')
plt.legend()

# Graphique de l'Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Précision (Accuracy)')
plt.legend()

plt.show()

# Sauvegarde finale
torch.save(model.state_dict(), 'melanoma_model.pth')
print("💾 Modèle sauvegardé avec succès.")