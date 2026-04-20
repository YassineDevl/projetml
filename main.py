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