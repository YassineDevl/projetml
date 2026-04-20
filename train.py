import os

TRAIN_DIR = "train" 

# On liste tout, mais on garde uniquement ce qui ne commence PAS par un point
classes = [nom for nom in os.listdir(TRAIN_DIR) if not nom.startswith('.')]
classes = sorted(classes) # On trie par ordre alphabétique

num_classes = len(classes)
print(f"Nombre de classes : {num_classes}")
print(f"Classes : {classes}")

for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    print(f"{classe} : {nb_images} images")