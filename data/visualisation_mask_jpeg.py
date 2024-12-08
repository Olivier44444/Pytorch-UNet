import os
import numpy as np
from PIL import Image

def analyze_masks(mask_dir):
    all_classes = set()  # Utiliser un ensemble pour éviter les doublons

    # Parcourir tous les fichiers du répertoire
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith(('.jpeg', '.png', '.tiff')):  # Vérifier les formats d'image pris en charge
            mask_path = os.path.join(mask_dir, mask_file)

            # Charger le masque et extraire les classes uniques
            mask = np.array(Image.open(mask_path))
            mask_classes = np.unique(mask)
            all_classes.update(mask_classes)  # Ajouter les classes au set

            print(f"Masque : {mask_file} - Classes trouvées : {mask_classes}")

    # Afficher toutes les classes trouvées dans l'ensemble des masques
    print("\nClasses uniques dans tous les masques :", sorted(all_classes))

# Répertoire contenant les masques
mask_directory = "./data/masks"
analyze_masks(mask_directory) 
