import os
from PIL import Image

# Chemins des dossiers
imgs_dir = "data/imgs"
masks_dir = "data/masks"

# Liste des fichiers de masques
mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

# Parcourir chaque fichier masque
for mask_file in mask_files:
    mask_path = os.path.join(masks_dir, mask_file)
    img_file = mask_file.replace("_mask", "")
    img_path = os.path.join(imgs_dir, img_file)

    # Charger le masque
    with Image.open(mask_path) as mask:
        # Vérifier si le masque est entièrement noir
        if not mask.getbbox():  # La méthode getbbox() retourne None si tout est noir
            print(f"Supprime {mask_path} et {img_path} (masque vide)")

            # Supprimer le masque et l'image associée
            os.remove(mask_path)
            if os.path.exists(img_path):
                os.remove(img_path)

print("Traitement terminé.")
