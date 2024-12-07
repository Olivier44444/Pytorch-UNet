import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image

def prepare_directories():
    """Vide les dossiers imgs et masks."""
    paths = ["./data/imgs", "./data/masks"]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)  # Supprime le dossier s'il existe
        os.makedirs(path)  # Crée un dossier vide

def save_slices_3d(nii_file_path_img, nii_file_path_mask, output_dir_img, output_dir_mask, global_counter):
    """
    Charge les fichiers .nii pour les images et les masques et sauvegarde les slices dans les dossiers appropriés.

    Args:
        nii_file_path_img (str): Chemin du fichier .nii d'image.
        nii_file_path_mask (str): Chemin du fichier .nii de masque.
        output_dir_img (str): Répertoire des images.
        output_dir_mask (str): Répertoire des masques.
        global_counter (int): Compteur global pour nommer les fichiers de manière séquentielle.

    Returns:
        int: Le compteur mis à jour après avoir ajouté les slices.
    """
    # Charge les fichiers .nii
    nii_img = nib.load(nii_file_path_img)
    data_img = nii_img.get_fdata()

    nii_mask = nib.load(nii_file_path_mask)
    data_mask = nii_mask.get_fdata()

    # Découpe suivant les trois axes
    axes = [0, 1, 2]  # Axes x, y, z
    for axis in axes:
        for i in range(data_img.shape[axis]):
            # Récupère une slice selon l'axe spécifié
            slice_2d_img = np.take(data_img, i, axis=axis)
            slice_2d_mask = np.take(data_mask, i, axis=axis)

            # Normalise les données de l'image pour les convertir en image 8 bits
            slice_2d_img_normalized = ((slice_2d_img - slice_2d_img.min()) / np.ptp(slice_2d_img) * 255).astype(np.uint8)

            # Sauvegarde de l'image
            img = Image.fromarray(slice_2d_img_normalized)
            img_path = os.path.join(output_dir_img, f"image{global_counter}.png")

            # Si le masque n'est pas vide, on enregistre le masque et l'image associée
            if np.any(slice_2d_mask):  # Si le masque n'est pas vide
                # Sauvegarde du masque
                mask = Image.fromarray(slice_2d_mask.astype(np.uint8))
                mask_path = os.path.join(output_dir_mask, f"image{global_counter}_mask.png")
                mask.save(mask_path, format="PNG")

                # Sauvegarde de l'image associée
                img.save(img_path, format="PNG")
            # Si le masque est vide, on ne sauvegarde ni l'image ni le masque

            global_counter += 1

    return global_counter

def process_data():
    """Traite les fichiers .nii pour les images et les masques."""
    # Prépare les dossiers imgs et masks
    prepare_directories()

    # Compteur global pour nommer les fichiers séquentiellement
    global_counter = 1

    # Traite les fichiers dans ./data/raw_data_imgs et ./data/raw_data_masks
    raw_data_dirs = {
        "./rawdata/rawimgs": "./rawdata/rawmasks",  # Associe les images et masques
    }

    for raw_data_path_img, raw_data_path_mask in raw_data_dirs.items():
        if os.path.exists(raw_data_path_img) and os.path.exists(raw_data_path_mask):
            for file_name_img in os.listdir(raw_data_path_img):
                if file_name_img.endswith(".nii"):
                    # Crée les chemins des fichiers image et masque
                    nii_file_path_img = os.path.join(raw_data_path_img, file_name_img)
                    nii_file_path_mask = os.path.join(raw_data_path_mask, file_name_img.replace(".nii", "_mask.nii"))

                    if os.path.exists(nii_file_path_mask):
                        # Sauve les slices de l'image et du masque
                        global_counter = save_slices_3d(nii_file_path_img, nii_file_path_mask, "./data/imgs", "./data/masks", global_counter)

if __name__ == "__main__":
    process_data()
