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

def save_slices_2d(nii_file_path, output_dir, global_counter, is_mask=False):
    """
    Charge un fichier .nii et sauvegarde ses slices dans un seul dossier avec des noms séquentiels.
    
    Args:
        nii_file_path (str): Chemin du fichier .nii.
        output_dir (str): Répertoire de sortie.
        global_counter (int): Compteur global pour nommer les fichiers de manière séquentielle.
        is_mask (bool): Indique si le fichier est un masque.
    
    Returns:
        int: Le compteur mis à jour après avoir ajouté les slices.
    """
    # Charge le fichier .nii
    nii_img = nib.load(nii_file_path)
    data = nii_img.get_fdata()
    
    # Parcourt les coupes axiales et les enregistre en PNG
    for i in range(data.shape[1]):  # Coupe axiale
        slice_2d = data[:, i, :]
        # Normalise les données pour les convertir en image 8 bits
        slice_2d_normalized = ((slice_2d - slice_2d.min()) / np.ptp(slice_2d) * 255).astype(np.uint8)
        # Définit le suffixe pour les masques
        suffix = "_mask" if is_mask else ""
        # Sauvegarde l'image avec un nom séquentiel basé sur le compteur global
        img = Image.fromarray(slice_2d_normalized)
        img.save(os.path.join(output_dir, f"image{global_counter}{suffix}.png"), format="PNG")
        global_counter += 1
    
    return global_counter

def process_data():
    """Traite les fichiers .nii pour les images et les masques."""
    # Prépare les dossiers imgs et masks
    prepare_directories()

    # Compteur global pour nommer les fichiers séquentiellement
    global_counter_imgs = 1
    global_counter_masks = 1

    # Traite les fichiers dans ./data/raw_data_imgs et ./data/raw_data_masks
    raw_data_dirs = {
        "./rawdata/rawimgs": ("./data/imgs", False),  # Pas un masque
        "./rawdata/rawmasks": ("./data/masks", True)  # Ajouter "_mask"
    }
    
    for raw_data_path, (output_path, is_mask) in raw_data_dirs.items():
        if os.path.exists(raw_data_path):
            for file_name in os.listdir(raw_data_path):
                if file_name.endswith(".nii"):
                    nii_file_path = os.path.join(raw_data_path, file_name)
                    if is_mask:
                        global_counter_masks = save_slices_2d(nii_file_path, output_path, global_counter_masks, is_mask)
                    else:
                        global_counter_imgs = save_slices_2d(nii_file_path, output_path, global_counter_imgs, is_mask)

if __name__ == "__main__":
    process_data()