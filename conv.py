import os
import shutil
import nibabel as nib
import numpy as np
from PIL import Image

def prepare_directories():
    # Vide les dossiers imgs et masks
    paths = ["./data/imgs", "./data/masks"]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)  # Supprime le dossier s'il existe
        os.makedirs(path)  # Crée un dossier vide

def save_slices_2d(nii_file_path, output_dir, prefix):
    # Charge le fichier .nii
    nii_img = nib.load(nii_file_path)
    data = nii_img.get_fdata()
    
    # Parcourt les coupes axiales et les enregistre en JPEG
    for i in range(data.shape[2]):  # Coupe axiale
        slice_2d = data[:, :, i]
        # Normalise les données pour les convertir en image 8 bits
        slice_2d_normalized = ((slice_2d - slice_2d.min()) / np.ptp(slice_2d) * 255).astype(np.uint8)
        # Sauvegarde l'image avec un nom basé sur le préfixe
        img = Image.fromarray(slice_2d_normalized)
        img.save(os.path.join(output_dir, f"{prefix}{i}.jpeg"))

def process_data():
    # Prépare les dossiers imgs et masks
    prepare_directories()

    # Traite les fichiers dans ./data/raw_data_imgs et ./data/raw_data_masks
    raw_data_dirs = {
        "./rawdata/rawimgs": ("./data/imgs", "image"),
        "./rawdata/rawmasks": ("./data/masks", "mask")
    }
    
    for raw_data_path, (output_path, prefix) in raw_data_dirs.items():
        if os.path.exists(raw_data_path):
            for file_name in os.listdir(raw_data_path):
                if file_name.endswith(".nii"):
                    nii_file_path = os.path.join(raw_data_path, file_name)
                    save_slices_2d(nii_file_path, output_path, prefix)

if __name__ == "__main__":
    process_data()