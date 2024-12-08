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

def remap_classes(slice_2d):
    """
    Remappe les classes originales (0, 1, 2, 3, 4) en nouvelles valeurs (0, 63, 127, 191, 255).
    """
    mapping = {
        0: 0,     # Fond
        1: 63,    # Rein droit
        2: 127,   # Rein gauche
        3: 191,   # Rate
        4: 255    # Foie
    }
    # Applique le mappage à chaque pixel
    remapped = np.zeros_like(slice_2d, dtype=np.uint8)
    for original, new_value in mapping.items():
        remapped[slice_2d == original] = new_value
    return remapped

def save_slices_2d(nii_file_path, output_dir, prefix, is_mask, start_index):
    # Charge le fichier .nii
    nii_img = nib.load(nii_file_path)
    data = nii_img.get_fdata(),(256,256),mode='reflect',anti_aliasing=True

    for i in range(data.shape[2]):  # Parcourt les coupes axiales
        slice_2d = data[:, :, i]

        if is_mask:
            # Remappe les classes avant d'enregistrer le masque
            slice_2d_remapped = remap_classes(slice_2d.astype(np.uint8))
            img = Image.fromarray(slice_2d_remapped, mode='L')  # Mode 'L' pour niveaux de gris
        else:
            # Normalise les images pour conversion en 8 bits
            slice_2d_normalized = ((slice_2d - slice_2d.min()) / np.ptp(slice_2d) * 255).astype(np.uint8)
            img = Image.fromarray(slice_2d_normalized)

        # Détermine le suffixe en fonction du type
        suffix = "_mask" if is_mask else ""
        img.save(os.path.join(output_dir, f"{prefix}{start_index}{suffix}.png"), format="PNG")
        start_index += 1

    return start_index

def process_data():
    # Prépare les dossiers imgs et masks
    prepare_directories()

    # Compteurs globaux pour nommer les slices
    img_index = 1
    mask_index = 1

    # Traite les fichiers dans ./data/raw_data_imgs et ./data/raw_data_masks
    raw_data_dirs = {
        "./data/raw_data_imgs": ("./data/imgs", "image", False),  # Pas un masque
        "./data/raw_data_masks": ("./data/masks", "image", True)  # Ajouter "_mask"
    }

    for raw_data_path, (output_path, prefix, is_mask) in raw_data_dirs.items():
        if os.path.exists(raw_data_path):
            for file_name in os.listdir(raw_data_path):
                if file_name.endswith(".nii"):
                    nii_file_path = os.path.join(raw_data_path, file_name)
                    if is_mask:
                        mask_index = save_slices_2d(nii_file_path, output_path, prefix, is_mask, mask_index)
                    else:
                        img_index = save_slices_2d(nii_file_path, output_path, prefix, is_mask, img_index)

if __name__ == "__main__":
    process_data()
