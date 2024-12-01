import os
import numpy as np
from pathlib import Path
from PIL import Image
import nibabel as nib

def normalize_to_8bit(data):
    """
    Normalise les données en une plage [0, 255] pour compatibilité avec les images 8 bits.
    """
    data_min, data_max = data.min(), data.max()
    if data_max > data_min:  # Evite la division par zéro
        normalized = (data - data_min) / (data_max - data_min) * 255
    else:
        normalized = np.zeros_like(data)  # Tout noir si min == max
    return normalized.astype(np.uint8)

def convert_nii_to_slices(input_dir: Path, output_dir: Path, normalize: bool = False, mask_suffix: str = ''):
    """
    Convertit des fichiers NIfTI en slices 2D et les sauvegarde sous forme d'images PNG.

    :param input_dir: Dossier contenant les fichiers NIfTI à convertir.
    :param output_dir: Dossier où sauvegarder les slices convertis.
    :param normalize: Si True, applique la normalisation des données en [0, 255].
    :param mask_suffix: Suffixe pour identifier les masques (par exemple "_mask").
    """
    # Assurez-vous que le dossier de sortie existe
    output_dir.mkdir(parents=True, exist_ok=True)

    image_counter = 1  # Compteur pour les images
    mask_counter = 1  # Compteur pour les masques

    # Parcours des fichiers .nii dans le dossier d'entrée
    for nii_file in input_dir.glob('*.nii*'):
        # Chargement de l'image NIfTI
        nii_data = nib.load(str(nii_file)).get_fdata()

        # Vérifier si les données sont volumétriques
        if nii_data.ndim != 3:
            print(f"Fichier {nii_file.stem} ignoré (données non volumétriques).")
            continue

        print(f"Traitement de {nii_file.stem}, taille : {nii_data.shape[2]} slices.")

        # Traitement des slices selon l'axe z
        for z in range(nii_data.shape[2]):
            slice_data = nii_data[:, :, z]

            # Normalisation si nécessaire
            if normalize:
                slice_data = normalize_to_8bit(slice_data)

            # Conversion en image PNG
            slice_img = Image.fromarray(slice_data.astype(np.uint8))

            

            # Si c'est un masque, on applique le suffixe "_mask"
            if mask_suffix:
                slice_img.save(output_dir / f'image{mask_counter}_mask.png')
                print(f"Masque {mask_counter} sauvegardé.")
                mask_counter += 1  # Incrémente le compteur des masques

            else:
                # Sauvegarde de l'image (image1, image2, ...)
                slice_img.save(output_dir / f'image{image_counter}.png')
                print(f"Image {image_counter} sauvegardée.")

            # Incrémenter le compteur d'images après chaque slice
            image_counter += 1

# Exemple d'utilisation
raw_imgs_dir = Path('./rawdata/rawimgs/')  # Dossier des images volumétriques
raw_masks_dir = Path('./rawdata/rawmasks/')  # Dossier des masques volumétriques
imgs_dir = Path('./data/imgs/')  # Dossier de sortie des images 2D
masks_dir = Path('./data/masks/')  # Dossier de sortie des masques 2D

# Conversion des images sans normalisation
convert_nii_to_slices(raw_imgs_dir, imgs_dir, normalize=False)

# Conversion des masques avec normalisation (si applicable)
convert_nii_to_slices(raw_masks_dir, masks_dir, normalize=True, mask_suffix='_mask')
