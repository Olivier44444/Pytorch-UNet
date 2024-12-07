import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
import torch
import os

# Création du dossier pour enregistrer les images
output_dir = 'output_images_and_mask'
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Lecture et extraction de données des fichiers NIfTI
file_path_mask = 'image1_mask.nii'  # Remplacez par le chemin de votre fichier NIfTI du masque
file_path_image = 'image1.nii'      # Remplacez par le chemin de votre fichier NIfTI associé

nifti_mask = nib.load(file_path_mask)
nifti_image = nib.load(file_path_image)
mask_data = nifti_mask.get_fdata()
image_data = nifti_image.get_fdata()

print("Dimensions du volume du masque :", mask_data.shape)
print("Dimensions du volume de l'image :", image_data.shape)

# Vérification que les dimensions des deux volumes sont identiques
assert mask_data.shape == image_data.shape, "Les dimensions du masque et de l'image doivent être identiques."

# Convertir en tenseurs TorchIO
mask_tensor = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)  # Ajouter la dimension C=1
image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)

# Créer les sujets TorchIO
subject = tio.Subject(
    mask=tio.ScalarImage(tensor=mask_tensor),   # Charger le masque
    image=tio.ScalarImage(tensor=image_tensor)  # Charger l'image associée
)

# Appliquer une transformation 10 fois
for i in range(1, 11):
    # Définir une rotation aléatoire autour de l'axe Z (plan XY)
    rotation_transform = tio.transforms.RandomAffine(
        degrees=(0, 0, 180),  # Rotation entre -180° et 180° autour de Z
        scales=(0, 0, 0),     # Pas de redimensionnement
        translation=(50, 50, 0)  # Déplacement en X et Y
    )

    # Appliquer la transformation aux deux volumes
    transformed_subject = rotation_transform(subject)

    # Récupérer le masque transformé
    transformed_mask = transformed_subject['mask'][tio.DATA]
    transformed_image = transformed_subject['image'][tio.DATA]

    print(f"Dimensions après transformation (masque) : {transformed_mask.shape}")
    print(f"Dimensions après transformation (image) : {transformed_image.shape}")

    # Sélectionner une coupe axiale au centre
    slice_index = mask_data.shape[2] // 2  # Indice au centre

    # Extraire les coupes 2D
    original_mask_2d = mask_data[:, :, slice_index]
    transformed_mask_2d = transformed_mask[0, :, :, slice_index].numpy()

    original_image_2d = image_data[:, :, slice_index]
    transformed_image_2d = transformed_image[0, :, :, slice_index].numpy()

    # Enregistrer le masque original une seule fois
    if i == 1:
        plt.imshow(original_mask_2d, cmap='gray')
        plt.title('Masque Original')
        plt.axis('off')
        mask_original_path = os.path.join(output_dir, 'masque_original.png')
        plt.savefig(mask_original_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        plt.imshow(original_image_2d, cmap='gray')
        plt.title('Image Originale')
        plt.axis('off')
        image_original_path = os.path.join(output_dir, 'image_originale.png')
        plt.savefig(image_original_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Enregistrer le masque transformé
    plt.imshow(transformed_mask_2d, cmap='gray')
    plt.title(f'Masque Transformé {i}')
    plt.axis('off')
    mask_transformed_path = os.path.join(output_dir, f'masque_transformé_{i}.png')
    plt.savefig(mask_transformed_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Enregistrer l'image transformée
    plt.imshow(transformed_image_2d, cmap='gray')
    plt.title(f'Image Transformée {i}')
    plt.axis('off')
    image_transformed_path = os.path.join(output_dir, f'image_transformée_{i}.png')
    plt.savefig(image_transformed_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Masque transformé {i} enregistré sous {mask_transformed_path}")
    print(f"Image transformée {i} enregistrée sous {image_transformed_path}")

print("Transformations terminées et images enregistrées.")
