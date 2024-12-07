import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
import torch
import os

# Création du dossier pour enregistrer les images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)  # Crée le dossier s'il n'existe pas

# Lecture et extraction de données d'un fichier NIfTI
file_path = 'image1_mask.nii'  # Remplacez par le chemin de votre fichier NIfTI
nifti_img = nib.load(file_path)
image_data = nifti_img.get_fdata()

print("Dimensions du volume :", image_data.shape)

# Convertir directement en un tenseur 4D avec C=1, H=256, W=256, D=36
image_tensor = torch.tensor(image_data, dtype=torch.float32)
image_tensor = image_tensor.unsqueeze(0)  # Ajouter la dimension C=1
print(f"Dimensions du tenseur 4D : {image_tensor.shape}")  # Affiche (1, 256, 256, 36)

# Créer un sujet TorchIO à partir du tenseur
subject = tio.Subject(
    image=tio.ScalarImage(tensor=image_tensor)  # Charger à partir du tenseur
)

# Appliquer une transformation 10 fois
for i in range(1, 11):
    # Définir une rotation aléatoire autour de l'axe Z (plan XY)
    rotation_transform = tio.transforms.RandomAffine(
        degrees=(0, 0, 180),  # Rotation entre -30° et 30° autour de Z
        scales=(0, 0, 0),     # Pas d'échelle
        translation=(50, 50, 0) # Pas de translation
    )

    # Appliquer la rotation à l'image
    transformed_subject = rotation_transform(subject)

    # Récupérer l'image transformée
    transformed_image = transformed_subject['image'][tio.DATA]
    print(f"Dimensions après transformation : {transformed_image.shape}")

    # Sélectionner une coupe axiale au centre de l'image originale et transformée
    slice_index = image_data.shape[2] // 2  # Indice au centre pour l'image originale

    # Extraire la coupe 2D de l'image originale
    original_image_2d = image_data[:, :, slice_index]

    # Extraire la coupe 2D de l'image transformée (en utilisant le même indice)
    transformed_image_2d = transformed_image[0, :, :, slice_index].numpy()

    # Enregistrer l'image originale (au format .png)
    if i == 1:  # Seulement pour la première itération, enregistrer l'image originale
        plt.imshow(original_image_2d, cmap='gray')
        plt.title('Image Originale')
        plt.axis('off')
        original_image_path = os.path.join(output_dir, 'image_originale.png')
        plt.savefig(original_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Enregistrer l'image transformée
    plt.imshow(transformed_image_2d, cmap='gray')
    plt.title(f'Image Transformée {i}')
    plt.axis('off')
    transformed_image_path = os.path.join(output_dir, f'image_transformée_{i}.png')
    plt.savefig(transformed_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Image transformée {i} enregistrée sous {transformed_image_path}")

print("Transformation terminée et images enregistrées.")
