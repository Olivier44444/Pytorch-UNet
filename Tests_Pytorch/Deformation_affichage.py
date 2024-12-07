import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio
import torch

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

# Définir une rotation aléatoire autour de l'axe Z (plan XY)
rotation_transform = tio.transforms.RandomAffine(
    degrees=(0, 0, 0),  # Rotation 
    scales=(0, 0, 0),     # Changement d'échelle
    translation=(50, 50, 0) # Translation
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
transformed_image_2d = transformed_image[0, :, :, slice_index].numpy()  # Correction ici

# Afficher les deux images côte à côte
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Afficher l'image originale
axes[0].imshow(original_image_2d, cmap='gray')
axes[0].set_title('Image Originale')
axes[0].axis('off')

# Afficher l'image transformée
axes[1].imshow(transformed_image_2d, cmap='gray')
axes[1].set_title('Image Transformée')
axes[1].axis('off')

plt.show()
