import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torchio as tio









# Lecture et extraction de données d'un fichier NIfTI
file_path = 'image1.nii'  # Remplacez par le chemin de votre fichier NIfTI
nifti_img = nib.load(file_path)
image_data = nifti_img.get_fdata()

print("Dimensions du volume :", image_data.shape)

# Choisir une coupe axiale au centre
slice_index = image_data.shape[2] // 2
slice_data = image_data[:, :, slice_index]

# Affichage de la coupe originale
plt.figure(figsize=(6, 6))
plt.imshow(slice_data, cmap='gray')
plt.colorbar()
plt.title(f'Coupe axiale originale (index {slice_index})')
plt.axis('off')
plt.show()

# Convertir la coupe en objet TorchIO pour appliquer les transformations
slice_tensor = torch.tensor(slice_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [C, H, W] format attendu

subject = tio.Subject(image=tio.ScalarImage(tensor=slice_tensor))

# Définir un pipeline d'augmentations avec TorchIO
transformations = tio.Compose([
    tio.RandomAffine(scales=(1, 1), degrees=(30, 30, 0), translation=(20, 30, 0)),  # Rotation + Translation
    tio.RandomFlip(axes=(0,), flip_probability=1.0),  # Symétrie verticale (exemple)
])

# Appliquer les transformations
augmented_subject = transformations(subject)
augmented_image = augmented_subject['image'].data.squeeze().numpy()  # Extraire la slice transformée

# Affichage de la coupe transformée
plt.figure(figsize=(6, 6))
plt.imshow(augmented_image, cmap='gray')
plt.colorbar()
plt.title('Coupe après transformation (rotation + translation)')
plt.axis('off')
plt.show()
