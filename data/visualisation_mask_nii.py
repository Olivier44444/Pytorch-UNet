import nibabel as nib
import numpy as np

# Charger un masque en format NIfTI
nii_mask = nib.load("C:/Users/ptyme/Desktop/IMT/UE B Patient numérique/Projet/Pytorch-UNet/Pytorch-UNet/data/raw_data_masks/image1_mask.nii")
mask = nii_mask.get_fdata()

# Trouver les classes uniques
classes = np.unique(mask)
print("Classes trouvées :", classes) 