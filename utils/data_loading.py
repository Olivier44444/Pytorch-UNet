import logging
import numpy as np
import torch
import nibabel as nib
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from scipy.ndimage import zoom  # Importation de la fonction zoom
from torch.utils.data import Dataset
from tqdm import tqdm
from skimage.transform import resize

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return np.load(filename)
    elif ext in ['.pt', '.pth']:
        return torch.load(filename).numpy()
    elif ext in ['.nii', '.nii.gz']:
        return nib.load(filename).get_fdata()
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = mask_dir / f"{idx}{mask_suffix}.nii"
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 3:
        return np.unique(mask)
    else:
        raise ValueError(f"Loaded masks should have 3 dimensions, found {mask.ndim}")


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.image_counter = 1  # Compteur global pour les noms des images

        # Obtenir les IDs des fichiers
        self.ids = [file.split('_mask')[0] for file in listdir(mask_dir)
                    if isfile(join(mask_dir, file)) and file.endswith('.nii')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Scanner les fichiers de masque pour déterminer les valeurs uniques
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique)).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        # La taille du dataset est le nombre total de slices 2D
        total_slices = 0
        for img_id in self.ids:
            img_file = self.images_dir / f"{img_id}.nii"
            img = load_image(img_file)  # Charger l'image 3D
            total_slices += img.shape[2]  # Profondeur (nombre de slices 2D)
        return total_slices

    def preprocess(self, mask_values, img, scale, is_mask):
        # Redimensionner et normaliser les images et les masques
        if is_mask:
            mask = np.zeros_like(img, dtype=np.int64)
            for i, v in enumerate(mask_values):
                mask[img == v] = i
            return mask
        else:
            if img.max() > 1:
                img = img / 255.0  # Normalisation des images
            return img

    def __getitem__(self, idx):
        # Identifier quel fichier (image et masque) utiliser pour cet index
        cumulative_slices = 0
        for img_id in self.ids:
            img_file = self.images_dir / f"{img_id}.nii"
            mask_file = self.mask_dir / f"{img_id}{self.mask_suffix}.nii"   

            # Charger les images et masques 3D
            img = load_image(img_file)
            mask = load_image(mask_file)

            #print('TYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYPE : ', type(img))
            #print('SHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPE : ', img.shape)

            # Si l'index idx est dans cette image, récupérer la slice correspondante
            depth = img.shape[2]
            if cumulative_slices <= idx < cumulative_slices + depth:
                # Identifier l'index exact de la slice
                z = idx - cumulative_slices
                img_slice = resize(img[:, :, z],(256,256),mode='reflect',anti_aliasing=True)
                mask_slice = resize(mask[:, :, z],(256,256),mode='reflect',anti_aliasing=True)
                

                
                # Prétraiter l'image et le masque
                img_slice = self.preprocess(self.mask_values, img_slice, self.scale, is_mask=False)
                mask_slice = self.preprocess(self.mask_values, mask_slice, self.scale, is_mask=True)

                #print('SHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPE : ', img_slice.shape)

                #Permet d'avoir une matrice de la forme (1, x, y)
                img_slice = img_slice[None, :, :]
                #mask_slice = mask_slice[None, :, :]

                # Générer les noms uniques
                image_name = f"image{self.image_counter}"
                mask_name = f"{image_name}_mask"
                self.image_counter += 1  # Incrémenter le compteur global

                # Retourner l'image, le masque et leurs noms
                return {
                    'image': torch.as_tensor(img_slice.copy()).float().contiguous(),
                    'mask': torch.as_tensor(mask_slice.copy()).long().contiguous(),
                    'image_name': image_name,
                    'mask_name': mask_name
                }

            # Sinon, augmenter l'index de cumulative_slices pour le prochain fichier
            cumulative_slices += depth

        # Si on arrive ici, c'est qu'on a dépassé l'index prévu
        raise IndexError(f"Index {idx} out of range in dataset.")


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
