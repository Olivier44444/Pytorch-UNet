import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import nibabel as nib
import numpy as np


#Changement fait par Pol
#-------------------------------------------------------------------
def load_image(filename):
    filename = str(filename)  # Convertir en chaîne de caractères
    
    if filename.endswith(('.nii', '.nii.gz')):
        # Charger un fichier NIfTI
        nii_img = nib.load(filename)
        return nii_img.get_fdata()
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        # Charger une image PNG ou JPEG et la convertir en tableau NumPy
        with Image.open(filename) as img:
            return np.array(img)
    else:
        # Charger d'autres formats d'image standard avec Pillow
        return Image.open(filename)
#----------------------------------------------------------------------

def unique_mask_values(idx, mask_dir, mask_suffix):
    #print("SUFFIXXXXXXXXXXXXXXXXXXXXEEEEEEEEEEEEEEEEEEEEEEE : ", idx + mask_suffix + '.*')
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 3:
        # Prendre une coupe centrale si les données sont volumétriques
        mask = mask[:, :, mask.shape[2] // 2]
    if mask.ndim == 2:
        return np.unique(mask)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')
#----------------------------------------------------------------------


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
    
    def __len__(self):
        return len(self.ids)
    #Autre modification par Pol
    #--------------------------------------------------------------------------------------------------------
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        if isinstance(pil_img, np.ndarray):  # Si c'est un tableau NumPy
            pil_img = Image.fromarray(pil_img.astype(np.uint8))

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img


    def __getitem__(self, idx):
        name = self.ids[idx]
        logging.info(f'Idx: {idx}')
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # Si les masques ou images sont volumétriques, prendre une coupe
        if mask.ndim == 3:
            mask = mask[:, :, mask.shape[2] // 2]
        if img.ndim == 3 and img.shape[2] > 3:  # Pour les images volumétriques
            img = img[:, :, img.shape[2] // 2]

        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(self.mask_values, Image.fromarray(img.astype(np.uint8)), self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, Image.fromarray(mask.astype(np.uint8)), self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
#----------------------------------------------------------------------------------------------------------------------------

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
