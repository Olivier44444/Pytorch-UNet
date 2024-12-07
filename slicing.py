import nibabel
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize



def read_train_exam(exam_nb):
    image = nibabel.load(f'data/imgs/image{exam_nb}.nii')
    mask = nibabel.load(f'data/masks/image{exam_nb}_mask.nii')
    return image, mask

def read_test_exam(exam_nb):
    image = nibabel.load(f'data/imgs/image{exam_nb}.nii.gz')
    return image


train_ids = [1,2,3,5,8,10,13,19]
#test_ids = [21,22,32,39]



img_slices, mask_slices = [], []
for train_id in train_ids:
    image, mask = read_train_exam(train_id)
    a = np.unique(np.where(mask.get_fdata()>0.)[2])
    z = a[int(1*len(a)/2)]
    img_slices.append(resize(image.get_fdata()[:,:,z],(256,256),mode='reflect',anti_aliasing=True))
    mask_slices.append(resize(mask.get_fdata()[:,:,z],(256,256),mode='reflect',anti_aliasing=True))

plt.figure(figsize=(20, 15))
for i in range(len(train_ids)):
    plt.subplot(1, len(train_ids), i+1)
    plt.imshow(img_slices[i], cmap='gray', interpolation='nearest')
    plt.imshow(mask_slices[i], cmap='gray', alpha=0.3)
plt.show()
