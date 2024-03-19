import numpy as np
import random
import SimpleITK as sitk
import copy
import os
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from utils import mkdir, read_img, augment_img, mask2one_hot, preprocess
from utils import get_image_paths
from imgaug import augmenters as iaa
import torchio as tio
np.random.seed(0)


def AddNoise(img):
    max = torch.max(img).data.numpy()
    noise = tio.RandomNoise(0, (0, 0.1 * max))
    aug_image = noise(img)
    return aug_image


def GaussianBlur(img):
    aug_GaussianBlur = iaa.Sequential(iaa.GaussianBlur(sigma=(0.1, 3.0)))
    aug_image = aug_GaussianBlur(images=img)
    return aug_image


def fourier_broken(patch, num, nb_rows, nb_cols):
    """
    window内的傅里叶变换
    """
    img_recon = np.empty((num, nb_rows, nb_cols))
    for i in range(num):
        x = patch[i]
        aug_a = iaa.GaussianBlur(sigma=0.5)
        aug_p = iaa.Jigsaw(nb_rows=nb_rows, nb_cols=nb_cols, max_steps=(1, 5))
        fre = np.fft.fft2(x)
        fre_a = np.abs(fre)
        fre_p = np.angle(fre)
        fre_a_normalize = fre_a / (np.max(fre_a) + 0.0001)
        fre_p_normalize = fre_p
        fre_a_trans = aug_a(image=fre_a_normalize)
        fre_p_trans = aug_p(image=fre_p_normalize)
        fre_a_trans = fre_a_trans * (np.max(fre_a) + 0.0001)
        fre_p_trans = fre_p_trans
        fre_recon = fre_a_trans * np.e ** (1j * fre_p_trans)
        img_recon[i, :, :] = np.abs(np.fft.ifft2(fre_recon))
    return img_recon


def Fourier_transform(img):
    image_temp = copy.deepcopy(img)
    orig_image = copy.deepcopy(img)
    img_deps, img_rows, img_cols = img.shape
    num_block = 8
    for _ in range(num_block):
        block_noise_size_x = random.randint(img_rows // 8 - 4, img_rows // 8)
        block_noise_size_y = random.randint(img_cols // 8 - 4, img_cols // 8)
        block_noise_size_z = random.randint(0, img_deps // 8)
        noise_x = random.randint(2 * block_noise_size_x, img_rows - 2 * block_noise_size_x)
        noise_y = random.randint(2 * block_noise_size_y, img_cols - 2 * block_noise_size_y)
        noise_z = random.randint(2 * block_noise_size_z, img_deps - 2 * block_noise_size_z)
        window = orig_image[noise_z:noise_z + block_noise_size_z, noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y]
        window = fourier_broken(window, block_noise_size_z, block_noise_size_x, block_noise_size_y)
        image_temp[noise_z:noise_z + block_noise_size_z, noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y] = window
    transform_img = image_temp

    return transform_img


class AEDataset(Dataset):
    def __init__(self, io, root, transform=True, ssl_transform=True, partition='train'):
        super(AEDataset, self).__init__()
        # data path
        self.files1 = get_image_paths(os.path.join(root, './t1'))
        self.files2 = get_image_paths(os.path.join(root, './t2-flair'))

        # basic operation
        self.transform = transform
        self.ssl_transform = ssl_transform
        self.num_examples = len(self.files1)
        if self.partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.train_ind)
        elif self.partition == 'valid':
            self.val_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.val_ind)
        elif self.partition == 'test':
            self.test_ind = [i for i in range(self.num_examples)]

        io.cprint("number of " + partition + " examples in dataset" + ": " + str(self.num_examples))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        img1_path = self.files1[idx]
        img2_path = self.files2[idx]

        T1 = read_img(img1_path)
        T2 = read_img(img2_path)

        if self.transform:
            mode = random.randint(0, 7)
            T1 = augment_img(T1, mode)
            T2 = augment_img(T2, mode)

        if self.ssl_transform:
            img_add_noise_org1 = T1.copy()
            img_add_noise_org1 = self._tensor(img_add_noise_org1)
            img_add_noise1 = AddNoise(img_add_noise_org1)

            img_blur_org1 = T1.copy()
            img_blur1 = GaussianBlur(img_blur_org1)
            img_blur1 = self._tensor(img_blur1)

            img_fourier_org1 = T1.copy()
            img_fourier1 = Fourier_transform(img_fourier_org1)
            img_fourier1 = self._tensor(img_fourier1)

            img1 = self._tensor(np.expand_dims(T1, axis=-1))

            img_add_noise_org2 = T2.copy()
            img_add_noise_org2 = self._tensor(img_add_noise_org2)
            img_add_noise2 = AddNoise(img_add_noise_org2)

            img_blur_org2 = T2.copy()
            img_blur2 = GaussianBlur(img_blur_org2)
            img_blur2 = self._tensor(img_blur2)

            img_fourier_org2 = T2.copy()
            img_fourier2 = Fourier_transform(img_fourier_org2)
            img_fourier2 = self._tensor(img_fourier2)

            img2 = self._tensor(np.expand_dims(T2, axis=-1))

            return img1, img_blur1, img_add_noise1, img_fourier1, \
                   img2, img_blur2, img_add_noise2, img_fourier2
        else:
            img1 = self._tensor(np.expand_dims(T1, axis=-1))
            img2 = self._tensor(np.expand_dims(T2, axis=-1))
            return img1, img2

    def _tensor(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)


def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
    """
    Only do intensity shift on brain voxels
    :param imgs_array: The whole input image with shape of (4, 48, 128, 128)
    :param brain_mask:
    :param limit:
    :return:
    """

    shift_range = 2 * limit
    for i in range(len(imgs_array) - 1):
        factor = -limit + shift_range * np.random.random()
        std = imgs_array[i][brain_mask].std()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
    return imgs_array


def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1, :]
            if axis == 3:
                imgs_array = imgs_array[:, :, :, ::-1]
    return imgs_array


class FusionDataset(Dataset):
    def __init__(self, io, root, transform=True, partition='train'):
        super(FusionDataset, self).__init__()
        # data path
        self.img1_paths = get_image_paths(os.path.join(root, './t1/'))  # T1
        self.img2_paths = get_image_paths(os.path.join(root, './t2-flair/'))  # T2
        self.label_paths = get_image_paths(os.path.join(root, './label/'))  # Segmentation label

        assert len(self.img1_paths) == len(self.img2_paths), "the number of image pair should be the same"
        assert len(self.img1_paths) == len(self.label_paths), 'The label should correspond to the image'
        # basic operation
        self.palette = [0, 1, 2, 3, 4]  # [BK:0, CSF:1, GM:2, WM:3, MS:4]
        self.transform = transform
        self.partition = partition

        self.num_examples = len(self.label_paths)

        if self.partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.train_ind)
        elif self.partition == 'valid':
            self.val_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.val_ind)
        elif self.partition == 'test':
            self.test_ind = [i for i in range(self.num_examples)]
        else:
            raise AssertionError("the declare parameter of partition should be train or test")
        io.cprint("number of " + partition + " examples in dataset" + ": " + str(self.num_examples))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]
        label_path = self.label_paths[idx]

        T1 = read_img(img1_path)
        T2 = read_img(img2_path)
        seg = read_img(label_path)

        if self.transform:
            mode = random.randint(0, 7)
            T1 = augment_img(T1, mode=mode)
            T2 = augment_img(T2, mode=mode)
            seg = augment_img(seg, mode=mode)

        T1 = np.expand_dims(T1, axis=-1)
        T2 = np.expand_dims(T2, axis=-1)
        seg = mask2one_hot(seg, self.palette)  # [BK:0, CSF:1, GM:2, WM:3, MS:4]
        img1 = self._tensor(T1)
        img2 = self._tensor(T2)
        seg = self._tensor(seg)
        return img1, img2, seg

    def _tensor(self, x):
        return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)




