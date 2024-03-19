import numpy as np
import SimpleITK as sitk
import os
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',
                  '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.nii',
                  '.nii.gz']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def preprocess(img):
    img[img < 0] = 0
    img = ((img - img.min()) / img.max()).astype(np.float32)
    return img


def postprocess(img):
    img[img < 0] = 0
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img


def GetPatch(out_root, patch_size):
    """
    Get image patch
    """
    t1 = get_image_paths('')
    flair = get_image_paths('')
    label = get_image_paths('')

    mkdir(out_root)
    mkdir(os.path.join(out_root, 't1'))
    mkdir(os.path.join(out_root, 't2-flair'))
    mkdir(os.path.join(out_root, 'label'))

    length = min(len(t1), len(flair), len(label))
    for num in range(length):
        img1 = read_img(t1[num]).astype(np.float32)
        # img1 = preprocess(img1)
        img2 = read_img(flair[num]).astype(np.float32)
        # img2 = preprocess(img2)
        seg = read_img(label[num]).astype(np.float32)
        seg = label_process(seg)

        D, H, W = img1.shape
        cont = 0
        slice = [0, 12]
        for d in range(len(slice)):
            i = slice[d]
            for h in range(0, H-patch_size, H-patch_size-1):
                for w in range(0, W-patch_size, W-patch_size-1):
                    cont += 1
                    img1_patch = img1[i:i+32, h:h+patch_size, w:w+patch_size]
                    img2_patch = img2[i:i+32, h:h + patch_size, w:w + patch_size]
                    seg_patch = seg[i:i+32, h:h + patch_size, w:w + patch_size]
                    img1_patch = sitk.GetImageFromArray(img1_patch, isVector=False)
                    img2_patch = sitk.GetImageFromArray(img2_patch, isVector=False)
                    seg_patch = sitk.GetImageFromArray(seg_patch, isVector=False)
                    sitk.WriteImage(img1_patch, os.path.join(out_root, f't1/T1_{num:02d}_{cont:03d}.nii'))
                    sitk.WriteImage(img2_patch, os.path.join(out_root, f't2-flair/Flair_{num:02d}_{cont:03d}.nii'))
                    sitk.WriteImage(seg_patch, os.path.join(out_root, f'label/Label_{num:02d}_{cont:03d}.nii'))


def label_process(label):
    """
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> [BK:0, CSF:1, GM:2, WM:3, MS:4]
    :param label:
    :return:
    """
    D, H, W = label.shape
    temp = np.zeros((D, H, W))
    # CSF
    temp[label == 5] = 1
    temp[label == 6] = 1
    # GM
    temp[label == 1] = 2
    temp[label == 2] = 2
    # WM
    temp[label == 3] = 3
    # MS
    temp[label == 4] = 4

    return temp.astype(np.float32)


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if m.ndim == 3:
                m = np.flip(m, axis)
            else:
                channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).
    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m, k):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


randflip = RandomFlip(np.random.RandomState(47), axis_prob=1, axis=0)
randrot = RandomRotate90(np.random.RandomState(47))


def augment_img(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return randflip(randrot(img, k=1))
    elif mode == 2:
        return randflip(img)
    elif mode == 3:
        return randrot(img, k=3)
    elif mode == 4:
        return randflip(randrot(img, k=2))
    elif mode == 5:
        return randrot(img, k=1)
    elif mode == 6:
        return randrot(img, k=2)
    elif mode == 7:
        return randflip(randrot(img, k=3))



def mask2one_hot(mask, palette):
    """
    :param mask:
    :param palette: [BK:0, CSF:1, GM:2, WM:3, MS:4]
    :return:
    """
    if mask.ndim == 3:
        mask = np.expand_dims(mask, axis=-1)
    semantic_map = []
    for label in palette:
        equality = np.equal(mask, label)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def one_hot2mask(mask, palette):
    """
    Converts a mask (D, H, W, K) to (D, H, W, C)
    palette: [BK:0, CSF:1, GM:2, WM:3, MS:4]
    """

    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


color = {
    'CSF': [253, 180, 98],
    'GM': [255, 255, 179],
    'WM': [251, 128, 114],
    'MS': [128, 177, 211],
    'BK': [0, 0, 0]  # [239, 247, 249]
}  # this is a color map for cv2, it means the channel is [B, G, R]


def slice_visualization(mask, num):
    """
    :param mask: [D, H, W]
    :return:
    """
    slice = mask[num]
    H, W = slice.shape[0], slice.shape[1]
    pixel = np.empty((H, W, 3))
    for x in range(H):
        for y in range(W):
            if slice[x, y] == 3:
                pixel[x, y, :] = color['WM']
            elif slice[x, y] == 4:
                pixel[x, y, :] = color['MS']
            elif slice[x, y] == 2:
                pixel[x, y, :] = color['GM']
            elif slice[x, y] == 1:
                pixel[x, y, :] = color['CSF']
            else:
                pixel[x, y, :] = color['BK']
    return pixel.astype(np.uint8)


def calculate_dice(input, target):
    dice = []
    for i in range(1, 5):
        gt = target[i, :, :, :]
        my_mask = input[i, :, :, :]
        gt = sitk.GetImageFromArray(gt, isVector=False)
        my_mask = sitk.GetImageFromArray(my_mask, isVector=False)
        dice_dist = sitk.LabelOverlapMeasuresImageFilter()
        dice_dist.Execute(gt > 0.5, my_mask > 0.5)
        dice.append(dice_dist.GetDiceCoefficient())
    return np.array(dice).astype(np.float32)


def calculate_accuracy(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):
    # batch_size = targets.size(0)
    y_pred = outputs[1:5, :, :, :]
    y_truth = targets[1:5, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    dice = []
    for i in range(1, 5):
        res = dice_coefficient_single_label(y_pred[i - 1:i, :, :, :], y_truth[i - 1:i, :, :, :], eps)
        dice.append(res.data)

    return dice


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):
    y_pred = outputs[0:1, :, :, :]
    y_truth = targets[0:1, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):

    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2
    union = torch.sum(y_pred, dim=(-3, -2, -1)) + torch.sum(y_truth, dim=(-3, -2, -1)) + eps
    dice = 2 * intersection / union
    return dice.mean()


if __name__ == '__main__':
    out_path = ''
    GetPatch(out_path, 128)











