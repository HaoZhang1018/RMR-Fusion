import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss


def tf_fspecial_gauss_3d_torch(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data, z_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1,
                             -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=1)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=1)

    z_data = np.expand_dims(z_data, axis=0)
    z_data = np.expand_dims(z_data, axis=1)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    z = torch.tensor(z_data, dtype=torch.float32)

    g = torch.exp(-((x ** 2 + y ** 2 + z ** 2) / (3.0 * sigma ** 2)))
    return g / torch.sum(g)


class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, img1, img2, k1=0.01, k2=0.03, L=2, window_size=11):  # (img1, img2)
        window = tf_fspecial_gauss_3d_torch(window_size, 1.5)
        window = window.cuda()
        mu1 = torch.nn.functional.conv3d(img1, window, stride=1, padding=0)
        mu2 = torch.nn.functional.conv3d(img2, window, stride=1, padding=0)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = torch.nn.functional.conv3d(img1 * img1, window, stride=1, padding=0) - mu1_sq
        sigma2_sq = torch.nn.functional.conv3d(img2 * img2, window, stride=1, padding=0) - mu2_sq
        sigma1_2 = torch.nn.functional.conv3d(img1 * img2, window, stride=1, padding=0) - mu1_mu2
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        return 1-torch.mean(ssim_map)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        d = r.shape[2]
        h = r.shape[3]
        w = r.shape[4]
        tv1 = torch.pow((r[:, :, 1:, :, :] - r[:, :, :d - 1, :, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:, :] - r[:, :, :, :h - 1, :]), 2).mean()
        tv3 = torch.pow((r[:, :, :, :, 1:] - r[:, :, :, :, :w - 1]), 2).mean()
        return tv1 + tv2 + tv3


class FeatureAELoss(nn.Module):
    def __init__(self):
        super(FeatureAELoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.TV = TV_Loss()

    def forward(self, feat, org):
        mse = self.MSE(feat, org)
        tv = self.TV(feat, org)
        return 40 * mse + 20 * tv


class ImageAELoss(nn.Module):
    def __init__(self):
        super(ImageAELoss, self).__init__()
        self.MSE = nn.MSELoss()
        self.TV = TV_Loss()
        self.SSIM = SSIM()

    def forward(self, img, org):
        mse = self.MSE(img, org)
        tv = self.TV(img, org)
        ssim = self.SSIM(img, org)
        return 40 * mse + 20 * tv + ssim


class Intensity(nn.Module):
    def __init__(self):
        super(Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        Loss_intensity = F.l1_loss(image_fused, image_B)

        return Loss_intensity


class Grad(nn.Module):
    def __init__(self):
        super(Grad, self).__init__()
        self.sobelconv = Sobelxyz()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :, :]
        image_B_Y = image_B[:, :1, :, :, :]
        image_fused_Y = image_fused[:, :1, :, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)

        return Loss_gradient


def Sobel3D():
    hx = [1, 2, 1]
    hy = [1, 2, 1]
    hz = [1, 2, 1]
    hpx = [1, 0, -1]
    hpy = [1, 0, -1]
    hpz = [1, 0, -1]
    gx = np.zeros((3, 3, 3))
    gy = np.zeros((3, 3, 3))
    gz = np.zeros((3, 3, 3))
    for m in range(3):
        for n in range(3):
            for k in range(3):
                gx[m, n, k] = hpx[m] * hy[n] * hz[k]
                gy[m, n, k] = hx[m] * hpy[n] * hz[k]
                gz[m, n, k] = hx[m] * hy[n] * hpz[k]

    return gx, gy, gz


class Sobelxyz(nn.Module):
    def __init__(self):
        super(Sobelxyz, self).__init__()
        kernelx, kernely, kernelz = Sobel3D()
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        kernelz = torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.weightz = nn.Parameter(data=kernelz, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv3d(x, self.weightx, padding=1)
        sobely = F.conv3d(x, self.weighty, padding=1)
        sobelz = F.conv3d(x, self.weightz, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely) + torch.abs(sobelz)


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.Grad = Grad()

    def forward(self, image_A, image_B, image_fused):

        loss_gradient = self.Grad(image_A, image_B, image_fused)

        return loss_gradient


class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()
        self.Intensity = Intensity()

    def forward(self, image_A, image_B, image_fused):

        loss_intensity = self.Intensity(image_A, image_B, image_fused)

        return loss_intensity


class SegmentationLoss(nn.Module):
    """
    the input and target size is [5, 32, 128, 128],
    should change it when change the size
    """

    def __init__(self):
        super(SegmentationLoss, self).__init__()

        self.DiceLoss = DiceLoss(include_background=True,
                                 softmax=True
                                 )

    def forward(self, input, target):

        Loss_seg = self.DiceLoss(input, target)

        return Loss_seg

