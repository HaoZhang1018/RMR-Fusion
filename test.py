import argparse
import numpy as np
from glob import glob
import os
from models.dualSwinAE import DualSwinAE
from models.FusionNet import FusionNet
from monai.networks.nets import SwinUNETR
from utils import mkdir, preprocess, label_process
from PIL import Image
import torch
import log
import SimpleITK as sitk
from utils import read_img, postprocess, mask2one_hot, one_hot2mask
from utils import get_image_paths, calculate_accuracy, slice_visualization
import time


index = [i for i in range(23, 32)]


def _tensor(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)
    return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)


def patches2image(seg, patch_size=128):
    image = torch.zeros((5, 48, 240, 240))
    cont = 0
    slice = [0, 12]
    for d in range(len(slice)):
        i = slice[d]
        for h in range(0, 240 - patch_size, 240 - patch_size - 1):
            for w in range(0, 240 - patch_size, 240 - patch_size - 1):
                image[:, i:i + 32, h:h + patch_size, w:w + patch_size] = seg[cont]
                cont += 1
    return image


def run_fusion(io, args):
    # ---------------
    # image path
    # ---------------
    img_path1 = get_image_paths(args.root + '/whole/t1/')
    img_path2 = get_image_paths(args.root + '/whole/t2-flair/')
    img_num = len(img_path1)
    mkdir(args.result_path)
    # ---------------
    # load model
    # ---------------
    # encoder-decoder model path
    TransNet_model_path = os.path.join(args.save_path, './SwinAE/model.pth')

    # fusion model path
    Fusion_model_path = os.path.join(args.save_path, './Train/Fusion/model.pth')

    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    if args.cuda:
        io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
                  str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.seed)
    else:
        io.cprint('Using CPU')

    # encoder-decoder Network
    state_dict = torch.load(TransNet_model_path, map_location=device)
    feature_net = DualSwinAE().to(device)
    feature_net.load_state_dict(state_dict['model'])
    # fusion Network
    state_dict = torch.load(Fusion_model_path, map_location=device)
    model = FusionNet().to(device)
    model.load_state_dict(state_dict['model'])

    feature_net.eval()
    model.eval()
    with torch.no_grad():
        for i in range(int(img_num)):
            start = time.time()
            img1 = read_img(img_path1[i])
            img1 = preprocess(img1)
            img2 = read_img(img_path2[i])
            img2 = preprocess(img2)
            img1 = _tensor(img1).unsqueeze(0).to(device)
            img2 = _tensor(img2).unsqueeze(0).to(device)

            # feature extract
            feat1 = feature_net.encoder1(img1)
            feat2 = feature_net.encoder2(img2)

            # feature fuse
            out_feat = model(feat1, feat2)

            # image reconstruction
            out = feature_net.decoder(out_feat)

            out = out.squeeze().detach().cpu()
            out = out.data.numpy()
            out = postprocess(out)
            for k in range(len(index)):
                fused_img = out[index[k]]
                fused_img = (fused_img - fused_img.min()) / (fused_img.max() - fused_img.min()) * 255
                fused_img = Image.fromarray(np.uint8(fused_img))
                fused_img.save(os.path.join(args.result_path, f'fused_img_{i}{k + 1}.png'))
            used_time = time.time() - start
            io.cprint(f'fuse image {i + 1} spend {used_time} times')


def run_seg_only(io, args):
    # ---------------
    # image path
    # ---------------
    img_path1 = get_image_paths(args.root + '/patches/t1/')
    img_path2 = get_image_paths(args.root + '/patches/t2-flair/')
    img_num = len(img_path1)
    seg_path = get_image_paths(args.root + '/whole/label/')
    mkdir(os.path.join(args.result_path, 'Segmentation'))
    # ---------------
    # load model
    # ---------------
    # encoder-decoder model path
    TransNet_model_path = os.path.join(args.save_path, './SwinAE/model.pth')
    # fusion model path
    Fusion_model_path = os.path.join(args.save_path, './Train/Fusion/model.pth')
    # segmentation model path
    Segmentation_model_path = os.path.join(args.save_path, './Train/Segmentation/model.pth')

    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    if args.cuda:
        io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
                  str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.seed)
    else:
        io.cprint('Using CPU')

    # encoder-decoder Network
    state_dict = torch.load(TransNet_model_path, map_location=device)
    feature_net = DualSwinAE().to(device)
    feature_net.load_state_dict(state_dict['model'])
    # fusion Network
    state_dict = torch.load(Fusion_model_path, map_location=device)
    model = FusionNet().to(device)
    model.load_state_dict(state_dict['model'])
    # segmentation Network
    state_dict = torch.load(Segmentation_model_path, map_location=device)
    seg_model = SwinUNETR(
        img_size=(32, 128, 128),
        in_channels=16,
        out_channels=5,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0
    ).to(device)
    seg_model.load_state_dict(state_dict['model'])

    feature_net.eval()
    model.eval()
    seg_model.eval()
    with torch.no_grad():
        start = time.time()
        final_out = []
        for i in range(int(img_num)):
            img1 = read_img(img_path1[i])
            # img1 = preprocess(img1)
            img2 = read_img(img_path2[i])
            # img2 = preprocess(img2)
            img1 = _tensor(img1).unsqueeze(0).to(device)
            img2 = _tensor(img2).unsqueeze(0).to(device)

            # feature extract
            feat1 = feature_net.encoder1(img1)
            feat2 = feature_net.encoder2(img2)

            # feature fuse
            out_feat = model(feat1, feat2)

            # Segmentation
            seg_out = seg_model(out_feat)
            seg_out = torch.softmax(seg_out, dim=1)

            seg_out = seg_out.squeeze(0).detach().cpu()
            final_out.append(seg_out)

        final_out = patches2image(final_out)
        seg = read_img(seg_path[0])
        seg = label_process(seg)
        seg = mask2one_hot(seg, palette=[0, 1, 2, 3, 4])
        seg = _tensor(seg)
        dice = calculate_accuracy(final_out, seg)
        io.cprint("DSC--GSF GM WM MS:" + str(dice))

        final_out = final_out.permute(1, 2, 3, 0).data.numpy()
        final_out = one_hot2mask(final_out, palette=[0, 1, 2, 3, 4])
        for k in range(len(index)):
            seg_RGB = slice_visualization(final_out, index[k])
            seg_RGB = Image.fromarray(seg_RGB, mode='RGB')
            seg_RGB.save(os.path.join(args.result_path, f'./Segmentation/seg_results_RGB_0{k + 1}.png'))
        final_out = sitk.GetImageFromArray(final_out, isVector=False)
        sitk.WriteImage(final_out, os.path.join(args.result_path, f'./Segmentation/seg_results_0.nii'))
        used_time = time.time() - start
        io.cprint(f'Segmentation {i + 1} spend {used_time} times')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model save and load')
    parser.add_argument('--exp_name', type=str, default='Fusion_test', help='Name of the experiment')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--root', type=str, default='./test_img', help='data path root')
    parser.add_argument('--save_path', type=str, default='./train_result', help='model and pics save path')
    parser.add_argument('--result_path', type=str, default='./results', help='fused images save path')
    parser.add_argument('--workers', type=float, default=0, help='to determine the number of worker in dataloader')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    io = log.IOStream(args)
    io.cprint(str(args))

    run_fusion(io, args)
    run_seg_only(io, args)

