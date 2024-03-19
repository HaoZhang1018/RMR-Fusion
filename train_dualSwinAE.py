import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import AEDataset
from models.dualSwinAE import DualSwinAE
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import FeatureAELoss, ImageAELoss
import log
from tqdm import tqdm
from utils import mkdir
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model save and load')
    parser.add_argument('--exp_name', type=str, default='Fusion_experiments', help='Name of the experiment')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--root', type=str, default='./dataset64/', help='data path root')
    parser.add_argument('--save_path', type=str, default='./train_result/SwinAE', help='model and pics save path')
    parser.add_argument('--ssl_transform', type=bool, default=True, help='use ssl_transformations or not')
    parser.add_argument('--workers', type=float, default=0, help='to determine the number of worker in dataloader')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--summary_name', type=str, default='Semantic_Fusion_', help='Name of the tensorboard summary')

    args = parser.parse_args()
    writer = SummaryWriter('./runs/SemanticFusion')  # 为你的存储日志命名

    io = log.IOStream(args)
    io.cprint(str(args))

    # ==================
    # init: before training, you should set the init and args
    # ==================

    pretrained = False
    # model path
    use_model_path = os.path.join(args.save_path, '')
    start_epoch = torch.load(use_model_path)['epoch'] + 1 if pretrained else 0
    # if you don't need the read epoch, change it with a number you want like 0
    # start_epoch = 0

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
    if args.cuda:
        io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
                  str(torch.cuda.device_count()) + ' devices available')
        torch.cuda.manual_seed_all(args.seed)
    else:
        io.cprint('Using CPU')

    # ==================
    # Read Data
    # ==================

    train_dataset = AEDataset(io, args.root+'training', transform=True, ssl_transform=args.ssl_transform, partition='train')
    val_dataset = AEDataset(io, args.root+'validation', transform=True, ssl_transform=args.ssl_transform, partition='val')

    # Creating data indices for training and validation splits:
    train_indices = train_dataset.train_ind
    val_indices = val_dataset.val_ind

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=args.batch_size,
                            sampler=valid_sampler)
    start = time.time()
    # ==================
    # Init Model
    # ==================
    model = DualSwinAE().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.wd) if args.optimizer == "SGD" \
        else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if pretrained:
        state_dict = torch.load(use_model_path, map_location='cuda:0')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])

    scheduler = CosineAnnealingLR(optimizer, 300)
    # loss
    featloss = FeatureAELoss()
    imgloss = ImageAELoss()

    # Handle multi-gpu
    if (device.type == 'cuda') and len(args.gpus) > 1:
        model = nn.DataParallel(model, args.gpus)

    # ==================
    # Model Training
    # ==================
    mkdir(args.save_path)
    last_index = len(train_loader) - 1
    io.cprint(f"the loop times of a epoch is {(last_index + 1):d}")
    for epoch in tqdm(range(start_epoch, args.epoch)):
        img_loss_self_per_epoch = 0.
        feat_loss_noise_per_epoch, feat_loss_blur_per_epoch, feat_loss_fourier_per_epoch = 0., 0., 0.
        img_loss_noise_per_epoch, img_loss_blur_per_epoch, img_loss_fourier_per_epoch = 0., 0., 0.
        total_loss_per_epoch = 0.
        model.train()
        for index, image in enumerate(train_loader):
            img_orig1 = image[0].to(device)
            img_blur1 = image[1].to(device)
            img_noise1 = image[2].to(device)
            img_fourier1 = image[3].to(device)
            img_orig2 = image[4].to(device)
            img_blur2 = image[5].to(device)
            img_noise2 = image[6].to(device)
            img_fourier2 = image[7].to(device)

            optimizer.zero_grad()

            feat_orig1, img_recon_orig1, feat_orig2, img_recon_orig2 = model(img_orig1, img_orig2)
            feat_blur1, img_recon_blur1, feat_blur2, img_recon_blur2 = model(img_blur1, img_blur2)
            feat_noise1, img_recon_noise1, feat_noise2, img_recon_noise2 = model(img_noise1, img_noise2)
            feat_fourier1, img_recon_fourier1, feat_fourier2, img_recon_fourier2 = model(img_fourier1, img_fourier2)

            image_orig1 = img_orig1[0].squeeze(0).detach().cpu()
            image_bright1 = img_recon_blur1[0].squeeze(0).detach().cpu()
            image_shuffling1 = img_recon_noise1[0].squeeze(0).detach().cpu()
            image_fourier1 = img_recon_fourier1[0].squeeze(0).detach().cpu()

            image_orig2 = img_orig2[0].squeeze(0).detach().cpu()
            image_bright2 = img_recon_blur2[0].squeeze(0).detach().cpu()
            image_shuffling2 = img_recon_noise2[0].squeeze(0).detach().cpu()
            image_fourier2 = img_recon_fourier2[0].squeeze(0).detach().cpu()

            feat_loss_blur = featloss(feat_blur1, feat_orig1) + featloss(feat_blur2, feat_orig2)
            feat_loss_noise = featloss(feat_noise1, feat_orig1) + featloss(feat_noise2, feat_orig2)
            feat_loss_fourier = featloss(feat_fourier1, feat_orig1) + featloss(feat_fourier2, feat_orig2)
            feat_loss_blur_per_epoch += feat_loss_blur
            feat_loss_noise_per_epoch += feat_loss_noise
            feat_loss_fourier_per_epoch += feat_loss_fourier

            img_loss_self = imgloss(img_recon_orig1, img_orig1) + imgloss(img_recon_orig2, img_orig2)
            img_loss_noise = imgloss(img_recon_blur1, img_orig1) + imgloss(img_recon_blur2, img_orig2)
            img_loss_blur = imgloss(img_recon_noise1, img_orig1) + imgloss(img_recon_noise2, img_orig2)
            img_loss_fourier = imgloss(img_recon_fourier1, img_orig1) + imgloss(img_recon_fourier2, img_orig2)
            img_loss_self_per_epoch += img_loss_self
            img_loss_noise_per_epoch += img_loss_noise
            img_loss_blur_per_epoch += img_loss_blur
            img_loss_fourier_per_epoch += img_loss_fourier

            Loss1 = feat_loss_blur + feat_loss_noise + feat_loss_fourier
            Loss2 = img_loss_self + img_loss_noise + img_loss_blur + img_loss_fourier

            Loss = Loss1 + Loss2
            total_loss_per_epoch += Loss
            Loss.backward()
            optimizer.step()

            if index == last_index and epoch % 10 == 0:
                writer.add_image('Train/image_org1', image_orig1[8], epoch, dataformats='HW')
                writer.add_image('Train/image_bright1', image_bright1[8], epoch, dataformats='HW')
                writer.add_image('Train/image_fourier1', image_fourier1[8], epoch, dataformats='HW')
                writer.add_image('Train/image_shuffling1', image_shuffling1[8], epoch, dataformats='HW')
                writer.add_image('Train/image_org2', image_orig2[8], epoch, dataformats='HW')
                writer.add_image('Train/image_bright2', image_bright2[8], epoch, dataformats='HW')
                writer.add_image('Train/image_fourier2', image_fourier2[8], epoch, dataformats='HW')
                writer.add_image('Train/image_shuffling2', image_shuffling2[8], epoch, dataformats='HW')

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        io.cprint(f"Epoch:[{epoch:d}/{args.epoch:d}]-----learning rate: {learning_rate:.10f}")
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Train------LOSS:{(total_loss_per_epoch / (len(train_loader))):.4f}')
        writer.add_scalar('Train/feat_loss_blur', feat_loss_blur_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/feat_loss_noise', feat_loss_noise_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/feat_loss_fourier', feat_loss_fourier_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/img_loss_self', img_loss_self_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/img_loss_noise', img_loss_noise_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/img_loss_blur', img_loss_blur_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/img_loss_fourier', img_loss_fourier_per_epoch / (len(train_loader)), epoch)

        scheduler.step()
        # ==================
        # Model Validation
        # ==================
        model.eval()
        with torch.no_grad():
            img_loss_self_per_epoch = 0.
            feat_loss_noise_per_epoch, feat_loss_blur_per_epoch, feat_loss_fourier_per_epoch = 0., 0., 0.
            img_loss_noise_per_epoch, img_loss_blur_per_epoch, img_loss_fourier_per_epoch = 0., 0., 0.
            total_loss_per_epoch = 0.
            for index, image in enumerate(val_loader):
                img_orig1 = image[0].to(device)
                img_blur1 = image[1].to(device)
                img_noise1 = image[2].to(device)
                img_fourier1 = image[3].to(device)
                img_orig2 = image[4].to(device)
                img_blur2 = image[5].to(device)
                img_noise2 = image[6].to(device)
                img_fourier2 = image[7].to(device)

                feat_orig1, img_recon_orig1, feat_orig2, img_recon_orig2 = model(img_orig1, img_orig2)
                feat_blur1, img_recon_blur1, feat_blur2, img_recon_blur2 = model(img_blur1, img_blur2)
                feat_noise1, img_recon_noise1, feat_noise2, img_recon_noise2 = model(img_noise1, img_noise2)
                feat_fourier1, img_recon_fourier1, feat_fourier2, img_recon_fourier2 = model(img_fourier1, img_fourier2)

                image_orig1 = img_orig1[0].squeeze(0).detach().cpu()
                image_bright1 = img_recon_blur1[0].squeeze(0).detach().cpu()
                image_shuffling1 = img_recon_noise1[0].squeeze(0).detach().cpu()
                image_fourier1 = img_recon_fourier1[0].squeeze(0).detach().cpu()

                image_orig2 = img_orig2[0].squeeze(0).detach().cpu()
                image_bright2 = img_recon_blur2[0].squeeze(0).detach().cpu()
                image_shuffling2 = img_recon_noise2[0].squeeze(0).detach().cpu()
                image_fourier2 = img_recon_fourier2[0].squeeze(0).detach().cpu()

                feat_loss_blur = featloss(feat_blur1, feat_orig1) + featloss(feat_blur2, feat_orig2)
                feat_loss_noise = featloss(feat_noise1, feat_orig1) + featloss(feat_noise2, feat_orig2)
                feat_loss_fourier = featloss(feat_fourier1, feat_orig1) + featloss(feat_fourier2, feat_orig2)
                feat_loss_blur_per_epoch += feat_loss_blur
                feat_loss_noise_per_epoch += feat_loss_noise
                feat_loss_fourier_per_epoch += feat_loss_fourier

                img_loss_self = imgloss(img_recon_orig1, img_orig1) + imgloss(img_recon_orig2, img_orig2)
                img_loss_noise = imgloss(img_recon_blur1, img_orig1) + imgloss(img_recon_blur2, img_orig2)
                img_loss_blur = imgloss(img_recon_noise1, img_orig1) + imgloss(img_recon_noise2, img_orig2)
                img_loss_fourier = imgloss(img_recon_fourier1, img_orig1) + imgloss(img_recon_fourier2, img_orig2)
                img_loss_self_per_epoch += img_loss_self
                img_loss_noise_per_epoch += img_loss_noise
                img_loss_blur_per_epoch += img_loss_blur
                img_loss_fourier_per_epoch += img_loss_fourier

                Loss1 = feat_loss_blur + feat_loss_noise + feat_loss_fourier
                Loss2 = img_loss_self + img_loss_noise + img_loss_blur + img_loss_fourier

                Loss = Loss1 + Loss2
                total_loss_per_epoch += Loss
            val_loss = total_loss_per_epoch / (len(val_loader))
            io.cprint(
                f'Epoch:[{epoch:d}/{args.epoch:d}]-----Valid------LOSS:{val_loss:.4f}')
            writer.add_scalar('Valid/feat_loss_blur', feat_loss_blur_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/feat_loss_noise', feat_loss_noise_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/feat_loss_fourier', feat_loss_fourier_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/img_loss_self', img_loss_self_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/img_loss_noise', img_loss_noise_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/img_loss_blur', img_loss_blur_per_epoch / (len(val_loader)), epoch)
            writer.add_scalar('Valid/img_loss_fourier', img_loss_fourier_per_epoch / (len(val_loader)), epoch)

        # ==================
        # Model Saving
        # ==================
        if epoch % 10 == 0:
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }

            torch.save(state, os.path.join(args.save_path, args.summary_name +
                                           'epoch_' + str(epoch) + '_' + f'{val_loss:.4f}.pth'))
        if epoch == args.epoch - 1:
            state = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            torch.save(state, os.path.join(args.save_path, 'model.pth'))
    end = time.time()

    io.cprint("The training process has finished! ")







