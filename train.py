import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
from datasets import FusionDataset
from models.dualSwinAE import DualSwinAE
from models.FusionNet import FusionNet
from monai.networks.nets import SwinUNETR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from loss import IntensityLoss, GradLoss
from loss import SegmentationLoss
import log
import time
from tqdm import tqdm
from utils import mkdir, calculate_dice
from tensorboardX import SummaryWriter

toPILImage = transforms.ToPILImage()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model save and load')
    parser.add_argument('--exp_name', type=str, default='Fusion_experiments', help='Name of the experiment')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--root', type=str, default='./dataset/', help='data path root')
    parser.add_argument('--save_path', type=str, default='./train_result', help='model and pics save path')
    parser.add_argument('--workers', type=float, default=0, help='to determine the number of worker in dataloader')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--epoch', type=int, default=150, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--int', type=float, default=12, help='intensity weight')
    parser.add_argument('--grad', type=float, default=1, help='gradient weight')
    parser.add_argument('--seg', type=float, default=10, help='segmentation weight')
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm for FusionNet")
    parser.add_argument("--optim_name_seg", default="adamw", type=str, help="optimization algorithm for SegNet")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler for FusionNet")
    parser.add_argument("--lrschedule_seg", default="warmup_cosine", type=str, help="type of learning rate scheduler for SegNet")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--summary_name', type=str, default='Semantic_Fusion_', help='Name of the tensorboard summary')

    args = parser.parse_args()
    writer = SummaryWriter('./runs/SemanticFusion')

    io = log.IOStream(args)
    io.cprint(str(args))
    save_path = os.path.join(args.save_path, 'Train')
    mkdir(save_path)

    # ==================
    # init: before training, you should set the init and args
    # ==================

    # encoder-decoder model path
    TransNet_model_path = os.path.join(args.save_path, './SwinAE/model.pth')

    pretrained = False
    # fusion model path
    use_model_path = os.path.join(args.save_path, '')
    # segmentation model path
    Segmentation_model_path = os.path.join(args.save_path, '')
    start_epoch = torch.load(use_model_path)['epoch'] + 1 if pretrained else 0
    # if you don't need the read epoch, change it with a number you want like 0
    # start_epoch = 0

    np.random.seed(args.seed)  # to get the same images and leave it fixed
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

    train_dataset = FusionDataset(io, args.root+'training', transform=True, partition='train')
    val_dataset = FusionDataset(io, args.root+'validation', transform=False, partition='val')

    # Creating data indices for training and validation splits:
    train_indices = train_dataset.train_ind
    val_indices = val_dataset.val_ind

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)  # sampler will assign the whole data according to batchsize.
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batch_size,
                              sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=args.workers, batch_size=1,
                            sampler=valid_sampler)

    start = time.time()

    # ==================
    # Init Model
    # ==================
    # encoder-decoder Network
    state_dict = torch.load(TransNet_model_path, map_location=device)
    feature_net = DualSwinAE().to(device)
    feature_net.load_state_dict(state_dict['model'])
    del state_dict
    # Segmentation Network
    seg_model = SwinUNETR(
        img_size=(32, 128, 128),
        in_channels=16,
        out_channels=5,
        feature_size=48,
    ).to(device)
    # fusion Network
    model = FusionNet().to(device)

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epoch
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = CosineAnnealingLR(optimizer, T_max=300)
    else:
        scheduler = None

    if args.optim_name_seg == "adam":
        optimizer_seg = torch.optim.Adam(seg_model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name_seg == "adamw":
        optimizer_seg = torch.optim.AdamW(seg_model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name_seg == "sgd":
        optimizer_seg = torch.optim.SGD(
            seg_model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True,
            weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule_seg == "warmup_cosine":
        scheduler_seg = LinearWarmupCosineAnnealingLR(
            optimizer_seg, warmup_epochs=args.warmup_epochs, max_epochs=args.epoch
        )
    elif args.lrschedule_seg == "cosine_anneal":
        scheduler_seg = CosineAnnealingLR(optimizer_seg, T_max=300)
    else:
        scheduler_seg = None

    if pretrained:
        # fusion Network
        state_dict = torch.load(use_model_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        del state_dict
        # Segmentation Network
        state_dict = torch.load(Segmentation_model_path, map_location=device)
        seg_model.load_state_dict(state_dict['model'])
        optimizer_seg.load_state_dict(state_dict['optimizer'])
        del state_dict
        if args.lrschedule == "cosine_anneal":
            scheduler.step(epoch=start_epoch)
            scheduler_seg.step(epoch=start_epoch)

    # loss function
    # content loss
    Loss_intensity = IntensityLoss()
    Loss_gradient = GradLoss()
    # semantic loss
    Loss_Seg = SegmentationLoss()
    # Handle multi-gpu
    if (device.type == 'cuda') and len(args.gpus) > 1:
        model = nn.DataParallel(model, args.gpus)

    # ==================
    # Model Training
    # ==================
    best_dice = 0.
    mkdir(args.save_path)
    feature_net.requires_grad_(False)
    feature_net.eval()
    last_index = len(train_loader) - 1
    io.cprint(f"the loop times of a epoch is {(last_index + 1):d}")
    for epoch in tqdm(range(start_epoch, args.epoch)):
        Loss_per_epoch = 0.
        Loss_grad_per_epoch = 0.
        Loss_int_per_epoch = 0.
        Loss_seg_per_epoch = 0.
        model.train()
        seg_model.train()
        for index, image in enumerate(train_loader):
            img1 = image[0].to(device)  # T1
            img2 = image[1].to(device)  # T2
            seg = image[2].to(device)  # seg ground truth

            # feature extract
            feat1 = feature_net.encoder1(img1)
            feat2 = feature_net.encoder2(img2)

            # feature fuse
            out_feat = model(feat1, feat2)

            # image reconstruction
            out = feature_net.decoder(out_feat)
            # segmentation results
            seg_out = seg_model(out_feat)
            # loss
            # content loss
            Loss_grad = Loss_gradient(img1, img2, out)
            Loss_int = Loss_intensity(img1, img2, out)
            Loss1 = args.grad * Loss_grad + args.int * Loss_int
            Loss_grad_per_epoch += Loss_grad
            Loss_int_per_epoch += Loss_int
            # semantic loss
            Loss2 = Loss_Seg(seg_out, seg)
            Loss_seg_per_epoch += Loss2

            optimizer_seg.zero_grad()
            optimizer.zero_grad()
            Loss = Loss1 + args.seg * Loss2
            Loss_per_epoch += Loss
            Loss.backward()
            optimizer_seg.step()
            optimizer.step()

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        io.cprint(f"Epoch:[{epoch:d}/{args.epoch:d}]-----learning rate: {learning_rate:.10f}")
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Train-----LOSS:{(Loss_per_epoch / (len(train_loader))):.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Grad-------LOSS:{(Loss_grad_per_epoch / len(train_loader)):.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Intensity--LOSS:{(Loss_int_per_epoch / len(train_loader)):.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----SEG-------LOSS:{(Loss_seg_per_epoch / len(train_loader)):.4f}')
        writer.add_scalar('Train/Total_loss', Loss_per_epoch / (len(train_loader)), epoch)
        writer.add_scalar('Train/Grad_loss', Loss_grad_per_epoch / len(train_loader), epoch)
        writer.add_scalar('Train/L1norm_loss', Loss_int_per_epoch / len(train_loader), epoch)
        writer.add_scalar('Train/SEG_loss', Loss_seg_per_epoch / len(train_loader), epoch)

        scheduler.step()
        scheduler_seg.step()
        # ==================
        # Model Validation
        # ==================
        model.eval()
        seg_model.eval()
        with torch.no_grad():
            Loss_per_epoch = 0.
            Loss_grad_per_epoch = 0.
            Loss_int_per_epoch = 0.
            # Loss_per_per_epoch = 0.
            Loss_seg_per_epoch = 0.
            for index, image in enumerate(val_loader):
                img1 = image[0].to(device)  # T1
                img2 = image[1].to(device)  # T2
                seg = image[2].to(device)  # seg ground truth

                # feature extract
                feat1 = feature_net.encoder1(img1)
                feat2 = feature_net.encoder2(img2)

                # feature fuse
                out_feat = model(feat1, feat2)

                # image reconstruction
                out = feature_net.decoder(out_feat)
                # loss
                # content loss
                Loss_grad = Loss_gradient(img1, img2, out)
                Loss_int = Loss_intensity(img1, img2, out)
                Loss1 = args.grad * Loss_grad + args.int * Loss_int
                Loss_grad_per_epoch += Loss_grad
                Loss_int_per_epoch += Loss_int

                # semantic loss
                seg_out = seg_model(out_feat)
                seg_out = torch.softmax(seg_out, dim=1)
                seg_out = seg_out.squeeze(0).detach().cpu()
                seg = seg.squeeze(0).detach().cpu()
                dice = calculate_dice(seg_out, seg)
                io.cprint(
                    f'iter:[{index}/{len(val_loader)}]-----dice-----:{dice[0]:.4f},{dice[1]:.4f},{dice[2]:.4f},{dice[3]:.4f}')
                Loss_seg_per_epoch += dice.mean()

                Loss = Loss1
                Loss_per_epoch += Loss

        val_loss = Loss_per_epoch / (len(val_loader))
        seg_loss = Loss_seg_per_epoch / (len(val_loader))
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Val-------LOSS:{val_loss:.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Grad-------LOSS:{(Loss_grad_per_epoch / len(val_loader)):.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----Intensity--LOSS:{(Loss_int_per_epoch / len(val_loader)):.4f}')
        io.cprint(
            f'Epoch:[{epoch:d}/{args.epoch:d}]-----SEG-------LOSS:{seg_loss:.4f}')
        writer.add_scalar('Val/Total_loss', val_loss, epoch)
        writer.add_scalar('Val/Grad_loss', Loss_grad_per_epoch / len(val_loader), epoch)
        writer.add_scalar('Val/L1norm_loss', Loss_int_per_epoch / len(val_loader), epoch)
        writer.add_scalar('Val/SEG_loss', seg_loss, epoch)

        if seg_loss > best_dice:
            best_dice = seg_loss
            best_epoch = epoch
            # save best model
            state = {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            mkdir(os.path.join(save_path, './Fusion'))
            torch.save(state, os.path.join(save_path, './Fusion/model.pth'))

            state1 = {
                'optimizer': optimizer_seg.state_dict(),
                'model': seg_model.state_dict(),
            }
            mkdir(os.path.join(save_path, './Segmentation'))
            torch.save(state1, os.path.join(save_path, './Segmentation/model.pth'))
            io.cprint(f'save the new best epoch at epoch:{best_epoch} and the best dice is {best_dice:.4f}!!!')

        # ==================
        # Model Saving
        # ==================
        if epoch % 10 == 0 and epoch != 0 and epoch != args.epoch - 1:
            state = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }
            torch.save(state, os.path.join(save_path, 'Fusion_model_' +
                                           'epoch_' + str(epoch) + '_' + f'{val_loss:.4f}.pth'))
            state1 = {
                'epoch': epoch,
                'optimizer': optimizer_seg.state_dict(),
                'model': seg_model.state_dict(),
            }
            torch.save(state1, os.path.join(save_path, 'Segmentation_model_' + 'epoch_' + str(epoch) + '_' +
                                            f'{(Loss_seg_per_epoch / len(val_loader)):.4f}.pth'))


