"""Train SATNet for underwater image enhancement.

Example:
    python train.py \
        --dataroot_lq ./datasets/UIEB/train/input \
        --dataroot_gt ./datasets/UIEB/train/gt \
        --Val_lq ./datasets/UIEB/test/input \
        --Val_gt ./datasets/UIEB/test/gt \
        --save_path ./experiments
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.structure_dataset import EdgeTestDataset, EdgeTrainDataset
from losses.losses import CharbonnierLoss, GradientLoss, VGG19PerceptualLoss
from models.satnet import SATNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_psnr(img, imclean, data_range=1.0):
    img = img.detach().cpu().numpy().astype(np.float32)
    imclean = imclean.detach().cpu().numpy().astype(np.float32)
    return np.mean(
        [compare_psnr(imclean[i], img[i], data_range=data_range) for i in range(img.shape[0])]
    )


def batch_ssim(imgx, imgy, data_range=1.0):
    imgx = imgx.detach().cpu().numpy().astype(np.float32)
    imgy = imgy.detach().cpu().numpy().astype(np.float32)
    ssims = []
    for i in range(imgx.shape[0]):
        per_channel = [
            compare_ssim(imgx[i, c], imgy[i, c], data_range=data_range) for c in range(3)
        ]
        ssims.append(np.mean(per_channel))
    return np.mean(ssims)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SATNet (underwater image enhancement)")
    # data
    parser.add_argument("--dataroot_lq", type=str, default="./datasets/UIEB/train/input", help="train low-quality images")
    parser.add_argument("--dataroot_gt", type=str, default="./datasets/UIEB/train/gt", help="train reference images")
    parser.add_argument("--Val_lq", type=str, default="./datasets/UIEB/test/input", help="test low-quality images")
    parser.add_argument("--Val_gt", type=str, default="./datasets/UIEB/test/gt", help="test reference images")
    parser.add_argument("--patch_size", type=int, default=256, help="training patch size")
    parser.add_argument("--edge_type", type=str, default="sobel", choices=["sobel", "canny"], help="edge map type")
    parser.add_argument("--threads", type=int, default=0, help="number of data loader workers")
    # training
    parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
    parser.add_argument("--test_batch_size", type=int, default=1, help="testing batch size")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--n_save", type=int, default=10, help="save a checkpoint every n epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--lr_step", type=int, default=40, help="LR decay step (epochs)")
    parser.add_argument("--lr_gamma", type=float, default=0.8, help="LR decay factor")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    # environment
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--save_path", type=str, default="./experiments", help="output directory")
    parser.add_argument("--dataset_name", type=str, default="UIEB", help="dataset name (for logging)")
    parser.add_argument("--model_name", type=str, default="SATNet", help="experiment name")
    return parser.parse_args()


def main():
    opt = parse_args()
    set_seed(opt.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda_id)
    device = torch.device(f"cuda:{opt.cuda_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SATNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.4f} M")

    # losses (same combination as the paper configuration)
    loss_per = VGG19PerceptualLoss().to(device)
    loss_l1 = CharbonnierLoss().to(device)
    loss_grid = GradientLoss().to(device)
    lambda_per, lambda_l1, lambda_grid = 0.1, 1.0, 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)

    train_set = EdgeTrainDataset(opt)
    test_set = EdgeTestDataset(opt)
    train_loader = DataLoader(
        train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False
    )

    exp_dir = os.path.join(opt.save_path, opt.model_name)
    model_dir = os.path.join(exp_dir, "model")
    best_dir = os.path.join(exp_dir, "best")
    image_dir = os.path.join(exp_dir, "images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    best_psnr, best_ssim = 0.0, 0.0

    for epoch in range(opt.n_epochs):
        model.train()
        running_loss, psnrs, ssims = 0.0, [], []
        for step, data in enumerate(train_loader, 1):
            start = time.time()
            image = data["lq"].to(device)
            label = data["gt"].to(device)
            gt_edge = data["gt_edge"].to(device)

            optimizer.zero_grad()
            output = model(image, is_train=True)
            pred, pred_edge = output[0], output[1]

            loss_vgg = loss_per(pred, label)
            loss_l1_val = loss_l1(pred, label)
            loss_grid_val = loss_grid(pred, label)
            loss_edge = loss_l1(pred_edge, gt_edge)
            loss = (
                lambda_per * loss_vgg
                + lambda_l1 * loss_l1_val
                + lambda_grid * loss_grid_val
                + lambda_l1 * loss_edge
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_pred = torch.clamp(pred, 0.0, 1.0)
            psnrs.append(batch_psnr(train_pred, label))
            ssims.append(batch_ssim(train_pred, label))

            if step % 50 == 0 or step == len(train_loader):
                sys.stdout.write(
                    f"\r[epoch {epoch + 1}/{opt.n_epochs}] "
                    f"[batch {step}/{len(train_loader)}] "
                    f"[time {time.time() - start:.3f}s] "
                    f"[PSNR {psnrs[-1]:.4f}] [SSIM {ssims[-1]:.4f}] "
                    f"[loss {loss.item():.4f}]"
                )
                sys.stdout.flush()
        print()
        scheduler.step()

        # validation
        model.eval()
        val_psnr, val_ssim = [], []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                image = data["lq"].to(device)
                label = data["gt"].to(device)
                pred = model(image, is_train=True)[0]
                pred = torch.clamp(pred, 0.0, 1.0)
                val_psnr.append(batch_psnr(pred, label))
                val_ssim.append(batch_ssim(pred, label))
                if random.randint(1, 50) == 1:
                    save_image(pred, os.path.join(image_dir, f"epoch_{epoch}_batch_{i}.png"))

        mean_psnr = float(np.mean(val_psnr))
        mean_ssim = float(np.mean(val_ssim))
        print(f"[epoch {epoch + 1}/{opt.n_epochs}] "
              f"train loss {running_loss / len(train_loader):.4f} | "
              f"val PSNR {mean_psnr:.4f} | val SSIM {mean_ssim:.4f}")

        if (epoch + 1) % opt.n_save == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{epoch + 1}.pth"))
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(model.state_dict(), os.path.join(best_dir, "best_psnr.pth"))
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save(model.state_dict(), os.path.join(best_dir, "best_ssim.pth"))

    print(f"Training finished. Best PSNR {best_psnr:.4f}, best SSIM {best_ssim:.4f}.")


if __name__ == "__main__":
    main()
