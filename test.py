"""Evaluate a trained SATNet checkpoint on a test set.

Computes PSNR, SSIM, LPIPS (if installed) and UIQM, saves the enhanced
images, and reports dataset averages. The test sets follow the paper:
Test-U90 (UIEB), Test-L400 (LSUI), Test-E330 (EUVP), Test-U300 (UCCS) and
Test-C60 (UIEB Challenge).

Example:
    python test.py \
        --Val_lq ./datasets/UIEB/test/input \
        --Val_gt ./datasets/UIEB/test/gt \
        --ckpt ./experiments/SATNet/best/best_psnr.pth \
        --save_path ./results
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data.structure_dataset import TestDataset
from metrics.uiqm_utils import getUIQM
from models.satnet import SATNet


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


def pad_to_multiple(x, factor=8):
    """Pad the spatial size of x to a multiple of `factor` (reflection padding)."""
    _, _, h, w = x.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def remove_padding(x, pad_h, pad_w):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def parse_args():
    parser = argparse.ArgumentParser(description="Test SATNet (underwater image enhancement)")
    parser.add_argument("--Val_lq", type=str, default="./datasets/UIEB/test/input", help="test low-quality images (e.g. Test-U90 / Test-L400 input)")
    parser.add_argument("--Val_gt", type=str, default="./datasets/UIEB/test/gt", help="test reference images, optional for non-reference sets (e.g. Test-E330 / Test-U300 / Test-C60)")
    parser.add_argument("--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--save_path", type=str, default="./results", help="output directory")
    parser.add_argument("--threads", type=int, default=0, help="number of data loader workers")
    parser.add_argument("--test_batch_size", type=int, default=1, help="testing batch size")
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device id")
    return parser.parse_args()


def load_model(ckpt_path, device):
    model = SATNet().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, SATNet):
        model = ckpt.to(device)
    elif isinstance(ckpt, dict):
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    model.eval()
    return model


def main():
    opt = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda_id)
    device = torch.device(f"cuda:{opt.cuda_id}" if torch.cuda.is_available() else "cpu")

    model = load_model(opt.ckpt, device)
    gt_dir = opt.Val_gt if os.path.isdir(opt.Val_gt) else None
    test_set = TestDataset(lq_dir=opt.Val_lq, gt_dir=gt_dir)
    test_loader = DataLoader(
        test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False
    )

    os.makedirs(opt.save_path, exist_ok=True)
    with_gt = gt_dir is not None

    try:
        import lpips

        lpips_model = lpips.LPIPS(net="vgg").to(device)
    except ImportError:
        lpips_model = None
        print("lpips not installed; skipping LPIPS metric.")

    psnrs, ssims, lpipses, uiqms = [], [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data["lq"].to(device)
            filename = data["filename"][0]

            image_padded, pad_h, pad_w = pad_to_multiple(image, factor=8)
            pred = model(image_padded)
            pred = remove_padding(pred, pad_h, pad_w)
            pred = torch.clamp(pred, 0.0, 1.0)

            save_image(pred, os.path.join(opt.save_path, f"{filename}.png"))

            if with_gt:
                label = data["gt"].to(device)
                psnr = batch_psnr(pred, label)
                ssim = batch_ssim(pred, label)
                psnrs.append(psnr)
                ssims.append(ssim)
                if lpips_model is not None:
                    lpipses.append(lpips_model(pred, label).item())
                else:
                    lpipses.append(float("nan"))
                print(f"[{i}] {filename}: PSNR {psnr:.4f}, SSIM {ssim:.4f}")

            np_pred = pred.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
            uiqms.append(getUIQM((np_pred * 255).astype(np.uint8)))

    if with_gt:
        print("\nAverage over dataset:")
        print(f"  PSNR  : {np.mean(psnrs):.4f}")
        print(f"  SSIM  : {np.mean(ssims):.4f}")
        print(f"  LPIPS : {np.nanmean(lpipses):.4f}")
    print(f"  UIQM  : {np.mean(uiqms):.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
