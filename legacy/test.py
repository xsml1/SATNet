import argparse
import os
import cv2

import torch
import torchvision.transforms as transforms
from Model.restormer_cascaded_arch import Restormer_Cascaded, Restormer
from model_cascaded import Cascaded_Spec
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from torchvision.utils import save_image
import time
torch.backends.cudnn.benchmark = True
from thop import profile
from structure_dataset import EdgeTestDataset

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_SSIM(imgx, imgy, data_range):
    imgx = imgx.data.cpu().numpy().astype(np.float32)
    imgy = imgy.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(imgx.shape[0]):
        img1 = imgx[i, :, :, :]
        img2 = imgy[i, :, :, :]
        ssim1 = compare_ssim(img1[0, :, :], img2[0, :, :], data_range=data_range)
        ssim2 = compare_ssim(img1[1, :, :], img2[1, :, :], data_range=data_range)
        ssim3 = compare_ssim(img1[2, :, :], img2[2, :, :], data_range=data_range)
        SSIM += (ssim1+ssim2+ssim3)/3.0
    return (SSIM / imgx.shape[0])


# Testing settings
parser = argparse.ArgumentParser(description='Spectroformer-implementation')
parser.add_argument('--dataset', default='UIEB', required=False, help='facades')
parser.add_argument('--model_name', default='Cascaded_Spec', required=False, help='facades')
parser.add_argument('--save_path', default='/home/data1/xj/Full/Results', required=False, help='facades')
parser.add_argument('--ckpt', default='/home/data1/xj/Full/Experiment/UIEBD/Cascaded_Spec/best/best_psnr.pth', required=False, help='facades')
parser.add_argument('--Val_gt', type=str, default='/home/data1/fjy/IR_datasets/test/underwater/labels', help='dataroot_gt')
parser.add_argument('--Val_lq', type=str, default='/home/data1/fjy/IR_datasets/test/underwater/images', help='dataroot_lq')
parser.add_argument('--edge_type', type=str, default='sobel', help='edge_type')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--cuda_id', type=int, default=2, help='cuda_id')
parser.add_argument('--cuda', action='store_false', help='use cuda')
opt = parser.parse_args()


save_path = os.path.join(os.path.join(opt.save_path, opt.dataset), opt.model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

device_id = opt.cuda_id
if torch.cuda.is_available():
    torch.cuda.set_device(opt.cuda_id)


model = Cascaded_Spec().to(device_id)
model.load_state_dict(torch.load(opt.ckpt))
model.eval()

test_set = EdgeTestDataset(opt)
test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

start = time.time()
avg_psnr=0
a = 0
import lpips
model1 = lpips.LPIPS(net='vgg')
model1.to(device_id)
loss_fn_alex = lpips.LPIPS(net='vgg')
PSNR = []
SSIM = []
LPIPS = []

with torch.no_grad():
    for i, data in enumerate(test_loader):

        image, label = data['lq'].to(device_id), data['gt'].to(device_id)
        edge = data['edge'].to(device_id)
        filename = data['filename'][0]
        img_name = filename + '.png'

        pred = model(image, edge)
        psnr = batch_PSNR(pred, label, 1.)
        ssim = batch_SSIM(pred, label, 1.)
        Lpips = model1.forward(pred, label).item()

        print(i, psnr, ssim, Lpips)
        PSNR.append(psnr)
        SSIM.append(ssim)
        LPIPS.append(Lpips)
        pred = pred.detach().squeeze(0).cpu()

        save_image(pred, os.path.join(save_path, img_name))

        edge_path = os.path.join(save_path, 'Edge')
        if not os.path.exists(edge_path):
            os.makedirs(edge_path)
        e_path = os.path.join(edge_path, img_name)
        save_image(edge, e_path)


PSNR = np.array(PSNR)
SSIM = np.array(SSIM)
LPIPS = np.array(LPIPS)
psnr_value = PSNR.mean()
ssim_value = SSIM.mean()
lpips_value = LPIPS.mean()



print("PSNR value:{}\n".format(psnr_value))
print("SSIM value:{}\n".format(ssim_value))
print("LPIPS value:{}\n".format(lpips_value))

print('done....')

'''
restormer:
PSNR value:23.87116336335499

SSIM value:0.9087131837666264

LPIPS value:0.09239690639078617


PSNR value:24.147880650972795

SSIM value:0.9060302603684725

LPIPS value:0.09019248051764933
'''

'''
Cascaded_Spec:
PSNR value:24.65046390385439

SSIM value:0.9111835762318407

LPIPS value:0.08520402101750829
'''