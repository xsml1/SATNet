import argparse
import os
import cv2

import torch
import torchvision.transforms as transforms
from Model.Aqua import Aqua
from Model.WWE_UIE import WWE_UIE
from Model.CDF_UIE_arch import CDFNet
from Abalation_Model.Full_model import SGCTransformer
from UWEnhancer.UWEnhancer import UWEnhancer
from Model.Uformer import Uformer
from Model.Restormer import Restormer
from Final_model_AGSSF import SpectroFormer
from Model.GuidedHybSensUIR import GuidedHybSensUIR
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from torchvision.utils import save_image
import time
from metrics.uiqm_utils import getUIQM
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
parser.add_argument('--dataset', default='UCCS', required=False, help='facades')
parser.add_argument('--with_gt', type=bool, default=False, help='w/wo the reference image')
parser.add_argument('--model_name', default='UWEnhancer', required=False, help='facades')
parser.add_argument('--save_path', default='/home/data1/xj/Full/UIEB_Exp_800/Results', required=False, help='facades')
parser.add_argument('--ckpt', default='/home/data1/xj/Full/UIEB_Exp_800/UWEnhancer/Stage1/model_80.pth', required=False, help='facades')
# parser.add_argument('--Val_gt', type=str, default='/home/data1/xj/Full/datasets/UIEB/test/gt', help='dataroot_gt')
# parser.add_argument('--Val_lq', type=str, default='/home/data1/xj/Full/datasets/UIEB/test/input', help='dataroot_lq')

parser.add_argument('--Val_gt', type=str, default='/home/data1/xj/Full/Test_data/LSUI/gt', help='dataroot_gt')
parser.add_argument('--Val_lq', type=str, default='/home/data1/xj/Full/Test_data/UCCS/input', help='dataroot_lq')
parser.add_argument('--edge_type', type=str, default='sobel', help='edge_type')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--cuda_id', type=int, default=2, help='cuda_id')
parser.add_argument('--cuda', action='store_false', help='use cuda')
opt = parser.parse_args()

with_gt = opt.with_gt
save_path = os.path.join(os.path.join(opt.save_path, opt.dataset), opt.model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

device_id = torch.device(f'cuda:{opt.cuda_id}')
if torch.cuda.is_available():

    torch.cuda.set_device(opt.cuda_id)


model = UWEnhancer()
checkpoint = torch.load(opt.ckpt, map_location='cpu')
model.load_state_dict(checkpoint)
model = model.to(device_id)
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
UIQM = []
total_time = 0
cnt = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        if with_gt:
            image, label = data['lq'].to(device_id), data['gt'].to(device_id)
            filename = data['filename'][0]
            img_name = filename + '.png'
            factor = 8
            # Padding in case images are not multiples of 8
            _, _, h, w = image.shape


            # pred = model(image_r)

            output1, pred, B_c, metric_z, z, z_J_c, exp_negative_beta_b = model(image)  # if CDFNet

            # pred, _ = model(image)

            # print(i,pred.shape, label.shape)
            psnr = batch_PSNR(pred, label, 1.)
            ssim = batch_SSIM(pred, label, 1.)
            Lpips = model1.forward(pred, label).item()

            np_pred = pred.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
            np_pred = (np_pred * 255).astype('uint8')
            uiqm = getUIQM(np_pred)
            UIQM.append(uiqm)

            print(i, psnr, ssim, Lpips, uiqm)
            PSNR.append(psnr)
            SSIM.append(ssim)
            LPIPS.append(Lpips)
            pred = pred.detach().squeeze(0).cpu()

            save_image(pred, os.path.join(save_path, img_name))
        else:
            image = data['lq'].to(device_id)
            filename = data['filename'][0]
            img_name = filename + '.png'
            factor = 8
            # Padding in case images are not multiples of 8
            _, _, h, w = image.shape


            output1, pred, B_c, metric_z, z, z_J_c, exp_negative_beta_b = model(image)



            np_pred = pred.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
            np_pred = (np_pred * 255).astype('uint8')
            uiqm = getUIQM(np_pred)
            UIQM.append(uiqm)

            print(i, uiqm)
            pred = pred.detach().squeeze(0).cpu()

            save_image(pred, os.path.join(save_path, img_name))

# print('time 4 infer:', total_time/cnt, cnt)
# PSNR = np.array(PSNR)
# SSIM = np.array(SSIM)
# LPIPS = np.array(LPIPS)
UIQM = np.array(UIQM)
# psnr_value = PSNR.mean()
# ssim_value = SSIM.mean()
# lpips_value = LPIPS.mean()
uiqm_value = UIQM.mean()



# print("PSNR value:{}\n".format(psnr_value))
# print("SSIM value:{}\n".format(ssim_value))
# print("LPIPS value:{}\n".format(lpips_value))
print("UIQM value:{}\n".format(uiqm_value))

print('done....')

'''
WWE_UIE
WACV2026
UIEB:
PSNR value:24.958582976039903

SSIM value:0.9191729887618695

LPIPS value:0.0773571493724982

UIQM value:2.997992943349517

LSUI:
PSNR value:21.863504680384693

SSIM value:0.8450.8457752880943642

LPIPS value:0.18742525838315488

UIQM value:2.96192076584409
'''

'''
Aqua 100
TCE2025
time 4 infer: 0.15716227520717663 89
PSNR value:25.278489747567964

SSIM value:0.9298110634738904

LPIPS value:0.06936859204433858

UIQM value:2.9612299025222106

LSUI
time 4 infer: 0.09323364750185706 399
PSNR value:21.623468280941996

SSIM value:0.847905034229587

LPIPS value:0.18767654458526523

UIQM value:2.832202712805048
'''

'''
CDFNet 300
TGRS2025
UIEB
time 4 infer: 0.022963687275232892
PSNR value:25.29484681961271

SSIM value:0.9233159119491219

LPIPS value:0.075791244312293

UIQM value:2.9466747941250593


LSUI
time 4 infer: 0.023023422499348346
PSNR value:21.775000540018468

SSIM value:0.8458326573496777

LPIPS value:0.1840874642552808

UIQM value:2.943127331236013

'''

'''
GuidedHybSensUIR

TCSVT2025 

PSNR value:24.709705443975064

SSIM value:0.9248030930205469

LPIPS value:0.07419299646798107

UIQM value:3.0624088333618795

time 4 infer: 0.08463299662845775 399
PSNR value:21.890219589539228

SSIM value:0.8529888619170266

LPIPS value:0.1834622295340523

UIQM value:2.964824887886694
'''

'''
SGCTrans

time 4 infer: 0.07582

PSNR value:25.590252191315045

SSIM value:0.93213348156439

LPIPS value:0.06311782580903835

UIQM value:2.9605433306278925
LSUI:

PSNR value:21.655352377047347

SSIM value:0.8524408966577148

LPIPS value:0.17148503418080507

UIQM value:2.902037837438419


time 4 infer: 0.07534627926379517 399
PSNR value:22.33770457923878

SSIM value:0.8537962163685553

LPIPS value:0.17727120746858419

UIQM value:2.8674217583901065

'''

'''
SGCTrans_2

PSNR value:25.925791472453717

SSIM value:0.9310251903354724

LPIPS value:0.06294028492540948

UIQM value:2.931109512787267

done....

PSNR value:25.925570901555663

SSIM value:0.9310273992318376

LPIPS value:0.06293960894561476

UIQM value:2.930269389859956
'''