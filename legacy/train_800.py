import os
import time
from torch.optim import lr_scheduler
from Abalation_Model.Full_model import SGCTransformer

import argparse
import random
from lossfun import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import sys
from structure_dataset import EdgeTrainDataset, EdgeTestDataset

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

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
        SSIM = (ssim1+ssim2+ssim3)/3.0
    return (SSIM / imgx.shape[0])

# Training settings
parser = argparse.ArgumentParser(description='Spectroformer-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--finetune', default=False, help='to finetune')
parser.add_argument('--cuda_id', type=int, default=1, help='cuda_id')
parser.add_argument('--n_epochs', type=int, default=500, help='n_epochs')
parser.add_argument('--n_save', type=int, default=10, help='save time')
parser.add_argument('--epoch_count', type=int, default=1,help='the starting epoch count')
parser.add_argument('--niter', type=int, default=1500, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=1500, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate for adam')
# parser.add_argument('--dataroot_gt', type=str, default='/home/data1/xj/Full/datasets/UIEB/train/gt', help='dataroot_gt')
# parser.add_argument('--dataroot_lq', type=str, default='/home/data1/xj/Full/datasets/UIEB/train/input', help='dataroot_lq')
parser.add_argument('--dataroot_gt', type=str, default='/home/data10t/xujie/SGCTrans/datasets/UIEB/train/gt', help='dataroot_gt')
parser.add_argument('--dataroot_lq', type=str, default='/home/data10t/xujie/SGCTrans/datasets/UIEB/train/input', help='dataroot_lq')
parser.add_argument('--edge_type', type=str, default='sobel', help='edge_type')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')

# parser.add_argument('--Val_gt', type=str, default='/home/data1/xj/Full/datasets/UIEB/test/gt', help='dataroot_gt')
# parser.add_argument('--Val_lq', type=str, default='/home/data1/xj/Full/datasets/UIEB/test/input', help='dataroot_lq')
parser.add_argument('--Val_gt', type=str, default='/home/data10t/xujie/SGCTrans/datasets/UIEB/test/gt', help='dataroot_gt')
parser.add_argument('--Val_lq', type=str, default='/home/data10t/xujie/SGCTrans/datasets/UIEB/test/input', help='dataroot_lq')
# parser.add_argument('--save_path', type=str, default='/home/data1/xj/Full/UIEB_Exp_800', help='save_path')
parser.add_argument('--save_path', type=str, default='/home/data10t/xujie/SGCTrans/UIEB_Exp_800', help='save_path')
parser.add_argument('--dataset_name', type=str, default='UIEB_800', help='save_path')
parser.add_argument('--model_name', type=str, default='SGCTransformer', help='save_path')
parser.add_argument('--lr_decay_iters', type=int, default=1500, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--edge_loss', default=False, help='apply edge loss for training')
parser.add_argument('--psnr', default=15.9509, help='psnr-value')
parser.add_argument('--edge_loss_type', default='canny', help='apply canny or sobel loss loss for training')
opt = parser.parse_args()

# save_path = os.path.join(opt.save_path, opt.dataset_name)
save_path = os.path.join(opt.save_path, opt.model_name)
device_id = opt.cuda_id
if torch.cuda.is_available():
    torch.cuda.set_device(opt.cuda_id)

n_epochs = opt.n_epochs
start_epoch = 0


model = SGCTransformer()
model.to(device_id)

use_pretrain = False
if use_pretrain:
    model.load_state_dict(torch.load(
        "/media/aa/a4b46d17-0f49-4392-98d6-49a5c9dee8e9/xujie/MFCR/saved_models_21_3/best/best_psnr_%d.pth" % (start_epoch)))
    print('successfully load pretrained model！')
else:
    print('No pretrain model found, training will start from scratch！')

Loss_per = VGG19_PercepLoss()
Loss_L1 = CharbonnierLoss()
Loss_Grid = Gradient_Loss()

lambda_per = 0.1
lambda_l1 = 1
lambda_grid = 0.02
Loss_per.to(device_id)
Loss_L1.to(device_id)
Loss_Grid.to(device_id)


# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)

best_loss = 99999999.0


best_psnr = 0.
best_ssim = 0.
print('start training @ epoch: {} !!!'.format(start_epoch))

train_set = EdgeTrainDataset(opt)
test_set = EdgeTestDataset(opt)
train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)
ckpt_path = None
for e in range(start_epoch, n_epochs):
    running_loss = 0
    PSNR = []
    SSIM = []
    step = 0

    for i, data in enumerate(train_loader):

        start_time = time.time()

        image, label = data['lq'].to(device_id), data['gt'].to(device_id)
        gt_edge = data['gt_edge'].to(device_id)

        step += 1
        optimizer.zero_grad()
        output = model(image, is_train=True)
        out, pred_edge = output[0], output[1]
        loss_vgg = Loss_per(out, label)
        loss_l1 = Loss_L1(out, label)
        loss_grid = Loss_Grid(out, label)

        loss_edge = Loss_L1(pred_edge, gt_edge)
        loss = lambda_per*loss_vgg + lambda_l1*loss_l1 + lambda_grid*loss_grid+ lambda_l1*loss_edge
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_pre = torch.clamp(out, 0., 1.)
        psnr_train = batch_PSNR(train_pre, label, 1.)
        ssim_train = batch_SSIM(train_pre, label, 1.)
        PSNR = np.append(PSNR, psnr_train)
        SSIM = np.append(SSIM, ssim_train)

        len_batch = len(train_loader)
        batch_time = time.time()
        batch_use_time = batch_time - start_time

        sys.stdout.write(
            '\r[epoch:%d/%d], [batch:%d/%d],[Time4batch:%f],[PSNR:%f], [SSIM: %f],[VGG_loss:%f],[L1_loss:%f],[Grid_loss:%f],[Edge_loss:%f]\n'
            % (
                e,
                n_epochs,
                step,
                len_batch,
                batch_use_time,
                psnr_train,
                ssim_train,
                loss_vgg.item(),
                loss_l1.item(),
                loss_grid.item(),
                loss_edge.item()
            )
        )
    scheduler.step()

    psnr_value = PSNR.mean()
    ssim_value = SSIM.mean()

    if (e + 1) % opt.n_save == 0:
        p_save = os.path.join(save_path, "model")
        if not os.path.exists(p_save):
            os.makedirs(p_save)
        pth_str = "model_%d.pth"% (e + 1)
        pth_save = os.path.join(p_save, pth_str)
        torch.save(model.state_dict(), pth_save)

    with torch.no_grad():
        model.eval()
        test_ssim = []
        test_psnr = []

        for i, data1 in enumerate(test_loader):
            image, label = data1['lq'].to(device_id), data1['gt'].to(device_id)
            edge = data1['edge'].to(device_id)
            output = model(image, is_train=True)
            pred, pred_edge = output[0], output[1]
            p_img = os.path.join(save_path, "images")
            if not os.path.exists(p_img):
                os.makedirs(p_img)
            iters = '%s_%s.png' % (e, i)
            img_save = os.path.join(p_img, iters)
            if random.randint(1, 50) == 1:
                save_image(pred, img_save, nrow=5, normalize=True)
            batch_psnr = batch_PSNR(pred, label, 1.)
            batch_ssim = batch_SSIM(pred, label, 1.)
            test_psnr = np.append(test_psnr, batch_psnr)
            test_ssim = np.append(test_ssim, batch_ssim)

        psnr_value = test_psnr.mean()
        ssim_value = test_ssim.mean()
        best_save = os.path.join(save_path, "best")
        psrn_save = os.path.join(best_save, "best_psnr.pth")
        ssim_save = os.path.join(best_save, "best_ssim.pth")
        if psnr_value > best_psnr:
            best_psnr = psnr_value
            if not os.path.exists(best_save):
                os.makedirs(best_save)
            torch.save(model.state_dict(), psrn_save)
            ckpt_path = psrn_save
        if ssim_value > best_ssim:
            best_ssim = ssim_value
            torch.save(model.state_dict(),ssim_save)

print('Done Training...')

model.load_state_dict(torch.load(ckpt_path))
model.eval()

avg_psnr=0
a = 0
import lpips
model1 = lpips.LPIPS(net='vgg')
model1.to(device_id)
loss_fn_alex = lpips.LPIPS(net='vgg')
PSNR = []
SSIM = []
LPIPS = []
r_path = "/home/data1/xj/Full/UIEB_Exp_800/Results"
result_path = os.path.join(os.path.join(opt.r_path, opt.dataset_name), opt.model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
with torch.no_grad():
    for i, data in enumerate(test_loader):

        image, label = data['lq'].to(device_id), data['gt'].to(device_id)
        filename = data['filename'][0]
        img_name = filename + '.png'
        pred, _ = model(image)


        psnr = batch_PSNR(pred, label, 1.)
        ssim = batch_SSIM(pred, label, 1.)
        Lpips = model1.forward(pred, label).item()

        print(i, psnr, ssim, Lpips)
        PSNR.append(psnr)
        SSIM.append(ssim)
        LPIPS.append(Lpips)
        pred = pred.detach().squeeze(0).cpu()

        save_image(pred, os.path.join(result_path, img_name))




PSNR = np.array(PSNR)
SSIM = np.array(SSIM)
LPIPS = np.array(LPIPS)
psnr_value = PSNR.mean()
ssim_value = SSIM.mean()
lpips_value = LPIPS.mean()



print("PSNR value:{}\n".format(psnr_value))
print("SSIM value:{}\n".format(ssim_value))
print("LPIPS value:{}\n".format(lpips_value))

print('Done Testing...')