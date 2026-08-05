"""Loss functions used for training SATNet.

All losses follow the standard ``(prediction, target)`` signature and expect
float tensors in [0, 1].
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

__all__ = ["CharbonnierLoss", "EdgeLoss", "GradientLoss", "PSNRLoss", "VGG19PerceptualLoss"]


class CharbonnierLoss(nn.Module):
    """Robust L1-style loss: mean(sqrt((x - y)^2 + eps^2))."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class GradientLoss(nn.Module):
    """L1 loss between Laplacian gradients of prediction and target."""

    def __init__(self):
        super().__init__()
        kernel = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]] * 3
        kernel = torch.FloatTensor(kernel).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        gx = F.conv2d(x, self.weight, groups=3)
        gy = F.conv2d(y, self.weight, groups=3)
        return self.criterion(gx, gy)


class EdgeLoss(nn.Module):
    """Charbonnier loss on Laplacian-pyramid edges (blur-diff edges)."""

    def __init__(self):
        super().__init__()
        k = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, x, y):
        return self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))


class VGG19PerceptualLoss(nn.Module):
    """Perceptual loss in VGG19 feature space (``conv5_2`` by default)."""

    _LAYER_NAMES = {
        "3": "conv1_2",
        "8": "conv2_2",
        "13": "conv3_2",
        "22": "conv4_2",
        "31": "conv5_2",
    }

    def __init__(self, layer="conv5_2"):
        super().__init__()
        try:
            self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except (AttributeError, TypeError):
            # Fallback for older torchvision versions.
            self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        self.layer = layer

    def get_features(self, image):
        x = image
        for name, module in self.vgg._modules.items():
            x = module(x)
            if self._LAYER_NAMES.get(name) == self.layer:
                return x
        raise ValueError(f"Layer {self.layer} not found in VGG19")

    def forward(self, pred, true):
        return torch.mean((self.get_features(pred) - self.get_features(true)) ** 2)


class PSNRLoss(nn.Module):
    """Negative PSNR used as an optimization objective."""

    def __init__(self, loss_weight=1.0, toY=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.scale = 10.0 / np.log(10.0)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)

    def forward(self, pred, target):
        if self.toY:
            coef = self.coef.to(pred.device)
            pred = (pred * coef).sum(dim=1, keepdim=True) + 16.0
            target = (target * coef).sum(dim=1, keepdim=True) + 16.0
            pred, target = pred / 255.0, target / 255.0
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return -self.loss_weight * self.scale * torch.log(mse + 1e-8).mean()
