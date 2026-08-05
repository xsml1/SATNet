"""Datasets for training and evaluating SATNet.

Data layout convention (paired folders, same file names in both):

    datasets/<name>/train/input/*.png   low-quality (underwater) images
    datasets/<name>/train/gt/*.png      reference (clear) images
    datasets/<name>/test/input/*.png
    datasets/<name>/test/gt/*.png

* :class:`EdgeTrainDataset` also returns a target edge map (Canny or Sobel)
  computed from the GT image, which supervises the learned edge branch.
* :class:`EdgeTestDataset` returns an edge map computed from the low-quality
  input, which is used by some ablation variants that take edges as input.
* :class:`TestDataset` is a plain low-quality / (optional) GT loader used by
  the evaluation script.

All images are returned as float tensors in [0, 1] with shape [3, H, W].
Edge maps are single-channel tensors in [0, 1].
"""

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ["EdgeTrainDataset", "EdgeTestDataset", "TestDataset", "build_edge_map"]


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _pair_paths(lq_dir, gt_dir):
    """Pair low-quality paths with GT paths by file name (stem)."""
    lq_paths = sorted(glob(os.path.join(lq_dir, "*")))
    assert lq_paths, f"No images found under: {lq_dir}"
    if gt_dir is None:
        return lq_paths, {}
    gt_by_stem = {_stem(p): p for p in sorted(glob(os.path.join(gt_dir, "*")))}
    paired = {}
    for p in lq_paths:
        stem = _stem(p)
        if stem in gt_by_stem:
            paired[p] = gt_by_stem[stem]
    assert paired, f"No matching GT found under: {gt_dir}"
    return list(paired.keys()), paired


def build_edge_map(img_bgr, edge_type="sobel", low=100, high=200):
    """Extract a single-channel edge map (float32 in [0, 1]) from a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if edge_type == "canny":
        edge = cv2.Canny(gray, low, high)
    elif edge_type == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip(cv2.magnitude(gx, gy), 0, 255)
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")
    return (edge / 255.0).astype(np.float32)


class _PairedDataset(Dataset):
    def __init__(self, lq_dir, gt_dir=None, patch_size=256, edge_type="sobel", train=False):
        super().__init__()
        self.lq_paths, self.gt_by_lq = _pair_paths(lq_dir, gt_dir)
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.edge_type = edge_type
        self.train = train
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.lq_paths)

    @staticmethod
    def _read(path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        return img

    def _prepare(self, img):
        """Make sure the image is at least patch_size x patch_size (train only)."""
        if not self.train:
            return img
        h, w = img.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            scale = self.patch_size / min(h, w)
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_LINEAR)
        return img

    def _crop_and_augment(self, imgs):
        """Random crop (+ flip) applied consistently to images and edge maps."""
        if not self.train:
            return imgs
        h, w = imgs[0].shape[:2]
        size = self.patch_size
        top = np.random.randint(0, h - size) if h > size else 0
        left = np.random.randint(0, w - size) if w > size else 0
        imgs = [im[top:top + size, left:left + size] for im in imgs]
        if np.random.rand() < 0.5:
            imgs = [cv2.flip(im, 1) for im in imgs]
        if np.random.rand() < 0.5:
            imgs = [cv2.flip(im, 0) for im in imgs]
        return imgs

    def _load_pair(self, index):
        lq_path = self.lq_paths[index]
        lq = self._prepare(self._read(lq_path))
        gt = None
        if self.gt_by_lq:
            gt = self._prepare(self._read(self.gt_by_lq[lq_path]))
        if self.train:
            imgs = self._crop_and_augment([lq] + ([gt] if gt is not None else []))
            lq, gt = imgs[0], (imgs[1] if gt is not None else None)
        return lq, gt, _stem(lq_path)

    def _edge_tensor(self, img_bgr):
        edge = build_edge_map(img_bgr, self.edge_type)
        return torch.from_numpy(edge[None])  # [1, H, W]


class EdgeTrainDataset(_PairedDataset):
    """Paired training set with a GT edge map for edge supervision."""

    def __init__(self, opt):
        super().__init__(
            lq_dir=opt.dataroot_lq,
            gt_dir=opt.dataroot_gt,
            patch_size=opt.patch_size,
            edge_type=opt.edge_type,
            train=True,
        )

    def __getitem__(self, index):
        lq, gt, _ = self._load_pair(index)
        assert gt is not None, "EdgeTrainDataset requires paired GT images."
        gt_edge = self._edge_tensor(gt)
        return {
            "lq": self.to_tensor(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)),
            "gt": self.to_tensor(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)),
            "gt_edge": gt_edge,
        }


class EdgeTestDataset(_PairedDataset):
    """Test set that also provides an edge map of the low-quality input."""

    def __init__(self, opt):
        super().__init__(
            lq_dir=opt.Val_lq,
            gt_dir=opt.Val_gt,
            edge_type=opt.edge_type,
            train=False,
        )

    def __getitem__(self, index):
        lq, gt, stem = self._load_pair(index)
        sample = {
            "lq": self.to_tensor(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)),
            "edge": self._edge_tensor(lq),
            "filename": stem,
        }
        if gt is not None:
            sample["gt"] = self.to_tensor(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
        return sample


class TestDataset(_PairedDataset):
    """Plain test set: low-quality images with optional GT references."""

    def __init__(self, lq_dir, gt_dir=None):
        super().__init__(lq_dir=lq_dir, gt_dir=gt_dir, train=False)

    def __getitem__(self, index):
        lq, gt, stem = self._load_pair(index)
        sample = {"lq": self.to_tensor(cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)), "filename": stem}
        if gt is not None:
            sample["gt"] = self.to_tensor(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))
        return sample
