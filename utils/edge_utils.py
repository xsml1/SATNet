"""Offline edge-map generation helpers.

These utilities are used to pre-process datasets (e.g. white balance + CLAHE)
and to extract Canny / Sobel edge maps for edge supervision or evaluation.
"""

import argparse
import os
from glob import glob

import cv2
import numpy as np

__all__ = ["white_balance", "apply_clahe", "canny_edge", "sobel_edge", "resize_img", "process"]


def white_balance(img):
    """Simple gray-world white balance (in-place on a copy)."""
    img = img.astype(np.float32)
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] = img[:, :, 0] * (avg_gray / max(avg_b, 1e-6))
    img[:, :, 1] = img[:, :, 1] * (avg_gray / max(avg_g, 1e-6))
    img[:, :, 2] = img[:, :, 2] * (avg_gray / max(avg_r, 1e-6))
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_clahe(img):
    """CLAHE contrast enhancement in LAB space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def canny_edge(img, low=100, high=200):
    """Canny edge map (uint8)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)


def sobel_edge(img):
    """Sobel magnitude edge map (uint8)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


def resize_img(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def process(args):
    """Batch-generate edge maps for a paired dataset:
    GT edges, input edges, and edges of the WB+CLAHE enhanced input.
    """
    gt_paths = sorted(glob(os.path.join(args.Val_gt, "*")))
    lq_paths = sorted(glob(os.path.join(args.Val_lq, "*")))
    assert len(gt_paths) == len(lq_paths), "GT and input image counts differ."

    out_gt = os.path.join(args.save_dir, "GT")
    out_wb_clahe = os.path.join(args.save_dir, "Pred")
    out_input = os.path.join(args.save_dir, "Input")
    for d in (out_gt, out_wb_clahe, out_input):
        os.makedirs(d, exist_ok=True)

    for gt_path, lq_path in zip(gt_paths, lq_paths):
        name = os.path.basename(gt_path)
        gt = resize_img(cv2.imread(gt_path))
        lq = resize_img(cv2.imread(lq_path))
        if gt is None or lq is None:
            print(f"Failed to read: {name}")
            continue

        edge_gt = canny_edge(gt)
        wb_clahe = apply_clahe(white_balance(lq))
        edge_wb_clahe = canny_edge(wb_clahe)
        edge_input = canny_edge(lq)

        cv2.imwrite(os.path.join(out_gt, name), edge_gt)
        cv2.imwrite(os.path.join(out_wb_clahe, name), edge_wb_clahe)
        cv2.imwrite(os.path.join(out_input, name), edge_input)
        print(f"Processed: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Canny edge maps for a paired dataset.")
    parser.add_argument("--Val_gt", type=str, required=True, help="GT image folder")
    parser.add_argument("--Val_lq", type=str, required=True, help="Input image folder")
    parser.add_argument("--save_dir", type=str, required=True, help="Output folder")
    args = parser.parse_args()
    process(args)
