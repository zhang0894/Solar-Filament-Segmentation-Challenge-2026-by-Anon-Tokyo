"""Mask geometry and local image contrast, derived only from allowed inputs."""

import cv2
import numpy as np
from pycocotools import mask as mu

FEATURE_NAMES = [
    "log_area",
    "log_width",
    "log_height",
    "bbox_fill",
    "elongation",
    "compactness",
    "solidity",
    "components",
    "mean",
    "std",
    "q10",
    "q50",
    "q90",
    "ring_mean",
    "ring_std",
    "contrast",
    "normalized_contrast",
    "radial_distance",
    "dark_fraction",
]


def candidate_features(image, rle):
    x, y, w, h = mu.toBbox(rle).astype(int)
    mask = mu.decode(rle)
    x0, y0, x1, y1 = (
        max(0, x - 16),
        max(0, y - 16),
        min(image.shape[1], x + w + 16),
        min(image.shape[0], y + h + 16),
    )
    mask = mask[y0:y1, x0:x1].copy()
    gray = image[y0:y1, x0:x1].astype(np.float32) / 255
    inside = gray[mask != 0]
    if not len(inside):
        raise ValueError("Empty proposal")
    ring = (
        cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) != 0
    ) & (mask == 0)
    outside = gray[ring]
    if not len(outside):
        outside = inside
    moments = cv2.moments(mask)
    area = moments["m00"]
    covariance = np.array(
        [[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]]
    ) / max(area, 1)
    eig = np.maximum(np.linalg.eigvalsh(covariance), 0.25)
    elongation = np.sqrt(eig[1] / eig[0])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    hull_area = (
        cv2.contourArea(cv2.convexHull(np.concatenate(contours))) if contours else area
    )
    cx, cy = moments["m10"] / area + x0, moments["m01"] / area + y0
    mean, std, ring_mean, ring_std = (
        inside.mean(),
        inside.std(),
        outside.mean(),
        outside.std(),
    )
    features = [
        np.log1p(area),
        np.log1p(w),
        np.log1p(h),
        area / max(w * h, 1),
        np.log1p(elongation),
        4 * np.pi * area / max(perimeter**2, 1),
        area / max(hull_area, area),
        len(contours),
        mean,
        std,
        *np.quantile(inside, [0.1, 0.5, 0.9]),
        ring_mean,
        ring_std,
        ring_mean - mean,
        (ring_mean - mean) / max(ring_std, 0.01),
        np.hypot(cx - 1024, cy - 1024) / 1024,
        (inside < 0.1).mean(),
    ]
    return np.asarray(features, np.float32)
