"""Convert model outputs to non-duplicated native-resolution instance masks."""

import math

import cv2
import numpy as np
import torch
from pycocotools import mask as mu
from torch.nn import functional as F

from filament.metrics import encode_crop_mask, encode_mask


def rcnn_candidates_cpu(output, output_size=2048, min_area=64):
    """Release dense ROI masks per batch instead of retaining a validation set."""
    tensor = output["masks"][:, 0].detach()
    scores = output["scores"].detach().float().cpu().numpy()
    if not len(scores):
        return []
    if "boxes" in output:
        height, width = tensor.shape[-2:]
        boxes = output["boxes"].detach().float().cpu().numpy()
        # Torchvision pastes each mask only inside its padded ROI. Transfer
        # those small patches together instead of N nearly empty full images.
        # Align crop bounds to the rational resize grid so local OpenCV resize
        # has exactly the same pixel centers as full-image resize (e.g. 4/3).
        qx = width // math.gcd(width, output_size)
        qy = height // math.gcd(height, output_size)
        bounds, patches = [], []
        for mask, (bx0, by0, bx1, by1) in zip(tensor, boxes):
            margin = max(bx1 - bx0, by1 - by0) * 0.1 + 4
            x0 = max(0, math.floor((bx0 - margin) / qx) * qx)
            y0 = max(0, math.floor((by0 - margin) / qy) * qy)
            x1 = min(width, math.ceil((bx1 + margin) / qx) * qx)
            y1 = min(height, math.ceil((by1 + margin) / qy) * qy)
            bounds.append((x0, y0, x1, y1))
            patches.append(mask[y0:y1, x0:x1].reshape(-1))
        packed = torch.cat(patches).float().cpu().numpy()
        regions, offset = [], 0
        for x0, y0, x1, y1 in bounds:
            count = (x1 - x0) * (y1 - y0)
            patch = packed[offset : offset + count].reshape(y1 - y0, x1 - x0)
            offset += count
            ox0, oy0 = x0 * output_size // width, y0 * output_size // height
            ox1, oy1 = x1 * output_size // width, y1 * output_size // height
            regions.append((patch, (ox1 - ox0, oy1 - oy0), ox0, oy0))
    else:
        masks = tensor.float().cpu().numpy()
        regions = [(mask, (output_size, output_size), 0, 0) for mask in masks]
    candidates = []
    for (probability, size, ox, oy), score in zip(regions, scores):
        probability = cv2.resize(probability, size, interpolation=cv2.INTER_LINEAR)
        binary = probability > 0.5
        if binary.sum() < min_area:
            continue
        x, y, w, h = cv2.boundingRect(binary.astype(np.uint8))
        candidates.append(
            {
                "rle": encode_crop_mask(
                    binary[y : y + h, x : x + w],
                    x + ox,
                    y + oy,
                    output_size,
                    output_size,
                ),
                "score": float(score),
            }
        )
    return candidates


def query_candidates_cpu(
    raw, output_size=2048, score_threshold=0.05, mask_threshold=0.5, min_area=64
):
    """CPU equivalent of query decoding, allowing GPU training to continue."""
    from scipy.special import expit

    classes = raw["class_logits"].astype(np.float32)
    exponential = np.exp(classes - classes.max(axis=-1, keepdims=True))
    class_score = (exponential / exponential.sum(axis=-1, keepdims=True))[:, 0]
    boundary = np.log(mask_threshold / (1 - mask_threshold))
    candidates = []
    for idx in np.flatnonzero(class_score > score_threshold):
        logits = cv2.resize(
            raw["mask_logits"][idx].astype(np.float32),
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        )
        binary = logits > boundary
        area = int(binary.sum())
        if area < min_area:
            continue
        # Quality only needs foreground pixels; evaluating sigmoid on the empty
        # solar background wastes most of the full-resolution CPU work.
        score = float(class_score[idx] * expit(logits[binary]).mean())
        if score > score_threshold:
            x, y, w, h = cv2.boundingRect(binary.astype(np.uint8))
            candidates.append(
                {
                    "rle": encode_crop_mask(
                        binary[y : y + h, x : x + w], x, y, output_size, output_size
                    ),
                    "score": score,
                }
            )
    return candidates


def semantic_instances(
    probability,
    threshold=0.5,
    min_area=64,
    min_score=0.0,
    output_size=2048,
    group_radius=0,
    connect_threshold=None,
    seed_threshold=None,
    seed_min_area=64,
):
    probability = cv2.resize(
        probability, (output_size, output_size), interpolation=cv2.INTER_LINEAR
    )
    binary = np.ascontiguousarray(probability > threshold, dtype=np.uint8)
    if connect_threshold is not None and not 0 <= connect_threshold <= threshold:
        raise ValueError("Connection threshold must be below the mask threshold")
    if seed_threshold is not None and not threshold <= seed_threshold <= 1:
        raise ValueError("Seed threshold must be above the mask threshold")
    grouping = (
        binary
        if connect_threshold is None
        else np.ascontiguousarray(probability > connect_threshold, dtype=np.uint8)
    )
    if group_radius:
        if group_radius < 0:
            raise ValueError("group_radius must be nonnegative")
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * group_radius + 1,) * 2
        )
        grouping = cv2.dilate(grouping, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouping, connectivity=8)
    instances = []
    for i in range(1, count):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        # Dilation associates nearby fragments but never expands the final mask.
        x, y, w, h = stats[i, :4]
        region = (labels[y : y + h, x : x + w] == i) & (
            binary[y : y + h, x : x + w] != 0
        )
        if (group_radius or connect_threshold is not None) and int(
            region.sum()
        ) < min_area:
            continue
        if seed_threshold is not None:
            seeds = region & (probability[y : y + h, x : x + w] > seed_threshold)
            if int(seeds.sum()) < seed_min_area:
                continue
        score = float(probability[y : y + h, x : x + w][region].mean())
        if score >= min_score:
            instances.append(
                {
                    "rle": encode_crop_mask(
                        region, int(x), int(y), output_size, output_size
                    ),
                    "score": score,
                }
            )
    return instances


@torch.no_grad()
def query_candidates(
    outputs,
    index=0,
    output_size=2048,
    score_threshold=0.05,
    mask_threshold=0.5,
    min_area=64,
):
    class_score = outputs.class_queries_logits[index].float().softmax(-1)[:, 0]
    candidates = []
    # Process candidates in small chunks to avoid a Q x 2048 x 2048 allocation.
    selected = torch.where(class_score > score_threshold)[0]
    for idx in selected.split(8):
        logits = outputs.masks_queries_logits[index, idx].float()
        probability = F.interpolate(
            logits[:, None],
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0].sigmoid()
        binary = probability > mask_threshold
        area = binary.sum(dim=(1, 2))
        quality = (probability * binary).sum(dim=(1, 2)) / area.clamp_min(1)
        scores = class_score[idx] * quality
        for k in range(len(idx)):
            if area[k] >= min_area and scores[k] > score_threshold:
                candidates.append(
                    {
                        "rle": encode_mask(binary[k].cpu().numpy()),
                        "score": float(scores[k]),
                    }
                )
    return candidates


def filter_instances(candidates, score_threshold=0.4, nms_threshold=0.5, min_area=64):
    order = sorted(
        (
            p
            for p in candidates
            if p["score"] >= score_threshold and int(mu.area(p["rle"])) >= min_area
        ),
        key=lambda x: -x["score"],
    )
    keep = []
    for candidate in order:
        if keep:
            overlaps = mu.iou(
                [candidate["rle"]], [p["rle"] for p in keep], [0] * len(keep)
            )[0]
            if np.any(overlaps > nms_threshold):
                continue
        keep.append(candidate)
    return keep


def disjoint_instances(candidates, min_area=64):
    """Assign contested pixels once, in decreasing model-confidence order."""
    if not candidates:
        return []
    height, width = candidates[0]["rle"]["size"]
    occupied = np.zeros((height, width), bool)
    result = []
    for item in sorted(candidates, key=lambda p: -p["score"]):
        x, y, w, h = mu.toBbox(item["rle"]).astype(int)
        region = mu.decode(item["rle"])[y : y + h, x : x + w].astype(bool)
        region &= ~occupied[y : y + h, x : x + w]
        if int(region.sum()) < min_area:
            continue
        occupied[y : y + h, x : x + w] |= region
        result.append(
            item | {"rle": encode_crop_mask(region, int(x), int(y), height, width)}
        )
    return result
