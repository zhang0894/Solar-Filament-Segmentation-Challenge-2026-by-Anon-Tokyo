"""Official v6 PQ semantics, plus strict matching diagnostics.

Source: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
Each annotator-image contributes separately to globally accumulated counts.
The official notebook counts all pairs with IoU > .5; it does not run an
assignment algorithm. The strict score exposes duplicate prediction errors.
"""

from dataclasses import asdict, dataclass

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def encode_mask(mask: np.ndarray) -> dict:
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"].decode("ascii")}


def decode_mask(rle: dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


def encode_crop_mask(mask, x0, y0, height=2048, width=2048):
    """Embed a small binary ROI in a full COCO RLE without a dense full image."""
    h, w = mask.shape
    if min(x0, y0) < 0 or x0 + w > width or y0 + h > height:
        raise ValueError("Mask crop is outside the full image")
    positions = np.flatnonzero(np.asarray(mask, dtype=bool).T.reshape(-1))
    if not len(positions):
        counts = [height * width]
    else:
        positions = (positions // h + x0) * height + (positions % h + y0)
        gaps = np.diff(positions) > 1
        starts = positions[np.r_[True, gaps]]
        ends = positions[np.r_[gaps, True]] + 1
        counts = np.empty(2 * len(starts) + 1, dtype=np.int64)
        counts[0] = starts[0]
        counts[1:-1:2] = ends - starts
        counts[2:-1:2] = starts[1:] - ends[:-1]
        counts[-1] = height * width - ends[-1]
        counts = counts.tolist()
        if counts[-1] == 0:
            counts.pop()
    rle = mask_utils.frPyObjects(
        {"size": [height, width], "counts": counts}, height, width
    )
    return {"size": [height, width], "counts": rle["counts"].decode("ascii")}


def pairwise_iou(gt: list[dict], pred: list[dict]) -> np.ndarray:
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    # COCO returns detection x ground-truth; official notebook uses the transpose.
    return np.asarray(mask_utils.iou(pred, gt, [0] * len(gt))).T


@dataclass
class PQCounts:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, ious: np.ndarray, strict: bool = False):
        n_gt, n_pred = ious.shape
        hit = ious > 0.5
        if strict and n_gt and n_pred:
            # Prioritize cardinality of valid matches, then summed IoU.
            rows, cols = linear_sum_assignment(
                -(hit.astype(float) * (min(n_gt, n_pred) + 1) + ious * hit)
            )
            matched = hit[rows, cols]
            rows, cols = rows[matched], cols[matched]
            tp = len(rows)
            self.iou_sum += float(ious[rows, cols].sum())
            self.tp += tp
            self.fp += n_pred - tp
            self.fn += n_gt - tp
        else:
            self.iou_sum += float(ious[hit].sum())
            self.tp += int(hit.sum())
            self.fp += int((hit.sum(axis=0) == 0).sum())
            self.fn += int((hit.sum(axis=1) == 0).sum())

    def result(self) -> dict:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return {
            **asdict(self),
            "pq": self.iou_sum / denominator if denominator else 0.0,
            "sq": self.iou_sum / self.tp if self.tp else 0.0,
            "rq": self.tp / denominator if denominator else 0.0,
        }


class Evaluator:
    def __init__(self):
        self.official = PQCounts()
        self.strict = PQCounts()
        self.entries = []

    def update(
        self, image_id: str, annotator_id: str, gt: list[dict], pred: list[dict]
    ):
        ious = pairwise_iou(gt, pred)
        self.official.update(ious)
        self.strict.update(ious, strict=True)
        entry = PQCounts()
        entry.update(ious)
        overlap = ious > 0
        self.entries.append(
            {
                "image_id": image_id,
                "annotator_id": annotator_id,
                **entry.result(),
                "n_gt": len(gt),
                "n_pred": len(pred),
                "split_gt": int((overlap.sum(axis=1) > 1).sum()),
                "merged_predictions": int((overlap.sum(axis=0) > 1).sum()),
            }
        )

    def result(self) -> dict:
        return {
            "official": self.official.result(),
            "strict": self.strict.result(),
            "n_entries": len(self.entries),
        }
