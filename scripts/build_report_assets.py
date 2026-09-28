"""Build publication figures and uncertainty summaries from frozen OOF evidence."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from pycocotools import mask as mu

from filament.data import ROOT, read_manifest
from filament.metrics import pairwise_iou


def pq(x):
    return x[..., 0] / np.maximum(x[..., 1] + 0.5 * (x[..., 2] + x[..., 3]), 1e-12)


def bootstrap(entries, rows, included, draws=5000):
    names = sorted({r["group"] for r in rows.values() if r["fold"] in included})
    lookup = {g: i for i, g in enumerate(names)}
    totals = []
    for kind in ["coarse", "refined"]:
        x = np.zeros((len(names), 4))
        for e in entries[kind]:
            r = rows[e["image_id"]]
            if r["fold"] in included:
                x[lookup[r["group"]]] += [e[k] for k in ["iou_sum", "tp", "fp", "fn"]]
        totals.append(x)
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(names), (draws, len(names)))
    samples = [pq(x[indices].sum(1)) for x in totals]
    return {
        "groups": len(names),
        "resamples": draws,
        "coarse_pq": float(pq(totals[0].sum(0))),
        "refined_pq": float(pq(totals[1].sum(0))),
        "refined_95pct": np.quantile(samples[1], [0.025, 0.975]).tolist(),
        "delta": float(pq(totals[1].sum(0)) - pq(totals[0].sum(0))),
        "delta_95pct": np.quantile(samples[1] - samples[0], [0.025, 0.975]).tolist(),
        "note": "Paired temporal-group bootstrap, conditional on fitted models; not nested model-selection uncertainty.",
    }


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="deliverables/AnonTokyo/report")
    parser.add_argument("--oof", default="runs/oof_fixed_recipe_v1")
    args = parser.parse_args()
    out = Path(args.output)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.titleweight": "bold",
        }
    )
    teal, gold, navy = "#167C80", "#DB9440", "#22374A"
    oof = ROOT / args.oof
    metrics = json.loads((oof / "metrics.json").read_text())
    entries = {
        k: json.loads((oof / f"{k}_entries.json").read_text())
        for k in ["coarse", "refined"]
    }
    rows = {r["image_id"]: r for r in read_manifest()["images"]}
    summary = {
        "all_folds": bootstrap(entries, rows, set(range(5))),
        "confirmation_folds": bootstrap(entries, rows, {1, 2, 3, 4}),
        "folds": {str(f): bootstrap(entries, rows, {f}) for f in range(5)},
        "manifest_sha256": metrics["manifest_sha256"],
    }
    predictions = {}
    for directory in metrics["pipelines"]:
        predictions.update(
            json.loads((ROOT / directory / "final/predictions.json").read_text())
        )
    if set(predictions) != set(rows):
        raise ValueError("Report requires complete, disjoint OOF observations")
    best_gt, matched, best_dice, examples, relations = [], [], [], [], []
    for name, row in rows.items():
        pred = [q["rle"] for q in predictions[name]]
        for ann in row["annotators"]:
            iou = pairwise_iou(ann["rles"], pred)
            best = iou.max(1) if len(pred) else np.zeros(len(ann["rles"]))
            best_gt.extend(best.tolist())
            best_dice.extend((2 * best / (1 + best)).tolist())
            matched.extend(iou[iou > 0.5].tolist())
            relations.append(
                {
                    "image_id": name,
                    "annotator_id": ann["id"],
                    "split_gt": int(((iou > 0).sum(1) > 1).sum()),
                    "merged_predictions": int(((iou > 0).sum(0) > 1).sum()),
                    "n_gt": len(ann["rles"]),
                    "n_pred": len(pred),
                }
            )
            # Predeclared examples: one small miss, one near miss, one moderate
            # match, one strong match. All are held-out, never test images.
            for j, value in enumerate(best):
                area = float(mu.area(ann["rles"][j]))
                if 256 <= area <= 6000:
                    examples.append((float(value), name, ann["id"], j, ann["rles"][j]))
    summary["distribution"] = {
        "gt_instances": len(best_gt),
        "matched_pairs": len(matched),
        "gt_best_iou_mean": float(np.mean(best_gt)),
        "gt_best_dice_mean": float(np.mean(best_dice)),
        "matched_iou_mean": float(np.mean(matched)),
        "split_gt_nonzero_overlap": sum(r["split_gt"] for r in relations),
        "merged_pred_nonzero_overlap": sum(r["merged_predictions"] for r in relations),
        "predictions_repeated_per_annotation": sum(r["n_pred"] for r in relations),
        "overlap_definition": "Any nonzero pixel overlap; diagnostic, not an IoU>.5 match.",
    }
    (out / "relationship_entries.json").write_text(json.dumps(relations))
    np.savez_compressed(
        out / "distribution_values.npz",
        best_gt_iou=best_gt,
        best_gt_dice=best_dice,
        matched_iou=matched,
    )

    fig, ax = plt.subplots(figsize=(7.05, 1.35))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    labels = [
        ("H-alpha image", "2048 x 2048\nNormalized grayscale"),
        ("Global proposals", "ConvNeXt-Tiny U-Net\n1536 px / 3-view TTA"),
        ("Native refiner", "ResNet18 / 384 px crops\nMask + quality"),
        ("Instance export", "Blend, filter, allocate\n2048 px COCO RLE"),
    ]
    for i, (title, detail) in enumerate(labels):
        x = 0.06 + i * 2.53
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.65),
                2.28,
                1.05,
                boxstyle="round,pad=0.03,rounding_size=.09",
                edgecolor=teal if i in (1, 2) else navy,
                facecolor="#F2F7F8",
                lw=1,
            )
        )
        ax.text(
            x + 1.14, 1.43, title, ha="center", color=navy, weight="bold", fontsize=8
        )
        ax.text(x + 1.14, 1.00, detail, ha="center", va="center", fontsize=7)
        if i < 3:
            ax.annotate(
                "",
                (x + 2.51, 1.18),
                (x + 2.29, 1.18),
                arrowprops={"arrowstyle": "->", "color": navy},
            )
    ax.text(
        5,
        0.25,
        "Five temporal-group folds | all annotators kept together | fixed post-processing",
        ha="center",
        color=teal,
        fontsize=8,
    )
    save(fig, figs / "pipeline")

    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.15), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    ax = axes[0]
    x = np.arange(5)
    coarse = [r["coarse"]["official"]["pq"] for r in metrics["folds"]]
    refined = [r["refined"]["official"]["pq"] for r in metrics["folds"]]
    intervals = np.array([summary["folds"][str(f)]["refined_95pct"] for f in range(5)])
    ax.plot(x - 0.07, coarse, "o--", color=gold, ms=4, label="Coarse, area >=256")
    ax.errorbar(
        x + 0.07,
        refined,
        yerr=np.array([refined - intervals[:, 0], intervals[:, 1] - refined]),
        fmt="o-",
        color=teal,
        capsize=2,
        ms=4,
        label="Refined, area >=64",
    )
    ax.set(
        xticks=x,
        xticklabels=["0*", "1", "2", "3", "4"],
        xlabel="Held-out fold (* development)",
        ylabel="Panoptic quality",
        ylim=(0.35, 0.51),
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.15)
    audit = json.loads((oof / "error_audit.json").read_text())["gt_area_bands"]
    axes[1].bar(
        np.arange(len(audit)),
        [a["matched"] / a["n"] for a in audit],
        color=teal,
        width=0.7,
    )
    axes[1].set(
        xticks=np.arange(len(audit)),
        xticklabels=["<256", "256-\n512", "512-\n1k", "1-2k", "2-4k", "4-8k", ">=8k"],
        ylabel="GT match rate (IoU > 0.5)",
        xlabel="Native instance area (pixels)",
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", labelsize=6.5)
    axes[1].grid(axis="y", alpha=0.15)
    fig.tight_layout(w_pad=2)
    save(fig, figs / "validation")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.55))
    axes[0].hist(best_gt, bins=np.linspace(0, 1, 21), color=teal, rwidth=0.9)
    axes[0].axvline(0.5, color=gold, ls="--", lw=1)
    axes[0].set(xlabel="Best prediction IoU per GT", ylabel="GT instances")
    axes[1].hist(best_dice, bins=np.linspace(0, 1, 21), color=teal, rwidth=0.9)
    axes[1].set(xlabel="Best prediction Dice per GT", ylabel="GT instances")
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 1000]
    sp = np.histogram([r["split_gt"] for r in relations], bins=bins)[0]
    mg = np.histogram([r["merged_predictions"] for r in relations], bins=bins)[0]
    x = np.arange(6)
    axes[2].bar(x - 0.18, sp, 0.36, color=teal, label="Split GT")
    axes[2].bar(x + 0.18, mg, 0.36, color=gold, label="Merged prediction")
    axes[2].set(
        xticks=x,
        xticklabels=["0", "1", "2", "3", "4", "5+"],
        xlabel="Relations per annotator-image",
        ylabel="Annotation entries",
    )
    axes[2].legend(frameon=False, fontsize=6)
    fig.tight_layout(w_pad=1)
    save(fig, figs / "distributions")

    fig, axes = plt.subplots(1, 4, figsize=(7.05, 1.70), squeeze=False)
    selected = []
    used = set()
    for col, target in enumerate([0.0, 0.45, 0.65, 0.85]):
        candidate = min(
            (e for e in examples if e[1] not in used),
            key=lambda e: (abs(e[0] - target), e[1], e[2], e[3]),
        )
        value, name, annotator, j, rle = candidate
        used.add(name)
        image = cv2.imread(
            str(
                ROOT
                / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
                / rows[name]["filename"]
            ),
            0,
        )
        x, y, w, h = mu.toBbox(rle)
        side = int(max(240, max(w, h) + 90))
        cx, cy = x + w / 2, y + h / 2
        x0, y0 = (
            max(0, min(2048 - side, int(cx - side / 2))),
            max(0, min(2048 - side, int(cy - side / 2))),
        )
        crop = image[y0 : y0 + side, x0 : x0 + side]
        gt = mu.decode(rle)[y0 : y0 + side, x0 : x0 + side]
        pred_mask = np.zeros_like(gt)
        for p in predictions[name]:
            pred_mask |= mu.decode(p["rle"])[y0 : y0 + side, x0 : x0 + side]
        for row in range(1):
            ax = axes[row, col]
            ax.imshow(
                crop,
                cmap="gray",
                vmin=max(0, np.percentile(crop, 1) - 5),
                vmax=np.percentile(crop, 99) + 5,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.contour(gt, levels=[0.5], colors=["#27D5BF"], linewidths=0.8)
            if pred_mask.any():
                ax.contour(pred_mask, levels=[0.5], colors=["#F2AF52"], linewidths=0.8)
            if row == 0:
                ax.set_title(
                    ["Miss", "Near match", "Moderate match", "Strong match"][col]
                    + f"\nIoU {value:.2f}",
                    fontsize=8,
                )
        selected.append(
            {
                "image_id": name,
                "annotator_id": annotator,
                "gt_index": j,
                "best_iou": value,
                "fold": rows[name]["fold"],
                "crop_xywh": [x0, y0, side, side],
            }
        )
    fig.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.15)
    save(fig, figs / "qualitative")
    summary["qualitative_selection"] = selected
    summary["qualitative_protocol"] = (
        "Closest GT best-IoU to fixed targets 0/.45/.65/.85, area 256..6000 px, distinct held-out images; examples illustrate strengths and failures, not prevalence."
    )
    (out / "report_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
