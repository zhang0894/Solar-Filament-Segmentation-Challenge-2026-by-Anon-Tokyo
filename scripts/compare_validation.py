"""Paired group-bootstrap comparison of two frozen validation predictions."""

import argparse
import json
from pathlib import Path

import numpy as np

from filament.data import read_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = {r["image_id"]: r for r in read_manifest()["images"] if r["fold"] == a.fold}
    groups = sorted({r["group"] for r in rows.values()})
    group_index = {g: i for i, g in enumerate(groups)}
    expected = {
        (r["image_id"], ann["id"]) for r in rows.values() for ann in r["annotators"]
    }
    arrays = []
    for file in [a.baseline, a.candidate]:
        entries = json.loads(Path(file).read_text())
        if (
            len(entries) != len(expected)
            or {(e["image_id"], e["annotator_id"]) for e in entries} != expected
        ):
            raise ValueError(
                "Comparison must cover exactly all held-out annotator-image entries"
            )
        totals = np.zeros((len(groups), 4))
        for e in entries:
            totals[group_index[rows[e["image_id"]]["group"]]] += [
                e[k] for k in ["iou_sum", "tp", "fp", "fn"]
            ]
        arrays.append(totals)

    def pq(counts):
        return counts[..., 0] / np.maximum(
            counts[..., 1] + 0.5 * (counts[..., 2] + counts[..., 3]), 1e-10
        )

    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(groups), (5000, len(groups)))
    baseline, candidate = [pq(v.sum(0)) for v in arrays]
    deltas = pq(arrays[1][indices].sum(1)) - pq(arrays[0][indices].sum(1))
    result = {
        "baseline": a.baseline,
        "candidate": a.candidate,
        "fold": a.fold,
        "observation_groups": len(groups),
        "baseline_pq": float(baseline),
        "candidate_pq": float(candidate),
        "delta": float(candidate - baseline),
        "paired_bootstrap_delta_95pct": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "bootstrap_fraction_positive": float((deltas > 0).mean()),
        "note": "Development-fold uncertainty; threshold/model selection on this fold can bias the gain.",
    }
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
