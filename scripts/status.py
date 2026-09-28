"""Compact progress snapshot without reading credentials or large tensors."""

import json
import shutil
import time

from filament.data import ROOT


def main():
    state = json.loads((ROOT / "RUN_STATE.json").read_text())
    histories = sorted(
        (ROOT / "runs").glob("*/history.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:5]
    models = []
    for file in histories:
        rows = [json.loads(line) for line in file.read_text().splitlines() if line]
        epochs = [
            r
            for r in rows
            if r.get("event") == "epoch" or ("epoch" in r and "loss" in r)
        ]
        validations = [r for r in rows if r.get("event") == "validation"]
        last = epochs[-1] if epochs else {}
        models.append(
            {
                "name": file.parent.name,
                "epoch": last.get("epoch"),
                "epoch_seconds": last.get("train_seconds", last.get("seconds")),
                "best_raw_pq": max(
                    (r["validation"]["official"]["pq"] for r in validations),
                    default=None,
                ),
            }
        )
    print(
        json.dumps(
            {
                "active": state.get("active_experiment"),
                "remaining_minutes": round(
                    (state["session_deadline_unix"] - time.time()) / 60, 1
                ),
                "disk_free_GiB": round(shutil.disk_usage(ROOT).free / 2**30, 1),
                "recent_models": models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
