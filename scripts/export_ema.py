"""Export the final-epoch inference weights from a completed training run."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.train import save_atomic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with Path(a.checkpoint).open("rb") as f:
        source = torch.load(f, map_location="cpu", weights_only=False)
        f.seek(0)
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if source["epoch"] != source["config"]["epochs"]:
        raise ValueError("Final-epoch export requires completed training")
    state = {
        "model": source.get("ema") or source["model"],
        "config": source["config"],
        "epoch": source["epoch"],
        "selection": "final_epoch_no_validation_selection",
        "source_checkpoint_sha256": digest,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(out, state)
    print(
        json.dumps(
            {"epoch": state["epoch"], "output": str(out), "source_sha256": digest}
        )
    )


if __name__ == "__main__":
    main()
