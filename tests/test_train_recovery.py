"""Exercise checkpoint/recovery ordering with a tiny CPU model and fake device I/O."""

import json
import math
import sys
from contextlib import nullcontext

import pytest
import torch

from scripts import train


class TinyDataset:
    def __init__(self, fold, training, size, kind, **kwargs):
        self.count = 2 if training else (0 if fold < 0 else 1)
        self.rows = [{}] * self.count
        self.fast_metadata = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {"image": torch.ones(3, 8, 8), "target": torch.ones(2, 8, 8)}


class TinyLoader:
    def __init__(self, dataset, batch_size, **kwargs):
        self.dataset, self.batch_size = dataset, batch_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        for start in range(0, len(self.dataset), self.batch_size):
            items = [
                self.dataset[i]
                for i in range(start, min(start + self.batch_size, len(self.dataset)))
            ]
            yield {key: torch.stack([item[key] for item in items]) for key in items[0]}


@pytest.fixture
def cpu_training(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text("{}")
    monkeypatch.setattr(train, "ROOT", tmp_path)
    monkeypatch.setattr(train, "FilamentDataset", TinyDataset)
    monkeypatch.setattr(train, "DataLoader", TinyLoader)
    monkeypatch.setattr(train, "CUDAPrefetcher", lambda loader, **kwargs: loader)
    monkeypatch.setattr(
        train, "make_model", lambda *args, **kwargs: torch.nn.Conv2d(3, 2, 1)
    )
    monkeypatch.setattr(train, "remaining_session_seconds", lambda: 100000)
    monkeypatch.setattr(train.signal, "signal", lambda *args: None)
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    def invoke(name, *extra):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--name",
                name,
                "--epochs",
                "1",
                "--batch",
                "2",
                "--workers",
                "0",
                "--val-workers",
                "0",
                *extra,
            ],
        )
        train.main()

    return tmp_path, invoke


def test_validation_failure_preserves_epoch_and_can_replay_on_resume(
    cpu_training, monkeypatch
):
    root, invoke = cpu_training

    def fail(*args, **kwargs):
        raise RuntimeError("simulated device failure during validation")

    monkeypatch.setattr(train, "collect_predictions", fail)
    with pytest.raises(RuntimeError, match="simulated device"):
        invoke("recover")
    run = root / "runs/recover"
    checkpoint = torch.load(run / "last.pt", weights_only=False)
    assert checkpoint["epoch"] == 1 and checkpoint["validation_due"] is True
    assert not (run / "best.pt").exists()
    # A real failed process has exited; this in-process test removes its record.
    (run / "process.json").unlink()
    monkeypatch.setattr(train, "collect_predictions", lambda *args, **kwargs: {})

    def score(*args, save_dir, **kwargs):
        best = {"threshold": 0.5, "official": {"pq": 0.42}}
        save_dir.mkdir()
        for name in ["validation.json", "predictions.json", "entries.json"]:
            (save_dir / name).write_text(json.dumps(best))
        return best, [best]

    monkeypatch.setattr(train, "score_predictions", score)
    invoke("recover", "--resume")
    best = torch.load(run / "best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert best["validation"]["official"]["pq"] == 0.42
    for key, value in checkpoint["ema"].items():
        torch.testing.assert_close(best["model"][key], value)


def test_full_training_exports_ema_without_inventing_validation(
    cpu_training, monkeypatch
):
    root, invoke = cpu_training

    def forbid(*args, **kwargs):
        raise AssertionError("All-data training must not run validation")

    monkeypatch.setattr(train, "collect_predictions", forbid)
    invoke("full", "--all-data")
    run = root / "runs/full"
    best = torch.load(run / "best.pt", weights_only=False)
    last = torch.load(run / "last.pt", weights_only=False)
    assert best["config"]["fold"] == -1
    assert "validation" not in best
    assert last["validation_due"] is False
    for key, value in last["ema"].items():
        torch.testing.assert_close(best["model"][key], value)
