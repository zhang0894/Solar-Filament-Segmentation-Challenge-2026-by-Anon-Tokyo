import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from filament.data import FilamentDataset
from filament.performance import (
    EMAUpdater,
    accumulation_fraction,
    check_finite,
    cpu_state_dict,
    validate_initialization_config,
    validate_resume_config,
)


def test_fast_cache_is_identical_under_same_augmentation():
    fast = FilamentDataset(fold=0, train=True, size=1024, limit=2, fast_cache=True)
    slow = FilamentDataset(fold=0, train=True, size=1024, limit=2, fast_cache=False)
    assert fast.fast_metadata is not None
    random.seed(41)
    np.random.seed(41)
    a = fast[0]
    random.seed(41)
    np.random.seed(41)
    b = slow[0]
    torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
    torch.testing.assert_close(a["target"], b["target"], rtol=0, atol=0)


def test_foreach_ema_and_cpu_snapshot():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    ema, reference = deepcopy(model), deepcopy(model)
    updater = EMAUpdater(ema, model)
    snapshot = cpu_state_dict(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.25)
        model[1].num_batches_tracked.add_(2)
        for name, value in reference.state_dict().items():
            if value.is_floating_point():
                value.lerp_(model.state_dict()[name], 0.1)
            else:
                value.copy_(model.state_dict()[name])
    updater.update(0.9)
    for key, value in ema.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    assert not torch.equal(snapshot["0.weight"], model.state_dict()["0.weight"])


def test_resume_can_rebatch_but_cannot_change_effective_batch():
    previous = {
        "kind": "semantic",
        "fold": 0,
        "size": 1024,
        "epochs": 30,
        "lr": 0.0003,
        "seed": 1,
        "manifest_sha256": "same",
        "ema": 0.99,
        "limit": 0,
        "batch": 2,
        "accum": 2,
    }
    current = {**previous, "batch": 4, "accum": 1}
    validate_resume_config(previous, current)
    with pytest.raises(ValueError):
        validate_resume_config(previous, {**current, "batch": 8})
    with pytest.raises(FloatingPointError):
        check_finite(torch.tensor(float("nan")), "bad")


def test_short_microbatch_matches_full_batch_mean_gradient():
    # The final accumulation group has two images followed by just one.
    x = torch.tensor([1.0, 3.0, 7.0])
    target = torch.tensor([2.0, 5.0, 8.0])
    full = torch.tensor(0.7, requires_grad=True)
    ((full * x - target) ** 2).mean().backward()
    accumulated = torch.tensor(0.7, requires_grad=True)
    for step, start in enumerate(range(0, 3, 2)):
        loss = (
            (accumulated * x[start : start + 2] - target[start : start + 2]) ** 2
        ).mean()
        (loss * accumulation_fraction(step, 2, 2, 3)).backward()
    torch.testing.assert_close(accumulated.grad, full.grad)


def test_warm_start_can_expand_to_all_data_but_cannot_leak_into_validation():
    fold0 = {"fold": 0, "manifest_sha256": "frozen"}
    full = {"fold": -1, "all_data": True, "manifest_sha256": "frozen"}
    validate_initialization_config(fold0, full)
    with pytest.raises(ValueError, match="validation fold"):
        validate_initialization_config(full, fold0)
    with pytest.raises(ValueError, match="validation fold"):
        validate_initialization_config(fold0, {**fold0, "fold": 1})
    with pytest.raises(ValueError, match="manifest"):
        validate_initialization_config(fold0, {**full, "manifest_sha256": "changed"})
