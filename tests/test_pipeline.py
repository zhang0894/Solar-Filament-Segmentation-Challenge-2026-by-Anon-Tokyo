import hashlib
import json

import pytest

from scripts.predict_pipeline import normalized_weights, validate_refiner_cache


def test_cached_refiner_requires_identical_proposals_and_excluded_fold(tmp_path):
    proposals = tmp_path / "proposals"
    cache = tmp_path / "cache"
    proposals.mkdir()
    cache.mkdir()
    data = b'{"example": []}'
    (proposals / "predictions.json").write_bytes(data)
    config = {"fold": 0, "manifest_sha256": "frozen"}
    (proposals / "metadata.json").write_text(json.dumps({"model_config": config}))
    meta = {
        "inference": {"split": "valid"},
        "proposal_sha256": hashlib.sha256(data).hexdigest(),
        "refiner_config": config.copy(),
    }
    path = cache / "metadata.json"
    path.write_text(json.dumps(meta))
    validate_refiner_cache(cache, proposals, "valid")
    meta["refiner_config"]["fold"] = 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="validation fold"):
        validate_refiner_cache(cache, proposals, "valid")
    meta["inference"]["split"] = "test"
    path.write_text(json.dumps(meta))
    validate_refiner_cache(cache, proposals, "test")
    (proposals / "predictions.json").write_bytes(b'{"different": []}')
    with pytest.raises(ValueError, match="different proposals"):
        validate_refiner_cache(cache, proposals, "test")


def test_nonfinite_or_empty_ensemble_weights_rejected():
    for items in [[], [{"weight": float("nan")}], [{"weight": -1}], [{"weight": 0}]]:
        with pytest.raises(ValueError):
            normalized_weights(items)
