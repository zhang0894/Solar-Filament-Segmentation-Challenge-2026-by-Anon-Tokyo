import pytest

from filament.data import FilamentDataset, read_manifest


def test_all_annotators_of_an_observation_stay_in_one_fold():
    manifest = read_manifest()
    rows = manifest["images"]
    image_ids = [r["image_id"] for r in rows]
    assert len(image_ids) == len(set(image_ids))
    for fold in range(5):
        train = {r["group"] for r in rows if r["fold"] != fold}
        val = {r["group"] for r in rows if r["fold"] == fold}
        assert train.isdisjoint(val)


def test_all_data_training_has_no_held_out_validation_observations():
    train = FilamentDataset(fold=-1, train=True, kind="image")
    valid = FilamentDataset(fold=-1, train=False, kind="image")
    assert len(train) == len(read_manifest()["images"])
    assert len(valid) == 0


@pytest.mark.parametrize("kind", ["semantic", "instance"])
def test_training_targets_have_correct_shapes_and_mask_only_labels(kind):
    dataset = FilamentDataset(fold=0, train=True, size=512, kind=kind, limit=2)
    item = dataset[0]
    assert item["image"].shape == (3, 512, 512)
    if kind == "semantic":
        assert item["target"].shape == (2, 512, 512)
        assert item["target"].max() <= 1
    else:
        assert item["masks"].shape[-2:] == (512, 512)
        assert item["labels"].sum() == 0
        assert len(item["labels"]) == len(item["masks"])
