import numpy as np
import pytest

from filament.metrics import (
    Evaluator,
    PQCounts,
    decode_mask,
    encode_crop_mask,
    encode_mask,
    pairwise_iou,
)


def test_crop_rle_matches_dense_at_edges_and_across_columns():
    rng = np.random.default_rng(92)
    for height, width, h, w, x, y in [
        (23, 31, 5, 7, 4, 6),
        (23, 31, 23, 31, 0, 0),
        (23, 31, 3, 5, 26, 20),
        (1, 1, 1, 1, 0, 0),
    ]:
        for crop in [
            rng.random((h, w)) > 0.7,
            np.zeros((h, w), bool),
            np.ones((h, w), bool),
        ]:
            dense = np.zeros((height, width), bool)
            dense[y : y + h, x : x + w] = crop
            actual = encode_crop_mask(crop, x, y, height, width)
            np.testing.assert_array_equal(decode_mask(actual), dense)
            assert actual == encode_mask(dense)


def test_asymmetric_rle_roundtrip_and_overlap():
    mask = np.zeros((13, 17), dtype=bool)
    mask[1:9, 3:5] = True
    mask[7:11, 4:14] = True
    rle = encode_mask(mask)
    assert isinstance(rle["counts"], str)
    np.testing.assert_array_equal(mask, decode_mask(rle))
    np.testing.assert_array_equal(pairwise_iou([rle], [rle]), [[1.0]])


def test_threshold_is_strict_and_global_counts():
    counts = PQCounts()
    counts.update(np.array([[0.5]]))
    assert counts.result()["pq"] == 0
    counts.update(np.array([[0.75]]))
    assert counts.result()["pq"] == pytest.approx(0.375)
    assert counts.result()["fp"] == counts.result()["fn"] == 1


def test_empty_predictions_and_empty_ground_truth():
    counts = PQCounts()
    counts.update(np.zeros((3, 0)))
    counts.update(np.zeros((0, 2)))
    counts.update(np.zeros((0, 0)))
    assert counts.fn == 3 and counts.fp == 2
    assert counts.result()["pq"] == 0


def test_duplicate_predictions_are_exposed_by_strict_metric():
    official, strict = PQCounts(), PQCounts()
    iou = np.array([[1.0, 1.0]])
    official.update(iou)
    strict.update(iou, strict=True)
    assert official.result()["pq"] == 1
    assert strict.result()["pq"] == pytest.approx(2 / 3)


def test_one_prediction_is_scored_against_each_annotator():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:7, 2:6] = True
    rle = encode_mask(mask)
    ev = Evaluator()
    ev.update("obs", "annotator_a", [rle], [rle])
    ev.update("obs", "annotator_b", [rle, rle], [rle])
    assert ev.result()["official"]["tp"] == 3
    assert ev.result()["strict"]["fn"] == 1
    assert ev.result()["n_entries"] == 2
