import numpy as np

from filament.metrics import decode_mask, encode_mask
from filament.prediction import filter_instances, semantic_instances


def test_separate_objects_remain_separate_and_tiny_noise_removed():
    p = np.zeros((64, 64), np.float32)
    p[5:25, 5:9] = 0.9
    p[30:50, 40:44] = 0.8
    p[60, 60] = 0.99
    predictions = semantic_instances(p, threshold=0.5, min_area=10, output_size=64)
    assert len(predictions) == 2


def test_mask_nms_rejects_duplicates_without_removing_adjacent_instances():
    a = np.zeros((20, 20), np.uint8)
    b = a.copy()
    a[1:18, 3:7] = 1
    b[1:18, 9:13] = 1
    predictions = [
        {"rle": encode_mask(a), "score": 0.9},
        {"rle": encode_mask(a), "score": 0.8},
        {"rle": encode_mask(b), "score": 0.7},
    ]
    selected = filter_instances(predictions, min_area=10)
    assert len(selected) == 2
    assert selected[0]["score"] == 0.9


def test_fragment_grouping_preserves_original_pixels():
    p = np.zeros((64, 64), np.float32)
    p[10:25, 30:34] = 0.9
    p[28:45, 30:34] = 0.9
    separate = semantic_instances(p, min_area=10, output_size=64)
    grouped = semantic_instances(p, min_area=10, output_size=64, group_radius=2)
    assert len(separate) == 2 and len(grouped) == 1
    np.testing.assert_array_equal(decode_mask(grouped[0]["rle"]), p > 0.5)


def test_cpu_query_postprocessing_matches_tensor_implementation():
    from types import SimpleNamespace

    import torch

    from filament.prediction import query_candidates, query_candidates_cpu

    torch.manual_seed(113)
    outputs = SimpleNamespace(
        masks_queries_logits=torch.randn(1, 4, 16, 16),
        class_queries_logits=torch.randn(1, 4, 2),
    )
    expected = query_candidates(outputs, output_size=64, min_area=1)
    actual = query_candidates_cpu(
        {
            "mask_logits": outputs.masks_queries_logits[0].numpy(),
            "class_logits": outputs.class_queries_logits[0].numpy(),
        },
        output_size=64,
        min_area=1,
    )
    assert len(expected) == len(actual)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(decode_mask(a["rle"]), decode_mask(b["rle"]))
        np.testing.assert_allclose(a["score"], b["score"], rtol=1e-5)


def test_disjoint_assignment_preserves_union_without_duplicate_pixels():
    from filament.prediction import disjoint_instances

    a = np.zeros((30, 30), bool)
    b = a.copy()
    a[3:20, 5:15] = 1
    b[6:23, 12:24] = 1
    result = disjoint_instances(
        [{"rle": encode_mask(a), "score": 0.9}, {"rle": encode_mask(b), "score": 0.8}],
        min_area=1,
    )
    masks = [decode_mask(p["rle"]) for p in result]
    assert not (masks[0] & masks[1]).any()
    np.testing.assert_array_equal(masks[0] | masks[1], a | b)
    np.testing.assert_array_equal(masks[0], a)


def test_hysteresis_groups_faint_bridge_without_expanding_final_mask():
    p = np.zeros((32, 32), dtype=np.float32)
    p[14:16, 4:8] = 0.9
    p[14:16, 18:22] = 0.9
    p[14:16, 8:18] = 0.3
    separate = semantic_instances(p, threshold=0.5, min_area=1, output_size=32)
    grouped = semantic_instances(
        p,
        threshold=0.5,
        min_area=1,
        output_size=32,
        connect_threshold=0.2,
        seed_threshold=0.8,
        seed_min_area=4,
    )
    assert len(separate) == 2 and len(grouped) == 1
    assert int(decode_mask(grouped[0]["rle"]).sum()) == 16
