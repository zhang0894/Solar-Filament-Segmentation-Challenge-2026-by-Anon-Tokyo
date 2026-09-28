import torch
from transformers.models.mask2former.modeling_mask2former import sample_point

from filament.sparse_loss import SparseMatcher, positive_coordinates


def test_foreground_points_cover_single_pixel_instances():
    mask = torch.zeros(2, 31, 37)
    mask[0, 3, 7] = 1
    mask[1, 23, 17] = 1
    coordinates = positive_coordinates(mask, 32)
    values = sample_point(mask[:, None], coordinates, align_corners=False)
    torch.testing.assert_close(values, torch.ones_like(values), atol=1e-5, rtol=0)


def test_sparse_matching_finds_tiny_instances_with_swapped_queries():
    masks = torch.zeros(2, 32, 32)
    masks[0, 3:5, 7:9] = 1
    masks[1, 23:25, 17:19] = 1
    queries = (masks.flip(0) * 20 - 10)[None]
    matcher = SparseMatcher(cost_class=2, cost_mask=5, cost_dice=5, num_points=256)
    result = matcher(
        queries,
        torch.tensor([[[5.0, -5.0], [5.0, -5.0]]]),
        [masks],
        [torch.zeros(2, dtype=torch.long)],
    )[0]
    assert result[0].tolist() == [0, 1]
    assert result[1].tolist() == [1, 0]
