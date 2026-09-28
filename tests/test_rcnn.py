import pytest
import torch
from torchvision.models.detection.roi_heads import paste_masks_in_image

from filament.metrics import decode_mask
from filament.prediction import rcnn_candidates_cpu
from filament.rcnn import boxes_from_masks


def test_roi_boxes_preserve_single_pixel_and_thin_instances():
    masks = torch.zeros(3, 20, 30)
    masks[0, 5, 7] = 1
    masks[1, 4:18, 9] = 1
    masks[2, 1:6, 12:21] = 1
    expected = torch.tensor(
        [[7, 5, 8, 6], [9, 4, 10, 18], [12, 1, 21, 6]], dtype=torch.float32
    )
    torch.testing.assert_close(boxes_from_masks(masks), expected)
    assert boxes_from_masks(masks[:0]).shape == (0, 4)


def test_roi_decoder_keeps_separate_masks_and_native_coordinates():
    masks = torch.zeros(2, 1, 32, 32)
    masks[0, 0, 4:20, 5:8] = 0.9
    masks[1, 0, 10:25, 20:22] = 0.9
    candidates = rcnn_candidates_cpu(
        {"masks": masks, "scores": torch.tensor([0.8, 0.7])}, output_size=32, min_area=1
    )
    assert len(candidates) == 2
    for candidate, original in zip(candidates, masks[:, 0]):
        assert (decode_mask(candidate["rle"]) == (original.numpy() > 0.5)).all()


@pytest.mark.parametrize("source_size,output_size", [(64, 128), (96, 128), (96, 96)])
def test_roi_cropped_transfer_matches_full_image_resize_exactly(
    source_size, output_size
):
    generator = torch.Generator().manual_seed(37)
    boxes = torch.tensor(
        [
            [0, 0, 19.8, 26.2],
            [13.7, 21.3, 40.1, 49.2],
            [source_size - 15.3, source_size - 19.7, source_size, source_size],
            [0, 0, source_size, source_size],
        ],
        dtype=torch.float32,
    )
    probabilities = torch.rand((4, 1, 56, 56), generator=generator)
    masks = paste_masks_in_image(probabilities, boxes, (source_size, source_size))
    output = {"masks": masks, "scores": torch.tensor([0.9, 0.8, 0.7, 0.6])}
    dense = rcnn_candidates_cpu(output, output_size, min_area=1)
    cropped = rcnn_candidates_cpu(output | {"boxes": boxes}, output_size, min_area=1)
    assert cropped == dense
