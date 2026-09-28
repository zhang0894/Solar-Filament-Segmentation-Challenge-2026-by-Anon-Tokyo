import numpy as np

from filament.refinement import crop_square
from scripts.predict import InferenceDataset
from scripts.predict_refiner import tile_origins


def test_crop_padding_preserves_coordinates():
    image = np.arange(30, dtype=np.uint8).reshape(5, 6)
    crop, (x0, y0) = crop_square(image, 0, 0, 4)
    assert (x0, y0) == (-2, -2)
    np.testing.assert_array_equal(crop[2:, 2:], image[:2, :2])
    assert not crop[:2].any()


def test_refinement_training_predictions_exclude_outer_validation():
    train = InferenceDataset("train", 0, 1024)
    valid = InferenceDataset("valid", 0, 1024)
    assert len(train) == 565 and len(valid) == 142
    assert set(train.paths).isdisjoint(valid.paths)


def test_tiles_cover_long_instances_and_image_edges_without_gaps():
    for start, end in [(0, 31), (0, 1024), (1500, 2048), (120, 1750)]:
        covered = np.zeros(end - start, bool)
        for left in tile_origins(start, end, 384, 256):
            covered[max(left, start) - start : min(left + 384, end) - start] = True
        assert covered.all()
