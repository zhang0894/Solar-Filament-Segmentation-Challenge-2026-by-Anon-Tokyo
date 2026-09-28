import torch

from filament.refinement import DetailResidualRefiner


class ToyRefiner(torch.nn.Module):
    def forward(self, image):
        return image[:, :1] * 0.3, image.mean((1, 2, 3))[:, None]


def test_detail_refiner_starts_as_identity_but_can_learn_pixel_corrections():
    model = DetailResidualRefiner(ToyRefiner())
    image = torch.randn(2, 4, 32, 32)
    original, original_quality = model.base(image)
    prediction, quality = model(image)
    torch.testing.assert_close(prediction, original, rtol=0, atol=0)
    torch.testing.assert_close(quality, original_quality, rtol=0, atol=0)
    prediction.square().mean().backward()
    assert model.detail[-1].weight.grad.abs().sum() > 0
