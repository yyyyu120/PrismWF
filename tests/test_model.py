import torch

from prismwf import PrismWF


def test_prismwf_output_shape() -> None:
    model = PrismWF(num_classes=12, num_layers=1).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 6, 256))
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()
