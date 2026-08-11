import torch

from src.utils import load_config
from src.train import build_model


def test_neural_model_prefix_causality():
    torch.manual_seed(42)

    cfg = load_config(
        "configs/config_full.yaml"
    )

    model = build_model(cfg).eval()

    x = torch.randn(
        1,
        200,
        cfg["features"]["n_mels"],
    )

    with torch.no_grad():
        full = model(x)

        for cut in (
            1, 5, 10, 25,
            50, 100, 150,
        ):
            prefix = model(
                x[:, :cut]
            )

            torch.testing.assert_close(
                prefix,
                full[:, :cut],
                rtol=1e-5,
                atol=1e-6,
            )
