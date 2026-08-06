"""Model tests: output shape, parameter budget, and STRICT CAUSALITY."""
import torch
from src.models.crnn import SepConvGRUVAD


def test_forward_shape():
    model = SepConvGRUVAD(n_mels=64).eval()
    x = torch.randn(4, 300, 64)          # [B, T, n_mels]
    y = model(x)
    assert y.shape == (4, 300)           # [B, T] logits


def test_param_budget():
    model = SepConvGRUVAD(n_mels=64)
    n = model.num_parameters()
    print(f"\n[params] {n:,} ({n/1e6:.3f} M)")
    assert n < 500_000, f"model too large: {n}"


def test_strict_causality():
    """Output at frames < t0 must not change when input at t0 is perturbed."""
    torch.manual_seed(0)
    model = SepConvGRUVAD(n_mels=64).eval()   # eval -> BN uses running stats
    x = torch.randn(1, 200, 64)
    t0 = 120
    x2 = x.clone(); x2[0, t0, :] += 10.0      # large perturbation at t0
    with torch.no_grad():
        y1, y2 = model(x), model(x2)
    assert torch.allclose(y1[:, :t0], y2[:, :t0], atol=1e-5), \
        "non-causal: past outputs changed by a future input"
    assert not torch.allclose(y1[:, t0:], y2[:, t0:]), \
        "perturbation had no effect at all (sanity)"
