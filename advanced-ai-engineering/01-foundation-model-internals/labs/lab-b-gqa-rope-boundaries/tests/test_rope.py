import math

import pytest
import torch

import attention as A


def vecs(dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 2, 1, 16, generator=g, dtype=dtype), torch.randn(1, 2, 1, 16, generator=g, dtype=dtype)


def test_position_zero_and_one_hot_pair_boundary():
    cos, sin = A.rope_cos_sin(torch.tensor([0, 1]), 8)
    assert torch.equal(cos[0], torch.ones(8)) and torch.equal(sin[0], torch.zeros(8)), "HINT: position 0 has zero angle and must be identity."
    x = torch.zeros(1, 1, 1, 8); x[..., 0] = 1
    y = A.apply_rope(x, cos[1:2], sin[1:2])[0, 0, 0]
    assert math.isclose(y[0].item(), math.cos(1), abs_tol=1e-6), "HINT: coordinate 0 retains cos(theta_0)."
    assert math.isclose(y[4].item(), math.sin(1), abs_tol=1e-6), (
        "HINT: one-hot coordinate 0 rotates into coordinate d_h/2 under half-split pairing; adjacent pairing preserves shape but breaks parity.")


def test_odd_dimension_rejected():
    with pytest.raises(ValueError):
        A.rope_cos_sin(torch.arange(2), 7)


@pytest.mark.parametrize("offset,shift", [(0, 0), (1, 1), (2, 1000), (17, 1000000)])
def test_relative_score_shift_invariance_float64(offset, shift):
    q, k = vecs()
    s0 = A.relative_score(q, k, torch.tensor([3]), torch.tensor([3 + offset]))
    s1 = A.relative_score(q, k, torch.tensor([3 + shift]), torch.tensor([3 + shift + offset]))
    assert torch.allclose(s0, s1, atol=1e-8), (
        f"HINT: same relative offset {offset} at shift {shift} changed a float64 score. Compute angles from exact position IDs "
        "in float64 and rotate Q and K with the same frequencies/pairing.")


def test_relative_score_changes_with_offset():
    q, k = vecs()
    a = A.relative_score(q, k, torch.tensor([11]), torch.tensor([12]))
    b = A.relative_score(q, k, torch.tensor([11]), torch.tensor([18]))
    assert not torch.allclose(a, b, atol=1e-4), "HINT: RoPE removes absolute shift from the score, not relative displacement."


def test_norm_preserved_at_large_position():
    x = torch.randn(2, 3, 1, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    c, s = A.rope_cos_sin(torch.tensor([10**9]), 16, dtype=torch.float64)
    y = A.apply_rope(x, c, s)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-12), "HINT: each RoPE pair is an orthogonal rotation; large position does not change the norm in float64."


def test_naive_low_precision_angle_boundary_is_measurable():
    q, k = vecs(torch.float32)
    precise = A.relative_score(q, k, torch.tensor([10**8]), torch.tensor([10**8 + 7]))
    # deliberately naive: convert absolute positions to float32 before angle multiplication
    dh = q.shape[-1]
    inv = 10000.0 ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)
    def naive(x, pos):
        ang = torch.tensor([pos], dtype=torch.float32)[:, None] * inv
        emb = torch.cat((ang, ang), -1)
        return A.apply_rope(x, emb.cos(), emb.sin())
    bad = (naive(q, 10**8) * naive(k, 10**8 + 7)).sum(-1)
    err = (bad - precise).abs().max().item()
    assert err > 1e-3, (
        "HINT: this test should expose float32 absolute-position angle loss near 1e8. If it does not, ensure the reference "
        "computes angles in float64 and the naive path casts positions before multiplication.")
