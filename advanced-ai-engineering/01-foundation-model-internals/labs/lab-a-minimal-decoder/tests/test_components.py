import math

import pytest
import torch
import torch.nn.functional as F

import decoder as D


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------- RMSNorm (Lesson 1.5) ----------

def test_rmsnorm_matches_builtin():
    x = torch.randn(2, 5, 16, generator=g())
    w = torch.rand(16, generator=g(1)) + 0.5
    for eps in (1e-6, 0.5):  # large eps exposes epsilon placement
        got = D.rms_norm(x, w, eps)
        ref = F.rms_norm(x, (16,), weight=w, eps=eps)
        assert got.shape == x.shape, "HINT: RMSNorm must keep [..., d]; reduce with keepdim and broadcast back."
        assert torch.allclose(got, ref, atol=1e-6), (
            f"HINT: mismatch at eps={eps}. Compare the mean square at one position against a hand calculation; "
            "is epsilon inside the square root? Is the mean NOT subtracted?")


def test_rmsnorm_reduces_feature_axis_only():
    x = torch.randn(1, 6, 8, generator=g())
    w = torch.ones(8)
    y0 = D.rms_norm(x, w, 1e-6)
    x2 = x.clone()
    x2[0, 5] *= 7.0  # change only the last position
    y1 = D.rms_norm(x2, w, 1e-6)
    assert torch.allclose(y0[0, :5], y1[0, :5]), (
        "HINT: changing one position moved another position's output. You reduced over S, not d; "
        "that also breaks causal isolation.")
    rms = y0.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), "HINT: with weight=1 every output row should have RMS 1."


def test_rmsnorm_fp16_large_values_finite():
    x = torch.full((1, 1, 8), 300.0, dtype=torch.float16)
    y = D.rms_norm(x, torch.ones(8, dtype=torch.float16), 1e-6)
    assert y.dtype == torch.float16, "HINT: cast the normalized value back to the input dtype before scaling."
    assert torch.isfinite(y).all(), (
        "HINT: 300^2 exceeds the FP16 max (65504). Square and average in float32, then cast back.")


# ---------- RoPE (Lesson 1.4) ----------

def test_rope_one_hot_pairing():
    dh, pos = 8, 3
    cos, sin = D.rope_cos_sin(torch.tensor([pos]), dh, 10000.0)
    assert cos.shape == (1, dh) and sin.shape == (1, dh), "HINT: cos/sin are [S, d_h] (angles repeated for both halves)."
    e0 = torch.zeros(1, 1, 1, dh)
    e0[..., 0] = 1.0
    out = D.apply_rope(e0, cos, sin)[0, 0, 0]
    assert math.isclose(out[0].item(), math.cos(pos), abs_tol=1e-6), "HINT: coordinate 0 should keep cos(p*theta_0)."
    assert math.isclose(out[dh // 2].item(), math.sin(pos), abs_tol=1e-6), (
        "HINT: rotate a one-hot vector and check which coordinate receives the sine term. "
        "The reference pairs (i, i + d_h/2) (half-split), not (2i, 2i+1).")
    assert out.abs().sum() - abs(out[0]) - abs(out[dh // 2]) < 1e-6, "HINT: a rotation of pair 0 must not leak into other pairs."


def test_rope_preserves_norm_and_position_zero_is_identity():
    x = torch.randn(2, 3, 7, 16, generator=g(), dtype=torch.float64)
    cos, sin = D.rope_cos_sin(torch.arange(7), 16, 10000.0)
    y = D.apply_rope(x, cos.double(), sin.double())
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1)), "HINT: each pair is rotated, so vector norms must be preserved."
    assert torch.allclose(y[:, :, 0], x[:, :, 0]), "HINT: angle at position 0 is 0, so RoPE must be the identity there."


def test_rope_score_depends_only_on_offset():
    dh = 16
    q = torch.randn(1, 1, 1, dh, generator=g(), dtype=torch.float64)
    k = torch.randn(1, 1, 1, dh, generator=g(1), dtype=torch.float64)

    def score(p, r):
        cp, sp = D.rope_cos_sin(torch.tensor([p]), dh, 10000.0)
        cr, sr = D.rope_cos_sin(torch.tensor([r]), dh, 10000.0)
        return (D.apply_rope(q, cp.double(), sp.double()) * D.apply_rope(k, cr.double(), sr.double())).sum().item()

    assert math.isclose(score(3, 5), score(13, 15), abs_tol=1e-5), (
        "HINT: (R(p)q)·(R(r)k) must depend only on r - p. Rotate q AND k with the same convention and frequencies.")
    assert not math.isclose(score(3, 5), score(3, 9), abs_tol=1e-3), "HINT: different offsets should give different scores."


# ---------- Mask and grouping (Lesson 1.3) ----------

def test_causal_mask_semantics():
    m = D.causal_mask(4, 4)
    assert m.shape == (4, 4), "HINT: mask is [S_q, S_k]."
    assert (torch.diagonal(m) == 0).all(), "HINT: the diagonal is allowed (j <= i): a token attends to itself."
    assert torch.isinf(m[0, 1]) and m[1, 0] == 0, "HINT: future keys (j > i) get -inf; check the mask orientation."
    dec = D.causal_mask(1, 6)
    assert (dec == 0).all(), (
        "HINT: one new query after 5 cached tokens sits at absolute position 5 and may see all 6 keys. "
        "Offset query positions by S_k - S_q.")
    m2 = D.causal_mask(2, 5)
    assert m2[0, 3] == 0 and torch.isinf(m2[0, 4]), "HINT: with a cached prefix, query i sits at absolute position S_k - S_q + i."


def test_expand_kv_contiguous_groups():
    kv = torch.randn(2, 3, 5, 4, generator=g())
    out = D.expand_kv(kv, 4)
    assert out.shape == (2, 12, 5, 4), "HINT: [B,Hkv,T,dh] -> [B,Hkv*n_rep,T,dh]."
    ref = torch.repeat_interleave(kv, 4, dim=1)
    for h in range(12):
        assert torch.equal(out[:, h], ref[:, h]), (
            f"HINT: query head {h} reads the wrong KV head. Query head h uses group h // g (contiguous blocks), "
            "not h % H_kv (interleaved). Compare per-head outputs against a materialized reference.")


@pytest.mark.parametrize("n_q,n_kv", [(4, 4), (4, 2), (4, 1)])
def test_attention_matches_sdpa(n_q, n_kv):
    q = torch.randn(2, n_q, 6, 8, generator=g())
    k = torch.randn(2, n_kv, 6, 8, generator=g(1))
    v = torch.randn(2, n_kv, 6, 8, generator=g(2))
    got = D.attention(q, k, v, D.causal_mask(6, 6))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert got.shape == ref.shape, "HINT: attention returns [B,Hq,S,dh]."
    assert torch.allclose(got, ref, atol=1e-5), (
        f"HINT: Hq={n_q}, Hkv={n_kv} mismatch. Check the 1/sqrt(d_h) scale, softmax over the KEY axis, "
        "mask orientation, and the head -> group map.")


def test_swiglu_branch_roles():
    x = torch.randn(3, 8, generator=g())
    wg, wu, wd = torch.randn(12, 8, generator=g(1)), torch.randn(12, 8, generator=g(2)), torch.randn(8, 12, generator=g(3))
    ref = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
    assert torch.allclose(D.swiglu(x, wg, wu, wd), ref, atol=1e-5), (
        "HINT: SiLU applies to the GATE branch only; the up branch is linear. nn.Linear-style weights are [out, in].")
