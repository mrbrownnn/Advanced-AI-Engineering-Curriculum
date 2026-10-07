import pytest
import torch

import attention as A


@pytest.mark.parametrize("n_kv,family,group", [(8, "MHA", 1), (4, "GQA", 2), (1, "MQA", 8)])
def test_geometry_endpoints(n_kv, family, group):
    g = A.geometry(d=64, n_q=8, n_kv=n_kv, batch=2, sequence=7, dtype_bytes=2)
    assert g["family"] == family and g["group_size"] == group, (
        "HINT: MHA is Hkv=Hq, MQA is Hkv=1, and intermediate divisors are GQA; group g=Hq/Hkv.")
    assert g["head_dim"] == 8, "HINT: standard head width is d/Hq."
    assert g["q_elements"] == 2 * 8 * 7 * 8, "HINT: Q is [B,Hq,S,dh]."
    assert g["k_elements"] == g["v_elements"] == 2 * n_kv * 7 * 8, "HINT: K/V are [B,Hkv,S,dh], not Hq."
    assert g["q_parameters"] == g["o_parameters"] == 64 * 64, "HINT: reducing KV heads does not shrink Q or O."
    assert g["k_parameters"] == g["v_parameters"] == 64 * n_kv * 8, "HINT: K/V projection width is Hkv*dh."
    assert g["kv_bytes"] == 2 * 2 * 7 * n_kv * 8 * 2, "HINT: K AND V payload: 2*B*S*Hkv*dh*dtype_bytes."


@pytest.mark.parametrize("args", [(63, 8, 2), (64, 8, 3), (64, 0, 1), (64, 8, 0)])
def test_invalid_geometry_rejected(args):
    with pytest.raises(ValueError):
        A.validate_geometry(*args)


def test_repeat_kv_maps_contiguous_groups():
    x = torch.zeros(1, 3, 2, 4)
    x[:, 0], x[:, 1], x[:, 2] = 10, 20, 30
    got = A.repeat_kv(x, 3)
    assert got.shape == (1, 9, 2, 4), "HINT: output head count is Hkv*group_size."
    assert got[:, :, 0, 0].tolist() == [[10, 10, 10, 20, 20, 20, 30, 30, 30]], (
        "HINT: expected contiguous head blocks; query h reads KV h//g, not h%Hkv. Inspect each head, not only shape.")


@pytest.mark.parametrize("n_kv", [8, 4, 2, 1])
def test_gqa_matches_materialized_reference(n_kv):
    torch.manual_seed(0)
    m = A.GQAAttention(64, 8, n_kv).eval()
    x = torch.randn(2, 9, 64)
    with torch.no_grad():
        got, ref = m(x), A.mha_reference(m, x)
    assert got.shape == x.shape, "HINT: merge query heads back to [B,S,d] before output projection."
    assert torch.allclose(got, ref, atol=2e-6), (
        f"HINT: Hkv={n_kv} differs from explicit repeated-KV attention. Compare per-head output; check contiguous "
        "grouping, mask diagonal/orientation, softmax key axis, and 1/sqrt(dh) scale.")


def test_mha_endpoint_matches_plain_sdpa():
    torch.manual_seed(2)
    m = A.GQAAttention(32, 4, 4).eval()
    x = torch.randn(1, 5, 32)
    q = m.q_proj(x).view(1, 5, 4, 8).transpose(1, 2)
    k = m.k_proj(x).view(1, 5, 4, 8).transpose(1, 2)
    v = m.v_proj(x).view(1, 5, 4, 8).transpose(1, 2)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = m.o_proj(ref.transpose(1, 2).reshape(1, 5, 32))
    assert torch.allclose(m(x), ref, atol=1e-6), "HINT: when Hkv=Hq, GQA must reduce exactly to ordinary MHA."


def test_parameter_count_matches_module_geometry():
    for n_kv in (8, 4, 1):
        m = A.GQAAttention(64, 8, n_kv)
        actual = sum(p.numel() for p in m.parameters())
        g = A.geometry(64, 8, n_kv, 1, 1)
        expected = sum(g[k] for k in ("q_parameters", "k_parameters", "v_parameters", "o_parameters"))
        assert actual == expected, (
            "HINT: derive counts from each instantiated projection's matrix; do not hardcode a family total.")


def test_causal_isolation_all_families():
    for n_kv in (8, 4, 1):
        torch.manual_seed(0)
        m = A.GQAAttention(64, 8, n_kv).eval()
        x = torch.randn(1, 7, 64)
        x2 = x.clone(); x2[:, -1] += 10
        assert torch.allclose(m(x)[:, :-1], m(x2)[:, :-1], atol=1e-6), (
            f"HINT: future-token change leaked backward for Hkv={n_kv}; mask must allow j<=i and reject j>i.")
