import pytest
import torch

import decoder as D
from reference import reference_logits


@pytest.fixture
def model():
    torch.manual_seed(0)
    return D.TinyDecoder(D.Config()).eval()


def ids(b=2, s=8, seed=1, vocab=64):
    return torch.randint(0, vocab, (b, s), generator=torch.Generator().manual_seed(seed))


def test_shapes_and_finite(model):
    x = ids()
    logits, cache = model(x)
    assert logits.shape == (2, 8, model.cfg.vocab), "HINT: logits are [B,S,V] - one next-token score vector per position."
    assert torch.isfinite(logits).all(), "HINT: non-finite logits; count finite values after every layer (norm dtype? -inf rows in the mask?)."
    assert len(cache) == model.cfg.n_layers, "HINT: return one (k, v) pair per layer."
    k, v = cache[0]
    assert k.shape == (2, model.cfg.n_kv, 8, model.cfg.d_h) and v.shape == k.shape, (
        "HINT: cache K/V are [B,H_kv,T,d_h]: K/V heads, not query heads.")


def test_future_token_perturbation(model):
    x = ids()
    base, _ = model(x)
    for j in (7, 3):
        x2 = x.clone()
        x2[:, j] = (x2[:, j] + 1) % model.cfg.vocab
        out, _ = model(x2)
        assert torch.allclose(out[:, :j], base[:, :j], atol=1e-6), (
            f"HINT: changing token {j} moved logits at positions < {j}. Change only the last token: which "
            "positions' logits are allowed to move? Check mask orientation and any axis mixing in reshapes/norms.")
        assert not torch.allclose(out[:, j], base[:, j]), f"HINT: position {j} must see its own token."


def test_parity_with_transparent_reference(model):
    x = ids()
    got, _ = model(x)
    with torch.no_grad():
        ref = reference_logits(model, x)
    err = (got - ref).abs().max().item()
    assert err < 1e-4, (
        f"HINT: final-logit error {err:.2e} vs a built-in reference. Do not stop at the final logits: hook each "
        "layer and find the first operation that diverges (norm, RoPE pairing, head/sequence transpose, residual order).")


def test_incremental_decode_matches_full_recompute(model):
    x = ids(s=10)
    full, _ = model(x)
    logits, cache = model(x[:, :6])
    steps = [logits]
    for t in range(6, 10):
        logits, cache = model(x[:, t:t + 1], cache)
        steps.append(logits)
    inc = torch.cat(steps, dim=1)
    assert inc.shape == full.shape, "HINT: each decode step returns [B,1,V]."
    assert torch.allclose(inc, full, atol=1e-5), (
        "HINT: cached decode diverges from full recompute. New tokens need absolute positions T_past.. for RoPE, "
        "the mask must offset query positions by the cached length, and the cache must hold ROTATED keys.")


def test_dtype_parity_float64(model):
    x = ids()
    f32, _ = model(x)
    m64 = D.TinyDecoder(model.cfg)
    m64.load_state_dict(model.state_dict())
    f64, _ = m64.double()(x)
    assert f64.dtype == torch.float64, "HINT: a float64 model should produce float64 logits; do not hard-cast to float32."
    assert torch.allclose(f32.double(), f64, atol=1e-4), (
        "HINT: float32 and float64 disagree beyond accumulation error; look for a cast that loses precision or a wrong axis.")


def test_gradients_reach_every_parameter(model):
    model.train()
    logits, _ = model(ids())
    logits.logsumexp(-1).mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"HINT: {name} gets no finite gradient; is it used in forward?"
        assert p.grad.abs().sum() > 0, f"HINT: {name} has an all-zero gradient; a branch may be detached or unused."
