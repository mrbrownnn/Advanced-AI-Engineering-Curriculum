import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import decoder as D
import sampling as SM

LESSON = torch.tensor([[2.0, 1.0, 0.0, -1.0]])  # Lesson 1.2 worked example
TIE = torch.tensor([[3.0, 1.0, 1.0, 0.0]])      # Lesson 1.2 guided practice


def kept(scores):
    return set(torch.nonzero(torch.isfinite(scores[0])).flatten().tolist())


def test_temperature_zero_is_an_error():
    with pytest.raises(ValueError, match="do_sample"):
        SM.apply_temperature(LESSON, 0.0)
    with pytest.raises(ValueError):
        SM.apply_temperature(LESSON, -1.0)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(-20, 20), min_size=2, max_size=20, unique=True), st.floats(0.05, 10.0))
def test_temperature_preserves_argmax(vals, t):
    z = torch.tensor([vals], dtype=torch.float64)
    assert SM.apply_temperature(z, t).argmax() == z.argmax(), "HINT: dividing by T > 0 preserves logit order, so the argmax cannot move."


def test_temperature_entropy_monotone():
    def entropy(t):
        p = torch.softmax(SM.apply_temperature(LESSON, t), -1)
        return -(p * p.log()).sum().item()
    assert entropy(0.5) < entropy(1.0) < entropy(2.0), "HINT: T < 1 sharpens, T > 1 flattens; check you divide (not multiply) by T."


def test_top_k_kept_set_and_ties():
    assert kept(SM.top_k_filter(LESSON, 2)) == {0, 1}, "HINT: top-k=2 on Lesson 1.2 logits keeps {A, B}."
    tie = kept(SM.top_k_filter(TIE, 2))
    assert tie == {0, 1, 2}, (
        "HINT: the pinned rule removes scores strictly below the k-th value (scores < kth), so a tie at the "
        "boundary keeps more than k tokens. Count the -inf entries.")


def test_top_p_kept_set():
    assert kept(SM.top_p_filter(LESSON, 0.9)) == {0, 1, 2}, (
        "HINT: smallest set whose mass reaches 0.9 is {A,B,C}. Ascending rule: remove cumulative <= 1 - p.")
    assert kept(SM.top_p_filter(LESSON, 0.5)) == {0}, "HINT: A alone already exceeds 0.5."
    assert len(kept(SM.top_p_filter(LESSON, 0.0))) == 1, "HINT: top-p always keeps at least min_tokens_to_keep (1) token."


def test_truncation_renormalizes_over_kept_set_only():
    p_raw = torch.softmax(LESSON, -1)[0]
    for scores in (SM.top_k_filter(LESSON, 2), SM.top_p_filter(LESSON, 0.9)):
        p = torch.softmax(scores, -1)[0]
        ks = sorted(kept(scores))
        assert math.isclose(p.sum().item(), 1.0, rel_tol=1e-6), "HINT: post-truncation probabilities must sum to 1."
        assert p[[i for i in range(4) if i not in ks]].sum() == 0, "HINT: removed tokens must get exactly 0 probability (-inf score)."
        assert math.isclose((p[ks[1]] / p[ks[0]]).item(), (p_raw[ks[1]] / p_raw[ks[0]]).item(), rel_tol=1e-5), (
            "HINT: renormalize over the kept set; ratios between surviving tokens must not change.")


def test_processor_order_can_change_kept_set():
    a = SM.process(LESSON, temperature=2.0, top_p=0.9, order=("temperature", "top_k", "top_p"))
    b = SM.process(LESSON, temperature=2.0, top_p=0.9, order=("top_p", "top_k", "temperature"))
    assert kept(a) != kept(b), (
        "HINT: temperature before top-p flattens the distribution first, so the nucleus grows. "
        "process() must honour the order argument.")
    assert SM.process(LESSON).dtype == torch.float32, "HINT: the contract says processors run on float32 scores."


def test_seeded_sampling_distribution():
    contract = SM.sampling_contract(do_sample=True, top_k=2)
    logits = LESSON.expand(4000, 4)
    toks = SM.sample_next(logits, contract, torch.Generator().manual_seed(0))
    assert set(toks.tolist()) <= {0, 1}, "HINT: top-k=2 must make tokens C and D impossible."
    p_a = torch.softmax(LESSON[0, :2], -1)[0].item()
    freq = (toks == 0).float().mean().item()
    se = math.sqrt(p_a * (1 - p_a) / 4000)
    assert abs(freq - p_a) < 5 * se, "HINT: empirical frequency of A is off the renormalized top-2 probability; sample from softmax(processed scores)."
    again = SM.sample_next(logits, contract, torch.Generator().manual_seed(0))
    assert torch.equal(toks, again), "HINT: same seed and generator must reproduce the draw; pass the generator to torch.multinomial."


def test_greedy_ignores_temperature():
    z = torch.randn(3, 10, generator=torch.Generator().manual_seed(0))
    for t in (0.3, 3.0):
        c = SM.sampling_contract(do_sample=False, temperature=t)
        assert torch.equal(SM.sample_next(z, c), z.argmax(-1)), "HINT: greedy is argmax of the logits, not sampling at low T."


REQUIRED = {"logit_position", "logit_dtype", "processor_order", "temperature", "top_k", "top_p", "do_sample",
            "library_defaults", "seed", "generator_device", "stop_token_ids", "max_new_tokens",
            "context_limit", "truncation_policy", "tokenizer"}


def test_contract_fields():
    c = SM.sampling_contract()
    missing = REQUIRED - set(c)
    assert not missing, f"HINT: a sampling contract records more than temperature; missing {sorted(missing)} (Lesson 1.2)."
    print("SAMPLING CONTRACT:", c)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return D.TinyDecoder(D.Config()).eval()


def test_generate_greedy_cache_and_stop(model):
    prompt = torch.tensor([[1, 2, 3, 4]])
    c_cache = SM.sampling_contract(do_sample=False, max_new_tokens=6, use_cache=True)
    c_full = SM.sampling_contract(do_sample=False, max_new_tokens=6, use_cache=False)
    a, b = SM.generate(model, prompt, c_cache), SM.generate(model, prompt, c_full)
    assert len(a) == 6 and a == b, "HINT: greedy decode with and without the KV cache must emit identical tokens."
    with torch.no_grad():
        first = model(prompt)[0][0, -1].argmax().item()
    assert a[0] == first, "HINT: the first new token is argmax of the LAST-position logits of the prompt."
    stop = SM.sampling_contract(do_sample=False, max_new_tokens=6, stop_token_ids=[a[1]])
    assert SM.generate(model, prompt, stop) == a[:2], "HINT: stop after emitting a stop token (include it), not before."
    with pytest.raises(ValueError):
        SM.generate(model, prompt, SM.sampling_contract(max_new_tokens=100, context_limit=64))


def test_generate_sampled_reproducible(model):
    prompt = torch.tensor([[5, 6]])
    c = SM.sampling_contract(do_sample=True, temperature=1.5, top_p=0.95, max_new_tokens=8)
    a = SM.generate(model, prompt, c, torch.Generator().manual_seed(7))
    b = SM.generate(model, prompt, c, torch.Generator().manual_seed(7))
    assert a == b, "HINT: identical contract + seed + generator device must reproduce sampled tokens on CPU."
