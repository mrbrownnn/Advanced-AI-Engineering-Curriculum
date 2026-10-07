"""Lab A: analytical KV payload model (Lesson 3.1). Worked-example numbers are the README's own."""
import pytest

from kv_model import GIB, concurrency_bound, kv_bytes_per_token, kv_heads_for, mla_bytes_per_token

HMEM = "HINT: After subtracting weights and buffers, how much memory is actually free for KV?"
HMLA = "HINT: For MLA, which tensors does the model retain?"
HFORM = "HINT: 2 (K and V) x bytes_per_element x cached_layers x num_kv_heads x head_dim."


def test_mha_worked_example():
    b = kv_bytes_per_token(32, kv_heads_for("MHA", 32), 128, 2)
    assert b == 524_288, f"Lesson 3.1 MHA example: got {b}. {HFORM}"
    assert 2048 * b == GIB, HFORM


def test_scaling_is_linear_in_each_factor():
    base = kv_bytes_per_token(10, 4, 64, 2)
    assert kv_bytes_per_token(20, 4, 64, 2) == 2 * base, HFORM
    assert kv_bytes_per_token(10, 8, 64, 2) == 2 * base, HFORM
    assert kv_bytes_per_token(10, 4, 128, 2) == 2 * base, HFORM
    assert kv_bytes_per_token(10, 4, 64, 1) == base / 2, HFORM


def test_variant_head_counts():
    assert kv_heads_for("MHA", 32) == 32, "HINT: MHA stores one KV head per query head."
    assert kv_heads_for("GQA", 32, 8) == 8, "HINT: GQA stores one KV head per group."
    assert kv_heads_for("MQA", 32) == 1, "HINT: MQA stores one KV head shared by all queries."
    with pytest.raises(ValueError):
        kv_heads_for("GQA", 32, 5)
    with pytest.raises(ValueError):
        kv_heads_for("MLA", 32)
    mha = kv_bytes_per_token(32, kv_heads_for("MHA", 32), 128, 2)
    gqa = kv_bytes_per_token(32, kv_heads_for("GQA", 32, 8), 128, 2)
    mqa = kv_bytes_per_token(32, kv_heads_for("MQA", 32), 128, 2)
    assert mha / gqa == 4 and mha / mqa == 32, HFORM


def test_mla_worked_example_and_head_independence():
    b = mla_bytes_per_token(60, 512, 64, 2)
    assert b == 69_120, f"Lesson 3.1 MLA example: got {b}. {HMLA}"
    assert 4096 * b == 270 * 1024 ** 2, HMLA
    gqa_misuse = kv_bytes_per_token(60, 128, 128, 2)
    assert gqa_misuse / b > 50, f"GQA formula on an MLA config should be ~57x larger. {HMLA}"


def test_concurrency_bound_guided_practice():
    per_tok = kv_bytes_per_token(32, 32, 128, 2)
    n = concurrency_bound(80, 15, 2, 4096, per_tok)
    assert isinstance(n, int), "HINT: a request count is an integer: floor, not round."
    assert n == 31, f"Lesson 3.1 guided practice: got {n}. {HMEM}"
    assert concurrency_bound(80, 0, 0, 4096, per_tok) > n, HMEM
    assert concurrency_bound(16, 15, 2, 4096, per_tok) == 0, HMEM
