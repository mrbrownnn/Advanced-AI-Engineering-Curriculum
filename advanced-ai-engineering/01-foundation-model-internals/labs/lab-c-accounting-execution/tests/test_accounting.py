import pytest
import torch

import accounting as A


@pytest.mark.parametrize("overrides", [
    {}, {"bias":True}, {"tie_embeddings":True}, {"adapter_rank":4},
    {"bias":True,"tie_embeddings":True,"adapter_rank":3},
    {"experts":3,"experts_per_token":1}, {"experts":3,"experts_per_token":2,"bias":True},
])
def test_analytical_count_reconciles_with_actual(overrides):
    c=A.Config(**overrides); torch.manual_seed(0); m=A.TinyModel(c)
    analytical=A.analytical_parameter_count(c); actual=A.actual_parameter_count(m,unique_storage=True)
    assert analytical==actual, (
        f"HINT: analytical={analytical}, unique actual={actual}, delta={analytical-actual}. Inspect one manifest row "
        "per real matrix; represent tying, biases, adapters, router, experts, and norm scales explicitly.")
    rows=A.parameter_manifest(c)
    assert all({"name","formula","assumptions","count"}<=set(r) for r in rows), (
        "HINT: each manifest row needs name, formula, assumptions, analytical count so its delta can have a named cause.")
    assert len({r["name"] for r in rows})==len(rows), "HINT: manifest group names must be unique and map to one logical tensor group."


def test_tied_storage_not_double_counted():
    u=A.Config(tie_embeddings=False); t=A.Config(tie_embeddings=True)
    mu,mt=A.TinyModel(u),A.TinyModel(t)
    assert A.actual_parameter_count(mu)-A.actual_parameter_count(mt)==t.vocab*t.d, (
        "HINT: tying aliases lm_head.weight to embed.weight, removing one unique V*d storage, not both logical names.")
    named_with_alias=A.actual_parameter_count(mt,unique_storage=False)
    assert named_with_alias-A.actual_parameter_count(mt,True)==t.vocab*t.d, (
        "HINT: request named_parameters(remove_duplicate=False) to expose the alias; deduplicate by storage pointer/offset/numel.")


def test_bias_and_adapter_deltas_derive_from_modules():
    base=A.Config(layers=1); biased=A.Config(layers=1,bias=True); adapted=A.Config(layers=1,adapter_rank=5)
    mb,mi,ma=A.TinyModel(base),A.TinyModel(biased),A.TinyModel(adapted)
    actual_bias=A.actual_parameter_count(mi)-A.actual_parameter_count(mb)
    expected_from_modules=sum(mod.bias.numel() for mod in mi.modules() if isinstance(mod,torch.nn.Linear) and mod.bias is not None)
    assert actual_bias==expected_from_modules, "HINT: bias delta is the sum of instantiated Linear bias vectors; include Q/K/V/O and 3 MLP biases."
    actual_adapter=A.actual_parameter_count(ma)-A.actual_parameter_count(mb)
    expected=sum(p.numel() for n,p in ma.named_parameters() if "adapter_" in n)
    assert actual_adapter==expected, "HINT: a bias-free rank-r adapter has down [r,d] plus up [d,r] weights."


def test_logical_flops_match_runtime_flop_counter():
    from torch.utils.flop_counter import FlopCounterMode
    # Bias-free dense case: exclusions avoid elementwise/norm ops; PyTorch counter counts the same mm/addmm FMA=2.
    c=A.Config(vocab=31,d=32,m=48,n_q=4,n_kv=4,layers=1,bias=False)
    m=A.TinyModel(c).eval(); ids=torch.randint(0,c.vocab,(2,7))
    with FlopCounterMode(display=False) as fc, torch.no_grad(): m(ids)
    measured=fc.get_total_flops(); expected=A.logical_flops(c,2,7)
    assert expected["fma_convention"]==2, "HINT: state beside every number that one fused multiply-add counts as 2 FLOPs."
    # PyTorch 2.14 FlopCounterMode does not register CPU aten::_scaled_dot_product_flash_attention_for_cpu.
    # Reconcile all registered projection/MLP/head matmuls, and name the missing fused-attention term.
    assert measured==expected["total"]-expected["attention_flops"], (
        f"HINT: logical non-attention matmuls {expected['total']-expected['attention_flops']} != profiler {measured}. "
        "Include Q,K,V,O, 3 SwiGLU projections and LM head. The CPU fused SDPA op is unregistered here; its "
        "nominal QK+AV term remains an analytical row rather than silently disappearing.")


def test_gqa_reduces_projection_work_not_attention_term():
    mha=A.logical_flops(A.Config(n_kv=4),2,11); mqa=A.logical_flops(A.Config(n_kv=1),2,11)
    assert mqa["weight_flops"]<mha["weight_flops"], "HINT: K/V projection width shrinks with Hkv."
    assert mqa["attention_flops"]==mha["attention_flops"], (
        "HINT: query heads and attention over context remain; nominal QK+AV is 4*B*S^2*d independent of Hkv.")


def test_kv_bytes_against_measured_tensors():
    for nkv in (4,2,1):
        c=A.Config(n_kv=nkv,layers=3); b,s=2,13
        k=torch.empty(c.layers,b,nkv,s,c.dh,dtype=torch.float32); v=torch.empty_like(k)
        measured=k.numel()*k.element_size()+v.numel()*v.element_size()
        assert A.kv_bytes(c,b,s,4)==measured, (
            "HINT: compare actual K and V tensor numel*element_size; formula is 2*L*B*S*Hkv*dh*dtype_bytes.")


def test_mla_pinned_geometry_rejects_gqa_formula():
    # Pinned DeepSeek-V2 config at revision 4461458...: L=60, kv_lora_rank=512, qk_rope_head_dim=64.
    out=A.mla_retained_elements(60,512,64)
    assert out["per_layer_per_token"]==512+64 and out["all_layers_per_token"]==60*(512+64), (
        "HINT: MLA retains c_KV (kv_lora_rank) plus shared k_R (qk_rope_head_dim) per token/layer.")
    assert set(out["cached_tensors"])=={"c_KV","k_R"} and out["gqa_formula_applies"] is False, (
        "HINT: per-head K/V are reconstructed by W^UK/W^UV; num_key_value_heads is not the MLA retained-state formula.")
    wrong=2*60*128*128
    assert wrong!=out["all_layers_per_token"], "HINT: this explicit trap must differ: using num_key_value_heads would classify MLA as MHA."


def test_moe_total_selected_and_executed_distinct():
    x=A.moe_counts(d=32,m=48,experts=8,selected=2)
    assert x["total_expert_parameters"]==4*x["selected_weight_elements_per_token"], (
        "HINT: total uses E experts; selected per token uses k. Neither alone predicts padding, routing, or wall-clock.")
    with pytest.raises(ValueError): A.moe_counts(32,48,2,3)


def test_model_shapes_and_finite():
    for c in (A.Config(),A.Config(n_kv=1,bias=True,tie_embeddings=True,adapter_rank=4),A.Config(experts=3,experts_per_token=2)):
        torch.manual_seed(0); y=A.TinyModel(c)(torch.randint(0,c.vocab,(2,6)))
        assert y.shape==(2,6,c.vocab), "HINT: real module must execute [B,S] -> [B,S,V], not serve only as parameter containers."
        assert torch.isfinite(y).all(), "HINT: profile real finite execution; inspect norm/mask/router if non-finite."
