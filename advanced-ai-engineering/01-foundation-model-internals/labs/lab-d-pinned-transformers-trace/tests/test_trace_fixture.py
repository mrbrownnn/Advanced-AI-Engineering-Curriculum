import pathlib
import yaml

LAB=pathlib.Path(__file__).resolve().parents[1]
DATA=yaml.safe_load((LAB/'fixtures'/'pinned-trace.yaml').read_text())


def test_revision_and_paths_are_pinned():
    assert len(DATA['commit'])==40 and DATA['commit']=='11c16613d93911300c38ec8c9c2460567be53281', (
        'HINT: source evidence needs an exact immutable revision, not main or a package version.')
    assert set(DATA['files'])=={'llama','masking','generation','logits','config'}, (
        'HINT: trace model orchestration, mask construction, generation loop, warpers, and generation defaults at the same revision.')


def test_required_symbols_covered():
    s=set(DATA['symbols'])
    required={'LlamaForCausalLM.forward','LlamaModel.forward','create_causal_mask','LlamaRotaryEmbedding.forward',
              'LlamaDecoderLayer.forward','LlamaAttention.forward','eager_attention_forward','LlamaRMSNorm.forward',
              'GenerationMixin._get_logits_processor','GenerationMixin._sample','GenerationConfig._get_default_generation_params',
              'TemperatureLogitsWarper','TopKLogitsWarper','TopPLogitsWarper'}
    assert required<=s, f"HINT: source trace is incomplete; missing {sorted(required-s)}. Map every §06 required boundary to a symbol."


def test_static_facts_record_dispatch_and_sampling_boundaries():
    f=DATA['static_facts']
    assert f['mask_dispatch_field']==f['attention_dispatch_field']=='config._attn_implementation', (
        'HINT: record the configured backend field before making a kernel claim; eager source is not proof eager ran.')
    assert f['generation_logit_slice']=='outputs.logits[:, -1]' and f['generation_logit_copy_dtype']=='float32', (
        'HINT: _sample selects last-position logits then copies them to float32 before processors.')
    assert f['sampler_order'][:4]==['temperature','top_h','top_k','top_p'], (
        'HINT: read _get_logits_processor in source order; at this revision top_h sits between temperature and top-k.')
    assert f['default_top_k']==50 and f['default_do_sample'] is False, (
        'HINT: unset fields are filled by _get_default_generation_params; sampled default top_k can truncate.')
    assert f['top_k_removal_comparison']=='scores_strictly_less_than_kth', (
        'HINT: strict < means ties at the kth score survive.')
    assert f['top_p_removal_comparison']=='cumulative_less_than_or_equal_to_one_minus_p', (
        'HINT: pinned top-p sorts ascending and removes cumulative <= 1-p while retaining at least one token.')


def test_static_and_executed_evidence_not_conflated():
    e=DATA['execution_status']
    assert e['forward_executed'] is False and e['backend_observed'] is False and e['processors_observed'] is False, (
        'HINT: static source proves possible orchestration, not which backend or effective processors ran. Mark unrun paths TODO_VERIFY.')
    assert 'TODO_VERIFY' in e['note'], 'HINT: every unexecuted branch must carry TODO_VERIFY, not an inferred runtime claim.'
