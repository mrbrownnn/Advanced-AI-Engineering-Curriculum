# Expert Track — Modules 25–36

Modules 00–24 build a senior inference/agent/evaluation engineer. This track
covers the remaining ~10% that usually separates senior from expert: work done
by people who **train, post-train, compress, interpret, and hand-tune** frontier
models, not only serve them.

Selection rule: a topic enters this track only if (a) Modules 00–24 do not own
it (checked by keyword sweep of module READMEs on 2026-10-12), (b) it requires
reasoning that cannot be done from framework docs alone, and (c) it can be
broken and measured on CPU, Colab-class GPU, or a pinned trace.

Status: **PROPOSED**. Each module below is a scope contract, not authored
content. Authoring follows [INSTRUCTIONAL_SCHEMA.md](INSTRUCTIONAL_SCHEMA.md)
and [DEPTH-CONTRACT.md](DEPTH-CONTRACT.md); evidence follows
[agent.md](../agent.md) (O/D/H, Tier 1 first).

> **Source status.** Sources listed are discovery leads from model memory,
> not yet re-opened for this file. Each must be opened and verified (title,
> version, authors, URL) before an authored module cites it. Mark `TODO_VERIFY`
> until then. No numbers from these papers are quoted here on purpose.

Hardware tiers: `CPU` · `COLAB-GPU` · `TRACE` (pinned source/log replay).

---

## Gap map

| Gap | Nearest existing module | Why existing coverage is not enough |
|---|---|---|
| Pretraining at scale | 19 (adaptation only) | No scaling laws, μP, instabilities, data mixtures |
| RL post-training systems | 19 (GRPO named) | No rollout/train asymmetry, RLVR, reward hacking dynamics |
| Interpretability | — | Not covered |
| Kernel engineering below Triton | 05 | No TMA/warp-specialization, CUTLASS/CuTe, MX/FP4 |
| Numerics & determinism | 02, 16 | Nondeterminism named, batch invariance not taught |
| Non-attention / sparse architectures | 01, 24 | MLA/MoE only; no SSM, linear/sparse attention, MTP, diffusion LM |
| MoE training dynamics | 20 (inference EP) | No router collapse, load-balancing losses |
| Compression beyond quantization | 05 | No distillation, structured pruning, merging |
| Multi-tenant adapter serving | 19, 04 | No S-LoRA/Punica-style batching |
| Privacy, memorization, unlearning | 08, 18 | Contamination only; no extraction, DP, unlearning evaluation |
| Computer-use / long-horizon agent eval | 12, 15 | Generic loops; no environment-grounded benchmarks |
| Edge / on-device inference | — | Not covered |

---

## Phase 9 — Training Systems

### 25 — Pretraining Dynamics & Scaling
- **Owns:** compute-optimal allocation, scaling-law fitting and its failure
  (extrapolation error, data-repetition regime), μP and hyperparameter transfer,
  loss spikes and logit growth, warmup/LR schedules (WSD vs cosine), data-mixture
  optimization, checkpoint-averaging.
- **Expert signal:** predict a large run's loss from small proxies *and* state
  when the prediction is invalid.
- **Break lab (COLAB-GPU):** train tiny decoders across width with SP vs μP;
  show LR optimum drift under SP; induce attention-logit divergence, fix with
  QK-norm / z-loss, measure.
- **Leads:** Kaplan et al. 2020 (arXiv:2001.08361); Hoffmann et al. 2022
  "Chinchilla" (arXiv:2203.15556); Yang et al. 2022 Tensor Programs V
  (arXiv:2203.03466); Wortsman et al. 2023 small-scale proxies for instability
  (arXiv:2309.14322); Xie et al. 2023 DoReMi (arXiv:2305.10429); Groeneveld et
  al. 2024 OLMo (arXiv:2402.00838).

### 26 — Distributed Training & Low-Precision Training
- **Owns:** ZeRO stages / FSDP sharding, activation checkpointing trade-off,
  TP/PP/CP for training (1F1B, interleaving, bubbles), FP8 training (scaling
  recipes, delayed vs current scaling), MX formats, failure/restart economics
  (checkpoint interval vs MTBF).
- **Boundary:** Module 20 owns inference parallelism; this owns backward pass,
  optimizer state, and gradient communication.
- **Break lab (CPU/COLAB):** memory ledger per ZeRO stage, verified against
  `torch.cuda.max_memory_allocated`; simulate pipeline bubble vs microbatch count;
  inject FP8 overflow.
- **Leads:** Rajbhandari et al. 2019 ZeRO (arXiv:1910.02054); Zhao et al. 2023
  PyTorch FSDP (arXiv:2304.11277); Shoeybi et al. 2019 Megatron-LM
  (arXiv:1909.08053); Micikevicius et al. 2022 FP8 formats (arXiv:2209.05433);
  Rouhani et al. 2023 Microscaling (arXiv:2310.10537); DeepSeek-AI 2024
  DeepSeek-V3 report, FP8 training section (arXiv:2412.19437).

### 27 — RL Post-Training Systems
- **Owns:** RLHF vs RLVR, PPO/GRPO estimator bias (length normalization, std
  normalization), KL placement, verifier design, reward hacking and
  over-optimization, rollout/training engine split, weight sync, off-policy
  staleness in async RL, entropy collapse.
- **Boundary:** Module 19 owns *whether* to adapt and SFT/DPO mechanics; this
  owns the RL loop as a distributed system and as an estimator.
- **Break lab (COLAB-GPU):** GRPO on a toy verifiable task; ablate length
  normalization and show length drift; plant an exploitable verifier and detect
  hacking via held-out verifier.
- **Leads:** Shao et al. 2024 DeepSeekMath/GRPO (arXiv:2402.03300); DeepSeek-AI
  2025 R1 (arXiv:2501.12948); Liu et al. 2025 Dr. GRPO (arXiv:2503.20783); Yu et
  al. 2025 DAPO (arXiv:2503.14476); Gao et al. 2022 reward-model
  overoptimization (arXiv:2210.10760); Sheng et al. 2024 HybridFlow/verl
  (arXiv:2409.19256); Lambert et al. 2024 Tülu 3 RLVR (arXiv:2411.15124).

### 28 — MoE Training Dynamics
- **Owns:** router collapse, auxiliary vs aux-loss-free balancing, capacity
  factor and token dropping, expert specialization measurement, router z-loss,
  dropless MoE kernels.
- **Boundary:** Module 20 owns EP placement at inference; Module 24 owns
  dense-vs-MoE choice.
- **Break lab (COLAB-GPU):** small MoE; remove balancing, observe collapse;
  compare aux loss vs bias-update balancing on load entropy and quality.
- **Leads:** Lepikhin et al. 2020 GShard (arXiv:2006.16668); Fedus et al. 2021
  Switch (arXiv:2101.03961); Zoph et al. 2022 ST-MoE (arXiv:2202.08906); Gale et
  al. 2022 MegaBlocks (arXiv:2211.15841); Wang et al. 2024 auxiliary-loss-free
  balancing (arXiv:2408.15664).

---

## Phase 10 — Deep Systems

### 29 — Kernel Engineering Below Triton
- **Owns:** Hopper/Blackwell execution features (TMA, warpgroup MMA, warp
  specialization, async pipelines), CUTLASS/CuTe layouts, persistent kernels,
  CUDA Graphs and capture hazards, `torch.compile` graph breaks, FP4/MX GEMM.
- **Boundary:** Module 05 owns *choosing* optimizations and Triton basics; this
  owns writing and reading production kernels.
- **Break lab (TRACE + COLAB-GPU):** source-trace a pinned FlashAttention-3 or
  DeepGEMM kernel; reproduce a CUDA-Graph capture bug (dynamic shape, host
  sync); measure graph-break cost.
- **Leads:** Shah et al. 2024 FlashAttention-3 (arXiv:2407.08608); Spector et
  al. 2024 ThunderKittens (arXiv:2410.20399); Ye et al. 2025 FlashInfer
  (arXiv:2501.01005); NVIDIA CUTLASS repo; deepseek-ai/DeepGEMM repo.

### 30 — Numerics, Determinism & Reproducibility
- **Owns:** floating-point non-associativity, batch-invariance failure in
  serving (same prompt, different batch, different output), split-K and
  reduction order, deterministic kernels cost, RL train/infer logprob mismatch,
  numerical drift across hardware.
- **Boundary:** Module 02 owns timing; Module 16 owns differential testing as
  technique. This owns the numerical root cause.
- **Break lab (CPU):** show order-dependent sum; build a batch-size-dependent
  matmul reduction, then a batch-invariant one; measure divergence of greedy
  outputs.
- **Leads:** Horace He and Thinking Machines Lab, 2025-09-10, "Defeating
  Nondeterminism in LLM Inference"
  (https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/,
  DOI 10.64434/tml.20250910);
  Goldberg 1991 "What Every Computer Scientist Should Know About Floating-Point
  Arithmetic".

### 31 — Beyond Dense Attention
- **Owns:** SSMs (selective scan, chunked parallel form), linear/gated
  attention, hybrid stacks, trainable sparse attention, multi-token prediction,
  diffusion language models; serving consequences (constant state vs KV,
  prefix caching breaks for recurrent state).
- **Boundary:** Module 01 teaches standard decoder; Module 24 does trade-off at
  system level. This teaches mechanisms and their serving contract.
- **Break lab (CPU):** implement selective scan recurrent vs chunked forms,
  prove equivalence; show associative-recall failure of fixed-size state.
- **Leads:** Gu & Dao 2023 Mamba (arXiv:2312.00752); Dao & Gu 2024 Mamba-2
  (arXiv:2405.21060); Yang et al. 2023 GLA (arXiv:2312.06635); Lieber et al. 2024
  Jamba (arXiv:2403.19887); Yuan et al. 2025 NSA (arXiv:2502.11089); Gloeckle et
  al. 2024 multi-token prediction (arXiv:2404.19737); Nie et al. 2025 LLaDA
  (arXiv:2502.09992).

### 32 — Compression: Distillation, Pruning, Merging
- **Owns:** logit vs sequence distillation, forward vs reverse KL, on-policy
  distillation, unstructured vs structured pruning (and why unstructured rarely
  speeds up), width/depth pruning + distillation recovery, weight merging
  (task arithmetic, TIES) and its interference failure.
- **Boundary:** Module 05 owns quantization.
- **Break lab (COLAB-GPU):** distill a small model with forward vs reverse KL,
  compare mode coverage; prune 50% unstructured, measure zero speedup on dense
  kernel; merge two LoRA tasks, measure interference.
- **Leads:** Hinton et al. 2015 (arXiv:1503.02531); Gu et al. 2023 MiniLLM
  (arXiv:2306.08543); Frantar & Alistarh 2023 SparseGPT (arXiv:2301.00774); Sun
  et al. 2023 Wanda (arXiv:2306.11695); Muralidharan et al. 2024 Minitron
  (arXiv:2407.14679); Ilharco et al. 2022 task arithmetic (arXiv:2212.04089);
  Yadav et al. 2023 TIES (arXiv:2306.01708).

---

## Phase 11 — Model Understanding & Trust

### 33 — Mechanistic Interpretability for Engineers
- **Owns:** linear probes and their confounds, activation patching, sparse
  autoencoders (dead latents, reconstruction vs downstream loss), attribution
  graphs, steering vectors and their side effects; using interp as a debugging
  tool for regressions and jailbreaks, not as proof.
- **Break lab (COLAB-GPU):** train SAE on a small model's residual stream;
  find a feature; steer; measure off-target capability loss; show probe
  succeeding on a spurious feature.
- **Leads:** Bricken et al. 2023 Towards Monosemanticity
  (transformer-circuits.pub); Templeton et al. 2024 Scaling Monosemanticity
  (transformer-circuits.pub); Cunningham et al. 2023 SAE (arXiv:2309.08600);
  Lieberum et al. 2024 Gemma Scope (arXiv:2408.05147); Zou et al. 2023
  Representation Engineering (arXiv:2310.01405); Ameisen et al. 2025-03-27
  "Circuit Tracing: Revealing Computational Graphs in Language Models"
  (https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

### 34 — Memorization, Privacy & Unlearning
- **Owns:** extractable vs discoverable memorization, membership inference
  limits, DP-SGD cost, PII in fine-tuning data, unlearning methods and why
  evaluations overclaim (relearning, probing residual knowledge).
- **Boundary:** Module 08 owns contamination; Module 18 owns runtime attacks.
- **Break lab (COLAB-GPU):** fine-tune with planted canaries; measure exposure;
  apply an unlearning method; recover canary with few-shot relearning.
- **Leads:** Carlini et al. 2020 (USENIX Security 2021) extraction (arXiv:2012.07805); Carlini et al.
  2022 quantifying memorization (arXiv:2202.07646); Nasr et al. 2023 scalable
  extraction (arXiv:2311.17035); Abadi et al. 2016 DP-SGD (arXiv:1607.00133);
  Maini et al. 2024 TOFU (arXiv:2401.06121).

---

## Phase 12 — Deployment Frontiers

### 35 — Multi-Tenant Adapter & Long-Context Serving
- **Owns:** batching heterogeneous LoRA adapters (segmented gather MM, adapter
  paging, cold-start), context extension (RoPE scaling, YaRN), ring/context
  parallel training for long context, effective-context evaluation.
- **Boundary:** Module 11 owns context budgeting at app level; Module 20 owns
  CP at inference.
- **Leads:** Sheng et al. 2023 S-LoRA (arXiv:2311.03285); Chen et al. 2023
  Punica (arXiv:2310.18547); Peng et al. 2023 YaRN (arXiv:2309.00071); Liu et al.
  2023 Ring Attention (arXiv:2310.01889); Hsieh et al. 2024 RULER
  (arXiv:2404.06654).

### 36 — Edge Inference & Computer-Use Agents
- **Owns:** on-device constraints (flash-to-DRAM streaming, NPU op coverage,
  GGUF/k-quants, energy per token), and environment-grounded agent evaluation
  (GUI/web/terminal environments, reset determinism, reward leakage in
  benchmarks).
- **Leads:** Alizadeh et al. 2023 LLM in a Flash (arXiv:2312.11514); Song et al.
  2023 PowerInfer (arXiv:2312.12456); ggml-org/llama.cpp repo; Xie et al. 2024
  OSWorld (arXiv:2404.07972); Zhou et al. 2023 WebArena (arXiv:2307.13854);
  Jimenez et al. 2023 SWE-bench (arXiv:2310.06770); Yao et al. 2024 τ-bench
  (arXiv:2406.12045).

---

## Expansion protocol (not fixed at 36)

Track is open-ended. Add a module when a Paper Radar entry
([paper-library/](../paper-library/)) passes all three:

1. **Ownership:** no existing module owns the mechanism (keyword sweep +
   boundary sections).
2. **Durability:** mechanism appears in ≥2 independent Tier 1/2 sources or one
   widely deployed official runtime.
3. **Breakability:** a lab can falsify a claim on `CPU`, `COLAB-GPU`, or
   `TRACE`.

Otherwise extend the nearest module's lesson or park it as `FRONTIER` in Paper
Radar. Retire a module when its mechanism becomes `LEGACY`.

## Suggested order

25 → 26 → 27 → 28 (training) can run after Module 19. 29 → 30 after Module 05.
31, 32 after Module 24. 33, 34 after Module 18. 35, 36 after Modules 20 and 12.
