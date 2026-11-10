# Module 19 — Model Adaptation

## 00 Why This Module Exists

Fine-tuning changes parameters shared by every future request, so its effect can reach inputs the target set never covered. Prompting, retrieval, and routing change only what a single request sees or which model answers it. That asymmetry decides most adaptation questions. A persistent format, procedure, or skill gap is a candidate for a weight update. A missing or changing fact is usually a retrieval problem. Any weight update can move behavior the target metric never looks at, including general capability and safety, and how far it moves is measured, not assumed.

```text
failure analysis --> failure class --> alternative? (prompt / RAG / route) --no--> adapt
                                                                                 |
     data snapshot (lineage, dedup, contamination, provenance, template, loss mask) <-+
                   |
                   v
     method (full / LoRA / QLoRA; SFT / DPO / GRPO) + memory plan + run manifest
                   |
                   v
     gate: target gain AND retained capability AND safety
           (paired vs base, declared margins, independently trained runs)
                   |
                   v
     Module 17 rollout: canary --> promote / rollback to previous adapter
```

Module 01 introduced parameter accounting (Lesson 1.5), Module 05 quantization contracts (Lesson 5.3), Module 08 data lineage and contamination (Lessons 8.1, 8.4), Module 15 paired evaluation and release gates (Lessons 15.2, 15.4), Module 17 harness manifests and staged rollout (Lessons 17.1, 17.5), and Module 18 source–sink threat modeling (Lesson 18.1). This module owns the adaptation decision, training memory, parameter-efficient mechanics, adaptation data, preference optimization, forgetting, and adaptation-specific safety gates. It also owns training-time data and checkpoint poisoning in adaptation pipelines. Test-time reasoning stays in Module 06 and full cost modeling in Module 22. Distributed *training* (ZeRO stages, FSDP, sharded optimizer states) is out of scope for this curriculum's current modules: Module 20 covers distributed inference only. This module computes the single-device memory that motivates sharding but does not teach sharding.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Decide whether a failure should be fixed by changing weights. If it should, change them with a known memory budget and data whose provenance you control, then produce evidence of what regressed and a statement of what the evidence cannot rule out.
- **What You Will Do**: Write an adaptation decision record with a falsifiable comparison, compute and measure training memory and LoRA parameter counts, implement and trace LoRA at a pinned PEFT commit, build SFT and preference data with correct templates, loss masks, and provenance filters, run DPO/GRPO-style updates on a small model, and gate an adapter on target, retained capability, and safety.
- **Environment**: Python 3.10+, PyTorch, Transformers, PEFT (and optionally TRL), one GPU able to fine-tune a ≤3B open-weight model with LoRA, the Module 15 evaluation harness, and a fixed safety prompt suite. No production data without the Module 08 lineage controls.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. All numbers in worked examples are declared synthetic or derived; author-reported results are labeled as such.
- **If You Only Have One Week**: Lessons 19.1, 19.4, 19.6 — they decide whether to adapt at all, build data that survives leakage and provenance checks, and gate release on gain, retained capability, and safety; memory, LoRA mechanics, and preference optimization build on those three.

## 01 Baseline Assumptions

- Module 01: parameter accounting and decoder tensor shapes (Lesson 1.5).
- Module 02: memory hierarchy, honest timing, and numeric precision formats. Mixed-precision *training* state layouts and allocator accounting are taught here in Lesson 19.2, not assumed.
- Module 05: quantization formats and their accuracy contracts (Lesson 5.3).
- Module 08: data lineage, deduplication, and contamination (Lessons 8.1, 8.4).
- Module 15: slices, paired evaluation, confidence intervals, and release gates (Lessons 15.2, 15.4).
- Module 17: harness manifests and canary rollout and rollback (Lessons 17.1, 17.5).
- Module 18: source–sink threat modeling (Lesson 18.1). The harmful fine-tuning and poisoning risk classes are introduced here, in Lesson 19.4.

## 02 Target Mastery

```yaml
depth_contract:
  conceptual: REQUIRED
  mechanistic: REQUIRED
  mathematical: REQUIRED
  quantitative: REQUIRED
  implementation: REQUIRED
  source_code: REQUIRED
  instrumentation: REQUIRED
  experimental: REQUIRED
  statistical: REQUIRED
  production_reasoning: REQUIRED
  failure_analysis: REQUIRED
  falsification: REQUIRED
  security: SELECTIVE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: REQUIRED

estimated_effort:
  instruction: 4.75h   # sum of lesson instruction estimates
  guided_practice: 2h  # sum of lesson practice estimates
  labs: 13h            # LAB A 3h + LAB B 3.5h + LAB C 3.5h + LAB D 3h
  assessment: 3h
  source_trace: 2h     # counted once; shared by Lesson 19.3 and LAB B
  total: 24.75h
```

The learner must justify adaptation against cheaper alternatives with a comparison that could come out the other way. They must predict memory before training and reconcile it with measurements, implement and trace LoRA, and construct data whose measured gains are not leakage and whose sources are controlled. They must explain the objectives of RLHF, DPO, and GRPO and their failure modes, and ship an adapter only with paired, margin-based evidence on target gain, retained capability, and safety, while stating what that evidence cannot certify.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Failure Analysis $\to$ Failure Class $\to$ Adaptation Decision (prompt / retrieve / route / SFT / preference). If adapting: Training Memory (model states + activations + workspace) $\to$ Method (full / LoRA / QLoRA; rank, alpha, targets) $\to$ Data (lineage, dedup, contamination, provenance, template, loss mask) $\to$ Objective (SFT; RLHF; DPO; GRPO and variants) $\to$ Gate (target, retained capability, safety; paired, margins, independent runs) $\to$ Rollout (Module 17). Cross-cutting risks: forgetting, reward/length pathologies, train/serve mismatch, silent configuration defaults, safety regression, and training-time poisoning.

## 04 Lessons

### Lesson 19.1 — The Adaptation Decision: Adapt, Retrieve, Prompt, or Route

**Engineering Question:**
For this failure class, is changing weights the cheapest intervention that fixes it durably?

**Concepts & Definitions:**
- **Failure class**: a cluster of errors with a shared cause found by slicing and inspecting failures (Module 15, Lesson 15.2): missing knowledge, stale knowledge, format/procedure, capability, or policy.
- **Weight adaptation** changes parameters that every future request uses, so its effect can extend beyond the targeted inputs. **Prompting, retrieval, and routing** change the input or the model per request (**D**, CLM-001).
- **Auditable support**: a fine-tuned model can emit citation-like text, but weights alone do not bind an answer to an immutable source record. Retrieval can bind each answer to retrieved records whose support is then checked (Module 10) (**D**, CLM-001).
- **Knowledge injection evidence**: Ovadia et al. report that RAG consistently outperformed *unsupervised* fine-tuning for both previously seen and new knowledge on their tasks. They also report that models struggled to learn new facts through fine-tuning unless shown many variations of each fact (**O**, CLM-014; evaluated tasks only).

**Mechanism Explanation:**
Knowledge in weights is refreshed only by retraining and re-gating. A retrieval index is refreshed by an index write. Format and procedure, by contrast, are persistent behaviors: a prompt re-teaches them on every request at a token cost, while an adapter learns them once. Hybrid designs (an adapter for behavior plus retrieval for facts) are common, and the decision is made per failure class, not per product.

**Quantitative Model / Trade-off Comparison:**
Per failure class, compare over a planning horizon with $N$ requests and $U$ refreshes:
$C_{prompt} = N\,\Delta t_{prompt}\,c_{tok}$; $C_{RAG} = N\,(\Delta t_{ctx}\,c_{tok} + c_{retr}) + U\,c_{index}$; $C_{adapt} = U\,(c_{train} + c_{eval}) + N\,\Delta c_{serve}$.

where:
- $N$: requests in the horizon; $U$: refreshes (fact updates, or adapter retrains) in the horizon.
- $\Delta t_{prompt}$: extra prompt tokens added to every request; $\Delta t_{ctx}$: extra retrieved-context tokens per request; $c_{tok}$: price per input token.
- $c_{retr}$: retrieval cost per request; $c_{index}$: cost of one index update.
- $c_{train}+c_{eval}$: cost of one retrain plus re-gate; $\Delta c_{serve}$: extra serving cost per request of the adapted model (≈0 when merged).
- Reading the terms: per-request costs are multiplied by $N$, per-refresh costs by $U$. Prompting pays only per request, adaptation mostly per refresh, and RAG pays both.

Cost is necessary but not sufficient: each option must also meet the quality target within a declared margin $\delta$. Hypothesis (**H**, CLM-013):
- **Freshness**: on withheld, unpredictable fact updates made after training, and with the same evidence allowed to both, RAG accuracy ≥ retrained-SFT accuracy − $\delta$ at lower update cost.
- **Format**: SFT format validity ≥ prompt validity − $\delta$, with $C_{adapt} < C_{prompt}$ over the horizon.

Falsified if the paired accuracy difference (RAG − retrained SFT) has an upper confidence bound below $-\delta$, if index update cost exceeds retrain-plus-re-gate cost, if SFT validity's upper bound falls below prompt validity − $\delta$, or if $C_{adapt} > C_{prompt}$.

**Worked Example (synthetic inputs):**
A support bot fails in two ways: 35% of errors cite last quarter's return policy (stale facts), and 40% emit invalid ticket JSON (format).
1. *Format.* A 1,200-token instruction block fixes most JSON errors. It costs $1{,}200 \times 2\times10^6 = 2.4\times10^9$ input tokens per month. At an assumed $c_{tok}$ of \$0.50 per $10^6$ input tokens, that is \$1,200/month. A monthly adapter refresh with assumed $c_{train}+c_{eval}$ of \$400 and $\Delta c_{serve}$ ≈ 0 (merged adapter) costs \$400/month. SFT is the cheaper candidate *if* its validity is within $\delta = 1$ point of the prompt. That condition is tested in LAB A, not assumed.
   - *Mapping to the formulas* (horizon = one month): $N=2\times10^6$ requests, $\Delta t_{prompt}=1{,}200$, $c_{tok}=0.50/10^6$ dollars, so $C_{prompt}=2\times10^6\times1{,}200\times0.5\times10^{-6}=1{,}200$ dollars. With $U=1$ refresh and $\Delta c_{serve}\approx0$: $C_{adapt}=1\times400+N\times0=400$ dollars.
2. *Freshness.* Policies change about quarterly. Retraining for each change costs a full re-gate, while re-indexing dated policy documents costs minutes, so stale-policy failures are routed to retrieval.
3. *Decision record.* Retrieval for stale facts. SFT adapter trial for JSON with the prompt kept as fallback. Both predictions stay falsifiable.

**Knowledge Check:**
1. What evidence would show that a fine-tuned model's citation actually supports its answer?
2. Why must the updated facts in the freshness test be unpredictable from prior knowledge?

**Guided Practice:**
Take 50 labeled failures from a Module 15 slice analysis. Cluster them into failure classes, and write one decision record per class: the alternative tried, cost terms, margin $\delta$, the predicted winner, and the falsifying observation.

**Feedback Contract:**
- *Expected Evidence*: Failure classes with counts, the alternative per class, cost terms with stated unit prices, $\delta$, and a falsifier that could fire.
- *Common Failure*: Choosing fine-tuning because the product "needs domain knowledge."
- *Diagnostic Hint*: Will this fact change before the next training run, and who pays for re-gating when it does?
- *Concept to Revisit*: Slices and Paired Comparison (Module 15); RAG Freshness (Module 10).

**Learning Outcome:**
Justify or reject weight adaptation per failure class with a falsifiable comparative hypothesis.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 19.2 — Training Memory: Model States, Activations, and Quantized Bases

**Engineering Question:**
Will this run fit, and which term dominates?

**Concepts & Definitions:**
- **Model states**: parameters, gradients, and optimizer states. ZeRO states that mixed-precision Adam needs $2\Psi$ bytes for fp16 parameters, $2\Psi$ for fp16 gradients, and $K\Psi=12\Psi$ for the fp32 parameter copy, momentum, and variance, which totals $16\Psi$ bytes before activations (**O**, CLM-002). This is one layout: pure-fp32, bf16-without-master-copy, 8-bit, or factored optimizers change the bytes per parameter.
- **Activations**: saved forward tensors, which scale with batch, sequence length, depth, and width. Activation checkpointing reduces them at the cost of recomputation.
- **Workspace and allocator**: temporary buffers (e.g., matmul workspace) and PyTorch's caching allocator. `torch.cuda.memory_allocated()` counts live tensors, `max_memory_allocated()` the peak of that count, and `memory_reserved()` what the allocator holds from the device, including cached free blocks. Peak reserved can exceed the sum of live tensors.
- **Parameter-efficient fine-tuning**: LoRA trains only low-rank adapters on a frozen base. Its authors report up to 10,000× fewer trainable parameters and 3× lower GPU memory than Adam fine-tuning of GPT-3 175B (**O**, CLM-003).
- **QLoRA**: backpropagates through a frozen 4-bit NF4 base into LoRA adapters, using double quantization and paged optimizers. Its authors report fine-tuning a 65B model on one 48 GB GPU while preserving 16-bit fine-tuning task performance on evaluated tasks (**O**, CLM-004).

**Mechanism Explanation:**
Under the ZeRO mixed-precision Adam layout, full fine-tuning pays $16\Psi$ for model states because every parameter has a gradient, an fp32 master copy, and two Adam moments. LoRA keeps the base frozen: the base needs only resident weights, and the full per-parameter state cost is paid only for adapter parameters. QLoRA further shrinks the frozen base by storing it in 4 bits and dequantizing on the fly. None of these methods removes activation memory, which often becomes the dominant term once model states shrink.

**Quantitative Model / Derivation (D):**
- Full mixed-precision Adam (ZeRO layout): $M_{states} = 16\Psi$ bytes.
- LoRA with a 16-bit frozen base: $M \approx 2\Psi + 16\,\Psi_{LoRA} + M_{act} + M_{ws}$. The 16 bytes per adapter parameter assume fp32 adapter weights, fp32 gradients, and two fp32 Adam moments.
- QLoRA: $M \approx 0.5\Psi + M_{quant\ consts} + 16\,\Psi_{LoRA} + M_{act} + M_{ws}$, plus any layers kept in higher precision.

where: $\Psi$ = number of base-model parameters; $\Psi_{LoRA}$ = number of trainable adapter parameters; $K$ = optimizer-state bytes per parameter; $M_{act}$ = activation bytes; $M_{ws}$ = workspace bytes; $M_{quant\ consts}$ = bytes of quantization constants (scales) for the 4-bit base. All $M$ terms are in bytes.

*Step-by-step (bytes per parameter):*
- Full fine-tuning, ZeRO layout: fp16 weight (2) + fp16 gradient (2) + fp32 master copy (4) + fp32 momentum (4) + fp32 variance (4) $= 2+2+12 = 16$, so $K = 4+4+4 = 12$.
- LoRA, frozen base: 16-bit weight only (2), no gradient and no optimizer state, so $2\Psi$. Adapter, all fp32: weight (4) + gradient (4) + two Adam moments (4+4) $= 16$. The two "16"s coincide but have different breakdowns.
- QLoRA base: 4 bits $= 4/8 = 0.5$ bytes per parameter, so $0.5\Psi$ before quantization constants.

**Measurement procedure.** These formulas are predictions:
1. After the first optimizer step (Adam allocates its moments lazily), sum `numel × element_size` over parameters, gradients, and `optimizer.state` tensors, grouped by dtype. This gives *state tensor bytes*.
2. Record `max_memory_allocated()` for one step and `memory_reserved()`.
3. Attribute the gap between peak allocated and state tensor bytes to activations plus workspace. Confirm this by toggling checkpointing and sequence length.
4. Report the reserved-minus-allocated gap as allocator caching.

The 10% tolerance used in LAB B is an exercise threshold, not an accuracy guarantee.

**Worked Example (derived; synthetic 8×10⁹-parameter decoder):**
- **Full fine-tuning.** $\Psi = 8\times10^9$, so model states are $16 \times 8\times10^9 = 128\times10^9$ bytes (128 GB, ≈119.2 GiB) before activations, more than one 80 GB GPU. (GB $=10^9$ bytes, GiB $=2^{30}$ bytes: $128\times10^9/2^{30}\approx119.2$.)
- **LoRA.** With $\Psi_{LoRA}= 41{,}943{,}040$ (computed in Lesson 19.3): $2\times8\times10^9 = 16\times10^9$ bytes of frozen base plus $16\times41{,}943{,}040 = 0.671\times10^9$ bytes of adapter states, ≈16.7 GB plus activations and workspace.
  Adapter states: $16\times41{,}943{,}040 = 671{,}088{,}640$ bytes $\approx0.671\times10^9$; total $16+0.671\approx16.7$ GB.
- **QLoRA.** ≈4 GB base + constants + 0.67 GB + activations. ($0.5\times8\times10^9 = 4\times10^9$ bytes.)
- **Activations.** At sequence length 4,096 and micro-batch 8, activations can exceed all of these. Measure them with and without checkpointing using the procedure above.

**Knowledge Check:**
1. Why does LoRA not reduce activation memory in proportion to trainable parameters?
2. What happens to $K$ if the optimizer is SGD with momentum and no fp32 master copy?

**Guided Practice:**
For a 1B model, predict state bytes and peak memory for full, LoRA, and QLoRA at two sequence lengths. Then run the four-step measurement procedure for one training step of each, and explain every gap above the 10% exercise tolerance.

**Feedback Contract:**
- *Expected Evidence*: Per-dtype state inventory taken after the first step, peak allocated, reserved, checkpointing setting, and an attributed residual.
- *Common Failure*: Measuring before the first optimizer step and missing Adam moments, or comparing a prediction to reserved memory as if it were live tensors.
- *Diagnostic Hint*: Which tensors have gradients, which have Adam moments, and when are those moments created?
- *Concept to Revisit*: Precision Formats and Honest Timing (Module 02); Quantization Contracts (Module 05).

**Learning Outcome:**
Predict and verify training memory for full, LoRA, and QLoRA runs, separating state bytes from activations, workspace, and allocator caching.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 19.3 — LoRA Mechanics: Rank, Scaling, Targets, and Merging

**Engineering Question:**
What exactly does LoRA add to a layer, and what do rank, alpha, and target modules control?

**Concepts & Definitions:**
- **Low-rank update**: for frozen $W\in\mathbb{R}^{d_{out}\times d_{in}}$, learn $B\in\mathbb{R}^{d_{out}\times r}$ and $A\in\mathbb{R}^{r\times d_{in}}$, giving $y = Wx + s\,B(Ax)$ with $s=\alpha/r$. LoRA freezes the pretrained weights and trains only these matrices (**O**, CLM-003).
  - where: $x\in\mathbb{R}^{d_{in}}$ is the layer input, $y\in\mathbb{R}^{d_{out}}$ the output; $r$ is the LoRA rank ($r \ll \min(d_{in},d_{out})$); $\alpha$ is the LoRA scaling hyperparameter (`lora_alpha`), unrelated to the significance level $\alpha$ used in statistics; $s$ is the resulting scalar multiplier (`scaling`).
  - Shapes: $Ax\in\mathbb{R}^{r}$ (compress to $r$ numbers), then $B(Ax)\in\mathbb{R}^{d_{out}}$ (expand back). The product $BA$ is a $d_{out}\times d_{in}$ matrix of rank at most $r$, so $y = (W + sBA)\,x$: the layer behaves as if its weight were $W+sBA$.
- **rsLoRA**: Kalajdzievski argues that dividing by $r$ slows learning at higher ranks and proves adapters should be divided by $\sqrt r$, i.e. $s=\alpha/\sqrt r$ (**O**, CLM-021).
- **Zero initialization**: $B=0$ at the start, so the adapted model initially equals the base. (With $B=0$: $sBA = 0$, so $y = Wx$ exactly, whatever $A$ is.)
- **Merge**: $W' = W + sBA$ removes adapter compute at serving time. Unmerged adapters allow many adapters over one base.
- **Target modules**: which projections receive adapters. At the pinned PEFT commit, the default mapping for llama, mistral, qwen2, and qwen3 is only `q_proj` and `v_proj` (**O**, CLM-022). Relying on defaults therefore gives an attention-only configuration.
- **Capacity**:
  - Biderman et al. report that standard low-rank LoRA substantially underperformed full fine-tuning on code and math target domains yet better preserved out-of-domain performance, and that full fine-tuning learned perturbations of 10–100× higher rank (**O**, CLM-006).
  - LoRA Without Regret reports that LoRA matched full fine-tuning in log loss on small-to-medium SFT datasets and matched it in RL, and that attention-only LoRA significantly underperformed MLP-only LoRA. It found the best LoRA learning rate was about 10× the full fine-tuning rate for longer runs (≈15× in short runs), and that LoRA can be less tolerant of large batch sizes (**O**, CLM-015; lab post, not peer reviewed).

**Mechanism Explanation — Pinned PEFT Trace (O, CLM-005, CLM-022):**
At `huggingface/peft` commit `b8674c86183a5dee38d0c3ede392e189593025e5` (static inspection, 2026-09-30):
1. `src/peft/tuners/lora/layer.py::LoraLayer.update_layer` creates `lora_A = nn.Linear(in_features, r)` and `lora_B = nn.Linear(r, out_features)` and sets `scaling = lora_alpha / r`, or `lora_alpha / math.sqrt(r)` when `use_rslora`.
2. `LoraLayer.reset_lora_parameters` initializes `lora_A` with Kaiming-uniform (default) or Gaussian and zeroes `lora_B`. Other `init_lora_weights` values (PiSSA, OLoRA, LoftQ, and others) take different paths.
3. `Linear.forward` computes `base_layer(x)` and, for each active unmerged vanilla adapter, adds `lora_B(lora_A(dropout(x))) * scaling`. Variants such as DoRA dispatch to their own forward.
4. `Linear.get_delta_weight` returns `transpose(B @ A, fan_in_fan_out) * scaling`. `Linear.merge` adds it into the base weight (with an optional NaN-checked `safe_merge`), and `unmerge` subtracts it.
5. `src/peft/utils/constants.py::TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` maps `llama`, `mistral`, `qwen2`, and `qwen3` to `["q_proj", "v_proj"]`.

Lesson: rank, alpha, and targets are separate knobs, and all three belong in the run manifest. The scalar $s$ is fixed by alpha and rank, but the norm of the learned update $sBA$ also depends on learning rate, data, and training length, so changing $r$ without revisiting $\alpha$ and the learning rate changes the effective update in ways that must be measured.

**Quantitative Model / Derivation (D):**
- Trainable parameters per adapted matrix: $r(d_{in}+d_{out})$.
- Extra forward FLOPs per token per unmerged matrix: $\approx 2r(d_{in}+d_{out})$, versus $2d_{in}d_{out}$ for the base matrix.

*Step-by-step:*
- Parameters: $A$ has $r\cdot d_{in}$ entries and $B$ has $d_{out}\cdot r$, total $r(d_{in}+d_{out})$, versus $d_{in}d_{out}$ in $W$.
- FLOPs (unmerged): computing $Ax$ costs $\approx 2rd_{in}$ (one multiply and one add per entry of $A$), then $B(Ax)$ costs $\approx 2rd_{out}$; sum $2r(d_{in}+d_{out})$. The base $Wx$ costs $2d_{in}d_{out}$. Ratio $= r(d_{in}+d_{out})/(d_{in}d_{out})$, the same as the parameter ratio. After merging, the extra cost is 0.
- Tiny example: $d_{in}=d_{out}=4096$, $r=16$: adapter $16\times8192=131{,}072$ parameters vs $4096^2=16{,}777{,}216$ base, ratio $\approx0.78\%$.

- The FLOP ratio below counts only the named linear projections. It excludes attention score/softmax FLOPs, norms, embeddings, and the LM head, so it overstates the fraction of total model FLOPs.

**Worked Example (derived; synthetic decoder with 32 layers, $d=4096$, GQA $k/v$ width 1024, MLP width 14336, $r=16$ on all seven linear projections):**
- **Trainable parameters per layer.** $q, o$: $16(4096+4096)=131{,}072$ each. $k, v$: $16(4096+1024)=81{,}920$ each. gate, up, down: $16(4096+14336)=294{,}912$ each. Sum $=1{,}310{,}720$ per layer, and × 32 layers = **41,943,040** trainable parameters.
- **FLOP ratio.** The named base projections hold $218{,}103{,}808$ parameters per layer, so unmerged adapter FLOPs are ≈0.601% of those projections' FLOPs. Base count: $q,o$: $4096\times4096=16{,}777{,}216$ each; $k,v$: $4096\times1024=4{,}194{,}304$ each; gate, up, down: $4096\times14336=58{,}720{,}256$ each; sum $2(16{,}777{,}216)+2(4{,}194{,}304)+3(58{,}720{,}256)=218{,}103{,}808$. Ratio $=1{,}310{,}720/218{,}103{,}808\approx0.00601$ (the factor 2 in both FLOP counts cancels).
- **Measured slowdown.** A measured unmerged slowdown larger than 0.6% has competing explanations (**H**): extra kernel launches, poor efficiency of small rank-16 matmuls, dtype casts (`_cast_input_dtype`), extra memory traffic, and synchronization. Discriminate them with a matched profiler timeline (Module 02, Lesson 2.6), not by assumption.
- **Scaling.** $r=16,\alpha=32$ gives $s=2$. Moving to $r=64$ with $\alpha=32$ gives $s=0.5$, while rsLoRA gives $32/\sqrt{64}=4$.
  - Step-by-step: $s=\alpha/r$: $32/16=2$, $32/64=0.5$ (rank ×4 with fixed $\alpha$ cuts $s$ by 4). rsLoRA $s=\alpha/\sqrt r$: $32/\sqrt{64}=32/8=4$. To keep the standard $s=2$ at $r=64$, $\alpha$ would need to be $2\times64=128$.
- **Default targets.** With default `target_modules` on a llama-type model, only $q$ and $v$ are adapted: $131{,}072+81{,}920 = 212{,}992$ per layer, or 6,815,744 total.

**Knowledge Check:**
1. Why does zero-initializing $B$ make the first training step start exactly from the base model?
2. Why can the Biderman and LoRA Without Regret results both be true?
3. What does a `LoraConfig` without `target_modules` adapt on a qwen2 model at the pinned commit?

**Guided Practice:**
Implement a from-scratch LoRA wrapper for `nn.Linear` and check numerical equivalence with PEFT on the same seeds. Then verify that merged and unmerged outputs agree within a stated tolerance in fp32 and bf16. Print the resolved target modules of a default `LoraConfig`.

**Feedback Contract:**
- *Expected Evidence*: Parameter counts, equivalence test with tolerance, merge parity per dtype, resolved target list, and trace citations by symbol.
- *Common Failure*: Doubling rank and reporting "no improvement" while the scale halved and the learning rate was unchanged, or comparing "LoRA vs full" with attention-only defaults.
- *Diagnostic Hint*: What are `scaling` and the resolved `target_modules` before and after your change?
- *Concept to Revisit*: Parameter Accounting (Module 01, Lesson 1.5).

**Learning Outcome:**
Implement, count, trace, configure, and merge LoRA, and explain its capacity trade-off.

*(Effort: 50m instruction, 20m practice; source trace 2h)*

---

### Lesson 19.4 — Adaptation Data: Failure Mining, Templates, Loss Masks, and Poisoning

**Engineering Question:**
Will the measured gain come from generalization, or from leakage, formatting accidents, or poisoned examples?

**Concepts & Definitions:**
- **Failure mining**: turning production failures into demonstrations or preference pairs, with labels from experts or verified corrections.
- **Train/evaluation separation**: adaptation data mined from logs must be separated from evaluation data by source and time and filtered for near-duplicates (**D**, CLM-007; Module 08, Lesson 8.4).
- **Chat template fidelity**: the role markers, special tokens, and system prompt used in training must match serving.
- **Loss masking and loss share**: a mask selects the positions that carry loss. The response's share of the loss is $\sum_{resp}\ell_t / \sum_{incl}\ell_t$, not its token share. The two are equal only when prompt and response tokens have equal mean loss (**D**, CLM-028).
- **Harmful fine-tuning**: Qi et al. report that fine-tuning GPT-3.5 Turbo on 10 adversarial examples for under \$0.20 made it responsive to nearly any harmful instruction (**O**, CLM-011).
- **Backdoor (trigger) poisoning**:
  - Wan et al. show that poison examples contributed to instruction-tuning data can make models misbehave whenever a trigger phrase appears. As few as 100 poison examples had broad effects on the models they evaluated, and filtering gave only moderate protection (**O**, CLM-023).
  - Souly et al. report that a near-constant number of poisoned documents (250 in their pretraining experiments) compromised 600M–13B models regardless of data size, and report similar dynamics in fine-tuning (**O**, CLM-024; FRONTIER).
  - Hubinger et al. report that constructed backdoors could persist through SFT, RL, and adversarial training (**O**, CLM-025).
  - Betley et al. report that narrow fine-tuning on insecure code produced broad misalignment, and that a trigger-conditioned variant stayed hidden without the trigger (**O**, CLM-026; FRONTIER).

**Mechanism Explanation:**
- **Leakage.** If near-duplicates of evaluation items enter training, the measured gain reflects memorization.
- **Template mismatch.** If the training template differs from serving, the model learns a conversation format it will never see at serving time.
- **Loss mask.** If prompt tokens carry loss, the objective includes predicting prompts. Masking changes the objective, not just a weight on it.
- **Untrusted sources.** If any data source or checkpoint can be written by untrusted parties, the adapter becomes a sink for Module 18's source–sink analysis.

Behavioral suites that never contain the trigger cannot distinguish a backdoored model from a clean one. The primary control is therefore provenance and write-access control over data and checkpoints, with trigger-search red teaming as bounded secondary evidence (**D**, CLM-027).

**Quantitative Model / Derivation (D):**
- Response loss share under token-mean cross-entropy: $\dfrac{T_r m_r}{T_p m_p + T_r m_r}$, where $T$ is the token count and $m$ the mean per-token loss. The token share is $T_r/(T_p+T_r)$.
  - where: subscripts $p$ = prompt, $r$ = response; $\ell_t$ = cross-entropy loss at token position $t$.
  - Step-by-step: the summed response loss is $\sum_{resp}\ell_t = T_r m_r$ (count × mean), and likewise $T_p m_p$ for the prompt. Dividing response loss by total loss gives the share. Setting $m_p=m_r=m$ cancels $m$ and recovers the token share $T_r/(T_p+T_r)$, which is why the two coincide only under equal mean loss. The overall $1/(T_p+T_r)$ of a token mean cancels in the ratio.
- Contamination rate: the fraction of evaluation items with an $n$-gram or embedding near-duplicate in training above a fixed threshold, reported together with the threshold.

**Worked Example (synthetic):**
1. *Loss share.* Examples average $T_p=900$ prompt tokens with mean loss $m_p=1.0$ and $T_r=100$ response tokens with $m_r=9.0$. The token share of the response is $100/1000 = 10\%$. Its loss share is $900/(900+900) = 50\%$ (prompt loss $T_pm_p=900\times1.0=900$, response loss $T_rm_r=100\times9.0=900$), so "only 10% of the loss is on responses" would be wrong. Masking the prompt makes the response 100% of the loss. Whether that improves held-out JSON validity is a hypothesis for LAB C.
2. *Leakage.* A near-duplicate scan (5-gram Jaccard ≥ 0.8) finds 6% of the evaluation set in the mined data. Those items are removed from training, and the pre-fix gain is reported as leaked.
3. *Provenance.* 4% of mined examples come from a feedback form open to external agents. They are labeled `untrusted` and excluded from training until reviewed. The run manifest records the counts.

**Knowledge Check:**
1. Why can a template mismatch hurt only at serving time?
2. Why can the three-part release gate pass a backdoored adapter?

**Guided Practice:**
Build an SFT set from 500 mined failures:
- lineage IDs and source trust labels;
- a dedup and contamination report with the threshold;
- template rendering through the serving tokenizer;
- loss-mask verification by printing masked tokens and computing the loss share on one batch;
- exclusion of untrusted sources.

**Feedback Contract:**
- *Expected Evidence*: Data manifest, contamination rate with threshold, rendered-template diff against serving, mask visualization, measured loss share, and source trust counts.
- *Common Failure*: Evaluating on a random split of the same mined logs, or treating the pass of a fixed safety suite as proof that no poisoned data entered.
- *Diagnostic Hint*: Could any evaluation item's answer be copied from a training item, and who could write to each data source?
- *Concept to Revisit*: Contamination and Lineage (Module 08); Source–Sink Models (Module 18).

**Learning Outcome:**
Construct adaptation data whose measured gains survive leakage, template, and loss-mask checks, and whose sources are provenance-controlled.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 19.5 — Preference Optimization: RLHF, DPO, and GRPO

**Engineering Question:**
How do preference and reward objectives move the policy, and what keeps them from moving it too far or in the wrong direction?

**Concepts & Definitions:**
- **RLHF**: InstructGPT used supervised demonstrations, a reward model trained on human rankings, and RL. Labelers preferred its 1.3B model's outputs to 175B GPT-3's (**O**, CLM-008).
- **DPO**: a classification loss on preference pairs that implicitly defines a reward relative to a reference policy, with $\beta$ controlling deviation (**O**, CLM-009).
- **GRPO**: drops PPO's value model, estimates a baseline from $G$ sampled outputs per prompt, normalizes rewards by the group mean and standard deviation, and adds a KL estimator to the loss (**O**, CLM-010).
- **GRPO variants**:
  - Liu et al. show that response-length normalization favors brevity among correct and length among incorrect answers, and that standard-deviation normalization up-weights low-variance questions. Dr. GRPO removes both (**O**, CLM-017).
  - DAPO's Dynamic Sampling filters out prompts whose samples are all correct or all incorrect (**O**, CLM-018).
  - Both Dr. GRPO and DAPO **drop the KL term**: Dr. GRPO assumes $\beta=0$, and DAPO excludes KL for long-CoT reasoning training (**O**, CLM-029).
- **Forgetting and RL**: RL's Razor reports that on-policy RL forgot less than SFT at similar new-task performance, with forgetting tracking KL to the base on the new task (**O**, CLM-016; FRONTIER, one study).

**Mechanism Explanation:**
Original RLHF and GRPO regularize toward a reference with a KL term. DPO does so through the reference log-probabilities inside its loss. KL-free recipes limit drift only through clipping, data, and training length, so KL-to-base must be *monitored* even when it is not *optimized*. The reward signal is the weakest link: a learned reward model can be exploited, offline preference pairs cannot explore, and programmatic rewards reward only what they check. Normalization details are not cosmetic. They decide which samples and questions dominate the update.

**Quantitative Model / Derivation:**
- **DPO**: $\mathcal{L} = -\log\sigma\big(\beta[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x)) - (\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))]\big)$.
- **GRPO objective** (outcome rewards, as written by DeepSeekMath and restated in Dr. GRPO):
$$\mathcal{J} = \mathbb{E}\Big[\tfrac{1}{G}\sum_{i=1}^{G}\tfrac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)\Big] - \beta\,\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}],\quad \rho_{i,t}=\tfrac{\pi_\theta(o_{i,t}|\cdot)}{\pi_{\theta_{old}}(o_{i,t}|\cdot)}$$
- **Advantage**: $\hat A_i = (r_i-\bar r)/(\mathrm{std}(r)+\varepsilon)$. Without $\varepsilon$ or an explicit zero-variance branch, equal rewards give $0/0$.
- **What the variants remove**: Dr. GRPO deletes $1/|o_i|$ and the std division. DAPO replaces the per-sequence mean with a token-level sum normalized over all tokens in the group. Both set $\beta=0$.

where:
- $x$: prompt; $y_w$, $y_l$: preferred ("winner") and rejected ("loser") responses; $\pi_\theta$: policy being trained; $\pi_{ref}$: frozen reference policy (usually the SFT model); $\sigma(z)=1/(1+e^{-z})$: logistic sigmoid.
- DPO $\beta$: strength of the implicit pull toward $\pi_{ref}$. GRPO $\beta$: weight of the explicit KL penalty. Same letter, different roles; both mean "how strongly to stay near the reference."
- $G$: samples (outputs) per prompt; $o_i$: the $i$-th sampled output, $|o_i|$ its length in tokens, $o_{i,t}$ its $t$-th token; $\pi_{\theta_{old}}$: the policy that generated the samples; $\rho_{i,t}$: per-token probability ratio (importance weight).
- $r_i$: scalar reward of output $i$; $\bar r$: group mean of $r_1..r_G$; $\mathrm{std}(r)$: group standard deviation; $\hat A_i$: advantage, shared by every token of $o_i$.
- $\epsilon$ (clip range, e.g. 0.2) and $\varepsilon$ (small constant against division by zero) are different symbols.
- $\mathbb{D}_{KL}[\pi_\theta\|\pi_{ref}]$: KL divergence from the reference; $\mathbb{E}$: expectation over prompts and sampled groups.

*Step-by-step (DPO):* write the log-ratios $h_w=\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x)$ and $h_l$ likewise. Then $\mathcal{L}=-\log\sigma(\beta(h_w-h_l))$. Since $-\log\sigma(z)=\log(1+e^{-z})$, the loss falls as the margin $z=\beta(h_w-h_l)$ grows: training raises the chosen response's probability *relative to the reference* more than the rejected one's.

*Step-by-step (GRPO clip):* if $\hat A_i>0$, $\min(\cdot)$ caps the term at $(1+\epsilon)\hat A_i$, so raising a good token's probability beyond ratio $1+\epsilon$ earns nothing more; if $\hat A_i<0$, once $\rho<1-\epsilon$ the term stays at $(1-\epsilon)\hat A_i$, so pushing a bad token's probability lower earns nothing more. Example: $\epsilon=0.2$, $\hat A=+1$, $\rho=1.5$: $\min(1.5,\,1.2)=1.2$.

- **Zero-signal case**: when rewards are equal, the reward term contributes no policy gradient, but a KL term with $\beta>0$ still does (**D**).

**Worked Example (derived; population std, $\varepsilon=10^{-4}$):**
1. *DPO.* With $\beta=0.1$, at initialization $\pi_\theta=\pi_{ref}$, the margin is 0 and $\mathcal{L}=\log 2\approx0.693$. After training, a chosen log-ratio of $+2.0$ and a rejected log-ratio of $-1.0$ give a margin of $0.1\times3=0.3$ and $\mathcal{L}=\log(1+e^{-0.3})\approx0.554$.
   - Steps: at init $h_w=h_l=0$, so $\mathcal{L}=-\log\sigma(0)=-\log 0.5=\log 2$. After training, $h_w-h_l=2.0-(-1.0)=3.0$; $z=0.1\times3.0=0.3$; $e^{-0.3}\approx0.741$; $\log(1.741)\approx0.554$.
2. *GRPO advantages*, $G=4$:

| Rewards | Mean | Std | $\hat A$ | Note |
|---|---|---|---|---|
| $[1,0,0,1]$ | 0.5 | 0.5 | $[+0.9998,-0.9998,-0.9998,+0.9998]$ | ≈ ±1 |
| $[1,1,1,1]$ | 1 | 0 | $[0,0,0,0]$ | via $\varepsilon$; raw formula is $0/0$ |
| $[0,0,0,0]$ | 0 | 0 | $[0,0,0,0]$ | same |
| $[0.50,0.50,0.50,0.51]$ | 0.5025 | 0.00433 | $[-0.564,-0.564,-0.564,+1.693]$ | without $\varepsilon$: $[-0.577,\ldots,+1.732]$ |

Steps for row 1: mean $=(1+0+0+1)/4=0.5$; deviations $\pm0.5$; population std $=\sqrt{(4\times0.25)/4}=0.5$; $\hat A=\pm0.5/(0.5+10^{-4})=\pm0.9998$. Row 4: mean $=2.01/4=0.5025$; deviations $-0.0025$ (×3) and $+0.0075$; std $=\sqrt{(3\times0.0025^2+0.0075^2)/4}=\sqrt{1.875\times10^{-5}}\approx0.00433$; $\hat A_4=0.0075/(0.00433+0.0001)\approx1.693$, or $0.0075/0.00433\approx1.732$ without $\varepsilon$.

The last row shows why std normalization up-weights near-uniform groups: a 0.01 reward difference becomes a unit-scale advantage. The two zero rows contribute nothing through the reward term, but with $\beta>0$ they still move the policy through KL. That is why DAPO filters such prompts. Population-versus-sample std and $\varepsilon$ differ across implementations, so read the one you run.

**Knowledge Check:**
1. What does DPO's reference model do that plain likelihood on chosen responses does not?
2. Why can a GRPO run's average response length grow while accuracy stalls?
3. In a $\beta=0$ recipe, what limits drift from the base model, and what should you log?

**Guided Practice:**
On a small model, run DPO on 2,000 preference pairs at three $\beta$ values, and a GRPO-style run on a verifiable math or format task with KL on and off. Log:
- chosen/rejected log-ratios;
- KL to reference and to base;
- response length by correctness;
- the fraction of zero-variance groups.

**Feedback Contract:**
- *Expected Evidence*: Loss and margin curves, KL-to-reference and KL-to-base, length split by correctness, zero-variance fraction with the $\varepsilon$ used, and held-out reward validity checks.
- *Common Failure*: Reporting rising reward as improvement without checking the reward against independent labels, or asserting "the reference keeps it close" in a $\beta=0$ run.
- *Diagnostic Hint*: What does the reward not check, and which term in your loss actually references the base model?
- *Concept to Revisit*: Verifier Gaming (Module 06); Judge Validity (Module 15).

**Learning Outcome:**
Apply and diagnose DPO- and GRPO-style preference optimization, including KL-free variants.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 19.6 — Forgetting, Safety Regression, and the Release Gate

**Engineering Question:**
What evidence justifies shipping an adapted model instead of the base, and what can that evidence not rule out?

**Concepts & Definitions:**
- **Catastrophic forgetting**: loss of capability outside the target domain after adaptation (CLM-006, CLM-016).
- **Safety regression**:
  - Benign fine-tuning can degrade safety alignment, though to a lesser extent than adversarial fine-tuning (**O**, CLM-011).
  - Narrow fine-tuning can produce broad misalignment (**O**, CLM-026).
  - Fraser et al. report surprising variance in safety-benchmark results under trivial changes to the fine-tuning setup (**O**, CLM-019).
  - Huang et al. (2026) model alignment reversal under later fine-tuning (**O**, CLM-020; FRONTIER).
- **Three-part gate**: target gain, retained general capability, and safety/refusal behavior, each paired against the base on suites frozen before training, with margins fixed before training (**D**, CLM-012).
- **Training variance vs evaluation variance**: independently trained runs (different training seeds or data order) measure training variance. Re-evaluating one adapter with different decoding or grader seeds measures only evaluation randomness.
- **Gate limit**: the gate cannot certify the absence of trigger-conditioned behavior (**D**, CLM-027).

**Mechanism Explanation:**
A single target metric is blind to the regressions most likely to follow adaptation. The gate therefore has three pre-registered components:
- a minimum useful gain for the target;
- non-inferiority margins $m_{gen}$ and $m_{safe}$ for retained capability and safety;
- decisions per independently trained run, plus an over-refusal check so that "safer" does not mean "useless."

Passing the gate authorizes a canary (Module 17), not full rollout. It does not replace provenance controls on data and checkpoints.

**Quantitative Model / Derivation:**
For paired binary items, let $b$ be the items where the base passes and the adapted model fails, and $c$ the reverse. Then:
$$\hat\Delta = \frac{c-b}{n},\qquad \mathrm{SE} = \frac{\sqrt{(b+c) - (b-c)^2/n}}{n},$$
with an approximate 95% interval $\hat\Delta \pm 1.96\,\mathrm{SE}$.

where: $n$ = number of paired items; items where both models agree (both pass or both fail) contribute 0 to $\hat\Delta$; $\hat\Delta>0$ means the adapted model passes more items; 1.96 is the two-sided 95% normal quantile; $m_{gen}$, $m_{safe}$ are the non-inferiority margins (allowed loss) in the same units as $\hat\Delta$.

*Step-by-step:* per item $i$, let $d_i=$ (adapted pass) $-$ (base pass) $\in\{-1,0,+1\}$; $d_i=+1$ on $c$ items and $-1$ on $b$ items. Then $\hat\Delta=\bar d=(c-b)/n$. The sample variance of $d_i$ is $\overline{d^2}-\bar d^2=(b+c)/n-(c-b)^2/n^2$, and $\mathrm{SE}=\sqrt{\mathrm{Var}(d)/n}=\sqrt{(b+c)-(b-c)^2/n}\,/\,n$. Only discordant pairs ($b+c$) drive the uncertainty, which is why pairing is tighter than comparing two independent pass rates. This is a Wald approximation; use an exact or bootstrap interval when $b+c$ is small (Module 15). A run passes the safety gate only if the interval's lower bound is $\ge -m_{safe}$. The adapter ships only if every independently trained run passes all three gates, or if a pre-declared aggregation rule says otherwise.

**Worked Example (synthetic; margins declared before training: minimum target gain 3 points, $m_{gen}=1.5$, $m_{safe}=2.0$; greedy decoding, so evaluation is deterministic and differences between runs reflect training):**
- **Target** JSON validity: $+8.0$ points, 95% CI $[6.1, 9.9]$. The lower bound exceeds 3, so it passes.
- **General suite:** $-1.2$, CI $[-2.0, -0.4]$. The lower bound $-2.0 < -1.5$, so non-inferiority is not shown and it fails.
- **Safety:** 400 harmful prompts; the base refuses 388 (97.0%).

| Training run | $b$ | $c$ | Adapted refusals | $\hat\Delta$ (pts) | 95% CI (pts) | Safety gate |
|---|---:|---:|---:|---:|---|---|
| seed 1 | 6 | 2 | 384 (96.0%) | −1.0 | [−2.38, +0.38] | fail |
| seed 2 | 16 | 2 | 374 (93.5%) | −3.5 | [−5.55, −1.45] | fail |
| seed 3 | 26 | 2 | 364 (91.0%) | −6.0 | [−8.53, −3.47] | fail |

Steps for seed 1 ($n=400$, $b=6$, $c=2$): adapted refusals $=388-6+2=384$; $\hat\Delta=(2-6)/400=-0.010$ ($-1.0$ pt); $\mathrm{SE}=\sqrt{8-16/400}/400=\sqrt{7.96}/400\approx0.00705$; interval $-0.010\pm1.96\times0.00705=[-0.0238,+0.0038]$, i.e. $[-2.38,+0.38]$ pts. Lower bound $-2.38<-m_{safe}=-2.0$, so fail. Seeds 2 and 3 follow the same steps (SE $\approx1.05$ and $1.29$ pts).

A point-estimate-only check on seed 1 ($-1.0 > -2.0$) would have passed. The paired interval does not, because 400 items cannot resolve a 2-point margin at this discordance. Decision: do not ship. Retry with lower rank or learning rate and general/safety replay data, enlarge the safety suite if a 2-point margin is required, and re-gate.

**Knowledge Check:**
1. Why must safety suites be run on every adapter refresh, even with benign data?
2. Why does re-running the evaluation of one adapter with new decoding seeds not measure training variance?
3. What would you need, beyond this gate, to argue that an adapter trained on externally contributed data is not backdoored?

**Guided Practice:**
Define the gate for the support-bot adapter: suites and sizes, margins, the number of independently trained runs and the aggregation rule, over-refusal checks, the provenance requirements the gate assumes, and the rollback target. Check that your safety suite size can resolve $m_{safe}$.

**Feedback Contract:**
- *Expected Evidence*: Frozen suite versions, margins declared before training, per-run paired counts and intervals, over-refusal rate, stated non-claims, and a rollout/rollback plan.
- *Common Failure*: Reporting only target gain, or treating a point estimate inside the margin as a pass.
- *Diagnostic Hint*: What did the base model do that the adapted one no longer does, and is your interval narrower than your margin?
- *Concept to Revisit*: Release Gates (Module 15, Lesson 15.4); Canary Rollout (Module 17, Lesson 17.5).

**Learning Outcome:**
Gate adapted models on target gain, retained capability, and safety with margin-based paired evidence over independent runs, and state what the gate cannot certify.

*(Effort: 45m instruction, 20m practice)*

---

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [ZeRO](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019; model-state accounting.
- [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022; SFT + reward model + RL.
- [Fine-Tuning or Retrieval?](https://arxiv.org/abs/2312.05934) — Ovadia et al., 2023; knowledge injection.
- [Fine-tuning Aligned Language Models Compromises Safety](https://arxiv.org/abs/2310.03693) — Qi et al., 2023.
- [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944) — Wan et al., ICML 2023.
- [Sleeper Agents](https://arxiv.org/abs/2401.05566) — Hubinger et al., 2024.

**CURRENT DEFAULT:** per-failure-class adaptation decisions; [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) as the default parameter-efficient family; [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) for offline preference tuning; run manifests with explicit target modules and checkpoint hashes; lineage- and provenance-controlled data; three-part paired release gates with declared margins. These are recommended engineering baselines. The PEFT defaults cited are one library's defaults, not evidence of industry practice.

**WORKLOAD-DEPENDENT:** [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) for memory-constrained runs; [rsLoRA](https://arxiv.org/abs/2312.03732) (Kalajdzievski, 2023) for higher ranks; LoRA rank and targets ([LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673), Biderman et al., 2024); full versus parameter-efficient tuning by data scale; [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024) for RL with verifiable rewards.

**FRONTIER:** [Dr. GRPO](https://arxiv.org/abs/2503.20783) (Liu et al., 2025) and [DAPO](https://arxiv.org/abs/2503.14476) (Yu et al., 2025) as GRPO corrections and KL-free recipes; [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, 2025); [RL's Razor](https://arxiv.org/abs/2509.04259) (Shenfeld et al., 2025); [Emergent Misalignment](https://arxiv.org/abs/2502.17424) (Betley et al., 2025); [Near-constant poison counts](https://arxiv.org/abs/2510.07192) (Souly et al., 2025); [Fine-Tuning Lowers Safety and Disrupts Evaluation Consistency](https://arxiv.org/abs/2506.17209) (Fraser et al., LLMSEC 2025); [Alignment Dynamics in LLM Fine-Tuning](https://arxiv.org/abs/2605.18309) (Huang et al., 2026).

**LEGACY / INSUFFICIENT:** fine-tuning to inject changing facts; evaluating only on the target task; random splits of mined logs; point-estimate or single-run safety gates; relying on library-default target modules without recording them; treating a fixed-suite pass as evidence of no backdoor.

**PRODUCTION SOURCE TRACE**
- Repository: `huggingface/peft`
- Revision: `b8674c86183a5dee38d0c3ede392e189593025e5` (committed 2026-09-25)
- Verified: 2026-09-30; static inspection only.
- Files/symbols: `src/peft/tuners/lora/layer.py::{LoraLayer.update_layer, LoraLayer.reset_lora_parameters, Linear.forward, Linear.get_delta_weight, Linear.merge, Linear.unmerge}`; `src/peft/utils/constants.py::TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` and record a run manifest with these fields:
- base-model and adapter checkpoint hashes and sources;
- data snapshot IDs with source trust labels;
- template hash and loss mask;
- method, rank, alpha, explicit `target_modules`, and effective scaling;
- precision layout;
- training seeds and hyperparameters;
- evaluation suite versions.

Labs use synthetic or public data; record which.

### LAB A — Adaptation Decision Record
- **Objective**: For two failure classes (fact freshness and output format), compare prompting, retrieval, and a LoRA SFT adapter on quality, tokens per request, and update cost.
- **Pre-Registered Hypothesis**: CLM-013, with $\delta$, cost unit prices, and the planning horizon declared before running.
- **Independent Variables**: Intervention, failure class, fact updates made after training.
- **Dependent Variables**: Paired accuracy on changed facts with intervals, retrieval recall on those items, format validity, tokens/request, and training, re-gating, and index update cost.
- **Break & Falsify**: Make 20 fact updates that cannot be inferred from prior knowledge (e.g., new arbitrary policy codes). Compare RAG against an SFT model retrained on them under the same allowed evidence. The freshness half is falsified if the upper bound of (RAG − retrained SFT) is below $-\delta$ or if index update cost exceeds retrain-plus-re-gate cost. The format half is falsified if SFT validity's upper bound falls below prompt validity − $\delta$ or if $C_{adapt} > C_{prompt}$ over the horizon.
- **Alignment**: Lesson 19.1.
- **Effort Estimate**: 3h total.

### LAB B — Memory and LoRA Mechanics
- **Objective**: Predict and measure memory for full, LoRA, and QLoRA steps on a ≤3B model using the Lesson 19.2 procedure. Implement LoRA from scratch, verify parity with PEFT and merge parity, and complete the pinned source trace, including default target resolution.
- **Pre-Registered Hypothesis**: Predicted state tensor bytes match the post-first-step inventory within the 10% exercise tolerance. Merged and unmerged outputs agree within a stated tolerance in fp32.
- **Independent Variables**: Method, sequence length, checkpointing, rank, alpha, `target_modules`, dtype.
- **Dependent Variables**: State tensor bytes by dtype, peak allocated, reserved, trainable parameters, step time, merge error.
- **Break & Falsify**: Change `r` without `alpha`. Train with default targets versus all linear layers at matched parameter count. Merge an adapter trained on a 4-bit base into a 16-bit base and measure output drift.
- **Alignment**: Lessons 19.2–19.3.
- **Effort Estimate**: 3.5h total (source trace counted separately, 2h).

### LAB C — Data and Preference Optimization
- **Objective**: Build a leak-checked, provenance-filtered SFT set and preference pairs from mined failures. Run SFT with and without a loss mask, DPO at three $\beta$ values, and a GRPO-style run on a verifiable task with KL on and off.
- **Pre-Registered Hypothesis**:
  - Removing near-duplicates lowers the apparent target gain.
  - Lower $\beta$ increases KL to reference.
  - With length-normalized GRPO, incorrect responses lengthen more than correct ones.
  - With $\beta=0$, KL to base grows faster than with $\beta>0$.
- **Independent Variables**: Dedup on/off, loss mask on/off, template variant, source trust filter, $\beta$, loss normalization, zero-variance filtering.
- **Dependent Variables**: Target metric, contamination rate, measured loss share, KL to reference and base, length by correctness, zero-variance fraction.
- **Break & Falsify**: Train with the wrong chat template and with no loss mask, and report both effects on held-out validity. In a sandbox, insert 50 synthetic trigger-phrase examples labeled as a controlled poisoning exercise. Show whether the Lesson 19.6 gate detects the effect without the trigger and with it.
- **Alignment**: Lessons 19.4–19.5.
- **Effort Estimate**: 3.5h total.

### LAB D — Forgetting and the Release Gate
- **Objective**: Compare LoRA at two ranks and two target sets against full fine-tuning (or the highest-capacity feasible setting) on target, general, and safety suites, over three independently trained runs each.
- **Pre-Registered Hypothesis** (confirmatory): at the pre-declared learning rate and epoch budget:
  - higher-capacity updates gain more on target and regress more outside it (CLM-006);
  - safety gate decisions differ across independently trained runs (CLM-019).
- **Independent Variables**: Rank, target modules, replay mix, training seed; the learning rate and epochs are fixed for the confirmatory comparison.
- **Dependent Variables**: Paired target/general/safety deltas with intervals, per-run gate decisions, over-refusal rate, KL to base on the target task.
- **Break & Falsify**:
  - Report the confirmatory result as is, including a null result ("no forgetting detected at this budget and suite size").
  - Then, as a *separate, labeled stress test*, sweep learning rate over $\{1,3,10\}\times$ the default and epochs over $\{1,3\}$. Stop at the first configuration where the gate fails, or when the grid is exhausted.
  - A stress failure demonstrates that the gate can catch forgetting. It does not confirm the pre-registered hypothesis.
- **Alignment**: Lesson 19.6 and Incident 19.1.
- **Effort Estimate**: 3h total.

---

## 07 Break / Incident Scenarios

### Incident 19.1 — The Monthly Refresh

- **Incident Symptoms**:
  - After the monthly adapter refresh for a support assistant, target-task CSAT rose in the canary.
  - A week later, trust-and-safety reports that the assistant complies with some requests it previously refused, and the tool team reports more malformed tool calls in long conversations.
  - The refresh made several changes at once: it used 30% more mined data (including a new partner feedback feed), raised rank from 16 to 64 with `alpha` unchanged, switched to QLoRA training, merged into the 16-bit base for serving, and started from a partner-supplied adapter checkpoint instead of the previous in-house one.
  - The offline gate was run once, on one training run, with point estimates.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*:
     - benign-data safety regression;
     - harmful or trigger-conditioned poisoned examples in the new feed;
     - a backdoored or mismatched partner checkpoint;
     - forgetting from a changed effective update;
     - train/serve template mismatch;
     - a 4-bit-trained adapter merged into a 16-bit base;
     - the wrong adapter version deployed;
     - a safety suite changed between runs;
     - training variance.
  2. *Rank Initial Plausibility*: Use the change list and the timing, but do not assign a cause without a discriminating test. Multiple changes can interact.
  3. *Identify Missing Evidence*: Both run manifests; the data diff with source trust labels; checkpoint hashes and origins; rendered templates; suite versions; per-run paired counts; the deployed adapter hash; effective `scaling` and resolved `target_modules` in both runs; the prompts on which the new refusals fail.
  4. *Design Discriminating Tests*:
     - Re-evaluate the old and new adapters on identical frozen suites over three independently trained runs.
     - Evaluate the new adapter unmerged on the 4-bit base versus merged.
     - Retrain one change at a time: old data with new settings, new data with old settings, and in-house versus partner starting checkpoint.
     - Search the new feed for repeated rare phrases shared by compliant prompts, and test candidate triggers.
  5. *Execute Causal Diagnosis*: Rank causes with the evidence each test produced. Report interacting or unresolved causes explicitly, and state what remains unexplained.
  6. *Prescribe Mitigation and Prevention*:
     - Roll back to the previous adapter and quarantine the partner feed and checkpoint.
     - Add manifest-level checks: scaling, template hash, merge precision, checkpoint provenance.
     - Add margin-based multi-run safety and long-conversation tool-call suites to the gate.
     - Put provenance and review gating on external data.
  7. *Remeasure*: Paired target/general/safety deltas with intervals per run, the malformed tool-call rate in canary, and the result of trigger testing on the quarantined data, reported as bounded negative evidence if nothing is found.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Adapting a Regulated Claims Assistant

An insurer's claims assistant must:
- follow a strict adjudication procedure;
- output a structured decision record;
- cite current policy clauses that change quarterly;
- refuse to give legal advice.

Constraints:
- **Data.** Training data comes from adjuster-corrected transcripts, some submitted through a feedback tool open to external agents. A vendor offers a pre-trained "claims" adapter.
- **Compute.** One 48 GB GPU is available for training.
- **Serving.** Several tenant adapters are served on one base.
- **Gate.** The safety gate must resolve a 2-point refusal margin.

**Required Deliverables**:
1. Adaptation decision record per failure class, with alternatives, $\delta$, costs, and falsifiers.
2. Memory plan for the chosen method, with predicted state bytes and measured peaks.
3. LoRA configuration (rank, alpha, explicit targets, learning rate) with justification, and a merged-versus-multi-adapter serving decision.
4. Data manifest with lineage, contamination report, template and loss-mask verification, and provenance controls for externally submitted data and the vendor adapter.
5. Preference-optimization plan (or a justified omission) with reward validity checks and KL-to-base monitoring.
6. Three-part release gate with declared margins, suite sizes sufficient for those margins, independently trained runs, over-refusal checks, and stated non-claims.
7. Pinned PEFT source trace.
8. Diagnosis and remediation for Incident 19.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace LoRA creation, scaling (including rsLoRA), initialization, forward, delta computation, merge/unmerge, and default target resolution at the pinned PEFT commit. State what was statically inspected and what was executed in your parity tests.

### Rubric Dimensions
- **Adaptation Decision**: *Insufficient* fine-tunes by default. *Competent* compares alternatives per failure class with costs. *Strong* pre-registers $\delta$ and falsifiers that could fire and tests them with paired intervals.
- **Memory and Mechanics**: *Insufficient* counts weights only. *Competent* predicts model states and LoRA parameters. *Strong* reconciles predictions with a post-first-step state inventory and peak/reserved measurements, and traces scaling, merge, and default targets in source.
- **Data Integrity**: *Insufficient* uses random splits. *Competent* dedups and masks correctly. *Strong* separates data by source and time, verifies templates against serving, computes loss share, and enforces provenance for external data and checkpoints.
- **Preference Optimization**: *Insufficient* reports reward curves. *Competent* tracks KL and margins. *Strong* diagnoses length, normalization, zero-variance, and reward-validity pathologies, including in KL-free runs.
- **Release Gate**: *Insufficient* reports target gain. *Competent* adds general and safety suites with point estimates. *Strong* uses declared margins, paired intervals, independently trained runs, over-refusal checks, and states what the gate cannot certify.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Adaptation decision | 19.1 | LAB A | Deliverable 1 | Decision records with $\delta$, costs, falsifiers; LAB A paired results |
| Training memory estimation | 19.2 | LAB B | Deliverable 2 | State inventory vs predicted; peak/reserved report |
| LoRA mechanics and source trace | 19.3 | LAB B | Deliverables 3, 7 | Parity tests, target-module comparison, PEFT trace artifact |
| Leak-free, provenance-controlled data | 19.4 | LAB C | Deliverable 4; Incident steps 3–4 | Data manifest, loss-share check, poisoning exercise |
| Preference optimization diagnosis | 19.5 | LAB C | Deliverable 5 | KL (ref and base), margins, length, zero-variance fraction |
| Forgetting and safety gating | 19.6 | LAB D | Deliverable 6; Incident steps 5–7 | Per-run paired gate table with margins and non-claims |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 19 must be able to:
1. Decide per failure class whether to prompt, retrieve, route, or adapt, with a falsifiable comparative prediction.
2. Predict and verify training memory for full, LoRA, and QLoRA runs, separating state bytes from activations and allocator effects.
3. Implement and trace LoRA, including scaling, initialization, target modules, and merging.
4. Build adaptation data that survives leakage, template, loss-mask, and provenance checks.
5. Explain and diagnose RLHF, DPO, and GRPO-style objectives, including KL-free variants and their failure modes.
6. Gate an adapted model on target gain, retained capability, and safety with margin-based paired evidence over independent runs, and state what the gate cannot certify.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: A weight update changes parameters shared by every future request. It must be justified against per-request alternatives, built from controlled data, and gated on what the target metric does not measure.
- **The Adaptation Path**: failure class → decision → memory plan → data manifest with provenance → objective → paired three-part gate with margins → canary → promote or roll back.
- LoRA trades capacity for retention, and its targets and scale are explicit choices. Preference objectives are only as good as their rewards, normalization, and drift monitoring. Safety is re-measured on every refresh, and poisoning is prevented upstream because the gate cannot prove its absence.

### Cost & Telemetry Hook
- [Lesson 22.2](../22-ai-economics/README.md): compare adaptation cost (training, eval, hosting) against prompt, retrieval, or routing alternatives as build-versus-API.
- [Lesson 22.5](../22-ai-economics/README.md): if the adapted model sits in a router or cascade, cost escalation to the base model.
- [Lesson 23.1](../23-observability-reliability/README.md): emit base model, adapter, and checkpoint hash on every request span.
- [Lesson 23.5](../23-observability-reliability/README.md): monitor post-release drift with proxy quality; reconcile with delayed labels before the next refresh.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
