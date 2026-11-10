# Module 24 — Model-System Co-design

## 00 Why This Module Exists
Changing a model's architecture changes the work the system must do. A different attention variant changes the retained state per token. A sparse expert layer changes which weights must stay resident, which weights are touched per token, and which bytes cross the network. Those changes reach the runtime's page layout and capacity metadata. The scheduler then turns that capacity into batches and latency. The quality gate and the cost boundary decide whether any of it was worth doing.

Earlier modules teach each layer on its own: architecture accounting (Module 01), device bounds (Module 02), KV representation (Module 03), scheduling and capacity (Module 04), and kernel, quantization, and speculation contracts (Module 05). This module builds the coupling model across them. It asks one question: **when an architecture changes, which constraint binds next, and what useful outcome changes?**

The outline for this module asked for the "exact financial impact of switching from MHA to GQA". That request has no answer. A KV-byte ratio is one input. The financial effect depends on whether KV capacity was the binding constraint, on what binds after the change, on quality, on the measured SLO-goodput of the matched workload, and on dated prices. This module replaces "exact impact" with a **conditional estimate with uncertainty and switching thresholds**.

**Module Orientation**
- **Engineering Problem**: Choose and defend a model architecture and system mapping (attention variant, dense versus MoE, precision, parallel layout, scheduler limits) for a declared workload. The choice is judged on SLO-constrained goodput that passes a shared quality gate, and its cost is reported as a conditional estimate.
- **What You Will Do**:
  - Compile real model configs into a resource vector.
  - Trace GQA and MLA fields through a pinned vLLM cache path.
  - Predict whether a bottleneck stays or moves, then test that prediction with a sweep.
  - Model dense versus MoE communication on a declared topology.
  - Run a matched-workload comparison with one quality and system contract.
  - Write a conditional cost and sensitivity analysis.
  - Diagnose an incident where a KV saving did not reduce cost.
  - Defend an architecture decision with reversal thresholds.
- **Environment**: Python 3.10+ for the manifest compiler, simulators, and analysis (Labs A–C). Lab D needs a vLLM checkout at the pinned commit for static tracing. Executing it on one GPU is optional but valuable. Multi-node MoE experiments are optional; without them, Lab B uses a labeled synthetic topology model.
- **Evidence Rule**: Every statement is labeled **O** (source observation), **D** (derivation), or **H** (hypothesis), with the claim ID next to it. Synthetic inputs are labeled. No number in this module is a measurement of your system.

## 01 Baseline Assumptions
These prerequisites are taught in the cited lessons. They are used here without being re-taught.
- **Attention geometry**: query heads, KV heads, the head-to-group map, and what GQA does and does not shrink (Module 01, Lesson 1.3).
- **Parameter and FLOP accounting**: attention and SwiGLU parameter formulas, the 2-FLOPs-per-FMA convention, and the decode attention term $4dS_k$ per layer (Module 01, Lesson 1.6).
- **MoE and MLA accounting**: total, resident, selected, and executed parameters; MLA retained state $(d_c+d_h^R)$; why the GQA formula does not apply to MLA (Module 01, Lesson 1.7).
- **Roofline**: $T\ge\max(W/P_{peak},Q/\beta)$, ridge intensity, matched byte boundary, and shape-dependent regime transitions (Module 02, Lessons 2.4 and 2.5).
- **KV payload and paging**: logical bytes per token for uniform attention and MLA, and the gap between logical payload and allocator occupancy (Module 03, Lessons 3.1 and 3.3).
- **Serving outcomes**: Little's Law boundaries, goodput, admission, and capacity screening with resident concurrency (Module 04, Lessons 4.4, 4.6, and 4.7).
- **Deployment contracts**: quantization as a contract (Module 05, Lesson 5.3), exact speculative decoding (Module 05, Lesson 5.5), and composing optimizations without losing attribution (Module 05, Lesson 5.7).
- **Comparison statistics**: confidence intervals (Module 00, Lesson 0.5), and paired comparison with noninferiority margins and critical-slice gates (Module 15, Lesson 15.4).

Two topics are **taught at first use here** because no earlier lesson owns them:
- **all-to-all expert dispatch/combine cost** (Lesson 24.4). Module 20 — Distributed Inference covers collectives, topology, and expert parallelism in depth; this module uses only the screening model.
- **sensitivity, switching thresholds, and Pareto dominance** (Lessons 24.5–24.6). Module 22 — AI Economics owns full cost accounting; this module uses only the conditional replica-cost model needed for an architecture decision.

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
  statistical: SELECTIVE
  production_reasoning: REQUIRED
  failure_analysis: REQUIRED
  falsification: REQUIRED
  security: NOT_APPLICABLE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 5h        # sum of lesson instruction estimates: 50+50+50+50+50+50 min
  guided_practice: 1.5h  # sum of in-lesson practice: 20 (24.1) + 20 (24.3) + 15 (24.4) + 15 (24.5) + 20 (24.6) min; 24.2 practice is the source trace
  labs: 13.5h            # LAB A 3h + LAB B 3.5h + LAB C 4h + LAB D 3h
  assessment: 3.5h       # Mastery transfer problem 3h + Incident 24.1 0.5h
  source_trace: 2h       # Lesson 24.2 guided trace and the Section 09 Production Source Trace artifact are one activity, counted once here
  total: 25.5h
```
Each category is counted once. The source trace is not also counted as Lesson 24.2 practice or as Lab D time. Lab D reuses the trace but adds measurement and reconciliation work only.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map
Architecture manifest (attention variant, heads, latent widths, experts, routing limits, precision) $\to$ logical resource vector (weights, retained state, FLOPs, bytes moved, communication payload) $\to$ runtime mapping (sharding, replication, page layout, expert placement) $\to$ physical capacity and per-step cost $\to$ scheduler service curve (batch limits, admission, phase layout) $\to$ SLO-goodput on a matched workload $\to$ shared quality gate $\to$ conditional cost and sensitivity $\to$ decision with reversal thresholds $\to$ remeasurement.

Feedback runs backwards at every arrow: a quality failure, a moved bottleneck, or a topology constraint can send the design back to the architecture manifest. DeepSeek-V3's node-limited routing is an example: a routing rule designed around a network topology (**O**, CLM-007).

## 04 Lessons

### Lesson 24.1 — The Co-design Contract and the Resource Vector

**Engineering Question:**
Which quantities does an architecture fix, which does the runtime and hardware mapping fix, and which only a loaded measurement can reveal?

**Concepts & Definitions:**
- **Architecture manifest**: the exact model fields that determine computation and state, pinned to a revision. Examples are layers, hidden size, query and KV heads, MLA latent widths, expert count, top-$k$, routing limits, vocabulary, tying, and storage dtype. A model name is not a manifest.
- **Resource vector** $R$: the per-request or per-step demands a manifest implies, each with a unit and a boundary:
  $$R=(M_{weights},\,M_{state}(S),\,W_{prefill},\,W_{decode},\,Q_{HBM},\,Q_{comm},\,D_{sched})$$
  These are weight bytes, retained state bytes at context $S$, prefill and decode FLOPs, bytes moved at the HBM boundary, network bytes, and scheduler-visible demands such as resident sequences and token budget.
- **Value classes**: each entry is labeled **logical** (from the manifest), **analytical bound** (Roofline-style), **measured**, or **hypothesis**. Mixing classes without labels is the main error this module guards against.
- **Co-design contract**: the manifest, the runtime and hardware mapping, the workload, the shared quality and system gates, and the cost boundary, all versioned together. Two architectures are comparable only under one contract (**D**, CLM-013).

**Mechanism Explanation:**
The manifest determines logical counts exactly under stated assumptions. The mapping turns them into physical quantities: TP degree splits or replicates heads, page size rounds state, and precision changes bytes. The scheduler converts physical capacity and per-step cost into a service curve: how many requests, at what TTFT and TPOT, for this workload. Gates turn the service curve into useful outcomes. An architecture change perturbs the first box. Whether anything downstream changes depends on which constraint was binding (**D**, CLM-012).

Sparse experts make the boxes diverge sharply. Total parameters set resident bytes, selected parameters set expert FLOPs, and every selected token crosses the expert placement boundary twice (**D**, CLM-006).

**Quantitative Model / Derivation:**
For a dense GQA decoder with $L$ layers, hidden size $d$, $H_q$ query heads, $H_{kv}$ KV heads, $d_h=d/H_q$, MLP width $m$, and vocabulary $V$, bias-free except where noted:
- Attention weights per layer: $2d^2+2dH_{kv}d_h$ (Module 01, Lesson 1.6).
- Logical KV bytes per token: $2LH_{kv}d_hb$ (**D**, CLM-002).
- Decode FLOPs per token at retained context $S$: $2P_{matmul}+4dSL$, where $P_{matmul}$ counts the projection, MLP, and LM-head matrix parameters.
- One decode step for $B$ sequences at context $S$, as a screen:
  $$Q_{HBM}\approx M_{weights}+B\,S\,k_{KV},\qquad W\approx B(2P_{matmul}+4dSL),\qquad T_{step}\ge\max\!\left(\frac{W}{C},\frac{Q_{HBM}}{\beta}\right)$$
  Here $k_{KV}$ is KV bytes per token, $C$ is the attainable compute ceiling, and $\beta$ is the attainable HBM bandwidth. Both are aggregated over the GPUs of one replica, and the bound ignores communication and scheduler time (**O**, CLM-008).

*Where the formulas come from (step by step):*
- **Symbols**: $b$ = bytes per stored element (2 for BF16); $S$ = tokens already retained in the context (written $S_k$ in Module 01's $4dS_k$); $B$ = sequences decoded together in one step; $M_{weights}$ = weight bytes ($=bP$ when every parameter is stored at $b$ bytes); $W$ = FLOPs per step; $T_{step}$ = time per step.
- **Attention weights**: $W_Q$ and $W_O$ each map $d\to d$ ($d^2$ each, since $H_qd_h=d$). $W_K$ and $W_V$ each map $d\to H_{kv}d_h$ ($dH_{kv}d_h$ each). Sum: $2d^2+2dH_{kv}d_h$. With $H_{kv}=H_q$ (MHA) this becomes $4d^2$.
- **KV bytes per token**: each layer stores one K and one V vector (factor $2$) per KV head ($H_{kv}$), each of $d_h$ elements at $b$ bytes, in each of $L$ layers: $2\cdot L\cdot H_{kv}\cdot d_h\cdot b$.
- **Decode FLOPs**: a matmul with $P$ weights costs one FMA $=2$ FLOPs per weight per token, so $2P_{matmul}$. Attention over $S$ past tokens per layer costs $QK^\top$ ($d$ multiply-adds per past token, $2dS$ FLOPs) plus the weighted sum of $V$ (another $2dS$), so $4dS$ per layer and $4dSL$ in total. The input-embedding lookup is a gather, not a matmul, so only the LM head contributes $V\cdot d$ to $P_{matmul}$.
- **Step bytes**: one decode step reads the weights once, shared by all $B$ sequences ($M_{weights}$), and each sequence reads its own $S$ tokens of KV ($S\,k_{KV}$). Hence $Q_{HBM}\approx M_{weights}+B\,S\,k_{KV}$.
- **Step bound**: the step cannot finish faster than its FLOPs at full compute ($W/C$) or its bytes at full bandwidth ($Q_{HBM}/\beta$), so $T_{step}$ is at least the larger of the two. Arithmetic intensity is $I=W/Q_{HBM}$; the ridge is $C/\beta$. If $I<C/\beta$, the bytes term is larger.

**Worked Example (real config, synthetic ceilings):**
- *Input*: the pinned Qwen2.5-72B-Instruct config: $L=80$, $d=8192$, $H_q=64$, $H_{kv}=8$, $m=29{,}568$, $V=152{,}064$, untied embeddings, BF16, and q/k/v biases in the Qwen2 attention (**O**, CLM-022).
- *Step 1 — parameters* (derived widths: $d_h=8192/64=128$, so K and V each have $H_{kv}d_h=8\cdot128=1024$ outputs):
  - attention weights per layer: $2\cdot8192^2+2\cdot8192\cdot8\cdot128=150{,}994{,}944$;
  - q/k/v biases: $8192+2\cdot1024=10{,}240$ (one bias per output: $d$ for Q, $1024$ each for K and V);
  - SwiGLU: $3\cdot8192\cdot29{,}568=726{,}663{,}168$ (gate, up, and down matrices, each $d\times m$);
  - two norms: $16{,}384$ ($2\cdot d$ scale vectors);
  - block total: $877{,}684{,}736$.
  - $80$ blocks $=70{,}214{,}778{,}880$, plus final norm $8{,}192$, plus embedding and LM head $2\cdot152{,}064\cdot8192=2{,}491{,}416{,}576$, gives $P=72{,}706{,}203{,}648$. This matches the Hub metadata exactly (**O**, CLM-022).
- *Step 2 — bytes*: $2P=145{,}412{,}407{,}296$ bytes $=145.41$ GB $\approx135.43$ GiB of logical weight payload. KV: $2\cdot80\cdot8\cdot128\cdot2=327{,}680$ bytes $=320$ KiB per token.
  - Units: GB $=10^9$ bytes, GiB $=2^{30}=1{,}073{,}741{,}824$ bytes, KiB $=1024$ bytes. So $145{,}412{,}407{,}296/10^9=145.41$ GB and $145{,}412{,}407{,}296/2^{30}\approx135.43$ GiB; $327{,}680/1024=320$ KiB.
- *Step 3 — FLOPs per decode token at $S=8192$*:
  - $P_{matmul}=80(150{,}994{,}944+726{,}663{,}168)+152{,}064\cdot8192=71{,}458{,}357{,}248$, so matmuls cost $142{,}916{,}714{,}496$ FLOPs. (Biases, norms, and the input embedding are excluded: they are adds, scales, or a lookup, not matrix products. $P_{matmul}=P-80(10{,}240+16{,}384)-8{,}192-152{,}064\cdot8192$.);
  - attention: $4\cdot8192\cdot8192\cdot80=21{,}474{,}836{,}480$ FLOPs ($4\cdot d\cdot S\cdot L$ with $d=8192$ and $S=8192$ — the two equal numbers are a coincidence of this example);
  - total per token: $142{,}916{,}714{,}496+21{,}474{,}836{,}480=164{,}391{,}550{,}976$ FLOPs, so attention is about $13\%$ of decode FLOPs at this context.
- *Step 4 — one step, $B=32$*, with **synthetic** replica ceilings $C=6.4\times10^{15}$ FLOP/s and $\beta=24\times10^{12}$ B/s (ridge $266.7$ FLOP/byte):
  - $Q_{HBM}=145{,}412{,}407{,}296+32\cdot8192\cdot327{,}680=231{,}311{,}753{,}216$ bytes (weights $145.41$ GB plus KV reads $85{,}899{,}345{,}920$ bytes $\approx85.90$ GB);
  - $W=32\cdot164{,}391{,}550{,}976=5{,}260{,}529{,}631{,}232$ FLOPs, where $164{,}391{,}550{,}976=142{,}916{,}714{,}496+21{,}474{,}836{,}480$ is the per-token total from Step 3;
  - $I=W/Q_{HBM}=5.2605\times10^{12}/(2.3131\times10^{11})\approx22.74$ FLOP/byte, far below the ridge $C/\beta=(6.4\times10^{15})/(24\times10^{12})=266.7$;
  - bounds: $T_{mem}=Q_{HBM}/\beta=2.3131\times10^{11}/(24\times10^{12})\ \text{s}\ge9.638$ ms and $T_{comp}=W/C=5.2605\times10^{12}/(6.4\times10^{15})\ \text{s}\ge0.822$ ms.
- *Result*: under these assumptions, the step screen is bounded by bytes moved, not by FLOPs, by roughly $11.7\times$ ($9.638/0.822\approx11.7$, equivalently ridge $/\,I=266.7/22.74$).
- *Interpretation / limits*: this is a lower bound for one decode step and one replica. It excludes TP all-reduce, sampling, scheduler gaps, attention kernel efficiency, and quantization. It is a hypothesis to test with a profiler (Module 02). It does not say the deployment is "memory-bound" at every batch size or context.

**Knowledge Check:**
1. Which entries of $R$ change when the same checkpoint moves from TP=8 to TP=16, and which do not?
2. Why can $P$ be exact while $M_{weights,resident}$ is only a measurement?
3. In Step 4, how does the bound change if $B$ doubles? Which term grows, and which does not?

**Guided Practice:**
Compile the same resource vector for the Qwen2.5-7B-Instruct config (28 layers, $d=3584$, 28 query heads, 4 KV heads, $m=18{,}944$, untied, BF16). Check your total against $7{,}615{,}616{,}512$. Then state which entries you would label logical, analytical, measured, or hypothesis.

**Feedback Contract:**
- *Expected Evidence*: a per-term parameter table that sums to the published total; KV bytes per token of $2\cdot28\cdot4\cdot128\cdot2=57{,}344$ bytes $=56$ KiB; a step bound with the ceilings labeled synthetic; every entry labeled with its value class.
- *Common Failure*: dropping the untied LM head or the q/k/v biases, or treating the Roofline step bound as a measured ITL.
- *Diagnostic Hint*: if your total is off by $V\cdot d$, which tensor did you count once that the config stores twice?
- *Concept to Revisit*: parameter and FLOP accounting (Module 01, Lesson 1.6); Roofline as a qualified bound (Module 02, Lesson 2.4).

**Learning Outcome:**
Compile a pinned architecture manifest into a labeled resource vector and state which downstream quantities it does not determine.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 24.2 — Attention Variants Through a Production Cache Path

**Engineering Question:**
How does an attention-variant choice become physical pages and a concurrency number in a real runtime, and where can the analytical KV ratio and the runtime's number diverge?

**Concepts & Definitions:**
- **MHA / MQA / GQA**: GQA uses more than one and fewer KV heads than query heads. Its authors report quality close to MHA with speed close to MQA in their evaluated settings (**O**, CLM-001). That quality result belongs to their trained models; swapping in fewer KV heads without training is a different experiment.
- **MLA**: retains one joint latent plus a decoupled RoPE key per token per layer, not per-head K/V. DeepSeek-V2 reports retained state equal to GQA with 2.25 groups, and better evaluated quality than MHA in its matched ablations (**O**, CLM-003).
- **Logical versus physical state**: logical bytes come from the manifest (**D**, CLM-002). Physical bytes add TP replication, page rounding, alignment padding, packed formats, and quantization scales.
- **Cache spec**: the runtime object that declares, per layer, how many heads are stored, their sizes, the dtype, and the page size.

**Mechanism Explanation:**
At the pinned vLLM commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`:
1. `GPUModelRunner.get_kv_cache_spec` asks every attention layer for its spec (**O**, CLM-011).
2. A standard decoder `Attention.get_kv_cache_spec` returns a `FullAttentionSpec` built from the *rank-local* `num_kv_heads`, `head_size`, `head_size_v`, dtype, and block size. `AttentionSpec.state_content_size_bytes` is `(head_size + head_size_v) * dtype_size`, and the page size is heads × states × that value (**O**, CLM-009).
3. Local KV heads come from the model file. `Qwen2Attention.__init__` partitions KV heads across TP ranks when there are at least as many KV heads as ranks. Otherwise it requires the TP size to be a multiple of the KV-head count and replicates, keeping `max(1, total_kv_heads // tp_size)` per rank (**O**, CLM-021).
4. MLA follows another class. `MLAAttention` sets `head_size = kv_lora_rank + qk_rope_head_dim`, returns an `MLAAttentionSpec` with `num_kv_heads=1` and `head_size_v=0`, and may substitute packed `state_content_bytes` and alignment padding (**O**, CLM-010).
5. `_get_kv_cache_bytes_per_block` sums page bytes per group. `get_kv_cache_config_from_groups` sets `num_blocks = available_memory // bytes_per_block`. `get_max_concurrency_for_kv_cache_config` divides blocks by whole blocks per request at `max_model_len` (**O**, CLM-011).

The runtime's capacity number is therefore a function of architecture × TP layout × dtype × page size × `max_model_len`. It is not a function of the KV ratio alone.

**Quantitative Model / Derivation:**
- Uniform attention, logical: $k_{KV}=2LH_{kv}d_hb$ (**D**, CLM-002).
- MLA, logical: $k_{MLA}=L(d_c+d_h^R)b$, from the paper's retained state (**O**, CLM-003).
- Physical KV per token across a TP group of size $t$, uniform attention, under the pinned Qwen2 rule: $k_{phys}=t\cdot2L\max(1,\lfloor H_{kv}/t\rfloor)d_hb$, before page rounding. When $t>H_{kv}$, this exceeds the logical value by $t/H_{kv}$.

*Where:* $d_c$ = MLA joint KV latent width (`kv_lora_rank`); $d_h^R$ = decoupled RoPE key width per token (`qk_rope_head_dim`); $t$ = TP degree; $\lfloor\cdot\rfloor$ = round down.

*Step by step:*
- **GQA versus MHA ratio**: everything in $2LH_{kv}d_hb$ is unchanged except $H_{kv}$, so $k_{MHA}/k_{GQA}=H_q/H_{kv}$. For Qwen2.5-72B: $64/8=8\times$.
- **MLA**: one latent of $d_c$ elements plus one RoPE key of $d_h^R$ elements per layer, with no separate V and no per-head copies, so $L(d_c+d_h^R)b$. Equivalently, it equals GQA with $g$ groups when $2gd_h=d_c+d_h^R$.
- **TP rule**: when $t\le H_{kv}$ (and $t$ divides $H_{kv}$), each rank holds $H_{kv}/t$ heads and $t$ ranks hold $t\cdot H_{kv}/t=H_{kv}$: no excess. When $t>H_{kv}$, each rank holds $\max(1,0)=1$ head, so $t$ ranks hold $t$ head copies instead of $H_{kv}$: ratio $t\cdot1/H_{kv}=t/H_{kv}$.

**Worked Example (pinned configs; same-dimension MHA is a counterfactual):**
- *GQA, real*: Qwen2.5-72B has 320 KiB/token (Lesson 24.1). An 8,192-token request holds $8192\cdot327{,}680=2{,}684{,}354{,}560$ bytes $=2.5$ GiB.
- *MHA counterfactual, same $L,d,H_q$*: $2\cdot80\cdot64\cdot128\cdot2=2{,}621{,}440$ bytes $=2{,}560$ KiB/token, which is $20$ GiB per 8,192-token request. K/V projections grow by $80\cdot2\cdot8192\cdot(8192-1024)=9{,}395{,}240{,}960$ weights (each of K and V widens from $1024$ to $8192$ outputs in each of 80 layers). The K/V biases grow by $80\cdot2\cdot(8192-1024)=1{,}146{,}880$, so with the added biases the model has $72{,}706{,}203{,}648+9{,}395{,}240{,}960+1{,}146{,}880=$ $82{,}102{,}591{,}488$ parameters, or $164.21$ GB $\approx152.93$ GiB in BF16.
- *Capacity screen*: synthetic 8 GPUs × 68 GiB usable for weights plus KV $=544$ GiB.
  - GQA KV pool: $544-135.43=408.57$ GiB, so $\lfloor408.57/2.5\rfloor=163$ requests at 8,192 tokens.
  - MHA KV pool: $544-152.93=391.07$ GiB, so $\lfloor391.07/20\rfloor=19$ requests.
  - The KV ratio is $8\times$; the capacity ratio is $163/19\approx8.6\times$, because the weights also changed.
  - Why more than $8\times$: the MHA weights are $17.5$ GiB larger, which shrinks its KV pool ($391.07<408.57$ GiB), and flooring ($\lfloor19.55\rfloor=19$ versus $\lfloor163.43\rfloor=163$) costs MHA a larger fraction.
- *MLA, real*: DeepSeek-V3 has $L=61$, $d_c=512$, $d_h^R=64$ (**O**, CLM-004).
  - $61\cdot576\cdot2=70{,}272$ bytes $=68.625$ KiB/token ($576=d_c+d_h^R=512+64$; $70{,}272/1024=68.625$). This matches the 70.272 KB (decimal) in Table 1 of the DeepSeek co-design paper (**O**, CLM-018).
  - An 8,192-token request holds $8192\cdot70{,}272/2^{20}=549$ MiB.
  - Trap: the config's `num_key_value_heads = 128` in the GQA formula gives $2\cdot61\cdot128\cdot128\cdot2=3{,}997{,}696$ bytes/token ($\approx3.81$ MiB), about $56.9\times$ too large ($3{,}997{,}696/70{,}272\approx56.9$).
- *TP replication, real Qwen2.5-72B at $t=16$*: each rank keeps $\max(1,\lfloor8/16\rfloor)=1$ KV head. Per rank: $2\cdot80\cdot1\cdot128\cdot2=40{,}960$ bytes/token. Across 16 ranks: $655{,}360$ bytes $=640$ KiB per token, twice the logical 320 KiB.
- *Interpretation / limits*: the counterfactual MHA model has no trained weights, so its capacity says nothing about quality. Page rounding, MLA alignment padding, packed FP8 MLA formats, and non-KV allocations change these numbers. Lab D reconciles them against the runtime's reported block count.

**Knowledge Check:**
1. Why is the runtime's maximum-concurrency number computed at `max_model_len` not the concurrency your workload achieves?
2. A team moves Qwen2.5-72B from TP=8 to TP=16 to cut latency. What happens to cluster KV bytes per token, and why?
3. Which vLLM symbol decides that MLA has no separate value state, and what would break if you sized MLA with `FullAttentionSpec`?

**Guided Practice (source trace, counted under `source_trace`):**
At commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`, trace:
1. `vllm/v1/worker/gpu_model_runner.py` `GPUModelRunner.get_kv_cache_spec`, to
2. `vllm/model_executor/layers/attention/attention.py` `Attention.get_kv_cache_spec` (full-attention branch), and `vllm/model_executor/layers/attention/mla_attention.py` `MLAAttention.get_kv_cache_spec`, to
3. `vllm/v1/kv_cache_interface.py` `AttentionSpec.state_content_size_bytes`, `MLAAttentionSpec`, and `_apply_alignment_padding`, to
4. `vllm/v1/core/kv_cache_utils.py` `_get_kv_cache_bytes_per_block`, `get_kv_cache_config_from_groups`, and `get_max_concurrency_for_kv_cache_config`.

Also open `vllm/model_executor/models/qwen2.py` `Qwen2Attention.__init__` for the TP rule. For each hop, record the inputs, the output, and which architecture field reaches it.

**Feedback Contract:**
- *Expected Evidence*: a hop table with file, symbol, inputs, and outputs; for Qwen2.5-72B at TP=8, state bytes per token per rank of $2\cdot80\cdot1\cdot128\cdot2$ (8 KV heads over 8 ranks) and why the page size multiplies it by `block_size`; for MLA, `head_size=576`, `num_kv_heads=1`, `head_size_v=0`, and the optional packed `state_content_bytes`; a statement that the trace is static.
- *Common Failure*: quoting the logical ratio as the runtime capacity ratio, or sizing MLA from `num_key_value_heads`.
- *Diagnostic Hint*: which rank-local head count reaches `FullAttentionSpec`, and does the model file split or copy KV heads at your TP degree?
- *Concept to Revisit*: logical versus physical KV (Module 03, Lesson 3.1); MLA retained state (Module 01, Lesson 1.7).

**Learning Outcome:**
Trace an attention variant from config fields to runtime pages and capacity metadata, and explain every factor between the logical KV ratio and the runtime's number.

*(Effort: 50m instruction. The guided source trace is counted once, under `source_trace`.)*

---

### Lesson 24.3 — Does the Bottleneck Stay or Move?

**Engineering Question:**
After an architecture change relieves one constraint, which constraint binds next, and what evidence distinguishes "the bottleneck moved" from "it stayed and only its level changed"?

**Concepts & Definitions:**
- **Normalized pressure** $p_r=D_r/C_r$: demand on resource $r$ divided by its attainable capacity at a declared boundary. The resources are KV capacity (resident sequences over the sequences that fit), HBM bandwidth (step bytes over $\beta\cdot T_{SLO}$), compute, network, scheduler limits (for example a configured maximum number of sequences), and host work.
- **Screening limiter**: $\arg\max_r p_r$ for a candidate operating point. It is a hypothesis about the active constraint, not a diagnosis (**D**, CLM-012).
- **Bottleneck transition**: the ordering of $p_r$ changes after a change in architecture, mapping, or workload. "Stays" means the same resource still has the largest pressure, possibly at a different level.
- **Static-label fallacy**: "decode is memory-bound" or "attention is the bottleneck" stated without a shape, batch, kernel, and topology. One recent study argues that reordered MLA raises decode attention intensity toward accelerator ridge points, and that MoE scale-out moves the binding constraint toward interconnect bandwidth and expert skew. Much of its multi-node evidence comes from a simulator calibrated against node-level runs (**O**, CLM-017).

**Mechanism Explanation:**
For a TPOT target $T_{SLO}$, each resource implies a maximum batch:
- KV capacity: $B_{KV}=\lfloor M_{KV}/(S\,k)\rfloor$.
- HBM bandwidth: $B_{BW}=\lfloor(\beta T_{SLO}-M_{weights})/(S\,k)\rfloor$, from the Lesson 24.1 screen.
- Compute: $B_{C}=\lfloor C\,T_{SLO}/(2P_{matmul}+4dSL)\rfloor$.
- Scheduler: the configured cap $B_{sched}$.

*Where:* $M_{KV}$ = bytes left for KV after weights and non-KV allocations; $k$ = KV bytes per token ($k_{KV}$ of Lesson 24.1); $S\,k$ = KV bytes held (and read per step) by one sequence at context $S$; $T_{SLO}$ = TPOT target in seconds.

*Step by step (each $B_r$ solves "demand at batch $B$ $\le$ capacity" for $B$):*
- **KV capacity**: $B\cdot S\,k\le M_{KV}$, so $B\le M_{KV}/(S\,k)$.
- **HBM bandwidth**: require the Lesson 24.1 bytes bound to fit the budget, $(M_{weights}+B\,S\,k)/\beta\le T_{SLO}$. Multiply by $\beta$: $M_{weights}+B\,S\,k\le\beta T_{SLO}$. Subtract $M_{weights}$ and divide by $S\,k$: $B\le(\beta T_{SLO}-M_{weights})/(S\,k)$. $\beta T_{SLO}$ is the byte budget of one step; weights are paid once per step whatever $B$ is.
- **Compute**: $B(2P_{matmul}+4dSL)/C\le T_{SLO}$, so $B\le C\,T_{SLO}/(2P_{matmul}+4dSL)$.
- Floor each, because batches are whole sequences.

The feasible batch is $\min(B_{KV},B_{BW},B_C,B_{sched},\dots)$, and the minimizer is the screening limiter. An architecture change moves several of these at once. GQA versus MHA changes both $k$ and $M_{weights}$. MoE changes $M_{weights}$, the per-token FLOPs, and adds a network term (Lesson 24.4). The bound uses attainable ceilings, so if the measured $\beta$ is lower, the bandwidth bound tightens (**O**, CLM-008).

**Quantitative Model / Derivation:**
$$B^*=\min_r B_r,\qquad r^*=\arg\min_r B_r$$
This matches the pressure definition: at a fixed batch $B$, resource $r$ hits its limit when $B=B_r$, and each $p_r$ grows with $B$. As $B$ rises, the first resource to reach $p_r=1$ is the one with the smallest $B_r$, so $r^*$ is the screening limiter of the definition above. (For KV and compute $p_r=B/B_r$ exactly before flooring; the bandwidth pressure is not proportional to $B$ because weight bytes are fixed.)
Goodput still requires queueing, TTFT, tail behavior, and quality. $B^*$ is a ceiling on resident decode concurrency, not a throughput prediction (**D**, CLM-014).

**Worked Example (Lesson 24.2 weights and KV; synthetic ceilings and scheduler cap):**
Inputs: $S=8192$, $T_{SLO}=40$ ms TPOT, $C=6.4\times10^{15}$ FLOP/s, a scheduler cap of $B_{sched}=128$, and $\beta\in\{24,12\}\times10^{12}$ B/s ("nameplate-like" versus a "measured sustainable" value, both synthetic).

| Candidate | $\beta$ | $B_{KV}$ | $B_{BW}$ | $B_{sched}$ | $r^*$ |
|---|---:|---:|---:|---:|---|
| MHA counterfactual | 24 TB/s | 19 | 37 | 128 | KV capacity |
| GQA (Qwen2.5-72B) | 24 TB/s | 163 | 303 | 128 | scheduler cap |
| GQA (Qwen2.5-72B) | 12 TB/s | 163 | 124 | 128 | HBM bandwidth |

- $B_{KV}$ comes from the Lesson 24.2 capacity screen: GQA $\lfloor408.57/2.5\rfloor=163$, MHA $\lfloor391.07/20\rfloor=19$ (GiB of pool over GiB per 8,192-token request).
- $B_{BW}$ for GQA at 24 TB/s: $(0.04\cdot24\times10^{12}-145{,}412{,}407{,}296)/(8192\cdot327{,}680)=303.46$, so 303.
  - Numerator: step byte budget $0.04\cdot24\times10^{12}=9.6\times10^{11}$ bytes minus $1.454\times10^{11}$ weight bytes $=8.146\times10^{11}$ bytes left for KV reads.
  - Denominator: $8192\cdot327{,}680=2{,}684{,}354{,}560$ bytes ($2.5$ GiB) per sequence.
- At 12 TB/s: $(4.8\times10^{11}-1.454\times10^{11})/2{,}684{,}354{,}560=124.64$, so 124. Halving $\beta$ more than halves $B_{BW}$, because the fixed weight bytes take a larger share of a smaller budget.
- MHA at 24 TB/s: $(0.04\cdot24\times10^{12}-164{,}205{,}182{,}976)/(8192\cdot2{,}621{,}440)=37.06$, so 37. ($164{,}205{,}182{,}976=2\cdot82{,}102{,}591{,}488$, the BF16 weight bytes from Lesson 24.2.)
- Compute is far from binding: at $B=128$ the GQA compute time is $128\cdot164{,}391{,}550{,}976/(6.4\times10^{15})=3.29$ ms of the 40 ms. Equivalently, $B_C=\lfloor6.4\times10^{15}\cdot0.04/164{,}391{,}550{,}976\rfloor=1{,}557$, far above every other $B_r$.
- *Result*: moving from MHA to GQA moves the screening limiter off KV capacity. Where it lands depends on two inputs that the attention variant does not set: the scheduler configuration and the attainable bandwidth (**D**, CLM-012).
- *Interpretation / limits*: an 8.6× KV-capacity gain yields at most $128/19\approx6.7\times$ resident batch with the cap, or $124/19\approx6.5\times$ at the lower bandwidth. Neither number is a throughput or cost ratio. The screen ignores TP communication, prefill interference, mixed context lengths, and kernel efficiency. Lab C tests it with a sweep.

**Knowledge Check:**
1. Raising $B_{sched}$ to 256 at $\beta=24$ TB/s: what is $r^*$ now?
2. Why does reducing KV bytes raise $B_{BW}$ as well as $B_{KV}$?
3. Name one measurement that would falsify "HBM bandwidth binds" at $\beta=12$ TB/s.

**Guided Practice:**
Repeat the table for DeepSeek-V3's MLA state (68.625 KiB/token) on a synthetic 16-GPU replica with 1,088 GiB usable for weights plus KV, $641.29$ GiB of checkpoint, and $\beta=48\times10^{12}$ B/s. Ignore communication first. Then state which missing term (Lesson 24.4) you expect to bind, and why the decode-step bytes screen alone cannot settle it.

**Feedback Contract:**
- *Expected Evidence*: $M_{KV}=1088-641.29=446.71$ GiB and $B_{KV}=\lfloor446.71\cdot2^{30}/(8192\cdot70{,}272)\rfloor$ (the $2^{30}$ converts GiB to bytes; this evaluates to $833$); a bandwidth bound that counts only selected expert weights per step *if* you assume batched expert reuse, stated as an assumption; a ranked hypothesis that network dispatch/combine or expert imbalance becomes the binding term; the measurement that would discriminate.
- *Common Failure*: counting all 671B parameters as bytes read per decode step, or declaring "MLA makes it compute-bound" from the cited study without reproduction.
- *Diagnostic Hint*: which expert weights does one decode step actually touch at your batch size, and over which links do the tokens travel?
- *Concept to Revisit*: shape-dependent regime transitions (Module 02, Lesson 2.5).

**Learning Outcome:**
Predict whether an architecture change keeps or moves the screening limiter, state the conditions that flip it, and design the measurement that tests it.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 24.4 — Dense versus MoE on a Real Topology

**Engineering Question:**
When does a sparse MoE's per-token compute saving survive the resident bytes, routing, and network traffic it adds on the topology you actually have?

**Concepts & Definitions:**
- **Total / resident / selected / executed parameters** (Module 01, Lesson 1.7). With $E$ experts and top-$k$ routing, the expert work selected per token scales with $k$. The checkpoint holds all $E$ experts, and every routed token travels to an expert owner and back (**D**, CLM-006).
- **Expert parallelism (EP) all-to-all, taught at first use**: experts live on different GPUs. In *dispatch*, each GPU sends each token's hidden state to the GPUs that own its selected experts. In *combine*, the expert outputs come back and are summed. The payload is proportional to tokens × selected experts × hidden size × bytes. The slowest receiver sets completion, so imbalance matters. Module 20 — Distributed Inference covers the collectives and placement in depth.
- **Topology tiers**: fast scale-up links within a node (for example NVLink) and slower scale-out links between nodes (for example InfiniBand).
- **Node-limited routing**: a routing constraint that keeps each token's experts on at most $M$ nodes. DeepSeek-V3 uses $M=4$. The authors motivate this as deduplicating slow inter-node traffic through faster intra-node forwarding (**O**, CLM-007).

**Mechanism Explanation:**
DeepSeek-V3 has 256 routed experts plus one shared expert, eight routed experts selected per token, 61 layers with MLA, and its first three FFN layers dense (**O**, CLM-004). Its reported deployment uses different layouts per phase:
- prefill: TP4/SP attention, DP8, EP32 on 32 GPUs;
- decode: TP4/SP, DP80, EP320 on 320 GPUs, with one expert per GPU and direct point-to-point all-to-all (**O**, CLM-005).

The co-design paper frames MLA, MoE, FP8, overlap schedules, node-limited routing, and a multi-plane network as one coupled design (**O**, CLM-018). The architecture fixed a communication pattern; the deployment chose phase-specific layouts to fit it.

**Quantitative Model / Derivation:**
Per MoE layer and decode step, as a payload-only screen:
$$Q_{a2a}=B\,k'\,d\,(b_{disp}+b_{comb}),\qquad T_{a2a}\ge\Gamma\frac{Q_{a2a}}{BW_{eff}}+\alpha$$
Here $k'$ counts selected experts including the shared one where it is treated as routed, $\Gamma\ge1$ is the max-over-receivers imbalance factor, and $\alpha$ collects latency terms. Also: $B$ = tokens per device in the step, $d$ = hidden size, $b_{disp}$ and $b_{comb}$ = bytes per element in the dispatch and combine directions, $BW_{eff}$ = effective per-device link bandwidth (bytes/s).

*Step by step:* each token's hidden state ($d$ elements) goes to $k'$ experts at $b_{disp}$ bytes per element ($k'\,d\,b_{disp}$ bytes), and $k'$ outputs of $d$ elements return at $b_{comb}$ bytes ($k'\,d\,b_{comb}$). Over $B$ tokens: $B\,k'\,d\,(b_{disp}+b_{comb})$. Time is bytes over bandwidth; the most loaded receiver gets $\Gamma$ times the average share, so the collective finishes $\Gamma$ times later, plus fixed latency $\alpha$. Check of the paper's figure: $32\cdot9\cdot7000\cdot3/(50\times10^9)\ \text{s}=120.96\ \mu$s (so "$d\approx7$K" is used as $7000$), and $14.76\text{ ms}/(61\cdot120.96\ \mu\text{s})=2$, so that bound counts twice the pair time per layer (the reason for the factor 2 is in the paper's Section 2.3.2, not restated here). The worked example below uses one pair per layer. The DeepSeek co-design paper uses this form, with FP8 dispatch (1 byte), BF16 combine (2 bytes), $B=32$, $k'=9$, $d\approx7$K, and 50 GB/s. It obtains $120.96\ \mu$s per all-to-all pair and a best-case $14.76$ ms TPOT bound over 61 layers, explicitly excluding latency and inefficiency (**O**, CLM-018). The 2025–2026 bottleneck study models the same term with an explicit imbalance factor (**O**, CLM-017).

**Worked Example (DeepSeek-V3 config; synthetic bandwidth and imbalance):**
- *Input*: $d=7168$, $k'=9$, $B=32$ tokens per device, 1 + 2 bytes, $BW_{eff}=40$ GB/s (the paper's effective inter-node figure, used here as a synthetic input), 58 MoE layers ($61-3$ dense), no overlap.
- *Step 1 — payload*: $32\cdot9\cdot7168\cdot3=6{,}193{,}152$ bytes $=5.906$ MiB per layer ($3=b_{disp}+b_{comb}=1+2$; $6{,}193{,}152/2^{20}=5.906$).
- *Step 2 — time*: $6{,}193{,}152/(40\times10^9)\ \text{s}=154.83\ \mu$s at $\Gamma=1$; $1.5\cdot154.83=232.24\ \mu$s at $\Gamma=1.5$ ($\alpha$ set to 0).
- *Step 3 — per decode step*: $58\cdot154.83\ \mu\text{s}=8.98$ ms, or $58\cdot232.24\ \mu\text{s}=13.47$ ms at $\Gamma=1.5$, of communication alone if nothing overlaps.
- *Step 4 — resident versus selected*: one routed expert has $3\cdot7168\cdot2048=44{,}040{,}192$ parameters (a SwiGLU expert: three $d\times m_e$ matrices with expert width $m_e=2048$). Over 58 layers the routed experts hold $256\cdot58\cdot44{,}040{,}192=653{,}908{,}770{,}816$ parameters, while a token selects $9\cdot58\cdot44{,}040{,}192=22{,}988{,}980{,}224$ expert parameters (8 routed + 1 shared, the shared expert assumed the same size as one routed expert).
  - Ratio: selected/resident routed-expert parameters $=9/256\approx3.5\%$ per token, so expert FLOPs scale with $9$ while resident bytes scale with $256$.
- *Step 5 — node-limited dispatch*: in a deduplicated scheme with one inter-node copy per destination node, FP8 inter-node dispatch per token drops from $8\cdot7168=57{,}344$ bytes (8 nodes) to at most $4\cdot7168=28{,}672$ bytes (4 nodes) (**O**, CLM-007).
- *Result*: on this synthetic link, all-to-all alone takes 9–13 ms per decode step. A 40 ms TPOT budget leaves room, but a 15 ms interactive budget does not unless communication overlaps with compute or the effective bandwidth rises.
- *Interpretation / limits*: payload only; ignores $\alpha$, kernel launch, intra-node forwarding, overlap (which DeepSeek uses), and shared-expert placement. The deployment layouts are author-reported, not defaults (**O**, CLM-005).

**Knowledge Check:**
1. Why does halving the active parameters not halve the GPU count for an MoE model?
2. Which term in $T_{a2a}$ does node-limited routing change, and which does expert-load rebalancing change?
3. Why can the same MoE model prefer TP at low concurrency and EP at high concurrency?

**Guided Practice:**
Recompute Steps 2–3 for an intra-node-only EP group at $BW_{eff}=160$ GB/s (synthetic). Then state the batch size at which compute per expert, not communication, would dominate under a Roofline screen with your declared ceilings.

**Feedback Contract:**
- *Expected Evidence*: $6{,}193{,}152/(160\times10^9)=38.71\ \mu$s per layer and $\approx2.25$ ms per step at $\Gamma=1$; an explicit compute model for one expert's batched GEMM, with its ceilings labeled synthetic; a statement that the crossover depends on expert-load skew.
- *Common Failure*: using total parameters as per-token FLOPs, or omitting the combine direction.
- *Diagnostic Hint*: how many bytes does one token send, to how many owners, in each direction?
- *Concept to Revisit*: MoE accounting (Module 01, Lesson 1.7); collectives and expert parallelism (Module 20 — Distributed Inference).

**Learning Outcome:**
Compare dense and MoE candidates on a declared topology by computing resident bytes, selected work, and all-to-all time, and identify which routing or placement constraint changes the outcome.

*(Effort: 50m instruction, 15m practice)*

---

### Lesson 24.5 — Matched Workloads, One Gate, Conditional Cost

**Engineering Question:**
How do we compare two architectures so that the measured difference belongs to the architecture, and how do we turn that difference into a cost statement that is honest about its conditions?

**Concepts & Definitions:**
- **Matched workload**: the same request population (prompt and output length distributions, prefix structure, tenant mix), the same arrival process (open-loop for capacity), the same SLOs, the same runtime revision and flags, and the same failure treatment for every candidate (**D**, CLM-013).
- **Shared quality and system gate**: one versioned contract per comparison. It contains paired quality metrics with noninferiority margins (aggregate and critical slices, Module 15, Lesson 15.4), output-length or behavior checks, latency predicates, reliability predicates, and failure counting. A candidate that fails any part leaves the comparison. MLPerf Inference is one public example of a contract that joins a quality target with a loaded-latency constraint. Its Server scenario uses Poisson arrivals, a benchmark-specific 99th-percentile latency constraint, and reports the maximum supported arrival rate (**O**, CLM-019).
- **SLO-goodput** $g$: successful requests per second, per replica, that meet every predicate on the matched workload (**D**, CLM-014).
- **Conditional cost**: cost per successful request given $g$, the replica count, utilization, dated prices, and a declared cost boundary (**D**, CLM-015).

**Mechanism Explanation:**
1. Freeze the contract.
2. Run quality evaluation on each candidate, paired by prompt.
3. Discard failing candidates.
4. Run open-loop load sweeps and report $g$ with an interval across seeded runs.
5. Size replicas from a *conservative* $g$ (for example the interval's lower bound).
6. Cost the replicas with dated prices.

The KV ratio enters only through $g$. If KV capacity was not binding (Lesson 24.3), it may not change $g$ at all (**D**, CLM-015).

**Quantitative Model / Derivation:**
$$n=\left\lceil\frac{\lambda_{success}}{g_{cons}}\right\rceil,\qquad c_{success}=\frac{n\cdot p_{replica\text{-}hour}}{3600\,\lambda_{success}}$$
*Where:* $\lambda_{success}$ = target successful requests per second; $g_{cons}$ = conservative per-replica goodput (req/s); $n$ = replicas; $p_{replica\text{-}hour}$ = price of one replica for one hour; $\lceil\cdot\rceil$ = round up (a fraction of a replica cannot be bought).
*Step by step:* $n$ replicas cost $n\,p_{replica\text{-}hour}$ per hour, and in one hour they serve $3600\,\lambda_{success}$ successes. Cost per success is the ratio; multiply by $10^6$ for cost per million successes.
Report $c$ at both ends of the $g$ interval. Add fixed costs and migration cost only inside the declared boundary. Full cost accounting is Module 22 — AI Economics.

**Worked Example (all inputs synthetic):**
- *Contract*: target $\lambda_{success}=40$ req/s. Quality: paired noninferiority, margin $\tau=1.0$ point, with the long-context slice gated separately at $\tau_s=1.0$. Latency: P95 TTFT and P95 TPOT thresholds fixed in advance. 8-GPU replica at \$24 per replica-hour (synthetic, dated to the exercise).
- *Gate*: GQA − MHA paired difference: aggregate interval $[-0.6,+0.3]$, long-context slice $[-0.9,+0.4]$. Both lower bounds are above $-1.0$, so GQA passes.
- *Measured $g$ intervals*: MHA $[1.9,2.1]$, GQA $[5.6,6.0]$ req/s per replica.
- *Replicas*: MHA $\lceil40/1.9\rceil=22$ to $\lceil40/2.1\rceil=20$; GQA $\lceil40/5.6\rceil=8$ to $\lceil40/6.0\rceil=7$.
  - Example: $40/1.9=21.05$, rounded up to 22 replicas; $40/5.6=7.14$, rounded up to 8.
- *Cost at the conservative end*: MHA $22\cdot24=\$528$/h, or $528/(3600\cdot40)\cdot10^6=\$3{,}666.67$ per million successes ($3600\cdot40=144{,}000$ successes per hour). GQA $8\cdot24=\$192$/h, or $192/144{,}000\cdot10^6=\$1{,}333.33$ per million.
- *Ratios*: the KV ratio is $8\times$; the conservative replica ratio is $22/8=2.75\times$. (The goodput ratio $5.6/1.9\approx2.95\times$ is the uncapped driver; integer rounding of $n$ turns it into $2.75\times$.)
- *Volume sensitivity*: at 730 h/month, the conservative saving is $(22-8)\cdot24\cdot730=\$245{,}280$/month at $\lambda=40$. At $\lambda=5$ it is $(3-1)\cdot24\cdot730=\$35{,}040$/month ($\lceil5/1.9\rceil=3$ MHA versus $\lceil5/5.6\rceil=1$ GQA replica). A synthetic one-time migration cost of \$150,000 pays back in $150{,}000/245{,}280=0.61$ months at $\lambda=40$ and $150{,}000/35{,}040=4.28$ months at $\lambda=5$.
- *Result*: "GQA cuts infrastructure cost per success by about $2.75\times$ *for this workload, gate, measured goodput, and price*." The conditions are part of the result.
- *Interpretation / limits*: the $8\times$ KV ratio did not become an $8\times$ cost ratio. Goodput gained less, and integer replicas and volume shape the saving. All numbers are synthetic; your contract must supply measured $g$ and dated prices (**D**, CLM-015).

**Knowledge Check:**
1. Why must candidates that fail the quality gate be removed *before* cost comparison, not penalized afterwards?
2. Why size from the lower bound of $g$ rather than its mean?
3. What changes if the MHA baseline also gets FP8 weights but GQA does not?

**Guided Practice:**
Rewrite the worked example for a contract where the long-context slice interval is $[-1.4,-0.2]$ with $\tau_s=1.0$. State the decision, what you would measure next, and what you would *not* report.

**Feedback Contract:**
- *Expected Evidence*: GQA fails the slice gate (lower bound $-1.4<-1.0$), so no cost comparison is reported for it. A next step: a slice-specific investigation, or a GQA checkpoint trained for long context, evaluated under the same contract. An explicit refusal to report the \$1,333 figure.
- *Common Failure*: reporting the cheaper candidate with a footnote about quality.
- *Diagnostic Hint*: is the quality predicate part of goodput, or an afterthought?
- *Concept to Revisit*: noninferiority and critical-slice gates (Module 15, Lesson 15.4); goodput (Module 04, Lesson 4.6).

**Learning Outcome:**
Run an architecture comparison under one quality and system contract and report cost as a conditional estimate with uncertainty, never as a ratio of KV bytes.

*(Effort: 50m instruction, 15m practice)*

---

### Lesson 24.6 — Sensitivity, Switching Thresholds, and the Defended Decision

**Engineering Question:**
How do we recommend an architecture when the inputs are uncertain, and what tells us when to reverse the recommendation?

**Concepts & Definitions:**
- **One-at-a-time sensitivity**: vary one input across its plausible range while holding the others fixed. This gives local elasticities and reversal points, but it can create impossible combinations (**D**, CLM-016).
- **Scenario analysis**: vary linked inputs together in coherent futures (for example, longer contexts *and* more document traffic *and* lower attainable bandwidth) (**D**, CLM-016).
- **Switching threshold**: the input value $x^\dagger$ where $\text{score}_A(x^\dagger)=\text{score}_B(x^\dagger)$. Above or below it, the recommendation flips.
- **Pareto dominance, taught at first use**: A dominates B when A passes every shared gate, is no worse on every declared constrained objective, and is better by a practically meaningful margin on at least one. Without dominance, the result is a workload-scoped trade-off (**H**, CLM-020).

**Mechanism Explanation:**
1. List the uncertain inputs and their ranges: goodput, interconnect bandwidth, workload mix, price, utilization, quality margins.
2. Compute the decision score for each candidate over the ranges.
3. Find the thresholds.
4. Check whether the current operating point is near any of them.
5. Turn every nearby threshold into a monitored guardrail with a rollback rule.

The DeepSeek case shows why topology inputs belong on this list: its architecture and deployment were tuned to a specific bandwidth hierarchy (**O**, CLM-018).

**Quantitative Model / Derivation:**
For candidates A and B, with continuous replicas as a first screen:
$$c_X(x)=\frac{p_X}{3600\,g_X(x)}\cdot10^6\ \text{per million successes},\qquad x^\dagger:\ c_A(x^\dagger)=c_B(x^\dagger)$$
Then recheck at integer replica counts and the conservative $g$ bounds.

*Where:* $p_X$ = price per replica-hour of candidate $X$; $g_X(x)$ = its per-replica goodput as a function of the uncertain input $x$. This is the Lesson 24.5 formula with $n=\lambda/g_X$ (no rounding), so $\lambda$ cancels.

*Step by step for the threshold:* $c_A=c_B$ means $p_A/g_A=p_B/g_B$, so the goodput B needs is $g_B^\dagger=g_A\cdot p_B/p_A$. If $g_B(x)$ is known, solve $g_B(x^\dagger)=g_B^\dagger$ for $x^\dagger$.

**Worked Example (all inputs synthetic):**
- *Candidates*:
  - A: dense GQA, 8-GPU replica, \$24/h, $g_A=5.6$ req/s.
  - B: MoE+MLA, 16-GPU replica across two nodes, \$48/h. Its goodput depends on the effective inter-node bandwidth $BW$ (GB/s) through a synthetic fitted line, $g_B=4.5+0.09\,BW$, valid on $BW\in[50,100]$.
- *Step 1*: $c_A=24/(3600\cdot5.6)\cdot10^6=\$1{,}190.48$ per million.
- *Step 2*: at $BW=50$, $g_B=4.5+0.09\cdot50=9.0$ and $c_B=48/(3600\cdot9.0)\cdot10^6=\$1{,}481.48$. At $BW=100$, $g_B=4.5+9.0=13.5$ and $c_B=48/(3600\cdot13.5)\cdot10^6=\$987.65$.
- *Step 3 — threshold*: $c_B=c_A$ requires $g_B=48\cdot5.6/24=11.2$ (B costs $2\times$ per hour, so it needs $2\times$ the goodput), so $4.5+0.09\,BW^\dagger=11.2$ gives $BW^\dagger=(11.2-4.5)/0.09=74.44$ GB/s. Check: $c_B(74.44)=48/(3600\cdot11.2)\cdot10^6=\$1{,}190.48=c_A$.
- *Step 4 — dominance*: B is cheaper only above 74.44 GB/s. If B also has a higher P99 TPOT, or a quality slice near its margin, neither candidate dominates.
- *Result*: "Prefer B only if sustained effective inter-node bandwidth stays above ~75 GB/s under production expert skew, and B passes the shared gate. Otherwise prefer A." Guardrail: monitor achieved all-to-all bandwidth and expert imbalance $\Gamma$, with rollback to A if a declared window falls below threshold.
- *Interpretation / limits*: the linear $g_B(BW)$ is a synthetic fit; a real one comes from Lab C sweeps with intervals. Integer replica rounding moves the threshold. The thresholds are not probabilities (**D**, CLM-016).

**Knowledge Check:**
1. Why can one-at-a-time sensitivity mislead when context length and document share rise together?
2. What evidence would falsify the claim "B dominates A" (CLM-020)?
3. Which guardrail metric would you page on, and which would you only review weekly?

**Guided Practice:**
Add a third uncertain input, the price ratio $p_B/p_A\in[1.6,2.4]$. Compute $BW^\dagger$ at both ends, and state which input the decision is more sensitive to over the declared ranges.

**Feedback Contract:**
- *Expected Evidence*: $g_B^\dagger=5.6\cdot(p_B/p_A)$, so $BW^\dagger=(8.96-4.5)/0.09=49.56$ GB/s at 1.6 and $(13.44-4.5)/0.09=99.33$ GB/s at 2.4. Both thresholds sit near the edges of the valid range, so the decision is highly price-sensitive; a stated recommendation conditional on the contracted price.
- *Common Failure*: reporting one cost number at mid-range inputs.
- *Diagnostic Hint*: where on the input range does the sign of $c_A-c_B$ change?
- *Concept to Revisit*: conditional cost (Lesson 24.5).

**Learning Outcome:**
Defend an architecture decision with sensitivities, switching thresholds, a dominance check, and rollback guardrails.

*(Effort: 50m instruction, 20m practice)*

---

## 05 Literature & Production Source Map

Every entry was opened on 2026-09-30 unless marked otherwise. *Scope* says what the module uses it for, and nothing more.

**REFERENCE / BASELINE**
- *Fast Transformer Decoding: One Write-Head is All You Need* (Shazeer, 2019). — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150). *Inspection*: abstract only. *Scope*: origin of MQA and the decode memory-bandwidth motivation; not used for numbers.
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., EMNLP 2023). — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245). *Inspection*: abstract only. *Scope*: GQA definition and author-reported quality/speed trade-off (CLM-001).
- *Roofline: An Insightful Visual Performance Model for Multicore Architectures* (Williams, Waterman, Patterson, CACM 2009). — [DOI](https://dl.acm.org/doi/10.1145/1498765.1498785). *Inspection*: citation metadata re-checked through Crossref; model content as taught in Module 02. *Scope*: step bounds (CLM-008).

**CURRENT DEFAULT (scoped to the named snapshot)**
- vLLM at commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad` (committed 2026-09-25). Statically inspected 2026-09-30; not executed. Files and symbols are listed in the Production Source Trace below (CLM-009, CLM-010, CLM-011, CLM-021). *Scope*: one upstream snapshot of one runtime; not an industry default.
- *Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures* (Zhao et al., ISCA 2025). — [arXiv:2505.09343](https://arxiv.org/abs/2505.09343). *Key sections*: 2.1.2 and Table 1 (KV per token), 2.3.2 (all-to-all TPOT bound), 4.2–4.3 (hardware-aware parallelism, node-limited routing), 5 (multi-plane network). *Scope*: primary co-design case study (CLM-018); its speed limits are analytical best cases.
- MLPerf Inference rules at `mlcommons/inference_policies` commit `ff7edba545fded369e7e7e3d5a2f0bab4a95eece`, `inference_rules.adoc`. *Scope*: an example of one contract joining quality and loaded-latency validity (CLM-019); not a production SLO.

**WORKLOAD-DEPENDENT**
- *DeepSeek-V2* (DeepSeek-AI, 2024). — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). *Key sections*: 2.1.2–2.1.4 (MLA, Table 1), 2.2.2 (device-limited routing), 3.2.3 (inference efficiency), Appendix D (MHA/GQA/MQA and MLA/MHA ablations). *Scope*: MLA state and quality ablations for its own models (CLM-003). The 93.3% KV reduction and 5.76× throughput are relative to its DeepSeek 67B baseline, with FP8 weights and 6-bit average KV quantization, and are not used as general multipliers.
- *DeepSeek-V3 Technical Report* (DeepSeek-AI, 2024). — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437). *Key sections*: 2.1.2 (node-limited routing), 3.4 (inference and deployment), 4.2 (hyper-parameters). *Scope*: architecture fields and author-reported phase layouts (CLM-004, CLM-005, CLM-007).
- Released configs: [Qwen2.5-72B-Instruct @ `495f39366efef23836d0cfae4fbe635880d2be31`](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/config.json) (CLM-022) and [DeepSeek-V3 @ `e815299b0bcbac849fa540c768ef21845365c9eb`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json) (CLM-004). *Scope*: exact fields for worked examples.
- *Mixtral of Experts* (Jiang et al., 2024). — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088). *Inspection*: abstract only. *Scope*: a second MoE comparison point (8 experts, top-2, 47B total / 13B active as stated by the authors) for practice; not used for claims.

**FRONTIER**
- *Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts* (Yun et al., 2025; v3, 2026-01-29). — [arXiv:2507.15465v3](https://arxiv.org/abs/2507.15465v3). *Key sections*: IV (MLA reordering, Observations 1–5), V (MoE batch limits and communication, Observations 6–8), VI (setup: DGX H100 node-level runs plus a calibrated simulator), VII. *Scope*: counterexample to static bottleneck labels (CLM-017); its multi-node results are simulation.
- *Rethinking Network Topologies for Cost-Effective Mixture-of-Experts LLM Serving* (Choi et al., 2026). — [arXiv:2605.00254](https://arxiv.org/abs/2605.00254). *Inspection*: abstract only. *Scope*: reading lead on topology cost-effectiveness for MoE; no claim relies on it (`TODO_VERIFY` before use).
- *Moebius: Serving Mixture-of-Expert Models with Seamless Runtime Parallelism Switch* (Wang et al., 2026). — [arXiv:2606.26607](https://arxiv.org/abs/2606.26607). *Inspection*: abstract only. *Scope*: reading lead on TP-versus-EP crossovers with concurrency; author-reported, unreproduced.
- *Coordinated Scheduling for MoE LLM Serving* (Sun et al., 2026). — [arXiv:2606.15177](https://arxiv.org/abs/2606.15177). *Inspection*: abstract only. *Scope*: reading lead on joint request and expert scheduling; no claim relies on it.

**LEGACY**
- Treating "MHA→GQA" as a pure KV-size switch with a fixed financial multiplier. It is rejected by Lessons 24.3 and 24.5: the KV ratio is an input to goodput, not a cost.
- Sizing every attention variant with $2LH_{kv}d_hb$. It is rejected for MLA by CLM-003 and CLM-010.

**Production Source Trace (pinned)**
- Repository: `vllm-project/vllm`, commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`, verified 2026-09-30, static inspection only.
- Path:
  1. `vllm/v1/worker/gpu_model_runner.py` `GPUModelRunner.get_kv_cache_spec`
  2. `vllm/model_executor/models/qwen2.py` `Qwen2Attention.__init__` (local KV heads under TP)
  3. `vllm/model_executor/layers/attention/attention.py` `Attention.get_kv_cache_spec` (`FullAttentionSpec`)
  4. `vllm/model_executor/layers/attention/mla_attention.py` `MLAAttention.__init__` and `MLAAttention.get_kv_cache_spec` (`MLAAttentionSpec`, `head_size = kv_lora_rank + qk_rope_head_dim`)
  5. `vllm/v1/kv_cache_interface.py` `AttentionSpec.state_content_size_bytes`, `AttentionSpec.unpadded_page_size_bytes`, `MLAAttentionSpec`, `_apply_alignment_padding`
  6. `vllm/v1/core/kv_cache_utils.py` `_get_kv_cache_bytes_per_block`, `get_kv_cache_config_from_groups`, `get_max_concurrency_for_kv_cache_config`
- *Generalizable*: PARTIAL. The architecture-to-page-size dependency is general; the symbols, the replication rule, and packed MLA formats are specific to this snapshot.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow the loop:
$$\text{PREDICT} \to \text{BUILD} \to \text{MEASURE} \to \text{EXPLAIN} \to \text{BREAK} \to \text{IMPROVE} \to \text{FALSIFY} \to \text{DEFEND}$$
Every lab writes rows under the same co-design contract (Lesson 24.5): manifest hashes, workload ID, gate version, runtime revision, seeds, and value class per number (**D**, CLM-013).

### LAB A — Architecture Manifest Compiler and Matched Workload
- **Objective**: Build a compiler from pinned config files to a labeled resource vector for GQA, a same-dimension MHA counterfactual, and MLA. Pair it with a matched workload generator and a shared gate definition file.
- **Pre-Registered Hypothesis**: For each config, the compiler reproduces the published parameter total exactly. For the MHA counterfactual, the screening KV-capacity ratio exceeds the logical KV ratio, because weights grow too (Lesson 24.2). State the predicted ratio before running.
- **Independent Variables**: Config (Qwen2.5-72B, Qwen2.5-7B, DeepSeek-V3, MHA counterfactual), context length $S\in\{2048,8192,32768\}$, dtype, TP degree $t\in\{4,8,16\}$.
- **Dependent Variables**: Parameter totals and their difference from the published total; logical and physical (TP-replicated) KV bytes/token; screening $B_{KV}$; weight bytes in GB and GiB.
- **Break & Falsify**: Feed the compiler the DeepSeek-V3 config through the GQA path. The lab passes only if the compiler refuses or flags the MLA fields instead of returning $\approx3.81$ MiB/token. Also generate two "matched" workloads with different seeds: if their prompt/output length distributions differ beyond a declared tolerance, the generator is not matched.
- **Alignment**: Lessons 24.1, 24.2, and 24.5. Claims CLM-001, CLM-002, CLM-013, CLM-014, CLM-019, CLM-022.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB B — Dense versus MoE Topology Simulator
- **Objective**: Simulate decode steps for a dense GQA candidate and an MoE+MLA candidate on a declared two-tier topology (intra-node and inter-node links), including expert placement, top-$k$ routing with a node limit, Zipf-skewed expert popularity, and optional overlap.
- **Pre-Registered Hypothesis**: Under the declared synthetic topology and a 15 ms TPOT budget, MoE decode time is dominated by all-to-all at inter-node bandwidth below a threshold you predict from Lesson 24.4. A node limit of $M=4$ reduces inter-node bytes per token by at least $2\times$ relative to $M=8$ when experts are spread over 8 nodes.
- **Independent Variables**: Inter-node bandwidth, node limit $M\in\{2,4,8\}$, expert-skew exponent, tokens per device, overlap on/off, number of nodes.
- **Dependent Variables**: Per-layer and per-step communication time, imbalance factor $\Gamma$, compute time per expert, TPOT estimate, and the binding term per configuration.
- **Break & Falsify**: Increase expert skew until a single hot expert's owner saturates. The hypothesis "node-limited routing is sufficient" is falsified if the step is still communication-bound at $M=2$ because of $\Gamma$. Then add redundant hot-expert replicas (as in the DeepSeek-V3 deployment, CLM-005) and remeasure.
- **Alignment**: Lesson 24.4. Claims CLM-004, CLM-005, CLM-006, CLM-007, CLM-017, CLM-018.
- **Effort Estimate**: 2.5h implementation, 1h analysis (3.5h total).

### LAB C — Bottleneck Transition Sweep and Conditional Cost Sensitivity
- **Objective**: Sweep batch, context, bandwidth, and scheduler caps for the Lab A candidates, using a simulator (and, optionally, real GPU runs). Locate where the screening limiter changes, convert per-replica SLO-goodput intervals into conditional cost, and compute switching thresholds.
- **Pre-Registered Hypothesis**: For GQA at the Lesson 24.3 inputs, the limiter is the scheduler cap at $\beta=24$ TB/s and HBM bandwidth at $\beta=12$ TB/s. Measured goodput gains are smaller than the KV-capacity ratio. Declare the predicted switching bandwidth for the Lesson 24.6 decision before running.
- **Independent Variables**: Candidate, $\beta$, $B_{sched}$, context mix, arrival rate (open-loop Poisson), price, inter-node bandwidth.
- **Dependent Variables**: $B_r$ per resource, the observed limiter (from queue, step-time, and occupancy telemetry), P95/P99 TTFT and TPOT, SLO-goodput with intervals across seeds, replicas, cost per million successes, and switching thresholds.
- **Statistical Rigor**: Seeded independent runs, declared warm-up, intervals per Module 00 Lesson 0.5. Size from the conservative bound.
- **Break & Falsify**: Raise $B_{sched}$ until the next limiter appears. The Lesson 24.3 prediction is falsified if measured goodput rises past the predicted $B_{BW}$ ceiling at the lower bandwidth, which would mean the bandwidth screen or the boundary is wrong. Check the dominance hypothesis (CLM-020) across scenarios; one plausible scenario that reverses the decision without a guardrail falsifies the recommendation.
- **Alignment**: Lessons 24.3, 24.5, and 24.6. Claims CLM-008, CLM-012, CLM-014, CLM-015, CLM-016, CLM-017, CLM-019, CLM-020.
- **Effort Estimate**: 3h implementation, 1h analysis (4h total).

### LAB D — Runtime Cache-Path Reconciliation
- **Objective**: Using the Lesson 24.2 trace, predict vLLM's per-layer page bytes, block count, and maximum concurrency for Qwen2.5-72B (or Qwen2.5-7B on one GPU) at two TP degrees and two block sizes. Then reconcile the predictions with the runtime's logged values, statically or by execution.
- **Pre-Registered Hypothesis**: Predicted `num_blocks` matches the runtime within the difference explained by measured non-KV memory, and TP above the KV-head count doubles physical KV bytes per token (CLM-021).
- **Independent Variables**: Model, TP degree, `block_size`, `max_model_len`, KV dtype.
- **Dependent Variables**: Page bytes, bytes per block, `num_blocks`, reported maximum concurrency, measured available KV memory, and the residual between prediction and runtime.
- **Break & Falsify**: Predict for an MLA model with the uniform-attention formula and show the mismatch. If the runtime's number matches the uniform-attention formula, your trace of `MLAAttentionSpec` is wrong. Change `max_model_len` and confirm that the reported concurrency changes while the workload's achieved concurrency (from Lab C) does not.
- **Alignment**: Lesson 24.2 and the Section 09 source trace. Claims CLM-009, CLM-010, CLM-011, CLM-013, CLM-021, CLM-022.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total; the static trace itself is counted under `source_trace`).

---

## 07 Break / Incident Scenarios

### Incident 24.1: The KV Saving That Did Not Pay
- **Incident Symptoms** (synthetic):
  - A team migrated a chat service from an MHA 70B-class checkpoint to a GQA checkpoint of the same family (8 KV heads), trained by the vendor. They expected the "8× smaller KV" to cut serving cost by roughly that factor.
  - To meet a latency target they also moved from TP=8 to TP=16.
  - After two weeks, free KV blocks per replica are about 4× higher than before, but successful requests per replica-hour rose only 1.3×.
  - P99 TPOT is 8% worse.
  - Retries rose 6% on long-document conversations.
  - The finance dashboard shows cost per successful request down 18%, not the promised ~85%.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: at least four, for example:
     - (A) KV capacity was never the binding constraint. The scheduler cap or HBM bandwidth binds (Lesson 24.3).
     - (B) TP=16 with 8 KV heads replicates KV across ranks and adds all-reduce cost, so physical KV per token is twice the logical value (CLM-021) and per-step communication grew.
     - (C) A quality regression on long documents drives retries and longer outputs, lowering successful goodput.
     - (D) The workload drifted toward longer contexts after launch, so the before/after comparison is unmatched.
     - (E) Weight bandwidth, not KV bandwidth, dominates decode at the operating batch.
  2. *Rank by Initial Plausibility*: use the symptoms. Free blocks rising while goodput barely moves weakens a pure KV-capacity story. Worse TPOT after a TP change supports (B) or (E). Retries concentrated on one slice support (C).
  3. *Identify Missing Evidence*: scheduler running-sequence counts versus `max_num_seqs`; step-time breakdown (attention, MLP, all-reduce) from a profiler; physical KV bytes per token from the runtime's cache config; paired quality on the long-document slice under the Lesson 24.5 gate; prompt/output length histograms before and after; replica-hour accounting boundaries.
  4. *Design Discriminating Tests*:
     - Rerun the old and new checkpoints on one frozen matched workload at TP=8 and TP=16 (a 2×2 design).
     - Raise the scheduler cap on one canary.
     - Run the paired long-document evaluation.

     State the falsifier for each hypothesis. For example, (A) is falsified if raising the cap does not raise running sequences or goodput; (C) is falsified if the paired slice difference is inside the margin.
  5. *Execute Causal Diagnosis*: rank the supported mechanisms. Allow for interaction: for example, a scheduler cap plus TP replication can each explain part of the gap.
  6. *Prescribe Mitigation & Long-Term Intervention*:
     - Immediate: return to TP=8 if it meets the latency SLO, and raise the cap within the measured bandwidth bound.
     - Quality: gate the long-document slice and route it to a fallback until a fix passes.
     - Long-term: adopt the Lesson 24.5 contract for every architecture change.
  7. *Remeasurement & Post-Mortem Defense*: report SLO-goodput intervals, the limiter per configuration, and conditional cost per success under the frozen contract. Replace the "~85%" claim with a conditional estimate and its switching thresholds (CLM-015, CLM-016).

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem: Dense GQA versus MoE+MLA for a Mixed Chat and Document Service
You own the architecture decision for a service handling interactive chat and long-document Q&A. Two candidates have passed internal pre-screening:
- **Candidate A**: a dense GQA model with the Qwen2.5-72B-Instruct manifest, served on 8-GPU replicas.
- **Candidate B**: an MoE+MLA model with the DeepSeek-V3 manifest, served on 16-GPU replicas spanning two nodes.

The deliverables link every earlier module: accounting (01), device bounds (02), KV representation (03), scheduling and capacity (04), precision and speculation contracts (05), and the comparison statistics of Modules 00 and 15.

**Workload and System Fixture (SYNTHETIC — exercise assumptions, not measurements of any real model, GPU, network, or price):**

| Input | Fixture value |
|---|---|
| Traffic | Target 30 successful req/s. 70% chat with a mean of 2,400 retained tokens per request; 30% documents with a mean of 13,500 retained tokens per request. Mean mix 5,730 tokens. |
| Device memory | 76 GiB usable per GPU, 8 GiB non-KV + headroom per GPU, so 68 GiB per GPU for weights plus KV |
| A weights / KV | $145{,}412{,}407{,}296$ bytes ($\approx135.43$ GiB) BF16; 320 KiB/token; even sharding over 8 GPUs |
| B weights / KV | $688{,}574{,}839{,}360$ bytes ($\approx641.29$ GiB) as stored (FP8 experts, BF16/F32 remainder per Hub metadata); 68.625 KiB/token MLA logical; even sharding over 16 GPUs |
| SLOs | Chat P95 TTFT 800 ms, P95 TPOT 40 ms; documents P95 TTFT 6 s, P95 TPOT 60 ms |
| Quality gate | Paired noninferiority versus the current production model, $\tau=1.0$ aggregate, $\tau_s=1.0$ for the long-document slice. A: aggregate $[-0.4,+0.5]$, slice $[-0.8,+0.3]$. B: aggregate $[-0.2,+0.9]$, slice $[-1.8,+0.6]$. |
| Measured SLO-goodput per replica | A: $[1.4,1.7]$ req/s. B: $[3.0,4.6]$ req/s, measured at 60 GB/s effective inter-node bandwidth |
| Prices | A replica \$24/h; B replica \$48/h (dated to the exercise) |
| Unknown by design | B's goodput at other bandwidths, expert-skew distribution in production, speculative-decoding acceptance for each model. Carry these as symbols or measure them. |

You may replace any fixture value with your own measurement, recording model, hardware, runtime revision, configuration, and method. Answers are graded conditionally on the inputs you declare. There is no single correct architecture.

**Required Deliverables**:
1. **Manifest and resource vector**: for both candidates, compute weight bytes (GB and GiB), KV bytes/token, per-request KV for chat and documents, the KV pool per replica, and selected versus total parameters for B. Label every number's value class.
2. **Gate before cost**: apply the shared quality gate. State which candidate passes, which fails, and what you are therefore allowed to report.
3. **Bottleneck prediction**: for each candidate, predict the screening limiter at the fixture's operating point (KV capacity, HBM bandwidth, compute, scheduler, or all-to-all), using Lessons 24.3–24.4. State the measurement that would falsify each prediction.
4. **Topology analysis for B**: compute the all-to-all screen per decode step at the fixture bandwidth with $\Gamma\in\{1,1.5\}$. State whether node-limited routing and overlap are needed to meet the 40 ms chat TPOT.
5. **Conditional cost**: compute replicas and cost per million successes at both ends of each passing candidate's goodput interval. Show why the KV ratio between A and B does not determine this.
6. **Sensitivity and thresholds**: identify the two inputs your decision is most sensitive to, and compute at least one switching threshold.
7. **Interventions that change the comparison**: propose two changes that could alter the outcome, for example a long-context-trained B checkpoint, FP8 weights for A under a Module 05 quantization contract, or speculative decoding with a declared distribution contract. For each, state which gate it must re-pass and which limiter it targets.
8. **Decision, uncertainty, and rollback**: give a recommendation, the guardrail metrics, and the rollback trigger. State explicitly what you are *not* claiming, for example any exact financial impact.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Submit a static (and, if possible, executed) trace of vLLM commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad` covering:
1. Entry at `GPUModelRunner.get_kv_cache_spec` and how each layer's spec is collected.
2. `Qwen2Attention.__init__` local KV-head computation under TP, with the replication branch.
3. `Attention.get_kv_cache_spec` producing `FullAttentionSpec`, and `AttentionSpec.state_content_size_bytes` and page size.
4. `MLAAttention.get_kv_cache_spec` producing `MLAAttentionSpec`, including `head_size`, `head_size_v=0`, packed `state_content_bytes`, and `_apply_alignment_padding`.
5. `_get_kv_cache_bytes_per_block`, `get_kv_cache_config_from_groups`, and `get_max_concurrency_for_kv_cache_config`, with one numeric prediction reconciled against Lab D.
6. One place where the runtime's capacity number would mislead an architecture comparison, and why.

### Reference Checks for Mastery Deliverables 1, 2, and 5 (fixture inputs only)
Reviewers use these to check arithmetic, not to grade a design.
- **KV pool per replica**: A $8\cdot68-135.43=408.57$ GiB; B $16\cdot68-641.29=446.71$ GiB.
- **Per-request KV**:
  - A: chat $2400\cdot320$ KiB $=750$ MiB; documents $13{,}500\cdot320$ KiB $=4{,}218.75$ MiB.
  - B: chat $2400\cdot68.625$ KiB $=160.84$ MiB; documents $13{,}500\cdot68.625$ KiB $=904.72$ MiB (divide KiB by 1024 for MiB).
- **B parameters**: routed experts $653{,}908{,}770{,}816$ of the stored total; selected expert parameters per token $22{,}988{,}980{,}224$ (Lesson 24.4).
- **Gate**: A passes (aggregate $-0.4>-1.0$, slice $-0.8>-1.0$). B fails the long-document slice ($-1.8<-1.0$). B's cost may not be reported as a deployable comparison for the full service. A submission may still analyze B as "chat-only" if it declares a routing split and re-gates.
- **Cost**: A needs $\lceil30/1.4\rceil=22$ to $\lceil30/1.7\rceil=18$ replicas, which is \$528 to \$432 per hour, or \$4,888.89 to \$4,000.00 per million successes ($528/(3600\cdot30)\cdot10^6$ and $432/108{,}000\cdot10^6$). For reference only (B fails the gate): B needs 10 to 7 replicas, which is \$480 to \$336 per hour, or \$4,444.44 to \$3,111.11 per million. The intervals overlap, so even if B passed, neither would dominate on cost.
- **All-to-all for B at 60 GB/s**: $6{,}193{,}152/(60\times10^9)=103.22\ \mu$s per layer at $\Gamma=1$, or $5.99$ ms over 58 layers; $8.98$ ms at $\Gamma=1.5$.

### Rubric Dimensions
- **Conditional Quantitative Answers** (Deliverables 1, 4, 5):
  - *Insufficient*: reports a single cost or GPU count as "the answer", derives cost from the KV ratio, or mixes GB and GiB.
  - *Competent*: declares every input with its source, keeps units and value classes straight, and reports cost at both ends of the goodput interval.
  - *Strong*: also identifies the inputs where the answer flips, and the measurement that would resolve each.
- **Gate Discipline** (Deliverable 2):
  - *Insufficient*: compares cost for a candidate that fails a gate, or applies different gates to different candidates.
  - *Competent*: applies one shared gate and excludes failing candidates.
  - *Strong*: proposes a re-gated variant (routing split or retrained checkpoint) and states exactly what must be re-measured.
- **Mechanistic Cross-Layer Reasoning** (Deliverables 3, 4):
  - *Insufficient*: lists vocabulary (GQA, MLA, MoE, EP) without connecting it to a constraint.
  - *Competent*: links each architecture field to the resource it changes and predicts a limiter.
  - *Strong*: states the conditions under which the limiter stays or moves, and ties them to topology and scheduler settings.
- **Experiment Design & Falsification** (Deliverables 3, 7; Labs):
  - *Insufficient*: unmatched comparisons, or no falsifier.
  - *Competent*: matched workload with a pre-registered hypothesis and a falsifier.
  - *Strong*: designs discriminating experiments (for example the Incident 2×2) that separate interacting causes.
- **Failure Diagnosis** (Incident 24.1):
  - *Insufficient*: names one cause from symptoms.
  - *Competent*: ranks multiple hypotheses with the missing evidence for each.
  - *Strong*: eliminates hypotheses systematically and accounts for interactions.
- **Source Reasoning** (Production Source Trace):
  - *Insufficient*: lists files without the data flow.
  - *Competent*: traces each hop with inputs and outputs at the pinned commit.
  - *Strong*: reconciles a numeric prediction against the runtime and explains the residual.
- **Architecture Decision** (Deliverables 6, 8):
  - *Insufficient*: recommends an architecture by reputation or peak throughput.
  - *Competent*: explains trade-offs with conditional numbers.
  - *Strong*: states switching thresholds, guardrails, rollback triggers, and what is not claimed.

---

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| **Architecture manifest → resource vector** | Lesson 24.1 | Lesson 24.1 Guided Practice; LAB A | Mastery Deliverable 1 | Manifest compiler output with value classes |
| **Attention variant through runtime cache path** | Lesson 24.2 | Lesson 24.2 source trace; LAB D | Section 09 Production Source Trace items 1–6; Mastery Deliverable 1 | Pinned trace plus Lab D reconciliation table |
| **Bottleneck stay/move prediction** | Lesson 24.3 | Lesson 24.3 Guided Practice; LAB C | Mastery Deliverable 3; Incident 24.1 steps 1–5 | Limiter table with falsifiers and sweep results |
| **Dense versus MoE on a topology** | Lesson 24.4 | Lesson 24.4 Guided Practice; LAB B | Mastery Deliverable 4 | Topology simulator report with $\Gamma$ and node-limit sweeps |
| **Matched workload and shared gate** | Lesson 24.5 | Lesson 24.5 Guided Practice; LABs A and C | Mastery Deliverable 2; Incident 24.1 step 4 | Versioned contract file plus paired gate results |
| **Conditional cost** | Lesson 24.5 | Lesson 24.5 Worked Example; LAB C | Mastery Deliverable 5 | Replica and cost table at interval bounds |
| **Sensitivity and defended decision** | Lesson 24.6 | Lesson 24.6 Guided Practice; LAB C | Mastery Deliverables 6–8; Incident 24.1 step 7 | Threshold analysis, guardrails, rollback rule |

---

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner completing Module 24 can:
1. Compile a pinned architecture manifest into a labeled resource vector and reproduce published parameter totals.
2. Trace GQA and MLA fields through a pinned runtime cache path and explain every factor between a logical KV ratio and runtime capacity.
3. Predict whether an architecture change keeps or moves the binding constraint, and test the prediction.
4. Compute resident, selected, and communication terms for dense versus MoE candidates on a declared topology.
5. Run a matched comparison under one quality and system gate and report conditional cost with uncertainty.
6. Defend a decision with switching thresholds, guardrails, and rollback triggers, and refuse to state an "exact financial impact".

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: an architecture change is a perturbation of a coupled system. Its effect on users and cost equals its effect on the binding constraint, under a fixed quality gate, for a declared workload.
- **The Chain**:
  $$\text{Manifest} \to \text{Resource Vector} \to \text{Runtime Mapping} \to \text{Limiter} \to \text{Service Curve} \to \text{Gated Goodput} \to \text{Conditional Cost} \to \text{Thresholds}.$$
- **Key Failures**: the ratio fallacy (FAIL-01), representation mismatch (FAIL-02), static bottleneck labels (FAIL-03), sparse-compute fallacy (FAIL-04), unmatched comparisons (FAIL-05), quality-blind optimization (FAIL-06), and point-estimate economics (FAIL-07).
- **Bridge**: Module 20 — Distributed Inference deepens the placement and communication side. Module 22 — AI Economics deepens the cost side. This module is the point where both meet the architecture.

## 12 Competency Targets

```yaml
competency:
  sfia: 5-6
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
