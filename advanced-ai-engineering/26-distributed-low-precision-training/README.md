# Module 26 — Distributed & Low-Precision Training

## 00 Why This Module Exists

A training configuration can fit every GPU and still run slower, fail more often, or converge differently. Sharding replaces resident state with transient gathers and collectives. Activation checkpointing replaces saved tensors with recomputation. TP, PP, CP, and EP replace local dependencies with communication and scheduling constraints. FP8, MXFP8, and NVFP4 replace one dtype choice with a recipe of formats, scales, accumulation, rounding, and high-precision islands. Checkpointing replaces failure loss with stalls, memory pressure, and persistence risk. These choices form one coupled system.

```text
model + sequence + optimizer + topology + failure trace
        |
        v
memory ledger: persistent state + activations + transient gathers + workspace + reserve
        |
        v
placement: DP/FSDP × TP/SP × PP × CP × EP --> collective and schedule ledger
        |
        v
timeline: compute | communication | exposed tail | pipeline bubble
        |
        v
precision recipe: encoding + scale timing/granularity + accumulation + high-precision islands
        |
        v
restart contract: snapshot --> stage --> persist --> verify --> restore
```

Module 01 provides parameter accounting and tensor shapes; Module 02 provides memory hierarchy and asynchronous timing; Module 05 provides quantization contracts; Module 19 provides single-device Adam state accounting; Module 20 provides collective and topology fundamentals; Module 25 owns scaling laws and optimization instability. This module owns distributed *training*: ZeRO/FSDP, activation checkpointing, TP/SP/PP/CP/EP training, communication overlap, low-precision training recipes, and restart/checkpoint economics. Inference parallelism stays in Module 20; router learning dynamics stay in Module 28.

**Research cutoff:** 2026-09-30. Sources were opened on 2026-10-12.

**Module Orientation**
- **Engineering Problem**: Build a training plan that fits, sustains throughput, preserves the intended numerical recipe, and recovers from interruption with bounded expected waste.
- **What You Will Do**: Reconcile memory and communication ledgers for ZeRO/FSDP; choose activation recomputation from tensor-level evidence; schedule TP/SP/PP/CP/EP forward and backward work; measure overlap from timelines; break FP8/MX/NVFP4 recipes; and select checkpoint intervals and asynchronous persistence controls from observed failure and storage behavior.
- **Environment**: Python 3.10+, NumPy and PyTorch. Hardware tiers are `CPU`, `COLAB-GPU` (one free/low-tier Colab GPU), and `TRACE` (pinned source or published-log replay). No local GPU or multi-node cluster is assumed.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. Author-reported results remain scoped to their configurations; synthetic inputs are labeled.

## 01 Baseline Assumptions

- Module 01: parameter accounting and tensor shapes (Lessons 1.5–1.6).
- Module 02: GPU memory hierarchy, asynchronous timing, and roofline reasoning (Lessons 2.1–2.6).
- Module 05: quantization contract—encoding, scale, rounding, accumulation, and accuracy boundary (Lesson 5.3).
- Module 19: single-device mixed-precision Adam states (Lesson 19.2).
- Module 20: distributed collectives, alpha-beta communication reasoning, rank meshes, and topology. Training-specific collective placement is taught here.
- Module 25: pretraining optimization and stability. This module measures distributed and numeric execution; it does not reteach scaling laws.
- Not assumed: ZeRO stages, FSDP1/FSDP2/DTensor, activation checkpoint placement, TP/SP/PP/CP/EP backward paths, overlap-tail analysis, FP8/MX/NVFP4 training recipes, or checkpoint-interval economics.

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
  economics: REQUIRED
  architecture_tradeoff: REQUIRED
  research_connection: REQUIRED

estimated_effort:
  instruction: 5h        # 6 lessons x 50m
  guided_practice: 2.5h  # 6 lessons x 25m
  labs: 13h              # LAB A 3h + LAB B 3.5h + LAB C 3.5h + LAB D 3h
  assessment: 3h
  source_trace: 1.5h     # counted once; shared by Lessons 26.1, 26.5, 26.6 and LAB D
  total: 25h
```

Learners must make a coupled plan, not select techniques independently. They must predict persistent and peak memory, collective bytes and latency exposure, pipeline bubbles, precision failure signals, and restart waste; then compare those predictions against CPU simulations, a Colab-scale run, traces, and pinned source.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Training Memory Ledger → ZeRO stages → FSDP1/FSDP2 state transitions → Activation Ledger → full versus selective recomputation → Training Parallelism (TP/SP, PP, CP, EP; forward and backward) → Topology Mapping → Buckets and Readiness → Overlap and Exposed Tail → Precision Recipe (E4M3/E5M2, delayed/current/block scaling, MXFP8, NVFP4, accumulation and high-precision islands) → Long-Horizon Validation → Failure Process → Blocking versus Asynchronous Checkpoint → Verified Durability and Restart Economics. Cross-cutting invariant: reducing one resident term usually creates compute, communication, temporary memory, numerical, or recovery work elsewhere.

## 04 Lessons

### Lesson 26.1 — ZeRO and FSDP: State, Materialization, and Communication

**Engineering Question:**
Which sharding stage makes the run fit, and what communication and transient memory does that choice create?

**Concepts & Definitions:**
- **Training memory ledger** separates persistent parameters, gradients, optimizer states, saved activations, communication buffers, workspaces, allocator reserve, and transient materialization.
- **ZeRO-1/2/3** shard optimizer state, then gradients, then parameters. Under ZeRO's mixed-precision Adam layout, the per-rank persistent model-state terms are $4\Psi+12\Psi/N$, $2\Psi+14\Psi/N$, and $16\Psi/N$ bytes respectively (**O**, CLM-001).
- **FSDP1** uses flattened parameter storage; **FSDP2** uses per-parameter DTensors and composable module hooks (**O**, CLM-004, CLM-005). The FSDP paper reports DDP-comparable performance and near-linear TFLOPS scaling only for its evaluated configurations (**O**, CLM-003).
- **Materialization window** is the period in which a full parameter group exists for compute. It can set peak memory even when steady state is sharded.
- **HSDP** shards within one mesh dimension and replicates across another, adding an all-reduce after reduce-scatter in the pinned path (**O**, CLM-004).

**Mechanism Explanation:**
At FSDP2 commit `38cca96300da024842405ecefa081e4761254922`, `fully_shard` installs state and hooks. A pre-forward hook unshards parameters through grouped all-gather; post-forward may reshard; pre-backward unshards again; post-backward reshares storage and reduce-scatters gradients. Optimizer states follow sharded parameters. Storage saved by stage 3 therefore coexists with transient gathered parameters and collective buffers.

**Quantitative Model / Derivation:**
For $N$ data-parallel ranks and $\Psi$ parameters under the cited 16-byte Adam layout:
$$M_1=4\Psi+12\Psi/N,\quad M_2=2\Psi+14\Psi/N,\quad M_3=16\Psi/N.$$
ZeRO's element-volume model gives $2\Psi$ communicated elements/step for stages 1–2 and $3\Psi$ for stage 3 (**O**, CLM-002). A ring collective over payload $S$ moves approximately $2(N-1)S/N$ bytes per rank (**D**, CLM-016). These are volume models, not elapsed-time predictions.

**Worked Example (derived; synthetic):**
A $10^9$-parameter model on $N=8$ ranks has persistent state predictions: stage 1, $4+12/8=5.5$ GB/rank; stage 2, $2+14/8=3.75$ GB/rank; stage 3, $16/8=2$ GB/rank. Add a synthetic 3.0 GB activation peak, 0.8 GB workspace/reserve allowance, and one 0.5 GB transient gather: stage 3 predicts 6.3 GB peak, not 2 GB. At two-byte communicated elements, the stage-3 ZeRO model is 6 GB/rank/step versus 4 GB for stages 1–2. Interpretation: stage 3 buys fit with more communication and materialization sensitivity. Limits: decimal GB, ideal partitioning, no padding or topology model.

**Knowledge Check:**
1. Why can stage 3 OOM although $16\Psi/N$ fits?
2. Which state first becomes sharded at ZeRO-1?
3. Why is communicated byte volume insufficient to predict step time?

**Guided Practice:**
Build a ledger for a synthetic 7B model across $N\in\{8,64\}$ and ZeRO stages 1–3. Add explicit activation, largest gather group, workspace, reserve, and padding terms. Predict the first OOM term and one topology-sensitive communication risk.

**Feedback Contract:**
- *Expected Evidence*: Per-term bytes, stage formulas, transient peak, collective payloads, and stated dtype/layout assumptions.
- *Common Failure*: Reporting only $16\Psi/N$ or treating FSDP2 as “ZeRO-3 with no implementation differences.”
- *Diagnostic Hint*: At the observed peak, which parameter group and communication buffer are simultaneously live?
- *Concept to Revisit*: Parameter Accounting (Module 01); Collective Cost (Module 20).

**Learning Outcome:**
Select and defend a ZeRO/FSDP plan from persistent, transient, and communication ledgers, and trace its runtime state transitions.

*(Effort: 50m instruction, 25m practice; source trace shared, 1.5h total)*

---

### Lesson 26.2 — Activation Checkpointing: Save, Recompute, or Select

**Engineering Question:**
Which forward tensors should survive until backward, and which should be recomputed?

**Concepts & Definitions:**
- **Activation checkpointing** saves boundaries and replays omitted forward operations during backward. Chen et al. give an $O(\sqrt n)$-memory schedule for $n$ layers at roughly one extra forward pass (**O**, CLM-006).
- **Full layer recomputation** stores a layer boundary and replays the whole layer.
- **Selective recomputation** preserves cheap/small tensors while replaying memory-heavy operations, such as attention-score paths, when recomputation cost is favorable. Korthikanti et al. report fivefold activation-memory reduction and over 90% lower recomputation overhead for their evaluated sequence-parallel selective policy; those factors do not transfer universally (**O**, CLM-008).
- **Pipeline in-flight multiplicity** multiplies saved activations by concurrent microbatches; it must be in the ledger.

**Mechanism Explanation:**
Forward ordinarily saves tensors needed by local backward formulas. A checkpoint boundary keeps only chosen inputs plus RNG/autocast state. During backward, the forward segment is re-executed to regenerate omitted tensors, then gradients are computed. Selective policies use tensor shape and operation cost: do not replay every GEMM merely because one $s\times s$ attention tensor dominates memory.

**Quantitative Model / Derivation:**
Under Korthikanti et al.'s 16-bit GPT-style assumptions, per-layer activation storage is
$$sbh(34+5as/h)\ \text{bytes},$$
where $s$ is sequence length, $b$ microbatch, $h$ hidden width, and $a$ attention heads. TP+SP divides the estimate by TP degree $t$; selective recomputation leaves about $34sbh/t$; full layer recomputation stores about $2sbh$ before optional partitioning (**O**, CLM-007). These formulas exclude allocator and kernel workspace.

**Worked Example (derived; synthetic):**
Let $s=4096$, $b=1$, $h=4096$, $a=32$, $t=8$. Then $5as/h=0.0391$, and baseline storage is about $571.1$ MB/layer before TP+SP, or $71.4$ MB/layer after division by 8. Selective recomputation is about $570.4/8=71.3$ MB/layer here because the chosen $a/h$ ratio makes the score term small; full checkpointing is $33.6$ MB/layer before partitioning. This counterexample shows the published symbolic ledger must be instantiated: selective recomputation does not always dominate. At longer $s$, the quadratic term grows and the ranking changes.

**Knowledge Check:**
1. What state must be preserved so dropout recomputation matches forward?
2. Why can activation memory rise when pipeline depth rises even if each layer is unchanged?
3. When does selective recomputation beat full-layer replay?

**Guided Practice:**
Evaluate the formula across $s\in\{2K,8K,32K\}$, $b\in\{1,4\}$, and $t\in\{1,8\}$. Choose a policy from predicted bytes and replay FLOPs. Mark formula terms that your tiny PyTorch measurement cannot observe directly.

**Feedback Contract:**
- *Expected Evidence*: Tensor-level ledger, replayed-op list, peak measurement, RNG correctness check, and one counterexample to the chosen policy.
- *Common Failure*: Saying “checkpoint every layer halves memory” without an activation inventory.
- *Diagnostic Hint*: Which saved tensor scales as $s^2$, and which operations recreate it?
- *Concept to Revisit*: Tensor Shapes (Module 01); Memory Hierarchy (Module 02).

**Learning Outcome:**
Choose activation saving and recomputation from tensor shapes, replay cost, in-flight multiplicity, and measured peak memory.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 26.3 — Training Parallelism: TP/SP, PP, CP, and EP

**Engineering Question:**
How should model, sequence, layers, context, and experts map onto ranks without hiding backward or synchronization costs?

**Concepts & Definitions:**
- **Tensor parallelism (TP)** shards linear algebra dimensions. Megatron's cited layout performs two all-reduces in forward and two in backward per Transformer layer (**O**, CLM-009).
- **Sequence parallelism (SP)** shards sequence-dimension activations around TP regions, replacing all-reduces with reduce-scatter/all-gather pairs of equal first-order bandwidth in the cited formulation (**O**, CLM-009).
- **Pipeline parallelism (PP)** assigns layers to stages and microbatches to a schedule. **1F1B** alternates forward/backward after warmup; **interleaving** gives each rank multiple virtual chunks.
- **Context parallelism (CP)** distributes one sequence. Ring Attention rotates K/V blocks while overlapping block attention; Llama 3 reports an alternative all-gather-K/V design for mask flexibility (**O**, CLM-013).
- **Expert parallelism (EP)** dispatches routed token-expert pairs and combines outputs with all-to-all; load skew exposes the slowest rank (**D**, CLM-014).

**Mechanism Explanation:**
Each dimension cuts one tensor axis or layer set and introduces matching forward and backward dependencies. TP collectives occur within layers. PP sends activations and gradients between stages. CP communicates K/V blocks or gathers them before local-Q attention. EP dispatches tokens before expert compute and combines results afterward. The rank mesh must map frequent/high-volume collectives to fast links and avoid multiplying incompatible dimensions beyond the world size.

**Quantitative Model / Derivation:**
For $p$ balanced stages and $m$ microbatches, ideal-compute-normalized 1F1B bubble is $\beta=(p-1)/m$; elapsed idle fraction is $\beta/(1+\beta)$. With $v$ virtual chunks, $\beta=(p-1)/(mv)$ and pipeline communication grows by $v$ (**O**, CLM-010). Zero Bubble splits backward into input-gradient $B$ and parameter-gradient $W$ work to fill idle slots; feasibility depends on measured $F/B/W$ times and memory (**O**, CLM-011).

**Worked Example (derived; synthetic):**
For $p=8$, $m=32$, 1F1B gives $\beta=7/32=0.21875$, so elapsed idle fraction is $0.21875/1.21875=17.95\%$. With $v=2$, $\beta=7/64=0.109375$, idle fraction is $9.86\%$, but pipeline communication doubles. If one stage takes 25% longer, the balanced-stage formula becomes optimistic; increasing $m$ cannot remove that bottleneck. A zero-bubble schedule may fill nominal gaps but can raise activation residency or alter optimizer synchronization. PACI instead trades exact synchronous semantics for bounded forward/backward weight-version drift through local accumulation; it remains a 2026 frontier preprint verified only at abstract level here (**O**, CLM-012).

**Knowledge Check:**
1. Why does SP preserve first-order bandwidth while reducing activation residency?
2. What does increasing $m$ fix, and what does it not fix?
3. Why are Ring Attention and all-gather-K/V both valid CP designs?

**Guided Practice:**
Given a synthetic 64-GPU topology (8 GPUs/node), build two factorizations for DP×TP×PP×CP×EP=64. Draw forward and backward collectives, place each process group, calculate pipeline bubble, and identify the most topology-sensitive edge.

**Feedback Contract:**
- *Expected Evidence*: Rank mesh, tensor axes, forward/backward collective table, PP schedule, bubble math, and imbalance sensitivity.
- *Common Failure*: Naming five parallel dimensions without showing group membership or backward traffic.
- *Diagnostic Hint*: Which collective crosses nodes most often, and which stage sets the clock?
- *Concept to Revisit*: Distributed Collectives and Topology (Module 20).

**Learning Outcome:**
Compose TP/SP/PP/CP/EP into a valid training mesh and defend it with forward/backward communication, schedule, memory, and topology evidence.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 26.4 — Communication Overlap and the Exposed Tail

**Engineering Question:**
Did communication become hidden, or did the profiler merely report concurrent work?

**Concepts & Definitions:**
- **Gradient bucket**: parameter gradients grouped for one collective. DDP hooks launch asynchronous all-reduce when a bucket becomes ready (**O**, CLM-015).
- **Readiness order**: backward order in which all gradients in each bucket become available.
- **Overlap window**: independent compute that can run while communication progresses.
- **Exposed tail**: dependency-critical communication remaining after useful compute ends.
- **Critical-path time** differs from the sum of kernel and collective durations; concurrent intervals must be unioned.

**Mechanism Explanation:**
Small buckets become ready earlier but pay more launch latency. Large buckets amortize launch overhead but delay their start. Reordering or unused parameters can delay an entire bucket. Communication can also slow concurrent compute through shared links, copy engines, memory bandwidth, or SM resources. The optimizer waits for the last required gradient, so the final bucket often defines the exposed tail.

**Quantitative Model / Derivation:**
For message size $S$, a first-order collective model is $T_{comm}\approx\alpha k+V/B$, with launch/step latency $\alpha$, algorithmic step count $k$, transferred bytes $V$, and achieved bandwidth $B$. Step time is the critical path, not $T_{compute}+\sum T_{comm}$. Useful overlap for a collective is bounded by independent compute after its readiness time; remaining duration is exposed (**D**, CLM-016).

**Worked Example (synthetic):**
Backward compute lasts 120 ms. Four buckets become ready at 30, 55, 80, and 110 ms; each collective lasts 25 ms without contention. First three finish by 105 ms and hide completely. The last finishes at 135 ms, exposing 15 ms. Aggregate communication duration is 100 ms, but exposed communication is 15 ms. Doubling bucket size produces two 42 ms collectives ready at 55 and 110 ms: the second finishes at 152 ms, exposing 32 ms despite lower aggregate duration (84 ms). Interpretation: aggregate collective time improved while step time worsened.

**Knowledge Check:**
1. Why can lower summed communication duration coincide with a slower step?
2. What evidence distinguishes contention from a late-ready bucket?
3. Why can a tiny final parameter create a long tail?

**Guided Practice:**
Use a supplied synthetic Chrome trace with compute and collective intervals. Compute readiness, completion, overlap, and exposed tail per bucket. Recommend one bucket reorder or size change and predict step time before applying it.

**Feedback Contract:**
- *Expected Evidence*: Time-aligned trace, readiness table, critical path, contention check, prediction, and remeasurement.
- *Common Failure*: Calculating “overlap percentage” from summed durations whose intervals overlap each other.
- *Diagnostic Hint*: Which event gates optimizer start, and when did its bucket become ready?
- *Concept to Revisit*: Asynchronous Timing (Module 02).

**Learning Outcome:**
Measure and tune communication overlap from readiness and dependency timelines rather than aggregate profiler totals.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 26.5 — FP8, MXFP8, and NVFP4 Are Recipes, Not Dtypes

**Engineering Question:**
Which tensors may use low precision, with which scale and accumulator, without hiding delayed instability?

**Concepts & Definitions:**
- **E4M3/E5M2** trade precision and range. The cited FP8 proposal gives maxima 448 and 57,344 and recommends E4M3 for weights/activations and E5M2 for gradients in its hybrid recipe (**O**, CLM-017).
- **Delayed scaling** derives a scale from prior amax history; **current scaling** derives it from current data; **block scaling** assigns scales to local groups. These change synchronization, staleness, outlier sensitivity, and metadata.
- **MXFP8** shares an E8M0 power-of-two scale across 32 values. Quantization axis matters, so quantize-then-transpose is not transpose-then-quantize (**O**, CLM-021).
- **NVFP4** uses E2M1 values with two-level 16-value scaling, Hadamard transforms, stochastic gradient rounding, 2D weight scaling, and selected higher-precision layers in the cited recipe (**O**, CLM-024).
- **High-precision islands** are operators or state retained in BF16/FP32: reductions, normalization, embeddings, selected layers, master weights, gradients, or optimizer state depending on recipe (**O**, CLM-018, CLM-020).

**Mechanism Explanation:**
A scale maps a high-precision tensor into a representable low-precision range. Delayed scales can lag distribution shifts; current scales need current amax; coarse scales let one outlier waste resolution for a whole tensor; blocks reduce that coupling but add metadata and transpose constraints. GEMMs may accept low-precision inputs while accumulating or outputting higher precision. The pinned Transformer Engine recipe objects make these choices explicit—delayed/current/block scaling, 32-value MXFP8 blocks, and NVFP4 controls—but configuration presence does not prove hardware support or accuracy (**O**, CLM-019). A 2025 MXFP8 study reports BF16-matched runs up to 8B parameters and 15T tokens with E4M3 and a specific shared-scale rounding rule; that result is recipe- and workload-scoped (**O**, CLM-022).

**Quantitative Model / Trade-off Comparison:**
For symmetric scale $s_q=x_{max}/q_{max}$, quantization is $q=\operatorname{round}(x/s_q)$ with clipping and reconstruction $\hat x=s_q q$. Track saturation rate, zero/underflow rate, amax/scale lag, block outlier ratio, accumulator growth, and loss delta against BF16. A successful short proxy does not bound long-run risk: Fishman et al. report prolonged FP8 instability tied to SwiGLU outlier amplification (**O**, CLM-023).

**Worked Example (synthetic):**
A 32-value block has 31 values near 1 and one outlier 128. Using a toy signed 4-bit maximum 7, one shared scale is $128/7=18.29$; all 31 values round to zero. If transformed or split so local maxima are 8, scale is $8/7=1.14$ and unit values survive near one code step. This is not an NVFP4 accuracy claim; it demonstrates why block size, outlier transforms, and scale hierarchy are part of the recipe. A delayed scale based on prior amax 8 would clip the new 128 outlier, while a current scale avoids clipping but coarsens every other value.

**Knowledge Check:**
1. Why does “FP8 GEMM” not imply FP8 optimizer state or accumulator?
2. Why can reusing an MX-quantized matrix transpose be wrong?
3. Which signals can warn before loss diverges?

**Guided Practice:**
On CPU, emulate E4M3-like and block-scaled quantization for synthetic normal, heavy-tail, and drifting tensors. Compare delayed/current scales and block sizes. On `COLAB-GPU`, optionally repeat with supported FP8 operations; otherwise replay published traces. Pre-register a saturation/underflow boundary that rejects the recipe.

**Feedback Contract:**
- *Expected Evidence*: Full recipe table, scale/amax traces, saturation and underflow, BF16 comparison, operator coverage, hardware/software versions, and rejection boundary.
- *Common Failure*: Reporting a format name and endpoint loss while omitting scale policy and accumulator.
- *Diagnostic Hint*: Did the distribution change before the scale, and which block contains the outlier?
- *Concept to Revisit*: Quantization Contract (Module 05).

**Learning Outcome:**
Specify, test, and falsify an FP8/MX/NVFP4 training recipe without conflating encoding, implementation availability, and convergence evidence.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 26.6 — Checkpoint and Restart Economics

**Engineering Question:**
How often should the run checkpoint, and when is a checkpoint actually durable?

**Concepts & Definitions:**
- **MTBF** $M$ is mean time between modeled failures. Correlated failures and maintenance violate an independent exponential model.
- **Checkpoint interval** $\tau$ is useful work between checkpoint starts; **blocking cost** $C$ is exposed checkpoint stall; **restart cost** $R$ includes detection, allocation, restore, and warm restart.
- **Staging** copies state to another memory tier. **Persistence** writes durable storage. A staging completion is not durability.
- **Asynchronous checkpointing** overlaps persistence with training, but consumes CPU memory and I/O and needs bounded in-flight work and verified completion (**O**, CLM-027, CLM-028).

**Mechanism Explanation:**
A blocking checkpoint pauses useful work, persists state, and resumes. A failure loses work since the last durable checkpoint plus restart time. Llama 3's 54-day 405B snapshot reports 466 interruptions (419 unexpected), showing why interruption telemetry matters while not supplying a universal failure distribution (**O**, CLM-025). Asynchronous save shortens the exposed stall by staging quickly and persisting later. CheckFreq reports bounded 3.5% runtime overhead and recovery reduction from hours to seconds only on its evaluated workloads (**O**, CLM-027); TorchTitan reports a setup-specific 5–15× checkpoint-overhead reduction (**O**, CLM-029). If writes lag the checkpoint cadence, queued snapshots consume resources and the latest completed durable checkpoint becomes older than the staged snapshot. At the pinned PyTorch commit, `async_save` exposes separate `staging_completion` and `upload_completion` futures (**O**, CLM-028).

**Quantitative Model / Derivation:**
Under independent exponential failures and first-order assumptions,
$$W(\tau)\approx C/\tau+\tau/(2M)+R/M.$$
Differentiating gives $\tau^*=\sqrt{2CM}$ (**D**, CLM-030). Young's commonly stated first-order result is used only as a planning approximation; its primary-source statement and exact Daly higher-order variants remain `TODO_VERIFY` because publisher full text was inaccessible (CLM-026).

**Worked Example (derived; synthetic):**
Let $C=120$ s, $M=24$ h $=86{,}400$ s, and $R=600$ s. $\tau^*=\sqrt{2\cdot120\cdot86{,}400}=4{,}553.7$ s = 75.9 min. Waste is $120/4553.7 + 4553.7/(172800)+600/86400=5.97\%$. If asynchronous staging reduces exposed $C$ to 15 s and persistence keeps up, $\tau^*=1{,}609.9$ s=26.8 min and waste is $2.56\%$. If upload needs 45 min, a 26.8-min cadence accumulates writes; the assumption fails and the policy needs backpressure or a longer interval.

**Knowledge Check:**
1. Why does reducing $R$ not change the first-order stationary point?
2. Which future proves durability in the pinned API?
3. Why is $54$ days divided by interruption count not automatically an exponential MTBF estimate?

**Guided Practice:**
Fit empirical intervals from a supplied synthetic interruption log; compare exponential and bursty traces. Sweep $C$, $R$, persistence bandwidth, and maximum in-flight snapshots. Choose a policy with a durability SLO and identify where Young's assumptions break.

**Feedback Contract:**
- *Expected Evidence*: Failure-trace summary, units, waste curve, chosen interval, staging/persistence timeline, completion checks, backpressure, and restore drill.
- *Common Failure*: Marking a staged snapshot complete or copying another run's interval.
- *Diagnostic Hint*: Which checkpoint can be restored after killing the process and storage worker now?
- *Concept to Revisit*: Experimental Units (Module 00); Storage/Failure Telemetry.

**Learning Outcome:**
Choose and validate checkpoint cadence and asynchronous persistence from observed failures, exposed stalls, storage throughput, and verified restoration.

*(Effort: 50m instruction, 25m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [ZeRO](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., 2019; sharded-state memory and bandwidth accounting.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Chen et al., 2016; activation checkpoint schedules.
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019; tensor parallelism.
- [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021; 1F1B and interleaving.
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al., 2022; FP8 encodings and mixed-precision recipe.
- Young (1974), DOI `10.1145/361147.361115`; first-order checkpoint interval. Daly's exact higher-order formula is not quoted.

**CURRENT DEFAULT**
- FSDP/ZeRO-style sharding chosen from a full memory and collective ledger; activation recomputation selected from tensor evidence; topology-aware multidimensional meshes; timeline-based overlap measurement; mixed precision specified as a full recipe; checkpoint completion verified by restore. “Current default” is an engineering baseline, not a universal adoption claim.

**WORKLOAD-DEPENDENT**
- FSDP1 versus FSDP2; full versus selective recomputation; TP/SP/PP/CP/EP degree; 1F1B versus interleaving; Ring Attention versus K/V all-gather; bucket size; delayed versus current/block scaling; blocking versus asynchronous checkpointing.

**FRONTIER**
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) — Qi et al., 2024.
- [PACI](https://arxiv.org/abs/2606.07881) — Elam et al., 2026; asynchronous pipeline with bounded weight inconsistency; abstract-level verification.
- [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027) — Mishra et al., 2025.
- [Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517) — Fishman et al., ICLR 2025; long-horizon instability.
- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) — NVIDIA, 2026; 12B/10T author-reported recipe.

**LEGACY / INSUFFICIENT**
- Dividing $16\Psi$ by ranks and calling that peak memory; treating all collective bytes as exposed time; selecting a parallelism degree without a backward path; claiming “FP8 training” from dtype alone; treating staging completion as durable; importing a checkpoint interval from unrelated failure data.

**PRODUCTION SOURCE TRACES (static inspection, verified 2026-10-12)**
1. `pytorch/pytorch` commit `38cca96300da024842405ecefa081e4761254922`:
   - `torch/distributed/fsdp/_fully_shard/_fully_shard.py::fully_shard`
   - `torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py::{FSDPParamGroup.unshard, reshard, pre_forward, post_forward, pre_backward, post_backward}`
   - `torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py::{foreach_all_gather, foreach_reduce}`
   - `torch/distributed/checkpoint/state_dict_saver.py::{AsyncSaveResponse, async_save}`
2. `NVIDIA/TransformerEngine` commit `ef2bf753080c088aa3fb3f1adb3d49713272c67a`:
   - `transformer_engine/common/recipe/__init__.py::{Format, DelayedScaling, Float8CurrentScaling, MXFP8BlockScaling, Float8BlockScaling, NVFP4BlockScaling}`
   - `transformer_engine/common/recipe/delayed_scaling.cu::amax_and_scale_update_impl::kernel`
   - `transformer_engine/common/recipe/fp8_block_scaling.cu::kTileDim`
Runtime support and numerical results are not inferred from configuration presence.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs use `PREDICT → BUILD → MEASURE → BREAK → EXPLAIN → IMPROVE → FALSIFY → DEFEND`. Labs use synthetic inputs unless a public trace is named. This module supplies descriptions only; no lab files are created.

### LAB A — Memory Ledger and Recompute Boundary
- **Hardware**: `CPU` for ledger/simulator; optional `COLAB-GPU` for tiny measured peaks.
- **Objective**: Predict persistent and peak memory for DDP, ZeRO-1/2/3, FSDP1/FSDP2, and activation policies; reconcile predictions with a tiny run or synthetic allocation trace.
- **Pre-Registered Hypothesis**: A ledger including largest transient gather, activation peak, workspace, and reserve predicts the first OOM boundary more accurately than persistent model-state bytes alone.
- **Independent Variables**: Sharding stage/API, wrap-group size, sequence length, microbatch, checkpoint policy, TP/SP degree.
- **Dependent Variables**: Persistent bytes, peak allocated/reserved, transient gather, replayed FLOPs, collective bytes, prediction error.
- **Break & Falsify**: Inflate one wrap group; increase sequence length; enable checkpointing while increasing in-flight microbatches. Falsify if the full ledger does not improve boundary prediction against the state-only baseline.
- **Alignment**: Lessons 26.1–26.2.
- **Effort Estimate**: 3h total.

### LAB B — Parallelism and Overlap Trace Laboratory
- **Hardware**: `CPU` discrete-event simulation plus `TRACE` replay.
- **Objective**: Compose TP/SP/PP/CP/EP schedules, calculate pipeline bubbles and collective volumes, and diagnose exposed communication from a timeline.
- **Pre-Registered Hypothesis**: The best configuration by aggregate communication duration differs from the best configuration by critical-path step time when bucket readiness, stage imbalance, or topology is introduced.
- **Independent Variables**: Mesh factorization, $p,m,v$, stage imbalance, bucket size/order, CP design, EP skew, link bandwidth/latency.
- **Dependent Variables**: Step time, bubble, collective bytes, overlap, exposed tail, slowest expert rank.
- **Break & Falsify**: Delay the final bucket; move TP across nodes; skew EP assignments; slow one PP stage. Falsify if aggregate communication ranking still predicts every critical-path ranking.
- **Alignment**: Lessons 26.3–26.4.
- **Effort Estimate**: 3.5h total.

### LAB C — Low-Precision Recipe Breaker
- **Hardware**: `CPU` emulation and `COLAB-GPU` only when supported; otherwise `TRACE`.
- **Objective**: Compare BF16 reference behavior against delayed/current/block scaling, MXFP8-like blocks, and an NVFP4-inspired toy recipe on normal, heavy-tail, and drifting tensors.
- **Pre-Registered Hypothesis**: Finer/current scaling lowers clipping under distribution shifts but can raise metadata/reduction cost or underflow elsewhere; short stable runs fail to rule out long-horizon drift.
- **Independent Variables**: Format, scale timing, history length, block/axis, rounding, accumulation, outlier transform, high-precision islands.
- **Dependent Variables**: Saturation, underflow/zeros, reconstruction error, scale lag, loss/gradient delta, runtime when hardware supports it.
- **Break & Falsify**: Inject a late outlier, quantize before versus after transpose, reduce accumulator precision, and remove one high-precision island. Reject the recipe at pre-declared numeric boundaries.
- **Alignment**: Lesson 26.5.
- **Effort Estimate**: 3.5h total.

### LAB D — Source Trace and Restart Economics
- **Hardware**: `TRACE` for pinned PyTorch and Transformer Engine sources; `CPU` for failure/storage simulation.
- **Objective**: Trace FSDP2, Transformer Engine recipes, and PyTorch `async_save`; fit a checkpoint policy to synthetic independent and bursty interruption traces; execute a restore drill on tiny state.
- **Pre-Registered Hypothesis**: A policy based on measured exposed staging time and bounded persistence queue yields lower simulated useful-time waste than a copied fixed interval, while staging-only success permits false durability.
- **Independent Variables**: Failure process, interval, exposed stage time, upload bandwidth/failure, restart time, maximum in-flight snapshots.
- **Dependent Variables**: Useful-time fraction, checkpoint age, queue depth, memory/I/O interference, successful restore point, lost work.
- **Break & Falsify**: Kill after staging but before upload; throttle storage below production rate; inject correlated failures; corrupt latest checkpoint. Falsify if completion checks/backpressure do not prevent selecting a non-restorable snapshot.
- **Alignment**: Lessons 26.1, 26.5–26.6.
- **Effort Estimate**: 3h total (source trace counted separately, 1.5h).

## 07 Break / Incident Scenarios

### Incident 26.1 — The Run That Fits, Slows, and Cannot Resume

- **Incident Symptoms**:
  - A 256-GPU run changed from FSDP1+BF16 to FSDP2+FP8 and doubled context length.
  - Startup memory estimates showed 12% headroom, but rank 117 OOMs during the first long sequence.
  - After reducing microbatch size, step time is 28% slower although profiler summaries claim 71% communication overlap.
  - Loss is normal for 40B tokens, then gradient amax and loss spike.
  - Operations reports “checkpoint complete” every 20 minutes, but the latest two checkpoints fail restore after a rack outage.
- **Diagnostic Protocol (Task)**:
  1. *Competing Hypotheses*: omitted transient all-gather or activation term; allocator/workspace peak; PP imbalance; late FSDP/DP bucket; cross-node TP; FP8 scale lag/coarse outlier contamination/accumulation; optimizer instability from Module 25; upload backlog; staging mistaken for durability; correlated failure invalidating the interval model.
  2. *Missing Evidence*: Per-rank memory snapshots at OOM; wrap groups and mesh; sequence/microbatch distribution; synchronized compute/collective trace with readiness; complete precision recipe and amax/scale/saturation; checkpoint staging/upload futures, queue depth, storage errors, hashes, and restore logs.
  3. *Discriminating Tests*: Replay the offending sequence in BF16 and FP8; reduce one memory term at a time; compare critical path rather than duration sums; move TP within node; replay delayed versus current scales; wait for `upload_completion`, kill processes, and restore; test bursty failure simulation.
  4. *Ranked Diagnosis*: Rank only after tests; allow interacting causes. A late bucket can coexist with topology contention; FP8 drift can coexist with optimizer instability.
  5. *Intervention*: Reduce gather group or checkpoint selected activations; remap mesh/rebucket; restore BF16 or revise scale/high-precision islands; bound one in-flight checkpoint, verify upload and manifest, lengthen cadence until persistence keeps up.
  6. *Remeasure*: Peak allocated/reserved by rank, p50/p95 step and exposed tail, amax/saturation/loss against BF16, durable checkpoint age, useful-time fraction, and successful restore.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem: Design a 70B Long-Context MoE Pretraining Run

Design a training plan for a synthetic 70B-total/18B-active MoE decoder, sequence length 32K, 128 GPUs arranged as 16 eight-GPU nodes. Each GPU has a stated 80 GB capacity; intra-node and inter-node bandwidth/latency are supplied. Failure and storage traces include independent failures, one rack outage, persistence bandwidth, and restore time. The quality team allows BF16, FP8, or a frontier block-scaled experiment, but requires a BF16 comparator and rollback boundary.

**Constraints**:
- No physical cluster is provided: use `CPU`, `COLAB-GPU`, and `TRACE` evidence only.
- World-size factorization and process groups must be explicit.
- Peak memory must include activations, transient gathers, workspace/reserve, and pipeline in-flight multiplicity.
- Precision choice must include scale, accumulator, rounding, operator coverage, and high-precision islands.
- A staged snapshot is not durable until persistence and restore checks pass.

**Required Deliverables**:
1. Per-rank persistent and peak memory ledger for at least two sharding/recompute alternatives.
2. DP/FSDP×TP/SP×PP×CP×EP mesh with every forward/backward collective and topology placement.
3. Pipeline and overlap timeline with bubble, readiness, and exposed-tail calculations.
4. Precision recipe and long-horizon falsification plan with BF16 comparator and numeric stop boundaries.
5. Checkpoint/restart policy with assumptions, waste curve, async queue/backpressure, durability proof, and restore drill.
6. Two alternative designs and an ADR selecting one with rollback conditions.
7. Pinned source trace for FSDP2, Transformer Engine recipes, and `async_save`.
8. Incident 26.1 diagnosis using symptom → competing hypotheses → missing evidence → discriminating tests → ranked explanation → intervention → remeasurement.

## 09 Required Evidence & Rubric

### Required Artifacts
- Memory and communication ledger with units and assumptions.
- Rank-mesh/process-group diagram and schedule timeline.
- Profiler/trace analysis showing readiness and critical path.
- Precision manifest and numeric break report.
- Failure/storage model, checkpoint waste sweep, completion logs, and restore evidence.
- Production source trace at both pinned commits, distinguishing static inspection from executed checks.
- ADR and incident report.

### Rubric Dimensions
- **Memory & Sharding**: *Insufficient* divides state bytes by ranks. *Competent* includes persistent state and activations. *Strong* reconciles transient gathers, workspace/reserve, in-flight activations, padding, and measured residuals.
- **Parallelism & Topology**: *Insufficient* lists degrees. *Competent* maps process groups and forward/backward collectives. *Strong* compares alternatives with topology, imbalance, memory, and synchronization sensitivity.
- **Timeline Reasoning**: *Insufficient* reports summed communication. *Competent* identifies overlapped and exposed intervals. *Strong* predicts and remeasures critical-path change under readiness and contention perturbations.
- **Numerical Recipe**: *Insufficient* names FP8/MX/NVFP4. *Competent* records scales, accumulator, rounding, coverage, and BF16 delta. *Strong* constructs long-horizon/outlier/transpose counterexamples and enforces stop boundaries.
- **Restart Economics**: *Insufficient* chooses an arbitrary interval or trusts staging. *Competent* derives a policy and verifies upload. *Strong* tests violated failure assumptions, bounded queues, corruption, and full restore.
- **Source Reasoning**: *Insufficient* cites documentation only. *Competent* traces pinned entry points and symbols. *Strong* links source state transitions to observed memory, communication, numeric, and durability behavior while stating static-inspection limits.
- **Architecture Decision**: *Insufficient* optimizes fit alone. *Competent* balances fit, throughput, numerics, and recovery. *Strong* quantifies uncertainty, adversarial conditions, rollback, and what evidence would reverse the choice.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Build sharded training ledger | 26.1 | LAB A | Deliverable 1 | Persistent/peak ledger, residual reconciliation, FSDP trace |
| Choose activation policy | 26.2 | LAB A | Deliverable 1 | Tensor inventory, replay cost, peak comparison |
| Compose training parallelism | 26.3 | LAB B | Deliverable 2 | Rank mesh, collective table, schedule |
| Diagnose overlap and tail | 26.4 | LAB B | Deliverable 3; Incident steps 2–6 | Readiness/critical-path trace and remeasurement |
| Specify and falsify low precision | 26.5 | LAB C | Deliverable 4; Incident steps 1–6 | Recipe manifest, BF16 delta, break report, TE trace |
| Engineer restart policy | 26.6 | LAB D | Deliverable 5; Incident steps 2–6 | Waste sweep, futures, queue/backpressure, restore proof |
| Trace production implementations | 26.1, 26.5, 26.6 | LAB D | Deliverable 7 | Commit/file/symbol/execution-path trace |
| Defend coupled architecture | 26.1–26.6 | LAB A–D | Deliverables 6, 8 | ADR, rollback criteria, incident report |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 26 can:
1. Predict and reconcile per-rank persistent and peak training memory for ZeRO/FSDP, including transient materialization.
2. Select full, selective, or no activation recomputation from tensor and compute evidence.
3. Compose TP/SP/PP/CP/EP with explicit forward/backward communication and topology.
4. Distinguish aggregate communication from hidden work and exposed critical-path tail.
5. Specify low-precision training as a complete recipe and falsify it with outlier, transpose, accumulator, and long-horizon tests.
6. Derive a checkpoint cadence, bound asynchronous persistence, and prove durability by restore.
7. Trace pinned FSDP2, Transformer Engine, and distributed-checkpoint source paths without generalizing one snapshot into an industry guarantee.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **Core invariant**: Every memory saving creates another obligation—communication, recomputation, temporary residency, numeric validation, or recovery work.
- **Planning path**: ledger → mesh and schedule → critical path → precision recipe → restart contract.
- A run is not ready because it fits once. It is ready when peak states are explained, dependencies are mapped, numeric drift has stop boundaries, checkpoints are durable, and each choice has a measured rollback condition.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
