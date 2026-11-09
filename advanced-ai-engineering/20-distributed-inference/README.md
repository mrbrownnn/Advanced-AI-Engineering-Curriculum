# Module 20 — Distributed Inference

## 00 Why This Module Exists

A model is distributed for one of three reasons: its weights and KV state do not fit on one GPU, one GPU cannot meet the latency target, or one replica cannot carry the traffic. Each reason has a different remedy, and each remedy puts a different communication on the path of every token. Independent replicas add throughput and isolate failures without any per-token communication. Splitting one forward pass across GPUs—by tensor, pipeline stage, expert, or sequence—adds capacity or cuts per-request work, but every participating GPU must then wait for the others on every step.

```text
request --> router --(replica choice: load, prefix affinity)--> replica
                                                                  |
              one replica = one model-parallel group (unit of failure)
              +-----------------------------------------------------------+
              | TP: all-reduce per block      (within fastest link domain) |
              | PP: stage -> stage hop        (forward only at inference)  |
              | EP: all-to-all dispatch/combine of tokens to experts       |
              | CP: KV / Q block exchange along the sequence               |
              +-----------------------------------------------------------+
                                                                  |
      prefill pool --(KV transfer: bytes / achieved bandwidth)--> decode pool
```

Inference runs forward passes only. Training schedules such as 1F1B interleave one forward and one backward pass per microbatch; they appear in this module only as a contrast, never as an inference schedule.

Module 01 supplies model geometry; Module 02 single-GPU execution and timing; Module 03 KV state, paged allocation, and cache identity; Module 04 scheduling, queueing, and capacity; Module 05 single-node kernels and quantization. This module owns what changes when a forward pass or a request spans GPUs, nodes, or pools: placement, collective cost, topology, stragglers, where state lives, the cost of moving KV between prefill and decode, and failure and recovery. Cost modeling stays in Module 22 and production telemetry in Module 23.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Choose and defend a placement of weights, KV, and requests across GPUs, nodes, and pools so the system meets its TTFT/TPOT targets, and know what each fault or slow link does to capacity.
- **What You Will Do**: Write a placement plan from memory fit, SLOs, and topology; predict and measure collective cost; trace tensor parallelism and the KV-transfer contract at a pinned vLLM commit; model forward-only pipelines, expert skew, and context parallelism; size prefill/decode disaggregation including transfer bandwidth; and plan capacity by failure domain.
- **Environment**: Python 3.10+, PyTorch with `torch.distributed`. Two or more GPUs in one node are preferred for the measured parts of Labs B and D; a second node is optional. Without multiple GPUs, run the same code over CPU processes and label the numbers as non-representative. Labs A and C and parts of D use a discrete-event simulator that you write; every result must be labeled **measured** or **simulated**.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. All inputs to worked examples are synthetic and labeled; author-reported results are quoted with their scope. Units: 1 GB $=10^9$ bytes, 1 GiB $=2^{30}$ bytes; link rates in Gb/s are bits, bandwidths in GB/s are bytes.

## 01 Baseline Assumptions

- Module 01: GQA head geometry (Lesson 1.3), parameter and FLOP accounting with absolute parameter counts (Lesson 1.6), and sparse experts (Lesson 1.7).
- Module 02: asynchronous execution and honest timing (Lesson 2.3) and roofline regimes (Lesson 2.4). Collective communication and multi-GPU topology are taught here in Lesson 20.2, not assumed.
- Module 03: logical KV payload per token (Lesson 3.1), cache-aware routing basics (Lesson 3.7), and the cross-instance KV boundary (Lesson 3.10). This module does not re-teach block tables or eviction.
- Module 04: TTFT/TPOT/goodput definitions and the expectation form for latency (Lesson 4.1), prefill/decode interference and chunked prefill (Lesson 4.3), queueing (Lesson 4.4), and capacity planning (Lesson 4.7). This module does not re-teach scheduling policy.
- Module 05: quantization changes bytes per parameter and per KV element (Lesson 5.3); kernels are not revisited.

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
  security: SELECTIVE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: REQUIRED

estimated_effort:
  instruction: 4.5h    # sum of lesson instruction estimates
  guided_practice: 2h  # sum of lesson practice estimates
  labs: 13h            # LAB A 3h + LAB B 3.5h + LAB C 3h + LAB D 3.5h
  assessment: 3h
  source_trace: 2h     # counted once; shared by Lessons 20.3, 20.5 and LABs B, D
  total: 24.5h
```

The learner must derive a placement from memory fit, SLOs, and topology; predict collective cost with stated units and check it against measurement; explain what tensor, pipeline, expert, and context parallelism each buy and cost at inference; decide prefill/decode disaggregation from transfer cost and SLO regime rather than by default; and state the capacity lost to one fault or one slow rank under each plan.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Why distribute (fit / latency / throughput) $\to$ Placement: replicas (DP) versus model-parallel groups (TP, PP, EP, CP) $\to$ Communication on the critical path (all-reduce, stage hop, all-to-all, block exchange) $\to$ Topology (intra-node versus inter-node links, latency and bandwidth) $\to$ Phase specialization (prefill pool, decode pool, KV transfer) $\to$ Routing across replicas (load versus prefix affinity) $\to$ Failure domains, stragglers, and recovery. Evaluation throughout: TTFT, TPOT, and goodput at SLO under offered load (Module 04), plus capacity lost per fault.

## 04 Lessons

### Lesson 20.1 — Why and How to Distribute: Placement Plans

**Engineering Question:**
Given a model, a latency target, and a set of GPUs, what should each GPU hold and which GPUs must talk on every token?

**Concepts & Definitions:**
- **Replica (data parallelism, DP)**: an independent copy of the model serving its own requests. Replicas exchange no activations.
- **Model-parallel group**: GPUs that cooperate on one forward pass. Its dimensions are **tensor parallelism (TP)**—each layer's matrices are sharded; **pipeline parallelism (PP)**—consecutive layers are placed on different GPUs; **expert parallelism (EP)**—MoE experts are placed on different GPUs; **context/sequence parallelism (CP)**—the token sequence is sharded.
- **Placement plan**: replica count and the TP × PP × EP × CP degrees per replica, mapped onto nodes.
- **Critical-path communication**: an exchange that must finish before the step can proceed.

**Mechanism Explanation:**
A plan decides what is replicated and what is sharded, and therefore which communication every token pays (**D**, CLM-001). Replicas add throughput and failure isolation at no per-token communication cost. Every model-parallel dimension splits a forward pass, so its ranks must exchange partial results before continuing. The best layout also depends on the phase: Pope et al. select different partitioning layouts for latency-bound generation and throughput-bound prompt processing from an analytical model (**O**, CLM-009; TPU v4 and PaLM, abstract-level inspection). Start from fit, then spend the remaining GPUs on replicas unless a latency target requires a larger group.

**Quantitative Model / Derivation (D):**
With absolute parameter count $P$ and $B_{param}$ bytes per parameter, weight payload is $M_w = P\,B_{param}$ bytes. Logical KV payload is $M_{kv} = 2\,L\,H_{kv}\,d_{head}\,B_{kv}\,N_{tok}$ bytes for $N_{tok}$ resident tokens (Lesson 3.1). The smallest group that fits is $g_{min} = \lceil (M_w + M_{kv}) / M_{usable} \rceil$ GPUs, before runtime and activation headroom. These are logical payloads, not measured resident footprints.

where: $P$ = parameter count; $B_{param}$, $B_{kv}$ = bytes per weight element and per KV element (2 for bf16); $L$ = layers; $H_{kv}$ = KV heads; $d_{head}$ = head dimension; $N_{tok}$ = resident tokens across all sequences in the replica; $M_{usable}$ = usable bytes per GPU; $g_{min}$ = minimum GPUs per model-parallel group.

Step-by-step for $M_{kv}$: each token stores one key and one value vector (factor 2) per layer ($\times L$) per KV head ($\times H_{kv}$), each of $d_{head}$ elements of $B_{kv}$ bytes; multiply by $N_{tok}$. $g_{min}$ divides total bytes by bytes per GPU and rounds up, because a fractional GPU is not available.

**Worked Example (synthetic inputs):**
- *Input*: dense decoder with $P = 70\times10^9$ parameters in bf16 (2 bytes); $L=80$ layers, $H_{kv}=8$ KV heads, $d_{head}=128$, bf16 KV; target 400,000 resident tokens per replica; 8 GPUs in one node with 76.0 GiB usable each (declared fixture value).
- *Steps*:
  1. $M_w = 70\times10^9 \times 2 = 140\times10^9$ bytes $=140$ GB $\approx 130.39$ GiB.
  2. KV per token $= 2\times80\times8\times128\times2 = 327{,}680$ bytes (320 KiB). $M_{kv} = 327{,}680\times400{,}000 = 131.07$ GB $\approx 122.07$ GiB.
  3. $g_{min} = \lceil (130.39+122.07)/76.0 \rceil = \lceil 3.32 \rceil = 4$.
  4. At TP4, each GPU holds $252.46/4 \approx 63.11$ GiB, leaving $\approx 12.89$ GiB for activations and runtime.
  5. (How Plan B's token count follows.) At TP8 with the same 12.89 GiB headroom, each GPU has $76.0-12.89 = 63.11$ GiB for payload; weights take $130.39/8 \approx 16.30$ GiB, leaving $\approx 46.81$ GiB of KV per GPU, $\approx 374.5$ GiB across 8 GPUs; $374.5\times2^{30}/327{,}680 \approx 1.23$ million tokens.
- *Result*: two candidate plans for the node. Plan A: 2 replicas × TP4. Plan B: 1 replica × TP8, holding $\approx 31.56$ GiB of the same payload per GPU, or about 1.23 million resident tokens at the same per-GPU headroom.
- *Interpretation / limits*: Plan A gives two failure domains and smaller collectives; Plan B gives longer contexts or more concurrent sequences per replica and a larger group to stall on one fault. Which meets the SLO is a measurement, not a consequence of this arithmetic. TP4 shards KV evenly only because $H_{kv}=8$ is divisible by 4. Headroom needs are runtime-specific and must be measured.

**Knowledge Check:**
1. Which plan dimension adds throughput without adding per-token communication?
2. Why can the minimum group that fits still be the wrong plan?

**Guided Practice:**
For a synthetic $32\times10^9$-parameter bf16 model ($L=48$, $H_{kv}=8$, $d_{head}=128$) and 200,000 resident tokens on 48.0 GiB-usable GPUs, compute $g_{min}$ and list two plans for a 4-GPU node with their failure-domain counts.

**Feedback Contract:**
- *Expected Evidence*: $M_w = 64$ GB $\approx 59.60$ GiB; KV $=196{,}608$ bytes/token, $M_{kv}\approx 36.62$ GiB; $g_{min}=\lceil 96.23/48\rceil = 3$; plans such as 1 × TP4 (one domain) versus TP2 replicas that do not fit, with the fit failure stated.
- *Common Failure*: Mixing GB and GiB, or treating logical payload as resident footprint.
- *Diagnostic Hint*: Which bytes did you count twice, and which did you not count at all?
- *Concept to Revisit*: KV State & Quantitative Model (Lesson 3.1); Parameter Accounting (Lesson 1.6).

**Learning Outcome:**
Derive candidate placement plans from memory fit and state what each trades.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 20.2 — Collectives, Topology, and Communication Cost

**Engineering Question:**
How long does the communication on a token's critical path take, and which term dominates?

**Concepts & Definitions:**
- **Collective**: a group communication primitive. **All-reduce** sums a tensor across ranks and gives every rank the result; **all-gather** concatenates shards; **reduce-scatter** sums then shards; **all-to-all** sends a different slice from every rank to every rank.
- **Topology**: which links connect ranks—direct GPU links within a node, network fabric (InfiniBand/RoCE/Ethernet) between nodes. Each has a latency and an achieved bandwidth for a given message size.
- **Algorithm bandwidth and bus bandwidth**: nccl-tests defines algorithm bandwidth as data size over time and bus bandwidth as algorithm bandwidth times an operation factor—$2(n-1)/n$ for AllReduce, $(n-1)/n$ for ReduceScatter, AllGather, and AlltoAll, 1 for Broadcast and Reduce—so bus bandwidth reflects link utilization independent of rank count (**O**, CLM-004).

**Mechanism Explanation:**
A collective's time has a size-independent part (launch, synchronization, hops) and a size-dependent part (bytes over bandwidth). Large messages are bandwidth-dominated; small messages are latency-dominated. Decode sends small messages—one token per sequence per step—so the number of collectives matters more than their size. Prefill sends large messages, so bandwidth matters. The same collective on a slower inter-node link pays both a larger latency and a lower bandwidth.

**Quantitative Model / Derivation:**
First-order all-reduce time (**D**, CLM-005):

$$T_{AR}(S,n) \approx \alpha + \frac{2(n-1)}{n}\cdot\frac{S}{B_{bus}}$$

with $S$ in bytes, $\alpha$ in seconds, $B_{bus}$ in bytes/s measured on the target topology at that message size. For a TP decode step $S = N_{seq}\times d_{model}\times B_{act}$. $\alpha$ is not a constant of nature: it varies with algorithm, rank count, and size, so measure it.

where: $S$ = bytes of the tensor being all-reduced (the full tensor each rank holds, not a shard); $n$ = ranks in the group (the TP degree here); $\alpha$ = fixed per-collective latency (launch, synchronization, hops)—no relation to any other $\alpha$ in the course; $B_{bus}$ = bus bandwidth in bytes/s; $N_{seq}$ = sequences in the decode batch (one new token each); $d_{model}$ = hidden size; $B_{act}$ = bytes per activation element.

Step-by-step for the factor $2(n-1)/n$ (ring all-reduce view, consistent with the nccl-tests factors above):
1. All-reduce = reduce-scatter followed by all-gather.
2. Split the tensor into $n$ chunks of $S/n$ bytes. In reduce-scatter each rank sends $n-1$ chunks, i.e. $\tfrac{n-1}{n}S$ bytes; all-gather sends the same amount again.
3. Total bytes each rank sends: $2\cdot\tfrac{n-1}{n}\,S$. Dividing by per-link bandwidth gives the bandwidth term. Equivalently, nccl-tests' algorithm bandwidth is $S/T$ and bus bandwidth is $\tfrac{2(n-1)}{n}\cdot S/T$, so $T = \tfrac{2(n-1)}{n}\cdot S/B_{bus}$.
4. As $n\to\infty$, $\tfrac{n-1}{n}\to1$, so the factor approaches 2: each rank sends almost twice the tensor no matter how many ranks join. Example values: $n=2$: 1; $n=4$: 1.5; $n=8$: 1.75.

**Worked Example (synthetic inputs):**
- *Input*: $d_{model}=8192$, bf16 activations, $L=80$ with two all-reduces per layer (Lesson 20.3), decode batch of 64 sequences. Intra-node: $n=4$, $\alpha=15\ \mu s$, $B_{bus}=200$ GB/s. Cross-node: $n=8$ over two nodes, $\alpha=40\ \mu s$, $B_{bus}=20$ GB/s. All link values are assumptions, not measurements.
- *Steps*:
  1. $S = 64\times8192\times2 = 1{,}048{,}576$ bytes (1 MiB).
  2. Intra-node bandwidth term: $2\cdot\tfrac{3}{4}\cdot 1{,}048{,}576 / (200\times10^9) = 7.86\ \mu s$; $T_{AR} = 22.86\ \mu s$.
  3. Per decode step: $2L = 2\times80 = 160$ all-reduces, so $160\times22.86\ \mu s = 3.66$ ms.
  4. Cross-node bandwidth term: $2\cdot\tfrac{7}{8}\cdot 1{,}048{,}576/(20\times10^9) = 91.75\ \mu s$; $T_{AR}=131.75\ \mu s$; per step $160\times131.75\ \mu s = 21.08$ ms.
  5. Prefill chunk of 2,048 tokens intra-node: $S = 2{,}048\times8192\times2 = 33{,}554{,}432$ bytes (32 MiB); bandwidth term $251.66\ \mu s$; $T_{AR}=266.66\ \mu s$; per chunk $160\times266.66\ \mu s = 42.67$ ms.
- *Result*: intra-node decode communication is latency-dominated (15 of 22.86 µs); cross-node decode is bandwidth- and latency-heavier by 5.8× ($131.75/22.86\approx5.76$); prefill is bandwidth-dominated.
- *Interpretation / limits*: for decode, reducing the count or latency of collectives helps more than raising bandwidth. The model ignores overlap with compute, fused or custom all-reduce paths, and congestion; it is a prediction to compare with a profiler trace.

**Knowledge Check:**
1. Why does the AllReduce bus-bandwidth factor approach 2 as $n$ grows?
2. Which term would you attack first for decode, and which for prefill?

**Guided Practice:**
Given measured nccl-tests output of algorithm bandwidth 80 GB/s for a 4-rank AllReduce at 32 MiB, compute bus bandwidth and the time for one such all-reduce; then explain why the same link may show far lower algorithm bandwidth at 64 KiB.

**Feedback Contract:**
- *Expected Evidence*: bus bandwidth $=80\times1.5=120$ GB/s (factor $2\cdot3/4=1.5$ for $n=4$); time $=33{,}554{,}432/(80\times10^9)\approx0.42$ ms; small-message explanation in terms of the latency term.
- *Common Failure*: Dividing size by link rate in Gb/s without converting bits to bytes.
- *Diagnostic Hint*: What is the time for a zero-byte all-reduce?
- *Concept to Revisit*: Honest Timing (Lesson 2.3).

**Learning Outcome:**
Predict collective time with units and identify the dominant term for decode and prefill.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 20.3 — Tensor Parallelism in a Production Runtime

**Engineering Question:**
Where exactly does tensor parallelism communicate, and how far can its group stretch?

**Concepts & Definitions:**
- **Column-parallel linear**: weight split along the output dimension; each rank computes a slice of the output. No communication unless the full output is needed.
- **Row-parallel linear**: weight split along the input dimension; each rank computes a partial sum that must be all-reduced.
- **Megatron pattern**: split the first MLP GEMM by columns and the second by rows so the nonlinearity needs no communication; split attention by heads; result: two all-reduces in the forward path per transformer layer (**O**, CLM-002; training paper—its backward-path all-reduces do not exist at inference).

**Mechanism Explanation — Pinned vLLM Trace (O, CLM-003):**
At `vllm-project/vllm` commit `863475eb9cd78ae6a4a22f1b90df7fe85e0af438` (static inspection, 2026-09-30):
1. `vllm/model_executor/layers/linear.py::ColumnParallelLinear.forward` applies the local shard and calls `tensor_model_parallel_all_gather` only if `gather_output` is true and `tp_size > 1`.
2. `RowParallelLinear.forward` takes already-parallel input (or splits it), applies the local shard with the bias only on rank 0, and calls `tensor_model_parallel_all_reduce` when `reduce_results` is true and `tp_size > 1`.
3. `vllm/distributed/communication_op.py::tensor_model_parallel_all_reduce` calls `get_tp_group().all_reduce`; `vllm/distributed/parallel_state.py::GroupCoordinator.all_reduce` returns the input unchanged when the group's world size is 1, otherwise dispatches to the device communicator.
4. `initialize_model_parallel` lays ranks out as ExternalDP × DP × PP × PCP × TP with TP innermost, so each TP group is a run of consecutive ranks; it builds an expert-parallel group only for MoE models.
What was not established: which device communicator runs on a given machine, and whether consecutive ranks share a node—that is the launcher's rank-to-host mapping.

**Quantitative Model / Derivation (D):**
If one rank alone would need $T_c$ per step and compute divides evenly, $T_{step}(n) \approx T_c/n + 2L\,T_{AR}(S,n)$ and speedup is $T_c/T_{step}(n)$. This is an upper bound on TP benefit: it assumes perfectly divisible compute and no overlap. PTD-P's Takeaway #1 is to use tensor parallelism up to the GPUs in one server and pipeline parallelism across servers (**O**, CLM-006; training). The corresponding inference prediction—intra-node TP plus replicas beats cross-node TP at equal GPU count—is a hypothesis to test (**H**, CLM-026).

where: $T_c$ = compute time of one decode step if a single rank held the whole model; $n$ = TP degree; $2L$ = all-reduces per step (two per layer: after the attention output projection and after the MLP down projection); $T_{AR}(S,n)$ from Lesson 20.2. Step-by-step: compute shrinks as $T_c/n$; communication does not shrink with $n$ (the $\alpha$ term is fixed and the bandwidth factor grows toward 2), so speedup $=T_c/(T_c/n + 2L\,T_{AR})$ saturates and falls below the ideal $n$. "% of ideal" below is speedup$/n$.

**Worked Example (synthetic inputs; continues Lesson 20.2):**
- *Input*: $T_c = 60$ ms (hypothetical single-rank step time if the model fit), $L=80$, communication per step from Lesson 20.2: TP4 intra-node 3.66 ms; TP8 intra-node ($\alpha=15\ \mu s$, 200 GB/s) and TP8 across two nodes ($\alpha=40\ \mu s$, 20 GB/s). Shapes: 64 query heads, 8 KV heads, MLP width 28,672.
- *Steps*:
  1. Shards at TP4: $64/4=16$ query heads and $8/4=2$ KV heads per rank; `gate`/`up` column shards $8192\times7168$ ($28{,}672/4=7{,}168$); `down` row shard $7168\times8192$; each all-reduce carries $N_{seq}\times8192$ values.
  2. TP4: $60/4 + 3.66 = 18.66$ ms; speedup $60/18.66 = 3.22$ (80% of ideal).
  3. TP8 intra-node: bandwidth term $2\cdot\tfrac{7}{8}\cdot1{,}048{,}576/(200\times10^9) = 9.18\ \mu s$, so $T_{AR} = 15 + 9.18 = 24.18\ \mu s$; communication $160\times24.18\ \mu s = 3.87$ ms; step $7.5+3.87 = 11.37$ ms; speedup $5.28$ (66% of ideal).
  4. TP8 across nodes: $7.5 + 21.08 = 28.58$ ms; speedup $2.10$—slower than TP4 within one node.
- *Result*: communication share of the step (communication / step time, e.g. $3.66/18.66$) rises from 20% (TP4) to 34% (TP8 intra-node) to 74% (TP8 cross-node).
- *Interpretation / limits*: under these assumed links, stretching TP across nodes loses to a smaller intra-node group; two TP4 replicas would also double throughput. The result is conditional on the assumed $\alpha$ and $B_{bus}$; faster inter-node fabrics or overlapped communication can falsify it (CLM-026).

**Knowledge Check:**
1. Why does the row-parallel layer add its bias on rank 0 only?
2. Why does TP degree interact with the number of KV heads?

**Guided Practice:**
List every all-reduce in one decode step for a 32-layer model under the traced pattern, give each message size for a batch of 16 at $d_{model}=4096$ in bf16, and state which trace steps justify the count.

**Feedback Contract:**
- *Expected Evidence*: 64 all-reduces (after attention output projection and after MLP down projection per layer), each $16\times4096\times2 = 131{,}072$ bytes; citations to `RowParallelLinear.forward` and the `gather_output=False` default.
- *Common Failure*: Counting an all-gather after every column-parallel layer.
- *Diagnostic Hint*: Which layers set `reduce_results` or `gather_output`?
- *Concept to Revisit*: GQA Geometry (Lesson 1.3).

**Learning Outcome:**
Trace and count tensor-parallel communication in a pinned runtime and bound its benefit.

*(Effort: 45m instruction, 20m practice; source trace 2h shared with Lesson 20.5)*

---

### Lesson 20.4 — Pipeline, Expert, and Context Parallelism at Inference

**Engineering Question:**
What do the other three model-parallel dimensions buy at inference, and what limits each?

**Concepts & Definitions:**
- **Pipeline bubble**: stage idle time. For training, GPipe gives $O((K-1)/(M+K-1))$ and found it negligible for $M\ge4K$ in its experiments (**O**, CLM-007); PTD-P gives a bubble fraction $(p-1)/m$, reduced by $v$ with an interleaved schedule at $v$ times the communication (**O**, CLM-006). Both count forward and backward work. (Notation: GPipe's $K$ = stages, $M$ = micro-batches; PTD-P's $p$ = pipeline stages, $m$ = micro-batches, $v$ = model chunks per device. Below, $p$ and $m$ are reused for stages and circulating batches; $p$ is never a probability in this module.)
- **Forward-only schedule**: at inference there is no backward pass. Decode adds an autoregressive dependency: a batch's next step cannot start until its previous step leaves the last stage.
- **Expert parallelism**: MoE experts live on different ranks; tokens are dispatched to their experts and combined back by all-to-all.
- **Context parallelism**: the sequence is split across ranks; Ring Attention exchanges key-value blocks around a ring and overlaps that communication with blockwise attention (**O**, CLM-012).

**Mechanism Explanation:**
*Pipeline.* PP adds memory capacity with cheap point-to-point hops, but a token still traverses every stage: PP does not lower per-token latency, and one batch alone leaves most stages idle (**D**, CLM-008). Throughput comes from keeping several batches circulating; the slowest stage sets the rate.
*Expert.* The DeepSeek-V3 report describes separate prefill and decode deployments: prefill on 32 GPUs with TP4 plus sequence parallelism and DP8 for attention and EP32 for MoE; decode on 320 GPUs with DP80 attention and EP320; redundant copies of high-load experts chosen from online statistics and adjusted about every 10 minutes; all-to-all across nodes over InfiniBand then within nodes over NVLink; and two micro-batches processed at once to overlap compute with dispatch/combine (**O**, CLM-010; one vendor's deployment, not an industry default). MegaScale-Infer goes further and disaggregates attention from FFN (**O**, CLM-011; FRONTIER, abstract-level inspection).
*Context.* Yang et al. report near-linear long-prefill scaling to 128 H100 GPUs with exact pass-KV and pass-Q ring variants, and similar scalability over RDMA and TCP (**O**, CLM-013; author-reported, abstract-level inspection). CP helps when per-block compute is long enough to hide the block exchange—long prefill, not short decode steps.

**Quantitative Model / Derivation (D, CLM-008):**
Stage $i$ takes $t_i$ per batch step; each hop, including returning the sampled token to stage 1, takes $x_i$ and does not occupy a stage.
- One-shot pass of $m$ batches through $p$ equal stages (prefill): elapsed $(m+p-1)\,t_s$; utilization $m/(m+p-1)$.
- Decode ring with $m$ circulating batches: cycle $= \max\big(\sum_i (t_i+x_i),\; m\cdot\max_i t_i\big)$; throughput $= m/\text{cycle}$ batch-steps; per-batch TPOT $=$ cycle; stage utilization $m\,t_i/\text{cycle}$.
- Expert skew: with synchronous combine, the MoE layer finishes when the most-loaded rank finishes; if rank load is proportional to assigned tokens, layer time scales with $\max_r \text{share}_r \times n_{ranks}$ relative to uniform.

where: $t_s$ = time of each stage when all stages are equal; $t_i$ = time of stage $i$; $x_i$ = hop time after stage $i$; $p$ = stages; $m$ = batches in flight; $\text{share}_r$ = fraction of token-to-expert assignments landing on EP rank $r$; $n_{ranks}$ = EP ranks.

Step-by-step:
1. *One-shot pass.* The first batch leaves the last stage after $p\,t_s$; each later batch follows $t_s$ behind, so $m$ batches finish at $p\,t_s + (m-1)\,t_s = (m+p-1)\,t_s$. Each stage is busy $m\,t_s$ of that time, so utilization $= m/(m+p-1)$. (Hop times are ignored in this equal-stage form.)
2. *Decode ring, latency bound.* One batch must traverse every stage and hop before its next token can start: at least $\sum_i (t_i+x_i)$ per step.
3. *Decode ring, resource bound.* Stage $i$ must process all $m$ batches once per cycle, taking $m\,t_i$; the slowest stage needs $m\max_i t_i$.
4. The cycle is the larger of the two bounds. Each batch advances one step (one token per sequence) per cycle, so TPOT $=$ cycle and throughput $= m/\text{cycle}$.
5. *Expert skew.* With uniform routing each rank has share $1/n_{ranks}$; time ratio to uniform is $\max_r\text{share}_r/(1/n_{ranks}) = \max_r\text{share}_r\times n_{ranks}$. Example: $0.30\times8 = 2.4$.

**Worked Example (synthetic inputs; forward-only decode timeline):**
- *Input*: $p=4$ stages, $t_s = 6$ ms, $x=0.5$ ms per hop, batches of 32 sequences.
- *Steps*:
  1. $m=1$: cycle $=4\times(6+0.5)=26$ ms; each stage is busy 6 of 26 ms (23%); throughput $1/26\text{ ms} = 38.5$ batch-steps/s $=38.5\times32=1{,}231$ tokens/s; TPOT 26 ms.
  2. $m=4$: cycle $=\max(26, 4\times6)=\max(26,24)=26$ ms; utilization $24/26=92\%$; throughput $4/26\text{ ms}=153.8$ batch-steps/s $=4{,}923$ tokens/s; TPOT still 26 ms.
  3. $m=5$: cycle $=\max(26,30)=30$ ms; utilization 100%; throughput $166.7$ batch-steps/s; TPOT rises to 30 ms.
  4. Imbalance, stage times $6,6,6,9$ ms with $m=4$: $\sum(t_i+x_i) = 27 + 4\times0.5 = 29$ ms and $m\max t_i = 4\times9 = 36$ ms, so cycle $=\max(29, 36)=36$ ms; throughput $4/36\text{ ms}=111.1$ batch-steps/s; TPOT 36 ms (38% worse than balanced: $36/26\approx1.38$); the first three stages sit at $4\times6/36=67\%$.
  5. Expert skew: 8 EP ranks, 4,096 tokens × top-8 routing $=32{,}768$ assignments, uniform $32{,}768/8=4{,}096$ per rank. If one rank's experts receive 30%, it gets $0.30\times32{,}768\approx9{,}830$ (2.4× the mean). Hosting redundant copies of those experts on a ninth rank and splitting the load evenly leaves 4,915 on each of the two hot ranks.
- *Result*: PP multiplied throughput by 4 at unchanged TPOT only once four batches circulated; one slow stage cost 28% of throughput ($1-111.1/153.8$); expert redundancy halved the worst-rank load.
- *Interpretation / limits*: there is no backward work anywhere in this timeline. The ring model assumes one batch per stage at a time and fixed stage times; real runtimes mix prefill chunks and decode steps, so measure per-stage busy time. The skew arithmetic assumes rank time proportional to assigned tokens.

**Knowledge Check:**
1. Why does 1F1B not describe an inference pipeline?
2. Why does adding a fifth batch to a balanced four-stage ring raise TPOT?

**Guided Practice:**
For $p=3$, stage times 5, 8, 5 ms, $x=0.4$ ms, compute cycle, throughput, and per-stage utilization for $m=1,2,3$, then say how you would rebalance the layers.

**Feedback Contract:**
- *Expected Evidence*: $\sum(t_i+x_i)=19.2$ ms; $m=1$: cycle 19.2 ms, 52.1 steps/s; $m=2$: cycle 19.2 ms, 104.2 steps/s; $m=3$: cycle 24 ms, 125 steps/s, middle stage 100%, outer stages 62.5%; rebalance by moving layers out of stage 2.
- *Common Failure*: Applying the training bubble fraction $(p-1)/m$ to decode.
- *Diagnostic Hint*: When can batch 1's next token enter stage 1?
- *Concept to Revisit*: Sparse Experts (Lesson 1.7).

**Learning Outcome:**
Model forward-only pipelines, expert skew, and context parallelism, and say when each helps.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 20.5 — Prefill/Decode Disaggregation and KV Transfer Cost

**Engineering Question:**
When is it worth running prefill and decode on separate pools, given that the KV must move between them?

**Concepts & Definitions:**
- **Disaggregation**: prefill and decode run on different GPUs or pools; the prompt's KV is transferred. DistServe does this to remove phase interference and to choose resources and parallelism per phase, placing phases by cluster bandwidth (**O**, CLM-014). Splitwise adds phase-specific hardware (**O**, CLM-015). Mooncake separates prefill and decode clusters around a disaggregated KV cache and adds early rejection under overload (**O**, CLM-016). Reported gains in all three are author-reported for their workloads and baselines; abstract-level inspection.
- **Aggregation**: both phases on the same replica with chunked prefill (Lesson 4.3).
- **KV connector**: the runtime interface that saves, moves, and loads KV.

**Mechanism Explanation:**
Disaggregation trades interference for transfer. vLLM's documentation marks its disaggregated prefilling experimental, gives the motivations as tuning TTFT and inter-token latency separately and controlling tail inter-token latency, and states that it does not improve throughput (**O**, CLM-017). TaiChi reports aggregation is best for tight TTFT with relaxed TPOT, disaggregation for strict TPOT with relaxed TTFT, and neither under balanced SLOs (**O**, CLM-018; FRONTIER preprint). In multi-turn traffic, repeated transfers can saturate links, and running append-prefill on the decode node can be better (**O**, CLM-022; FRONTIER preprint). Smaller KV in hybrid-attention models may extend transfer across datacenters, with scheduling still required (**O**, CLM-021; FRONTIER preprint). The decision is therefore a hypothesis per workload (**H**, CLM-027).

*Pinned vLLM KV-connector contract (O, CLM-019)*, same commit as Lesson 20.3, `vllm/distributed/kv_transfer/kv_connector/v1/base.py::KVConnectorBase_V1`: scheduler-side `get_num_new_matched_tokens` reports how many prompt tokens can be loaded from external KV—only the prefix actually loadable at call time, or `None` to be asked again; `update_state_after_alloc` and `build_connector_meta` bind the transfer to allocated blocks; worker-side `start_load_kv`/`wait_for_layer_load` and `save_kv_layer`/`wait_for_save` move KV per layer; `request_finished` may take ownership of blocks for asynchronous sending; `get_block_ids_with_load_errors` reports blocks that failed to load. Interface and docstrings only; no connector's transport or recovery was traced or executed.

**Quantitative Model / Derivation (D, CLM-020):**
Bytes to move: $M_{xfer} = 2\,L\,H_{kv}\,d_{head}\,B_{kv}\times N_{uncached}$. In expectation form (Lesson 4.1) for one request class:

$$E[TTFT_{disagg}] = E[W_{prefill\ queue}] + E[T_{prefill}] + E[T_{xfer,\ non\text{-}overlapped}] + E[W_{decode\ admit}] + E[T_{first\ step}]$$

Sequential transfer adds $M_{xfer}/B + t_{setup}$. With layer-wise streaming on a link that keeps up, only the last layer's bytes remain after prefill ends. Aggregate link demand is $\lambda_{prefill}\times E[M_{xfer}]$ bytes/s and must stay below achieved bandwidth.

where: $N_{uncached}$ = prompt tokens whose KV is not already at the decode side; $W$ terms = queueing waits; $T_{prefill}$ = prefill compute time; $T_{xfer,\ non\text{-}overlapped}$ = transfer time not hidden behind prefill; $T_{first\ step}$ = first decode step; $B$ = achieved link bandwidth in bytes/s (not bytes per parameter); $t_{setup}$ = fixed per-transfer setup time; $\lambda_{prefill}$ = requests/s leaving the prefill pool.

Step-by-step for the streaming tail: if layer $\ell$'s KV is sent while layer $\ell+1$ is computed, and the whole transfer $M_{xfer}/B$ is no longer than $T_{prefill}$, every layer except the last is already sent when prefill ends. What remains is one layer: $(M_{xfer}/L)/B$, plus $t_{setup}$.

**Worked Example (synthetic inputs; Lesson 20.1 model):**
- *Input*: 327,680 bytes/token; prompt 8,000 tokens, none cached at the decode side; $E[T_{prefill}] = 300$ ms; $t_{setup}=5$ ms; achieved bandwidth 10 GB/s (a 100 Gb/s link at 80%) or 40 GB/s; TTFT budget 500 ms; queue and first-step terms set to zero to isolate transfer; prefill pool output 20 requests/s.
- *Steps*:
  0. Bandwidth conversion: $100\text{ Gb/s}/8 = 12.5$ GB/s; at 80%, $0.8\times12.5 = 10$ GB/s.
  1. $M_{xfer} = 327{,}680\times8{,}000 = 2.62$ GB ($\approx2.44$ GiB).
  2. Sequential at 10 GB/s: $2.62\times10^9/10^{10}\text{ s} = 262.1$ ms; TTFT $=300+262.1+5 = 567.1$ ms—over budget.
  3. Sequential at 40 GB/s: $65.5$ ms; TTFT $=300+65.5+5=370.5$ ms—within budget.
  4. Layer-wise streaming at 10 GB/s: total transfer (262 ms) is shorter than prefill (300 ms), so the link keeps up; the tail is one layer, $2.62\text{ GB}/80 = 32.8$ MB, $32.8\text{ MB}/(10\text{ GB/s}) = 3.3$ ms; TTFT $\approx 300+3.3+5 = 308.3$ ms.
  5. Aggregate demand: $20\times2.62 = 52.4$ GB/s—more than either link. With 75% of each prompt already cached at the decode side, $0.25\times8{,}000 = 2{,}000$ tokens move: $327{,}680\times2{,}000 = 0.655$ GB per request, $20\times0.655 = 13.1$ GB/s.
- *Result*: a single request fits the budget with streaming on the slower link, but the pool's aggregate demand is 5.2× the slower link ($52.4/10$) and 1.3× the faster one ($52.4/40$) unless prefix reuse or more links cut the bytes.
- *Interpretation / limits*: per-request arithmetic can pass while the fleet saturates; size links for aggregate bytes/s at peak. Payload is logical; block padding, layout conversion, and compression change wire bytes. Setting queue terms to zero hides the pool-ratio problem that Lab D measures.

**Knowledge Check:**
1. Why can disaggregation leave throughput unchanged yet improve goodput at an SLO?
2. Which connector method decides how much prefill the decode side can skip, and what must it not count?

**Guided Practice:**
For a model with 131,072 bytes/token of KV, prompts of 4,000 tokens with 50% cached, 30 requests/s, and 12.5 GB/s achieved bandwidth, compute per-request sequential transfer time and aggregate link utilization.

**Feedback Contract:**
- *Expected Evidence*: uncached 2,000 tokens $=131{,}072\times2{,}000=0.262$ GB; transfer $0.262/12.5\approx 21.0$ ms (plus any setup time); aggregate $30\times0.262=7.86$ GB/s $=7.86/12.5=63\%$ of the link; a note that burst arrivals can exceed it.
- *Common Failure*: Sizing the link from one request's transfer time.
- *Diagnostic Hint*: What is bytes/s at peak arrival rate, not on average?
- *Concept to Revisit*: Cross-Instance KV (Lesson 3.10); Prefill/Decode Interference (Lesson 4.3).

**Learning Outcome:**
Decide and size disaggregation from transfer cost, SLO regime, and the runtime's transfer contract.

*(Effort: 50m instruction, 20m practice; source trace shared with Lesson 20.3)*

---

### Lesson 20.6 — Routing Across Replicas, Stragglers, Failure, and Recovery

**Engineering Question:**
What does one hot prefix, one slow rank, or one dead GPU do to the fleet, and how should the plan absorb it?

**Concepts & Definitions:**
- **Failure domain**: the set of resources lost together. For model parallelism it is the whole group.
- **Straggler**: a rank that is slower than its peers (thermal throttling, degraded link, noisy neighbor).
- **Blast radius**: capacity lost per fault.
- **Prefix-aware routing**: sending requests to replicas that hold their prefix KV (Lesson 3.7). Preble co-optimizes KV reuse and load balance across a cluster (**O**, CLM-023; author-reported, abstract-level inspection).
- **State placement**: where a request's KV lives decides what must be recomputed or moved when its replica fails.

**Mechanism Explanation:**
Model-parallel ranks synchronize on every step, so one failed or slow rank stalls or slows its entire group; the group, not the GPU, is the unit of failure (**D**, CLM-025). The pinned vLLM layout code notes that ranks in the same in-model DP group must generate together or deadlock. Independent replicas share nothing, so a fault removes one replica. Routing has the same shape of trade-off: pinning a prefix that carries traffic share $f$ to one of $R$ replicas gives it at least $f$ of the load against a fair share of $1/R$; affinity must be bounded by load, and judged by goodput and tail latency rather than hit rate (**D**, CLM-024). Recovery has three costs: detecting the fault, reloading weights and rejoining the group, and re-running prefill for in-flight requests whose KV was lost—unless KV was placed in a shared pool (Lesson 20.5).

**Quantitative Model / Derivation (D):**
For $G$ GPUs split into $R$ replicas of $g$ GPUs: capacity lost per GPU fault $=1/R$. If each GPU is independently unavailable with probability $q$, a replica is up with probability $(1-q)^g$, which is also the expected available capacity fraction. A rank slower by factor $s$ slows its group's step by $s$: fleet throughput loss $=(1-1/s)/R$. The independence assumption fails for shared causes (node power, switch); count those as larger domains.

where: $G = R\,g$; $q$ = per-GPU unavailability probability; $s\ge1$ = straggler slowdown factor; $f$ = traffic share of a hot prefix. Step-by-step:
1. *Per-fault loss.* One GPU fault stops its whole group, i.e. one of $R$ equal replicas: $1/R$.
2. *Availability.* A replica runs only if all $g$ GPUs are up: $(1-q)^g$. By linearity of expectation, expected up replicas $=R(1-q)^g$, a fraction $(1-q)^g$ of capacity.
3. *Straggler.* The affected replica's step takes $s$ times longer, so its throughput becomes $1/s$ of normal and it loses $1-1/s$ of its $1/R$ share: fleet loss $(1-1/s)/R$.
4. *Hotspot.* Pinned load / fair share $= f/(1/R) = f\,R$; example $0.40\times4 = 1.6$.

**Worked Example (synthetic inputs):**
- *Input*: 64 GPUs; plans A: 16 × TP4; B: 8 × TP8; C: 2 × (TP8 × PP4, 32 GPUs each). $q = 0.01$. One rank runs 20% slow ($s=1.2$). Weight reload 140 GB at 2 GB/s. Hot prefix share $f=0.40$ with $R=4$ routing targets.
- *Steps*:
  1. Capacity lost per fault: A $1/16=6.25\%$; B $1/8=12.5\%$; C $1/2=50\%$.
  2. Expected available capacity ($g=4, 8, 32$): A $0.99^4=96.1\%$; B $0.99^8=92.3\%$; C $0.99^{32}=72.5\%$.
  3. Straggler loss ($1-1/1.2 = 0.1667$): A $(1-1/1.2)/16 = 1.04\%$; B $0.1667/8=2.08\%$; C $0.1667/2=8.33\%$.
  4. Reload after a fault: $140/2 = 70$ s before warmup, for every plan; plan C reloads 32 GPUs' shards.
  5. Hotspot: the pinned replica receives $\ge40\%$ of load versus a 25% fair share—1.6×.
- *Result*: to survive one fault at full demand, plan A needs one spare TP4 replica (6.25% extra), plan C needs a whole extra 32-GPU replica.
- *Interpretation / limits*: larger groups buy context length or single-request latency at the price of availability and straggler sensitivity. The numbers assume independent GPU faults and no elastic recovery inside a group; redundant experts or elastic EP can change the picture for MoE ranks.

**Knowledge Check:**
1. Why is "GPU utilization looks normal" compatible with a straggler-limited group?
2. What state is lost when a decode replica dies, and what does recovery cost per in-flight request?

**Guided Practice:**
For 32 GPUs, compare 8 × TP4 with 2 × TP16 on capacity lost per fault, expected availability at $q=0.02$, and straggler loss at $s=1.5$; recommend spare capacity for each.

**Feedback Contract:**
- *Expected Evidence*: per-fault loss 12.5% vs 50%; availability $0.98^4=92.2\%$ vs $0.98^{16}=72.4\%$; straggler loss $4.17\%$ vs $16.7\%$; spares of one replica each, i.e., 4 vs 16 GPUs.
- *Common Failure*: Counting failure domains as GPUs.
- *Diagnostic Hint*: Which GPUs stop producing tokens when this one stops?
- *Concept to Revisit*: Capacity Planning (Lesson 4.7); Cache-Aware Routing (Lesson 3.7).

**Learning Outcome:**
Quantify blast radius, straggler loss, and routing hotspots, and plan recovery and spares.

*(Effort: 40m instruction, 20m practice)*

---

## 05 Literature & Production Source Map

Section pointers are given where the full text was read; "abstract" marks abstract-level inspection.

**REFERENCE / BASELINE**
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019; Section 3 (tensor-parallel MLP and attention, all-reduce counts). Training paper.
- [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., 2018; Section 2.3 (bubble overhead). Training paper.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) — Narayanan et al., 2021; Sections 2.2.1–2.2.2 and Takeaway #1. Its 1F1B schedules are training schedules.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — Pope et al., 2022; abstract.
- [Ring Attention](https://arxiv.org/abs/2310.01889) — Liu, Zaharia, Abbeel, 2023; abstract.
- [DistServe](https://arxiv.org/abs/2401.09670) — Zhong et al., OSDI 2024; abstract. [Splitwise](https://arxiv.org/abs/2311.18677) — Patel et al., 2023; abstract.

**CURRENT DEFAULT (scoped):** replicas for throughput and isolation; tensor parallelism within the fastest link domain; bus-bandwidth accounting per [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md); sizing transfer links by aggregate bytes/s; counting failure domains as groups. These are derivations and common practice in the inspected sources, not a measured industry survey.

**WORKLOAD-DEPENDENT:** pipeline parallelism for capacity; context parallelism for long prefill ([Yang et al., 2024](https://arxiv.org/abs/2411.01783), abstract); expert parallelism layouts ([DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), Section 3.4); prefill/decode disaggregation ([Mooncake](https://arxiv.org/abs/2407.00079), abstract; [vLLM disaggregated prefilling docs](https://docs.vllm.ai/en/latest/features/disagg_prefill.html), experimental); prefix-aware distributed scheduling ([Preble](https://arxiv.org/abs/2407.00023), abstract).

**FRONTIER (preprints; abstract-level inspection):** [MegaScale-Infer](https://arxiv.org/abs/2504.02263) (2025); [TaiChi — PD aggregation or disaggregation](https://arxiv.org/abs/2508.01989) (2025); [PPD disaggregation for multi-turn serving](https://arxiv.org/abs/2603.13358) (2026); [Prefill-as-a-Service](https://arxiv.org/abs/2604.15039) (2026).

**LEGACY / INSUFFICIENT:** treating 1F1B or training bubble fractions as inference schedules; stretching tensor parallelism across slow links by default; "disaggregation always raises throughput"; cache hit rate as the routing objective; counting failure domains per GPU.

**PRODUCTION SOURCE TRACE**
- Repository: `vllm-project/vllm`
- Revision: `863475eb9cd78ae6a4a22f1b90df7fe85e0af438` (committed 2026-09-30)
- Verified: 2026-09-30; static inspection only; nothing executed.
- Files/symbols: `vllm/model_executor/layers/linear.py::{ColumnParallelLinear.forward, RowParallelLinear.forward}`; `vllm/distributed/communication_op.py::tensor_model_parallel_all_reduce`; `vllm/distributed/parallel_state.py::{GroupCoordinator.all_reduce, initialize_model_parallel}`; `vllm/distributed/kv_transfer/kv_connector/v1/base.py::KVConnectorBase_V1.{get_num_new_matched_tokens, update_state_after_alloc, build_connector_meta, request_finished, start_load_kv, wait_for_layer_load, save_kv_layer, wait_for_save, get_block_ids_with_load_errors}`.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`. Every reported number carries a label: **measured** (with hardware, software versions, message sizes, repetitions) or **simulated** (with the simulator's assumptions). Simulated results are not evidence about real hardware.

### LAB A — Placement Plan, Router, and Failure Domains (simulated)
- **Objective**: Write a placement plan for a synthetic model and cluster, then build a discrete-event simulator of replicas with a router and fault injection.
- **Pre-Registered Hypothesis**: With a skewed prefix distribution, pure affinity routing has lower goodput at SLO than load-bounded affinity (CLM-024); many small groups lose less capacity per fault than few large groups (CLM-025).
- **Independent Variables**: Replica count and group size, routing policy, prefix skew, fault and straggler injection.
- **Dependent Variables**: Goodput at SLO, p99 TTFT, per-replica load, capacity lost per fault, recovery time. All simulated.
- **Break & Falsify**: Find a prefix distribution where pure affinity wins; state the condition. If none exists in your sweep, report the sweep as bounded evidence.
- **Alignment**: Lessons 20.1 and 20.6.
- **Effort Estimate**: 3h total.

### LAB B — Collective Cost and Tensor Parallelism (measured)
- **Objective**: Measure all-reduce time versus message size and rank count with `torch.distributed` (or nccl-tests), fit $\alpha$ and $B_{bus}$, implement column/row-parallel linear layers, verify numerical parity with the unsharded layer, and complete the TP part of the source trace.
- **Pre-Registered Hypothesis**: At decode-sized messages, measured all-reduce time is within 2× of the fitted $\alpha$ and largely independent of size; predicted $2L\,T_{AR}$ is within 30% of the measured communication time per step (CLM-005).
- **Independent Variables**: Message size (16 KiB–64 MiB), rank count, link (intra-node, and inter-node if available), backend.
- **Dependent Variables**: Time per collective with synchronization (Lesson 2.3), fitted $\alpha$ and $B_{bus}$, parity error, communication share of step time. Measured; CPU-process runs labeled non-representative.
- **Break & Falsify**: Test CLM-026 if two nodes are available: compare TP across nodes with intra-node TP replicas at equal GPU count; report if cross-node TP matches.
- **Alignment**: Lessons 20.2–20.3.
- **Effort Estimate**: 3.5h total (plus source trace, counted once).

### LAB C — Forward-Only Pipeline, Expert Skew, and Context Blocks (simulated)
- **Objective**: Simulate the decode ring and one-shot prefill pipeline, an EP layer with skewed routing and redundant experts, and a context-parallel ring with block exchange.
- **Pre-Registered Hypothesis**: Measured cycle time in the simulator matches $\max(\sum(t_i+x_i), m\max t_i)$ (CLM-008); redundant experts reduce worst-rank load in proportion to the split.
- **Independent Variables**: Stages, stage times, in-flight batches, hop time, routing skew, redundancy, block size versus link bandwidth.
- **Dependent Variables**: Cycle, throughput, per-stage utilization, TPOT, worst-rank load, non-overlapped exchange time. All simulated.
- **Break & Falsify**: Add variable per-batch work (mixed prefill chunks and decode steps) and show where the equal-stage formula fails. The timeline must contain no backward work.
- **Alignment**: Lesson 20.4.
- **Effort Estimate**: 3h total.

### LAB D — Disaggregation and KV Transfer (measured transfer, simulated fleet)
- **Objective**: Measure tensor transfer time between two processes or hosts for KV-sized payloads (sequential and layer-wise), then simulate prefill and decode pools with that transfer model and compare with a colocated chunked-prefill deployment; complete the KV-connector part of the source trace.
- **Pre-Registered Hypothesis**: CLM-027—disaggregation raises goodput for long prompts with strict TPOT when transfer time is well below the TTFT budget, and matches or loses for short prompts with tight TTFT.
- **Independent Variables**: Prompt length distribution, cached fraction, link bandwidth, prefill:decode ratio, SLO pair, transfer mode.
- **Dependent Variables**: Transfer bytes and time (measured), TTFT decomposition, TPOT, goodput at SLO, link utilization (simulated).
- **Break & Falsify**: Inject transfer failures and multi-turn traffic; report where recompute or link saturation removes the benefit.
- **Alignment**: Lesson 20.5 and Incident 20.1.
- **Effort Estimate**: 3.5h total.

---

## 07 Break / Incident Scenarios

### Incident 20.1 — The Upgrade That Slowed Every Token

- **Incident Symptoms**: A team moved a dense model from 2 × TP4 replicas in one node to TP4 × PP2 replicas spanning two nodes to serve longer contexts, and enabled prefill/decode disaggregation with prefix-affinity routing. Afterwards: TPOT p99 roughly doubled; long-prompt TTFT became bimodal; one decode replica shows periodic stalls; prefill GPUs report high utilization and decode GPUs low; cache hit rate went up. No telemetry on collectives, per-stage busy time, or transfer bytes is available yet.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: stage hop or collective latency across nodes; pipeline stage imbalance or too few circulating batches; a straggler rank or degraded link; KV-transfer link saturation; failed KV loads forcing recompute; affinity hotspot on one decode replica; mis-set prefill:decode ratio; a scheduler or admission change (Module 04) unrelated to placement.
  2. *Rank Initial Plausibility*: Use the change list and symptom timing; state which symptoms each hypothesis explains and which it cannot.
  3. *Identify Missing Evidence*: Per-collective and per-hop timings, per-stage busy time, per-rank step time, transfer bytes/s and link utilization, load-error counts from the connector, per-replica load and queue, TTFT decomposition by prompt length.
  4. *Design Discriminating Tests*: Profile one decode step per rank; run the same replica with PP stages colocated; compare per-rank step times for a straggler; plot transfer time against prompt length and link utilization against arrival bursts; replay traffic with load-bounded routing.
  5. *Execute Causal Diagnosis*: Rank explanations with the evidence each test produced; allow more than one cause; state what remains unexplained.
  6. *Prescribe Mitigation and Prevention*: Immediate—shift traffic off the stalled replica, bound affinity by load, cap transfer concurrency. Long-term—rebalance stages or return TP to one link domain, fix the pool ratio, add transfer and collective telemetry, spare capacity per failure domain.
  7. *Remeasure*: Same offered load and boundaries: TPOT and TTFT distributions, goodput at SLO, link utilization, per-stage utilization.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Serving Two Workloads on a Four-Node Cluster

A company must serve one dense model to two workloads on a fixed cluster and decide the placement, whether to disaggregate, and how much spare capacity to hold.

**Fixture (synthetic; all values are exercise assumptions, not measurements):**
- **Model**: $P=70\times10^9$ parameters, bf16; $L=80$; 64 query heads, 8 KV heads, $d_{head}=128$; $d_{model}=8192$; bf16 KV.
- **Cluster**: 4 nodes × 8 GPUs, 76.0 GiB usable per GPU. Intra-node: $\alpha=15\ \mu s$, $B_{bus}=200$ GB/s. Inter-node: $\alpha=40\ \mu s$, achieved 40 GB/s per node pair. Weight storage read rate 2 GB/s per node.
- **Workload 1 (chat)**: 30 requests/s; prompt mean 1,500 tokens, 60% of each prompt shared across requests; output mean 300 tokens; SLO TTFT 500 ms, TPOT 40 ms.
- **Workload 2 (document QA)**: 2 requests/s; prompt mean 24,000 tokens, no sharing; output mean 500 tokens; SLO TTFT 4 s, TPOT 40 ms.
- **Reliability**: each GPU independently unavailable with $q=0.01$; the service must meet both SLOs with any one GPU down.
- **Compute model**: you may assume single-rank-equivalent decode step time 60 ms at a batch of 64 sequences and prefill throughput 20,000 tokens/s per TP4 group; state any further assumption you add.

**Required Deliverables**:
1. Placement plan: replicas and TP/PP/CP degrees per node, with memory fit in GB and GiB and headroom.
2. Communication budget: predicted collective and hop time per decode step and per prefill chunk for the plan, with units and the dominant term.
3. Forward-only pipeline or context-parallel analysis for Workload 2 (or a justified decision not to use either), with a timeline containing no backward work.
4. Disaggregation decision per workload: transfer bytes, per-request transfer time, aggregate link demand, expected TTFT in expectation form, and the falsifier.
5. Routing policy across replicas with a load bound and its evaluation metric.
6. Failure-domain analysis: capacity lost per fault, expected availability, straggler loss, recovery time, and the spare capacity that meets the reliability requirement.
7. Pinned vLLM source trace (TP communication and KV-connector contract).
8. Diagnosis and remediation for Incident 20.1.
9. Evidence ledger separating **measured**, **simulated**, and **derived** numbers, with open risks.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
At the pinned vLLM commit, trace where tensor-parallel layers communicate, how ranks are grouped, and the scheduler-side and worker-side KV-transfer methods including the failure-reporting path. State that inspection was static, list what was not traced (device communicator selection, concrete connectors), and separate it from anything you executed in the labs.

### Rubric Dimensions
- **Placement**: *Insufficient* picks a layout without fit arithmetic or mixes GB and GiB. *Competent* shows fit, headroom, and at least two candidate plans. *Strong* ties the choice to SLOs, topology, and failure domains and states what would change it.
- **Communication Modeling**: *Insufficient* quotes link speed. *Competent* computes per-step collective time with units. *Strong* identifies the dominant term per phase, reconciles prediction with measurement, and states model limits.
- **Parallelism Reasoning**: *Insufficient* applies training schedules to inference. *Competent* uses the forward-only models for PP, EP, and CP. *Strong* shows where they break (imbalance, skew, non-overlap) and tests it.
- **Disaggregation**: *Insufficient* adopts or rejects it by default. *Competent* computes transfer bytes, time, and link demand. *Strong* decides per workload with a falsifiable prediction, pool ratio, and failure handling grounded in the traced contract.
- **Reliability**: *Insufficient* counts GPUs. *Competent* computes blast radius and spares per group. *Strong* includes stragglers, correlated domains, state loss, and recovery time.
- **Evidence Discipline**: *Insufficient* mixes simulated and measured numbers. *Competent* labels each. *Strong* also separates O/D/H, pins sources, and lists non-claims.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Placement from fit, SLO, and topology | 20.1 | LAB A | Deliverable 1 | Fit arithmetic in GB/GiB; candidate plans |
| Collective cost prediction and measurement | 20.2 | LAB B | Deliverable 2; Incident steps 3–4 | Fitted $\alpha$, $B_{bus}$; predicted vs measured step communication |
| Tensor parallelism and source trace | 20.3 | LAB B | Deliverables 2, 7 | Parity test; trace artifact with symbols |
| Forward-only PP, EP, CP reasoning | 20.4 | LAB C | Deliverable 3; Incident steps 1, 4 | Simulated ring results; timeline without backward work |
| Disaggregation and KV transfer | 20.5 | LAB D | Deliverables 4, 7; Incident steps 3–6 | Transfer measurements; goodput comparison; connector trace |
| Routing, stragglers, failure, recovery | 20.6 | LAB A | Deliverables 5, 6; Incident steps 5–7 | Simulated hotspot and fault results; spare-capacity plan |
| Evidence labeling | 20.1–20.6 | LABs A–D | Deliverable 9 | Ledger of measured, simulated, derived |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 20 must be able to:
1. Derive placement plans from memory fit, SLOs, and topology, with correct units.
2. Predict collective time, name the dominant term for decode and prefill, and check the prediction by measurement.
3. Trace tensor-parallel communication and the KV-transfer contract in a pinned runtime.
4. Model forward-only pipelines, expert skew, and context parallelism without importing training schedules.
5. Decide prefill/decode disaggregation per workload from transfer cost, link demand, and SLO regime, with a falsifier.
6. Quantify blast radius, straggler loss, routing hotspots, recovery cost, and required spares.
7. Label every number as measured, simulated, or derived.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: Replicas add throughput and isolation for free; every split of a forward pass puts a synchronous exchange on each token's path and makes the group the unit of failure.
- **The Distributed Path**: fit → placement (replicas, then groups inside the fastest link domain) → communication budget with units → phase specialization only if transfer bytes/s fit the links and the SLO regime calls for it → routing bounded by load → spares per failure domain.
- Inference is forward-only. Decode is latency-dominated and autoregressive; prefill is bandwidth-dominated and parallelizable. Most distributed-inference mistakes come from applying the intuition of one to the other.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
