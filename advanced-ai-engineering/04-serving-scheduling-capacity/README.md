# Module 04 — Serving, Scheduling, & Capacity

## 00 Why This Module Exists
An LLM model weight checkpoint cannot serve user traffic on its own. Deploying a model into production requires a serving engine that manages request ingress, batches heterogeneous requests, schedules GPU compute across conflicting prefill and decode execution phases, and dynamically arbitrates constrained GPU High Bandwidth Memory (HBM).

When traffic surges or prompt distributions shift, static batching can waste capacity, long prefill work can delay active decodes, and KV growth can force queuing, rejection, preemption, migration, or recomputation. Which mechanism dominates is workload- and runtime-dependent and must be established from telemetry.

Mastering serving, scheduling, and capacity planning is what transforms an engineer from someone who merely runs inference scripts into a systems engineer capable of designing, provisioning, and defending production-grade LLM runtimes under rigorous Service Level Objectives (SLOs) and real-world economics.

**Module Orientation**
- **Engineering Problem**: Maximizing inference goodput and capacity utilization while enforcing strict bounds on Time-To-First-Token (TTFT) and Inter-Token Latency (ITL) under dynamic, multi-tenant workloads.
- **What You Will Do**: Implement a discrete-event continuous batching scheduler simulator, evaluate chunked prefill mechanics under heterogeneous prompt distributions, model queueing saturation knees using Little's Law and M/G/1 theory, diagnose a multi-factor production latency incident, trace vLLM scheduler execution paths, and design an enterprise serving platform capacity plan.
- **Environment**: A standard Python 3.10+ development environment for the scheduler simulator and queueing test harnesses (Labs A–D). Access to a single NVIDIA GPU (Ampere/Ada/Hopper) is optional but valuable for running live inference benchmarks.

## 01 Baseline Assumptions
- **Transformer Forward Pass & Dimensions**: You understand the sequence dimensions, self-attention tensor shapes, and linear projection FLOP counts for standard autoregressive decoders (Module 01).
- **GPU Execution Model & Memory Hierarchy**: You understand Streaming Multiprocessors (SMs), Tensor Cores, High Bandwidth Memory (HBM) bandwidth constraints, and the distinction between compute-bound and memory-bandwidth-bound regimes via the Roofline model (Module 02).
- **KV Cache Memory Footprint & Paged Management**: You know how to derive the logical K/V payload per token ($2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes per element}$), state what that estimate excludes, and understand how logical sequences map to physical blocks in paged KV memory pools (Module 03).

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
  instruction: 5h        # sum of lesson instruction estimates: 40+40+40+45+45+40+50 min
  guided_practice: 1.5h  # sum of in-lesson practice: 20 (4.1) + 15 (4.4) + 15 (4.5) + 20 (4.6) + 20 (4.7) min; 4.2/4.3 practice is inside Labs A/B
  labs: 14h              # LAB A 4h + LAB B 3.5h + LAB C 3h + LAB D 3.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 04.1 0.5h
  source_trace: 2h       # Lesson 4.5 guided trace and the Section 09 Production Source Trace artifact are one activity, counted once here
  total: 25.5h
```
Each category is counted once. The source trace is not also counted as Lesson 4.5 practice or assessment time, and lab analysis is not counted again as guided practice.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map
Request Arrival $\to$ Ingress Control $\to$ Admission / Waiting Queue $\to$ Iteration-Level Scheduler $\to$ Prefill and Decode Work $\to$ GPU Compute, Memory-Bandwidth, and KV-Capacity Constraints $\to$ Optional Preemption / Migration / Recomputation $\to$ Output Streaming $\to$ Completion or Cancellation $\to$ Telemetry Feedback. Queueing, rate limiting, backpressure, load shedding, and capacity planning act at different points in this lifecycle; their effectiveness depends on workload, runtime, hardware, and client behavior.

## 04 Lessons

### Lesson 4.1 — The Serving Request Lifecycle & Metrics Decomposition

**Engineering Question:**
How do we define client-visible latency components precisely enough to narrow competing explanations for an SLO violation?

**Concepts & Definitions:**
- **Request Lifecycle**: An incoming request can pass through client/network transit, gateway processing, tokenization and validation, admission, scheduler waiting, prompt processing, first-token sampling and emission, autoregressive decode, output buffering/streaming, and completion or cancellation. Implementations may overlap or reorder some stages, so timestamps must be defined at explicit boundaries.
- **Time-to-First-Token (TTFT)**: The elapsed wall-clock duration between a declared request-arrival boundary and receipt of the first output event. Client-observed TTFT can include ingress and egress transit, gateway and host work, scheduler queueing, prompt processing, sampling, serialization, and buffering; engine-reported TTFT may use narrower boundaries.
- **Inter-Token Latency (ITL)**: The elapsed duration between consecutive output events at a declared observation boundary. Client-observed ITL can include scheduling, execution, sampling, buffering, and transport; an engine-side token timestamp uses a narrower definition. In a streaming interface, the upper tail affects perceived generation smoothness.
- **Time-Per-Output-Token (TPOT)**: The average decode latency across all generated tokens for a request: $(T_{e2e} - T_{TTFT}) / (N_{out} - 1)$. While TPOT represents the mean, ITL tracks the per-token distribution, capturing tail jitter and stalls. Aggregating across requests requires a declared weighting. A request-weighted mean averages each request's TPOT. A gap-weighted mean divides all post-first-output time by all inter-output gaps. These differ when output length and TPOT are correlated (see Lesson 4.7, Step 2).
- **End-to-End Latency ($T_{e2e}$)**: Total wall-clock time from initial request arrival to final token emission.

**Quantitative Model / Derivation:**
For a request arriving at time $t_0$, scheduled for prefill at $t_1$, emitting token 1 at $t_2$, and emitting subsequent tokens at $t_3, t_4, \dots, t_{N_{out}}$:
$$T_{queue} = t_1 - t_0$$
$$T_{first\_output\_after\_queue} = t_2 - t_1$$
$$T_{TTFT} = t_2 - t_0 = T_{queue} + T_{first\_output\_after\_queue}$$
Let $o_i$ be the client-observed arrival time of output token or output event $i$, with $o_1=t_2$:
$$ITL_i = o_i-o_{i-1} \quad \text{for } i \in [2,N_{out}]$$
$$T_{e2e} = o_{N_{out}}-t_0 = T_{TTFT}+\sum_{i=2}^{N_{out}}ITL_i$$
$$TPOT = \frac{T_{e2e}-T_{TTFT}}{N_{out}-1}=\frac{1}{N_{out}-1}\sum_{i=2}^{N_{out}}ITL_i$$
These identities assume one observed event per token (**D**, CLM-008). If a runtime emits multiple tokens per event, event-level ITL and token-amortized TPOT differ.

*Where*: $t_0$ = request arrival at the declared boundary; $t_1$ = start of prefill scheduling; $t_2=o_1$ = first output event; $o_i$ = arrival of output event $i$; $N_{out}$ = number of output events (tokens, under the one-event-per-token assumption); $ITL_i$ = gap before event $i$.

*Step-by-step (why the sum and the $N_{out}-1$ appear)*:
1. The $N_{out}$ output events define $N_{out}-1$ gaps: $o_1\to o_2,\ \dots,\ o_{N_{out}-1}\to o_{N_{out}}$. The first token has no preceding gap; its latency is already in TTFT.
2. The gaps telescope: $\sum_{i=2}^{N_{out}}(o_i-o_{i-1}) = o_{N_{out}}-o_1$.
3. Add $o_1-t_0=T_{TTFT}$ to both sides: $T_{TTFT}+\sum_{i=2}^{N_{out}} ITL_i = o_{N_{out}}-t_0 = T_{e2e}$.
4. TPOT is the mean of those $N_{out}-1$ gaps, i.e. the post-first-output time divided by the gap count.

*Assumptions*: $t_0$, $t_1$, and the output timestamps share a clock and explicitly defined observation boundary; generation produces $N_{out} \ge 2$ output events. A narrower engine-side decomposition must separately account for host, sampling, serialization, buffering, and network time before comparing it with client-observed TTFT.

**Worked Example:**
A client issues a request with a 2,048-token prompt.
- Queue wait time: $T_{queue} = 120\text{ ms}$.
- Measured first-output work after queue admission: $T_{first\_output\_after\_queue} = 65\text{ ms}$.
- Output length: $N_{out} = 200\text{ tokens}$.
- Total autoregressive decode iterations take $4,975\text{ ms}$. This is the post-first-output time $o_{N_{out}}-o_1=\sum_{i=2}^{N_{out}}ITL_i$, spread over $200-1=199$ gaps.
- $T_{TTFT} = 120\text{ ms} + 65\text{ ms} = 185\text{ ms}$.
- $TPOT = 4,975\text{ ms} / (200 - 1) = 25.0\text{ ms/token}$.
- $T_{e2e} = 185\text{ ms} + 4,975\text{ ms} = 5,160\text{ ms}$.
If the target SLO requires $T_{TTFT} < 100\text{ ms}$, the system is violating the SLO. The breakdown demonstrates that $T_{queue}$ alone ($120\text{ ms}$) exceeds the entire $100\text{ ms}$ budget before compute ever begins. Optimizing the prefill kernel will not restore compliance; the queue delay must be resolved.

**Knowledge Check:**
1. A monitoring dashboard reports that mean $T_{e2e}$ increased by $400\text{ ms}$, but mean $TPOT$ remained completely unchanged at $20\text{ ms/token}$. Assuming output token count is constant, which metric component accounts for the entire regression?
2. Why is tracking P99 ITL critical for conversational chat applications even if mean TPOT meets the 30 ms target?

**Guided Practice:**
An inference node serves a steady stream of requests with $N_{in} = 1,024$ and $N_{out} = 128$.
Under load, telemetry reveals:
- $T_{queue} = 240\text{ ms}$
- $T_{prefill} = 40\text{ ms}$
- 126 decode steps take $22\text{ ms}$ each.
- Step 45 experiences an anomalous pause of $410\text{ ms}$ before subsequent steps resume at $22\text{ ms}$.
Assume an output contains 128 tokens: 126 inter-token gaps are $22\text{ ms}$ and one gap is $410\text{ ms}$. Calculate: (a) $T_{TTFT}$, (b) mean $TPOT$, (c) maximum ITL, and (d) $T_{e2e}$. Then calculate empirical P99 under a stated quantile convention rather than silently equating P99 with the maximum.

**Feedback Contract:**
- *Expected Evidence*: (a) $T_{TTFT}=280\text{ ms}$. (b) Total post-first-output time $=(126\times22)+410=3,182\text{ ms}$ and mean $TPOT=3,182/127\approx25.06\text{ ms/token}$. (c) Maximum ITL is $410\text{ ms}$. With only 127 gaps, common empirical P99 conventions need not return the maximum; the convention and sample size must be reported. (d) $T_{e2e}=3,462\text{ ms}$. The post-first-output anomaly leaves TTFT unchanged but affects TPOT and the upper tail/max of ITL.
- *Common Failure*: Blaming the 410 ms spike on prefill queueing or adding 410 ms to TTFT.
- *Diagnostic Hint*: At what step index did the 410 ms pause occur? Was token 1 already emitted?
- *Concept to Revisit*: Latency Decomposition Invariants.

**Learning Outcome:**
Decompose end-to-end serving latency into mathematically precise lifecycle stages and attribute SLO breaches to queueing, prefill compute, or decode jitter.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 4.2 — Static vs. Continuous Batching Mechanics

**Engineering Question:**
When does fixed batch membership waste execution slots in autoregressive workloads, and what scheduler mechanisms make iteration-level admission possible?

**Concepts & Definitions:**
- **Static Batching**: Requests are grouped into a fixed batch of size $B$ at arrival. The entire batch executes synchronously through prefill and all decode steps until every single request in the batch emits its end-of-sequence (`<eos>`) token or hits `max_tokens`.
- **Execution Bubbles (Wasted Slots)**: When requests in a fixed-membership batch have uneven completion lengths, finished slots cannot be replaced until the batch boundary. Some implementations execute masked or padded work for inactive slots; others avoid part of that work but still lose the opportunity to admit waiting requests.
- **Continuous Batching (Iteration-Level Scheduling)**: Batch composition is re-evaluated dynamically at each discrete token-generation step. Terminated sequences immediately exit the batch and release their resources, while waiting requests are admitted into empty batch slots on the very next iteration (**O**, CLM-001).

**Mechanism Explanation:**
In static batching, batch execution duration is governed by the maximum output length: $T_{batch} = \max_{j \in [1, B]} (N_{out, j}) \times \tau_{step}$. The total useful token compute is $\sum_{j=1}^B N_{out, j}$, while the expended compute is $B \times \max_{j} (N_{out, j})$.
Continuous batching decouples individual request lifecycles. At iteration $k$:
1. Check termination condition (`<eos>` or length limit) for each running sequence.
2. Evict finished sequences and release their KV cache blocks to the free pool.
3. If free slots and KV memory exist, select requests from the waiting queue and insert them into the active execution batch.
4. Execute one forward pass across all active sequences.

**Quantitative Model / Derivation:**
As an analytical toy model, let independent generation lengths be continuous $N_{out,j}\sim\mathrm{Uniform}(0,M)$. This is not a production output-length model.
Under static batching:
$$\mathbb{E}[\max_{j \in [1, B]} N_{out, j}] = \frac{B}{B + 1} M$$
$$\text{Efficiency}_{static} = \frac{\mathbb{E}[\sum_{j=1}^B N_{out, j}]}{B \cdot \mathbb{E}[\max_j N_{out, j}]} = \frac{B \cdot (M/2)}{B \cdot \frac{B}{B+1} M} = \frac{B + 1}{2B}$$
*Where*: $B$ = batch size; $N_{out,j}$ = output length of request $j$; $M$ = maximum possible output length; $\tau_{step}$ = time of one decode iteration (assumed constant in this toy model).

*Step-by-step*:
1. Numerator: by linearity of expectation, $\mathbb{E}[\sum_j N_{out,j}]=B\,\mathbb{E}[N_{out,j}]=B\cdot M/2$ (mean of $\mathrm{Uniform}(0,M)$).
2. CDF of the maximum: $\max_j N_{out,j}\le x$ only if every one of the $B$ independent lengths is $\le x$, so $\Pr[\max\le x]=(x/M)^B$ for $0\le x\le M$.
3. For a nonnegative variable bounded by $M$, $\mathbb{E}[X]=\int_0^M \Pr[X>x]\,dx$. Hence $\mathbb{E}[\max]=\int_0^M\left(1-(x/M)^B\right)dx = M-\frac{M}{B+1}=\frac{B}{B+1}M$.
4. Divide: $\frac{B M/2}{B\cdot\frac{B}{B+1}M}=\frac{B+1}{2B}$.
5. Check: $B=1$ gives $1$ (no bubbles with one request); $B=4$ gives $5/8=62.5\%$; $B\to\infty$ gives $1/2$.

For large $B$, this toy-model slot-occupancy ratio approaches $50\%$. It does not prove that half of GPU compute is wasted: masked execution, kernel shapes, memory traffic, admission gaps, and batching overhead determine measured resource waste.
Continuous batching can refill eligible slots and improve occupancy, but it does not imply $100\%$ useful hardware utilization or a universal throughput multiplier. Admission frequency, prefill interference, KV availability, scheduler overhead, and workload shape must be measured.

**Worked Example:**
Consider a static batch of 4 requests with generation lengths:
- Req 1: 10 tokens
- Req 2: 50 tokens
- Req 3: 20 tokens
- Req 4: 100 tokens
Total useful tokens: $10 + 50 + 20 + 100 = 180\text{ tokens}$.
Static batch duration: $100\text{ iterations}$.
Total slots processed: $4 \times 100 = 400\text{ token slots}$.
Bubble waste: $400 - 180 = 220\text{ wasted slots}$ ($55\%$ waste).
Equivalently, $\text{Efficiency}_{static}=180/400=45\%$. This specific batch is not drawn from the uniform toy model, so it need not match the $B=4$ toy value of $62.5\%$.
Under continuous batching, Req 1 vacates its slot at step 10, permitting Req 5 from the queue to start immediately.

**Knowledge Check:**
1. Under what theoretical workload distribution would static batching achieve identical efficiency to continuous batching?
2. Why can continuous batching introduce higher variance in decode ITL compared to static batching?

**Feedback Contract:**
- *Expected Evidence*: (1) A workload where all requests have strictly identical prompt and generation lengths. (2) Because new requests entering the batch introduce prefill compute or change the active batch size dynamically, altering step execution time.
- *Diagnostic Hint*: When does static batching have zero padding bubbles?
- *Concept to Revisit*: Static Batch Bubble Derivation.

**Independent Practice:**
Proceed to **LAB A** to construct the discrete-event scheduler simulator and measure throughput differences between static and continuous batching.

**Learning Outcome:**
Quantify the compute bubble waste of static batching and explain the iteration-level state transitions of continuous batching.

*(Effort: 40m instruction, Lab A integration)*

---

### Lesson 4.3 — Prefill/Decode Interference & Chunked Prefill

**Engineering Question:**
When does long prefill work interfere with active decodes, and how should chunk size and token budget be chosen without assuming chunking improves every metric?

**Concepts & Definitions:**
- **Prefill vs. Decode Regimes**: Prefill exposes parallel work across prompt tokens; decode advances active sequences incrementally. Prefill often reaches higher arithmetic intensity and decode often becomes sensitive to weight/KV traffic, but model, batch size, sequence length, precision, kernels, and hardware can change either regime (**O**, CLM-002).
- **Inter-Phase Interference**: Long prefill work can delay decode through launch ordering or contention for SM, cache, HBM, and runtime budgets. A time-aligned scheduler trace plus GPU counters is required to identify the actual cause (**H**, CLM-002b).
- **Chunked Prefill**: Dividing an input prompt of length $N_{in}$ into multiple chunks of size at most $C$ (e.g., $C = 512$). In a simple fixed-size model the prompt needs $\lceil N_{in}/C\rceil$ chunks; block alignment and runtime-specific constraints can alter boundaries. Chunking creates opportunities to schedule decode work between or alongside chunks, but does not guarantee that every iteration contains both phases.

**Mechanism Explanation:**
In a hypothetical scheduler that executes an arriving 4,096-token prompt as one non-overlapped prefill step, a measured $150\text{ ms}$ step can add a comparable delay to decodes queued behind it. Actual overlap and contention must be established from a timeline.
In the Sarathi-Serve scheduling pattern (**O**, CLM-003):
1. The scheduler maintains a total token budget per iteration: $T_{budget}$ (e.g., 512 tokens).
2. Active decodes are scheduled first: $B_{dec}$ tokens.
3. The next prefill allocation is bounded by both the configured chunk limit and remaining token budget, e.g. $C_{next}\le\min(C_{max},T_{budget}-B_{dec})$ in a simplified token-count model.
4. The long prompt executes chunk 1 ($C$ tokens), appends intermediate KV states to its block table, and yields execution.
5. In iteration 2, the next decode tokens execute alongside chunk 2.
The work budget limits scheduled token work, but does not create a strict wall-clock bound: token cost, cache state, kernel choice, synchronization, and hardware state can still vary.

**Quantitative Model / Derivation:**
Use a measured step-cost function rather than treating tokens as equal-cost work:
$$T_{step}=f(N_{prefill},N_{decode},\text{lengths},\text{cache state},\text{kernels},\text{hardware})$$
*Where*: $T_{step}$ = wall-clock time of one scheduler iteration; $N_{prefill}$ = prompt tokens scheduled in the iteration; $N_{decode}$ = decode tokens scheduled (one per running sequence, so $N_{decode}=B_{dec}$ in the list above); $T_{budget}$ = token budget per iteration (a token count, not a time); $C_{max}$ = configured chunk limit; $C_{next}$ = prefill tokens granted to the next chunk.

*Example of the budget rule*: with $T_{budget}=512$, $C_{max}=512$, and $B_{dec}=100$ running decodes, $C_{next}\le\min(512,\,512-100)=412$ prompt tokens this iteration.

The scheduler enforces a work constraint such as $N_{prefill}+N_{decode}\le T_{budget}$, while experiments estimate $f$. Smaller chunks can shorten individual decode interruptions but add more scheduling steps and may reduce kernel efficiency. Larger chunks can improve prefill efficiency while worsening ITL. TTFT, ITL, throughput, fairness, and goodput must all be remeasured.

**Worked Example:**
Treat the following values as exercise inputs, not hardware facts: a decode-only step measures $18\text{ ms}$, an unchunked 4,096-token prefill step measures $120\text{ ms}$, and eight 512-token chunks measure $35\text{ ms}$ each when mixed with the same decode load. Compare the longest injected step and cumulative prompt-processing time.
- Chunk count: $\lceil 4{,}096/512\rceil = 8$.
- Longest step a running decode can be stuck behind: $120\text{ ms}$ unchunked vs. $35\text{ ms}$ chunked (vs. $18\text{ ms}$ with no prompt in flight).
- Cumulative time of steps containing the prompt: $120\text{ ms}$ unchunked vs. $8\times35=280\text{ ms}$ chunked. Those chunked steps also advance decodes, so this is not pure prefill cost, but the prompt's own processing span grows.

Exact request TTFT and ITL require the arrival order, batch composition, and scheduling timeline, so construct that timeline before calculating them. Then test whether the measurements reproduce across randomized runs and include a counterexample workload with short prompts or high chunk overhead.

**Knowledge Check:**
1. Does chunking necessarily increase mathematical model FLOPs? Which implementation costs can nevertheless increase measured work or elapsed time?
2. If your workload consists exclusively of short prompts ($N_{in} \le 64$) and long decodes ($N_{out} \ge 1,024$), is chunked prefill necessary?

**Feedback Contract:**
- *Expected Evidence*: (1) No: causal-attention arithmetic can be organized without a universal increase in mathematical FLOPs, while kernel shapes, repeated scheduling/launch work, block boundaries, and cache traffic can increase measured cost. (2) Chunking may be unnecessary or harmful for short prompts, but the decision requires measured ITL and throughput rather than prompt length alone.
- *Diagnostic Hint*: How does attention work when prompt tokens are split across iterations?
- *Concept to Revisit*: Causal Attention Across Chunk Boundaries.

**Independent Practice:**
Proceed to **LAB B** to measure ITL variance under heterogeneous prompt arrivals with and without chunked prefill.

**Learning Outcome:**
Diagnose prefill/decode interference and select chunk budgets from workload-faithful TTFT, ITL, throughput, and goodput measurements.

*(Effort: 40m instruction, Lab B integration)*

---

### Lesson 4.4 — Queueing Dynamics, Little's Law, & The Saturation Knee

**Engineering Question:**
Under which assumptions does queueing delay grow nonlinearly near capacity, and how should analytical mean-value checks be combined with empirical load sweeps to select operating headroom?

**Concepts & Definitions:**
- **Arrival Rate ($\lambda$)**: Mean number of requests arriving per second.
- **Service Rate ($\mu$)**: In a single-server queue model with independent service times $S$, $\mu=1/\mathbb{E}[S]$. A batching LLM runtime has a state- and workload-dependent completion curve, so an empirical "requests/s" capacity is not automatically this model parameter.
- **Traffic Intensity / Utilization ($\rho$)**: In the stated single-server model, $\rho=\lambda\mathbb{E}[S]=\lambda/\mu$ and stationarity requires $\rho<1$. GPU utilization counters and token throughput are different quantities.
- **Little's Law**: Fundamental queueing theorem:
  $$L = \lambda W$$
  *Intuition*: draw each request as a bar from entry to exit. Area under the "requests present" curve over a window $[0,T]$ equals the summed bar lengths. Divide both by $T$: left side is $L$; right side is $(\text{arrivals}/T)\times(\text{mean bar length})=\lambda W$. Example: 4 req/s each staying 0.5 s means $L=2$ requests present on average.
  The average number of requests in the system ($L$) equals arrival rate ($\lambda$) multiplied by average residence time ($W = T_{e2e}$).
  Similarly, for the waiting queue alone: $L_q = \lambda W_q$, where $L_q$ is average queue depth and $W_q$ is average queue wait time ($T_{queue}$).
  *Conditions* ([Little, 2011, §2 and §3](https://pubsonline.informs.org/doi/10.1287/opre.1110.0940)):
  - *Finite window $[0,T]$*: If the boundary is empty at both $0$ and $T$, then $L=\lambda W$ holds exactly, with $L$ the time-average count, $\lambda$ = arrivals$/T$, and $W$ the mean time in the boundary. If requests are present at either end, the identity still holds only when $W$ counts time accrued inside the window. The full sojourn of a request that straddles the window edge does not satisfy it.
  - *Long run*: The identity holds when the long-run arrival rate and mean residence limits exist and are finite. It needs neither Poisson arrivals, stationarity, nor a particular queue discipline.
  - *Boundary*: $L$, $\lambda$, and $W$ must describe the same boundary and population. Use admitted arrivals, not offered arrivals, when rejected requests never enter. $L$ for an end-to-end boundary counts waiting, resident, and still-streaming requests. It is not the number of resident GPU sequences holding KV. That count needs its own boundary: $L_{resident}=\lambda_{admitted}\,\mathbb{E}[T_{resident}]$.
  - Little's Law is a conservation identity for averages (**D**, CLM-004). It says nothing about percentiles, and it does not predict $W$ from $\lambda$ without a service model.
- **Saturation Knee**: An empirically observed workload-specific region where additional offered load causes queue delay, rejection, or SLO misses to rise sharply relative to useful completions. There is no universal safe-utilization interval.

**Quantitative Model / Derivation:**
Approximating the serving system as an M/G/1 queue (Poisson arrivals, general service time distribution with mean $1/\mu$ and variance $\sigma^2$):
According to the **Pollaczek-Khinchine (P-K) formula** (**D**, CLM-005):
$$W_q = \frac{\lambda (\sigma^2 + 1/\mu^2)}{2(1 - \rho)} = \frac{\rho \cdot \frac{1}{\mu} \left(1 + C_v^2\right)}{2(1 - \rho)}$$
where $C_v = \sigma / (1/\mu) = \sigma \mu$ is the coefficient of variation of service time.

*Where*: $S$ = service time of one request; $\mathbb{E}[S]=1/\mu$; $\sigma^2=\mathrm{Var}(S)$; $\rho=\lambda/\mu$; $W_q$ = mean time waiting before service starts. Mean time in system is $W=W_q+1/\mu$.

*Step-by-step (from the first form to the second)*:
1. The standard P-K form is $W_q=\dfrac{\lambda\,\mathbb{E}[S^2]}{2(1-\rho)}$.
2. Second moment from variance: $\mathbb{E}[S^2]=\mathrm{Var}(S)+\mathbb{E}[S]^2=\sigma^2+1/\mu^2$. This gives the first expression above.
3. Factor out $1/\mu^2$: $\sigma^2+1/\mu^2=\frac{1}{\mu^2}(\sigma^2\mu^2+1)=\frac{1}{\mu^2}(1+C_v^2)$.
4. Then $\lambda\cdot\frac{1}{\mu^2}=\frac{\lambda}{\mu}\cdot\frac{1}{\mu}=\rho\cdot\frac{1}{\mu}$, which gives the second expression.
5. Special cases: deterministic service ($C_v=0$, M/D/1) halves the wait of exponential service ($C_v=1$, M/M/1) at the same $\rho$.
Notice two critical dynamics:
1. **The $(1 - \rho)$ Denominator**: In the M/G/1 model, mean waiting diverges as $\rho \to 1$. This is model behavior, not a percentile guarantee for a batching LLM server (**D**, CLM-009).
2. **The Variance Term ($C_v^2$ or $\sigma^2$)**: Under the model assumptions, higher service-time variance increases mean queue wait. Real LLM service demand depends on prompt/output lengths, batching, cache state, and scheduler decisions, so the distribution must be measured rather than inferred from output length alone.

**Worked Example:**
A serving instance has maximum capacity $\mu = 10\text{ req/s}$ with mean service time $1/\mu = 100\text{ ms} = 0.1\text{ s}$.
Workload has standard deviation $\sigma = 0.1\text{ s}$ ($C_v = 1.0$).
- At $\lambda = 5\text{ req/s}$ ($\rho = 0.50$):
  $$W_q = \frac{0.50 \cdot 0.1 \cdot (1 + 1)}{2(1 - 0.50)} = \frac{0.10}{1.0} = 0.10\text{ s} = 100\text{ ms}$$
  $$L_q = \lambda W_q = 5 \times 0.10 = 0.5\text{ requests in queue}$$
- At $\lambda = 8\text{ req/s}$ ($\rho = 0.80$):
  $$W_q = \frac{0.80 \cdot 0.1 \cdot (1 + 1)}{2(1 - 0.80)} = \frac{0.16}{0.40} = 0.40\text{ s} = 400\text{ ms}$$
  $$L_q = 8 \times 0.40 = 3.2\text{ requests in queue}$$
- At $\lambda = 9.5\text{ req/s}$ ($\rho = 0.95$):
  $$W_q = \frac{0.95 \cdot 0.1 \cdot (1 + 1)}{2(1 - 0.95)} = \frac{0.19}{0.10} = 1.90\text{ s} = 1,900\text{ ms}$$
  $$L_q = 9.5 \times 1.90 = 18.05\text{ requests in queue}$$
Notice: Moving from $\rho = 0.80$ to $\rho = 0.95$ (a $9.5/8-1\approx19\%$ increase in arrival rate) causes queue wait time to jump by nearly $5\times$ ($400\text{ ms} \to 1,900\text{ ms}$, a factor of $4.75$).
Why: the numerator grows only by $0.95/0.80\approx1.19$, while the denominator shrinks from $2(1-0.80)=0.40$ to $2(1-0.95)=0.10$, a factor of $4$. The product $1.1875\times4=4.75$ is the jump.
Mean time in system adds the $100\text{ ms}$ service time: $W=W_q+1/\mu=200$, $500$, and $2{,}000\text{ ms}$ at the three loads.

**Knowledge Check:**
1. If your average response time is $W = 2.5\text{ seconds}$ and your target peak arrival rate is $\lambda = 40\text{ req/s}$, what is the average concurrency $L$ that your cluster must maintain in steady state?
2. If an initially empty deterministic fluid queue receives $\lambda=15$ req/s while departures remain fixed at $\mu=10$ req/s for 20 seconds, with no drops or cancellations, what backlog does that model predict?

**Feedback Contract:**
- *Expected Evidence*: (1) By Little's Law, $L = \lambda W = 40 \times 2.5 = 100$ concurrent requests. (2) Unstable queue growth: $dL_q/dt = \lambda - \mu = 15 - 10 = 5\text{ req/s}$. Over 20 seconds, accumulated queue depth $= 5 \times 20 = 100$ requests waiting. On the hint: the finite-window form of Little's Law still holds as an accounting identity over the burst window if $W$ counts only in-window time. The long-run form cannot be used to predict residence time during the burst, because no limit exists while backlog grows.
- *Diagnostic Hint*: Can Little's Law be applied during the transient 20-second burst when $\lambda > \mu$?
- *Concept to Revisit*: Steady-State Assumption in Little's Law vs Transient Accumulation.

**Guided Practice:**
A capacity engineer observes P50 TTFT of $80\text{ ms}$ and P99 TTFT above $3,500\text{ ms}$. Use the P-K formula only to predict how the **mean** wait changes under an explicit M/G/1 approximation. Then state why the formula cannot explain or guarantee P99, and design an open-loop load experiment that measures the actual tail under the changed workload.

**Learning Outcome:**
Apply Little's Law and qualified M/G/1 intuition as mean-value checks, then determine SLO-constrained headroom empirically.

*(Effort: 45m instruction, 15m practice)*

---

### Lesson 4.5 — Runtime Scheduling Policies & Preemption Under Memory Pressure

**Engineering Question:**
How do serving runtimes respond when scheduled work cannot obtain KV capacity, and which parts of that behavior are general mechanisms versus version-specific policy?

**Concepts & Definitions:**
- **General mechanisms**: Delay admission, reject work, preempt and recompute, pause while retaining state, migrate/offload state, or reserve capacity. Each has latency, fairness, compute, bandwidth, and memory costs.
- **Current pinned vLLM V1 observation**: At commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`, the scheduler maintains `running` and request queues. When `KVCacheManager.allocate_slots()` fails, it can select a victim and call `_preempt_request()`, which frees KV blocks, marks the request `PREEMPTED`, resets `num_computed_tokens`, increments its counter, and prepends it to the waiting queue (**O**, CLM-013).
- **Historical vLLM V0 observation**: Older source/paper versions described SWAP versus RECOMPUTE modes. That path is useful as a design comparison, not as the current vLLM definition (**O**, CLM-007).
- **Thrashing hypothesis**: Repeated preemption or migration can reduce completion goodput, but counters and traces must show the loop before it is diagnosed (**H**, CLM-009b).

**Mechanism Explanation:**
At the pinned V1 revision, the relevant path is conceptually:
```python
new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
if new_blocks is None:
    victim = select_victim(policy, running)
    self._preempt_request(victim, scheduled_timestamp)
```
This pseudocode is explanatory; the required artifact must cite the actual pinned symbols and execution path. Victim selection depends on configured scheduling policy and queue state.

**Quantitative Model / Trade-off Comparison:**
- *Idealized cost of transferring state*:
  $$T_{swap\_roundtrip} = \frac{2 \times \text{KV\_Size\_Bytes}}{\text{PCIe\_Effective\_Bandwidth}} + 2 \times \text{DMA\_Overhead}$$
  *Where*: $\text{KV\_Size\_Bytes}$ = KV state of the preempted request(s); the factor 2 counts one copy out (device to host) and one copy back; $\text{PCIe\_Effective\_Bandwidth}$ = measured bytes/s of the host link, not the link's nominal rate; $\text{DMA\_Overhead}$ = fixed setup/synchronization cost per transfer.
  For an illustrative $4\text{ GiB}$ state and a measured $25\text{ GiB/s}$ effective transfer path, the payload-only round-trip lower bound is:
  $$T_{swap\_roundtrip} \ge \frac{2 \times 4\text{ GiB}}{25\text{ GiB/s}} = 0.32\text{ s} = 320\text{ ms}$$
- *Idealized cost of recomputation*:
  $$T_{recompute} = T_{prefill}(N_{in} + N_{gen\_so\_far}) \approx \frac{2 \cdot P \cdot (N_{in} + N_{gen\_so\_far})}{\text{Peak\_Compute\_FLOPs}}$$
  Here $N_{in}$ = prompt length and $N_{gen\_so\_far}$ = tokens already generated before preemption; both must be reprocessed because the KV for all of them was freed. $P$ is the absolute parameter count, not billions. $2P$ FLOP per token is the dense matrix-multiply approximation and excludes attention-score work, which grows with context length. The denominator is in FLOP/s, so the result is in seconds.
These are lower-bound comparisons. Effective bandwidth, overlap, serialization, current load, kernel efficiency, and timeout/cancellation behavior must be measured. Keep binary and decimal units apart: 1 GiB $=2^{30}$ bytes and 1 GB $=10^9$ bytes.

**Worked Example:**
*Inputs.* All values below are **synthetic exercise values** (CLM-016), not measurements of any system.
- Preempted state payload: 500 MiB. At the 128 KiB/token logical KV footprint of the Lesson 4.7 8B example, that is $500\cdot2^{20}/(128\cdot2^{10})=4{,}000$ tokens.
- Effective transfer bandwidth, stated as a synthetic "measured" value: 25 GiB/s in each direction. Transfer is not overlapped with other traffic.
- Recomputation of those 4,000 tokens on the same device, timed separately (synthetic value): 310 ms.
- Declared peak compute used only for an analytical bound (synthetic value): $500\times10^{12}$ FLOP/s.

*Step 1: transfer lower bound.* The unit conversion is $500\text{ MiB}/(25\text{ GiB/s})=500/(25\times1024)\text{ s}=19.53125\text{ ms}$ one way. Out and back:
$$T_{swap\_roundtrip}\ge2\times19.53125\text{ ms}=39.0625\text{ ms}$$
This is payload only. DMA setup, synchronization, block gather/scatter, and contention can only add time. If a real round-trip measurement comes out below 39.0625 ms, the bandwidth figure or the timing boundary is wrong.

*Step 2: analytical recompute lower bound.* This is for comparison, not prediction: $2\times(8\times10^9)\times4{,}000/(500\times10^{12})=0.128\text{ s}=128\text{ ms}$. The synthetic measured 310 ms lies above it, as expected, because kernels do not run at peak and attention work is excluded from the bound.

*Step 3: compare like with like.* Compare the measured recompute time (310 ms) with the measured transfer time. Neither should be replaced by a bound. The transfer *lower bound* is $310/39.0625\approx7.9\times$ smaller. The break-even effective bandwidth, at which payload-only transfer equals the 310 ms recompute, is $2\times500\text{ MiB}/0.310\text{ s}\approx3{,}226\text{ MiB/s}\approx3.15\text{ GiB/s}$.

*Interpretation and limits.* In this fixture, transfer wins only if its measured round trip, including overheads and contention, stays well below 310 ms. It must also not steal bandwidth that other requests need, and host memory must be available for the swapped state. The lower bound is not a latency prediction. The pinned vLLM V1 path in this lesson recomputes rather than swaps, so this comparison is a design exercise, not a description of that runtime. Recompute on a loaded device competes with other requests' prefill and decode work. Measure both alternatives under the same load before choosing (**D**, CLM-018).

**Knowledge Check:**
1. Under what condition does enabling CPU KV swapping degrade total cluster throughput worse than immediately aborting preempted requests?
2. How would FCFS, priority scheduling, youngest-first victim selection, and largest-memory-first selection change wasted work and fairness?

**Feedback Contract:**
- *Expected Evidence*: (1) A transfer policy is harmful when transfer/restore work consumes scarce bandwidth without increasing SLO-compliant completions. (2) No victim policy dominates universally: minimizing discarded work can conflict with fairness, priority, memory reclaimed, and deadlines.
- *Diagnostic Hint*: How much work has a request that just started done versus one that is at token 1,900 of 2,000?
- *Concept to Revisit*: LIFO Preemption Victim Selection.

**Guided Practice:**
Trace `vllm/v1/core/sched/scheduler.py` at commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`: `Scheduler.schedule()` → `KVCacheManager.allocate_slots()` failure → victim selection → `Scheduler._preempt_request()`. Contrast this verified V1 path with the older V0 SWAP/RECOMPUTE design. Also locate the separate `allocate_slots()` call for *waiting* requests, and state what happens when it fails. At the pinned revision, that failure stops admission for the step rather than selecting a victim. A trace that merges the two paths is incomplete.

**Learning Outcome:**
Analyze scheduler priority arbitration and evaluate when delay, rejection, recomputation, transfer, or reservation policies reduce useful work or create repeated preemption.

*(Effort: 45m instruction, 15m practice on the worked example and knowledge checks. The guided source trace is counted once, under `source_trace`.)*

---

### Lesson 4.6 — Admission Control, Backpressure, & Load Shedding

**Engineering Question:**
How do we prevent a serving system from entering goodput collapse when incoming request volume exceeds physical hardware capacity?

**Concepts & Definitions:**
- **Throughput vs. Goodput**:
  - *Throughput*: Total tokens or requests processed by the GPU per second, regardless of whether they meet SLOs or whether the client abandoned the connection.
  - *Goodput*: The rate of work that satisfies a declared usefulness predicate. In this module's exercises that predicate includes successful completion and stated latency SLOs; a production definition must also specify the measured population and treatment of rejections, cancellations, retries, and output-quality failures.
- **Goodput Collapse**: Under some timeout, retry, and cancellation policies, overload makes the system spend resources on work that no longer produces user-visible value. Goodput can fall even while raw throughput or device activity remains high.
- **Admission Control**: A gating mechanism at the gateway or scheduler boundary that evaluates whether sufficient system capacity exists to service a request within its SLO before admitting it into the queue.
- **Backpressure**: A signal or blocking mechanism that causes an upstream producer to slow down. HTTP 429 or gRPC `RESOURCE_EXHAUSTED` is rejection/rate-limit feedback; it becomes effective backpressure only if upstream behavior actually reduces offered load.
- **Load Shedding**: The deliberate, early rejection of excess requests when queues exceed a safe threshold, preserving full service capacity for already-admitted requests.

**Mechanism Explanation:**
Admission control algorithms employ four primary strategies:
1. **Concurrency Limits ($L_{max}$)**: Bound the total active requests in the serving engine:
   $$\text{If } |Queue_{waiting}| + |Queue_{running}| \ge L_{max} \implies \text{apply the configured admission outcome}$$
2. **Time-in-Queue Bounding (Virtual Deadline)**:
   A simple estimate $\widehat{W}_q=|Queue_{waiting}|/\mu$ requires homogeneous work and approximately constant service rate. Treat it as a calibrated predictor with uncertainty rather than proof that an SLO violation is guaranteed.
3. **Sojourn-Time Queue Control**: Use measured queue residence time as an overload signal and shed according to a declared policy. CoDel is one reference algorithm from packet queues; applying it to heterogeneous LLM jobs requires validation rather than copying its constants.
4. **Rate Limiting**: A token-bucket or related limiter bounds accepted arrivals over a time window and permits a configured burst. It controls ingress rate; it is not the same mechanism as sojourn-time-based queue dropping (**O**, CLM-010).

**Quantitative Model / Trade-off Comparison:**
For an initially empty deterministic fluid queue with fixed rates and no drops:
$$\lambda>\mu \implies Q(t)=(\lambda-\mu)t,\qquad \widehat{W}_q(t)=Q(t)/\mu$$
*Where*: $Q(t)$ = queue depth at time $t$ after overload begins; $\widehat{W}_q(t)$ = wait of a request that joins at $t$.

*Step-by-step*:
1. Requests enter at rate $\lambda$ and leave at rate $\mu$, so $dQ/dt=\lambda-\mu$. With $Q(0)=0$, integrating gives $Q(t)=(\lambda-\mu)t$.
2. Under FIFO a new arrival waits until the $Q(t)$ requests ahead of it drain at rate $\mu$: $\widehat{W}_q(t)=Q(t)/\mu$.
3. Combining: $\widehat{W}_q(t)=\frac{\lambda-\mu}{\mu}\,t$. Wait grows linearly and without bound while $\lambda>\mu$. A client timeout $T_{out}$ is first exceeded at $t^*=T_{out}\,\mu/(\lambda-\mu)$.

Client timeouts can reduce user-visible goodput when stale work is not cancelled, but neither the fluid model nor $\lambda>\mu$ alone proves goodput goes to zero. In a simplified steady-rate exercise, admitted rate cannot sustainably exceed measured completion capacity. Real goodput additionally depends on SLO attainment, failures, cancellations, retries, and output quality. Load shedding trades rejected-user utility for protection of selected objectives; it does not create 100% reliability.

**Worked Example:**
A fluid-model exercise assumes fixed completion capacity $\mu=100\text{ req/s}$, an initially empty queue, no service-rate change with batching, and a client timeout of $2.0\text{ s}$.
A traffic spike arrives at $\lambda = 250\text{ req/s}$ for 60 seconds.
- *Scenario A (No Admission Control, FIFO queue)*:
  Queue accumulates $(250 - 100) = 150\text{ req/s}$.
  After 15 seconds, queue depth reaches $15 \times 150 = 2,250\text{ requests}$.
  Wait time in queue for newly scheduled requests $= 2,250 / 100 = 22.5\text{ seconds}$.
  The modeled wait first crosses the $2.0\text{ s}$ timeout at $t^*=2.0\times100/150\approx1.33\text{ s}$ (queue depth $200$).
  Requests at the tail of this modeled backlog exceed the timeout. Exact goodput requires the arrival/departure timeline and cancellation semantics; it is not automatically zero.
- *Scenario B (Strict Admission Control with Load Shedding)*:
  Gateway limits queue size to 50 requests ($0.5\text{ s}$ max queue delay, from $\widehat{W}_q=50/100$).
  Gateway sheds the offered excess using the configured rejection contract.
  The admitted $100\text{ req/s}$ are processed with $T_{queue} < 500\text{ ms}$.
  The model predicts up to $100\text{ req/s}$ completions; measured SLO-compliant goodput may be lower.

**Knowledge Check:**
1. Under what timeout, retry, cancellation, and user-utility assumptions is early rejection preferable to queueing?
2. What client-side mechanism must accompany HTTP 429 load shedding to prevent an immediate retry storm (thundering herd)?

**Feedback Contract:**
- *Expected Evidence*: (1) Early rejection before engine admission avoids KV and model-execution work for that request, though gateway work remains. Queueing is harmful only when it later times out, is cancelled, or displaces more valuable work. (2) Retry budgets plus exponential backoff with jitter, or another policy that reduces synchronized retry load.
- *Diagnostic Hint*: What happens when 1,000 clients simultaneously retry an HTTP 429 after exactly 1.000 seconds?
- *Concept to Revisit*: Thundering Herd and Jittered Backoff.

**Guided Practice:**
Design an admission policy for an enterprise RAG endpoint. Prompt length is 3,000 tokens, generation length is 200 tokens, and the client timeout is 1,200 ms. The queue has 12 requests. Explain why a single aggregate "1,500 prompt tokens/sec" value is insufficient to decide admission, list the missing service/scheduling information, and propose a calibrated prediction with an uncertainty margin.

**Learning Outcome:**
Design, configure, and defend admission control and load shedding mechanisms to safeguard serving goodput under overload.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 4.7 — Capacity Planning & Bottleneck Discrimination

**Engineering Question:**
How do we rank interacting queueing, compute, host, network, and memory constraints, and how do we size a deployment without mistaking a lower-bound estimate for a capacity guarantee?

**Concepts & Definitions:**
- **Competing Constraint Hypotheses**:
  1. *Queueing pressure*: arrivals, bursts, or a reduced effective service rate create backlog; this describes system state, not a physical resource by itself.
  2. *Device execution pressure*: one or more kernels are limited by compute throughput, memory bandwidth, synchronization, launch behavior, or inefficient shapes. Aggregate SM activity alone does not identify which.
  3. *KV-capacity pressure*: allocation limits block admission or trigger runtime-specific preemption even when some compute counters appear low.
  4. *Host/network/control-plane pressure*: tokenization, serialization, communication, scheduler overhead, or downstream backpressure delays work outside the main GPU kernels.
Multiple constraints can interact or transition during one incident (**H**, CLM-011).

**Systematic Discrimination Matrix:**
To diagnose an underperforming cluster, collect the following telemetry metrics and cross-reference against the discrimination signatures:

| Signal | Supports a hypothesis when correlated with | Why it is insufficient alone |
|---|---|---|
| **Queue residence / depth** | arrival trace, completion rate, request-state transitions | backlog can be caused by any reduction in effective service rate |
| **Prefill/decode spans** | token counts, batch composition, kernel timeline | longer spans can reflect scheduler delay, shapes, contention, or host gaps |
| **GPU counters** | per-kernel arithmetic intensity, achieved bandwidth/FLOPs, occupancy | aggregate utilization hides idle gaps and mixed kernels |
| **Free KV blocks / allocation failures** | admission blocks, preemption events, sequence footprints | low free blocks may be an intentional steady state without harm |
| **Preemption rate** | recompute/transfer work and latency changes | correlation does not establish that preemption initiated the incident |
| **Host and network timeline** | CPU queues, serialization, transport, client timestamps | engine-only metrics omit end-to-end delays |

**Quantitative Capacity Planning Model:**
Given:
- Target peak arrival rate: $\lambda_{peak}\text{ req/s}$
- Mean prompt length: $\bar{N}_{in}$; Mean decode length: $\bar{N}_{out}$
- TTFT SLO: $T_{TTFT\_SLO}$; ITL SLO: $T_{ITL\_SLO}$
- Model parameters: $P$ = absolute parameter count (a "70B" model has $P=70\times10^9$); Precision: $B_{param}\text{ bytes/param}$
- GPU specs: HBM capacity $M_{gpu}$, Memory bandwidth $BW_{mem}$, FP16 Peak Tensor Compute $C_{peak}$

*Step 1: Weight and Static Memory Footprint*:
$$M_{weights,payload}\,[\text{bytes}] = P \times B_{param}$$
If $P$ is quoted in billions ($P_B$), use $M_{weights,payload}=10^9\,P_B\,B_{param}$ bytes. Convert explicitly: divide by $10^9$ for GB or by $2^{30}$ for GiB. Example: $70\times10^9\times2\text{ bytes}=140\times10^9\text{ bytes}=140\text{ GB}\approx130.39\text{ GiB}$, *not* 140 GiB. This is the logical payload. The resident weight footprint $M_{weights,resident}$ also depends on the runtime, sharding, replicated tensors, and allocator, so measure it or declare it as an exercise input (**D**, CLM-017).
$$M_{KV\_available} = M_{device,usable} - M_{weights,resident} - M_{nonKV,measured} - M_{headroom}$$
The non-KV term and allocator/headroom policy are measured for the selected runtime and configuration; they are not universal constants.

*Step 2: Concurrency Sizing via Little's Law*:
Per request $r$ with $N_r\ge2$ output events at one observation boundary, Lesson 4.1 gives the exact identity $T_{e2e,r}=TTFT_r+(N_r-1)\,TPOT_r$. A request with $N_r=1$ has $T_{e2e,r}=TTFT_r$ and no TPOT, so report it as a separate class; do not assign it a TPOT of zero. Averaging over the same population of $N\ge2$ requests:
$$\mathbb{E}[T_{e2e}] = \mathbb{E}[TTFT] + \mathbb{E}[(N_{out}-1)\,TPOT]$$
$$\mathbb{E}[(N_{out}-1)\,TPOT]=(\mathbb{E}[N_{out}]-1)\,\mathbb{E}[TPOT]+\mathrm{Cov}(N_{out},TPOT)$$
*Step-by-step*:
1. Take expectations of the per-request identity; expectation is linear, so $\mathbb{E}[T_{e2e}]=\mathbb{E}[TTFT]+\mathbb{E}[(N_{out}-1)TPOT]$.
2. For any two random variables, $\mathrm{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$. Rearranged: $\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y]+\mathrm{Cov}(X,Y)$.
3. Set $X=N_{out}-1$ and $Y=TPOT$. Since $\mathbb{E}[N_{out}-1]=\mathbb{E}[N_{out}]-1$ and subtracting a constant does not change covariance, this gives the second line.
4. Gap-weighted case: $(N_r-1)\,TPOT_r=T_{e2e,r}-TTFT_r$ by definition. Averaging over $n$ requests, $\frac{1}{n}\sum_r(T_{e2e,r}-TTFT_r)=\frac{\sum_r(N_r-1)}{n}\cdot\frac{\sum_r(T_{e2e,r}-TTFT_r)}{\sum_r(N_r-1)}=(\bar N_{out}-1)\,\overline{ITL}_{gap}$ with no covariance term.

The product of means $(\bar N_{out}-1)\times\overline{TPOT}$ is therefore exact only in two cases:
- the covariance is zero, for example when TPOT does not vary with output length; or
- $\overline{TPOT}$ is replaced by the gap-weighted mean over the same population, $\overline{ITL}_{gap}=\sum_r (T_{e2e,r}-TTFT_r)/\sum_r (N_r-1)$.

Otherwise it is an approximation, and its error is the covariance term (**D**, CLM-015).

*Counterexample (two requests):*

| Request | $N_{out}$ | $TPOT$ | Post-first-output time |
|---|---:|---:|---:|
| A | 2 | 10 ms | 10 ms |
| B | 4 | 2 ms | 6 ms |

- True mean post-first-output time: $(10+6)/2=8\text{ ms}$.
- Product of request means: $\overline{N_{out}-1}=2$ and $\overline{TPOT}=6\text{ ms}$, which gives $12\text{ ms}$. That overstates the mean by 50%. The population covariance is $-4\text{ ms}$, and $12-4=8$. (Deviations from the means: A has $N_{out}-1=1$, i.e. $-1$, and $TPOT$ $+4$; B has $+1$ and $-4$. $\mathrm{Cov}=\frac{(-1)(+4)+(+1)(-4)}{2}=-4\text{ ms}$.)
- Gap-weighted mean: $\overline{ITL}_{gap}=16\text{ ms}/4\text{ gaps}=4\text{ ms}$. Then $2\times4=8\text{ ms}$, which is correct.

Target average concurrency:
$$L = \lambda \times \mathbb{E}[T]$$
Here $\lambda$ and $\mathbb{E}[T]$ must share the boundary of the $L$ you need, under the Lesson 4.4 conditions. Use end-to-end time to get requests in the system. For KV sizing, use admitted rate times mean *resident* time, from KV allocation to release. Queued requests usually hold no KV, so end-to-end $L$ is not the resident-sequence count. $\lambda_{peak}$ inserted into a mean identity sizes a sustained peak only if that peak lasts long enough for the averages to settle. Size bursts and tails with load tests.

*Step 3: KV Cache Concurrency Ceiling*:
Per-request peak KV memory demand:
$$KV_{req} = 2 \times N_{layers} \times N_{kv\_heads} \times d_{head} \times B_{elem} \times (\bar{N}_{in} + \bar{N}_{out})$$
Maximum concurrent sequences supported by memory:
$$L_{KV\_max} = \frac{M_{KV\_available}}{KV_{req}}$$
*Where*: the leading 2 counts one K and one V tensor; $N_{layers}$ = transformer layers; $N_{kv\_heads}$ = KV heads (fewer than query heads under grouped-query attention); $d_{head}$ = head dimension; $B_{elem}$ = bytes per stored element (2 for FP16/BF16). Everything before $(\bar N_{in}+\bar N_{out})$ is the per-token footprint; the last factor is the token count at the end of generation, when the request's KV is largest. Round $L_{KV\_max}$ down, because a partial sequence cannot be resident.

If the rough $L_{KV\_max}$ estimate is below required resident concurrency, KV capacity is a candidate constraint. Mean sequence length, allocator overhead, weights, activations, prefix reuse, batching, and routing can invalidate the estimate; confirm with runtime measurements before selecting an intervention.

**Worked Example:**
- Exercise assumption: $80\text{ GiB}$ usable device memory after any platform reservation.
- Model payload estimate: $8\times 10^9$ parameters at 2 bytes each $=16\times10^9$ bytes $\approx14.90\text{ GiB}$; real resident weight memory must be measured.
- Measured non-KV allocations plus reserved headroom: $5.10\text{ GiB}$.
- Exercise KV budget: $M_{KV\_available}=80-14.90-5.10=60.00\text{ GiB}$.
- KV footprint for 8B model (32 layers, 8 KV heads, $d_{head}=128$, FP16):
  $$\text{Bytes/token} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072\text{ bytes} = 128\text{ KiB/token}$$
- Workload: $\bar{N}_{in} = 2,048$, $\bar{N}_{out} = 512$. Total tokens $= 2,560$.
- Logical KV payload per request: $2,560 \times 128\text{ KiB}=320\text{ MiB}=0.3125\text{ GiB}$, before allocator metadata, block rounding, and prefix sharing.
- Maximum memory concurrency:
  $$L_{KV\_max,logical}=\left\lfloor\frac{60\text{ GiB}}{0.3125\text{ GiB}}\right\rfloor=192\text{ concurrent requests}$$
  Unit check: $131{,}072/1024=128$ KiB; $2{,}560\times128\text{ KiB}=327{,}680\text{ KiB}=320\text{ MiB}$; $320/1024=0.3125$ GiB; $60/0.3125=192$ exactly.
Suppose the workload has an admitted rate of $\lambda = 50\text{ req/s}$ and a mean *resident* time of $\mathbb{E}[T_{resident}] = 5\text{ s}$, measured from KV allocation to release (exercise assumption). Then $L_{resident} = 50 \times 5 = 250$ resident sequences on average. Only the resident count is comparable with the KV ceiling. An end-to-end $W$ of 5 s would include queue time and overstate the resident count. The exact time-average KV demand is $\lambda\,\mathbb{E}[\int KV_r(t)\,dt]$ over each request's residency. The screening product $L_{resident}\times KV_{req}(\bar N_{in}+\bar N_{out})$ differs from it in two opposing ways:
- It overstates demand, because KV grows during decode and does not sit at its final size for the whole residency.
- It can understate demand, because longer requests both hold more KV and stay resident longer (positive covariance, the same issue as in Step 2).

Its direction is therefore not guaranteed. Neither form covers length tails or bursts.
Because the mean-value resident-concurrency estimate exceeds the simplified KV ceiling, this configuration fails the analytical screening check. It does not prove a crash or prescribe one remedy; run a workload-faithful load sweep and compare admission, replication, model placement, representation, and SLO trade-offs.

**Knowledge Check:**
1. A cluster exhibits elevated P99 TTFT, low sampled SM activity, and abundant free KV blocks. Give at least three competing hypotheses and the next discriminating measurement.
2. If doubling batch size doubles ITL while compute counters are high, what additional evidence distinguishes compute saturation from synchronization, queueing, or host-side delay?

**Feedback Contract:**
- *Expected Evidence*: A ranked hypothesis set—not an exactly-one classifier—using queue residency, request states, host/GPU timelines, kernel counters, KV state, and client/network timing.
- *Expected Evidence (Guided Practice)*:
  - $L=20\times8=160$ requests inside the end-to-end boundary. Because resident time is at most $W$, at most 160 are resident on average, and fewer if queueing time is positive.
  - At the stated footprint, 160 residents would need $160\times2.0=320\text{ GiB}$ of KV. This is a screening value, not a bound (see the caveats in the worked example above).
  - Weights are the declared 140 GiB resident per model copy. Do not recompute them as $70\times10^9\times2$ bytes, which is $\approx130.39$ GiB of payload. The declared figure is an independent exercise input.
  - The combined screening figure is $\approx460\text{ GiB}$ for one copy serving that load, before non-KV memory and headroom.
  - A submission that turns this into a GPU count without declaring usable memory per device, sharding, and replica count is incomplete.
- *Common Failure*: Computing $(\bar N_{out}-1)\times\overline{TPOT}$ from request-weighted means when length and TPOT covary, or comparing end-to-end $L$ with a KV ceiling.
- *Diagnostic Hint*: If SM utilization is 35% and memory is free, is the GPU doing work while the request waits? For sizing: which boundary does your $W$ describe, and do queued requests hold KV there?
- *Concept to Revisit*: Serving Bottleneck Discrimination Matrix; Little's Law boundary conditions (Lesson 4.4).

**Guided Practice:**
For a 70B exercise with a stated $140\text{ GiB}$ resident weight footprint (an independent exercise input, not derived from the parameter count), $\lambda=20\text{ req/s}$ admitted, mean end-to-end response time $W=8\text{ s}$, and a stated logical $KV_{req}=2.0\text{ GiB}$, do the following:
1. Compute the mean-concurrency requirement, and say which boundary it describes.
2. Compute the memory screening values.
3. List the missing placement, parallelism, per-replica service curve, resident-time, tail-latency, burst, allocator, failure-headroom, and workload-distribution measurements that prevent these averages from determining an exact H100 count.

**Learning Outcome:**
Discriminate queueing, compute, and memory bottlenecks from production telemetry and calculate hardware provisioning requirements from workload parameters.

*(Effort: 50m instruction, 20m practice)*

---

## 05 Literature & Production Source Map

Section pointers below were re-checked against the linked full texts on 2026-09-30. Each entry's *Scope* says what the module uses it for, and nothing more.

**CANONICAL**
- *Orca: A Distributed Serving System for {Transformer-Based} Generative Models* (Yu et al., OSDI 2022). [Mechanism: Continuous Batching & Iteration-Level Scheduling] — [USENIX page](https://www.usenix.org/conference/osdi22/presentation/yu) · [PDF](https://www.usenix.org/system/files/osdi22-yu.pdf)
  - *Why it matters*: Established iteration-level scheduling and selective batching as reference serving mechanisms.
  - *Key Sections*: Section 3 (Challenges and Proposed Solutions: S1 iteration-level scheduling, S2 selective batching); Section 4.2 (Scheduling Algorithm).
  - *Scope*: The reference mechanism for Lesson 4.2. Its evaluation numbers are specific to its models and hardware and are not used as general speedups.
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., SOSP 2023). [Mechanism: Paged KV Cache & vLLM Architecture] — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
  - *Why it matters*: Demonstrated paged KV management and memory-aware serving; its scheduler implementation is historical, not the current vLLM V1 path.
  - *Key Sections*: Section 4.5 (Scheduling and Preemption: swapping vs. recomputation); Section 7.3 (Comparing Recomputation and Swapping).
  - *Scope*: The historical swap/recompute design used for comparison in Lesson 4.5. Its Section 7.3 overhead ratios are measured on its own setup and do not transfer to other runtimes.
- *Little's Law as Viewed on Its 50th Anniversary* (J. D. C. Little, Operations Research 59(3), 2011). — [DOI](https://pubsonline.informs.org/doi/10.1287/opre.1110.0940)
  - *Key Sections*: Section 2 (finite-window theorems LL.1/LL.2: exactness, independence from stationarity and queue discipline); Section 3 (sample-path and stationary versions).
  - *Scope*: The conditions on $L=\lambda W$ stated in Lessons 4.4 and 4.7. It does not supply any LLM service model.

**PRODUCTION**
- `vllm/v1/core/sched/scheduler.py` and `vllm/v1/core/kv_cache_manager.py`. Pinned commit: `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`. Statically verified 2026-09-25 and statically re-inspected 2026-09-30; not executed. — [scheduler.py](https://github.com/vllm-project/vllm/blob/25b0add7b8a1c944d5c4e364f2de6aa82497a2ad/vllm/v1/core/sched/scheduler.py) · [kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/25b0add7b8a1c944d5c4e364f2de6aa82497a2ad/vllm/v1/core/kv_cache_manager.py)
  - *Inspection Focus*: `Scheduler.schedule()`, `KVCacheManager.allocate_slots()`, and `Scheduler._preempt_request()`.
  - *Observed*: `Scheduler.schedule()` assigns token work to running requests before traversing the waiting queues, bounds that work with scheduled/input token budgets, and calls `KVCacheManager.allocate_slots()` before scheduling work that needs KV capacity (**O**, CLM-006).
  - *Scope*: One upstream snapshot. It says nothing about other releases or runtimes.
- *Addressing Cascading Failures* (Google SRE Book, ch. 22, written by Mike Ulrich). — [sre.google](https://sre.google/sre-book/addressing-cascading-failures/)
  - *Key Sections*: Queue Management; Load Shedding and Graceful Degradation; Retries.
  - *Scope*: Generic overload-control practice for Lesson 4.6. Its thresholds, such as queue length relative to thread-pool size, are not LLM-serving constants.
- Other production runtimes should be traced only at a pinned revision. Their queue, budgeting, and preemption semantics are comparison points, not assumed copies of vLLM; unpinned implementation claims remain `TODO_VERIFY`.

**FRONTIER**
- *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (Agrawal et al., OSDI 2024). — [USENIX page](https://www.usenix.org/conference/osdi24/presentation/agrawal) · [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
  - *Why it matters*: Provides measured evidence for chunked-prefill/decode-piggyback trade-offs on evaluated configurations.
  - *Key Sections*: Section 3 (Motivation: prefill/decode cost, throughput–latency trade-off); Sections 4.1–4.3 (chunked prefills, stall-free batching, token-budget selection).
  - *Scope*: The scheduling pattern in Lesson 4.3. Its token budgets and reported gains are specific to its evaluated models, hardware, and SLOs.
- *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving* (Zhong et al., OSDI 2024). — [USENIX page](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) · [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
  - *Why it matters*: Evaluates phase disaggregation under explicit workloads and exposes placement and KV-transfer trade-offs (**O**, CLM-014).
  - *Key Sections*: Sections 2.1–2.3 and Section 3 (prefill/decode characterization and trade-off analysis); Section 4 (placement and online scheduling).
  - *Scope*: Evidence that phase behavior is workload- and hardware-dependent, and that disaggregation is a candidate intervention in the Mastery problem. Detailed disaggregated placement belongs to Module 20.
- *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (Patel et al., ISCA 2024). — [DOI](https://doi.org/10.1109/ISCA59077.2024.00019) · [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
  - *Reading Priority*: DIRECTED_READ for understanding phase separation economics and network KV transfer overheads.
  - *Key Sections*: Section III (Characterization); Section IV-C (KV-cache transfer).
  - *Scope*: Characterization of its own production traces and hardware. It is a comparison point, not a default architecture.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs require following the rigorous scientific loop:
$$\text{PREDICT} \to \text{BUILD} \to \text{MEASURE} \to \text{EXPLAIN} \to \text{BREAK} \to \text{IMPROVE} \to \text{FALSIFY}$$

### LAB A — Serving Scheduler Simulator: Static vs. Continuous Batching
- **Objective**: Build a discrete-event scheduler simulator in Python that processes heterogeneous prompt and generation workloads, comparing static batching against iteration-level continuous batching.
- **Pre-Registered Hypothesis**: Under the lab's declared execution-cost model, sustained backlog, and heterogeneous output lengths, continuous batching will reduce unfillable fixed-membership slot-time. State a minimum practically important throughput effect before running the experiment; do not treat a supplied multiplier as a fact.
- **Independent Variables**: Batching policy (`STATIC` vs `CONTINUOUS`), batch size $B \in [4, 8, 16, 32]$, generation length distribution variance ($\sigma \in [10, 100, 500]$ tokens).
- **Dependent Variables**: Token throughput (tokens/sec), execution bubble percentage, P50/P99 TTFT, P50/P99 TPOT.
- **Break & Falsify**: Inject a degenerate workload where all requests have strictly identical prompt and generation lengths ($N_{in}=128, N_{out}=128$). Falsify the claim that continuous batching is universally superior by measuring the scheduling loop overhead.
- **Alignment**: Explicitly exercises Lesson 4.2.
- **Effort Estimate**: 3h implementation, 1h analysis (4h total).

### LAB B — Chunked Prefill & ITL Variance Under Mixed Workloads
- **Objective**: Implement chunked prefill mechanics within the scheduler simulator and measure inter-token latency stability during long prompt arrivals.
- **Pre-Registered Hypothesis**: For the selected cost model and mixed workload, an intermediate chunk size will reduce the upper tail of ITL relative to monolithic prefill, while very small chunks can sacrifice prefill efficiency. Declare the expected effect and acceptable TTFT/throughput cost before measurement.
- **Independent Variables**: Prefill chunk size $C \in [128, 256, 512, 1024, \infty]$ (where $\infty$ denotes unchunked), prompt length distribution.
- **Dependent Variables**: P50/P95/P99 ITL, TTFT, throughput, scheduled step count, and simulator cost components. If an optional GPU experiment is run, add achieved FLOPs/bandwidth and kernel timelines rather than inventing hardware efficiency in the simulator.
- **Break & Falsify**: Lower chunk size to $C=32$ and test whether extra steps or implementation overhead reduce throughput. A result with no degradation falsifies the proposed overhead mechanism for that setup.
- **Alignment**: Explicitly exercises Lesson 4.3.
- **Effort Estimate**: 2.5h implementation, 1h analysis (3.5h total).

### LAB C — Queueing Dynamics, Little's Law, & Saturation Breakdown
- **Objective**: Subject the serving simulator to varying arrival rates modeled as Poisson processes, experimentally locating the saturation knee and validating Little's Law.
- **Pre-Registered Hypothesis**: For a stable run measured after warm-up with consistent arrival and residence boundaries, $L\approx\lambda W$ within a tolerance justified by finite-sample uncertainty. Near overload, the steady-state estimator becomes unusable if the run never equilibrates; finite queues, drops, or cancellations require reporting admitted/completed populations rather than claiming Little's Law itself failed.
- **Independent Variables**: Arrival rate $\lambda \in [0.2 \mu, 0.4 \mu, 0.6 \mu, 0.8 \mu, 0.95 \mu, 1.2 \mu]$, prompt length variance ($C_v \in [0.2, 1.0, 2.5]$).
- **Dependent Variables**: Mean queue depth $L_q$, queue wait time $W_q$, system occupancy $L$, goodput (requests meeting a 500 ms TTFT SLO).
- **Statistical Rigor**: Use independently seeded trials, predeclare warm-up and stopping rules, report the estimator and confidence-interval method, and increase trial count until uncertainty is adequate for the chosen effect size. Ten trials may be a starting point, not a universal guarantee.
- **Alignment**: Explicitly exercises Lesson 4.4 and Lesson 4.7.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — Preemption Under Memory Pressure & Admission Control
- **Objective**: Implement physical KV block accounting in the scheduler simulator. Simulate high-concurrency traffic that exceeds GPU HBM, triggering preemption (swapping vs recomputation), and integrate an admission control gateway.
- **Pre-Registered Hypothesis**: Under a declared timeout/cancellation/retry model, bounded admission will protect selected SLO-compliant goodput during overload at the cost of rejected utility. Sweep offered load and queue limits; a result where unbounded queueing preserves equal or better declared utility falsifies the proposed policy for that workload.
- **Action**: Compare at least two victim/admission policies, parameterize measured or explicitly synthetic transfer and recomputation costs, and report completions, SLO-goodput, rejections, cancellations, wasted work, and fairness. Keep the simulator's response contract generic unless a specific gateway protocol is part of the experiment.
- **Alignment**: Explicitly exercises Lesson 4.5 and Lesson 4.6.
- **Effort Estimate**: 2.5h implementation, 1h analysis (3.5h total).

---

## 07 Break / Incident Scenarios

### Incident 04.1: The Midnight Latency Collapse
- **Incident Symptoms**:
  At 00:14 UTC, production alerts fire for a multi-tenant enterprise LLM cluster serving customer support chatbots and backend document indexing jobs.
  - Overall P99 TTFT spikes from a baseline of $210\text{ ms}$ to $14,800\text{ ms}$.
  - Customer chat clients report widespread HTTP 504 Gateway Timeouts.
  - GPU compute metrics show that all 16 node GPUs remain pinned at $100\%$ SM utilization.
  - Total output token generation rate drops by $82\%$, despite the cluster being at 100% compute load.
  - Ingress traffic volume (requests per second) increased by only $12\%$ over normal midnight baseline.
- **Diagnostic Protocol (Task)**:
  The learner must act as the primary on-call systems engineer and execute the following investigation:
  1. *Formulate Competing Hypotheses*: Formulate at least 3 distinct, technically rigorous hypotheses that could explain the symptoms (e.g., Hypothesis A: Ingress arrival burst causing pure queueing saturation; Hypothesis B: Document indexing job submitted long-context prompts triggering prefill/decode interference; Hypothesis C: Dynamic sequence length growth caused KV cache exhaustion, triggering a preemption recomputation thrashing cascade).
  2. *Rank Hypotheses by Initial Plausibility*: Evaluate the reported symptoms against each hypothesis. Do not assume a proportional response: a modest offered-load increase can cross timeout, retry, cancellation, or batching thresholds and cause a much larger goodput change even without KV thrashing.
  3. *Identify Missing Evidence*: Specify exactly what telemetry metrics and logs must be inspected to discriminate the mechanism (e.g., scheduler preemption counters, KV allocation failures, transfer/offload bytes when that feature is enabled, prompt-length histograms, and queue-residency timers).
  4. *Design Discriminating Test*: Formulate a targeted query or controlled intervention whose predicted observations distinguish the leading hypotheses; state what result would falsify each explanation within the test's scope.
  5. *Execute Causal Diagnosis*: Rank the supported mechanism or interacting mechanisms and state which alternatives the evidence rules out.
  6. *Prescribe Immediate Mitigation & Long-Term Intervention*: Define the immediate remediation (e.g., shedding the background indexing queue via admission control) and the architectural prevention (e.g., separate tenant pools or chunked prefill with strict max batched tokens).
  7. *Remeasurement & Post-Mortem Defense*: State the quantitative verification criteria proving the incident is resolved.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Architecture Transfer Problem: Multi-Tenant Voice & Document Platform
You are the Principal Inference Architect for an enterprise AI platform that serves two drastically different workloads on a shared cluster of 8 $\times$ NVIDIA H100 SXM GPUs (vendor-listed as 80GB HBM3 on the [NVIDIA H100 page](https://www.nvidia.com/en-us/data-center/h100/); the usable capacity is the fixture value below, not the marketing figure) running a 70B parameter model:
1. **Interactive Real-Time Voice Agent**: Requires strict streaming latency SLOs: P95 TTFT $< 300\text{ ms}$, P95 ITL $< 30\text{ ms}$. Average prompt: 400 tokens; average generation: 150 tokens. Traffic: 40 requests/sec with high burstiness.
2. **Asynchronous Document Summarization**: Long-context batch jobs. Average prompt: 16,000 tokens; average generation: 800 tokens. Traffic: 2 requests/sec steady background volume.

Under the current baseline deployment, one engine with tensor parallelism across all 8 GPUs, the voice agent stutters badly whenever a document summarization request arrives (P99 ITL spikes to $180\text{ ms}$), and conversational users frequently disconnect.

You must design a comprehensive serving architecture, scheduling configuration, and capacity allocation plan that targets the voice agent's strict SLOs while maximizing document summarization throughput, and define the load and failure tests required before making a production guarantee.

**Workload Fixture (SYNTHETIC — exercise assumptions, not measurements of any real model, GPU, or runtime; registry CLM-012):**

| Input | Fixture value |
|---|---|
| Parameters / weight precision | $P=70\times10^9$; 2 bytes/param. Logical payload is $140\times10^9$ bytes ($\approx130.39$ GiB). Assume even sharding under TP degree $t$, i.e. $130.39/t$ GiB of weights per GPU, ignoring replicated tensors. |
| KV architecture | 80 layers, 8 KV heads, $d_{head}=128$, KV stored at 2 bytes/element on the same devices as the weights. KV heads shard evenly for $t\in\{1,2,4,8\}$. |
| Usable device memory | 76.0 GiB per GPU after platform/runtime reservation |
| Non-KV allocations + headroom | 6.0 GiB per GPU, the same for every $t$ (a simplification: state how a different measured value changes your answer) |
| Measured mean times at baseline load | Voice: end-to-end 3.2 s, resident (KV allocation to release) 3.0 s. Documents: end-to-end 30 s, resident 28 s. |
| Voice burst | 80 req/s for 15 s windows, a few times per hour |
| Baseline step costs (TP=8 engine, synthetic) | decode-only step with ≤160 decodes: 20 ms; decode + 512-token prefill chunk: 27 ms; + 1,024: 34 ms; + 2,048: 48 ms; decode + unchunked 16,000-token prefill: 175 ms |
| Unknown by design | Effective KV-transfer bandwidth between pools, step costs at other TP degrees, and the tail distributions. Measure them, or carry them as symbols. |

You may replace any fixture value with your own measurement. If you do, record the model, hardware, runtime revision, configuration, and method. Every numerical answer is graded *conditionally on the inputs you declare*, and there is no single correct GPU count or configuration. Where an input is unknown, answer as a function of it. For example, the voice KV-transfer time is $171.875\text{ MiB}/B_{eff}$, and you must state the $B_{eff}$ at which an intervention stops meeting the ITL/TTFT budget.

**Required Deliverables**:
1. **Quantitative Workload & Resource Model**:
   - Calculate parameter memory footprint (bytes, GB, and GiB), available KV cache memory per GPU for each placement you consider, and per-request KV demand for both workloads.
   - Apply Little's Law to calculate steady-state concurrency for both workloads, both end-to-end and resident. State the boundary and conditions for each, and give burst-window numbers separately from long-run means.
2. **Bottleneck Prediction & Theoretical Analysis**:
   - Formulate ranked competing hypotheses for the voice agent ITL spikes. Give the discriminating measurement for each, and say which fixture data support or weaken it.
   - Explain why standard continuous batching does not by itself isolate these workloads.
3. **Three Candidate Architectural Interventions**:
   - Formulate three technically distinct candidate interventions (e.g., (A) Aggressive Chunked Prefill with token budgeting; (B) Prefill-Decode Disaggregation with dedicated voice and summary pools; (C) Strict Priority Preemption with CPU KV Swapping).
4. **Quantitative Derivation of Expected Effects**:
   - For each intervention, derive the expected effect on voice agent TTFT, voice agent ITL, and document summarization throughput. State all explicit assumptions.
5. **Trade-Off & Failure Mode Analysis**:
   - Detail the operational risks, hardware utilization costs, and failure modes of each intervention.
6. **Rejection of Inappropriate Interventions**:
   - Explicitly analyze and reject at least one seemingly plausible intervention using the fixture or your declared measurements. For example, test whether simply increasing the global continuous batch size, or relying on CPU swapping, would worsen voice latency. Reject it only if your inputs support that conclusion.
7. **Final Architecture Defense**:
   - Defend your recommended architecture with concrete scheduler settings, such as `max_num_batched_tokens`, chunk sizes, queue limits, and admission control rules. Tie each to the runtime and revision it applies to, and derive it from your declared inputs. These are initial settings to validate by load test, not guaranteed values.
8. **Uncertainty, Falsification, & Rollback Strategy**:
   - Identify what workload change would falsify your design assumptions (e.g., voice prompts growing to 4,000 tokens).
   - Define exact operational metrics and automated rollback triggers that would reverse the architectural changes in production.

---

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
The learner must submit a verified code trace document analyzing the scheduling loop of a production serving runtime. The reference trace uses vLLM commit `25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`.
The trace must map out:
1. Entry point into `Scheduler.schedule()`.
2. How running work is budgeted and where `KVCacheManager.allocate_slots()` is called.
3. The path from allocation failure through victim selection to `Scheduler._preempt_request()`.
4. How chunked prefill slices waiting requests into the scheduled token budget.
5. One concrete policy trade-off where the production scheduler's heuristic favors one objective over another; do not assert a universally optimal queue discipline without an objective and queue model.

### Reference Checks for Mastery Deliverables 1–2 (fixture inputs only)
Reviewers use these values to check arithmetic, not to grade a design. A submission with different declared inputs is checked against its own inputs.

- **Weights**: $70\times10^9\times2=140\times10^9$ bytes $=140$ GB $\approx130.39$ GiB. Per GPU at $t=2/4/8$: $\approx65.19/32.60/16.30$ GiB.
- **KV pool**, from $76.0-130.39/t-6.0$ (usable memory minus this GPU's weight shard minus non-KV/headroom; per replica multiply by $t$, and $8/t$ replicas fit on 8 GPUs):
  - per GPU at $t=2/4/8$: $\approx4.81/37.40/53.70$ GiB;
  - per replica: $\approx9.61/149.61/429.61$ GiB, with $4/2/1$ replicas.
  - The 16,800-token document footprint alone ($\approx5.13$ GiB) exceeds a single $t=2$ GPU's KV pool. It fits only when sharded across the replica's pool of $\approx9.61$ GiB, so a $t=2$ replica holds at most one resident document at a time.
- **KV bytes/token**: $2\times80\times8\times128\times2=327{,}680$ B $=320$ KiB.
  - Voice at mean lengths: $550\times320\text{ KiB}=171.875$ MiB $\approx0.168$ GiB.
  - Documents: $16{,}800\times320\text{ KiB}=5{,}250$ MiB $\approx5.13$ GiB.
- **Little's Law, long-run means**:
  - Voice: $40\times3.2=128$ end-to-end and $40\times3.0=120$ resident.
  - Documents: $2\times30=60$ end-to-end and $2\times28=56$ resident.
  - The resident screening demand at mean lengths is $\approx20.14+287.11=307.25$ GiB ($120\times171.875\text{ MiB}=20{,}625\text{ MiB}$ and $56\times5{,}250\text{ MiB}=294{,}000\text{ MiB}$, each divided by 1,024). That is below the single $t=8$ pool but above the $\approx299.23$ GiB total of two $t=4$ replicas. The caveats from Lesson 4.7 apply, and burst-window and tail demand must be analyzed separately.
- **Step costs**: The fixture's 175 ms unchunked-prefill step is *consistent with* the 180 ms P99 ITL spike. It supports the prefill-interference hypothesis but does not prove it. At a 512-token chunk the mixed step is 27 ms, and at 1,024 tokens it is 34 ms, which already exceeds a 30 ms ITL target before any other overhead. With one 512-token chunk per step, a 16,000-token prompt needs $\lceil16{,}000/512\rceil=32$ steps, so at least $32\times27=864$ ms of document prefill span. That is the TTFT/throughput cost to weigh.

### Rubric Dimensions
- **Conditional Quantitative Answers** (Mastery Deliverables 1, 4, 7):
  - *Insufficient*: Reports one GPU count or setting as "the answer" from incomplete inputs, treats 70B × 2 bytes as 140 GiB, or compares end-to-end concurrency with a KV ceiling.
  - *Competent*: Declares every input with its source (fixture, measurement, or assumption), keeps units and boundaries straight, and states the conditions under which each conclusion holds.
  - *Strong*: Also gives the sensitivity to the unknown inputs, such as the transfer bandwidth or non-KV memory at which the recommendation flips, and the measurement that would resolve each one.
- **Mechanistic Reasoning**: *Insufficient* describes high-level concepts without execution mechanisms. *Competent* explains step-by-step scheduler decisions and hardware contention. *Strong* links scheduler loop logic directly to GPU hardware execution regimes (Tensor Core saturation vs HBM bandwidth).
- **Quantitative Reasoning**: *Insufficient* quotes formulas without assumptions. *Competent* applies Little's Law, qualified queueing models, and memory estimates with units and boundaries. *Strong* compares predictions with load measurements and explains where state-dependent service invalidates the analytical approximation.
- **Experiment Design & Rigor**: *Insufficient* runs uninstrumented tests. *Competent* isolates variables and validates hypotheses with controlled workloads. *Strong* constructs rigorous falsification experiments and accounts for statistical variance across Monte Carlo trials.
- **Failure Diagnosis**: *Insufficient* guesses root causes from high-level symptoms. *Competent* distinguishes queueing, compute, and memory bottlenecks using metric signatures. *Strong* executes a systematic elimination protocol during multi-factor incidents without leaking hypotheses.
- **Architecture Defense**: *Insufficient* recommends a popular library without justification. *Competent* balances throughput vs latency trade-offs using workload models. *Strong* provides a mathematically defended architecture plan with explicit rejection of inappropriate alternatives and rollback triggers.

---

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| **Latency Metrics Decomposition** | Lesson 4.1 | Lesson 4.1 Guided Practice | LAB A dependent variables (TTFT/TPOT) / Mastery Deliverable 1 | Latency Profiler Trace / Workload Model with declared timestamp boundaries |
| **Continuous Batching Dynamics** | Lesson 4.2 | Lesson 4.2 Independent Practice (LAB A) | LAB A Break & Falsify | Scheduler Simulator Code & Benchmark Report |
| **Chunked Prefill & ITL Stabilization** | Lesson 4.3 | Lesson 4.3 Independent Practice (LAB B) | LAB B / Mastery Deliverables 3–4 (checked against fixture step costs) | ITL Variance Measurement & Comparison Artifact |
| **Queueing Theory & Little's Law** | Lesson 4.4 | Lesson 4.4 Guided Practice; Lesson 4.7 Step 2 counterexample | LAB C / Mastery Deliverable 1 (end-to-end vs resident concurrency, with boundary conditions) | M/G/1 Model Derivation & Empirical Plot; Reference Checks in Section 09 |
| **Memory & Concurrency Sizing Units** | Lesson 4.7 Steps 1–3 | Lesson 4.7 Guided Practice | Mastery Deliverable 1 / Conditional Quantitative Answers rubric | Workload model with bytes/GB/GiB and declared inputs |
| **Preemption & Memory Arbitration** | Lesson 4.5 | Lesson 4.5 Worked Example and Knowledge Check | LAB D / Incident 04.1 steps 3–5 | Diagnostic Report with measured or labeled synthetic transfer/recompute costs |
| **Production Scheduler Source Trace** | Lesson 4.5 | Lesson 4.5 Guided Practice (pinned trace, including the waiting-path allocation failure) | Section 09 Required Artifact items 1–5 | Production Source Trace pinned to commit `25b0add7…` |
| **Admission Control & Load Shedding** | Lesson 4.6 | Lesson 4.6 Guided Practice | LAB D / Mastery Deliverable 7 | Gateway Simulator & Goodput Curve |
| **Bottleneck Discrimination** | Lesson 4.7 | Lesson 4.7 Guided Practice | Incident 04.1 steps 1–7 / Mastery Deliverable 2 | Root Cause Analysis Protocol & Defense |

---

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 04 must be able to:
1. Derive logical memory estimates, state their exclusions, and measure the saturation knee for a specified workload distribution.
2. Build and instrument a discrete-event continuous batching scheduler simulator.
3. Evaluate chunked prefill parameters across TTFT, ITL, throughput, fairness, and goodput, including a workload where chunking is harmful.
4. Trace execution flow through the scheduler loop of a production serving engine.
5. Diagnose complex serving incidents and systematically isolate queueing, compute, and memory exhaustion failures.
6. Design, defend, and specify an enterprise serving architecture under conflicting multi-tenant latency SLOs.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: An LLM serving engine is a state-dependent queueing and resource-arbitration system. Prefill and decode frequently present different execution shapes, but their actual bottlenecks must be measured for the selected model, workload, runtime, and hardware.
- **The Path to High Goodput**:
  $$\text{Arrivals} \to \text{Overload Controls} \to \text{Queue/Admission} \to \text{Scheduling and Batching} \to \text{GPU/KV Constraints} \to \text{Completion, Rejection, or Cancellation}.$$
- When degradation occurs, align client, queue, scheduler, host, GPU, KV-allocation, and network timelines; rank competing explanations; intervene; then repeat the same measurements. No single ordering or signal identifies every incident.

### Cost & Telemetry Hook

- [Lesson 22.1](../22-ai-economics/README.md) and [Lesson 22.2](../22-ai-economics/README.md): cost capacity plans per goodput-satisfying request, counting rejected, cancelled, and preempted work in the denominator and fixed reserved capacity.
- [Lesson 23.3](../23-observability-reliability/README.md): define TTFT/ITL SLIs over offered requests, including shed and timed-out ones, not only completions.
- [Lesson 23.4](../23-observability-reliability/README.md): turn the module's multi-tenant SLOs into error budgets with multiwindow burn alerts on queue delay and goodput.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
