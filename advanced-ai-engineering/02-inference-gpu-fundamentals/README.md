# Module 02 — Inference GPU Fundamentals

## 00 Why This Module Exists

An LLM forward pass is a graph of asynchronous host submissions, device kernels, memory transfers, allocations, and synchronization. A model-level FLOP count cannot tell you whether the live execution is limited by launch latency, exposed dependencies, insufficient parallelism, a compute pipeline, a memory boundary, resource residency, throttling, or an interaction among them.

This module builds the hardware/software measurement model needed before KV-cache engineering, serving, or optimization. It teaches the learner to define the measured boundary, calculate a scoped upper bound, collect timeline and kernel evidence, break one assumption at a time, and reject a clean bottleneck story when the telemetry does not discriminate it.

**Module Orientation**

- **Engineering Problem**: Connect tensor shapes and operations to GPU execution, memory traffic, time, profiler observations, and falsifiable performance explanations.
- **What You Will Do**: Implement reference kernels, induce coalescing and divergence failures, prove why unsynchronized timing is invalid, construct qualified Roofline bounds, profile at system and kernel scope, trace a pinned benchmark implementation, and defend a diagnosis against alternatives.
- **Environment**: Python 3.10+ plus CUDA C++ or a GPU kernel DSL; an NVIDIA GPU and current profiling tools are required for device-counter labs, while analytical exercises can run without them.
- **Research Cutoff**: evidence registry 2026-09-25; WP-F1 review 2026-09-30 corrected units, rubric, and examples and re-checked pinned symbol presence without re-reading the NVIDIA documentation. Current implementation claims remain pinned to the verified revisions below.

## 01 Baseline Assumptions

Prerequisites: Module 00 measurement discipline; Module 01 tensor shapes, matrix multiplication, parameter/FLOP conventions, and numerical invariants; basic Python and either CUDA C++ or a GPU kernel DSL.

This module owns:

- CUDA-style execution hierarchy, SIMT, warps, divergence, and synchronization;
- registers, shared memory, caches, device memory, memory transactions, and coalescing;
- asynchronous execution, streams, events, timing boundaries, warmup, and replication;
- arithmetic intensity, effective bandwidth, basic and hierarchical Roofline reasoning;
- occupancy/resource residency and why occupancy is not a performance objective;
- application-timeline versus kernel-counter profiling;
- profiler overhead, replay, cache control, and measurement intrusion;
- shape-, precision-, backend-, and hardware-dependent bottleneck transitions.

It cross-references but does not replace:

- KV-cache layout, paging, fragmentation, and eviction — Module 03;
- TTFT, ITL/TPOT, queueing, batching, goodput, and serving capacity — Module 04;
- FlashAttention, fusion, quantization, speculative decoding, and production kernel tuning — Module 05;
- collective communication and multi-GPU topology — Module 20;
- fleet observability, SLOs, and incident operations — Module 23.

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
  security: NOT_APPLICABLE
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 6h
  guided_practice: 3h
  labs: 16h
  assessment: 3h
  source_trace: 2h
  total: 30h
```

*Effort reconciliation*: lesson instruction sums to 360 min and lesson practice to 180 min. Labs are A 4h + B 3h + C 4h + D 4h + E's 1h comparison = 16h; LAB E's 2h source trace (which includes Lesson 2.7's trace practice) is counted once, under `source_trace`. The WP-F1 revision raised labs from 15h to match the lab estimates already stated (total 29h → 30h).

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
host code / framework
  ↓ asynchronous submission, allocation, dispatch, synchronization
CUDA streams and dependencies
  ↓ grids of thread blocks
SM residency and warp issue
  ↓ instructions, tensor/math pipelines, load/store pipelines
registers ↔ shared memory / L1 ↔ L2 ↔ device memory
  ↓
completed results observed at an explicit boundary
```

The diagnostic feedback loop is:

```text
shape / layout / precision / backend changes
  → kernel choice and launch geometry change
  → reuse, traffic, registers, shared memory, and parallelism change
  → stalls and attainable throughput change
  → elapsed time and downstream serving behavior change
```

That loop is a hypothesis template, not a universal mechanism. It applies only when the selected path actually changes those quantities. It is falsified when the kernel path and device evidence remain stable while latency changes elsewhere, for example in a CPU gap, synchronization, queue, or allocation.

Keep three knowledge types separate:

- **O — Source observation**: a profiler counter, timeline event, documented semantic, or pinned source behavior.
- **D — Derivation**: a result following from named units, formulas, and assumptions.
- **H — Engineering hypothesis**: a causal explanation that predicts what a discriminating intervention will change.

## 04 Lessons

### Lesson 2.1 — Execution Hierarchy, SIMT, and Resource Residency

**Engineering Question:** What can the program control, and what is only observed after the device schedules work?

**Concepts & Definitions:**

A kernel launch defines a grid of thread blocks. Blocks are scheduled onto streaming multiprocessors (SMs); correctness cannot depend on the order in which blocks run. Threads within an SM execute in groups of 32 called warps on current CUDA devices (**O**, CLM-001). Threads have individual state, but a warp is most efficient when active lanes follow the same instruction path.

Do not collapse these concepts:

- **grid size**: total blocks and exposed work;
- **block size**: threads that cooperate and share block-scoped resources;
- **warp execution**: active lanes, divergence, reconvergence, and instruction issue;
- **resident blocks/warps**: work that fits concurrently on an SM;
- **occupancy**: active warps divided by the architectural maximum;
- **utilization/throughput**: activity or work rate during a named interval.

Residency is jointly constrained by architectural block/thread limits and per-block registers and shared memory. More occupancy can help hide latency, but maximum occupancy is not a theorem of maximum performance (**O**, CLM-007). Reducing registers to increase occupancy can spill data; shrinking tiles can lower reuse; changing blocks can reduce instruction-level parallelism.

Step by step, ignoring shared memory and allocation granularity:

$$B_{SM}=\min\left(\left\lfloor\frac{R_{SM}}{T_b\,r}\right\rfloor,\ \left\lfloor\frac{T_{SM}}{T_b}\right\rfloor,\ B_{max}\right),\qquad \text{occupancy}_{theor}=\frac{B_{SM}\,T_b/32}{W_{max}},$$

where:

- $R_{SM}$: 32-bit registers per SM; $r$: registers per thread; $T_b$: threads per block;
- $T_{SM}$: maximum resident threads per SM; $B_{max}$: maximum resident blocks per SM;
- $B_{SM}$: resident blocks per SM (the tightest of the three limits); $T_b/32$: warps per block;
- $W_{max}$: maximum resident warps per SM ($=T_{SM}/32$).

Each term is one resource budget divided by one block's demand; the smallest quotient wins. *Achieved* occupancy is measured by the profiler and is usually lower.

**Worked Example (exercise limits, not a specific GPU):**
- *Input*: per-SM limits of 65,536 32-bit registers, 2,048 resident threads (64 warps), and 32 resident blocks. Kernel X uses 256 threads/block and 32 registers/thread; kernel Y uses 256 threads/block and 64 registers/thread. Register-allocation granularity and shared memory are ignored.
- *Steps*: X needs $256\times32=8{,}192$ registers/block, so registers allow $\lfloor65536/8192\rfloor=8$ blocks and threads allow $2048/256=8$, and the block limit allows 32, giving $\min(8,8,32)=8$ blocks $=8\times256/32=64$ warps. Y needs $256\times64=16{,}384$ registers/block, so $\min(\lfloor65536/16384\rfloor,8,32)=\min(4,8,32)=4$ blocks $=32$ warps.
- *Result*: theoretical occupancy is $64/64=100\%$ for X and $32/64=50\%$ for Y.
- *Interpretation / limits*: Y may still finish first if its extra registers hold reused operands and expose more instruction-level parallelism, while X spills or reloads. Query real limits from the device and the occupancy API or profiler before reasoning about a real kernel.

**Knowledge Check:** Which launch quantities are chosen by the program, and which residency/utilization quantities must be observed?

**Guided Practice:** Sweep block size and artificial register/shared-memory use. Record theoretical occupancy, achieved active warps, spills, stall mix, and kernel time. Seek a case where higher occupancy is slower.

**Feedback Contract:**
- *Expected Evidence*: launch geometry; per-kernel registers, shared memory, and spills; theoretical versus achieved occupancy; stall mix; kernel time; and one measured case where higher occupancy is not faster (or a stated falsifier if none is found).
- *Common Failure*: reporting theoretical occupancy as achieved, or tuning for occupancy without measuring time.
- *Diagnostic Hint*: when you cap registers to raise occupancy, do local-memory (spill) loads appear?
- *Concept to Revisit*: resource residency versus latency hiding.

**Learning Outcome:** Connect execution hierarchy and resource residency to measured performance without treating occupancy as a universal target.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 2.2 — Memory Hierarchy, Transactions, and Coalescing

**Engineering Question:** How many bytes did the program need, and how many bytes did the hardware move at each boundary?

**Concepts & Definitions:**

A useful simplified path is registers and shared memory/L1, then L2, then device memory. The exact topology, capacities, cache behavior, and instructions vary by architecture. Never substitute an unqualified “GPU memory” byte count for all levels.

When lanes in a warp access adjacent aligned values, hardware can serve requests with fewer memory transactions. Strided, scattered, or misaligned patterns can transfer sectors containing unused data (**O**, CLM-002). Coalescing therefore concerns the ratio of requested bytes to transferred bytes, not merely whether addresses are contiguous somewhere in source code.

For useful bytes read $B_r$, useful bytes written $B_w$, and synchronized elapsed time $t$ seconds:

$$BW_{useful}=\frac{B_r+B_w}{t}\quad\text{bytes/s}.$$

Here $B_r$ and $B_w$ are counted from the algorithm (elements × bytes/element), not from a profiler. Use $10^9$ for decimal GB/s and $2^{30}$ for GiB/s; do not mix the label and divisor (**D**, CLM-010). This value is not DRAM traffic. Compare it with profiler traffic at a named boundary to test transaction waste or reuse hypotheses.

**Necessary but insufficient signals**

- low useful bandwidth may mean poor coalescing, insufficient parallelism, dependency stalls, or simply little memory work;
- high DRAM throughput supports a DRAM-pressure hypothesis but does not prove every load is efficient;
- high L2 hit rate does not reveal whether L1/shared/register use is optimal;
- many transactions do not prove they are on the critical path.

**Worked Example:**
- *Input*: a kernel requests 4 MiB of useful bytes (read + write) and a synchronized interval of 1 ms.
- *Steps*: (1) to bytes: $4\text{ MiB}=4\times2^{20}=4{,}194{,}304$ bytes; (2) to seconds: $1\text{ ms}=10^{-3}$ s; (3) divide: $4{,}194{,}304/10^{-3}=4{,}194{,}304{,}000$ bytes/s; (4) convert once: $\div2^{30}=3.90625$ GiB/s or $\div10^{9}=4.194304$ GB/s.
- *Result*: exactly $4000$ MiB/s $=4000/1024=3.90625$ GiB/s $=4.194304$ GB/s. Rounded, “≈3.9 GiB/s” or “≈4.2 GB/s”; it is **not** 4 GiB/s, which would need 4 GiB/s × 1 ms $=4.096$ MiB.
- *Interpretation / limits*: this is useful bandwidth. If the profiler reports 8 MiB of DRAM reads for the same launch, the transferred-to-useful ratio is $8\text{ MiB}/4\text{ MiB}=2$ (half of the DRAM bytes moved were not requested), which supports (but does not prove) a transaction-waste hypothesis; reuse in L2 could equally make DRAM bytes smaller than useful bytes.

**Knowledge Check:** Why are useful bytes, L2 traffic, and DRAM traffic different, and what does coalescing change?

**Guided Practice:** Sweep aligned contiguous, offset, and strided accesses while preserving arithmetic; compare useful bandwidth with hierarchy-specific traffic.

**Feedback Contract:**
- *Expected Evidence*: a byte equation per access pattern with the unit convention stated; synchronized time; useful bandwidth; profiler-reported sectors/bytes at L1/L2/DRAM for the same launch; the transferred-to-useful ratio; and at least two competing explanations.
- *Common Failure*: dividing MiB by milliseconds and labeling the result GiB/s (a 2.4% error here: 4 versus 3.90625), or comparing useful bytes with a DRAM counter from a different launch.
- *Diagnostic Hint*: convert to bytes and seconds first, then divide by $2^{30}$ or $10^9$ exactly once.
- *Concept to Revisit*: useful versus transferred bytes at a named boundary.

**Learning Outcome:** Distinguish requested work from measured traffic and diagnose layout-dependent transaction waste.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 2.3 — Asynchrony and Honest Timing

**Engineering Question:** What exactly starts and stops the clock?

**Concepts & Definitions:**

Kernel launches normally return before device completion. A host timer around a launch can therefore measure submission time, not execution time (**O**, CLM-003). Three common boundaries are different measurements:

1. **device interval**: events recorded in the relevant stream;
2. **synchronized host interval**: wall time with completion enforced at the end points;
3. **application interval**: end-to-end work including host logic, allocation, dispatch, transfers, and required synchronization.

Record stream, start/end events, synchronization call, warmup, repetitions, aggregation, and whether transfers or setup are included. An event on one stream does not automatically bound unrelated work in another stream.

Warmup can include context creation, library initialization, memory-pool growth, code generation, autotuning, cache state, and clock ramp. Do not erase cold-start behavior when cold start is the question; report cold and steady-state separately.

**Anti-patterns**

- `time.time()` around an asynchronous launch with no completion boundary;
- synchronizing after every operator and then claiming the result represents production overlap;
- selecting the minimum of many runs without documenting the intended estimator;
- timing under a heavy profiler and treating the host duration as uninstrumented latency.

**Worked Example (synthetic timestamps on one host clock, device events on the same stream):**
- *Input*: host timer starts at 0 µs; the launch call returns at 20 µs; the start event records at 25 µs and the end event at 195 µs; `synchronize()` returns at 200 µs.
- *Steps*: submission interval $=20-0=20$ µs; device interval $=195-25=170$ µs; synchronized host interval $=200-0=200$ µs. The difference decomposes as $200-170=30=(25-0)+(200-195)$: 25 µs from timer start to the start event (20 µs submission + 5 µs launch-to-start) and 5 µs from the end event until `synchronize()` returns.
- *Result*: three different, non-conflicting measurements. The 30 µs between device and synchronized host time is submission plus launch-to-start delay (25 µs) plus synchronization return (5 µs).
- *Interpretation / limits*: reporting 20 µs as “kernel time” understates execution by $170/20=8.5\times$. Adding a second stream whose work is not bracketed by these events would leave the device interval unchanged while the application interval grows.

**Knowledge Check:** Why can an event on one stream fail to bound work on another, and when is per-operator synchronization unrepresentative?

**Guided Practice:** Time identical work using unsynchronized wall time, synchronized wall time, device events, and an application boundary; then add a second stream.

**Feedback Contract:**
- *Expected Evidence*: for each method, the clock, stream, synchronization point, warmup, repetitions, inclusion rule, raw samples, and profiler status; an explanation of why the four methods disagree; the second-stream case showing which boundary excludes queued work.
- *Common Failure*: `time.time()` around a launch without synchronization, or synchronizing after every operator and presenting the result as production overlap.
- *Diagnostic Hint*: does the reported time change when you add a synchronization before stopping the timer?
- *Concept to Revisit*: submission versus completion boundaries.

**Learning Outcome:** Measure asynchronous execution at an explicit boundary without destroying the concurrency being studied.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 2.4 — Roofline as a Qualified Bound

**Engineering Question:** Under which work, byte, hierarchy, precision, and ceiling assumptions does a Roofline bound discriminate a regime?

**Concepts & Definitions:**
Define:

- $W$: executed or modeled work in operations;
- $Q$: bytes transferred at one named memory-hierarchy boundary;
- $I=W/Q$: operational/arithmetic intensity in operations per byte;
- $P_{peak}$: attainable compute ceiling for the selected precision/instruction path;
- $\beta$: attainable bandwidth at the same selected memory boundary.

The basic Roofline bound is (**O**, CLM-004; **D**, CLM-005)

$$P\le\min(P_{peak},\beta I),$$

with ridge intensity

$$I^*=\frac{P_{peak}}{\beta}.$$

Equivalent time lower bounds are

$$T_{compute}\ge\frac{W}{P_{peak}},\qquad T_{memory}\ge\frac{Q}{\beta},$$

and under the basic overlap abstraction,

$$T\ge\max\left(\frac{W}{P_{peak}},\frac{Q}{\beta}\right).$$

where $P=W/T$ is achieved throughput (operations/s) and $T$ is elapsed time. Unit check: $\beta I$ is (bytes/s)·(operations/byte) = operations/s, the same unit as $P_{peak}$.

*Step-by-step derivation:*

1. Work cannot execute faster than the compute ceiling: $T\ge W/P_{peak}$.
2. $Q$ bytes cannot cross the boundary faster than its bandwidth: $T\ge Q/\beta$.
3. Both hold at once, so $T\ge\max(W/P_{peak},\,Q/\beta)$. Perfect overlap of compute and memory is the optimistic assumption; with no overlap the two terms would add.
4. Divide $W$ by this bound: $P=\dfrac{W}{T}\le\min\left(P_{peak},\ \dfrac{W}{Q/\beta}\right)=\min(P_{peak},\,\beta I)$, using $W/Q=I$.
5. The roofs meet where $\beta I=P_{peak}$, i.e. $I^*=P_{peak}/\beta$. If $I<I^*$ then $Q/\beta>W/P_{peak}$ and the memory term dominates; if $I>I^*$ the compute term dominates.

This is an analytical bound, not a latency guarantee. State all assumptions:

- FMA counting convention and which operations are counted;
- precision and instruction path, including whether tensor instructions execute;
- DRAM, L2, L1, or another byte boundary;
- measured sustainable versus specification/nameplate ceiling;
- sufficient parallelism and problem size;
- cache/reuse model and whether bytes are algorithmic or measured;
- excluded launch, synchronization, allocation, dependency, and control costs.

A point below both ceilings is not automatically “neither compute nor memory.” It may reflect a lower unmodeled ceiling, poor instruction mix, dependencies, divergence, occupancy/resource limits, insufficient waves, frequency/power state, or invalid counts. Hierarchical Roofline and profiler evidence refine the hypothesis.

**Worked Example (exercise ceilings, not a specific GPU):**
- *Input*: $P_{peak}=400$ TFLOP/s for the selected FP16 tensor path and $\beta=2.0$ TB/s at DRAM, both treated as attainable; FMA = 2 FLOPs; one-pass DRAM bytes (read $A$ and $B$ once, write $C$ once), 2 bytes/element.
- *Step 1 — ridge*: $I^*=\dfrac{400\times10^{12}\ \text{FLOP/s}}{2.0\times10^{12}\ \text{bytes/s}}=200$ FLOP/byte.
- *Step 2 — GEMV-like* ($M=K=4096$, $N=1$): $W=2MNK=2\cdot4096\cdot1\cdot4096=33{,}554{,}432$ FLOPs; $Q=s(MK+KN+MN)=2(4096^2+4096+4096)=33{,}570{,}816$ bytes (the $A$ term dominates); $I=W/Q\approx1.0$ FLOP/byte. Bounds: $W/P_{peak}=33{,}554{,}432/(400\times10^{12})\text{ s}=0.084$ µs and $Q/\beta=33{,}570{,}816/(2.0\times10^{12})\text{ s}=16.79$ µs, so $T\ge\max(0.084,16.79)$ µs and the memory term dominates ($I\approx1\ll I^*=200$).
- *Step 3 — GEMM* ($M=N=K=4096$): $W=2\cdot4096^3=137{,}438{,}953{,}472$ FLOPs; $Q=2\cdot3\cdot4096^2=100{,}663{,}296$ bytes; $I=W/Q=4096/3\approx1365$ FLOP/byte. Bounds: $W/P_{peak}=343.6$ µs and $Q/\beta=50.3$ µs, so $T\ge\max(343.6,50.3)$ µs and the compute term dominates ($I\approx1365>I^*=200$).
- *Result*: the screen assigns the GEMV to the bandwidth roof and the GEMM to the compute roof.
- *Interpretation / limits*: both are lower bounds on time under optimistic byte counts and attainable ceilings. A measured GEMV at 40 µs would sit well below the bandwidth roof and send you looking for launch overhead, insufficient parallelism, or a lower sustainable $\beta$. It would not refute the classification.

**Knowledge Check:** Which byte boundary defines $I$, and why can a point below both ceilings have an unmodeled limiter?

**Guided Practice:** Derive bounds using specification peaks and then measured sustainable ceilings; explain how the conclusion changes.

**Feedback Contract:**
- *Expected Evidence*: $W$, $Q$, $I$, $I^*$, and both time bounds with units; the FMA convention, precision/instruction path, and byte boundary; nameplate versus measured ceilings side by side; and at least one kernel whose classification changes when the measured ceiling or measured traffic replaces the optimistic one.
- *Common Failure*: combining a tensor-core FP16 peak with an FP32 CUDA-core kernel, or DRAM bandwidth with L2-level byte counts.
- *Diagnostic Hint*: are $W$ and $P_{peak}$ counted on the same instruction path, and are $Q$ and $\beta$ measured at the same boundary?
- *Concept to Revisit*: a Roofline is a bound, and the boundary must match.

**Learning Outcome:** Build and qualify a Roofline model without promoting it to a measured bottleneck or latency guarantee.

*(Effort: 60m instruction, 35m practice)*

---

### Lesson 2.5 — Shape-Dependent Regime Transitions

**Engineering Question:** Which shape and backend changes move execution among launch-, latency-, bandwidth-, and compute-limited regimes?

**Concepts & Definitions:**
For $C_{M\times N}=A_{M\times K}B_{K\times N}$ and FMA counted as two operations:

$$W_{GEMM}\approx2MNK.$$

Here $A$ is $M\times K$, $B$ is $K\times N$, and $C$ is $M\times N$. Each of the $MN$ outputs is a length-$K$ dot product, i.e. $K$ FMAs, so $MNK$ FMAs $=2MNK$ operations.

If each element occupies $s$ bytes and an optimistic screen assumes one read of $A$, one read of $B$, one write of $C$, and no read of the old $C$ (GEMM epilogue $C\leftarrow AB+\beta C$ with $\beta=0$; this $\beta$ is the GEMM scale factor, not the bandwidth $\beta$ of Lesson 2.4):

$$I_{one-pass}=\frac{2MNK}{s(MK+KN+MN)}.$$

Derivation: $A$, $B$, and $C$ hold $MK$, $KN$, and $MN$ elements, each touched once, so $Q=s(MK+KN+MN)$ bytes; divide $W_{GEMM}$ by $Q$. Two useful limits:

- **square** ($M=N=K=n$): $I=\dfrac{2n^3}{3sn^2}=\dfrac{2n}{3s}$, i.e. $n/3$ for FP16 ($s=2$): $4096/3\approx1365$ and $64/3\approx21$;
- **thin** ($N\ll M,K$): $MK$ dominates the denominator, so $I\approx\dfrac{2MNK}{sMK}=\dfrac{2N}{s}$, i.e. $I\approx N$ for FP16. Intensity grows roughly linearly with the batch-like dimension.

This is not measured intensity. Tiles reuse operands through registers/shared memory/cache, while imperfect implementations can reload them. The chosen $M,N,K$, layout, dtype, alignment, epilogue, library version, workspace, and concurrent work affect kernel selection and realized traffic.

Consequences:

- matrix-vector-like shapes often expose far less reuse than large matrix-matrix shapes;
- a larger batch can raise reuse and parallelism, then later hit another ceiling;
- padding can improve an instruction path while adding work;
- a nominally high-intensity kernel can remain latency-limited when the grid is too small;
- the same model phase can contain kernels in different regimes.

Therefore “prefill is compute-bound” and “decode is memory-bound” are useful hypotheses for some shapes, not architecture-independent facts. Module 04 measures their serving consequences; this module verifies the device path.

Numerical behavior remains part of the experiment. Floating-point addition is not generally associative, and fused or reordered reductions may differ. Define tolerances and downstream quality checks before declaring a faster path correct.

**Worked Example (same exercise ceilings as Lesson 2.4, $I^*=200$ FLOP/byte):**
- *Input*: FP16, $M=K=4096$, batch-like dimension $N\in\{1,8,64,512\}$, one-pass bytes.
- *Steps*: with $s=2$ the factor 2 in $2MNK$ cancels $s$, so $I=MNK/(MK+KN+MN)$. With $M=K=4096$, divide numerator and denominator by $4096^2$: $I=\dfrac{N}{1+N/2048}$, giving $1.0$, $7.97$, $62.1$, and $409.6$ FLOP/byte. Solving $I=I^*=200$ gives $N=200/(1-200/2048)\approx222$. The memory lower time is $Q/\beta=2(4096^2+2\cdot4096N)\text{ bytes}/(2.0\times10^{12}\text{ bytes/s})$: $16.8$, $16.8$, $17.3$, and $21.0$ µs for the four $N$, because the $4096^2$ weight-matrix term dominates $Q$.
- *Result*: the one-pass screen crosses the ridge between $N=64$ and $N=512$ (at $N\approx222$). The memory-bound lower time stays near 17–21 µs while $N$ grows 512×, which is why batching decode tokens can raise throughput until another ceiling appears (**O**, CLM-006).
- *Interpretation / limits*: a $64\times64\times64$ GEMM has $I=64/3\approx21$ FLOP/byte, but its $64\times64$ output may be covered by one or a few thread blocks, far too few to occupy every SM. Intensity does not encode grid size, and the kernel library may pick a different tiling for each $N$.

**Knowledge Check:** Why is the one-pass GEMM intensity not measured intensity, and how can padding both add work and improve execution?

**Guided Practice:** Sweep GEMV-like and GEMM-like shapes, dtype, alignment, and backend; record selected kernels, traffic, grid size, time, and numerical error.

**Feedback Contract:**
- *Expected Evidence*: for each shape, the selected kernel, grid size, one-pass and measured intensity, achieved throughput and bandwidth, time, and numerical error versus a reference (**O**, CLM-013); the $N$ at which the measured regime changes, compared with the screen's prediction.
- *Common Failure*: labeling all decode as memory-bound and all prefill as compute-bound without measuring the shapes involved.
- *Diagnostic Hint*: does the measured time grow with $N$ before the predicted ridge? If not, what else limits it?
- *Concept to Revisit*: shape-dependent reuse and parallelism.

**Learning Outcome:** Explain and measure shape-dependent bottleneck transitions while preserving numerical validity.

*(Effort: 55m instruction, 35m practice)*

---

### Lesson 2.6 — Profiler Ladder and Diagnostic Reasoning

**Engineering Question:** What is the least intrusive evidence needed to distinguish competing GPU-performance explanations?

**Concepts & Definitions:**
Use the least intrusive evidence that can answer the current question:

1. establish a synchronized unprofiled baseline with distributions;
2. use an application timeline to locate CPU gaps, launches, copies, streams, kernels, and synchronization;
3. select representative kernel instances, preserving shapes and context;
4. collect targeted kernel metrics for launch geometry, resource limits, achieved throughput, traffic, and stalls;
5. inspect source or SASS only when the hypothesis requires instruction-path evidence;
6. change one causal variable and remeasure the baseline.

Nsight Systems and Nsight Compute are complementary. Systems traces application scheduling and the CPU/GPU timeline. Compute collects detailed metrics for selected kernels (**O**, CLM-008). A kernel can be efficient while the application is slow; an application timeline can show a long kernel without explaining its internal limiter.

Profilers are interventions. Nsight Compute may replay kernels or ranges, save/restore memory, control caches/clocks, serialize launches, or patch instructions depending on the metric set (**O**, CLM-009). Keep replay mode, cache control, metric set, filtering, and concurrent activity in the evidence record. If the workload is nondeterministic or relies on concurrency, verify that the capture method preserves the behavior of interest.

Apply this chain:

```text
SYMPTOM
  → competing hypotheses
  → missing evidence
  → discriminating measurement or intervention
  → ranked explanation
  → change
  → unprofiled remeasurement
```

Example: “GPU utilization fell” can be explained by host launch gaps, smaller grids, dependency stalls, a faster kernel, throttling, memory faults, synchronization, or another process. Utilization alone cannot rank them (**H**, CLM-012).

**Worked Example (synthetic traces):**
- *Input*: a 10-iteration loop regressed from 4.0 ms to 6.0 ms per iteration after an upgrade. Aligned Nsight Systems timelines show the same kernel sequence with per-kernel durations within 2%, but a new 190 µs gap before each of 10 launches in the new trace.
- *Steps*: accumulated gaps $=10\times0.19=1.9$ ms, explaining $1.9$ of the $2.0$ ms regression; kernel-duration changes explain at most $0.02\times4.0=0.08$ ms (2% of the whole old iteration, an upper bound because kernels are only part of it). Together, at most $1.9+0.08=1.98$ ms accounts for nearly all of the $6.0-4.0=2.0$ ms regression.
- *Result*: host-side launch or dispatch gaps rank first, and kernel-internal limiters rank last.
- *Interpretation / limits*: the next measurement is a CPU sampling or API trace of the gap. Nsight Compute on the unchanged kernels would cost time without discriminating. Confirm by removing the suspected host cause and remeasuring *unprofiled*.

**Knowledge Check:** When should a timeline precede kernel metrics, and how can replay invalidate a concurrency-sensitive capture?

**Guided Practice:** Diagnose the same latency symptom after separately injecting a host gap, strided access, and register pressure.

**Feedback Contract:**
- *Expected Evidence*: unprofiled baseline distribution; the minimal capture for each injected fault with settings (replay mode, cache control, metric set); the alternatives ruled down and by which observation; the intervention; and unprofiled remeasurement.
- *Common Failure*: starting with a full Nsight Compute metric dump on every kernel, or reporting the profiled duration as the fix's effect.
- *Diagnostic Hint*: does the regression live inside kernels or between them on the timeline?
- *Concept to Revisit*: profiler scope and measurement intrusion.

**Learning Outcome:** Select profiler scope from a hypothesis and account for the profiler as an intervention.

*(Effort: 55m instruction, 35m practice)*

---

### Lesson 2.7 — Pinned Runtime Source Trace

**Engineering Question:** What timing semantics does the pinned benchmark utility implement, and what does it exclude?

**Concepts & Definitions:** A source trace distinguishes implementation-specific synchronization, warmup, repetition, aggregation, and returned raw measurements from general benchmarking principles.

At PyTorch commit `7ee5406f6686d190efc6571475f074ba1bc9a8c0`, inspect:

```text
torch/utils/benchmark/utils/timer.py
  Timer.blocked_autorange
    → Timer._estimate_block_size
        → Timer._timeit
            → timer
                → torch.accelerator.synchronize
                → timeit.default_timer
    → repeated measurement loop
    → Measurement(number_per_run, raw_times, task_spec)
```

Observed behavior at that revision:

- the module's default timer synchronizes an available accelerator before reading the host timer;
- block-size estimation acts as warmup and attempts to amortize timer overhead;
- `blocked_autorange` collects repeated timed blocks and returns raw times with the repetition count.

(**O**, CLM-011) Generalizable? **PARTIAL.** Synchronization, warmup, and replicates are general measurement concerns. The exact thresholds, call graph, and accelerator abstraction are PyTorch-revision-specific. This path does not measure an end-to-end serving request and its synchronization suppresses overlap outside the timed statement.

**Worked Example:**
- *Input*: a `Measurement` from `blocked_autorange` reporting `number_per_run=100` and raw times for 5 blocks.
- *Steps*: each raw time covers 100 executions bracketed by accelerator synchronization, so per-call time $t_{call,i}=t_{raw,i}/n$, where $t_{raw,i}$ is block $i$'s raw time and $n=$ `number_per_run` $=100$ (illustrative: $t_{raw,i}=2.0$ ms gives $t_{call,i}=20$ µs); the spread across the 5 blocks is block-level variation.
- *Result*: an operator-level, synchronized, amortized estimate.
- *Interpretation / limits*: the synchronization spreads launch overhead over 100 calls and removes overlap with other work. It does not include request queueing, and it is not evidence about production overlap.

**Knowledge Check:** Where does synchronization occur, and which surrounding asynchronous work falls outside the statement?

**Independent Practice:** Execute the pinned trace on one operator, compare it with device events and manual synchronized wall time, and mark unexecuted paths `TODO_VERIFY`.

**Feedback Contract:**
- *Expected Evidence*: revision, path, symbols, call path, configuration, static-versus-executed status, exact commands, and a comparison table of utility, event, and manual synchronized timings for one operator with an explanation of each difference.
- *Common Failure*: describing `blocked_autorange` as measuring end-to-end latency, or omitting that it synchronizes.
- *Diagnostic Hint*: where in the call path is `torch.accelerator.synchronize` invoked relative to `timeit.default_timer`?
- *Concept to Revisit*: utility-specific timing semantics.

**Learning Outcome:** Verify benchmark timing behavior from source without turning utility-specific semantics into a universal definition.

*(Effort: 45m instruction; trace practice counted in LAB E source trace)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- Williams, Waterman, and Patterson (2009), [*Roofline: An Insightful Visual Performance Model for Multicore Architectures*](https://escholarship.org/uc/item/78h8v7mr) — original bound and optimization model (CLM-004).
- NVIDIA, [*CUDA Programming Guide*](https://docs.nvidia.com/cuda/cuda-programming-guide/) — execution, synchronization, and architecture semantics (CLM-001, CLM-003, CLM-007; accessed 2026-09-25).
- NVIDIA, [*CUDA C++ Best Practices Guide*](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — correctness, timing, bandwidth, coalescing, and execution-configuration guidance (CLM-002, CLM-007, CLM-013; accessed 2026-09-25).

**CURRENT OPERATIONAL DOCUMENTATION**

- NVIDIA, [*Nsight Systems User Guide*](https://docs.nvidia.com/nsight-systems/UserGuide/) — CPU/GPU application timelines and CUDA tracing (CLM-008).
- NVIDIA, [*Nsight Compute Profiling Guide*](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — metrics, replay, overhead, reproducibility, and Roofline analysis (CLM-004, CLM-008, CLM-009).
- NVIDIA, [*GPU Performance Background*](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) and [*Matrix Multiplication Background*](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) — shape-dependent math/memory/latency intuition with explicitly historical hardware examples (CLM-006).
- PyTorch, [*CUDA semantics*](https://docs.pytorch.org/docs/main/notes/cuda.html) — framework-facing asynchronous execution and timing boundaries (CLM-003).

**CURRENT SOURCE SNAPSHOT**

- PyTorch commit `7ee5406f6686d190efc6571475f074ba1bc9a8c0`, `torch/utils/benchmark/utils/timer.py`, symbols `timer`, `Timer._estimate_block_size`, and `Timer.blocked_autorange`, statically verified 2026-09-25; symbol presence and the `torch.accelerator.synchronize` call re-checked 2026-09-30 (CLM-011).

**Currentness classification**

- **REFERENCE / BASELINE**: SIMT reasoning, explicit synchronization, effective bandwidth, and basic Roofline.
- **RECOMMENDED BASELINE WORKFLOW (this curriculum's recommendation, not a measured industry default)**: synchronized/warmed replicated benchmarks followed by timeline-first and targeted-kernel profiling. It is supported by the asynchrony, scope, and replay semantics documented in CLM-003, CLM-008, and CLM-009; how widely teams follow it is not claimed.
- **WORKLOAD-DEPENDENT**: compute-, memory-, latency-, launch-, occupancy-, or dependency-limited behavior; useful batch/shape; overlap; cache benefit.
- **FRONTIER / GENERATION-SPECIFIC**: new tensor instructions, asynchronous copy/memory engines, cluster-level features, graph/device launch, and hierarchy-specific optimizations. No source for these was opened in this revision, so they are topic pointers, not claims (`TODO_VERIFY`).
- **LEGACY WHEN UNIVERSALIZED**: fixed transaction rules from old compute capabilities, implicit warp-synchronous assumptions, and a single device's nameplate ridge used for every kernel.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow $\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}$.

### LAB A — Transactions, Layout, and Useful Bandwidth

- **Objective**: Determine when layout changes alter useful and physical traffic at named memory boundaries.
- **Pre-Registered Hypothesis**: Strided lane access will increase transferred traffic or reduce useful bandwidth on the selected path relative to aligned contiguous access.
- **Independent Variables**: Layout, offset, stride, problem size, and launch geometry.
- **Dependent Variables**: Correctness, useful bandwidth, hierarchy traffic, transactions, stalls, and kernel time.
- **Build**: implement copy/add and a tiled matrix multiply with shape and bounds assertions.
- **Measure**: sweep contiguous, offset, and strided accesses; calculate useful bytes/time and collect targeted L1/L2/DRAM traffic.
- **Break & Falsify**: Preserve element count and arithmetic while permuting layout so adjacent lanes access a large stride; if traffic efficiency and time do not respond on the same path, weaken the coalescing explanation.
- **Competing hypotheses**: transaction waste, cache reuse, insufficient grid size, alignment, compiler vectorization, or timing error.
- **Artifact**: source, correctness oracle, hardware/software manifest, byte equations, raw timings, targeted profile, and a boundary-labeled conclusion.
- **Alignment**: Lessons 2.1–2.2.
- **Effort Estimate**: 3h implementation, 1h analysis.

### LAB B — Asynchronous Timing Trap

- **Objective**: Establish which timing boundaries include queued device work and which perturb production overlap.
- **Pre-Registered Hypothesis**: Unsynchronized host timing will measure submission rather than completion for asynchronous work.
- **Independent Variables**: Timer, synchronization boundary, stream count, cold/warm state, and workload size.
- **Dependent Variables**: Submission, device, synchronized-wall, and application durations plus overlap visible in the timeline.
- **Build**: time identical GPU work with an unsynchronized host timer, synchronized host timer, device events, and `torch.utils.benchmark`.
- **Break & Falsify**: Add work on a second stream, move synchronization boundaries, and demonstrate which timer excludes queued work or destroys overlap.
- **Measure**: submission time, device interval, application interval, warm/cold distributions, and timeline.
- **Artifact**: a timing contract that makes start/end, streams, setup, and aggregation explicit.
- **Alignment**: Lesson 2.3.
- **Effort Estimate**: 2h implementation, 1h analysis.

### LAB C — Qualified Roofline Matrix

- **Objective**: Compare analytical bounds with measured execution across shape and arithmetic-intensity regimes.
- **Pre-Registered Hypothesis**: Regime classification will change for at least one shape when measured traffic and sustainable ceilings replace optimistic one-pass/specification values.
- **Independent Variables**: Operation family, shape, dtype, layout, backend, and hierarchy boundary.
- **Dependent Variables**: Work, traffic, intensity, selected path, grid size, throughput, duration, and numerical error.
- **Build**: create elementwise, reduction, GEMV-like, and GEMM-like cases across shapes and dtypes.
- **Derive**: $W$, algorithmic $Q$, $I$, attainable ceilings, ridge, and time lower bounds with units.
- **Measure**: selected instruction path, executed work where available, traffic at multiple hierarchy levels, throughput, grid size, and duration.
- **Break & Falsify**: Choose a tiny high-intensity problem that underfills the GPU and a large low-intensity problem with high bandwidth; revise classifications whose precision/path or traffic boundary does not match.
- **Artifact**: analytical-versus-measured table with every exclusion and no universal model-phase label.
- **Alignment**: Lessons 2.4–2.5.
- **Effort Estimate**: 3h implementation, 1h analysis.

### LAB D — Competing Bottleneck Diagnosis

- **Objective**: Distinguish multiple causes that present as slower elapsed time or lower aggregate utilization.
- **Pre-Registered Hypothesis**: Timeline, targeted counters, and a one-variable intervention can rule down at least two alternatives for each injected fault.
- **Independent Variables**: Injected fault and evidence available to the blinded diagnostician.
- **Dependent Variables**: Diagnosis rank, evidence requested, time-to-discrimination, correction effect, and false attribution.
- **Inject separately**: host launch gaps, divergence, strided loads, register pressure, synchronization, small grids, background GPU activity, and a constrained power/clock state where safe.
- **Blind diagnose**: start from the same symptom—higher elapsed time or lower aggregate utilization.
- **Required evidence**: unprofiled baseline, Systems timeline, selected Compute metrics, clocks/power, shape/backend, and controlled intervention.
- **Scoring rule**: no credit for naming a bottleneck without ruling down at least two alternatives.
- **Break & Falsify**: Include two faults with similar aggregate utilization and reject any diagnosis that cannot distinguish them through aligned evidence and intervention.
- **Artifact**: symptom → hypotheses → missing evidence → discrimination → ranked cause → intervention → remeasurement.
- **Alignment**: Lessons 2.1, 2.2, 2.5, and 2.6.
- **Effort Estimate**: 3h injection and diagnosis, 1h report.

### LAB E — Pinned Benchmark Source Trace

- **Objective**: Verify timing and synchronization semantics of the pinned benchmark utility.
- **Pre-Registered Hypothesis**: Different timing boundaries will report different values when asynchronous work exists outside the timed statement.
- **Independent Variables**: Timing method, external asynchronous work, block size, and warmup state.
- **Dependent Variables**: Raw times, repetitions, synchronization placement, and disagreement among timing boundaries.
- **Trace**: PyTorch `Timer.blocked_autorange` at the pinned commit through block-size estimation and synchronization.
- **Compare**: direct events, manual synchronized wall timing, and the utility on one stable operator.
- **Break & Falsify**: Add asynchronous work outside the timed statement and explain why each boundary reports a different result.
- **Artifact**: repository, commit, verification date, file, symbols, entry point, execution path, exact run commands, and `TODO_VERIFY` for any path not executed locally.
- **Alignment**: Lessons 2.3 and 2.7.
- **Effort Estimate**: 2h source trace (counted under `source_trace`), 1h comparison (3h total).

## 07 Break / Incident Scenarios

### Incident 02.1: Latency Doubled While “GPU Utilization” Fell

After a framework upgrade, a fixed-shape inference benchmark is twice as slow and a dashboard GPU-utilization average is lower. A teammate concludes that memory bandwidth is the bottleneck and proposes kernel fusion.

Competing hypotheses:

1. the selected kernel/backend changed;
2. host compilation or launch gaps entered the measured window;
3. synchronization or copies became serialized;
4. the grid underfills the GPU after a shape/padding change;
5. register/shared-memory changes reduced residency or caused spills;
6. memory layout increased transferred bytes;
7. clocks/power/thermal state changed;
8. the benchmark boundary or warmup changed;
9. another process or profiler perturbed execution.

**Diagnostic Protocol (Task):**

1. Freeze inputs, shape, dtype, seed, correctness tolerance, and software/hardware manifest.
2. Reproduce with synchronized unprofiled distributions.
3. Align old/new Systems timelines and identify changed gaps, kernels, copies, and sync.
4. Profile only representative changed kernels with the smallest discriminating metric set.
5. Verify source/backend selection.
6. Run one-variable interventions.
7. Remeasure unprofiled latency and numerical parity.
8. Separate immediate rollback from the long-term correction and define quantitative recovery criteria.

The aggregate utilization signal is useful for detecting change but insufficient to establish cause.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Defend a GPU Performance Diagnosis

The learner receives an unfamiliar GPU trace, selected kernel profiles, tensor shapes, and a regression report. They must:

1. define all timing and byte boundaries;
2. reconstruct launch geometry and resource constraints;
3. derive a unit-consistent Roofline screen with stated exclusions;
4. identify at least three competing explanations;
5. request the minimum discriminating measurements;
6. reproduce one failure in a controlled kernel;
7. propose an intervention with a falsifying prediction;
8. demonstrate numerical parity and unprofiled improvement;
9. state which conclusions are O, D, and H;
10. refuse unsupported translation from kernel behavior to TTFT/TPOT or capacity.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit the pinned PyTorch benchmark trace from Lesson 2.7/Lab E, including repository, revision, verification date, path, symbols, call path, runtime configuration, static-versus-executed status, commands, and `TODO_VERIFY` markers.

### Rubric Dimensions

A submission is **Competent** overall only if every dimension is at least Competent. Strong on one dimension does not offset Insufficient on another.

| Dimension (evidence) | Insufficient | Competent | Strong |
|---|---|---|---|
| **Correctness** (D8, Labs A/C) | Speed reported without parity, or tolerance unstated | Oracle, shape assertions, dtype/tolerance, and error distribution reported | Tolerance justified from precision/reduction order; parity checked on the changed and unchanged paths |
| **Timing boundary** (D1, Lab B) | Asynchronous submission timed as execution; clock/stream unstated | Clock, stream, synchronization, warmup, repetitions, and raw samples declared; boundary matches the question | Shows how each alternative boundary changes the number and which one production overlap needs |
| **Modeling: bandwidth/compute inference** (D3, Lab C) | Intensity computed without a named byte boundary or with mismatched peak/path; units mixed (e.g., MiB/ms labeled GiB/s) | Unit-consistent $W$, $Q$, $I$, ceilings, and bounds at one named boundary with exclusions | Compares optimistic and measured traffic/ceilings and predicts where the classification will flip |
| **Profiler intrusion** (D5, Lab D) | Counter dump without a question; profiled times reported as results | Timeline first, targeted metrics, capture settings (replay, cache control) recorded, unprofiled baseline kept | Demonstrates a case where the capture changed behavior and adjusts the method |
| **Diagnosis** (D4, D7, Incident) | One bottleneck named from utilization or from intensity alone | At least three alternatives, two ruled down by aligned evidence, ranked explanation | Ranks by discriminating power, including interacting causes, and states what remains unresolved |
| **Falsifier** (D7, all labs) | No condition stated under which the explanation fails | A falsifying observation stated before the intervention and checked | A falsifier that actually fired on one hypothesis and changed the ranking |
| **Reproducibility & remeasurement** (D6, D8, required artifact) | Environment cannot be reconstructed; only profiled improvement | Hardware/driver/toolkit/framework/revision/commands; unprofiled remeasurement with variance | Independent rerun or second device reproduces the direction of the effect |

*Calibration cases*: (a) a submission that computes arithmetic intensity correctly but never names the byte boundary or checks the hierarchy traffic is at most Competent on Modeling if its units are right, and Insufficient on Diagnosis if its bottleneck label rests only on that intensity; (b) a submission that times with an unsynchronized host timer is Insufficient on Timing boundary regardless of other work.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Explain GPU execution and residency | Lesson 2.1 | 2.1 Guided Practice; LAB A / LAB D | Mastery D2; Incident steps 3–5 | Launch/resource table (CLM-001, CLM-007); rubric: Diagnosis |
| Diagnose memory transactions | Lesson 2.2 | 2.2 Guided Practice; LAB A | Mastery D1, D3 | Useful-vs-transferred traffic report (CLM-002, CLM-010); rubric: Modeling |
| Time asynchronous work correctly | Lessons 2.3, 2.7 | LAB B / LAB E | Mastery D1; required source trace | Timing contract + pinned trace (CLM-003, CLM-011); rubric: Timing boundary |
| Build and qualify a Roofline model | Lessons 2.4–2.5 | LAB C | Mastery D3, D10 | Analytical-vs-measured table (CLM-004, CLM-005, CLM-006); rubric: Modeling |
| Select profiler scope and control intrusion | Lesson 2.6 | 2.6 Guided Practice; LAB D | Mastery D4–D5, D7; Incident steps 2–4, 7 | Diagnosis matrix with capture settings (CLM-008, CLM-009, CLM-012); rubric: Profiler intrusion, Falsifier |
| Preserve numerical validity | Lessons 2.5–2.6 | LAB C / LAB D | Mastery D8; Incident step 7 | Parity report (CLM-013); rubric: Correctness |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner can exit Module 02 when they can:

1. connect tensor shapes to grids, blocks, warps, resource residency, and memory traffic;
2. distinguish useful bytes from traffic at named hierarchy levels;
3. time asynchronous work with an explicit, defensible boundary;
4. derive Roofline bounds without presenting them as latency guarantees;
5. explain why occupancy and aggregate utilization are not unique bottleneck classifiers;
6. show a workload that transitions among latency, memory, and compute regimes;
7. use timeline and kernel profilers in a hypothesis-driven sequence;
8. account for profiler replay and measurement intrusion;
9. preserve correctness and numerical tolerance during performance experiments;
10. hand Module 04 measured device evidence without confusing it with serving metrics.

### Module Wrap-Up (Final Mental Model Reconstruction)

The final invariant: **a GPU bottleneck is a scoped causal claim supported by aligned work, byte, time, timeline, and intervention evidence—not a label read from one counter.**

## 12 Competency Targets

```yaml
competency:
  sfia: 4-5
  bloom: Analyze -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
