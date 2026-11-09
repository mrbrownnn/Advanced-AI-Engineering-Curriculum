# Module 05 — Inference Optimization

## 00 Why This Module Exists

An optimization is useful only when it improves the declared production objective without violating numerical, quality, memory, reliability, or portability constraints. A smaller checkpoint may execute through a slow conversion path. A faster attention microbenchmark may be irrelevant to end-to-end latency. Speculation may reduce target-model calls yet lose after draft and verification costs. Fusing more work may reduce launches while creating a resource-heavy kernel.

This module teaches a disciplined optimization loop:

$$
\text{workload contract} \to \text{baseline profile} \to \text{bottleneck hypothesis}
\to \text{mechanism} \to \text{correctness/quality gate}
\to \text{end-to-end measurement} \to \text{break and falsify}.
$$

The scope is single-node inference mechanisms: IO-aware attention, kernel fusion and Triton fundamentals, FP8/INT8/INT4 quantization, SmoothQuant/GPTQ/AWQ, and speculative decoding including Medusa and EAGLE. Scheduling policy remains in Module 04, KV allocation internals in Module 03, and distributed inference specialization in Module 20. This module considers those systems only where an optimization changes their inputs or costs.

**Research Cutoff:** 2026-09-30. The sources in Section 05 were re-read on 2026-10-01 at the depth stated per entry. No exhaustive search for work published between 2026-09-27 and the cutoff was done; that gap is an open verification item, not a claim that nothing new exists.

**Module Orientation**
- **Why This Matters**: Inference optimization sits where engineering and business constraints meet: every request consumes GPU time, so an unoptimized serving path raises both cost per request and latency. When traffic rises suddenly, a GPU running inefficient kernels reaches its capacity sooner, and users see queueing, timeouts, or rejected requests. This module stays within one GPU node and teaches how to choose, implement, and verify the mechanisms that reduce that cost: IO-aware attention, kernel fusion with Triton, quantization, and speculative decoding. None of these is a mandatory step. Each is adopted only when a measured bottleneck calls for it and the result still passes correctness and quality gates.
- **Engineering Problem**: Choose, implement, compose, and defend inference optimizations for a pinned model, runtime, hardware target, workload distribution, and SLO.
- **What You Will Do**: Compare reference and IO-aware attention; write and profile a fused Triton kernel; build and validate quantized artifacts; instrument speculative cycles; trace a current FlashAttention source path; and diagnose an optimization stack that regresses production goodput.
- **Environment**: Python 3.10+, PyTorch, Triton or an equivalent kernel environment, and a supported GPU for execution labs; analytical and artifact-manifest work can proceed without every backend.
- **Evidence Rule**: Distinguish source observation (**O**), assumption-backed derivation (**D**), and telemetry-dependent engineering hypothesis (**H**). Paper speedups do not transfer without their model, hardware, shapes, configuration, baseline, and measurement boundary.

## 01 Baseline Assumptions

- **Model semantics (Module 01):** causal attention and MHA/GQA tensor shapes (Lesson 1.3); autoregressive factorization, softmax with temperature, greedy/top-k/top-p, and the sampling contract (Lesson 1.2). Lesson 5.5 uses exactly these: the target and draft distributions are the *post-processing* next-token distributions of Lesson 1.2. The difference between semantic equivalence and bitwise identity is taught here at first use (Lesson 5.1), not assumed.
- **GPU performance (Module 02):** memory hierarchy, arithmetic intensity, Roofline bounds, launch overhead, warm-up and synchronization, kernel timelines, and profiler-counter limitations.
- **KV cache (Module 03):** logical KV payload, cache growth, layouts, and paged addressing. This module uses these facts but does not reteach allocation lifecycle.
- **Serving metrics (Module 04):** TTFT, ITL/TPOT, throughput, goodput, batching, arrival load, and scheduler interference. Optimization experiments must declare whether they are isolated kernels, offline model runs, or loaded serving runs.

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
  instruction: 6.25h      # lesson instruction lines: 60+50+50+50+70+45+50 min
  guided_practice: 3.5h   # lesson practice lines: 30+30+30+30+35+25+30 min
  labs: 16h               # LAB A 4h + LAB B 4h + LAB C 4h + LAB D 4h
  assessment: 3h          # Mastery transfer problem 2.5h + Incident 05.1 0.5h
  source_trace: 2h        # Section 09 Production Source Trace; LAB A attaches this artifact and does not count it again
  total: 30.75h
```
Each category is counted once. Lab analysis is not counted again as guided practice, and the source trace is not counted inside LAB A's 4h.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
Model + workload + hardware + runtime revision
                    |
             establish baseline
                    |
     profile time, bytes, FLOPs, shapes, memory
                    |
       form competing bottleneck hypotheses
                    |
  +-----------------+-------------------+
  |                 |                   |
attention IO     kernel/launch       representation / decode
tiling           and fusion          algorithm
  |                 |                   |
FlashAttention   Triton/compiler     quantization/speculation
  +-----------------+-------------------+
                    |
 correctness + numerical + task-quality gates
                    |
 cold/warm + single/batched + load measurements
                    |
      end-to-end latency, throughput, goodput
                    |
        ablation, adversarial workload, rollback
```

Every branch consumes resources and can shift the next bottleneck. The optimization target is not “fewer bits,” “fewer kernels,” or “more accepted tokens” in isolation. It is a declared end-to-end objective measured at a stable boundary.

## 04 Lessons

### Lesson 5.1 — IO-Aware Exact Attention

**Engineering Question:** How can dense attention retain its mathematical semantics while changing the memory traffic that dominates execution?

**Concepts & Definitions:** Exact dense-attention semantics, IO-aware tiling, online softmax, supported masks/layouts, and backend dispatch are separate contracts.

For one attention head, the reference expression is:

$$
S = QK^T/\sqrt{d}, \qquad P=\operatorname{softmax}(S), \qquad O=PV.
$$

where:
- $Q\in\mathbb{R}^{N_q\times d}$, $K\in\mathbb{R}^{N_k\times d}$, $V\in\mathbb{R}^{N_k\times d}$ — query, key, and value rows for $N_q$ queries and $N_k$ keys;
- $d$ — the head dimension (written $d_h$ in the source-trace paragraph below; same quantity), so $1/\sqrt{d}=d_h^{-1/2}$ is the default score scale;
- $S,P\in\mathbb{R}^{N_q\times N_k}$ — scaled scores and attention probabilities; softmax is applied to each row: $P_{ij}=e^{S_{ij}}/\sum_{j'}e^{S_{ij'}}$;
- $O\in\mathbb{R}^{N_q\times d}$ — the output; row $i$ is $o_i=\sum_jP_{ij}V_j$, a probability-weighted average of value rows.

The intermediate $S$ and $P$ each hold $N_q N_k$ entries, which is what a naive implementation writes to and reads back from HBM.

A straightforward implementation can materialize the $N_q\times N_k$ score or probability matrix in HBM. FlashAttention instead tiles Q, K, and V, maintains online softmax statistics, and avoids full intermediate materialization. The original paper classifies this as an exact dense-attention algorithm: it does not prune attention entries or approximate the softmax (**O**, CLM-001). “Exact” does not mean bitwise-identical results because tiling and reduction order change floating-point behavior. Two implementations are *semantically equivalent* when they compute the same mathematical function and agree within a declared numerical tolerance; they are *bitwise identical* only if every floating-point operation happens in the same order and precision.

**Online softmax: the state carried between tiles**

Softmax couples every key in a row through one normalizer, so a tile cannot be normalized on its own. The tiled algorithm keeps three pieces of state per query row (**O**, CLM-017):

- $m$ — the running maximum score over the keys seen so far;
- $\ell$ — the running normalizer $\sum_j e^{s_j-m}$ over the keys seen so far, expressed relative to the current $m$;
- an output accumulator — the partial weighted sum of value rows.

For one query row, let the key tiles be $s^{(1)},\dots,s^{(T)}$ with value rows $V^{(t)}$. Start from $m_0=-\infty$, $\ell_0=0$, $\tilde o_0=0$. For each tile $t$:

$$
m_t=\max\bigl(m_{t-1},\max_j s^{(t)}_j\bigr),\qquad
\alpha_t=e^{m_{t-1}-m_t},\qquad
\tilde p^{(t)}_j=e^{s^{(t)}_j-m_t},
$$

$$
\ell_t=\alpha_t\,\ell_{t-1}+\sum_j\tilde p^{(t)}_j,\qquad
\tilde o_t=\alpha_t\,\tilde o_{t-1}+\sum_j\tilde p^{(t)}_jV^{(t)}_j,\qquad
o=\tilde o_T/\ell_T .
$$

where $s^{(t)}_j$ is the $j$-th scaled score in tile $t$ (one row of $S$, split into $T$ tiles), $V^{(t)}_j$ the matching value row, $\tilde p^{(t)}_j$ an unnormalized probability relative to the current maximum, $\tilde o_t$ the unnormalized output accumulator, and $\alpha_t$ the rescale factor. ($\alpha_t$ here is unrelated to the acceptance parameter $\alpha$ of Lesson 5.5.)

*Why subtract the maximum.* Softmax is unchanged by subtracting any constant $c$ from every score: $e^{s_j-c}/\sum_{j'}e^{s_{j'}-c}=e^{-c}e^{s_j}/(e^{-c}\sum_{j'}e^{s_{j'}})=e^{s_j}/\sum_{j'}e^{s_{j'}}$. Choosing $c=\max_j s_j$ makes every exponent $\le0$, so no term overflows.

*Why multiplying by $\alpha_t$ is the right correction.* A term stored relative to the old maximum is $e^{s_j-m_{t-1}}$. Multiplying by $\alpha_t$ gives $e^{m_{t-1}-m_t}\,e^{s_j-m_{t-1}}=e^{s_j-m_t}$, the same term relative to the new maximum. Because $\ell_{t-1}$ and $\tilde o_{t-1}$ are sums of such terms, one multiplication rescales the whole sum.

Here $\alpha_1=e^{-\infty}=0$. The factor $\alpha_t$ re-expresses the old state relative to the new maximum; it equals 1 when the maximum did not change. This is the unscaled-accumulator form described in FlashAttention-2 Section 3.1.1, which divides by $\ell$ once at the end. FlashAttention Algorithm 1 instead keeps a normalized accumulator and rescales it on every tile, $o_t=(\ell_{t-1}\alpha_t\,o_{t-1}+\sum_j\tilde p^{(t)}_jV^{(t)}_j)/\ell_t$ (**O**, CLM-017). Both produce $\operatorname{softmax}(s)V$ in exact arithmetic: by induction, after tile $t$ the state is $\ell_t=\sum_{\text{seen}}e^{s_j-m_t}$ and $\tilde o_t=\sum_{\text{seen}}e^{s_j-m_t}V_j$, so their ratio is the softmax-weighted sum over all keys once every tile is processed (**D**, CLM-018).

*Step-by-step induction (unscaled form).*
1. Claim after tile $t$: $\ell_t=\sum_{\text{seen}}e^{s_j-m_t}$ and $\tilde o_t=\sum_{\text{seen}}e^{s_j-m_t}V_j$. It holds trivially before any tile ($\ell_0=0$, $\tilde o_0=0$).
2. Assume it for $t-1$. Then $\alpha_t\ell_{t-1}=\sum_{\text{old}}e^{s_j-m_t}$ by the rescaling identity above, and $\sum_j\tilde p^{(t)}_j=\sum_{\text{tile }t}e^{s_j-m_t}$. Their sum is $\sum_{\text{seen}}e^{s_j-m_t}$; the same argument applies to $\tilde o_t$.
3. After the last tile, $\tilde o_T/\ell_T=\sum_je^{s_j-m_T}V_j/\sum_{j'}e^{s_{j'}-m_T}=\sum_j\operatorname{softmax}(s)_jV_j$, since $m_T=\max_js_j$ cancels.

*Normalized form.* Writing $o_t=\tilde o_t/\ell_t$ and substituting $\tilde o_t=\alpha_t\tilde o_{t-1}+\sum_j\tilde p^{(t)}_jV^{(t)}_j$ with $\tilde o_{t-1}=\ell_{t-1}o_{t-1}$ gives exactly the Algorithm 1 update $o_t=(\ell_{t-1}\alpha_t\,o_{t-1}+\sum_j\tilde p^{(t)}_jV^{(t)}_j)/\ell_t$.

**Mechanistic invariants**

- Causal/local masks, scale, head layout, and GQA/MQA mapping must match the reference contract.
- The online softmax must maintain numerically stable row maxima and normalization across tiles.
- Backward support, dropout, head dimension, dtype, device generation, sequence shape, and variable-length packing are implementation capabilities, not properties implied by the name.
- A kernel can win its attention microbenchmark while leaving end-to-end inference unchanged if MLP, collectives, sampling, host work, or queueing dominates.

**Pinned interface path (static inspection, not executed).** At `Dao-AILab/flash-attention` commit `e9cf2c1651d2303191eb40a739a3c135fda00999`, `flash_attn_func` calls `FlashAttnFunc.apply`. `FlashAttnFunc.forward` sets the default scale to $d_h^{-1/2}$, pads the head dimension to a multiple of 8 when needed, and calls the wrapped `_flash_attn_forward`, which invokes `flash_attn_gpu.fwd`. `flash_attn_gpu` is bound at import time to the compiled `flash_attn_2_cuda` module, or to a Triton implementation when `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` or when a ROCm build cannot import the compiled module. The forward call returns the output together with `softmax_lse`, the per-row log-sum-exp of the scaled scores, which is the stored form of the $(m,\ell)$ statistics ($m+\log\ell$, because $\log\sum_je^{s_j}=\log\bigl(e^{m}\sum_je^{s_j-m}\bigr)=m+\log\ell$; in the worked example below, $5+\log1.2034380\approx5.18518$). `flash_attn_with_kvcache` calls `flash_attn_gpu.fwd_kvcache` directly and its docstring states that it does not support the backward pass (**O**, CLM-014). This shows which Python symbol dispatches to which backend object. It does not show which CUDA kernel variant ran for a given shape; that needs an executed trace.

**Currentness:** the original IO-aware exact-attention mechanism is **REFERENCE**, while selection of a particular backend is **WORKLOAD-DEPENDENT**. FlashAttention-3 targets Hopper GPUs and FlashAttention-4 (arXiv v1, March 2026) targets Blackwell GPUs; both are **FRONTIER** designs scoped to those hardware generations (**O**, CLM-002). Neither paper establishes portable speedups across other accelerators, and this module retains none of their reported speedup figures as general numbers.

**Worked Example (synthetic values).** One query row, four keys, two tiles of two keys each. Values are scalars so that the output is a single number.

- *Input*: scaled scores $s=[1,3\mid 2,5]$ and values $v=[10,20\mid 30,40]$.
- *Direct reference*: $m=\max s=5$; $e^{s-m}=[0.0183156,\,0.1353353,\,0.0497871,\,1]$; $\ell=1.2034380$; $p=e^{s-m}/\ell=[0.01522,\,0.11246,\,0.04137,\,0.83095]$; $o=\sum_jp_jv_j=36.88057$.
- *Tile 1*, $s=[1,3]$: $m_1=\max(-\infty,3)=3$, $\alpha_1=e^{-\infty-3}=0$, $\tilde p=[e^{1-3},\,e^{3-3}]=[0.1353353,\,1]$, $\ell_1=1.1353353$, $\tilde o_1=0.1353353\cdot10+1\cdot20=21.3533528$.
- *Tile 2*, $s=[2,5]$: $m_2=\max(3,5)=5$, $\alpha_2=e^{3-5}=0.1353353$, $\tilde p=[e^{2-5},\,e^{5-5}]=[0.0497871,\,1]$.
  - $\ell_2=0.1353353\cdot1.1353353+1.0497871=1.2034380$.
  - $\tilde o_2=0.1353353\cdot21.3533528+(0.0497871\cdot30+1\cdot40)=2.8898621+41.4936121=44.3834741$.
- *Result*: $o=44.3834741/1.2034380=36.88057$. It equals the direct reference, and $\ell_2$ equals the direct normalizer, which is the normalization check ($\sum_jp_j=1$).
- *Normalized-accumulator form*: $o_1=21.3533528/1.1353353=18.8079708$, then $o_2=(1.1353353\cdot0.1353353\cdot18.8079708+41.4936121)/1.2034380=36.88057$. Same result, one extra division per tile.
- *Tile order*: processing $[2,5]$ first gives $m=5$ immediately, $\ell: 1.0497871\to1.2034380$ and $\tilde o: 41.4936121\to44.3834741$ with $\alpha_2=1$. The mathematical result is unchanged.
- *The error this state prevents*: normalizing each tile by its own maximum and adding without $\alpha$ gives $\ell=1.1353353+1.0497871=2.1851224$, $\tilde o=62.8469649$, and $o=28.76130$. That is not attention over the four keys.
- *Interpretation and limits*: the four-key example was bit-identical in both tile orders in a float32 NumPy check. In 2,000 random eight-key float32 rows, 470 gave last-bit differences between the two tile orders. That is an illustrative CPU check of reduction-order sensitivity, not a measurement of any GPU kernel. Validate a kernel against a reference with a declared tolerance, not with bit equality.

**Knowledge Check:**
1. Why does “exact” not imply bitwise identity?
2. In the worked example, why must the tile-1 state be multiplied by $e^{m_1-m_2}$ before tile 2 is added?
3. Why does a faster attention kernel not establish service speedup?

**Guided Practice:** (a) By hand, process $s=[0,2\mid4,1]$, $v=[1,2\mid3,4]$ in two tiles and confirm the result against a direct softmax. (b) Build a shape matrix over batch, $N_q$, $N_k$, head dimension, causal mode, dtype, and GQA ratio. For each cell, predict the likely traffic advantage, validate outputs against a reference with declared tolerances, and measure warm kernel time, peak allocated memory, and end-to-end share. Identify at least one shape where dispatch or unsupported-path overhead weakens the expected benefit.

**Feedback Contract:**
- *Expected Evidence*: (a) tile 1 gives $m_1=2$, $\ell_1=1.1353353$, $\tilde o_1=2.1353353$; tile 2 gives $m_2=4$, $\alpha_2=0.1353353$, $\ell_2=1.2034380$, $\tilde o_2=3.4881345$, $o=2.89847$, equal to the direct softmax. (b) a shape matrix with semantic coverage, tolerance, selected backend, traffic and time boundaries, unsupported cases, and an end-to-end denominator. Knowledge check: reduction order changes rounding; the old state is relative to the old maximum; attention may be a small share of end-to-end time.
- *Common Failure*: adding per-tile normalizers without rescaling, initializing $m$ to 0 instead of $-\infty$ (wrong when all scores are negative), or reporting a kernel speedup as a service speedup.
- *Diagnostic Hint*: does your final $\ell$ equal $\sum_je^{s_j-\max s}$ computed directly? If not, find the first tile where the maximum changed.
- *Concept to Revisit*: online softmax state $(m,\ell,\text{accumulator})$; semantic equivalence versus bitwise identity.

**Learning Outcome:** Explain FlashAttention as a tiled IO transformation, carry the online-softmax state by hand, validate semantic coverage, and refuse to infer service speedup from an isolated favorable kernel result.

*(Effort: 60m instruction, 30m practice)*

---

### Lesson 5.2 — Kernel Fusion and Triton Fundamentals

**Engineering Question:** Which boundaries should be fused, and how do we know the fused kernel did not exchange one cost for another?

**Concepts & Definitions:** Vertical/horizontal fusion, launch amortization, intermediate traffic, resource pressure, compilation, and specialization are separate cost terms.

Vertical fusion can keep producer output on chip for a consumer and avoid intermediate HBM writes/reads. Horizontal fusion can combine independent small operations to amortize launch overhead. Compiler capture, legality, aliases, dynamic shapes, and graph breaks determine what can fuse. Generated code determines what actually fused.

A Triton kernel expresses a grid of program instances. Each program computes offsets, loads tiles under masks, transforms values, and stores results. Compile-time meta-parameters such as block sizes determine mappings and can be autotuned. This programming model lowers the barrier to custom kernels; it does not make an arbitrary kernel correct or fast (**O**, CLM-013).

**Fusion cost ledger**

$$
T_{candidate}=T_{launch}+T_{loads}+T_{compute}+T_{stores}+T_{sync}+T_{conversion}+T_{compile\ amortized}.
$$

where each $T$ is a time contribution for one candidate implementation: kernel launch overhead, HBM loads, arithmetic, HBM stores, synchronization, dtype/layout conversion, and compilation time divided by the number of calls that reuse the compiled kernel. A first-order traffic term is $T_{loads}+T_{stores}\approx\text{bytes moved}/\text{effective bandwidth}$.

This is an accounting decomposition, not a guarantee that terms add independently. Fusion targets launch and intermediate-memory terms, while potentially increasing register use, shared-memory use, synchronization, code size, compilation, or shape specialization.

**Required measurements**

- cold compile latency and warm steady-state latency;
- launch count and generated code;
- achieved bandwidth/FLOPs, registers, shared memory, occupancy, and spills where available;
- output error versus reference across adversarial sizes and boundary conditions;
- performance distribution over representative shapes, not one favorable tensor.

**Worked Example (synthetic exercise values, not measurements of any device).** Fuse `y = act(x + b)` from two kernels into one.

- *Input*: $N$ FP16 elements (2 bytes each); assumed launch cost 8 µs per kernel; assumed effective memory bandwidth 400 GB/s; compute time ignored.
- *Traffic*: unfused reads $x$, writes a temporary, reads it back, and writes $y$: $4\times2N$ bytes. Fused reads $x$ and writes $y$: $2\times2N$ bytes.
- *Time per term*: $T_{traffic}=\text{bytes}/(400\times10^9\ \text{B/s})$ and $T_{launch}=8\ \mu\text{s}\times\text{kernel count}$. For example, $8\times10^6\ \text{B}/(4\times10^{11}\ \text{B/s})=2\times10^{-5}$ s $=20\ \mu$s.
- *Large tensor*, $N=10^6$: unfused $8\times10^6$ B $\to$ 20 µs traffic + 16 µs launch = 36 µs. Fused $4\times10^6$ B $\to$ 10 µs + 8 µs = 18 µs.
- *Tiny tensor*, $N=10^3$: unfused 0.02 µs + 16 µs = 16.02 µs. Fused 0.01 µs + 8 µs = 8.01 µs. The whole gain is the removed launch.
- *Result*: this lower-bound ledger predicts a factor of two in both cases, for different reasons.
- *Interpretation and limits*: the ledger is **D**, not a measurement. If the fused kernel measures 30 µs on the large tensor, launch and traffic explain only 18 µs. The remaining 12 µs is unexplained by this model. Wider live ranges, register spills, lower occupancy, or a slower code path are candidate explanations (**H**) to test with generated code and counters (CLM-003).

**Knowledge Check:** Which cost term does fusion target, and which resource changes can reverse the expected gain?

**Guided Practice:** Fuse a bias-add plus activation or RMSNorm-like chain in Triton. Compare with a strong framework/compiler baseline. Sweep widths that are aligned and misaligned to the chosen block, and include a tiny tensor where launch overhead dominates and a large tensor where resource pressure matters. A slower fused result is valid evidence, not a failed lab.

**Feedback Contract:**
- *Expected Evidence*: numerical parity against the unfused reference; cold and warm times reported separately; launch counts and a generated-code excerpt showing what fused; resource counters where available; results over the swept shapes; a rollback criterion. Knowledge check: fusion targets launch and intermediate-traffic terms; register, shared-memory, occupancy, compile, and specialization costs can reverse the gain.
- *Common Failure*: timing the first (compiling) call as steady state, or comparing against an unoptimized eager baseline only.
- *Diagnostic Hint*: subtract the ledger's launch and traffic terms from the measured time. What is left, and which counter would explain it?
- *Concept to Revisit*: fusion cost ledger; cold versus warm boundaries (Module 02).

**Learning Outcome:** Implement and inspect a fused kernel, then attribute its outcome to measured traffic, launch, compilation, or resource effects.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 5.3 — Quantization Is a Deployment Contract

**Engineering Question:** What exactly has changed when a model is called “FP8,” “INT8,” or “INT4”?

**Concepts & Definitions:** Encoding, scaling, granularity, coverage, accumulator/output dtype, packing, executable kernel, quality, and memory are distinct deployment fields.

For uniform affine quantization:

$$
q=\operatorname{clamp}(\operatorname{round}(x/s)+z,q_{min},q_{max}),
\qquad \hat{x}=s(q-z).
$$

where:
- $x$ — the real value being quantized; $\hat x$ — its dequantized reconstruction;
- $s>0$ — the scale (real units per integer step); $z$ — the integer zero point, the code that represents real 0;
- $q$ — the stored integer code; $[q_{min},q_{max}]$ — the representable integer range, e.g. $[-8,7]$ for signed INT4 or $[0,255]$ for unsigned INT8;
- $\operatorname{round}$ — the rounding rule (e.g. round-to-nearest); $\operatorname{clamp}(v,a,b)=\min(\max(v,a),b)$.

*Step-by-step.* (1) Divide by $s$ to express $x$ in integer steps. (2) Round to the nearest step and shift by $z$. (3) Clamp into the code range. (4) Dequantize by undoing the shift and multiplying back by $s$. Two error regimes follow:
- *Inside the range* $s(q_{min}-z)\le x\le s(q_{max}-z)$: only rounding acts, so with round-to-nearest $|x-\hat x|\le s/2$.
- *Outside the range*: clamping acts, and the error is the distance from $x$ to the nearest endpoint, which can be arbitrarily large.

A larger $s$ widens the range (less clipping) but coarsens each step (more rounding error). For signed INT4 with $z=0$ and $s=0.05$, the range is $[-8s,7s]=[-0.40,0.35]$ and the rounding error is at most $0.025$; the worked example below shows one value of each regime.

The scale $s$, zero point $z$, clipping range, rounding rule, and granularity may be per tensor, channel, group, token, or block. Symmetric quantization often fixes $z=0$. Floating formats instead define exponent/mantissa encodings and are still commonly paired with scaling policies. FP8 E4M3 and E5M2 trade precision and dynamic range differently; “FP8” alone is incomplete (**O**, CLM-008). More generally, a bit-width label does not identify a quantized configuration (**D**, CLM-004).

For $N$ logical weights at $w$ packed payload bits:

$$
M_{payload}=\frac{Nw}{8}\ \text{bytes}.
$$

Here $Nw$ is the total payload in bits and dividing by 8 converts bits to bytes. With per-group metadata of $b_{meta}$ bits shared by a group of $g$ weights, the effective width is $w_{eff}=w+b_{meta}/g$ bits per weight and the total is $Nw_{eff}/8$ bytes. Unit reminder: 1 GB $=10^9$ B, 1 GiB $=2^{30}$ B $\approx1.0737\times10^9$ B, so GiB $=$ B$/2^{30}$.

This is a nominal lower bound. Add scales, zero points, padding, alignment, codebooks, duplicated weights, runtime workspaces, allocator fragmentation, and KV/activation memory before predicting device capacity. Do not convert the storage ratio into a latency ratio.

**Quantization manifest**

- model and tokenizer revisions;
- tensor coverage: weights, activations, KV, logits, or selected layers;
- encoding and group/granularity;
- calibration dataset and sampling procedure;
- clipping, smoothing, rounding, and exceptional layers;
- accumulator/output dtypes;
- packing layout, runtime, kernel/backend, and supported shapes;
- quality suite and numerical tolerances;
- memory, cold/warm latency, throughput, and goodput boundaries.

**Worked Example (hypothetical model; exercise assumptions).**

- *Input*: $N=8\times10^9$ weights, densely packed at $w=4$ bits; symmetric INT4 range $[-8,7]$; scale $s=0.05$, $z=0$; groups of 128 weights, each storing one 16-bit scale and one 16-bit zero point.
- *Step 1 — payload* (**D**, CLM-009): $M_{payload}=8\times10^9\times4/8=4\times10^9$ bytes $=4$ GB; $4\times10^9/2^{30}\approx3.725$ GiB.
- *Step 2 — per-group metadata*: $b_{meta}=16+16=32$ bits per group, so $32/128=0.25$ extra bits per weight and $w_{eff}=4.25$, so $8\times10^9\times4.25/8=4.25\times10^9$ bytes $\approx3.958$ GiB. The "4-bit" artifact already costs 4.25 bits per weight before padding, unquantized layers, or runtime workspace.
- *Step 3 — one value*: $x=0.37\Rightarrow x/s=7.4\Rightarrow q=7\Rightarrow\hat x=0.35$, rounding error $0.02$. $x=0.52\Rightarrow x/s=10.4\Rightarrow$ clamp to $7\Rightarrow\hat x=0.35$, clipping error $0.17$. (The first value is inside $[-0.40,0.35]$, so its error obeys $\le s/2=0.025$; the second is outside, so the bound does not apply.)
- *Result*: bit width fixes neither total bytes nor error. The same $w=4$ gives different bytes under a different group size and different error under a different scale or clipping range.
- *Interpretation and limits*: these numbers do not establish checkpoint size, device allocation, quality, or speed. Each needs its own measurement on the deployed artifact and runtime.

**Knowledge Check:** Why is bit width insufficient to identify arithmetic, kernel selection, total memory, or quality?

**Guided Practice:** (a) For $N=7\times10^9$ weights at 4 bits with groups of 64, each storing a 16-bit scale and a 4-bit zero point, compute payload bytes, effective bits per weight, and total bytes in GB and GiB. (b) Build a complete quantization manifest for two artifacts and reconcile packed payload, metadata, file bytes, device allocation, selected kernels, and quality gates.

**Feedback Contract:**
- *Expected Evidence*: (a) payload $7\times10^9\times4/8=3.5\times10^9$ B $\approx3.260$ GiB; metadata $(16+4)/64=20/64=0.3125$ bits per weight, so $w_{eff}=4.3125$; total $7\times10^9\times4.3125/8=3.7734375\times10^9$ B $\approx3.514$ GiB. (b) a manifest with every field in the list above filled or marked unknown, and file bytes reconciled against payload plus metadata. Knowledge check: bit width names storage width only; scale, granularity, coverage, accumulator dtype, packing, and kernel support are separate fields.
- *Common Failure*: reporting GB as GiB, omitting per-group metadata, or converting a 4× payload ratio into a 4× memory or latency claim.
- *Diagnostic Hint*: divide the artifact's file bytes by the logical weight count. If the answer is not close to your effective bits per weight, which byte category is missing?
- *Concept to Revisit*: quantization manifest; nominal payload as a lower bound.

**Learning Outcome:** Specify low precision as a reproducible artifact/runtime contract and separate representation, arithmetic, memory, quality, and performance.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 5.4 — SmoothQuant, GPTQ, and AWQ

**Engineering Question:** Which error source does each post-training method address, and which serving path realizes its promised benefit?

**Concepts & Definitions:** Calibration distribution, activation outliers, reconstruction objective, salient weights, artifact packing, and runtime kernel support determine different parts of the method contract.

- **SmoothQuant:** uses an equivalent offline rescaling to migrate activation-outlier difficulty into weights, enabling a W8A8 path (**O**, CLM-005). The smoothing parameter and calibration distribution influence the resulting ranges.
- **GPTQ:** performs one-shot weight quantization using approximate second-order information to control reconstruction error (**O**, CLM-006). The resulting artifact still needs a packing scheme and executable matrix kernel.
- **AWQ:** uses observed activations to identify/protect salient weights through per-channel scaling for low-bit weight-only quantization. Its system results include a tailored runtime (**O**, CLM-007).

These methods solve different problems. SmoothQuant is not “GPTQ for activations”; AWQ is not defined only by a bit width; and a GPTQ-format checkpoint does not prove that a selected runtime executes an optimized kernel rather than unpacking or dequantizing inefficiently.

**Evaluation matrix**

| Axis | Required evidence |
|---|---|
| Representation | actual tensor dtypes, scale tensors, group sizes, packed bytes |
| Numerical | layer/output error, NaN/Inf, outlier behavior, reference tolerance |
| Task quality | representative tasks, prompts, decoding settings, uncertainty |
| Kernel | selected operator/backend, conversion/dequant kernels, shape support |
| Memory | weights, metadata, workspace, activations/KV, peak and steady state |
| Serving | TTFT, ITL/TPOT, throughput, SLO-goodput at declared load |

**Break cases:** calibration drift; rare activation outliers; sensitive output heads; unsupported group size; small batches where unpack overhead dominates; long-prefill compute shapes where weight bandwidth is not dominant; and mixed batches that select a fallback path.

**Worked Example (synthetic values).** The equivalent rescaling that SmoothQuant relies on, on one output of a two-input linear map $y=\sum_jx_jw_j$.

- *Input*: activations $x=[40,\,0.5]$, weights $w=[0.1,\,0.2]$, per-channel smoothing factors $s=[8,\,1]$ (chosen by hand for the exercise, not by the paper's rule).
- *Step 1 — original*: $y=40\cdot0.1+0.5\cdot0.2=4.1$. Activation channel range ratio is $40/0.5=80$; weight ratio is $0.2/0.1=2$.
- *Step 2 — rescale*: $x'_j=x_j/s_j=[5,\,0.5]$ and $w'_j=s_jw_j=[0.8,\,0.2]$. (Here $s_j$ is a smoothing factor, not the quantization scale of Lesson 5.3.)
- *Step 3 — check*: $y'=5\cdot0.8+0.5\cdot0.2=4.1$. The function is unchanged in exact arithmetic (**D**). In general, $\sum_jx'_jw'_j=\sum_j(x_j/s_j)(s_jw_j)=\sum_jx_jw_j$ for any nonzero $s_j$, because each $s_j$ cancels.
- *Result*: the activation range ratio fell from 80 to $5/0.5=10$, and the weight range ratio rose from 2 to $0.8/0.2=4$. Difficulty moved from activations to weights; it did not disappear.
- *Interpretation and limits*: the identity holds before quantization. Whether the rescaled tensors quantize with acceptable error depends on the smoothing factors, calibration data, and granularity, and is an empirical question. A weight-only artifact from GPTQ or AWQ is a different case: it can reduce nominal weight bytes yet regress a small-batch workload if the selected path repeatedly converts or unpacks weights (**H**; verify from selected-operator logs).

**Knowledge Check:** Which method targets activation outliers, and why does a compatible checkpoint format not prove an optimized kernel executes?

**Guided Practice:** Compare two methods under a shared calibration/quality/runtime matrix, then shift the prompt domain and force one unsupported shape.

**Feedback Contract:**
- *Expected Evidence*: for each method, inspected tensor dtypes and scale tensors, calibration provenance, the selected operators for a supported and an unsupported shape, quality with uncertainty on the original and shifted domain, a memory breakdown, and loaded-serving metrics. Knowledge check: SmoothQuant targets activation outliers; a checkpoint format describes stored tensors, not the kernel the runtime selects.
- *Common Failure*: comparing methods with different calibration sets or runtimes, or reading "W4" from a file name instead of from the tensors.
- *Diagnostic Hint*: list the operators that actually executed for one decode step. Is there a dequantize or unpack operator in the list?
- *Concept to Revisit*: evaluation matrix; error source versus executable path.

**Learning Outcome:** Select and falsify a quantization method based on error source, runtime support, workload, and quality—not label popularity.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 5.5 — Exact Speculative Decoding and Its Cost Model

**Engineering Question:** When does doing extra draft work reduce time per committed target token?

**Concepts & Definitions:**

- **Target and draft distributions** $p(x)$, $q(x)$: the next-token distributions of the target and draft models for the same prefix, *after* the sampling contract of Module 01 Lesson 1.2 (temperature, top-k, top-p) has been applied to each. Greedy decoding is the special case where the processed distribution is one-hot (**O**, CLM-019).
- **Proposal length** $\gamma$ (often `k` in runtimes): draft tokens generated per cycle.
- **Committed reward** $A$: tokens appended to the output in one cycle. In the reference algorithm $A=n+1$, where $n\in[0,\gamma]$ is the number of accepted draft tokens and the extra token always comes from the target, so $1\le A\le\gamma+1$.
- **Per-position acceptance probability** $\beta=\sum_x\min(p(x),q(x))$ for one prefix. An "acceptance rate" is meaningless until its denominator is stated: proposed tokens, verified positions, or cycles.
- **Cycle time** $T_{cycle}$: draft, verification, and bookkeeping time for one cycle at a declared boundary.
- **Distribution contract**: the statement of what the speculative output is guaranteed to equal, and under which assumptions.

**Mechanism: the exact rule.** One cycle of the reference algorithm (Leviathan et al., Section 2.3 and Algorithm 1; Chen et al., Algorithm 2) is (**O**, CLM-019):

1. *Draft*: for $i=1,\dots,\gamma$, compute $q_i$ on the prefix plus earlier draft tokens and sample $x_i\sim q_i$.
2. *Verify*: one parallel target pass yields $p_1,\dots,p_{\gamma+1}$ for the prefix extended by $0,\dots,\gamma$ draft tokens.
3. *Accept or reject in order*: draw $r_i\sim U(0,1)$. Accept $x_i$ when $r_i\le p_i(x_i)/q_i(x_i)$, that is, with probability $\min(1,p_i(x_i)/q_i(x_i))$. Stop at the first rejection; $n$ is the number accepted before it.
4. *Emit one target token*: if a rejection happened at position $n+1$, sample from the **residual distribution**
   $$p'(x)=\frac{\max(0,\,p_{n+1}(x)-q_{n+1}(x))}{\sum_{x'}\max(0,\,p_{n+1}(x')-q_{n+1}(x'))}$$
   and discard all later draft tokens. If all $\gamma$ were accepted, sample from $p_{\gamma+1}$ unchanged.

**Why the emitted token is distributed as the target** (**D**, CLM-020, following Appendix A.1 of Leviathan et al.). For one position, token $x$ can be emitted in two ways: drafted and accepted, with probability $q(x)\min(1,p(x)/q(x))=\min(p(x),q(x))$; or after a rejection, with probability $(1-\beta)\,p'(x)$. The residual normalizer is $\sum_x\max(0,p-q)=1-\sum_x\min(p,q)=1-\beta$, so the second term is $p(x)-\min(p(x),q(x))$. The sum is $p(x)$ for every $x$ and for any $q$.

*Step-by-step.*
1. Accept term: $P(\text{draft }x)\cdot P(\text{accept}\mid x)=q(x)\min\bigl(1,p(x)/q(x)\bigr)$. If $p(x)\ge q(x)$ this is $q(x)$; otherwise it is $p(x)$. Either way it equals $\min(p(x),q(x))$.
2. Rejection probability: summing step 1 over $x$ gives $P(\text{accept})=\beta$, so $P(\text{reject})=1-\beta$.
3. Residual normalizer: for each $x$, $\max(0,p-q)=p-\min(p,q)$ (check both cases $p\ge q$ and $p<q$). Summing, $\sum_x\max(0,p-q)=1-\beta$.
4. Rejection term: $(1-\beta)\,p'(x)=(1-\beta)\,\dfrac{p(x)-\min(p(x),q(x))}{1-\beta}=p(x)-\min(p(x),q(x))$.
5. Total: $\min(p(x),q(x))+p(x)-\min(p(x),q(x))=p(x)$. (If $\beta=1$, then $p=q$ and rejection never happens.)

 Applying the rule position by position, with every draft after the first rejection discarded, gives a sequence distributed as target-only sampling.

**Assumptions behind the guarantee**

- $p$ and $q$ are the distributions that would actually be sampled: both are processed by the same sampling contract as the target-only decoder being matched.
- Each draft token is truly *sampled* from the $q_i$ used in the ratio. Drafting by argmax while plugging the full $q$ into the ratio breaks the guarantee (see the worked example).
- The uniform draws are independent of the draft tokens.
- The verification pass computes the same conditionals as sequential target decoding (same prefix, mask, cache, and precision). Chen et al. state the result as holding "within hardware numerics"; a batched verification kernel can differ in floating point from a sequential one.
- The guarantee is about the output *distribution*. With the same seed, a speculative run and a target-only run generally produce different text.

**Variants are separate contracts.** Do not carry the guarantee above to another acceptance rule.

| Variant | Acceptance rule | What it preserves | Status |
|---|---|---|---|
| Exact speculative sampling | $\min(1,p/q)$ plus residual resampling | The target's sampling distribution, under the assumptions above | **O**, CLM-019; hand check **D**, CLM-020 |
| Greedy verification | Both distributions one-hot: accept iff the draft token equals the target argmax, else emit the target argmax | The target's greedy sequence in exact arithmetic. It is the exact rule applied to argmax-processed distributions, so it says nothing about a sampled distribution. Near-ties need a stated rule | **D**, CLM-020 |
| Typical acceptance (Medusa, Section 2.3.1) | Accept when $p_{orig}(x)>\min(\epsilon,\delta e^{-H(p_{orig})})$; the first token is greedy and always accepted | Not the target distribution. The authors state that matching it is typically unnecessary and argue for quality empirically | **O**, CLM-021 |
| Tree or multi-candidate verification (Medusa trees, EAGLE draft trees) | A rule applied per tree node | Depends on the rule. EAGLE states that it applies speculative sampling recursively at each node and preserves the distribution. This module has not re-derived the multi-candidate proof | **O**, CLM-021 for the papers' statements; `TODO_VERIFY` for any specific runtime's tree verifier |
| Retrained backbone (Medusa-2) | Any | The backbone itself is fine-tuned, so the comparison target is the original model's quality, not its distribution | **O**, CLM-021 |

**Cost model.** Let a cycle propose $\gamma$ tokens, take

$$
T_{cycle}=T_{draft}(\gamma)+T_{verify}(\gamma)+T_{bookkeeping},
$$

and commit random reward $A$ output tokens. Under stationary ergodic renewal-style cycles with finite expectations (**D**, CLM-011):

$$
\text{token rate}=\frac{\mathbb{E}[A]}{\mathbb{E}[T_{cycle}]},
\qquad
\text{time/token}=\frac{\mathbb{E}[T_{cycle}]}{\mathbb{E}[A]}.
$$

where $T_{draft}(\gamma)$ is the time to produce $\gamma$ draft tokens, $T_{verify}(\gamma)$ the time of the target pass over $\gamma+1$ positions, $T_{bookkeeping}$ acceptance, cache, and scheduling work, and $\mathbb{E}[\cdot]$ an expectation over cycles.

*Why a ratio of expectations.* Over $C$ cycles, total time is $\sum_{c=1}^{C}T_c$ and total output is $\sum_{c=1}^{C}A_c$. Dividing numerator and denominator by $C$ gives $\bigl(\tfrac1C\sum T_c\bigr)/\bigl(\tfrac1C\sum A_c\bigr)$, which tends to $\mathbb{E}[T_{cycle}]/\mathbb{E}[A]$ as $C$ grows under the stated assumptions.

This is not generally $\mathbb{E}[T_{cycle}/A]$. For a finite set of measured cycles, the aggregate time per committed token is $\sum_cT_c/\sum_cA_c$, which equals the *token-weighted* mean of $T_c/A_c$, not the unweighted cycle mean. (Weight cycle $c$ by its token share $A_c/\sum_{c'}A_{c'}$: $\sum_c\frac{A_c}{\sum A}\cdot\frac{T_c}{A_c}=\frac{\sum_cT_c}{\sum A}$.) “Acceptance rate” is insufficient unless its denominator and proposal structure are defined. Record proposed tokens, committed tokens, first-rejection position, target calls, draft/verify times, synchronization, memory, and scheduler/batch context.

**What the reference paper's analytical model assumes.** If the per-position acceptance probabilities are treated as i.i.d. with mean $\alpha$, then $A$ is a capped geometric variable and $\mathbb{E}[A]=(1-\alpha^{\gamma+1})/(1-\alpha)$ (Leviathan et al., Eq. 1). If, in addition, $\gamma+1$ positions verify in the time of one target step and one draft step costs $c$ target steps, the expected wall-time improvement is $(1-\alpha^{\gamma+1})/((1-\alpha)(\gamma c+1))$ for long generations (their Theorem 3.8) (**O**, CLM-019). This is an analytical model, not a measurement. The same paper's Appendix A.3 compares it with measured run times, reports cases where the measured improvement is below the prediction, and attributes the gap to implementation differences and to the i.i.d. simplification.

*Step-by-step for the two model formulas.* Here $\alpha$ plays the role of an average $\beta$, and $c=$ (time of one draft step)/(time of one target step).
1. $A\ge k$ (for $k=1,\dots,\gamma+1$) happens exactly when the first $k-1$ drafts are all accepted, so $P(A\ge k)=\alpha^{k-1}$.
2. For a positive integer variable, $\mathbb{E}[A]=\sum_{k\ge1}P(A\ge k)=\sum_{k=1}^{\gamma+1}\alpha^{k-1}=\dfrac{1-\alpha^{\gamma+1}}{1-\alpha}$ (finite geometric series, $\alpha<1$).
3. Point probabilities: $P(A=k)=\alpha^{k-1}(1-\alpha)$ for $k\le\gamma$, and $P(A=\gamma+1)=\alpha^{\gamma}$ (the cap).
4. Cost of one cycle in target-step units: $\gamma$ draft steps cost $\gamma c$, plus one target step for verification, total $\gamma c+1$.
5. Target-only decoding commits one token per target step, so the improvement factor is (tokens per cycle)/(target steps per cycle) $=\mathbb{E}[A]/(\gamma c+1)$, which is the Theorem 3.8 expression.

**Adversarial regimes**

- hard or distribution-shifted prompts reduce accepted tokens;
- a draft model is too large or resides on a contested device;
- verifier cost grows sharply with tree/block shape;
- high target batching already amortizes weight movement;
- extra model/KV/workspace memory reduces serving concurrency;
- long accepted bursts change streaming event semantics and ITL measurement.

**Worked Example A — acceptance and residual on a three-token vocabulary (synthetic distributions).**

- *Input*: vocabulary $\{a,b,c\}$; target $p=[0.5,\,0.3,\,0.2]$; draft $q=[0.2,\,0.2,\,0.6]$; one position.
- *Steps*:

| Token | $p$ | $q$ | Accept prob. $\min(1,p/q)$ | Drafted and accepted $\min(p,q)$ | $\max(0,p-q)$ | Residual $p'$ | Emitted after rejection $(1-\beta)p'$ | Total |
|---|---|---|---|---|---|---|---|---|
| a | 0.5 | 0.2 | 1 | 0.2 | 0.3 | 0.75 | 0.3 | 0.5 |
| b | 0.3 | 0.2 | 1 | 0.2 | 0.1 | 0.25 | 0.1 | 0.3 |
| c | 0.2 | 0.6 | 1/3 | 0.2 | 0 | 0 | 0 | 0.2 |
| Sum | 1 | 1 | — | $\beta=0.6$ | 0.4 | 1 | 0.4 | 1 |

- *Row c, step by step*: accept probability $\min(1,0.2/0.6)=1/3$; drafted-and-accepted $0.6\cdot1/3=0.2=\min(0.2,0.6)$; $\max(0,0.2-0.6)=0$, so c is never emitted after a rejection. Row a: $\max(0,0.5-0.2)=0.3$, residual $0.3/0.4=0.75$.
- *Result*: the Total column equals $p$. The rejection mass $1-\beta=0.4$ equals the residual normalizer, which is the normalization check.
- *Counterexample 1 — resample from $p$ instead of the residual*: output $=\min(p,q)+(1-\beta)p=[0.2+0.4\cdot0.5,\;0.2+0.4\cdot0.3,\;0.2+0.4\cdot0.2]=[0.40,\,0.32,\,0.28]\ne p$. Token c is over-produced because it can arrive both ways.
- *Counterexample 2 — draft by argmax but use the full $q$ in the ratio*: the draft is always c, accepted with probability $1/3$, else the residual is sampled: output $=\tfrac13[0,0,1]+\tfrac23[0.75,0.25,0]=[0.5,\,1/6,\,1/3]\ne p$.
- *Greedy contract*: argmax processing turns $p$ into $[1,0,0]$ and $q$ into $[0,0,1]$. The draft c is always rejected ($p/q=0$), the residual is $[1,0,0]$, and a is emitted. The output is the target's greedy token and the reward is $A=1$ in every such cycle: correct, with no speculative gain.
- *Under the i.i.d. model*, $\alpha=0.6$ and $\gamma=2$ give $P(A=1,2,3)=0.4,\,0.24,\,0.36$ (that is, $1-\alpha$, $\alpha(1-\alpha)$, $\alpha^2$) and $\mathbb{E}[A]=(1-0.6^3)/0.4=0.784/0.4=1.96$; check: $1\cdot0.4+2\cdot0.24+3\cdot0.36=1.96$. The model's wall-time factor is $1.96/(2c+1)$: $1.96/1.2\approx1.63$ at $c=0.1$ and $1.96/2=0.98$ at $c=0.5$. The same acceptance gives a predicted gain or a predicted loss depending on draft cost.
- *Interpretation and limits*: the table proves the single-position identity for these distributions; the algebra above proves it for any $p$, $q$. It says nothing about speed. LAB D extends the check to two positions by exact enumeration.

**Worked Example B — two cycles with different reward and time (synthetic records).**

- *Input*: $\gamma=4$; target-only decoding at the same boundary takes 25 ms per token (synthetic).

| Cycle | Accepted drafts $n$ | $A=n+1$ | $T_{cycle}$ (ms) | $T/A$ (ms/token) |
|---|---|---|---|---|
| 1 | 3 | 4 | 60 | 15 |
| 2 | 0 | 1 | 40 | 40 |

- *Steps*: $\sum T/\sum A=100/5=20$ ms/token. Mean of $T/A=(15+40)/2=27.5$ ms/token. Token-weighted mean of $T/A=(4\cdot15+1\cdot40)/5=20$ ms/token.
- *Result*: five tokens were committed in 100 ms, so the aggregate is 20 ms/token and the speedup on these records is $25/20=1.25$. The unweighted mean of ratios, 27.5 ms/token, would report a regression that did not happen.
- *Same acceptance, opposite outcome*: keep the token counts (3 of 8 proposed drafts accepted, 37.5%) but let the cycles take 80 ms and 55 ms because the draft is slower. Then $\sum T/\sum A=135/5=27$ ms/token and the factor is $25/27\approx0.93$, a regression.
- *Interpretation and limits*: two cycles are not a stationary sample. A decision needs many cycles per workload stratum with uncertainty, the same output-distribution contract in both arms, and the memory and batching context. Acceptance rate alone fixed neither result.

**Knowledge Check:**
1. Why is the residual $\max(0,p-q)$ normalized by $1-\beta$, and what goes wrong if the rejected position is resampled from $p$?
2. Why is acceptance percentage insufficient to infer speedup?
3. Which assumptions support the renewal-style ratio, and which extra assumptions does the i.i.d. wall-time formula add?

**Guided Practice:** (a) For $p=[0.6,0.3,0.1]$ and $q=[0.3,0.3,0.4]$, compute the acceptance probabilities, $\beta$, the residual, and the emitted distribution. (b) For cycle records $(T,A)=(90,5),(30,1),(45,2)$ in ms and tokens, compute $\sum T/\sum A$ and the mean of $T/A$. (c) With your own measured cycle records—not a supplied speedup—compute both ratios, explain the difference, and compare against target-only decoding at an identical output-distribution contract, batch/concurrency, and timing boundary.

**Feedback Contract:**
- *Expected Evidence*: (a) acceptance probabilities $[1,\,1,\,0.25]$; $\beta=0.7$; residual $[1,0,0]$; emitted $[0.6,0.3,0.1]=p$. (b) $165/8=20.625$ ms/token versus $(18+30+22.5)/3=23.5$ ms/token. (c) cycle-level reward and cost, the denominator of every rate, the distribution contract of the variant used, memory, batching/load context, and a target-only baseline. Knowledge check 3: the ratio needs stationary cycles with finite means; the wall-time formula adds i.i.d. acceptance, free parallel verification, a single cost ratio $c$, and long generations.
- *Common Failure*: resampling the rejected position from $p$; counting the bonus target token as an "accepted draft"; averaging $T/A$ over cycles; inferring speedup from acceptance rate.
- *Diagnostic Hint*: does $\sum_x\max(0,p-q)$ equal $1-\beta$ in your table? Does your aggregate equal total time divided by total committed tokens?
- *Concept to Revisit*: acceptance and residual rule; ratio of sums versus mean of ratios.

**Learning Outcome:** State and hand-verify the exact acceptance/residual rule and its assumptions, keep other acceptance variants under their own contracts, and prove or disprove speculative benefit using total cycle reward and cost rather than acceptance rate.

*(Effort: 70m instruction, 35m practice)*

---

### Lesson 5.6 — Medusa, EAGLE, and Learned Drafting

**Engineering Question:** How do proposal architecture and training change the speculative trade space?

**Concepts & Definitions:** Auxiliary heads, draft models, feature-level prediction, candidate trees, verification shape, retraining, and approximation guarantees distinguish speculative families.

Medusa adds decoding heads that propose multiple future tokens and uses tree-based verification. Its variants differ in whether the backbone remains frozen (Medusa-1) or is fine-tuned with the heads (Medusa-2), and in their quality/training trade-offs (**O**, CLM-012). Medusa can verify with rejection sampling or with its typical-acceptance rule; the latter does not target the original model's distribution (**O**, CLM-021). EAGLE predicts at the target model's second-to-top-layer feature level and feeds a token sequence advanced by one step to resolve feature uncertainty; its paper states that it keeps the speculative-sampling acceptance rule and so preserves the output distribution (**O**, CLM-012, CLM-021). EAGLE-3 replaces feature prediction with direct token prediction and multi-layer feature fusion, and reports both batch-size-1 latency and batched throughput in a serving framework. The same paper notes that the spare compute speculation exploits shrinks as batch size grows (**O**, CLM-016).

The comparison contract is broader than speed:

- target/draft compatibility and retraining requirements;
- artifact and serving-memory overhead;
- candidate-tree width/depth and verification shape;
- exact-distribution versus declared approximate acceptance;
- acceptance by prompt domain and decoding settings;
- single-request latency versus batched throughput;
- failure/fallback behavior and rollout complexity.

Treat Medusa, EAGLE, and later variants as **WORKLOAD-DEPENDENT** or **FRONTIER**, not universal defaults. Their paper results motivate experiments but do not select a production method without replication on the target stack.

**Worked Example (synthetic cycle summaries for two hypothetical drafts on one workload).**

- *Input*: draft X commits $\mathbb{E}[A]=3.2$ tokens per cycle with $\mathbb{E}[T_{cycle}]=70$ ms. Draft Y commits $\mathbb{E}[A]=2.4$ with $\mathbb{E}[T_{cycle}]=48$ ms. Both use the exact acceptance rule.
- *Steps*: time per token is $\mathbb{E}[T_{cycle}]/\mathbb{E}[A]$ (Lesson 5.5 cost model). X costs $70/3.2=21.875$ ms/token. Y costs $48/2.4=20$ ms/token. Equivalently, X yields $3.2/0.070\approx45.7$ tokens/s and Y $2.4/0.048=50$ tokens/s.
- *Result*: the draft with the higher committed reward is slower per token, because its wider tree or larger draft makes each cycle more expensive.
- *Interpretation and limits*: this compares single-request time per token only. If X also holds more draft parameters, tree workspace, or KV state, it can lower the number of concurrent sequences the device can hold. That effect is a hypothesis (**H**) until memory and loaded throughput are measured on the target stack. If X used typical acceptance instead, the two would not even share an output contract, and a quality gate would be needed before any speed comparison.

**Knowledge Check:** Which comparisons require a distribution-preservation statement, and which costs lie outside acceptance rate?

**Guided Practice:** Build a comparison table covering training, target compatibility, proposal structure, quality contract, memory, verification, batching, and fallback behavior.

**Feedback Contract:**
- *Expected Evidence*: one row per family with the acceptance rule and its distribution contract (exact, greedy-equivalent, typical/approximate, or unverified), whether the backbone is modified, added parameters and runtime memory, verification shape, the batch conditions of any cited result, and fallback behavior. Knowledge check: any comparison of sampled outputs needs the statement; draft time, verification shape, memory, training, and batching interaction lie outside acceptance rate.
- *Common Failure*: ranking families by one paper's speedup or acceptance value, or comparing an exact method with an approximate one on speed alone.
- *Diagnostic Hint*: for each row, can you state what the output is guaranteed to equal? If not, what quality gate replaces the guarantee?
- *Concept to Revisit*: variant contracts table in Lesson 5.5; cycle cost ledger.

**Learning Outcome:** Distinguish speculative families mechanistically and design a fair comparison that includes training, quality, memory, batching, and operations.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 5.7 — Bottleneck-Driven Composition and Diagnosis

**Engineering Question:** How do we compose optimizations without losing causal attribution?

**Concepts & Definitions:** Main effects, interactions, bottleneck shifts, semantic/quality gates, end-to-end boundaries, canaries, and rollback govern composition.

Start with a pinned baseline. Measure the fraction of time and resources attributable to the target. Apply one mechanism. Re-run correctness/quality gates. Then measure at kernel, model, and loaded-serving boundaries. Only after main effects are understood should combinations be tested.

For identical boundaries:

$$
S_{e2e}=\frac{T_{baseline}}{T_{optimized}}.
$$

where $S_{e2e}$ is the end-to-end speedup and $T_{baseline}$, $T_{optimized}$ are times for the same work measured at the same boundary ($S>1$ is a gain, $S<1$ a regression).

*Amdahl-style bound, step by step.* Let $f$ be the target's fraction of baseline time and $k$ its local speedup. If everything else is unchanged, $T_{optimized}=T_{baseline}\bigl((1-f)+f/k\bigr)$, so $S_{e2e}=1/\bigl((1-f)+f/k\bigr)$. As $k\to\infty$ (the target costs nothing), $S_{e2e}\to1/(1-f)$. In the worked example below, $f=10/40=0.25$ and $k=2$ give $1/(0.75+0.125)\approx1.143$, with ceiling $1/0.75\approx1.333$.

An Amdahl-style upper bound is useful only if non-target work remains unchanged. Quantization can change kernel selection; fusion can change layouts; speculation changes decode shapes and scheduler work; attention kernels can change memory headroom. These interactions violate a naïve fixed-fraction assumption.

Use the incident protocol:

$$
\text{symptom}\to\text{competing hypotheses}\to\text{missing evidence}
\to\text{discriminating measurement}\to\text{ranked explanation}
\to\text{intervention}\to\text{remeasurement}.
$$

**Necessary but insufficient signals**

- lower model-weight bytes do not prove lower latency;
- fewer launches do not prove faster kernels;
- higher occupancy does not prove more useful work;
- higher acceptance percentage does not prove speculative speedup;
- lower kernel time does not prove better TTFT, ITL, throughput, or goodput;
- close perplexity does not prove application-quality equivalence.

**Worked Example (synthetic step times).**

- *Input*: a baseline decode step takes 40 ms: attention 10 ms, all other work 30 ms. Candidate 1 halves attention time. Candidate 2 (quantization) cuts other work to 20 ms.
- *Step 1 — candidate 1 alone*: $10\to5$ ms, step $=35$ ms, $S=40/35\approx1.143$. Amdahl bound if attention cost nothing: $40/30\approx1.333$.
- *Step 2 — candidate 2 alone*: step $=10+20=30$ ms, $S=40/30\approx1.333$.
- *Step 3 — both, if the two effects are independent and additive*: step $=5+20=25$ ms, $S=40/25=1.6$. The product of the two separate speedups is $1.143\times1.333\approx1.524$, which is not $1.6$. Speedups measured one at a time do not multiply, even with no interaction.
- *Step 4 — both, with an interaction*: suppose the quantized path changes the attention inputs' dtype and the attention kernel then takes 6 ms instead of 5 ms. Step $=26$ ms, $S=40/26\approx1.538$. Only a measurement of the combined configuration reveals this.
- *Result*: after candidate 2, attention is $10/30=33\%$ of the step instead of $25\%$, so the bottleneck share moved; the fixed-fraction assumption behind the Step 1 bound no longer describes the system.
- *Interpretation and limits*: all values are exercise inputs. The example shows why the plan measures main effects first, then the combination, at the same boundary (CLM-015 states the diagnostic hypothesis this supports).

**Knowledge Check:** Which observation indicates a bottleneck shift, and why should combined changes follow main-effect measurements?

**Guided Practice:** (a) A baseline step takes 50 ms: attention 20 ms, other 30 ms. An attention kernel is twice as fast. Compute the new step time, the speedup, and the Amdahl bound. (b) Design a factorial toggle matrix for attention, fusion, quantization, and speculation with quality gates and a workload that can falsify each local win.

**Feedback Contract:**
- *Expected Evidence*: (a) 40 ms, $S=1.25$, bound $50/30\approx1.667$. (b) a pinned baseline; every single-factor toggle and the needed combinations; a quality gate per mechanism; matched kernel, model, and loaded-serving boundaries; competing hypotheses for any interaction; canary metrics and rollback triggers. Knowledge check: a change in the time or resource share of the non-target work indicates a shift; without main effects, a combined result cannot be attributed.
- *Common Failure*: multiplying separately measured speedups, or enabling everything at once and attributing the net change to one mechanism.
- *Diagnostic Hint*: after each toggle, recompute each component's share of the step. Which share grew?
- *Concept to Revisit*: end-to-end speedup at identical boundaries; incident protocol.

**Learning Outcome:** Construct an optimization evidence chain that survives bottleneck shifts, interactions, and adversarial workloads.

*(Effort: 50m instruction, 30m practice)*

## 05 Literature & Production Source Map

Each entry states what was read on 2026-10-01 and what the module uses it for. "Abstract only" means the method and results sections were not re-read in this pass; such entries support only the scoped statement beside them. No reported speedup is used as a general number.

**REFERENCE / BASELINE**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao, Fu, Ermon, Rudra, and Ré (arXiv v2, 2022).
  - *Read*: Section 3.1 (tiling, the $m$/$f$/$\ell$ softmax decomposition, recomputation) and Algorithm 1.
  - *Scope*: the exact tiled algorithm and its per-row statistics in Lesson 5.1 (CLM-001, CLM-017).
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao (arXiv v1, 2023).
  - *Read*: Section 2.3.1 (online softmax in two blocks) and Section 3.1.1 (unscaled accumulator, stored log-sum-exp) with Algorithm 1.
  - *Scope*: the recurrence form used in the Lesson 5.1 worked example (CLM-017).
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan, Kalman, and Matias (ICML 2023; arXiv v2).
  - *Read*: Sections 2.2–2.3 and Algorithm 1 (standardized sampling, acceptance, residual), Sections 3.1–3.5 (acceptance rate, Eq. 1, Theorem 3.8 and its assumptions), Appendix A.1 (correctness proof), Appendix A.3 (predicted versus measured).
  - *Scope*: the exact rule, its proof, and the analytical model in Lesson 5.5 (CLM-010, CLM-019).
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al. (arXiv v1, 2023).
  - *Read*: Algorithm 2 and the modified-rejection-sampling section, including the "within hardware numerics" qualification.
  - *Scope*: independent statement of the same acceptance/residual rule (CLM-010, CLM-019).
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) — Micikevicius et al. (arXiv v2, 2022). *Read*: abstract only. *Scope*: E4M3/E5M2 encodings exist with different range/precision trade-offs (CLM-008); scaling and kernel policy remain separate.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al. (ICML 2023; arXiv v7). *Read*: abstract, and the smoothing-factor passage that defines migration strength. *Scope*: equivalent offline rescaling for W8A8 (CLM-005). The Lesson 5.4 example uses hand-chosen factors, not the paper's rule.
- [GPTQ](https://arxiv.org/abs/2210.17323) — Frantar et al. (ICLR 2023; arXiv v2). *Read*: abstract only. *Scope*: one-shot weight quantization from approximate second-order information (CLM-006).

**WORKLOAD-DEPENDENT**

- [AWQ](https://arxiv.org/abs/2306.00978) — Lin et al. (MLSys 2024; arXiv v6). *Read*: abstract and the scaling-search passage. *Scope*: activation-aware per-channel scaling, with system results that include its own runtime (CLM-007).
- [Medusa](https://arxiv.org/abs/2401.10774) — Cai et al. (ICML 2024; arXiv v3). *Read*: abstract, Section 2 overview, Section 2.3.1 (typical acceptance). *Scope*: heads plus tree verification; Medusa-1 versus Medusa-2; typical acceptance does not target the original distribution (CLM-012, CLM-021).
- [EAGLE](https://arxiv.org/abs/2401.15077) — Li et al. (arXiv v3, March 2025). *Read*: abstract and the passages stating that it keeps the speculative-sampling rule, including for its draft tree. *Scope*: feature-level drafting and the paper's own distribution statement (CLM-012, CLM-021). EAGLE attributes its tree verifier to SpecInfer (below).
- [SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification](https://arxiv.org/abs/2305.09781) — Miao et al. (arXiv v4; proof in v3). *Read*: Theorem 4.2 statement (multi-step speculative sampling equals the LLM's sampling distribution) and its proof in Appendix A.3 of [arXiv v3](https://arxiv.org/abs/2305.09781v3). *Scope*: the cited proof exists and covers SpecInfer's `VerifyStochastic`; not re-derived here and not evidence about any runtime's tree verifier (CLM-021).
- [Triton documentation](https://triton-lang.org/main/index.html) — project landing page and the vector-addition tutorial, unversioned `main` docs as served on 2026-10-01. *Scope*: the program-instance, masked load/store, and `constexpr` block-size model in Lesson 5.2 (CLM-013). The docs are not pinned to a release.
- [PyTorch compiler FAQ](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_faq.html) — `main` docs as served on 2026-10-01. *Read*: the sections on how compilation speeds up code (vertical and horizontal fusion), graph breaks, and guard-triggered recompilation. *Scope*: the fusion vocabulary and failure conditions in Lesson 5.2 (CLM-003). The `torch.compile` API page cited in the registry was not re-read in this pass.
- Triton and compiler fusion are tool/mechanism families. No claim is made here that either is a default in any runtime; an implementation is selected only after correctness and workload-specific measurement.

**FRONTIER**

- [FlashAttention-3](https://arxiv.org/abs/2407.08608) — Shah et al. (arXiv v2, July 2024). *Read*: abstract only. *Scope*: a Hopper-targeted design using asynchrony and FP8 (CLM-002).
- [EAGLE-3](https://arxiv.org/abs/2503.01840) — Li et al. (arXiv v3, April 2025). *Read*: abstract, contribution list, and Section 4.3. *Scope*: direct token prediction with multi-layer feature fusion; results at batch size 1 and in a batched serving framework; the paper's remark that spare compute shrinks with batch size (CLM-016).
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) — Zadouri et al. (arXiv v1, 5 March 2026). *Read*: abstract, contribution list, and Section 3.1.4 (which restates the online-softmax state). *Scope*: a Blackwell-targeted algorithm/kernel co-design (CLM-002). It is a preprint about one hardware generation, not evidence of adoption or of portable gains.

**PRODUCTION SOURCE TRACE**

- Repository: `Dao-AILab/flash-attention`
- Revision: `e9cf2c1651d2303191eb40a739a3c135fda00999` — [flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/flash_attn_interface.py)
- Verified: 2026-09-26 by static inspection; the same revision, file, and symbols were statically re-inspected on 2026-10-01. Not executed here.
- File: `flash_attn/flash_attn_interface.py`
- Symbols: `flash_attn_func`, `flash_attn_varlen_func`, `flash_attn_with_kvcache`, `FlashAttnFunc.forward`, `_flash_attn_forward`.
- Trace: `flash_attn_func` → `FlashAttnFunc.apply` → `FlashAttnFunc.forward` → `_flash_attn_forward` → `flash_attn_gpu.fwd`; `flash_attn_varlen_func` → `FlashAttnVarlenFunc.apply` → `flash_attn_gpu.varlen_fwd`; `flash_attn_with_kvcache` → `flash_attn_gpu.fwd_kvcache`. `flash_attn_gpu` is the compiled `flash_attn_2_cuda` module or a Triton ROCm implementation, selected at import.
- Scope: this proves behavior of the pinned upstream interface, not every wheel, fork, backend, or serving runtime. It is one repository snapshot, not an industry default.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow:

$$
\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}
\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}.
$$

### LAB A — Attention Parity and Shape-Regime Map

- **Objective**: Compare a trusted reference attention path with an available IO-aware path over a pre-registered shape matrix.
- **Pre-Registered Hypothesis**: IO-aware attention will reduce measured intermediate traffic for supported large shapes, while dispatch or underfill can erase the latency benefit on other cells.
- **Independent Variables**: Device, dtype, batch, query/key lengths, head dimension, causal/local mask, MHA/GQA ratio, and packed/variable-length representation.
- **Dependent Variables**: Correctness tolerance, NaN/Inf, warm kernel time, cold dispatch/compile time, peak memory, achieved bandwidth/FLOPs where reliable, and end-to-end attention share.
- **Break & Falsify**: Include tiny queries, unsupported or fallback-prone head sizes, ragged sequences, extreme logits, and shapes where attention is a small end-to-end fraction.
- **Required Artifact**: Pinned environment, raw shape-level results, profiler traces for at least three regimes, a reference online-softmax implementation that carries $(m,\ell,\text{accumulator})$ and reproduces the Lesson 5.1 worked example, and the Section 09 source trace of the pinned interface (attached, not redone here).
- **Alignment**: Lesson 5.1.
- **Effort Estimate**: 4h (the 2h source trace is counted separately).

### LAB B — Fusion and Triton Resource Trade-Off

- **Objective**: Implement one fused inference kernel and compare it with eager and compiler-generated strong baselines.
- **Pre-Registered Hypothesis**: Fusion will help where removed launch/intermediate traffic exceeds added compilation and resource costs, and regress at least one adversarial shape.
- **Independent Variables**: Block size, warps/stages where applicable, width, batch, aligned/misaligned dimensions, contiguous/strided layout, and cold/warm execution.
- **Dependent Variables**: Numerical error, launch count, compile/recompile events, kernel/end-to-end time, traffic, registers/shared memory, occupancy, and spills.
- **Break & Falsify**: Find a shape where fusion regresses and determine whether launch, compilation, fallback, memory access, or resource pressure explains it.
- **Required Artifact**: Kernel, tests, generated-code excerpt or trace, benchmark protocol, uncertainty, and rollback criterion.
- **Alignment**: Lesson 5.2.
- **Effort Estimate**: 4h.

### LAB C — Quantized Artifact: Memory, Quality, and Kernel Reality

- **Objective**: Produce at least two quantized configurations that differ in method or granularity and evaluate them against the same baseline.
- **Pre-Registered Hypothesis**: Nominal bit reduction will overpredict at least one of measured memory or serving improvement after metadata, kernels, and quality constraints are included.
- **Independent Variables**: Method, bit width, granularity, calibration data, batch/load, prompt/output regime, and supported/fallback shape.
- **Dependent Variables**: File/tensor/metadata bytes, peak/steady memory, selected kernels, conversion time, numerical error, application quality, TTFT, TPOT, throughput, and SLO-goodput.
- **Break & Falsify**: Evaluate calibration shift, outlier prompts, small/large batches, short decode/long prefill, and a fallback shape.
- **Required Artifact**: Complete quantization manifest, calibration provenance, quality uncertainty, profiler trace, and compatibility table.
- **Alignment**: Lessons 5.3–5.4.
- **Effort Estimate**: 4h.

### LAB D — Speculative Decoding Reward/Cost Surface

- **Objective**: Implement or instrument target-only and one speculative path, then measure full cycle reward/cost across proposal lengths and workloads.
- **Pre-Registered Hypothesis**: Speculation will improve the declared objective only where committed reward amortizes draft, verification, bookkeeping, and memory/concurrency cost.
- **Independent Variables**: Proposal length, draft family/size, prompt domain, sampling, output length, batch/concurrency, and scheduler load.
- **Dependent Variables**: Proposed/committed tokens, first rejection, component time, target calls, memory, client token-event timing, throughput, and goodput.
- **Break & Falsify**: Use a low-acceptance domain, expensive draft, and high target batching; locate where speculation loses using the full ledger. Replace the residual with the unmodified target distribution in the toy model and show that the enumeration check fails.
- **Required Artifact**:
  1. A toy reference implementation of the exact rule on a three-token vocabulary with fixed target and draft tables for two positions. Enumerate every draft path exactly (no sampling) and show that the joint distribution of the first two emitted tokens equals the target's $p_1(x_1)\,p_2(x_2\mid x_1)$. A sampled frequency test alone is not sufficient.
  2. Cycle records with proposed tokens, accepted drafts, committed tokens, and component times.
  3. $\sum T/\sum A$ and the mean of $T/A$ on the same records, with the difference explained.
  4. The distribution contract of the variant actually run (exact, greedy-equivalent, approximate, or unverified) and the matching equivalence or quality check.
  5. A measured enable/disable policy by workload stratum.
- **Alignment**: Lessons 5.5–5.7.
- **Effort Estimate**: 4h.

## 07 Break / Incident Scenarios

### Incident 05.1: The “Optimized” Release Loses Goodput

A release simultaneously enables weight-only INT4, compiler fusion, and speculative decoding. Offline model memory falls and a decode microbenchmark improves, yet loaded production shows higher P99 TTFT, intermittent ITL stalls, more out-of-memory restarts, and lower SLO-goodput. Some prompt domains also show a small but disputed quality change.

The incident deliberately does not identify one root cause. The learner must:

1. **Form competing hypotheses:** unsupported quantized shapes causing dequantization; fusion recompilation or register pressure; speculative draft/verification overhead; extra draft/workspace/KV memory reducing concurrency; calibration drift; scheduler interaction; or an unrelated traffic/runtime change.
2. **Identify missing evidence:** artifact manifests, selected-kernel logs, cold/warm traces, graph breaks/recompiles, generated kernels, GPU counters, memory breakdown, cycle-level acceptance/cost, prompt-domain labels, request queues, and matched quality results.
3. **Design discriminating experiments:** reproduce a pinned workload and use a factorial toggle matrix for INT4, fusion, and speculation. Preserve model/runtime/configuration except the declared factor and include interaction terms.
4. **Rank explanations:** use lead-lag timing and ablation effect sizes. Do not infer causality from occupancy, acceptance, or allocation alone.
5. **Intervene:** roll back or gate only the unsupported mechanism, constrain shapes/domains, or reserve memory based on evidence.
6. **Separate mitigation and prevention:** define immediate rollback/gating independently from the long-term artifact, capacity, or release-process correction.
7. **Remeasure:** repeat the same correctness, quality, latency, throughput, memory, and goodput protocol; define pass/fail and automatic rollback thresholds before redeployment.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Defend an Optimization Portfolio

You own one model service with two declared workload classes: short interactive conversations and long-prompt offline generation. A hardware refresh provides a newer GPU generation, and stakeholders propose enabling the newest attention kernel, INT4 weights, graph fusion, and learned speculative drafting simultaneously.

Produce an evidence-backed optimization plan with:

1. a pinned baseline and workload distribution, including prompt/output correlations, arrival or concurrency model, batch shapes, and SLO-goodput definition;
2. a profile-based bottleneck model for each workload class;
3. attention, fusion, quantization, and speculation candidates with explicit applicability and failure conditions;
4. derivations for nominal weight payload and speculative cycle throughput with units and assumptions, with time per committed token computed as total cycle time over total committed tokens;
5. a correctness/numerical/task-quality gate appropriate to each mechanism, including the acceptance rule of the chosen speculative variant and the distribution contract it does or does not carry;
6. a main-effect and interaction experiment design with warm/cold and isolated/loaded boundaries;
7. source inspection of one production kernel/interface at an exact revision;
8. a decision table that accepts, rejects, or gates each optimization by shape/domain/load;
9. deployment canary, observability, rollback thresholds, and a falsifying workload shift;
10. a defense explaining why stacking every locally favorable optimization is not an engineering conclusion.

No supplied speedup is accepted as a substitute for measurement. If the available evidence cannot support a deployment claim, record a bounded open verification item with the exact missing experiment.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

The reference trace uses FlashAttention commit `e9cf2c1651d2303191eb40a739a3c135fda00999`. The learner's trace must contain:

1. repository, exact revision, verification date, and file;
2. the public entry symbol and each symbol on the path to the backend call (for the reference: `flash_attn_func` → `FlashAttnFunc.forward` → `_flash_attn_forward` → `flash_attn_gpu.fwd`);
3. how the backend object is selected and under which configuration conditions;
4. one observed behavior with its condition (for example head-dimension padding, or the KV-cache path's lack of backward support);
5. static-versus-executed status for every statement, and whether each conclusion generalizes beyond the pinned revision.

### Reference Checks (arithmetic only; not a design answer)

Reviewers use these to check a submission's calculations. A submission with different declared inputs is checked against its own inputs.

- **Online softmax** (Lesson 5.1 inputs): final $m=5$, $\ell=1.2034380$, unscaled accumulator $44.3834741$, output $36.88057$.
- **Exact speculation** (Lesson 5.5 inputs): $\beta=0.6$, residual $[0.75,0.25,0]$, emitted distribution $[0.5,0.3,0.2]$. Resampling from $p$ gives $[0.40,0.32,0.28]$ and must be flagged as wrong.
- **Cycle ratio** (Lesson 5.5 records): $\sum T/\sum A=20$ ms/token; mean of $T/A=27.5$ ms/token.
- **Payload** (Lesson 5.3 inputs): $4\times10^9$ B payload; $4.25\times10^9$ B with the stated per-group metadata.

### Rubric Dimensions

- **Mechanistic reasoning** (Mastery 2–3; LAB A, LAB D artifact 1)
  - *Insufficient*: names mechanisms without their state or rule, or applies the exact-speculation guarantee to a different acceptance rule.
  - *Competent*: distinguishes mathematical function, representation, kernel, runtime dispatch, and serving behavior; states the online-softmax state and the acceptance/residual rule correctly.
  - *Strong*: also identifies which assumption fails in a given variant or implementation and how a test would expose it.
- **Quantitative reasoning** (Mastery 4; Reference Checks)
  - *Insufficient*: converts a bit ratio or an acceptance rate into a performance claim, or averages per-cycle ratios.
  - *Competent*: supplies equations, units, assumptions, boundaries, and exclusions; uses total time over total committed tokens.
  - *Strong*: also shows the sensitivity of the conclusion to the unmeasured inputs, such as draft cost or metadata overhead.
- **Correctness and quality** (Mastery 5; LAB A, LAB C, LAB D artifacts 1 and 4)
  - *Insufficient*: reports speed without a parity, distribution, or quality gate.
  - *Competent*: tests numerical parity and application quality with declared tolerances, populations, decoding settings, and uncertainty; states the distribution contract of the speculative variant.
  - *Strong*: also includes a negative control that the gate is shown to catch.
- **Experimental rigor** (Mastery 1, 6; all labs)
  - *Insufficient*: unpinned revisions, cold and warm runs mixed, or one favorable shape.
  - *Competent*: pins revisions and configuration; warms and synchronizes correctly; uses representative shape and workload distributions; reports raw results and variability.
  - *Strong*: also measures main effects before combinations and reports interaction terms at matched boundaries.
- **Failure diagnosis** (Incident 05.1 steps 1–7)
  - *Insufficient*: names one cause from a symptom.
  - *Competent*: ranks competing causes through discriminating measurements and controlled ablations.
  - *Strong*: allows interacting bottlenecks, states what evidence would overturn the ranking, and remeasures after intervention.
- **Source trace** (Mastery 7; Required Artifact items 1–5)
  - *Insufficient*: cites a repository or a comment without revision, symbol, or path.
  - *Competent*: all five items present and consistent with the pinned source.
  - *Strong*: also separates what static inspection cannot establish and lists the executed check that would.
- **Architecture defense** (Mastery 8–10)
  - *Insufficient*: recommends the full stack because each part is individually favorable.
  - *Competent*: accepts, rejects, or gates each mechanism by evidence and defines operational rollback.
  - *Strong*: also names the workload shift that would reverse each decision.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| IO-aware attention, online-softmax state, and parity | Lesson 5.1 | Lesson 5.1 Guided Practice (a); LAB A | Mastery 3, 5; Reference Check "Online softmax" | Reference online-softmax implementation, shape map, parity tests; rubric: Mechanistic reasoning, Correctness and quality |
| Fusion and Triton kernel reasoning | Lesson 5.2 | Lesson 5.2 Guided Practice; LAB B | Incident 05.1 steps 1–3 (fusion hypotheses and toggles); Mastery 3, 6 | Kernel, tests, generated-code excerpt, cold/warm profile; rubric: Experimental rigor |
| Quantization contract and methods | Lessons 5.3–5.4 | Lesson 5.3 Guided Practice (a)–(b); LAB C | Incident 05.1 steps 1–3 (quantization hypotheses); Mastery 3, 4, 5; Reference Check "Payload" | Quantization manifest, calibration provenance, selected-operator log, quality report; rubric: Quantitative reasoning |
| Exact speculation: acceptance/residual rule and distribution contract | Lesson 5.5 (mechanism, Worked Example A) | Lesson 5.5 Guided Practice (a); LAB D artifact 1 | Mastery 5; Reference Check "Exact speculation" | Toy reference implementation with exact enumeration; rubric: Mechanistic reasoning, Correctness and quality |
| Speculative reward/cost modeling | Lesson 5.5 (cost model, Worked Example B); Lesson 5.6 | Lesson 5.5 Guided Practice (b)–(c); LAB D artifacts 2–5 | Mastery 4, 6, 8; Reference Check "Cycle ratio" | Cycle records, ratio calculations, enable/disable policy; rubric: Quantitative reasoning |
| Draft-family comparison | Lesson 5.6 | Lesson 5.6 Guided Practice | Mastery 3, 8 | Comparison table with a distribution contract per family |
| Production source trace | Lesson 5.1 (pinned interface path); Section 05 trace | LAB A (attaches the trace) | Mastery 7; Section 09 Required Artifact items 1–5 | Trace pinned to commit `e9cf2c16…`; rubric: Source trace |
| Optimization composition diagnosis | Lesson 5.7 | Lesson 5.7 Guided Practice (a)–(b); LAB D Break & Falsify | Incident 05.1 steps 1–7; Mastery 6, 8–10 | Factorial ablation table, ranked hypotheses, rollback plan; rubric: Failure diagnosis, Architecture defense |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can:

1. explain how IO-aware attention changes traffic while preserving dense-attention semantics within numerical tolerance, and carry the online-softmax state across tiles by hand;
2. write, test, profile, and break a fused GPU kernel;
3. specify a quantized artifact beyond its bit width and measure its real memory, quality, and executable kernel path;
4. distinguish SmoothQuant, GPTQ, and AWQ by mechanism and applicability;
5. state the exact acceptance/residual rule, verify by hand that it reproduces the target distribution on a small vocabulary, and say which variants do not carry that guarantee;
6. derive and measure speculative cycle reward/cost without treating acceptance rate as speedup;
7. compare draft families by training, quality, memory, batch, and operational trade-offs;
8. trace a production source path at a pinned revision;
9. diagnose an interacting optimization regression and defend a rollback with remeasurement.

### Module Wrap-Up (Final Mental Model Reconstruction)

Inference optimization is constrained resource transformation. It changes bytes, operations, launch structure, precision, or serial target calls, then hands a new workload to the rest of the system. Validate semantics and quality first; measure the transformed bottleneck at the end-to-end boundary; retain the optimization only where its full cost ledger improves the declared objective.

### Cost & Telemetry Hook

- [Lesson 22.2](../22-ai-economics/README.md): an optimization's cost ledger includes calibration, kernel build, draft-model training/memory, and rollback, not only per-token savings.
- [Lesson 22.6](../22-ai-economics/README.md): report the speedup flip point — batch size or acceptance rate below which a speculative or quantized path stops paying.
- [Lesson 23.1](../23-observability-reliability/README.md) and [Lesson 23.5](../23-observability-reliability/README.md): emit backend/kernel path, precision, and acceptance length per request; treat offline quality deltas as proxy quality until labeled traffic confirms them.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
