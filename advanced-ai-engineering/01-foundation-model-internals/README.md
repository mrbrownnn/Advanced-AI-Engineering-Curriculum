# Module 01 — Foundation Model Internals

## 00 Why This Module Exists

Serving, caching, profiling, quantization, and distributed placement all depend on model geometry. An engineer who cannot reconstruct the forward pass will miscount parameters, confuse logical FLOPs with latency, allocate the wrong KV shape, or optimize an implementation detail that the selected architecture does not have.

This module builds a mechanistic decoder-only model while refusing the common shortcut that every modern model is “basically Llama.” The pinned Llama implementation (pre-norm RMSNorm, RoPE, grouped K/V heads, gated MLP) is a useful reference, not a universal definition. Sparse experts and latent-attention families extend the model and change the accounting.

**Module Orientation**

- **Engineering Problem**: Translate a model configuration into correct tensor shapes, equations, parameter counts, leading FLOP terms, numerical behavior, next-token distributions, and downstream systems constraints.
- **What You Will Do**: Implement a minimal decoder layer, prove its shape invariants, turn logits into a documented sampling contract, test causal isolation and RoPE, compare MHA/MQA/GQA with MLA latent state, reconcile analytical counts with real parameters, profile deviations from the model, and trace a pinned Transformers forward and sampling path.
- **Environment**: Python 3.10+ and PyTorch; a GPU is optional for profiler studies but not for shape, accounting, sampling, or invariant tests.
- **Research Cutoff**: Evidence registry updated 2026-09-30 (WP-F1 review). Implementation claims remain pinned to the revisions in §05.

## 01 Baseline Assumptions

Prerequisites: linear algebra, matrix multiplication, the definition of the softmax function, basic PyTorch, and Module 00 claim discipline. Temperature, truncation, and sampling are taught here (Lesson 1.2), not assumed.

This module owns architecture semantics, the logits-to-token contract, and logical accounting. It cross-references but does not replace:

- KV-cache bytes, layout, lifecycle, paging, reuse, and eviction — Module 03 (this module supplies the retained-state geometry, including MLA's latent state in Lesson 1.7);
- GPU execution, memory hierarchy, and profiler reasoning — Module 02;
- kernel optimization, quantization, and speculative decoding — Module 05;
- test-time compute policies built on sampling (best-of-N, self-consistency) — Module 06;
- calibration and uncertainty interpretation of token probabilities — Module 07;
- tensor/pipeline/expert/context parallelism — Module 20.

**Prerequisite contract exported to later modules** (what downstream owners may assume was taught, and where):

| Contract item | Taught in | Scope limit |
|---|---|---|
| Autoregressive factorization over token IDs and sequence log-likelihood | Lesson 1.2 | Probability of a token-ID sequence under one tokenizer and context policy, not of a text string |
| Softmax with temperature; greedy, top-k, top-p; renormalization | Lesson 1.2 | Definitions and one pinned implementation; quality effects are empirical |
| Sampling contract fields (processor order, library defaults, seed, stop rules, context limit) | Lesson 1.2 | Reproducibility across batch composition or kernels is not guaranteed |
| MHA/MQA/GQA K/V head geometry | Lesson 1.3 | Head counts only; bytes and layout in Module 03 |
| MLA latent state: cached $c^{KV}$ plus decoupled RoPE key $k^R$, $(d_c+d_h^R)$ elements per token per layer | Lesson 1.7 | Paper equations and one released config; runtime cache layout is Module 03 / `TODO_VERIFY` |

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
  research_connection: REQUIRED

estimated_effort:
  instruction: 6h
  guided_practice: 3h
  labs: 12h
  assessment: 3h
  source_trace: 2h
  total: 26h
```

*Effort reconciliation*: lesson instruction sums to 360 min and lesson practice to 180 min. Labs are A 5h + B 3h + C 3h + D's 1h execution = 12h; LAB D's 2h source trace (which includes Lesson 1.8's trace practice) is counted once, under `source_trace`.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
token IDs [B,S]
  → embeddings [B,S,d]
  → repeated decoder blocks
       RMSNorm
       Q/K/V projections (or latent projections) + position transform
       causal attention + output projection
       residual add
       RMSNorm
       gated or expert MLP
       residual add
  → final norm
  → vocabulary projection
  → logits [B,S,V]
  → last-position logits [B,V]
  → logits processors (temperature, top-k, top-p, …)
  → greedy argmax or seeded multinomial draw
  → next token appended; stop rule checked; repeat
```

Each arrow has four views that must agree:

1. **Semantics**: what dependency or transformation is intended?
2. **Shape**: what are the axes, sizes, strides, and broadcast rules?
3. **Accounting**: which weights and operations are logically required?
4. **Implementation**: what source path, dtype, backend, workspace, and fusion actually realize it?

A mismatch between these views is a diagnostic signal, not something to paper over with a familiar asymptotic formula.

## 04 Lessons

### Lesson 1.1 — Decoder Block and Residual Stream

**Engineering Question:** How does information move from input tokens to logits through a pre-normalized decoder?

**Concepts & Definitions:** Token, sequence, feature, residual, normalization, attention, MLP, and vocabulary axes form the decoder's executable contract.

Let input token IDs have shape $[B,S]$, model width $d$, and vocabulary size $V$. Embedding lookup produces

$$X_0\in\mathbb{R}^{B\times S\times d}.$$

A Llama-style pre-normalized layer can be written as

$$U_l=X_l+\mathrm{Attention}(\mathrm{RMSNorm}(X_l)),$$
$$X_{l+1}=U_l+\mathrm{MLP}(\mathrm{RMSNorm}(U_l)).$$

After $L$ layers, a final normalization and language-model head produce logits:

$$Z=\mathrm{LMHead}(\mathrm{RMSNorm}(X_L))\in\mathbb{R}^{B\times S\times V}.$$

where:
- $B$ = batch size (independent sequences), $S$ = sequence length (positions), $d$ = model width (features per position), $V$ = vocabulary size, $L$ = number of decoder layers;
- $X_l\in\mathbb{R}^{B\times S\times d}$ = residual stream entering layer $l$ ($l=0,\dots,L-1$); $X_0$ is the embedding output and $X_L$ the output of the last layer;
- $U_l$ = residual stream after layer $l$'s attention branch has been added, before its MLP branch;
- $\mathrm{LMHead}$ = a linear map $d\to V$ applied independently at every position; $Z$ = logits (unnormalized scores, one per vocabulary entry).

Read each layer equation as "normalize a copy, transform it, add the result back": the branch input is $\mathrm{RMSNorm}(X_l)$, but the addition uses the un-normalized $X_l$. That is why the branch outputs must have exactly the shape of $X_l$, namely $[B,S,d]$.

The pinned Transformers Llama implementation follows this order (**O**, CLM-010). It is a reference architecture: other decoders may use post-norm, parallel attention/MLP branches, additional norms, convolution or state-space blocks, different residual scaling, or shared layers.

**Invariants**

- Both residual additions require identical logical shapes.
- The causal dependency of position $i$ must not change when only tokens at positions $>i$ change.
- Vocabulary projection and input embedding may be tied or independent; parameter counting must inspect configuration and actual tensors.

**Worked Example:**
- *Input*: $B=2$, $S=8$, $d=64$, $V=1000$, two layers.
- *Steps*: embedding $[2,8]\to[2,8,64]$; each attention and MLP branch returns $[2,8,64]$, so both residual adds are $[2,8,64]+[2,8,64]$; final norm keeps $[2,8,64]$; LM head maps the last axis $64\to1000$.
- *Result*: logits $[2,8,1000]$, i.e. one next-token score vector for every position of every sequence.
- *Interpretation / limits*: a `view` that reorders $[B,S,d]$ as $[B,d,S]$ preserves all $2\cdot8\cdot64=1024$ elements per add yet mixes features across positions. Element count is necessary, not sufficient; axis names must be checked.

**Knowledge Check:** Which shapes must match at residual additions, and which perturbation tests causal isolation?

**Guided Practice:** Install hooks around every layer of a small model. Record shape, dtype, device, contiguity, min/max, finite count, and residual norm. Compare the trace with the equations before profiling speed.

**Feedback Contract:**
- *Expected Evidence*: a forward graph with named axes at every arrow; both residual boundaries per layer; a future-token perturbation showing logits at positions $\le i$ unchanged within tolerance; the first layer where any recorded statistic diverges from the reference.
- *Common Failure*: treating “the output shape is $[B,S,V]$” as proof of a correct forward pass.
- *Diagnostic Hint*: change only the last token of the input. Which positions' logits are allowed to move?
- *Concept to Revisit*: residual-stream invariants and causal isolation.

**Learning Outcome:** Reconstruct and instrument a decoder block as an executable tensor contract.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 1.2 — Logits, Autoregressive Factorization, and the Sampling Contract

**Engineering Question:** How does a vector of logits become the next token, and which choices must be recorded so that a generation, a likelihood, or a sampling-based policy can be reproduced and compared?

**Concepts & Definitions:**

- **Autoregressive factorization**: for a token-ID sequence $x_{1:n}$ produced by one fixed tokenizer, the chain rule gives
  $$P(x_{1:n})=\prod_{i=1}^{n}P(x_i\mid x_{<i}),\qquad \log P(x_{1:n})=\sum_{i=1}^{n}\log P(x_i\mid x_{<i}).$$
  Here $x_i$ is the token ID at position $i$, $x_{1:n}=(x_1,\dots,x_n)$, and $x_{<i}=(x_1,\dots,x_{i-1})$ is everything before position $i$ (empty for $i=1$, i.e. only the context). *Step-by-step for $n=3$*: $P(x_1,x_2,x_3)=P(x_1)\,P(x_2\mid x_1)\,P(x_3\mid x_1,x_2)$ by repeated use of $P(a,b)=P(a)P(b\mid a)$; taking $\log$ turns the product into a sum, which is numerically safer because a product of many probabilities $<1$ underflows quickly.
  Decoder language models are trained and used to produce these conditionals left to right, and generation draws tokens one at a time from them (**O**, CLM-013). The chain rule itself is exact; the model's conditionals are learned approximations.
- **Scope of the factorization**: it assigns probability to a sequence of token IDs under a stated tokenizer, context (prompt, system text, truncation), and position policy. It is not the probability of a text string (one string can have several tokenizations), and it changes if the context is truncated at the context limit.
- **Logits and softmax**: the LM head's last-position output $z\in\mathbb{R}^V$ defines $p_j=\exp(z_j)/\sum_{k}\exp(z_k)$.
  Here $z_j$ is the logit of vocabulary entry $j$, $p_j$ its probability, and the sum runs over all $k=1,\dots,V$. Exponentiation makes every term positive; dividing by the sum makes the $p_j$ add to 1. Adding the same constant $c$ to every logit does not change $p$ (the factor $e^{c}$ cancels between numerator and denominator), which is why implementations subtract $\max_k z_k$ before exponentiating to avoid overflow.
- **Temperature** $T>0$: $p_j(T)=\exp(z_j/T)/\sum_k\exp(z_k/T)$ (**O**, CLM-013). $T<1$ sharpens, $T>1$ flattens. Dividing by $T$ preserves the order of logits, so the argmax does not change (**D**, CLM-014).
  *Step-by-step*: (1) the probability ratio of two tokens is $p_i(T)/p_j(T)=\exp\big((z_i-z_j)/T\big)$, because the shared denominator cancels; (2) for $z_i>z_j$, a smaller $T$ makes the exponent $(z_i-z_j)/T$ larger, so the gap between the more and less likely token widens (sharper), and a larger $T$ shrinks it toward $\exp(0)=1$ (flatter, approaching uniform as $T\to\infty$); (3) because $T>0$, $z_i>z_j \iff z_i/T>z_j/T$, so the ordering and hence $\arg\max$ are unchanged.
- **Greedy decoding**: choose $\arg\max_j z_j$. It is not “sampling at $T=0$”: $T=0$ divides by zero, and the pinned Transformers warper rejects it and points users to `do_sample=False` (**O**, CLM-015).
- **Top-k**: keep the $k$ highest-probability tokens, renormalize, sample. **Top-p (nucleus)**: keep the smallest high-probability set whose mass is at least $p$, renormalize, sample (**O**, CLM-013).
  *In symbols*: let $\mathcal{K}$ be the kept set. Top-k: $\mathcal{K}$ = the $k$ indices with largest $p_j$. Top-p: sort tokens so $p_{(1)}\ge p_{(2)}\ge\dots$, and take $\mathcal{K}=\{(1),\dots,(r)\}$ with the smallest $r$ such that $\sum_{i=1}^{r}p_{(i)}\ge p$. In both cases *renormalize* means
  $$\tilde p_j=\frac{p_j}{\sum_{i\in\mathcal{K}}p_i}\ \text{for } j\in\mathcal{K},\qquad \tilde p_j=0\ \text{otherwise},$$
  i.e. divide by the kept mass, not by the full-vocabulary sum. Implementations do this by setting removed logits to $-\infty$ (so $\exp(-\infty)=0$) and re-running softmax. Note the symbol clash: $p$ in "top-p" is a threshold, while $p_j$ is a token probability.
- **Sampling contract**: the full list of choices that determine the next-token distribution and the draw—logit source position and dtype, processor chain and order, $T$, $k$, $p$, other processors, library and checkpoint defaults, greedy versus sampled, RNG seed/generator/device, stop tokens and maximum new tokens, tokenizer/detokenizer versions, and context-limit/truncation policy.

**Mechanism Explanation:**

At Transformers commit `11c16613d93911300c38ec8c9c2460567be53281`, `GenerationMixin._sample` takes the last-position logits, copies them to float32, applies the logits-processor list, and then either draws with `torch.multinomial(softmax(scores))` when `do_sample=True` or takes `torch.argmax` otherwise (**O**, CLM-015). When sampling, `_get_logits_processor` appends temperature, then top-k, then top-p warpers when those fields are set away from their neutral values. The generation-config docstring states that unset fields are filled from `_get_default_generation_params()`, which lists `top_k: 50`; a sampled call that never mentions `top_k` may therefore still truncate (**O**, CLM-016). Confirm the effective processor list at runtime before claiming what a run used (`TODO_VERIFY` until executed).

Two further boundary facts at this revision: the top-k warper removes scores strictly below the $k$-th value, so ties at the boundary can keep more than $k$ tokens; the top-p warper always keeps at least `min_tokens_to_keep` tokens (**O**, CLM-015).

**Quantitative Model / Derivation:**

The forward pass that produces $z$ is independent of the sampling choice. For a dense decoder, projection and MLP work is roughly $2$ FLOPs per weight use per token under the FMA-as-two convention (**D**, CLM-007). Temperature adds $V$ divisions and the softmax adds $O(V)$ elementwise work to the same logits. Changing $T$, $k$, or $p$ changes the distribution and therefore which tokens are generated, which can change how many tokens are generated. It does not change the cost of one forward pass.

**Worked Example (synthetic logits):**
- *Input*: vocabulary $\{A,B,C,D\}$, last-position logits $z=[2,1,0,-1]$.
- *Step 1 — softmax, $T=1$*: $e^{z}=[7.3891,2.7183,1,0.3679]$, sum $11.4752$, so $p=[0.6439,0.2369,0.0871,0.0321]$.
- *Step 2 — temperature*: $T=0.5$ uses $z/T=[4,2,0,-2]$, giving $[0.8650,0.1171,0.0158,0.0021]$; $T=2$ gives $[0.4551,0.2760,0.1674,0.1015]$. The entropy is $0.455$, $0.948$, and $1.245$ nats for $T=0.5,1,2$. Greedy chooses $A$ at every $T$.
  - *Detail for $T=0.5$*: $e^{[4,2,0,-2]}=[54.598,7.389,1,0.1353]$, sum $63.1225$; e.g. $p_A=54.598/63.1225=0.8650$. Ratio check: $p_A/p_B=e^{(2-1)/T}$ is $e^{2}=7.39$ at $T=0.5$, $e^{1}=2.72$ at $T=1$, $e^{0.5}=1.65$ at $T=2$.
  - *Entropy*: $H(p)=-\sum_j p_j\ln p_j$ (nats because of $\ln$); $0$ means one token has all the mass, $\ln V=\ln4=1.386$ is the uniform maximum here. Higher $H$ = flatter distribution.
- *Step 3 — top-k=2 at $T=1$*: keep $\{A,B\}$ with mass $0.8808$; renormalized $[0.7311,0.2689]$ (i.e. $0.6439/0.8808$ and $0.2369/0.8808$).
- *Step 4 — top-p=0.9 at $T=1$*: cumulative mass from the top is $0.6439$, $0.8808$, $0.9679$. The smallest set reaching $0.9$ is $\{A,B,C\}$; renormalized $[0.6652,0.2447,0.0900]$. The pinned implementation reaches the same set from the other direction: sorted ascending, cumulative $[0.0321,0.1192,0.3561,1]$, and it removes only entries $\le 1-0.9=0.1$, which is $D$. (Why the two directions agree: a token's ascending cumulative mass is the total mass of that token and every token ranked below it, so removing entries with ascending cumulative $\le 1-p$ removes a tail of total mass $\le 1-p$ and leaves a top set of mass $\ge p$.)
- *Step 5 — likelihood*: if the next step's model probability of $B$ after $A$ is $0.5$ (a second synthetic value), then $\log P(A,B\mid\text{ctx})=\ln0.6439+\ln0.5=(-0.4402)+(-0.6931)=-1.1333$ nats (equivalently $P=0.6439\times0.5=0.3220$). Compute that score from unprocessed logits unless the evaluation explicitly defines a processed distribution.
- *Result*: the same logits yield different next-token distributions under different contracts, while greedy output and the forward-pass FLOPs are unchanged.
- *Interpretation / limits*: $A$ with probability $0.6439$ is not a calibrated statement about the world (Module 07). Whether $T=0.5$ or top-p$=0.9$ improves task quality is an empirical question for the target model and workload.

**Knowledge Check:**
1. Why does lowering $T$ from 1 to 0.5 change sampled outputs but not the FLOPs of the forward pass that produced $z$?
2. A run reports `do_sample=True, temperature=0.7` and nothing else. Which additional field could still be truncating the distribution at the pinned revision, and how would you prove what actually ran?
3. Why is $\sum_i \log P(x_i\mid x_{<i})$ not the probability of the *text* “New York” in general?

**Guided Practice:** For logits $z=[3,1,1,0]$ compute $p$ at $T=1$; the set kept by top-k=2 (watch the tie); the set kept by top-p=0.8; and the renormalized distribution in each case. Then write the complete sampling contract for this toy run.

**Feedback Contract:**
- *Expected Evidence*: $p\approx[0.7573,0.1025,0.1025,0.0377]$. For top-k=2, the second-largest value is tied: a strict “exactly $k$” rule must break the tie by a stated rule, while the pinned threshold rule `scores < kth_value` removes only index 3 and keeps three tokens, renormalized to $[0.7870,0.1065,0.1065]$. For top-p=0.8, the smallest set from the top is index 0 plus *one* tied token (mass $0.8598$, renormalized $[0.8808,0.1192]$). The ascending-cumulative implementation (cumulative $[0.0377,0.1402,0.2427,1]$, remove $\le0.2$) drops index 3 and whichever tied token sorts first, so the survivor depends on sort order—record it as implementation behavior. A contract lists processor order, defaults, seed, stop rule, and context policy.
- *Common Failure*: renormalizing over the whole vocabulary after truncation, applying top-p before temperature when the contract says the opposite, or recording only `temperature`.
- *Diagnostic Hint*: print the processor list and the post-processing scores for one step; count how many entries are $-\infty$.
- *Concept to Revisit*: temperature-scaled softmax, truncation sets, and implementation defaults.

**Learning Outcome:** Derive next-token distributions from logits under a stated sampling contract, separate distribution changes from forward-pass cost, and record every choice needed to reproduce a sampled or greedy generation.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 1.3 — Causal Attention and MHA/MQA/GQA Geometry

**Engineering Question:** Which axes are shared, and which remain per query head?

**Concepts & Definitions:** Query heads, KV heads, grouping, causal masks, and logical versus materialized repetition must be distinguished.

For hidden states $X\in\mathbb{R}^{B\times S\times d}$, define query heads $H_q$, key/value heads $H_{kv}$, and head width $d_h$, with $d=H_qd_h$ for the standard case:

$$Q\in\mathbb{R}^{B\times H_q\times S\times d_h},$$
$$K,V\in\mathbb{R}^{B\times H_{kv}\times S_k\times d_h}.$$

If $g=H_q/H_{kv}$ is an integer, query head $h$ uses K/V group $\lfloor h/g\rfloor$. The logical attention for a query head is (**O**, CLM-001)

$$A_h=\mathrm{softmax}\left(\frac{Q_hK_{group(h)}^\top}{\sqrt{d_h}}+M\right),$$
$$O_h=A_hV_{group(h)}.$$

where:
- $S$ = number of query positions, $S_k$ = number of key/value positions (equal to $S$ without a cache; larger when a cached prefix is attended to);
- $Q_h\in\mathbb{R}^{S\times d_h}$ = query head $h$ for one sequence; $K_{group(h)},V_{group(h)}\in\mathbb{R}^{S_k\times d_h}$ = the K/V head it reads, with $group(h)=\lfloor h/g\rfloor$;
- $M\in\mathbb{R}^{S\times S_k}$ = additive mask; $A_h\in\mathbb{R}^{S\times S_k}$ = attention weights; $O_h\in\mathbb{R}^{S\times d_h}$ = head output. Softmax runs over the key axis (each row of $A_h$ sums to 1).

*Step-by-step for one query position $i$*:
1. Scores: $s_{ij}=q_i\cdot k_j/\sqrt{d_h}$ for every key position $j$ ($q_i$ = row $i$ of $Q_h$, $k_j$ = row $j$ of $K_{group(h)}$). The $\sqrt{d_h}$ keeps score magnitudes from growing with head width: if the $d_h$ coordinates of $q$ and $k$ were independent with variance 1, $q\cdot k$ would have variance $d_h$, and dividing by $\sqrt{d_h}$ brings it back to 1.
2. Mask: add $M_{ij}$; future keys get $-\infty$.
3. Normalize: $a_{ij}=\exp(s_{ij}+M_{ij})/\sum_{j'}\exp(s_{ij'}+M_{ij'})$; masked entries become $\exp(-\infty)=0$.
4. Mix: $o_i=\sum_j a_{ij}v_j$, a weighted average of value rows.
5. Concatenate the $H_q$ outputs $O_h$ along features ($H_qd_h=d$) and apply the output projection $W_O$.

For a simple no-prefix causal mask,

$$M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i.\end{cases}$$

*Tiny example* ($S=3$, query row $i=1$ with 0-based positions, unmasked scores $[1,0,5]$): the mask turns them into $[1,0,-\infty]$, so $a_{1,\cdot}=[e^1,e^0,0]/(e^1+e^0)=[0.7311,0.2689,0]$. The large future score $5$ has no effect—exactly the property the future-token perturbation test checks. Row $i=0$ always gets $[1,0,0]$: the first position attends only to itself.

The mask is what makes the per-position logits of Lesson 1.2 conditionals on $x_{<i}$ only. With cached prefixes, padding, packed sequences, sliding windows, or special attention patterns, absolute positions and valid-key boundaries require a richer mask.

- **MHA**: $H_{kv}=H_q$.
- **MQA**: $H_{kv}=1$.
- **GQA**: $1<H_{kv}<H_q$ in the intermediate case.

Fewer KV heads reduce K/V projection width and logical KV payload. They do not remove query heads, output projection work, softmax, or attention over context (**O**, CLM-005). A runtime can map groups without physically repeating K/V; a naive `repeat` may create avoidable memory traffic.

**Worked Example:**
- *Input*: $H_q=32$, $H_{kv}=8$, $d_h=128$, $d=4096$.
- *Steps*: $g=32/8=4$; query heads 0–3 read KV head 0, heads 4–7 read KV head 1, …, heads 28–31 read KV head 7. K and V projections map $4096\to8\cdot128=1024$ each, instead of $4096\to4096$ for MHA.
- *Result*: K/V projection width and per-token K/V elements shrink by $H_{kv}/H_q=1/4$; Q, the 32 score matrices, and the output projection are unchanged.
  - *Per-token, per-layer K/V elements*: $2H_{kv}d_h$ (factor 2 for K and V). MHA: $2\cdot32\cdot128=8192$; GQA: $2\cdot8\cdot128=2048$; MQA: $2\cdot1\cdot128=256$. Ratio GQA/MHA $=2048/8192=1/4$.
  - *Group map check*: head $h=13$ uses group $\lfloor13/4\rfloor=3$ (heads 12–15 share KV head 3).
- *Interpretation / limits*: the mapping does not require physically repeating stored K/V. Whether decode latency falls depends on kernels, batch, and context (Lab B); quality depends on training (**O**, CLM-005).

**Knowledge Check:** Which dimensions shrink under GQA, and why does attention over prior positions remain?

**Guided Practice:** Flip the mask orientation or permute head and sequence axes while preserving the element count. A prefix-invariance test and reference comparison should fail.

**Feedback Contract:**
- *Expected Evidence*: named axes, the explicit head→group map, mask semantics including the diagonal, parity tolerance, and one corrupted variant that passes a shape-only smoke test but fails prefix invariance.
- *Common Failure*: mapping query head $h$ to group $h \bmod H_{kv}$ (interleaved) when the checkpoint expects contiguous blocks, or vice versa.
- *Diagnostic Hint*: compare per-head outputs against a reference that materializes the repeat; which heads diverge first?
- *Concept to Revisit*: head grouping and causal masking.

**Learning Outcome:** Derive MHA/MQA/GQA geometry and detect mask or grouping corruption.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 1.4 — Rotary Position Embedding

**Engineering Question:** How can a position transformation alter attention scores without adding a position vector to the residual stream?

**Concepts & Definitions:** RoPE rotates paired query/key features using position-dependent angles before attention scoring.

For each paired feature coordinate, RoPE applies a rotation at position $p$:

$$R(p\theta)=\begin{bmatrix}\cos(p\theta)&-\sin(p\theta)\\\sin(p\theta)&\cos(p\theta)\end{bmatrix}.$$

where $p$ = token position (here the query's; $r$ below is the key's), $\theta$ = the rotation frequency assigned to this coordinate pair, and $q,k\in\mathbb{R}^2$ = one pair of query/key features. A fully rotated $d_h$-wide head has $d_h/2$ such pairs, each with its own $\theta_i$ given by the model's frequency schedule (read it from the actual config and source, not from memory).

Queries and keys are rotated before their dot product. Orthogonality gives

$$\left(R(p\theta)q\right)^\top\left(R(r\theta)k\right)=q^\top R((r-p)\theta)k,$$

*Step-by-step*:
1. Transpose of a product: $\left(R(p\theta)q\right)^\top=q^\top R(p\theta)^\top$.
2. Rotations are orthogonal, and the inverse of rotating by $p\theta$ is rotating by $-p\theta$: $R(p\theta)^\top=R(-p\theta)$.
3. Rotations about the same origin compose by adding angles: $R(-p\theta)R(r\theta)=R((r-p)\theta)$.
4. So the score is $q^\top R((r-p)\theta)k$: absolute positions $p$ and $r$ enter only through $r-p$. Summing over all $d_h/2$ pairs keeps this property, since each pair contributes its own term.

so the score can depend on relative displacement (**O**, CLM-003). Real models choose a frequency schedule, rotary fraction, coordinate pairing convention, maximum training positions, and possibly a scaling method.

**Do not infer:** “RoPE supports unlimited context.” Mathematical rotations can be evaluated at new positions, but quality and numerical behavior beyond training length are empirical properties of the trained model and scaling scheme.

**Worked Example:**
- *Input*: one 2-D pair, $q=k=(1,0)$, $\theta=\pi/6$; positions $(p,r)=(3,5)$ and then $(13,15)$.
- *Steps*: $q^\top R((r-p)\theta)k=\cos(2\cdot\pi/6)=\cos(\pi/3)$ in both cases; $\|R(p\theta)q\|=1$.
  - *Direct check for $(p,r)=(3,5)$*: $R(3\theta)q=(\cos\frac{\pi}{2},\sin\frac{\pi}{2})=(0,1)$ and $R(5\theta)k=(\cos\frac{5\pi}{6},\sin\frac{5\pi}{6})=(-0.8660,0.5)$; dot product $0\cdot(-0.8660)+1\cdot0.5=0.5$. For $(13,15)$ the angles are $13\pi/6$ and $15\pi/6$, which differ by the same $\pi/3$, giving again $0.5$.
- *Result*: score $0.5$ for both pairs; norms preserved.
- *Interpretation / limits*: the identity shows relative-position dependence for one frequency. It says nothing about quality at positions never seen in training, and at large $p$ low-precision angle computation can break the identity numerically.

**Knowledge Check:** Why does norm preservation not prove extrapolation quality, and which pairing convention can break parity while preserving shape?

**Guided Practice:** Test norm preservation of rotated vectors, relative-shift behavior, dtype sensitivity at large positions, and parity with the pinned reference. Include an intentionally wrong pairing convention.

**Feedback Contract:**
- *Expected Evidence*: frequency schedule, coordinate pairing (adjacent versus half-split), position IDs, dtype/tolerance, tested position range, and the shift-invariance error as a function of position.
- *Common Failure*: pairing features $(2i,2i{+}1)$ when the reference rotates $(i,i{+}d_h/2)$—shapes and norms stay correct while scores differ.
- *Diagnostic Hint*: rotate a one-hot vector and compare which coordinate receives the sine term.
- *Concept to Revisit*: rotary pairing conventions and relative-position identity.

**Learning Outcome:** Test RoPE mechanics without converting a mathematical identity into a quality guarantee.

*(Effort: 40m instruction, 25m practice)*

---

### Lesson 1.5 — RMSNorm, SwiGLU, and Numerical Details

**Engineering Question:** Which normalization, gating, accumulation, and casting contracts must match for numerical parity?

**Concepts & Definitions:** RMS reduction axis, epsilon placement, accumulation dtype, gate/up branches, and residual order are separate correctness conditions.

**RMSNorm** over the final feature axis:

$$\mathrm{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}.$$

where $x\in\mathbb{R}^d$ = the feature vector at one (batch, position), $x_j$ its $j$-th feature, $\gamma\in\mathbb{R}^d$ = learned per-feature scale (the norm "weight"), $\epsilon>0$ = small constant added *inside* the square root so the denominator is never 0, and $\odot$ = elementwise product. *Step-by-step*: (1) square each feature; (2) average over the $d$ features (mean square); (3) add $\epsilon$; (4) take the square root, giving the RMS; (5) divide every feature by it, so the result has RMS $\approx1$; (6) multiply by $\gamma$.

Unlike LayerNorm, the expression does not subtract the feature mean (**O**, CLM-002). The pinned Transformers Llama implementation converts hidden states to float32 for the mean-square and reciprocal-square-root calculation, casts normalized values back to the input dtype, and multiplies by the learned weight (**O**, CLM-011). A substituted fused kernel may realize equivalent semantics through a different execution path.

**SwiGLU** with intermediate width $m$ (**O**, CLM-004):

$$\mathrm{SwiGLU}(x)=W_{down}\left(\mathrm{SiLU}(W_{gate}x)\odot(W_{up}x)\right).$$

where:
- $x\in\mathbb{R}^d$ = normalized input at one position; $m$ = intermediate (hidden) MLP width;
- $W_{gate},W_{up}\in\mathbb{R}^{m\times d}$ map $d\to m$; $W_{down}\in\mathbb{R}^{d\times m}$ maps $m\to d$;
- $\mathrm{SiLU}(u)=u\cdot\sigma(u)=u/(1+e^{-u})$, applied elementwise; $\odot$ = elementwise product.

*Shape walk*: $x\,[d]\to$ gate $[m]$ and up $[m]\to$ $\mathrm{SiLU}(\text{gate})\odot\text{up}$ $[m]\to$ down $[d]$. The gate branch passes through SiLU, the up branch does not, so swapping the two weight matrices changes the output. Example: gate pre-activation $2$, up pre-activation $3$ gives $\mathrm{SiLU}(2)\cdot3=1.7616\cdot3=5.285$; swapped gives $\mathrm{SiLU}(3)\cdot2=2.8577\cdot2=5.715$.

Bias and orientation notation depend on the framework. In row-major PyTorch notation, `nn.Linear(d,m)` stores a weight shaped $[m,d]$; the scalar parameter count is still $dm$.

**Failure modes**

- applying epsilon outside the square root;
- reducing across the wrong axis;
- computing low-precision squares that overflow or underflow;
- swapping gate/up branches when the activation is not symmetric;
- applying normalization after rather than before the sublayer;
- comparing outputs without aligning dtype and tolerance.

**Worked Example:**
- *Input*: $x=(3,4)$, $\gamma=(1,1)$, $\epsilon=10^{-6}$, reduction over the feature axis $d=2$.
- *Steps*: mean square $=(9+16)/2=12.5$; $\sqrt{12.5+10^{-6}}\approx3.5355$; divide.
- *Result*: $(0.8485,1.1314)$, whose RMS is 1 (i.e. $3/3.5355$ and $4/3.5355$; check $\sqrt{(0.8485^2+1.1314^2)/2}=\sqrt{(0.72+1.28)/2}=1$).
- *Interpretation / limits*: on $[B,S,d]$, reducing over $S$ instead of $d$ still broadcasts and returns the same shape, but each output now depends on other positions, which also breaks causal isolation. In FP16, $x_j^2$ overflows above $|x_j|\approx 256$ (since $256^2=65536$ exceeds the FP16 maximum 65504), which is one reason the pinned path squares in float32.

**Knowledge Check:** Where is epsilon applied, which axis is reduced, and why can accumulation dtype matter?

**Guided Practice:** Inject a wrong reduction axis, epsilon placement, and low-precision square path into transparent references.

**Feedback Contract:**
- *Expected Evidence*: intermediate mean-square statistics per position, cast boundaries, finite counts, tolerance justification, and the first mismatching operation for each injected defect.
- *Common Failure*: declaring parity from final logits only, which hides a defect that a later norm partly compensates.
- *Diagnostic Hint*: compare the mean square at one position against a hand calculation before looking at the output.
- *Concept to Revisit*: reduction axes and accumulation dtype.

**Learning Outcome:** Implement and diagnose normalization and gated-MLP numerical contracts.

*(Effort: 40m instruction, 25m practice)*

---

### Lesson 1.6 — Parameter and FLOP Accounting

**Engineering Question:** Which parts are exact logical counts, and which parts are performance hypotheses?

**Concepts & Definitions:** Logical parameter, FLOP, activation, and payload counts are scoped derivations; latency and allocation require measurement.

Assume dense, bias-free projections, $d=H_qd_h$, and no padding or adapters.

**Attention projection parameters** (**D**, CLM-006)

$$P_Q=d(H_qd_h)=d^2,$$
$$P_K=P_V=d(H_{kv}d_h),\qquad P_O=d^2,$$
$$P_{attn}=2d^2+2dH_{kv}d_h=2d^2\left(1+\frac{H_{kv}}{H_q}\right).$$

*Step-by-step*: a bias-free linear map from $a$ inputs to $b$ outputs has $ab$ weights. $Q$ maps $d\to H_qd_h=d$, so $d^2$; $K$ and $V$ each map $d\to H_{kv}d_h$, so $dH_{kv}d_h$ each; $O$ maps $H_qd_h=d\to d$, so $d^2$. Sum: $2d^2+2dH_{kv}d_h$. To factor, substitute $d_h=d/H_q$: $2dH_{kv}d_h=2d^2\,H_{kv}/H_q$, hence $2d^2(1+H_{kv}/H_q)$. Checks: MHA ($H_{kv}=H_q$) gives $4d^2$; MQA ($H_{kv}=1$) gives $2d^2(1+1/H_q)$.

**SwiGLU parameters** (**D**, CLM-019)

$$P_{mlp}=dm+dm+md=3dm.$$

The three terms are $W_{gate}$ ($d\to m$), $W_{up}$ ($d\to m$), and $W_{down}$ ($m\to d$), each with $dm$ weights.

Two RMSNorm weights contribute $2d$ per standard block. Add embeddings, final norm, LM head, biases, adapters, expert/router tensors, and any untied output head separately.

**FLOP convention**

State whether one fused multiply-add counts as one or two operations. This module uses two (**D**, CLM-007):

$$\mathrm{GEMM}(m\times k,k\times n)\approx2mkn\text{ FLOPs}.$$

Here $m,k,n$ are generic matrix dimensions (this $m$ is not the MLP width): the output has $mn$ entries, each a length-$k$ dot product of $k$ multiplies and $k$ adds, so $2k$ FLOPs per entry and $2mkn$ in total. Corollary: a token passing through a $d\to b$ linear layer is a $(1\times d)(d\times b)$ GEMM, i.e. $2db$ FLOPs $=2\times$ the layer's weight count. This is the source of "2 FLOPs per weight per token".

For a dense full-sequence attention calculation, QK and AV together have a nominal leading term

$$4BH_qS^2d_h=4BS^2d.$$

*Step-by-step*: per sequence and head, $QK^\top$ is $(S\times d_h)(d_h\times S)$, costing $2S\,d_h\,S=2S^2d_h$; $AV$ is $(S\times S)(S\times d_h)$, again $2S^2d_h$. Together $4S^2d_h$; times $B$ sequences and $H_q$ heads gives $4BH_qS^2d_h$, and $H_qd_h=d$ gives $4BS^2d$. Softmax and masking are $O(BH_qS^2)$ elementwise work, lower order by a factor $d_h$.

For one decode token attending to $S_k$ retained positions, the corresponding term is $4H_qS_kd_h=4dS_k$ per layer (same derivation with one query row: $(1\times d_h)(d_h\times S_k)$ plus $(1\times S_k)(S_k\times d_h)$, i.e. $4S_kd_h$ per head). It uses $H_q$, not $H_{kv}$: every query head still scores every retained position. A causal-aware implementation may avoid part of the nominal square, and a fused kernel may never materialize the score matrix. Projections and MLP scale linearly with $S$ but can dominate at common dimensions. Logical FLOPs do not determine latency without shapes, backend, precision, achieved compute/bandwidth, and memory traffic.

**Worked Example:**
- *Input*: $d=4096$, $H_q=32$, $H_{kv}=8$, $d_h=128$, $m=11008$, one decode token with $S_k=4096$ retained positions.
- *Steps (parameters)*: attention $2\cdot4096^2\cdot(1+8/32)=41{,}943{,}040$; SwiGLU $3\cdot4096\cdot11008=135{,}266{,}304$; two norms $8{,}192$.
  - *Attention, term by term*: $Q$ and $O$: $4096^2=16{,}777{,}216$ each; $K$ and $V$: $4096\cdot(8\cdot128)=4{,}194{,}304$ each; sum $2\cdot16{,}777{,}216+2\cdot4{,}194{,}304=41{,}943{,}040$, matching the factored form.
  - *Subtotal*: $41{,}943{,}040+135{,}266{,}304+8{,}192=177{,}217{,}536$.
- *Result (parameters)*: block subtotal $177{,}217{,}536$, excluding embeddings, final norm, LM head, biases, storage padding, and all runtime memory.
- *Steps (FLOPs per decode token per layer)*: projections and MLP $2\times(41{,}943{,}040+135{,}266{,}304)=354{,}418{,}688$; attention QK+AV $4\cdot4096\cdot4096=67{,}108{,}864$.
  - The first uses "2 FLOPs per weight" (norm weights excluded as negligible elementwise work); the second is $4dS_k$ with $d=S_k=4096$, equivalently $4H_qS_kd_h=4\cdot32\cdot4096\cdot128$.
  - *Crossover*: the two terms are equal when $4dS_k=354{,}418{,}688$, i.e. $S_k=354{,}418{,}688/16{,}384=21{,}632$ retained positions (under these logical counts only).
- *Interpretation / limits*: at this context the weight-proportional term is about $5.3\times$ the attention term ($354{,}418{,}688/67{,}108{,}864=5.28$), but the attention term grows with $S_k$ while the weight term does not. Neither number predicts decode latency, which also depends on bytes moved per token (Module 02).

**OOM discipline:** Logical tensors provide component estimates, not an exact OOM boundary. Measure allocated and reserved peaks across fresh processes and record workspaces, backend, cache state, graph capture, padding, and concurrent allocations (**H**, CLM-012).

**Knowledge Check:** Which assumptions yield $3dm$ for SwiGLU, and why do fewer FLOPs not guarantee lower latency?

**Guided Practice:** Reconcile the analytical manifest with named unique tensor storage, profiler shapes, and peak memory after toggling tying or bias.

**Feedback Contract:**
- *Expected Evidence*: a manifest row per parameter group with formula, assumptions, analytical count, `named_parameters` count, and delta with a named cause (tying, bias, padding, adapter, expert); the FMA convention stated beside every FLOP number.
- *Common Failure*: double-counting a tied embedding/LM head, or quoting “2× parameters FLOPs per token” while silently dropping the context-dependent attention term.
- *Diagnostic Hint*: sum `p.numel()` over unique storage pointers and compare with the sum over names.
- *Concept to Revisit*: scoped logical accounting versus measured execution.

**Learning Outcome:** Derive scoped architecture counts and reconcile them with execution evidence.

*(Effort: 55m instruction, 30m practice)*

---

### Lesson 1.7 — Sparse Experts and Latent Attention

**Engineering Question:** How do sparse experts and latent attention change total, resident, selected, executed, and retained quantities?

**Concepts & Definitions:** Total expert parameters, selected experts, capacity/padding, routing locality, latent state, and up-projection/absorption are distinct accounting dimensions.

**Sparse MoE** replaces a dense MLP with a router and multiple expert MLPs (**O**, CLM-008). For $E$ bias-free SwiGLU experts of width $m$, a simplified total expert count is

$$P_{experts,total}=3Edm.$$

This is the Lesson 1.6 SwiGLU count $3dm$ repeated for $E$ experts. The router (a $d\to E$ scoring map) and any shared experts are counted separately. Here $k$ means the number of experts selected per token, not the top-k sampling parameter of Lesson 1.2.

If top-$k$ experts execute per token, the raw selected expert weight use is proportional to $3kdm$, plus router and shared components. This is not a wall-clock model. Expert imbalance, capacity, padding, token dropping/rerouting, locality, batching, and communication matter.

Always distinguish:

- total parameters;
- resident parameters per device/process;
- parameters selected per token;
- operations actually executed after padding/capacity decisions.

**Multi-head Latent Attention (MLA)** as defined in DeepSeek-V2 (**O**, CLM-017). For attention input $h_t\in\mathbb{R}^d$, $n_h$ heads of width $d_h$:

$$c_t^{KV}=W^{DKV}h_t\in\mathbb{R}^{d_c},\qquad k_t^{C}=W^{UK}c_t^{KV},\qquad v_t^{C}=W^{UV}c_t^{KV},$$
$$k_t^{R}=\mathrm{RoPE}(W^{KR}h_t)\in\mathbb{R}^{d_h^R}\ \text{(one key shared by all heads)},$$
$$q_{t,i}=[q_{t,i}^C;q_{t,i}^R],\qquad k_{t,i}=[k_{t,i}^C;k_t^R],\qquad \text{score scale } 1/\sqrt{d_h+d_h^R}.$$

where:
- $t$ = token position, $i$ = head index ($i=1,\dots,n_h$); $d$ = model width;
- $d_c$ = KV latent width (much smaller than $n_hd_h$); $c_t^{KV}$ = joint compressed key/value latent;
- $W^{DKV}\in\mathbb{R}^{d_c\times d}$ = down-projection; $W^{UK},W^{UV}\in\mathbb{R}^{n_hd_h\times d_c}$ = up-projections, whose outputs $k_t^C,v_t^C\in\mathbb{R}^{n_hd_h}$ are split into per-head $k_{t,i}^C,v_{t,i}^C\in\mathbb{R}^{d_h}$ ("C" = content/compressed part);
- $d_h^R$ = width of the decoupled RoPE part; $W^{KR}\in\mathbb{R}^{d_h^R\times d}$; $q_{t,i}^R\in\mathbb{R}^{d_h^R}$ = per-head RoPE query part;
- $[a;b]$ = concatenation, so each head's query and key have width $d_h+d_h^R$, which is why the score scale uses $\sqrt{d_h+d_h^R}$ instead of $\sqrt{d_h}$.

Queries are also produced through a latent $c_t^Q\in\mathbb{R}^{d_c'}$, but the paper states this reduces training activation memory, not KV cache.

**State model — what must be retained per token per layer:** $c_t^{KV}$ ($d_c$ elements) and $k_t^R$ ($d_h^R$ elements). Per-head $k^C$ and $v^C$ are recomputable from $c^{KV}$. The paper states that at inference $W^{UK}$ can be absorbed into $W^Q$ and $W^{UV}$ into $W^O$, so per-head keys and values need not be materialized (**O**, CLM-017). The RoPE part is decoupled because a position-dependent rotation between $W^Q$ and $W^{UK}$ would block that absorption.

*Why absorption works (one head, content part only)*: let $W_i^{UK}\in\mathbb{R}^{d_h\times d_c}$ be head $i$'s rows of $W^{UK}$, so $k_{t,i}^C=W_i^{UK}c_t^{KV}$. For query position $s$ and key position $t$, by associativity
$$\big(q_{s,i}^C\big)^\top k_{t,i}^C=\big(q_{s,i}^C\big)^\top W_i^{UK}c_t^{KV}=\Big(\big(W_i^{UK}\big)^\top q_{s,i}^C\Big)^\top c_t^{KV}.$$
The bracketed vector depends only on the query side, so the map producing it can be folded into the query projection, and scores are taken directly against the cached $c_t^{KV}$. If RoPE were applied to $k_{t,i}^C$, a rotation $R_t$ that depends on the key position $t$ would sit between $W_i^{UK}$ and $c_t^{KV}$; it cannot be folded into a fixed query-side matrix, which is why the RoPE key $k_t^R$ is kept separately. The value side is analogous: $\sum_t a_{t}W_i^{UV}c_t^{KV}=W_i^{UV}\sum_t a_tc_t^{KV}$, so $W_i^{UV}$ can be applied after attention and merged into the output projection.

**Why MLA is not GQA with a smaller $H_{kv}$:** GQA retains full-width $K$ and $V$ for each of $H_{kv}$ groups ($2H_{kv}d_h$ elements per token per layer). MLA retains one joint latent plus one shared RoPE key, $(d_c+d_h^R)$ elements, and reconstructs per-head keys and values by learned up-projections, with every query head keeping its own reconstructed key/value. Configuration field names can mislead: the released DeepSeek-V2 config sets `num_key_value_heads: 128` alongside `kv_lora_rank: 512` and `qk_rope_head_dim: 64` (**O**, CLM-017).

**Worked Example (DeepSeek-V2 paper values and pinned config):**
- *Input*: $l=60$ layers, $n_h=128$, $d_h=128$, $d_c=512$ (`kv_lora_rank`), $d_h^R=64$ (`qk_rope_head_dim`).
- *Step 1 — MLA retained elements*: $(512+64)=576$ per token per layer; $\times60=34{,}560$ per token.
- *Step 2 — MHA reference with the same heads*: $2n_hd_hl=2\cdot128\cdot128\cdot60=1{,}966{,}080$ per token.
- *Step 3 — GQA-equivalent group count*: solve $2H_{kv}d_h=576$ for $H_{kv}$ with $d_h=128$: $576/(2\cdot128)=2.25$ groups, matching the paper's Table 1 statement (**D**, CLM-018).
- *Step 4 — the trap*: plugging `num_key_value_heads=128` into the GQA formula reproduces the MHA figure, a $1{,}966{,}080/34{,}560\approx56.9\times$ over-count of retained elements.
- *Result*: retained state is set by $d_c+d_h^R$, not by the configured KV-head count.
- *Interpretation / limits*: these are element counts from the paper's equations. Bytes per element, whether a runtime caches the latent before or after the paper's additional latent normalization, padding of 576 to alignment, and page layout belong to Module 03 and remain runtime-specific. The paper's reported 93.3% KV reduction and throughput gains are relative to its own DeepSeek 67B baseline and setup, not general multipliers (**O**, CLM-009).

**Currentness classification**

- **REFERENCE / BASELINE**: scaled causal MHA and dense decoder blocks (Vaswani et al.).
- **OBSERVED IN PINNED SOURCE, NOT A PREVALENCE CLAIM**: the Transformers Llama path at the pinned commit uses pre-norm RMSNorm, RoPE, configurable K/V head count, and a gated MLP (**O**, CLM-010). How many current open-weight families share this combination is `TODO_VERIFY` (requires a dated survey of released configs).
- **WORKLOAD- AND MODEL-DEPENDENT**: MQA versus GQA versus MHA; dense versus sparse experts.
- **FRONTIER / ARCHITECTURE-SPECIFIC**: MLA as defined by DeepSeek-V2 (paper read 2026-09-30). Newer hybrid attention/state-space designs are `TODO_VERIFY` in this module: no source for them was opened in this revision.
- **REJECT AS A MODEL**: “all current LLMs are Llama with different weights.”

**Worked Example (MoE, illustrative configuration, not a specific model):**
- *Input*: $E=8$ experts, top-$k=2$, $d=4096$, $m=14336$, bias-free SwiGLU experts.
- *Steps*: total $3\cdot8\cdot4096\cdot14336=1{,}409{,}286{,}144$; selected per token $3\cdot2\cdot4096\cdot14336=352{,}321{,}536$.
- *Result*: a $4\times$ ratio of total to selected expert weights for this layer ($E/k=8/2$; the common factor $3dm=176{,}160{,}768$ cancels).
- *Interpretation / limits*: every expert still has to be resident somewhere. Executed work depends on capacity padding and routing balance, and latency depends on communication and batching.

**Knowledge Check:**
1. Why are total, resident, selected, and executed quantities not interchangeable?
2. Which two tensors must an MLA decoder retain per token per layer, and why can per-head $K$ and $V$ be recomputed?
3. A colleague sizes an MLA model's KV with $2\cdot L\cdot$`num_key_value_heads`$\cdot d_h$. What is wrong, and which config fields should they use?

**Guided Practice:** Build an accounting table for dense, top-$k$ expert, and latent-attention configurations. For MLA, derive retained elements from $d_c$ and $d_h^R$ and list which config fields map to them. Mark any formula you have not confirmed against the released config or source as `TODO_VERIFY`.

**Feedback Contract:**
- *Expected Evidence*: separate total/resident/selected/executed rows for MoE; for MLA, $(d_c+d_h^R)$ elements per token per layer with the config field for each symbol, a stated storage precision left to Module 03, and an explicit statement that the GQA formula does not apply.
- *Common Failure*: equating selected parameters with wall-clock cost; treating MLA as renamed GQA; using `num_key_value_heads` for MLA state.
- *Diagnostic Hint*: which tensors does the paper say must be cached, and which are reconstructed by $W^{UK}$ and $W^{UV}$?
- *Concept to Revisit*: total versus active parameters; retained state versus reconstructed state.

**Learning Outcome:** Extend decoder accounting to sparse experts and latent attention without forcing incompatible architecture families into one formula, and hand Module 03 a correct retained-state model.

*(Effort: 55m instruction, 20m practice)*

---

### Lesson 1.8 — Production Source Trace and Diagnosis

**Engineering Question:** How does a pinned implementation turn the abstract decoder graph and sampling contract into a backend-dispatched execution path?

**Concepts & Definitions:** A source trace records repository, revision, path, symbol, entry point, configuration conditions, execution path, and static-versus-executed status.

At Transformers commit `11c16613d93911300c38ec8c9c2460567be53281`, trace (**O**, CLM-010):

```text
LlamaForCausalLM.forward
  → LlamaModel.forward
      → embed_tokens
      → create_causal_mask
      → LlamaRotaryEmbedding
      → each LlamaDecoderLayer.forward
          → input_layernorm
          → LlamaAttention.forward
              → q_proj / k_proj / v_proj
              → apply_rotary_pos_emb
              → optional Cache.update
              → ALL_ATTENTION_FUNCTIONS backend interface
              → o_proj
          → residual add
          → post_attention_layernorm
          → LlamaMLP.forward
          → residual add
      → final LlamaRMSNorm
  → lm_head
generation (same commit, CLM-015):
GenerationMixin._sample
  → outputs.logits[:, -1] copied to float32
  → logits_processor(input_ids, next_token_logits)
  → softmax + torch.multinomial   (do_sample=True)
    or torch.argmax               (do_sample=False)
  → append token; stopping_criteria
```

The source file defines orchestration and an eager reference path, but configured attention may dispatch to a different backend. Record the selected backend before making a kernel claim.

Use a diagnostic chain:

```text
SYMPTOM
  → competing shape / mask / position / dtype / backend / memory / sampling-contract hypotheses
  → missing tensor and timeline evidence
  → discriminating invariant or intervention
  → ranked explanation
  → fix
  → parity and workload remeasurement
```

Examples of discriminating tests:

- Future-token perturbation changes earlier logits → mask or packing hypothesis.
- FP32 norm path fixes non-finite outputs → precision hypothesis strengthened.
- Eager backend matches reference but optimized backend diverges → backend path strengthened.
- Analytical parameters differ from named unique storage → tying, sharing, padding, or omitted tensors.
- OOM boundary shifts across fresh processes with unchanged shapes → allocator/workspace/state hypothesis.
- Greedy outputs match but sampled outputs differ between two stacks with the same seed → compare effective processor lists and defaults before blaming the model.

**Worked Example:**
- *Input*: the pinned source above, read statically.
- *Steps*: `LlamaAttention.forward` selects an attention function through `ALL_ATTENTION_FUNCTIONS` according to configuration; `_sample` applies whatever processor list `_get_logits_processor` assembled.
- *Result*: the source proves the orchestration and the sampling order at this revision.
- *Interpretation / limits*: it does not prove which compiled attention kernel ran or which processors a given call used; both require an executed trace with the configuration recorded.

**Knowledge Check:** Which configuration selects the attention backend, and what evidence is required before making a kernel claim or a claim about the effective sampling processors?

**Independent Practice:** Execute the pinned path with hooks, record backend selection and the processor list for one sampled call, and map symbols to equations and invariants. Mark unexecuted branches `TODO_VERIFY`.

**Feedback Contract:**
- *Expected Evidence*: exact revision/path/symbols, dynamic configuration (attention implementation, dtype, cache use, generation config), the printed processor list, first-divergence evidence, and generalizability limits.
- *Common Failure*: citing the eager attention code as the kernel that ran, or citing `generate()` arguments as the effective sampling contract.
- *Diagnostic Hint*: log `config._attn_implementation` (or the equivalent field at your revision) and `type(p).__name__` for every processor.
- *Concept to Revisit*: static source reading versus executed trace.

**Learning Outcome:** Trace and diagnose a production forward and sampling path without generalizing revision-specific behavior.

*(Effort: 30m instruction; trace practice counted in LAB D source trace)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- Vaswani et al. (2017), [*Attention Is All You Need*](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) — scaled dot-product and multi-head attention (CLM-001).
- Zhang and Sennrich (2019), [*Root Mean Square Layer Normalization*](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html) — RMSNorm mechanism (CLM-002).
- Shazeer (2020), [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) — SwiGLU family (CLM-004).
- Su et al. (2021), [*RoFormer*](https://arxiv.org/abs/2104.09864) — rotary position embedding (CLM-003).
- Holtzman et al. (ICLR 2020), [*The Curious Case of Neural Text Degeneration*](https://arxiv.org/abs/1904.09751) — left-to-right factorization (Section 2), nucleus, top-k, and temperature sampling definitions (Sections 3.1–3.3); read 2026-09-30 (CLM-013). Its quality findings are specific to its models and tasks.

**CURRENT ARCHITECTURE EXTENSIONS**

- Shazeer (2019), [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) — MQA (CLM-005).
- Ainslie et al. (2023), [*GQA*](https://aclanthology.org/2023.emnlp-main.298/) — grouped-query attention and uptraining study (CLM-005).
- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html) — sparse routing reference (CLM-008).
- DeepSeek-AI (2024), [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) v5 — MLA equations and cache comparison (Sections 2.1.1–2.1.4) and model hyper-parameters (Section 3.1.2), read 2026-09-30; plus the released [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/config.json) at revision `4461458f186c35188585855f28f77af5661ad489` (CLM-009, CLM-017). Empirical results remain setup-specific.

**CURRENT SOURCE SNAPSHOT**

- Hugging Face Transformers commit `11c16613d93911300c38ec8c9c2460567be53281`:
  - `src/transformers/models/llama/modeling_llama.py` — statically verified 2026-09-25; symbol presence re-checked 2026-09-30 (CLM-010, CLM-011).
  - `src/transformers/generation/utils.py` (`GenerationMixin._sample`, `_get_logits_processor`), `generation/logits_process.py` (`TemperatureLogitsWarper`, `TopKLogitsWarper`, `TopPLogitsWarper`), and `generation/configuration_utils.py` (`GenerationConfig._get_default_generation_params`) — statically verified 2026-09-30 (CLM-015, CLM-016).

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow $\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}$.

### LAB A — Minimal Decoder, Numerical Parity, and Sampling Contract

- **Objective**: Implement RMSNorm, RoPE, causal attention, GQA mapping, SwiGLU, residuals, final norm, logits, and a sampling step without using a high-level decoder block or `generate()`.
- **Pre-Registered Hypothesis**: The transparent implementation will match the reference within declared tolerances until an injected invariant violation reaches the affected layer; with a fixed seed and identical processor chain, its sampled tokens will match the pinned reference on CPU for a short sequence.
- **Independent Variables**: Injected defect, dtype, sequence length, attention family, temperature, top-k, top-p, and processor order.
- **Dependent Variables**: Layerwise error, prefix invariance, finite counts, gradients where applicable, final-logit parity, post-processing distributions, and sampled-token agreement.
- **Controls**: fixed tiny configuration, deterministic weights/inputs, explicit dtype and tolerance, no dropout, fixed RNG generator and device.
- **Required tests**: shape assertions, finite outputs, future-token prefix invariance, norm/rotation properties, gradient check where applicable, output parity with a transparent reference, argmax invariance under temperature, top-k/top-p kept sets equal to hand calculation on the Lesson 1.2 logits, and a printed sampling contract.
- **Break & Falsify**: Inject wrong mask orientation, wrong RMSNorm axis, epsilon outside the root, and a head/sequence transpose that preserves element count; each must fail the relevant invariant. Swap temperature and top-p order and show which synthetic logits change the kept set; set $T=0$ and show the required error path.
- **Alignment**: Lessons 1.1–1.5.
- **Effort Estimate**: 3.5h implementation, 1.5h analysis (5h total).

### LAB B — MHA/MQA/GQA and RoPE Boundary Study

- **Objective**: Compare parameter geometry, logical KV dimensions, output parity at equivalent weights where meaningful, and measured runtime across MHA/MQA/GQA.
- **Pre-Registered Hypothesis**: Fewer KV heads reduce the declared K/V geometry, while latency benefit remains backend- and shape-dependent.
- **Independent Variables**: $H_{kv}$, batch, prompt length, decode context, backend, and position range.
- **Dependent Variables**: Projection counts, tensor shapes, backend identity, allocated/reserved memory, kernel timeline, and quality/parity limitations.
- **Break & Falsify**: Find a workload where fewer KV heads do not improve end-to-end latency, or a position range where a naive RoPE extrapolation claim fails.
- **Alignment**: Lessons 1.3–1.4.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Accounting Versus Real Execution

- **Objective**: Generate an analytical manifest for every parameter group and leading matmul, then reconcile it with `named_parameters`, profiler shapes, and peak memory.
- **Pre-Registered Hypothesis**: Exact logical counts will reconcile only after tying, biases, adapters, experts, padding, and unique storage are represented explicitly.
- **Independent Variables**: $B$, $S$, $d$, $m$, $H_q$, $H_{kv}$, tying, bias, adapter, backend, and workspace state.
- **Dependent Variables**: Count deltas, profiler-shape deltas, allocated/reserved peaks, and latency across the sweep.
- **Design**: sweep $B$, $S$, $d$, $m$, $H_q$, and $H_{kv}$ on configurations small enough to reproduce. Add one MLA configuration read from a released config: derive retained elements per token from $d_c$ and $d_h^R$ and record why the `num_key_value_heads` formula is rejected.
- **Break & Falsify**: Enable biases, tie/untie embeddings, add an adapter, change backend/workspace, introduce an MoE layer, and find a case where lower logical work does not yield proportional latency or memory savings.
- **Alignment**: Lessons 1.6–1.7.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — Pinned Transformers Source Trace

- **Objective**: Execute the pinned Llama forward and sampling path with hooks and map every source symbol to the mathematical operation and tensor contract.
- **Pre-Registered Hypothesis**: Backend selection, numerical casts, and library generation defaults visible at runtime will explain at least one difference from a purely abstract decoder graph and hand-written sampling contract.
- **Independent Variables**: Attention backend, dtype, cache use, short/long position range, and generation config (explicit versus defaulted fields).
- **Dependent Variables**: Selected symbols/backend, effective processor list, layerwise parity, finite counts, and trace completeness.
- **Required trace**: entry point, configuration fields, causal-mask construction, RoPE, layer order, cache call boundary, attention backend dispatch, final norm, LM head, `_sample` logits position/dtype, and processor order.
- **Artifact**: repository, commit, verification date, file, symbol, entry point, execution path, selected backend, printed processor list, exact commands, and `TODO_VERIFY` for paths not run.
- **Break & Falsify**: Force an alternative backend or incompatible configuration and determine which assumed path no longer executes; call `generate(do_sample=True)` with and without an explicit `top_k` and show whether the effective processor list changes.
- **Alignment**: Lesson 1.8 (and Lesson 1.2's contract).
- **Effort Estimate**: 2h source trace (counted under `source_trace`), 1h execution (3h total).

## 07 Break / Incident Scenarios

### Incident 01.1: Plausible Logits, Corrupt Long Context

A converted model matches short-prompt top-1 tokens but loses quality beyond 8K tokens and occasionally produces non-finite activations in low precision. Parameter count appears close to the source checkpoint. Sampled outputs from the converted stack also “feel more random” than the source stack at the same stated temperature.

Investigate competing explanations:

1. wrong RoPE frequencies, scaling, pairing, or position IDs;
2. incorrect $H_{kv}$ grouping or reshape layout;
3. causal/padding-mask broadcast error;
4. RMSNorm epsilon or accumulation dtype mismatch;
5. tied versus untied output weights or missing parameters;
6. optimized attention backend discrepancy;
7. a different effective sampling contract (defaults, processor order, or missing top-k/top-p) rather than a model defect;
8. checkpoint/config mismatch unrelated to the suspected mechanism.

**Diagnostic Protocol (Task):**

1. Localize the first layer and position where parity diverges, using greedy decoding so sampling cannot mask the comparison.
2. Compare eager and optimized backends.
3. Run future-token perturbation and relative-position invariants.
4. Inspect norms, finite counts, shapes, dtypes, and configuration.
5. Print both stacks' effective processor lists and compare post-processing distributions on identical logits.
6. Rank explanations and state what remains unresolved.
7. Fix one mechanism and rerun the same short- and long-context tests.
8. Define immediate rollback, long-term prevention, and quantitative parity/quality evidence required before redeployment.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Reconstruct an Unfamiliar Decoder

Given an unfamiliar decoder configuration and a pinned implementation, produce:

1. a forward-pass graph with exact logical shapes and residual boundaries;
2. attention, MLP, norm, embedding, head, router/expert, and total parameter counts with exclusions;
3. leading FLOP equations with a declared FMA convention;
4. MHA/MQA/GQA or MLA classification without forcing incompatible formulas, including retained elements per token per layer;
5. a causal-mask and position-encoding invariant test suite;
6. a minimal executable reference for one layer;
7. a source trace showing backend dispatch and numerical dtype choices;
8. a profiler comparison explaining where analytical work and measured latency diverge;
9. an OOM prediction expressed as a component estimate plus empirical boundary, not fake exactness;
10. downstream implications for Modules 02–06 and 20 with scope boundaries preserved;
11. a sampling contract for the checkpoint's default generation, including the effective processor chain and defaults, verified on one step against a hand calculation.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit the pinned Transformers trace from Lesson 1.8/Lab D with exact revision, paths, symbols, runtime configuration, backend selection, effective processor list, static-versus-executed status, commands, and `TODO_VERIFY` markers.

### Rubric Dimensions

A submission is **Competent** overall only if every dimension is at least Competent. Strong on one dimension does not offset Insufficient on another.

| Dimension (evidence) | Insufficient | Competent | Strong |
|---|---|---|---|
| **Shape & mechanism** (D1, D4, Lab A) | Names components; axes missing or checked by element count only | Named axes at every arrow; head→group map and mask semantics stated; MLA/GQA distinguished | Predicts which silent reshape/mask defects each invariant catches and demonstrates one missed by a smoke test |
| **Parameter & FLOP accounting** (D2–D3, Lab C) | Formula quoted without bias/tie/padding/FMA conventions, or unreconciled | Counts derived from matrices with conventions; deltas to `named_parameters` explained | Separates weight-proportional and context-dependent terms and predicts where the accounting will diverge from measured latency or memory |
| **Numerical correctness** (D5–D6, Lab A) | Code runs and output shape is right, but mask, norm axis, dtype path, or backend is unverified or wrong — **this fails regardless of runtime success** | Invariants and parity pass at a justified tolerance; first divergence localized for injected defects | Tolerance derived from dtype/accumulation analysis; parity checked across two backends |
| **Sampling contract** (D11, Lab A, Lab D) | Records only `temperature` or `generate()` arguments | Complete contract; hand-computed step matches implementation's kept set | Shows one case where defaults, ties, or processor order change the distribution and records it |
| **Source dispatch** (D7, Lab D, required artifact) | Cites eager code as “the kernel”; no revision | Pinned revision/symbols; selected backend and processors recorded; static vs executed marked | Forces an alternate path and shows which assumed branch stops executing |
| **Architecture trade-off & transfer** (D4, D8–D10) | Treats the model as Llama-like by name | States which conclusions are Llama-, MoE-, MLA-, backend-, or workload-specific | Hands later modules correct inputs (e.g., MLA retained state) with explicit limits |
| **Failure analysis** (Incident 01.1) | Guesses from final logits | Competing hypotheses and a discriminating test, including sampling-contract drift | Ranks hypotheses by evidence, fixes one, and remeasures short and long context |

*Calibration cases*: (a) a submission whose decoder produces correctly shaped logits but fails the future-token perturbation test is Insufficient on Numerical correctness; (b) a submission that computes KV for an MLA config with `num_key_value_heads` is Insufficient on Shape & mechanism even if its arithmetic is correct.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Decoder forward path | 1.1 | 1.1 Guided Practice; Lab A; Lab D | Mastery D1, D6–D7; Incident step 1 | Named-axis graph; hook trace; rubric: Shape & mechanism |
| Logits → sampling contract | 1.2 | 1.2 Guided Practice; Lab A sampling tests; Lab D processor list | Mastery D11; Incident step 5 | Contract + hand-computed kept sets; rubric: Sampling contract |
| Attention geometry | 1.3 | 1.3 Guided Practice; Labs A/B | Mastery D1, D4–D5 | Head→group map; prefix-invariance test; rubric: Shape & mechanism |
| RoPE mechanics | 1.4 | 1.4 Guided Practice; Labs A/B | Mastery D5; Incident step 3 | Rotation/relative-shift tests; rubric: Numerical correctness |
| RMSNorm and SwiGLU | 1.5 | 1.5 Guided Practice; Lab A | Mastery D6–D7; Incident step 4 | Parity report with cast boundaries; rubric: Numerical correctness |
| Parameter/FLOP accounting | 1.6 | 1.6 Guided Practice; Lab C | Mastery D2–D3, D8–D9 | Analytical manifest vs `named_parameters`; rubric: Accounting |
| MoE and MLA distinctions | 1.7 | 1.7 Guided Practice; Lab C MLA row | Mastery D4, D10 | Retained-state table; rubric: Shape & mechanism, Transfer |
| Source-based diagnosis | 1.8 | Lab D | Required source trace; Mastery D7; Incident steps 1–8 | Pinned trace with backend + processors; rubric: Source dispatch, Failure analysis |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner can exit Module 01 when they can:

1. reconstruct an unfamiliar decoder from configuration and source;
2. preserve token, head, sequence, and feature axes through every operation;
3. turn logits into a next-token distribution under a written sampling contract and compute its effect by hand on a small example;
4. detect future-token leakage and position-transform errors with invariants;
5. distinguish MHA, MQA, GQA, sparse experts, and latent attention, including MLA's retained state;
6. derive scoped parameter and FLOP estimates and state every exclusion;
7. explain why logical work, active parameters, memory footprint, sampling settings, and latency are different quantities;
8. localize numerical or backend divergence before proposing a fix;
9. hand accurate architectural and sampling inputs to GPU, KV-cache, serving, optimization, reasoning, and distributed-inference modules.

### Module Wrap-Up (Final Mental Model Reconstruction)

The final invariant: **model architecture is an executable tensor contract, and generation is that contract plus a recorded sampling contract—not a bag of component names and a temperature.**

## 12 Competency Targets

```yaml
competency:
  sfia: 4-5
  bloom: Analyze -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
