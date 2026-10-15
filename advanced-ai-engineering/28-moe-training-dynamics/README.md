# Module 28 — MoE Training Dynamics

## 00 Why This Module Exists

A sparse mixture-of-experts (MoE) layer only saves compute if the router spreads tokens across experts, and the router is itself being trained. Nothing in the language-model loss asks for balance. The experts the router picks get gradient, get better for their tokens, and get picked more. Every balancing mechanism in use today, whether an auxiliary loss, a bias controller, a capacity limit, or a global-batch statistic, is an engineered counterweight to that loop. Each one trades some model quality, some specialization, or some step time for the balance it buys. In expert-parallel training the trade is paid twice: an imbalanced micro-batch slows the forward and the backward all-to-all on the most-loaded rank.

```text
tokens --> router logits --(z-loss)--> score fn (softmax | sigmoid) --(+bias, selection only)--> top-k
                                                                                     |
            capacity? --yes--> drop overflow / pad underflow     no --> dropless (block-sparse / grouped)
                                                                                     |
     dispatch all-to-all --> experts (+shared) --> combine all-to-all --> residual      (backward: transposes)
                                                                                     |
     balance signal: aux loss alpha*N*sum f_i P_i  @ scope {sequence | micro-batch | global}
                     or bias b_i += u*sign(mean - load_i) once per global batch
                                                                                     |
     measure: per-expert load, MaxVio, dead experts, per-domain shares, z-loss, drop rate, a2a time
```

Module 01 introduced MoE parameter accounting (Lesson 1.7). Module 20 covered expert parallelism and expert skew at inference (Lesson 20.4), and Module 24 the dense-versus-MoE decision and inference all-to-all cost (Lesson 24.4). Module 19 covered training memory and paired release gates (Lessons 19.2, 19.6). This module owns what happens to the router *during training*: score functions and selection, collapse dynamics, auxiliary and auxiliary-loss-free balancing, balance scope, capacity and dropless execution, router stability, fine-grained and shared experts, specialization measurement, upcycling, the forward-and-backward all-to-all cost of expert-parallel training, and MoE fine-tuning pitfalls. Inference expert placement stays in Module 20, the dense-versus-MoE choice in Module 24, and general distributed training (ZeRO/FSDP, TP/PP schedules, FP8 recipes) in Module 26, which is cross-referenced only. RL estimator design stays in Module 27; this module covers only the routing-drift hazard it creates for MoE.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Train or fine-tune an MoE model so that experts are used, balanced at the scope the hardware needs, specialized where the data allows, and stable, and diagnose which balancing mechanism is failing when quality, step time, or utilization regress.
- **What You Will Do**: Compute router outputs and gradients by hand, induce and stop router collapse on a toy MoE, compare auxiliary-loss and bias balancing at micro-batch and global scope, quantify dropping, padding, and dropless waste, model expert-parallel all-to-all cost in training, measure specialization separately from balance, decide on upcycling, and trace balancing code at pinned Megatron-LM and MegaBlocks commits.
- **Environment**: Python 3.10+ and PyTorch on CPU for simulators and toy MoE (LAB A, LAB C); a Colab-class GPU for a small MoE language model (LAB B) and a small fine-tuning arm (LAB D); pinned source checkouts for traces (TRACE). No local GPU is assumed, and no multi-node experiment is required: expert-parallel cost is modeled, not measured.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. All numbers in worked examples are synthetic or derived and labeled. Author-reported results are labeled as such and carry their model, scale, and recipe.

## 01 Baseline Assumptions

- Module 01: MoE total, resident, selected, and executed parameter accounting (Lesson 1.7); logits and softmax (Lesson 1.2); parameter and FLOP accounting (Lesson 1.6).
- Module 19: training memory, activation checkpointing (Lesson 19.2), and paired, margin-based release gates (Lesson 19.6).
- Module 20: collectives and topology cost (Lesson 20.2); expert parallelism and expert skew at inference, where the most-loaded rank sets layer time (Lesson 20.4).
- Module 24: inference all-to-all payload model, node-limited routing, and dense-versus-MoE trade-offs (Lesson 24.4).
- Module 15: paired comparison and release gates (Lesson 15.4).
- Taught here, not assumed: router gradients through top-k, all balancing losses and controllers, capacity and dropless execution, router z-loss, specialization metrics, upcycling, the backward-pass all-to-all, and MoE fine-tuning hazards. Module 26 (distributed training) is being authored in parallel; this module states its own training-communication model and does not depend on Module 26 lessons.

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
  instruction: 5h        # sum of lesson instruction estimates: 50+50+50+55+50+45 min
  guided_practice: 2h    # 6 lessons x 20 min
  labs: 13.5h            # LAB A 3h + LAB B 4h + LAB C 3h + LAB D 3.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 28.1 0.5h
  source_trace: 2h       # counted once; shared by Lesson 28.3 and LAB D
  total: 25.5h
```

The learner must compute what a router does to a token and to its gradient, predict when routing will collapse, choose and tune a balancing mechanism and its scope with stated quality and step-time consequences, size capacity or justify dropless execution, estimate expert-parallel training communication, measure specialization as something different from balance, decide whether to upcycle, and fine-tune an MoE while monitoring routing drift. Each claim about a recipe must name the model, scale, and source it came from.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Router Logits $\to$ Score Function (softmax / sigmoid) $\to$ Selection (token-choice top-$k$ / expert-choice) $\to$ Gradient Path (selected experts only) $\to$ Collapse Loop $\to$ Balancing Signal (auxiliary loss / bias controller) $\to$ Balance Scope (sequence / micro-batch / global) $\to$ Execution (capacity + drop + pad / dropless) $\to$ Expert-Parallel Communication (forward and backward all-to-all) $\to$ Stability (z-loss, router precision) $\to$ Architecture (fine-grained, shared experts) $\to$ Specialization Measurement $\to$ Initialization (from scratch / upcycled) $\to$ Fine-tuning and RL (routing drift, anchoring, replay). Cross-cutting risks: dead experts, over-balancing, silent token dropping, rank stragglers, router logit growth, causal leakage, bias-controller bugs, and routing drift.

## 04 Lessons

### Lesson 28.1 — Router Mechanics: Score Functions, Selection, and the Gradient Path

**Engineering Question:**
For one token, which experts run, with what weights, and which parameters receive gradient?

**Concepts & Definitions:**
- **Router**: a linear map from the hidden state to $E$ logits $x_1..x_E$, one per expert.
- **Softmax router**: Switch uses $p_i(x)=e^{h(x)_i}/\sum_j e^{h(x)_j}$ and routes each token to one expert ($k=1$) (**O**, CLM-002). Shazeer et al. introduced noisy top-$k$ gating, $G(x)=\mathrm{Softmax}(\mathrm{KeepTopK}(H(x),k))$, which applies softmax *after* top-$k$ (**O**, CLM-001). Softmax-then-top-$k$ and top-$k$-then-softmax give different gate values for the same logits.
- **Sigmoid router with renormalization**: DeepSeek-V3 scores each expert independently with $s_{i,t}=\mathrm{Sigmoid}(u_t^\top e_i)$ and normalizes the selected scores to sum to 1. It also adds $N_s$ shared experts that every token uses (**O**, CLM-009).
- **Token-choice vs expert-choice**: in token-choice, each token picks $k$ experts. In expert-choice, each expert picks its top-$k$ tokens with capacity $k=n c/e$, so a token can get several experts or none (**O**, CLM-006).
- **Causal leakage**: expert-choice selection compares tokens across positions. In causal language modeling, future tokens then influence earlier tokens' expert assignment. Wang et al. bound the leak at more than $K\log_2\frac{1-R}{R}$ bits per token, about 50 bits in their example (**O**, CLM-007). OLMoE found token choice better than expert choice at equal token budget (**O**, CLM-008).
- **Router gradient through top-$k$**: the selection is a non-differentiable index set. Only selected experts appear in the output, so only they and their gate values receive task gradient (**D**, CLM-010).

**Mechanism Explanation:**
State: router weights $W_r$, expert weights $\{W_e\}$. Operation for one token: logits $x=W_r h$; scores by the chosen function; top-$k$ indices $S$; gates $g_i$ for $i\in S$; output $y=\sum_{i\in S} g_i E_i(h)$ (plus shared experts). Transition after backward: $W_e$ changes only for $e\in S$; $W_r$ changes only through $\partial L/\partial g_i$ for $i\in S$ (plus whatever the normalizer couples in). An expert that is never selected for a kind of token never learns that kind of token, and the router never learns that it could have helped. That is the seed of the collapse loop in Lesson 28.2.

**Quantitative Model / Derivation (D, CLM-010):**
- Softmax-then-top-$k$: $g_i = e^{x_i}/\sum_{j=1}^{E}e^{x_j}$ for $i\in S$; $\sum_{i\in S} g_i<1$.
- Top-$k$-then-softmax: $g_i = e^{x_i}/\sum_{j\in S}e^{x_j}$; $\sum g_i = 1$.
- Sigmoid with renormalization: $g_i=\sigma(x_i)/\sum_{j\in S}\sigma(x_j)$; $\sum g_i=1$.
- $\partial L/\partial W_e = 0$ for $e\notin S$, for every variant.

**Worked Example (synthetic logits; $E=4$, $k=2$):**
- *Input*: $x=[2.0,\,1.0,\,0.5,\,-1.0]$.
- *Softmax over all*: $\sum e^{x_j}=12.1239$, probabilities $[0.6095, 0.2242, 0.1360, 0.0303]$. Top-2 = experts 1, 2 with gates $0.6095, 0.2242$ (sum $0.8337$).
- *Top-2 then softmax*: $e^{2}/(e^2+e^1)=0.7311$ and $0.2689$.
- *Sigmoid*: $[0.8808, 0.7311, 0.6225, 0.2689]$. Top-2 = experts 1, 2; renormalized gates $0.8808/1.6119=0.5464$ and $0.4536$.
- *Interpretation*: same logits, same experts, three different gate vectors. The softmax-over-all variant scales the output down by 0.8337, the sigmoid variant weights the two experts more evenly. Experts 3 and 4 get no gradient in any variant.
- *Limits*: one token. Selection sets diverge once a balancing bias or capacity limit is applied (Lessons 28.3, 28.4).

**Knowledge Check:**
1. Why does adding the same constant to all four logits leave the softmax gates unchanged but change the sigmoid scores?
2. In which variant does the router receive a gradient signal that involves unselected experts' logits, and through which term?
3. Why can expert-choice routing look better on training loss than on autoregressive evaluation?

**Guided Practice:**
Write a 30-line PyTorch MoE layer with $E=4$, $k=2$ and the three gate variants. Run one backward pass on a fixed token and print which expert weight tensors have nonzero `.grad`. Then switch to expert-choice over a batch of 8 tokens and count tokens that received zero experts.

**Feedback Contract:**
- *Expected Evidence*: The three gate vectors matching the worked example, the gradient mask per expert, and a zero-expert token count for expert choice.
- *Common Failure*: Assuming the router "sees" all experts' quality because softmax is over all logits.
- *Diagnostic Hint*: Which expert outputs appear in $y$? Only those can pass gradient back.
- *Concept to Revisit*: Logits and softmax (Module 01, Lesson 1.2); MoE accounting (Module 01, Lesson 1.7).

**Learning Outcome:**
Compute gates and gradient masks for softmax and sigmoid top-$k$ routing and justify token-choice routing for causal LM training.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 28.2 — Router Collapse and the Auxiliary Load-Balancing Loss

**Engineering Question:**
Why does an untrained-for-balance router collapse, and how strongly must an auxiliary loss push back?

**Concepts & Definitions:**
- **Router collapse**: Shazeer et al. observed the gating network converging to large weights for the same few experts, an imbalance that is self-reinforcing because favored experts train faster and are selected more (**O**, CLM-001). OLMoE reports that without a load-balancing loss most experts stay flat and become dead weights that hold memory but do no work (**O**, CLM-012).
- **Switch load-balancing loss**: $\mathcal{L}_{bal}=\alpha N\sum_{i=1}^{N} f_i P_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ the mean router probability for expert $i$. It is minimized by uniform routing; Switch uses $\alpha=10^{-2}$ (**O**, CLM-003). GShard used a related $\ell_{aux}=\frac1E\sum_e \frac{c_e}{S}m_e$ over local groups (**O**, CLM-005).
- **Interference gradient**: the gradient the balancing loss adds to the router, which is not aligned with the LM loss. Wang et al. swept $\alpha\in\{10^{-2},10^{-3},10^{-4},0\}$: small $\alpha$ caused routing collapse and large $\alpha$ notably degraded model performance (**O**, CLM-011).
- **MaxVio**: $(\max_i \mathrm{Load}_i-\overline{\mathrm{Load}})/\overline{\mathrm{Load}}$, measured per training batch (MaxVio$_{batch}$) or over a validation set (MaxVio$_{global}$) (**O**, CLM-013).

**Mechanism Explanation:**
The collapse loop: expert $j$ wins a few more tokens $\to$ it receives more gradient on those tokens $\to$ its output improves for them $\to$ the LM loss pushes $g_j$ up for similar tokens $\to$ the router selects $j$ more. The auxiliary loss inserts a negative feedback: $f_i$ is a count and carries no gradient, so the gradient flows only through $P_i$, scaled by the current load $f_i$. Heavily loaded experts have their probability pushed down hardest (**D**, CLM-014). The balance it reaches is a compromise between two gradients, which is why $\alpha$ is a quality knob, not just a balance knob.

**Quantitative Model / Derivation (D, CLM-014):**
- Uniform routing ($f_i=P_i=1/N$): $N\sum f_iP_i = 1$. Full collapse onto one expert: $N\sum f_iP_i=N$.
- $\partial \mathcal{L}_{bal}/\partial P_i = \alpha N f_i$ (with $f$ held constant).
- For top-$k$, normalize $f_i$ by $T k$ so uniform routing still gives 1 (the Megatron-Core form, Lesson 28.3).
- The value of $\mathcal{L}_{bal}$ is a proxy. Monitor per-expert load and MaxVio directly; a near-1 loss can coexist with a starved expert if $f$ and $P$ anti-correlate.

**Worked Example (synthetic, $N=4$, top-1, $\alpha=10^{-2}$):**
- *Input*: $f=[0.5, 0.25, 0.125, 0.125]$, $P=[0.4, 0.3, 0.15, 0.15]$.
- *Loss*: $4\times(0.5\cdot0.4+0.25\cdot0.3+0.125\cdot0.15+0.125\cdot0.15)=4\times0.3125=1.25$, so $\mathcal{L}_{bal}=0.0125$ against a minimum of $0.01$.
- *Gradient on $P$*: $\alpha N f = [0.02, 0.01, 0.005, 0.005]$. Expert 1's probability is pushed down four times harder than experts 3 and 4.
- *MaxVio$_{batch}$*: mean load 0.25, max 0.5, so $(0.5-0.25)/0.25=1.0$.
- *Interpretation*: a loss 25% above its floor corresponds to one expert at twice its fair share. With $N=64$ the collapse ceiling is 64, so loss curves must be read relative to 1, not as absolute values.
- *Limits*: the size of $\alpha$ that suffices depends on learning rate, initialization scale, and data clustering; it is measured (LAB A), not derived.

**Knowledge Check:**
1. Why is $f_i$ not differentiable, and what would go wrong if you replaced it with $P_i$ (i.e., $N\sum P_i^2$)?
2. A run shows $\mathcal{L}_{bal}/\alpha = 1.02$ but two experts receive under 0.5% of tokens. How is that possible?
3. Why does a larger $\alpha$ hurt LM quality even when balance is already good?

**Guided Practice:**
On a toy MoE ($E=8$, top-2, 2-layer MLP experts) trained on four Gaussian clusters, log per-expert load every 50 steps for $\alpha\in\{0, 10^{-3}, 10^{-2}\}$ and predict, before running, which $\alpha$ produces at least one expert under 1% load (**H**, CLM-049).

**Feedback Contract:**
- *Expected Evidence*: Per-expert load trajectories, MaxVio$_{batch}$ curves, the pre-run prediction, and task loss per $\alpha$.
- *Common Failure*: Reporting only the auxiliary-loss value, or computing $f_i$ with the wrong top-$k$ normalization so uniform does not equal 1.
- *Diagnostic Hint*: What is $N\sum f_iP_i$ at initialization, and what should it be?
- *Concept to Revisit*: Router gradient path (Lesson 28.1).

**Learning Outcome:**
Explain the collapse loop, compute the Switch loss and its gradient, and choose $\alpha$ from measured balance and quality.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 28.3 — Auxiliary-Loss-Free Balancing and Balance Scope

**Engineering Question:**
Can balance be enforced without a loss term, and over which set of tokens should balance be enforced?

**Concepts & Definitions:**
- **Loss-Free Balancing**: add a per-expert bias $b_i$ to the routing score *for top-$K$ selection only*; the gate that weights the expert output stays the unbiased score. After each batch, $b_i \leftarrow b_i + u\cdot\mathrm{sign}(e_i)$ with $e_i=\bar c - c_i$. The paper uses sigmoid gating and $u=0.001$ and argues the method adds no interference gradient (**O**, CLM-015). The authors report, for 1B and 3B models, validation perplexity 9.56 vs 9.50 and 7.97 vs 7.92, and MaxVio$_{global}$ 0.72 vs 0.04 and 0.52 vs 0.04 (auxiliary vs loss-free; author-reported, no variance in the table) (**O**, CLM-016).
- **DeepSeek-V3 recipe**: bias update speed $\gamma=0.001$ for the first 14.3T tokens, then 0.0 for the last 500B; a complementary *sequence-wise* balance loss with $\alpha=0.0001$; no token dropping (**O**, CLM-017). Its ablations report the auxiliary-loss-free strategy better on most benchmarks at 15.7B and 228.7B scale. At 1B, validation loss was 2.258 with a sequence-wise loss and 2.253 for both loss-free and batch-wise loss. It names two risks of batch-wise balance: imbalance within certain sequences or small batches, and domain-shift imbalance at inference (**O**, CLM-018).
- **Balance scope**: the token set over which $f_i$ is counted. Qiu et al. note that frameworks compute it per micro-batch, which at LLM scale is nearly per-sequence and forces even code-only sequences to spread over all experts. Synchronizing $f_i$ to the global batch gave about 0.1 lower pre-training perplexity and about 2 benchmark points, specialization emerged clearly only with global balance, and overhead was under 3% (author-reported) (**O**, CLM-019). Qwen3 MoE (128 experts, 8 active, no shared experts) adopts the global-batch loss (**O**, CLM-020).
- **Control view**: the sign update is a fixed-step integral controller. ID Balancing (2026) frames it so and proposes an integral-derivative controller for 768-expert, top-3 routing (**O**, CLM-022; FRONTIER, abstract-level). Han and Zhong analyze it as a single-step primal-dual method (**O**, CLM-023; FRONTIER, abstract-level). Kimi K3's Quantile Balancing is a further variant whose primary source was not opened (CLM-053, `TODO_VERIFY`).

**Mechanism Explanation — Pinned Megatron-Core Trace (O, CLM-024, CLM-025):**
At `NVIDIA/Megatron-LM` commit `e998be072d22ff57b4b49899819a1a153f4f3dc9` (main, 2026-09-30; static inspection 2026-10-12):
1. `TopKRouter.__init__` registers an fp32 `expert_bias` and an int64 `local_tokens_per_expert` buffer when `moe_router_enable_expert_bias` is set.
2. `moe_utils.py::topk_routing_with_score_function` with `score_function="sigmoid"` computes `scores_for_routing = scores + expert_bias`, takes top-$k$ indices from that, then gathers the *unbiased* `scores` and normalizes them over the selected experts.
3. `TopKRouter._apply_expert_bias` adds the routing map to `local_tokens_per_expert`, only when grad is enabled, so evaluation and recomputation passes do not count.
4. `finalize_model_grads` calls `_update_router_expert_bias` once per global batch. `get_updated_expert_bias` all-reduces counts over the TP×DP×CP group and applies `bias += rate * sign(total - count * num_experts)`. `moe_router_bias_update_rate` defaults to `1e-3`.
5. `TransformerConfig` validation rejects expert bias unless the score function is `sigmoid` or `sqrtsoftplus`.
6. Scope is a code path: `_apply_aux_loss` reduces counts over TP×CP (micro-batch scope), `_apply_seq_aux_loss` balances each sequence and divides by batch size, and `_apply_global_aux_loss` reduces over TP×DP×CP and averages over gradient-accumulation steps. `switch_load_balancing_loss_func` multiplies by `num_experts * coeff / (topk * T^2)`. `moe_aux_loss_coeff` defaults to 0.0.

Lesson: the same name ("aux loss") covers three balance scopes, and the bias controller's correctness depends on where it counts and when it updates. All of these belong in the run manifest.

**Quantitative Model / Derivation (D, CLM-021):**
- Bias selection: choose $S=\mathrm{TopK}(s_i+b_i)$; gate $g_i=s_i/\sum_{j\in S}s_j$.
- Controller step: each expert moves by $\pm u$ per global batch. Two experts whose score gap is $\Delta$ close it in about $\Delta/(2u)$ updates if one moves up and the other down.
- Scope: $\mathcal{L}_{global}=\frac{1}{N_P}\sum_j N\sum_i \bar f_i P_i^{j}$ with $\bar f$ from the whole batch (CLM-019's form) versus $\frac{1}{N_P}\sum_j N\sum_i f_i^{j}P_i^{j}$ per micro-batch.

**Worked Example (synthetic):**
1. *Bias changes selection, not weight.* Sigmoid scores from Lesson 28.1 are $[0.8808, 0.7311, 0.6225, 0.2689]$, with bias $b=[-0.2, 0, +0.3, 0]$ after expert 1 was overloaded and expert 3 underloaded. Selection scores are $[0.6808, 0.7311, 0.9225, 0.2689]$, so top-2 = experts 3, 2. Gates use unbiased scores: $0.6225/1.3536 = 0.4599$ and $0.7311/1.3536=0.5401$.
2. *Why not softmax.* Adding +1 to expert 3's logit before a softmax over all experts changes every probability, from $[0.6095, 0.2242, 0.1360, 0.0303]$ to $[0.4940, 0.1817, 0.2996, 0.0246]$ (**D**). A bias that leaks into gate values leaks into the gradient, which is what the method exists to avoid. The pinned config enforces sigmoid-family scores (CLM-024).
3. *Controller lag.* With $u=10^{-3}$, a selection-score gap of 0.05 between an overloaded and an underloaded expert takes about $0.05/(2\times10^{-3})=25$ global batches to close.
4. *Scope.* Two sequences: code routes all tokens to experts 1–2, prose to 3–4 ($N=4$, top-1, $P=f$ within each sequence). Per-sequence loss: $4(0.5\cdot0.5+0.5\cdot0.5)=2.0$ each, twice the floor. Global: $\bar f = [0.25]^4$, each term $4(0.25\cdot0.5\cdot2)=1.0$, the floor. The micro-batch loss penalizes exactly the domain split the global loss allows.
- *Interpretation and limits*: global balance permits specialization but not per-micro-batch balance, which is what sets expert-parallel step time (Lesson 28.4). DeepSeek-V3 pairs global-style bias balancing with a tiny sequence loss; Qwen3 uses a global loss. Neither establishes an industry default.

**Knowledge Check:**
1. Why must the bias be updated from counts all-reduced across data-parallel ranks rather than from local counts?
2. What happens to balance if the bias update runs on every micro-batch including activation recomputation?
3. A model trained with global-batch balance shows high step-time variance under expert parallelism. Which risk named by DeepSeek-V3 is this, and what does adding a small micro-batch loss trade?

**Guided Practice:**
Implement the sign-step bias controller in your LAB A toy MoE with sigmoid scores. Plot MaxVio$_{batch}$ for $u\in\{10^{-4},10^{-3},10^{-2}\}$, identify oscillation or lag, and complete the source trace steps 1–6 at the pinned commit, recording file and symbol for each.

**Feedback Contract:**
- *Expected Evidence*: MaxVio traces per $u$, a statement of where your counts are reduced and when the update fires, and the trace table with file::symbol entries.
- *Common Failure*: Adding the bias to the gate value (it then changes the LM gradient), or counting tokens during evaluation.
- *Diagnostic Hint*: Print the gate vector with and without the bias for one token; they must be equal up to which experts are selected.
- *Concept to Revisit*: Auxiliary loss gradient (Lesson 28.2).

**Learning Outcome:**
Implement and trace bias-based balancing, choose a balance scope, and state its quality and step-time consequences.

*(Effort: 50m instruction, 20m practice; source trace 2h)*

---

### Lesson 28.4 — Capacity, Token Dropping, Dropless Execution, and Expert-Parallel Training Cost

**Engineering Question:**
What does imbalance cost in dropped tokens, padded compute, and all-to-all time during training?

**Concepts & Definitions:**
- **Expert capacity**: Switch defines capacity = (tokens per batch / experts) × capacity factor; overflow tokens skip expert computation and pass through the residual. Switch reports better results at capacity factors 1.0–1.25 in its setting (**O**, CLM-004). GShard also passes overflow through residuals (**O**, CLM-005). ST-MoE recommends top-2 with capacity factor 1.25, reports that higher train capacity factors gave small gains offset by a 14% slowdown at larger scale, and found fine-tuning unaffected by dropping up to 10–15% of tokens (**O**, CLM-030).
- **Dropless MoE**: MegaBlocks expresses expert computation as block-sparse matrix products (SDD then DSD, 128×128 blocks) and never drops tokens. In its study, an MoE at capacity factor 1 reduced validation loss by 0.15 against the dense baseline while the dropless MoE reduced it by 0.26; it reports 1.38×, 2.0×, and 4.35× end-to-end speedups over padding-based drop avoidance in Tutel for MoE-XS/Small/Medium (author-reported, A100, 2022 baselines) (**O**, CLM-033). DeepSeek-V3 drops no tokens in training or inference (**O**, CLM-017).
- **Expert-parallel (EP) training communication**: DeepSeek-V3 reports a computation-to-communication ratio of about 1:1 under cross-node EP. Each forward and backward chunk is split into attention, dispatch, MLP, and combine for overlap, 20 SMs handle communication, each token is limited to 4 nodes, and activations are cast to FP8 before dispatch (**O**, CLM-035).
- **Backward all-to-all**: the adjoint of dispatch sends gradients back along the same routes, and likewise for combine. A training step therefore performs four all-to-alls per MoE layer per micro-batch, six if the MoE layer is recomputed (**D**, CLM-036).

**Mechanism Explanation — Pinned Traces (O, CLM-027, CLM-028):**
- *Megatron-Core* (`e998be07…`): `TopKRouter.routing` calls `apply_router_token_dropping` only when `moe_expert_capacity_factor` is set; the default `None` means dropless. Capacity is `get_capacity(num_tokens*topk, num_experts, cf) = ceil(num_tokens*topk/num_experts*cf)`. Overflow is decided per expert by router probability (`drop_policy="probs"`) or position, with optional padding to capacity.
- *MegaBlocks* (`databricks/megablocks` at `952db33d6eac334d22c61e47a0d5d41446298784`, committed 2026-03-25): `ParallelDroplessMLP.indices_and_padded_bins` sorts assignments by expert, histograms them, and rounds each count up to `blocking=128`; `sparse_forward_once` gathers, runs block-sparse expert MLPs, and scatters. The capacity path `ParallelMLP.expert_capacity` is `int(moe_capacity_factor * top_k * tokens * ep_world_size / num_experts)`.

**Quantitative Model / Derivation (D, CLM-034, CLM-036):**
- $C=\lceil Tk/E\cdot CF\rceil$; dropped $=\sum_e\max(0,n_e-C)$; padding $=EC-\sum_e\min(n_e,C)$; pad-to-max waste $=E\max_e n_e - Tk$; block-rounded waste $=\sum_e(\lceil n_e/b\rceil b - n_e)$.
- EP training bytes per MoE layer per micro-batch per rank: $n_{a2a}\cdot Bkd\cdot\text{bytes}\cdot\frac{R-1}{R}$ with $n_{a2a}=4$ (6 with recompute); time $\ge \Gamma\cdot\text{bytes}/BW_{eff}$ where $\Gamma\ge1$ is the max-over-receivers imbalance (Module 20, Lesson 20.4; Module 24, Lesson 24.4 for the inference form).

**Worked Example (synthetic):**
1. *Drop, pad, dropless.* $T=4{,}096$ tokens in one routing group, $E=8$, top-1, loads $n=[1400, 600, 500, 500, 500, 300, 100, 196]$ (mean 512, MaxVio $=(1400-512)/512=1.734$).

| Policy | Capacity | Dropped | Drop % | Slots | Padding |
|---|---:|---:|---:|---:|---:|
| CF 1.0 | 512 | 976 | 23.83% | 4,096 | 976 |
| CF 1.25 | 640 | 760 | 18.55% | 5,120 | 1,784 |
| CF 2.0 | 1,024 | 376 | 9.18% | 8,192 | 4,472 |
| Pad to max | 1,400 | 0 | 0% | 11,200 | 7,104 (63.4%) |
| Dropless, $b=128$ | — | 0 | 0% | 4,352 | 256 (5.9%) |

   Raising capacity buys fewer drops with more padding; neither reaches zero drops below CF ≈ 2.73 here. Block rounding wastes 256 rows.
2. *EP all-to-all in training.* $B=4{,}096$ tokens per rank per micro-batch, $k=8$, $d=4{,}096$, BF16, $R=8$ EP ranks, $BW_{eff}=50$ GB/s (synthetic), 24 MoE layers, no overlap. Per all-to-all: $4096\cdot8\cdot4096\cdot2=268{,}435{,}456$ bytes (256 MiB), of which $7/8$ ($234{,}881{,}024$ bytes) leaves the rank, so $4.70$ ms. Per layer: $4\times4.70=18.79$ ms, or $28.19$ ms with MoE recompute. Per micro-batch: $24\times18.79=451.0$ ms; at $\Gamma=1.5$, $676.5$ ms.
- *Interpretation*: the inference model in Module 24 counts two all-to-alls per layer; training counts four or six, and imbalance multiplies all of them. FP8 dispatch, node-limited routing, and overlap reduce exposed time; balance scope decides $\Gamma$.
- *Limits*: payload only; ignores latency terms, deduplication across experts on the same rank, and kernel efficiency.

**Knowledge Check:**
1. Why does raising capacity factor from 1.0 to 1.25 in the table cut dropped tokens by only 22% while adding 83% more padding?
2. Which training-time cost does dropless execution not remove?
3. Why is $\Gamma$ in training set by per-micro-batch load, even under global-batch balance?

**Guided Practice:**
Recompute the table for Zipf-skewed loads with exponent $s\in\{0.5,1.0,1.5\}$ over 8 experts at $T=4{,}096$, and the all-to-all time with FP8 dispatch (1 byte) and BF16 combine (2 bytes) in forward and the same in backward (**H**, CLM-051 for the skew trend).

**Feedback Contract:**
- *Expected Evidence*: Drop, padding, and block-waste values per skew; forward and backward byte counts with dtypes stated; total exposed time with and without recompute.
- *Common Failure*: Counting only forward all-to-alls, or applying the capacity formula to $T$ instead of $Tk$.
- *Diagnostic Hint*: Where does the gradient of the dispatched hidden state have to travel?
- *Concept to Revisit*: Expert skew (Module 20, Lesson 20.4); all-to-all payload (Module 24, Lesson 24.4).

**Learning Outcome:**
Size capacity, quantify dropping, padding, and dropless waste, and estimate forward-plus-backward EP communication for a training step.

*(Effort: 55m instruction, 20m practice)*

---

### Lesson 28.5 — Router Stability, Expert Granularity, Shared Experts, and Specialization

**Engineering Question:**
How do you keep the router numerically stable, and how do you tell specialization from balance?

**Concepts & Definitions:**
- **Router z-loss**: ST-MoE defines $L_z=\frac1B\sum_i(\log\sum_j e^{x_j^{(i)}})^2$, which penalizes large router logits that exacerbate roundoff in exponentials, with $c_z=0.001$ in $L=L_{CE}+c_BL_B+c_zL_z$. On sparse models FLOP-matched to T5-XL, the baseline was stable in 4 of 6 seeds (quality $-1.755$) and z-loss in 3 of 3 ($-1.741$); update clipping was stable in 3 of 3 but far worse ($-4.206$). Removing multiplicative components or adding jitter/dropout improved stability but significantly hurt quality (**O**, CLM-029; small seed counts).
- **Why z-loss is needed**: softmax is invariant to adding a constant to all logits, so the LM loss never constrains their common magnitude. $L_z$ rises with it, and its gradient on logit $j$ is $2\,\mathrm{lse}(x)\,\mathrm{softmax}(x)_j$ (**D**, CLM-054). Megatron-Core applies z-loss to router logits before top-$k$; `moe_z_loss_coeff` defaults to `None` (**O**, CLM-026).
- **Selective precision and init scale**: Switch casts router inputs to float32 inside the router only and reduces the initialization scale by 10× (**O**, CLM-002). OLMoE reports that z-loss (weight 0.001) reduced spikes and improved quality (**O**, CLM-038).
- **Fine-grained and shared experts**: DeepSeekMoE splits each expert into $m$ smaller experts and activates $m$ times as many, and isolates $K_s$ shared experts. It cites $\binom{16}{2}=120$ versus $\binom{64}{8}=4{,}426{,}165{,}368$ combinations, and reports DeepSeekMoE 16B comparable to LLaMA2 7B at about 40% of the compute (author-reported) (**O**, CLM-037). OLMoE found diminishing returns to granularity (about 10% from 8 to 32 experts, 1–2% more at 64) and that a shared expert performed slightly worse (**O**, CLM-038). Qwen3 removed shared experts (**O**, CLM-020). The shared-expert question is recipe-dependent.
- **Specialization metrics**: OLMoE measures domain specialization (share of a domain's tokens routed to an expert), vocabulary specialization (share of a token ID's occurrences), co-activation, and router saturation (agreement of an intermediate checkpoint's routing with the final one: up to about 60% after 1% of pretraining) (**O**, CLM-039). ST-MoE saw token-type specialization in encoder experts but little in decoder experts and no language specialization (**O**, CLM-032). DeepSeek-V3 reports greater specialization with loss-free balancing (**O**, CLM-018), and global-batch balance made domain specialization emerge (**O**, CLM-019).
- **Balance is not specialization**: global load can be perfectly uniform while each domain uses a disjoint expert subset. Specialization is measured conditionally, against the uniform share $k/E$ (**D**, CLM-048). Over-dispersed routing, which Kapusuzoglu et al. associate with aggressive balancing, made router-based importance signals uninformative for pruning (**O**, CLM-046; FRONTIER, abstract-level).

**Mechanism Explanation:**
Router logits can drift upward together without changing the softmax, until bf16 roundoff in $e^x$ or a few saturated gates destabilize training. Z-loss supplies the missing restoring force; fp32 router math reduces the roundoff it guards against. Granularity raises the number of expert combinations per token at constant compute, which gives the router more ways to specialize. Shared experts remove common computation from routing, but whether that helps depends on the recipe. Specialization is a property of *conditional* routing distributions, so it must be measured on labeled held-out data, not inferred from balance curves.

**Quantitative Model / Derivation (D, CLM-048, CLM-054):**
- $L_z=\mathrm{mean}_t\,\mathrm{lse}(x_t)^2$, $\partial L_z/\partial x_j = 2\,\mathrm{lse}\,\mathrm{softmax}_j$ per token.
- Domain share $D_{e}^{(d)} = \#\{t\in d: e\in S_t\}/|d|$; uniform baseline $k/E$; report $\max_e D_e^{(d)}/(k/E)$ per domain and layer.
- Combinations at fixed compute: $\binom{mE}{mk}$ versus $\binom{E}{k}$.

**Worked Example (synthetic):**
1. *Z-loss.* Logits $[2.0,1.0,0.5,-1.0]$: $\mathrm{lse}=2.4952$, $\mathrm{lse}^2=6.2259$, $c_zL_z=0.006226$. Shift all logits by +10: softmax unchanged, $\mathrm{lse}^2=156.13$, $c_zL_z=0.1561$, a 25× larger penalty for a change the LM loss cannot see. Gradient coefficients $2\,\mathrm{lse}\,p=[3.041, 1.119, 0.679, 0.151]$ (× $c_z$): the most probable expert's logit is pulled down most.
2. *Granularity.* 8 experts top-2 give $\binom82=28$ combinations; splitting each into 4 ($m=4$, 32 experts, top-8) gives $\binom{32}{8}=10{,}518{,}300$ at the same expert FLOPs.
3. *Specialization vs balance.* $E=8$, top-2, uniform share $k/E=0.25$. Four equal-size domains each route every token to a disjoint expert pair (code to experts 1–2, and so on). Every expert gets the same load, so MaxVio $=0$, yet each domain's share on its own pair is $1.0$, four times the baseline, and $0$ elsewhere. Balance metrics alone report nothing.
- *Interpretation and limits*: a z-loss value is only interpretable relative to its own history; specialization numbers depend on domain labels and layer.

**Knowledge Check:**
1. Why does z-loss matter less, in principle, for a sigmoid router? What would you measure to check?
2. A fine-grained model has uniform load and identical per-domain shares in every layer. What two explanations compete, and what measurement separates them?
3. Why can OLMoE's shared-expert result and DeepSeekMoE's design both be correct?

**Guided Practice:**
In the LAB B small MoE, log per-layer mean $\mathrm{lse}$ of router logits with and without z-loss, and compute per-domain expert shares on a labeled held-out set for micro-batch versus global balance.

**Feedback Contract:**
- *Expected Evidence*: lse trajectories, z-loss on/off loss curves (spikes marked), per-domain share heatmaps per layer with the $k/E$ baseline marked.
- *Common Failure*: Reading high MaxVio as specialization, or uniform load as lack of specialization.
- *Diagnostic Hint*: Condition on the domain before you compute the share.
- *Concept to Revisit*: Balance scope (Lesson 28.3).

**Learning Outcome:**
Stabilize router training with z-loss and precision controls, and measure specialization as a conditional quantity distinct from balance.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 28.6 — Upcycling and Fine-Tuning MoE Models

**Engineering Question:**
Should this MoE start from a dense checkpoint, and what changes when an MoE is fine-tuned or trained with RL?

**Concepts & Definitions:**
- **Sparse upcycling**: copy the dense MLP into every expert and add a randomly initialized router. Komatsuzaki et al. replaced half of the MLP layers, used capacity factor 2, and report beating dense continuation with about 50% of the original dense pretraining cost, with a from-scratch MoE catching up at about 120% of the dense budget, after an initial quality drop from the structural change (**O**, CLM-040). OLMoE's upcycled OLMo-1B was overtaken by a from-scratch MoE at around 600B tokens (**O**, CLM-041). Drop-Upcycling re-initializes part of the copied weights to speed specialization (**O**, CLM-042; FRONTIER, abstract-level).
- **Expert symmetry**: with identical experts and gates that sum to 1, the upcycled layer initially computes exactly the dense output. Only the router and data order break the symmetry (**D**, CLM-047).
- **MoE fine-tuning findings (ST-MoE)**: sparse models overfit more; updating only non-MoE parameters works about as well as updating all, updating only MoE parameters (about 80% of weights) is much worse; dropout helps modestly; sparse models preferred smaller batches and higher learning rates; balance losses helped fine-tuning (**O**, CLM-031; encoder-decoder SuperGLUE era, so a starting hypothesis, not a recipe).
- **Routing drift in RL**: in RL of Qwen3-30B-A3B-Base, about 10% of activated experts differed between the new and old policy after a gradient update. The team's earlier Routing Replay cached and replayed the old policy's experts; GSPO removed that dependency (**O**, CLM-043; Module 27 owns the estimator). Megatron-Core exposes `moe_enable_routing_replay`, which swaps top-$k$ computation for a recorded selection (**O**, CLM-044).
- **Router anchoring**: Router Prior Bias reports that re-imposing a uniform balancing loss during post-training flattens the base router's structure, and that softly anchoring to the base router gave 45.77 in-domain math accuracy on Moonlight-16B-A3B versus 31.91 with re-applied LBL and 29.44 unanchored (**O**, CLM-045; FRONTIER, abstract-level, author-reported).

**Mechanism Explanation:**
Upcycling buys a good starting function at the cost of symmetric experts that diverge slowly. Fine-tuning moves the router as well as the experts, so a small, narrow dataset can re-route tokens the base handled well. The balancing loss used in pretraining was tuned for pretraining data; on a narrow domain it pushes against the specialization the domain needs. In RL, the probability ratio between new and old policies is computed under different expert selections unless routing is pinned, so routing drift becomes estimator noise.

**Quantitative Model / Derivation (D, CLM-047):**
- Upcycled layer at init: $y=\sum_{i\in S}g_i\,\mathrm{MLP}(x)=\mathrm{MLP}(x)$ when $\sum g_i=1$. With softmax-then-top-$k$ and no renormalization, $y=(\sum_{i\in S}g_i)\mathrm{MLP}(x)$, a scaled-down output.
- Added parameters: $(E-1)\cdot P_{MLP}$ per converted layer; per-token expert compute $\times k$.
- Routing drift between checkpoints A and B on a fixed probe set: fraction of (token, layer) pairs with $S_A\neq S_B$.

**Worked Example (synthetic dense decoder; $d=2048$, SwiGLU width 8192, 24 layers, convert every other layer to $E=8$, top-2):**
- *MLP size*: $3\cdot2048\cdot8192=50{,}331{,}648$ parameters.
- *Added parameters*: $7\times50{,}331{,}648\times12=4{,}227{,}858{,}432$.
- *Per-token expert compute in a converted layer*: $2\times$ the dense MLP ($100{,}663{,}296$ weights touched).
- *Initial output*: with renormalized top-2, identical to the dense model; with softmax-over-8 then top-2 at a near-uniform fresh router, gates sum to about $2/8$, cutting the MLP contribution to about a quarter. This is one candidate cause of an initial drop.
- *Decision*: if the remaining budget is short relative to the original dense budget, upcycle; if it is long, the reported crossovers (CLM-040, CLM-041) favor from-scratch MoE. The crossover for your model is a measurement.
- *Fine-tuning plan*: freeze router or anchor it, tune non-expert parameters first, and monitor routing drift and retained capability with paired gates (Module 19, Lesson 19.6).

**Knowledge Check:**
1. Why does an upcycled MoE with identical experts specialize at all?
2. What does re-applying the pretraining balance loss during domain fine-tuning optimize against?
3. In RL, why does a 10% change in selected experts between policy snapshots matter for the importance ratio?

**Guided Practice:**
Upcycle a 2-layer toy dense MLP model into $E=4$, top-2 with and without renormalization, and verify the initial output equality numerically. Then fine-tune the LAB B MoE on one domain with (a) trainable router, (b) frozen router, (c) re-applied micro-batch balance loss, and measure routing drift on a probe set (**H**, CLM-052).

**Feedback Contract:**
- *Expected Evidence*: Max absolute output difference at init per variant, per-layer drift per arm, in-domain and retained eval deltas with paired intervals.
- *Common Failure*: Reporting fine-tuning gains without measuring retained capability, or comparing arms with different seeds and calling the difference an effect.
- *Diagnostic Hint*: Do the gates of the selected experts sum to 1 at step 0?
- *Concept to Revisit*: Paired release gates (Module 19, Lesson 19.6); balance scope (Lesson 28.3).

**Learning Outcome:**
Decide between upcycling and from-scratch MoE, and fine-tune an MoE with explicit router policy and routing-drift monitoring.

*(Effort: 45m instruction, 20m practice)*

---

## 05 Literature & Production Source Map

Sources were opened on 2026-10-12. Section pointers are those inspected; abstract-only inspection is marked.

**REFERENCE / BASELINE**
- [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) — Shazeer et al., 2017; noisy top-$k$, self-reinforcing imbalance (Sections 2.1, 4).
- [GShard](https://arxiv.org/abs/2006.16668) — Lepikhin et al., 2020; top-2, capacity, local groups, auxiliary loss (Section 2.2).
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus, Zoph, Shazeer, 2021 (v3 2022); load-balancing loss, capacity, selective precision (Sections 2.1–2.4).
- [ST-MoE](https://arxiv.org/abs/2202.08906) — Zoph et al., 2022; router z-loss, stability table, fine-tuning, specialization (Sections 3–5, 7; Table 4).

**CURRENT DEFAULT** (scoped to the open recipes inspected, not an industry survey): token-choice top-$k$ routing; dropless execution; explicit balancing with recorded scope.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024; fine-grained and shared experts (Sections 3.1–3.3).
- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) — Wang et al., 2024; bias balancing, MaxVio, expert-choice leakage (Sections 2.2, 3, 4, 5.2).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024 (v2 2025); sigmoid gating, bias plus sequence loss, no dropping, EP overlap (Sections 2.1.2, 3.2, 3.3.3, 4.2, 4.5.2–4.5.3; Appendix C).
- [Demons in the Detail (global-batch LBL)](https://arxiv.org/abs/2501.11873) — Qiu et al., 2025; balance scope (Sections 2.2, 3, 4, 5).
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Qwen Team, 2025; 128 experts, 8 active, no shared experts, global-batch LBL (Section 2).
- [OLMoE](https://arxiv.org/abs/2409.02060) — Muennighoff et al., 2024 (v2 2025); open ablations of routing, balancing, z-loss, granularity, shared experts, upcycling, specialization (Sections 4.1, 5).
- [MegaBlocks](https://arxiv.org/abs/2211.15841) — Gale et al., 2022; dropless block-sparse MoE (Sections 3, 4.1, 5.1, 6.1).

**WORKLOAD-DEPENDENT:** [Expert Choice routing](https://arxiv.org/abs/2202.09368) (Zhou et al., 2022) for non-causal settings; [Sparse Upcycling](https://arxiv.org/abs/2212.05055) (Komatsuzaki et al., 2022) when the remaining budget is short; capacity-factor execution where static shapes are required; shared experts; ST-MoE fine-tuning recommendations.

**FRONTIER:** [ID Balancing](https://arxiv.org/abs/2609.39137) (Jin et al., 2026, abstract only); [Theoretical framework for ALF-LB](https://arxiv.org/abs/2512.03915) (Han and Zhong, 2025, abstract only); [Drop-Upcycling](https://arxiv.org/abs/2502.19261) (Nakamura et al., 2025, abstract only); [GSPO](https://arxiv.org/abs/2507.18071) (Zheng et al., 2025; Section 5.3 on MoE routing volatility); [Router Prior Bias](https://arxiv.org/abs/2609.08115) (Lee et al., 2026, abstract only); [Over-dispersed MoE pruning](https://arxiv.org/abs/2609.04453) (Kapusuzoglu et al., 2026, abstract only). Kimi K3 Quantile Balancing: primary source not opened (`TODO_VERIFY`, CLM-053).

**LEGACY / INSUFFICIENT:** fixed capacity with silent dropping as a pretraining default for decoder LLMs; expert-choice routing in causal LM pretraining; reporting only the auxiliary-loss value as evidence of balance; inferring specialization from balance metrics; reusing pretraining balance settings unchanged for narrow fine-tuning.

**PRODUCTION SOURCE TRACE**
- Repository: `NVIDIA/Megatron-LM` (Megatron-Core)
  - Revision: `e998be072d22ff57b4b49899819a1a153f4f3dc9` (last commit on `main` on or before 2026-09-30; committed 2026-09-30). Not a tagged release.
  - Verified: 2026-10-12; static inspection only.
  - Files/symbols: `megatron/core/transformer/moe/router.py::{TopKRouter.__init__, TopKRouter.routing, TopKRouter.apply_z_loss, TopKRouter._apply_aux_loss, TopKRouter._apply_seq_aux_loss, TopKRouter._apply_global_aux_loss, TopKRouter._apply_expert_bias}`; `megatron/core/transformer/moe/moe_utils.py::{topk_routing_with_score_function, switch_load_balancing_loss_func, z_loss_func, get_capacity, apply_router_token_dropping, get_updated_expert_bias}`; `megatron/core/distributed/finalize_model_grads.py::{finalize_model_grads, _update_router_expert_bias, reset_model_temporary_tensors}`; `megatron/core/transformer/transformer_config.py::TransformerConfig` (fields `moe_router_enable_expert_bias`, `moe_router_bias_update_rate`, `moe_aux_loss_coeff`, `moe_z_loss_coeff`, `moe_expert_capacity_factor`, `moe_router_load_balancing_type`, `moe_enable_routing_replay`).
  - Not inspected: fused Transformer Engine router path, `router_replay.py`, the `quantile_balancing` algorithm.
- Repository: `databricks/megablocks`
  - Revision: `952db33d6eac334d22c61e47a0d5d41446298784` (head of `main`, committed 2026-03-25).
  - Verified: 2026-10-12; static inspection only.
  - Files/symbols: `megablocks/layers/dmoe.py::{ParallelDroplessMLP.__init__, ParallelDroplessMLP.indices_and_padded_bins, ParallelDroplessMLP.sparse_forward_once}`; `megablocks/layers/moe.py::{ParallelMLP.expert_capacity, ParallelMLP.load_balancing_loss, batched_load_balancing_loss}`.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` and record a run manifest: router score function, top-$k$, $E$, shared experts, balancing mechanism and coefficient or update rate, balance scope, capacity factor or dropless, z-loss coefficient, router precision, initialization (scratch or upcycled), seeds, and data snapshot. Data is synthetic or public; label which. Hardware tiers: `CPU`, `COLAB-GPU`, `TRACE`. No local GPU is assumed.

### LAB A — Router Collapse on a Toy MoE
- **Hardware tier**: CPU.
- **Objective**: Train a toy token-choice MoE ($E=8$, top-2) on clustered synthetic data and induce, observe, and stop router collapse.
- **Pre-Registered Hypothesis**: CLM-049: without balancing, load skews progressively and at least one expert falls below 1% of assignments; Switch loss at $\alpha=10^{-2}$ keeps MaxVio$_{batch}$ below the no-balancing run after warm-up.
- **Independent Variables**: $\alpha\in\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}$; score function (softmax, sigmoid); router initialization scale; number of clusters.
- **Dependent Variables**: Per-expert load, MaxVio$_{batch}$, count of experts under 1% load, task loss, $\mathcal L_{bal}/\alpha$.
- **Break & Falsify**: Set $\alpha=0$ and a 10× larger router init; confirm collapse. Then find an $\alpha$ large enough to visibly raise task loss (over-balancing). The hypothesis is falsified if, across 3 seeds, no expert drops below 1% without balancing and MaxVio does not grow.
- **Alignment**: Lessons 28.1–28.2.
- **Effort Estimate**: 3h total.

### LAB B — Balancing Mechanism and Scope on a Small MoE LM
- **Hardware tier**: COLAB-GPU.
- **Objective**: Train a small decoder MoE language model (for example 8–16 experts, top-2, a few layers) on a domain-labeled public mix (code and prose) under auxiliary loss at micro-batch scope, auxiliary loss at global scope (via gradient-accumulation steps), and sigmoid-bias balancing; add a z-loss on/off arm.
- **Pre-Registered Hypothesis**: CLM-050: bias balancing reaches MaxVio$_{global}$ no higher than auxiliary loss at $\alpha=10^{-2}$ with validation loss no worse beyond seed spread; global-scope loss yields higher per-domain expert concentration than micro-batch scope at similar validation loss.
- **Independent Variables**: Mechanism, scope, $\alpha$, $u$, z-loss coefficient, shared expert on/off.
- **Dependent Variables**: Validation loss, MaxVio$_{batch}$ and MaxVio$_{global}$, per-domain expert shares per layer, router lse, loss spikes, step time.
- **Break & Falsify**: Apply the bias to gate values instead of selection only and show the change in LM gradient and loss; update the bias on every micro-batch including recomputation; set $u$ 10× too large and show oscillation. Falsified if the bias arm's MaxVio$_{global}$ or validation loss is worse than the auxiliary arm beyond the 3-seed spread, or global versus micro-batch scope shows no difference in domain concentration.
- **Alignment**: Lessons 28.3, 28.5.
- **Effort Estimate**: 4h total.

### LAB C — Capacity, Dropping, and Expert-Parallel Cost Simulator
- **Hardware tier**: CPU.
- **Objective**: Simulate routing groups under Zipf-skewed expert popularity; compute dropped assignments, padding, and block-rounded dropless waste; model forward-plus-backward all-to-all time per training step for a declared EP topology.
- **Pre-Registered Hypothesis**: CLM-051 (simulation part): drop rate at CF 1.0 increases monotonically with skew; pad-to-max waste exceeds block-rounded waste at $b=128$ for every skew.
- **Independent Variables**: Zipf exponent, $E$, $k$, CF, block size, $R$, $BW_{eff}$, recompute on/off, dispatch dtype, $\Gamma$.
- **Dependent Variables**: Drop %, padding slots, block waste, bytes per layer, exposed all-to-all time per micro-batch.
- **Break & Falsify**: Construct a load vector where raising CF from 1.0 to 1.25 removes no drops (one expert far above both capacities) and one where block rounding wastes more than CF 1.0 padding (many experts with tiny loads). Optional COLAB-GPU arm: train the LAB B model with CF 1.0 dropping versus dropless and compare validation loss over 3 seeds.
- **Alignment**: Lesson 28.4.
- **Effort Estimate**: 3h total.

### LAB D — Source Trace, Upcycling, and MoE Fine-Tuning Drift
- **Hardware tier**: TRACE for the source trace; COLAB-GPU for the upcycling and fine-tuning arms.
- **Objective**: Complete the pinned Megatron-Core and MegaBlocks trace; upcycle a small dense model and verify initial-output equivalence; fine-tune the LAB B MoE on one domain with trainable router, frozen router, and re-applied micro-batch balance loss.
- **Pre-Registered Hypothesis**: CLM-052: routing drift is highest with re-applied balance loss and lowest with a frozen router; in-domain gain with a frozen or anchored router is at least that with re-applied balance loss.
- **Independent Variables**: Router policy (trainable, frozen, re-applied LBL), parameter subset (all, non-expert only, expert only), learning rate, upcycling gate normalization.
- **Dependent Variables**: Per-layer routing drift on a fixed probe set, in-domain and retained eval with paired intervals (Module 15, Lesson 15.4), initial output error after upcycling.
- **Break & Falsify**: Upcycle with softmax-over-all then top-$k$ and no renormalization and measure the initial loss jump; tune only expert parameters and compare with non-expert only. Falsified if re-applied LBL shows drift no higher than the trainable-router arm, or higher in-domain gain than the frozen-router arm beyond intervals.
- **Alignment**: Lessons 28.3, 28.6.
- **Effort Estimate**: 3.5h total (source trace counted separately, 2h).

---

## 07 Break / Incident Scenarios

### Incident 28.1 — The Migration That Lost Its Experts

- **Incident Symptoms**:
  - A team moved a 64-expert, top-8 MoE pretraining run to a new framework configuration and doubled expert parallelism from 8 to 16 ranks.
  - After 200B tokens, validation loss is 0.04 worse than the old run at the same token count, and code benchmarks regressed more than prose.
  - Step time rose 30% and varies widely between steps. Logged auxiliary loss is close to 1.0. Throughput dashboards show no errors.
  - Changes made at once: router switched from softmax to sigmoid with expert bias enabled; the auxiliary loss changed from global-batch scope to the framework default; `moe_expert_capacity_factor` was set to 1.25 "to bound memory"; a custom training loop calls the bias update hook after every micro-batch; z-loss was left unset.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*:
     - micro-batch-scope balance suppressing domain specialization;
     - token dropping from the new capacity factor;
     - bias controller updating per micro-batch or from unreduced local counts;
     - bias leaking into gate values;
     - stragglers from per-micro-batch imbalance at the larger EP degree;
     - router logit growth without z-loss;
     - data-order or data-mix change between runs;
     - training variance.
  2. *Rank Initial Plausibility*: Use the change list and symptom pattern (code-heavy regression, step-time variance, near-1 aux loss), but assign no cause before a discriminating test. Several changes can interact; for example, dropping concentrates on hot code experts.
  3. *Identify Missing Evidence*: Both run manifests; per-expert, per-layer load histograms per micro-batch; drop rate per layer; bias values over time; where token counts are reduced and when the bias update fires; per-domain expert shares; router lse; per-rank expert compute and all-to-all time; data snapshot IDs.
  4. *Design Discriminating Tests*:
     - Log drop rate directly; rerun a short window with capacity factor unset (dropless).
     - Rerun with the bias update moved to once per global batch on all-reduced counts.
     - Switch the auxiliary loss to global scope; compare per-domain shares on code versus prose.
     - Check one token's gate vector with and without bias.
     - Plot per-rank time against per-rank assigned tokens per micro-batch.
     - Rerun the old configuration on the new data snapshot with 2 seeds.
  5. *Execute Causal Diagnosis*: Rank causes by the evidence each test produced. State interactions (for example, drops plus micro-batch balance) and anything unexplained.
  6. *Prescribe Mitigation and Prevention*: Return to dropless execution; fix the bias update cadence and reduction; restore global-scope balance with an optional small micro-batch term if stragglers require it; add z-loss; add manifest checks for scope, capacity factor, bias cadence, and score function; add per-domain share and drop-rate panels.
  7. *Remeasure*: Validation loss gap at matched tokens over 2 seeds, MaxVio$_{batch}$ and MaxVio$_{global}$, drop rate (should be 0), per-domain shares, and step-time distribution.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Training a Domain-Heavy MoE on a Two-Tier Cluster

A team will pretrain a 128-expert, top-8 decoder MoE (fine-grained experts, one candidate shared expert) on a mix of 40% code, 40% web text, and 20% multilingual text, then fine-tune it for a code assistant and run RL on verifiable code tasks.

Constraints:
- **Cluster.** 64 GPUs in 8 nodes; fast intra-node links and slower inter-node links (declare synthetic bandwidths).
- **Budget.** A dense 3B checkpoint exists. The MoE token budget is either short (20% of the dense model's pretraining tokens) or long (150%); plan both.
- **Quality.** Code specialization is desired; step-time variance must stay within a declared bound.
- **Framework.** Megatron-Core at the pinned commit or an equivalent, with every routing setting recorded.

**Required Deliverables**:
1. Router design: score function, top-$k$, gate normalization, shared expert decision, with evidence and the counter-evidence considered.
2. Balancing plan: mechanism, coefficient or update rate, scope, and the predicted effects on specialization and step time.
3. Execution plan: dropless versus capacity, with drop/padding/block-waste estimates under a declared skew.
4. EP communication estimate for one training step (forward and backward, with or without recompute), and the levers (node limit, dispatch precision, overlap) with their effect on bytes or exposure.
5. Stability plan: z-loss, router precision, and the signals monitored.
6. Specialization measurement plan with domain labels and baselines.
7. Upcycle-versus-scratch decision for both budgets, with the crossover evidence cited and how you would measure your own.
8. Fine-tuning and RL plan: router policy, parameter subset, routing-drift metric, and paired gates.
9. Pinned source trace (Megatron-Core and MegaBlocks).
10. Diagnosis and remediation for Incident 28.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace, at the pinned Megatron-Core commit: router logits to z-loss, score function, biased selection with unbiased gates, optional capacity dropping, the three auxiliary-loss scopes, token counting for the bias, and the once-per-global-batch bias update in `finalize_model_grads`. At the pinned MegaBlocks commit: dropless sort, histogram, block rounding, and block-sparse compute. State what was statically inspected and what, if anything, was executed.

### Reference Checks (worked-example inputs only)
- Lesson 28.1 gates: softmax-all $[0.6095, 0.2242]$; top-2-then-softmax $[0.7311, 0.2689]$; sigmoid-renormalized $[0.5464, 0.4536]$.
- Lesson 28.2: $N\sum f_iP_i=1.25$; gradient $[0.02, 0.01, 0.005, 0.005]$; MaxVio 1.0.
- Lesson 28.3: biased selection picks experts 3, 2 with gates $0.4599, 0.5401$; per-sequence loss 2.0 vs global 1.0.
- Lesson 28.4: CF 1.0 drops 976 (23.83%); dropless waste 256 rows; 4.70 ms per all-to-all, 18.79 ms per layer, 451.0 ms per micro-batch at $\Gamma=1$.
- Lesson 28.5: $c_zL_z$ 0.006226 vs 0.1561 after a +10 shift; $\binom{32}{8}=10{,}518{,}300$.
- Lesson 28.6: 4,227,858,432 added parameters.

### Rubric Dimensions
- **Router Mechanics**: *Insufficient* treats routing as a black box. *Competent* computes gates and gradient masks per variant. *Strong* links the gradient path to collapse and to the choice of score function for bias balancing.
- **Balancing and Scope**: *Insufficient* reports the auxiliary-loss value as balance. *Competent* measures per-expert load and MaxVio and tunes the mechanism. *Strong* chooses scope from both specialization and step-time evidence and states the trade.
- **Execution and Communication**: *Insufficient* ignores dropped tokens or backward communication. *Competent* quantifies drops, padding, and forward-plus-backward bytes. *Strong* connects per-micro-batch imbalance to exposed all-to-all time and names the levers with their limits.
- **Stability and Specialization**: *Insufficient* conflates balance with specialization. *Competent* adds z-loss and measures conditional routing shares. *Strong* separates competing explanations for uniform-looking routing and tests them.
- **Initialization and Fine-Tuning**: *Insufficient* upcycles or fine-tunes by default. *Competent* cites budget crossovers and monitors drift. *Strong* pre-registers router policy arms, measures drift and retained capability with paired intervals, and states what remains uncertain.
- **Source Reasoning**: *Insufficient* cites config names only. *Competent* traces execution paths by file and symbol. *Strong* shows how a code-level detail (counting cadence, reduction group, default capacity) produces an observable training behavior.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Router mechanics and gradient path | 28.1 | 28.1 Guided Practice; LAB A | Deliverable 1 | Gate vectors, gradient masks, token- vs expert-choice note |
| Collapse diagnosis and auxiliary loss | 28.2 | LAB A | Deliverable 2; Incident steps 1–4 | Per-expert load, MaxVio, $\alpha$ sweep |
| Bias balancing and balance scope | 28.3 | LAB B; LAB D trace | Deliverables 2, 9; Incident steps 4–6 | Scope comparison, bias trace, source-trace artifact |
| Capacity, dropless, EP training cost | 28.4 | LAB C | Deliverables 3–4; Incident step 4 | Drop/padding tables, forward+backward byte model |
| Stability and specialization measurement | 28.5 | LAB B | Deliverables 5–6 | lse and spike logs, per-domain share heatmaps |
| Upcycling and MoE fine-tuning | 28.6 | LAB D | Deliverables 7–8 | Init-equivalence check, routing-drift table, paired gates |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 28 must be able to:
1. Compute gates and gradient masks for softmax and sigmoid top-$k$ routing and justify token choice for causal training.
2. Induce, detect, and stop router collapse, and tune an auxiliary loss from measured balance and quality.
3. Implement and trace bias-based balancing and choose a balance scope with its specialization and step-time consequences.
4. Quantify dropping, padding, and dropless waste, and estimate forward-plus-backward expert-parallel communication.
5. Stabilize routers with z-loss and precision controls and measure specialization separately from balance.
6. Decide between upcycling and from-scratch MoE and fine-tune an MoE with a stated router policy and drift monitoring.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: The language-model loss does not ask for balance, and top-$k$ selection hides unselected experts from the gradient, so routing collapses unless something pushes back. Every pushback trades quality, specialization, or step time.
- **The Training Path**: router logits → (z-loss) → score function → biased selection with unbiased gates → dropless or capacity execution → dispatch, experts, combine, and their backward transposes → balance signal at a declared scope → measurement of load, drops, specialization, and communication.
- Balance scope sets what may specialize and how stragglers behave; capacity sets what is silently dropped; the bias controller is code whose cadence and reduction group matter; upcycling and fine-tuning change routing as well as weights. Next: Module 26 for the general distributed-training machinery around this layer, and Module 27 for RL estimators that routing drift perturbs.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
