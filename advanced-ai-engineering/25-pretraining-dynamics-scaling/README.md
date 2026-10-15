# Module 25 — Pretraining Dynamics & Scaling

## 00 Why This Module Exists

A pretraining run is decided before it starts. Model size, token count, learning rate, schedule, parametrization, and data mixture are committed on the basis of a forecast from small proxy runs. The expensive run then either confirms the forecast or exposes it. This module teaches how to make that forecast, how to state the conditions under which it is invalid, and how to keep the run stable long enough to test it.

```text
compute budget C, unique tokens U, inference demand
        |
        v
proxy sweep (width/depth, LR, mixture) --- matched schedules, total-param accounting, seeds
        |
        v
fit: L(N,D) / mixing law / LR optimum vs width ---> prediction + interval + invalidating conditions
        |
        v
target config: N, D, parametrization (SP/muP), schedule (cosine/WSD), mixture, stability stack
        |                                            (QK-norm, z-loss, epsilon, warmup, skip/rollback)
        v
monitored run: loss, grad norm, max attention logit, log Z, LR, spikes
        |
        v
decay / cooldown --> checkpoint average / soup --> held-out loss vs forecast --> post-training (Module 19)
```

Module 00 introduced experimental units, confidence intervals, and falsification (Lessons 0.3, 0.5, 0.7). Module 01 taught softmax and logits (Lesson 1.2), attention scores (Lesson 1.3), RMSNorm (Lesson 1.5), and parameter and FLOP accounting (Lesson 1.6). Module 08 taught data lineage and deduplication (Lessons 8.1, 8.4). Module 19 computed Adam model states (Lesson 19.2). This module owns compute-optimal allocation, scaling-law fitting and its failure, parametrization and hyperparameter transfer, loss-spike and logit-growth instabilities, learning-rate schedules, data-mixture optimization, and checkpoint averaging. Fine-tuning and adaptation stay in Module 19. Data lineage, deduplication, and contamination mechanics stay in Module 08. Distributed training memory, sharding, parallelism, and FP8 training belong to Module 26 and are only cross-referenced here.

**Research cutoff:** 2026-09-30. Sources were opened on 2026-10-12; claims about later work are not made.

**Module Orientation**
- **Engineering Problem**: Predict a large run's loss and good hyperparameters from small proxies, state when the prediction is invalid, and keep the run stable enough to test it.
- **What You Will Do**: Derive compute-optimal allocations under competing published fits; fit and stress a scaling law on proxy runs; implement muP and show learning-rate drift under standard parametrization; induce and fix attention-logit and output-logit instabilities; trace z-loss, skip-step, and schedule code at a pinned OLMo-core commit; compare cosine and warmup-stable-decay schedules; optimize a data mixture on proxies and test whether it transfers; and average checkpoints.
- **Environment**: Python 3.10+, NumPy/SciPy, PyTorch. Hardware tiers are `CPU`, `COLAB-GPU` (one free/low-tier Colab GPU), and `TRACE` (pinned source or published-log replay). No local GPU is assumed. Models are tiny decoders (≤ ~50M parameters) on public text.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. Author-reported results are labeled as such. All numbers in worked examples are synthetic or recomputed from cited fits and are labeled.

## 01 Baseline Assumptions

- Module 00: claims and estimands, experimental units and pairing, confidence intervals, and falsification (Lessons 0.1, 0.3, 0.5, 0.7).
- Module 01: softmax and logits (Lesson 1.2), attention scores (Lesson 1.3), RMSNorm numerics (Lesson 1.5), and the FMA-as-two parameter/FLOP convention (Lesson 1.6). The $6ND$ training rule is taught here (Lesson 25.1), not assumed.
- Module 02: honest timing and profiling, used only to report throughput in labs.
- Module 08: data lineage, deduplication, and contamination (Lessons 8.1, 8.4). Mixture weighting and repetition scaling are taught here.
- Module 19: model states under mixed-precision Adam (Lesson 19.2). Adam's epsilon and per-layer learning rates are taught here in Lessons 25.3–25.4.
- Not assumed: scaling-law functional forms, μP, training-stability mechanisms, schedule design, and mixture optimization. All are taught on first use.

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
  research_connection: REQUIRED

estimated_effort:
  instruction: 5h        # 6 lessons x 50m
  guided_practice: 2.5h  # 6 lessons x 25m
  labs: 13h              # LAB A 3h + LAB B 3.5h + LAB C 3.5h + LAB D 3h
  assessment: 3h
  source_trace: 1.5h     # counted once; shared by Lessons 25.4-25.5 and LAB C
  total: 25h
```

The learner must allocate compute with a stated loss model and explain why published allocations disagree. They must fit a scaling law on proxies, forecast a held-out run with an interval, and name the changes that would void the forecast. They must implement μP and show when standard parametrization fails to transfer a learning rate, induce and diagnose logit-growth instabilities, and trace their mitigations in pinned source. They must choose a schedule with its budget consequences, optimize a mixture while testing whether it transfers across scale, and average checkpoints with a stated same-basin assumption.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Compute Accounting ($6ND$, parameter counting) $\to$ Loss Model ($E + A/N^\alpha + B/D^\beta$) $\to$ Compute-Optimal Allocation (and its confounds: schedule, warmup, counting, optimizer tuning; inference demand) $\to$ Forecast Validity (irreducible term, seeds, number of runs, repetition, downstream metrics) $\to$ Parametrization (SP, μP, depth transfer) so that proxy hyperparameters mean something at scale $\to$ Stability (attention-logit growth, output-logit drift, epsilon, spikes; QK-norm, z-loss, soft-cap, skip/rollback) so the run survives $\to$ Schedule (cosine vs WSD; cooldown; decay floor) so that checkpoints and budgets are comparable $\to$ Mixture and Averaging (proxy mixture search, transfer risk, checkpoint averaging and souping). Cross-cutting risks: confounded fits, invariants silently changed between proxy and target, and rankings inside seed noise.

## 04 Lessons

### Lesson 25.1 — Compute Accounting and Compute-Optimal Allocation

**Engineering Question:**
Given a FLOP budget, how many parameters and how many tokens, and why did the field publish two different answers?

**Concepts & Definitions:**
- **Training compute**: Kaplan et al. estimate non-embedding training compute as $C \approx 6N$ FLOPs per token, i.e. $C\approx 6ND$, with $N$ excluding vocabulary and positional embeddings (**O**, CLM-001). The 6 is $2$ (forward, FMA-as-two from Module 01, Lesson 1.6) $\times 3$ (backward ≈ twice forward). It omits the context-dependent attention term.
- **Compute-optimal allocation**: the $(N, D)$ that minimizes final loss at fixed $C$. It is defined only relative to a loss model and a training procedure.
- **Kaplan allocation**: $N_{opt}\propto C^{0.73}$, $D\propto C^{0.27}$, with large models stopped well short of convergence. All their runs used a fixed $2.5\times10^5$ steps with cosine decay to zero (**O**, CLM-002). Classified LEGACY as an allocation rule.
- **Chinchilla allocation**: three methods give $a = 0.50, 0.49, 0.46$ and $b = 0.50, 0.51, 0.54$. The parametric fit is $L(N,D) = E + A/N^\alpha + B/D^\beta$ with $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$. Chinchilla (70B, 1.4T tokens) used the same FLOPs as Gopher (**O**, CLM-003).
- **Why they disagree** (several confounds, none sufficient alone):
  - Hoffmann et al.: a fixed token count and schedule across model sizes makes intermediate losses overestimates; a cosine cycle overshooting the run by more than 25% noticeably degrades performance (**O**, CLM-004).
  - Pearce and Song: counting non-embedding parameters at small scale gives a local exponent of 0.74–0.78 (**O**, CLM-006).
  - Porian et al.: last-layer FLOPs, warmup duration, and scale-dependent optimizer tuning; correcting them moved their exponent from ≈0.86 to ≈0.50. They report careful LR decay was *not* essential in their reproduction (**O**, CLM-007).
  - Besiroglu et al.: the Chinchilla Approach-3 fit itself had procedural errors; their refit gives $E=1.8172$, $A=482.01$, $B=2085.43$, $\alpha=0.3478$, $\beta=0.3658$, about 20 tokens per parameter (**O**, CLM-005).
- **Inference-aware allocation**: adding $2N$ FLOPs per inference token to the objective favors smaller models trained longer when inference demand is large (~1B requests). Sardana et al. report one case of a 16B model on 3.35T tokens replacing a 30B-Chinchilla-quality model at 17% lower cost (**O**, CLM-010).

**Mechanism Explanation:**
At fixed $C$, a larger $N$ lowers the $A/N^\alpha$ term but forces fewer tokens, which raises $B/D^\beta$. The optimum balances the marginal gains, which is why the exponents depend only on $\alpha$ and $\beta$. Anything that biases the measured loss at a given $(N,D)$ biases $\alpha$ and $\beta$, and therefore the allocation. A schedule not decayed at the measured checkpoint inflates loss for short runs of large models. Excluding embeddings changes $N$ most at small scale. Untuned small models look worse than they are. The allocation is a property of the measurement procedure as much as of the model.

**Quantitative Model / Derivation (D, CLM-008):**
Minimize $E + AN^{-\alpha} + B(C/6N)^{-\beta}$ over $N$:
$$N_{opt}=G\left(\tfrac{C}{6}\right)^{a},\quad D_{opt}=G^{-1}\left(\tfrac{C}{6}\right)^{b},\quad G=\left(\tfrac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad a=\tfrac{\beta}{\alpha+\beta},\ b=\tfrac{\alpha}{\alpha+\beta}.$$
Assumptions: $C=6ND$ exactly; single epoch over unique data; the fitted form holds at target scale.

**Worked Example (recomputed from published fits; $C=10^{21}$ FLOPs):**
1. $C/6 = 1.667\times10^{20}$ token-parameters.
2. *Hoffmann (rounded)*: $a = 0.28/0.62 = 0.4516$. $N\approx1.82\times10^9$, $D\approx9.1\times10^{10}$, ≈50 tokens/parameter, predicted $L\approx2.33$.
3. *Besiroglu refit*: $a\approx0.5126$. $N\approx2.78\times10^9$, $D\approx6.0\times10^{10}$, ≈22 tokens/parameter, predicted $L\approx2.31$.
4. *Chinchilla's own budget*: $6\times70\times10^9\times1.4\times10^{12}=5.88\times10^{23}$. The rounded fit puts the optimum at ≈32.5B/3.0T; the refit at ≈73B/1.34T, close to the actual run.
5. *Interpretation and limits*: the same data and the same functional form give a 1.5× different model size at $10^{21}$ because of the fit procedure alone. The "20 tokens per parameter" heuristic is a property of one refit, data set, and tokenizer, not a constant. The predicted losses differ by only ≈0.02, so the loss surface is flat near the optimum, and inference cost (CLM-010) can move the decision more than the fit error.

**Knowledge Check:**
1. Why does Kaplan's fixed $2.5\times10^5$-step cosine schedule bias the allocation toward larger models?
2. What two quantities fully determine the allocation exponent $a$ in the parametric model?
3. Under what serving forecast is training past the compute-optimal point the cheaper total-cost choice?

**Guided Practice:**
For $C\in\{10^{20},10^{22}\}$, compute $N_{opt}, D_{opt}$ under both parameter sets. Then add an inference load of $I$ tokens at $2N$ FLOPs each and find the $I$ at which a model half the compute-optimal size (trained to the same predicted loss) becomes cheaper in total FLOPs. State every assumption.

**Feedback Contract:**
- *Expected Evidence*: Both allocations with $G, a, b$ shown; predicted losses; the inference break-even $I$ with the equal-loss constraint solved, not assumed.
- *Common Failure*: Quoting "20 tokens per parameter" as a law, or counting $6N$ with total parameters in one place and non-embedding in another.
- *Diagnostic Hint*: Which parameter count did the fit use, and was each training run's schedule matched to its own length?
- *Concept to Revisit*: FLOP Accounting (Module 01, Lesson 1.6).

**Learning Outcome:**
Allocate a compute budget under a stated loss model, explain each published confound behind the Kaplan/Chinchilla disagreement, and include inference demand in the decision.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 25.2 — Fitting Scaling Laws and Knowing When They Fail

**Engineering Question:**
How accurate is a forecast from proxy runs, and which changes make it meaningless?

**Concepts & Definitions:**
- **Irreducible loss** $E$: the asymptote of the loss model. A pure power law $kN^{-p}$ has none.
- **Absolute relative error (ARE)**: $|L_{pred}-L_{obs}|/L_{obs}$ on a held-out larger run. Choshen et al. report ≈4% as the best typically achieved, initialization noise of up to 4% in loss, better fits from intermediate checkpoints and from dropping the first ~10B tokens, and about 5 models as a safe minimum (**O**, CLM-011).
- **Fit-procedure sensitivity**: the Chinchilla replication shows that loss averaging versus summing, rounding, and optimizer termination change the fitted parameters and their stated intervals (**O**, CLM-005).
- **Data-constrained regime**: up to 4 epochs of repetition gave negligible loss change versus unique data. Effective tokens follow $D' = U_D + U_D R^*_D(1-e^{-R_D/R^*_D})$ with $R^*_D\approx15.4$, beyond which returns diminish extremely fast (**O**, CLM-009). $R^*_D$ is a decay constant, not the point where repeats become worthless.
- **Downstream unpredictability**: in one meta-analysis, downstream-task scaling was predictable in only 39% of cases (**O**, CLM-012; FRONTIER). Forecast validation loss; check task metrics separately.
- **Proxy invariants**: data mixture, tokenizer, parametrization, schedule family, and unique-data regime must be the same in proxy and target.

**Mechanism Explanation:**
In log–log space, $E + AN^{-\alpha}$ is concave. A straight line fitted over a short range is steeper than the curve's tail and runs below the asymptote when extrapolated. The fit can look excellent in range and be badly wrong out of range. Seed noise sets a floor on what any fit can resolve. Changing an invariant changes $E$, $A$, or $\alpha$ themselves, so the proxy points no longer describe the target. Repetition replaces $D$ with a smaller $D'$, so a compute-optimal plan that assumed unique data overstates its token benefit.

**Quantitative Model / Derivation:**
- Forecast with interval: fit on $k\ge5$ runs; bootstrap over runs and seeds; report $L_{pred}$ with a prediction interval widened by the seed spread at the largest proxy.
- Pre-registered validity test (**H**, CLM-014): with all invariants held, the held-out loss at 4–10× proxy compute lies within the declared tolerance; changing one invariant pushes it outside.

**Worked Example (synthetic, noiseless; D, CLM-013):**
1. *Truth*: $L = 1.8 + 400/N^{0.34}$. Proxies at $N = 1,2,4,8,16\times10^7$ give $L = 3.468, 3.117, 2.841, 2.622, 2.450$.
2. *Pure power-law fit*: $L \approx 25.76\,N^{-0.1252}$. Maximum in-range residual 1.3%.
3. *Extrapolation*: at $N=10^9$ it predicts 1.923 versus a true 2.148 (−10.5%). At $10^{10}$ it predicts 1.441 versus 1.959 (−26.4%), below the irreducible loss.
4. *With an $E$ term misjudged by ±0.1*: predictions at $10^{10}$ are 1.907 or 2.016 (−2.6% / +2.9%).
5. *Repetition (D, from CLM-009's form)*: with $U_D = 100$B unique tokens, 400B trained tokens ($R_D=3$) give $D'\approx373$B (93%). 1.6T ($R_D=15$) give $D'\approx1.06$T (66%). 4T ($R_D=39$) give $D'\approx1.52$T (38%).
6. *Interpretation and limits*: in-range fit quality says almost nothing about extrapolation. The irreducible term matters more than the exponent. Repetition silently turns a planned token count into fewer effective tokens. All values are synthetic or recomputed.

**Knowledge Check:**
1. Why can a fit with 1% in-range residual be 26% wrong at 60× scale?
2. Why must proxy schedules be matched to each proxy's own token budget?
3. Which of these voids a forecast: changing the tokenizer, adding a seed, switching SP to μP, going from 2 to 20 epochs?

**Guided Practice:**
Given loss curves from six proxy runs (provided synthetic CSV with seed replicates), fit both the pure and the $E$-term forms, bootstrap a 90% prediction interval at 8× the largest proxy, and state whether a supplied "observed" target loss falsifies the fit.

**Feedback Contract:**
- *Expected Evidence*: Both fits with parameters, bootstrap method, prediction interval including seed spread, a list of held invariants, and the decision.
- *Common Failure*: Reporting $R^2$ in range as forecast quality, or fitting to intermediate checkpoints of runs whose schedules had not decayed.
- *Diagnostic Hint*: Plot residuals against $\log N$. Is there curvature the model cannot express?
- *Concept to Revisit*: Confidence Intervals and Sample Size (Module 00, Lesson 0.5).

**Learning Outcome:**
Fit and stress a scaling law, forecast a held-out run with an honest interval, and list the conditions that invalidate the forecast.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 25.3 — Parametrization and Hyperparameter Transfer: SP versus μP

**Engineering Question:**
Can a learning rate tuned on a 40M-parameter proxy be used unchanged on a 7B target?

**Concepts & Definitions:**
- **Standard parametrization (SP)**: one global learning rate and fan-in initialization for all matrices. Under SP the optimal learning rate shifts with width (**O**, CLM-015).
- **Maximal Update Parametrization (μP)**: per-layer scaling of initialization, multipliers, and Adam learning rate so that updates stay order-one as width grows. In Tensor Programs V's Table 3, Adam LR is 1 for input weights and biases and $1/\text{fan\_in}$ for hidden and output weights; init variance is $1/\text{fan\_in}$ for input and hidden and $1/\text{fan\_in}^2$ for output. The equivalent Table 8 form puts a $1/\text{fan\_in}$ multiplier on the output layer. Attention uses $q^\top k/d$ instead of $q^\top k/\sqrt d$ (**O**, CLM-016).
- **μTransfer**: tune on a narrow proxy, copy to the wide target. Yang et al. report outperforming published 6.7B GPT-3 numbers from a 40M proxy at a tuning cost of 7% of pretraining, and published BERT-large numbers from a 13M proxy at a tuning cost equal to one BERT-large pretraining. Optimization HPs, init, and multipliers transfer; regularization HPs such as dropout and weight decay are listed as not transferring (**O**, CLM-015).
- **SP can transfer too**: Everett et al., at up to 26.8B parameters, report that all parameterizations can transfer with per-layer LR prescriptions and that their SP prescription outperformed μP (**O**, CLM-017; FRONTIER).
- **Depth transfer**: scaling residual branches by $L^{-\alpha}$ with $\alpha=1$ (CompleteP) is reported to give depth-wise transfer over 2–128 layers, with 11.8–34.4% FLOP savings over μP at 1.9B (**O**, CLM-018; FRONTIER, author-reported).
- **Adam epsilon**: gradient RMS shrinks with scale, so a fixed $\epsilon$ (1e-8) can dominate the denominator. Wortsman et al. report that lowering it to 1e-15 improved loss for a 4.8B model at LR 0.3 while 1e-6 diverged; μParam stabilized the optimal LR but did not reduce LR sensitivity (**O**, CLM-021).

**Mechanism Explanation:**
Under SP with Adam, each hidden-weight entry moves by about the learning rate regardless of width, but a matrix–vector product sums $n$ such changes. The change in activations therefore grows with width $n$, and the stable learning rate falls roughly as $1/n$. μP divides the hidden learning rate by fan-in so that the activation change stays constant, which pins the optimum. The output layer needs an extra $1/\text{fan\_in}$ so logits do not blow up as width grows. Transfer is a statement about *optimum location*, not about loss: μP does not promise a better loss, only a learning rate that means the same thing at every width.

**Quantitative Model / Derivation:**
For a hidden matrix $W\in\mathbb{R}^{n\times n}$ updated by Adam with entries $\approx\pm\eta$ and input entries $x_j=\Theta(1)$, the coordinate change is $\Delta(Wx)_i=\sum_j\Delta W_{ij}x_j=\Theta(\eta\, n)$ when updates and inputs are correlated (as they are after training aligns them). Holding $\Delta(Wx)_i=\Theta(1)$ requires $\eta\propto1/n$, which is μP's hidden-layer rule.

**Worked Example (synthetic proxy-to-target transfer):**
- *Setup*: proxy width 256 with tuned global Adam LR $\eta^*=3\times10^{-3}$; target width 4096 (16× wider).
- *SP, copied LR*: per the derivation, the target's hidden activations move ≈16× more per step than at the proxy. If the proxy's optimum sits within 4× of its own divergence point (an assumption typical of tuned optima), the copied LR is past divergence.
- *SP, hand-scaled*: dividing by 16 gives $1.875\times10^{-4}$ for hidden weights, but that also shrinks embedding LR, which should not scale with width. This is why a single global SP LR cannot match per-layer needs.
- *μP*: write the hidden-layer LR as $\eta_{base}\cdot256/\text{fan\_in}$. At the target that equals $3\times10^{-3}\times256/4096=1.875\times10^{-4}$ for hidden weights, while input weights keep $3\times10^{-3}$ and the output layer gets the $1/\text{fan\_in}$ multiplier. One base LR now has the same meaning at both widths.
- *Interpretation and limits*: the factor-16 argument is an asymptotic order, not an exact constant. Whether the optimum drifts in practice at small widths is the LAB B hypothesis (**H**, CLM-019).

**Knowledge Check:**
1. Why does μP not guarantee a lower loss than a well-tuned SP run?
2. Which hyperparameters did Tensor Programs V find do not transfer, and why might that be?
3. What happens to Adam's update when gradient RMS falls below $\epsilon$?

**Guided Practice:**
Take a 4-layer decoder. Implement μP per Table 8 (output multiplier, per-group Adam LRs, $1/d$ attention). Write a coordinate check: train 3 steps at widths 64–1024 and plot the RMS change of each layer's output. Show the curves are flat under μP and growing under SP.

**Feedback Contract:**
- *Expected Evidence*: Per-group LR table, init variances, attention scale, and coordinate-check plots for SP and μP.
- *Common Failure*: Scaling only the hidden LR and forgetting the output multiplier, or leaving $1/\sqrt d$ attention, so logits still grow with width.
- *Diagnostic Hint*: Which layer's activation RMS changes with width in the coordinate check?
- *Concept to Revisit*: Adam States (Module 19, Lesson 19.2); Attention Scores (Module 01, Lesson 1.3).

**Learning Outcome:**
Implement μP, verify it with a coordinate check, and predict when a proxy-tuned learning rate will or will not transfer under SP.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 25.4 — Instabilities: Logit Growth, Loss Spikes, and Their Mitigations

**Engineering Question:**
Why does a run that trained fine for 100k steps suddenly diverge, and which signal would have warned you?

**Concepts & Definitions:**
- **Attention-logit growth**: attention logits grow until attention weights collapse to one-hot vectors (attention entropy collapse) (**O**, CLM-020). ViT-22B saw this around 8B parameters with logits exceeding 50,000 in magnitude (**O**, CLM-022).
- **QK-norm**: normalize queries and keys before the dot product (**O**, CLM-020, CLM-022). OLMo 2 applies RMSNorm to query and key projections (**O**, CLM-024).
- **Output-logit divergence**: the LM-head logits drift. Wortsman et al. report this in models without weight decay regardless of scale and resolved by **z-loss**, an auxiliary $k\log^2 Z$ with $k=10^{-4}$ (**O**, CLM-020). PaLM used the same auxiliary loss (**O**, CLM-023).
- **Logit soft-capping**: $z\leftarrow s\tanh(z/s)$, with $s=50$ for attention and $30$ for the final layer in Gemma 2 (**O**, CLM-026). Kimi K2 notes that $q\cdot k$ can still grow before capping, that QK-norm does not apply to MLA, and introduces QK-Clip (per-head rescaling of query/key weights, $\tau=100$). It reports 15.5T tokens with zero loss spike (**O**, CLM-027; FRONTIER, author-reported).
- **Loss spike**: a sudden loss jump that may or may not recover. PaLM saw ~20 spikes in its largest run despite clipping and handled them by restarting ~100 steps earlier and skipping ~200–500 batches, attributing spikes to data batches combined with a parameter state (**O**, CLM-023). OLMo 2 reports that filtering documents with 32+ repeated n-grams eliminates many spikes, alongside architectural changes (**O**, CLM-024). Takase et al. report that small sub-layer init plus embeddings with standard deviation near 1 avoided spikes in their runs (**O**, CLM-025).
- **LR sensitivity**: the expected deviation from optimal loss across three orders of magnitude of LR. Longer warmup and LR-independent weight decay reduce it, and depth raises it faster than width (**O**, CLM-020, CLM-021).

**Mechanism Explanation — Pinned OLMo-core Trace (O, CLM-028):**
Cross-entropy depends only on logit *differences*, so a constant shift of all output logits costs nothing and can drift. Attention logits are $q\cdot k$, which grows with the product of the norms, so modest norm growth produces large logit gaps. At OLMo-core commit `5f6f58a133e7ef577d596295f2c8db4651c27857` (static inspection, 2026-10-12):
1. `src/olmo_core/nn/functional/cross_entropy_loss.py::cross_entropy_loss` casts logits to float32, computes cross-entropy, and when `compute_z_loss` is set adds `z_loss_multiplier * logsumexp(logits)^2` averaged over non-ignored labels (default multiplier `1e-4`).
2. `src/olmo_core/optim/skip_step_optimizer.py::SkipStepOptimizer.get_step_factor` returns 0 (skip) when the latest loss (and grad norm, when tracked) exceeds the rolling mean by more than `sigma_factor` (default 6) standard deviations over a window of `rolling_interval_length` (default 128), active after 64 entries. It is computed as a tensor to avoid a host sync.
3. `src/olmo_core/nn/transformer/config.py::TransformerConfig.olmo2_7B` defaults to `qk_norm=True` with the `reordered_norm` block.
4. `src/scripts/train/OLMo2/OLMo2-7B.py::build_train_module_config` sets `z_loss_multiplier=1e-5`, `max_grad_norm=1.0`, zero weight decay on `embeddings.weight`, and `CosWithWarmup(warmup_steps=2000)`.

The OLMo 2 paper's Table 1 gives the z-loss weight as $10^{-5}$ while Section 3.3.3 says $10^{-4}$ (**O**, CLM-024). The script matches the table. Which value the published run used is `TODO_VERIFY`.

**Quantitative Model / Derivation (D, CLM-029):**
- $\text{CE}=\log Z - z_y$ is unchanged by $z\to z+c$, while $\log Z\to\log Z+c$.
- z-loss gradient: $\partial(k\log^2Z)/\partial z_j = 2k\log Z\,p_j$, summing to $2k\log Z$ per position. This is a restoring force on the free shift.
- Two-key attention with logit gap $\Delta$: weight $=1/(1+e^{-\Delta})$. Scaling $|q|$ and $|k|$ by $g$ scales $\Delta$ by $g^2$.

**Worked Example (synthetic):**
1. *z-loss scale*: with $\log Z = 30$ and $k=10^{-4}$, z-loss $=0.09$ per token, and the summed logit gradient is $0.006$, small next to a CE near 2 but always pointing toward $\log Z=0$. At $k=10^{-5}$ both are 10× smaller.
2. *Attention collapse*: $\Delta=2$ gives a top weight of 0.881. If query and key norms each grow 3×, $\Delta=18$ and the top weight is $1-1.52\times10^{-8}$: the head attends to a single key, and the gradient through the softmax nearly vanishes for all others.
3. *Skip-step trigger*: rolling loss mean 2.40, standard deviation 0.02, `sigma_factor` 6. The threshold is $2.40+0.12=2.52$. A step at loss 2.55 is skipped; a step at 2.50 is applied.
4. *Interpretation and limits*: these mechanisms explain *how* logits drift and how mitigations bound them. They do not predict *when* a given run diverges. Skip-step thresholds trade missed spikes against skipped legitimate steps after a distribution change.

**Knowledge Check:**
1. Why can output logits drift without the training loss noticing?
2. Why does soft-capping not stop $q\cdot k$ from growing, and what does QK-Clip do instead?
3. A spike recurs at the same data position after a rollback. What does that suggest, and what does it not prove?

**Guided Practice:**
In a small decoder at a deliberately high learning rate, log per-layer max attention logit, attention entropy, mean $\log Z$, loss, and grad norm every step. Identify which signal moves first before divergence. Then add QK-norm, then z-loss, and record which signal each removes. Trace the four OLMo-core symbols above by file and line.

**Feedback Contract:**
- *Expected Evidence*: Time-aligned plots of the five signals for baseline and each mitigation; the leading indicator named with its lead time; a source trace with commit, file, symbol, and what was statically inspected.
- *Common Failure*: Logging only loss and grad norm, so the cause is invisible until divergence; or claiming z-loss "fixes spikes" without showing $\log Z$ was the drifting quantity.
- *Diagnostic Hint*: Did $|\log Z|$ or max attention logit trend upward before the loss did?
- *Concept to Revisit*: Softmax and Logits (Module 01, Lesson 1.2); RMSNorm Numerics (Module 01, Lesson 1.5).

**Learning Outcome:**
Instrument a pretraining run for logit-growth instabilities, attribute a divergence to its mechanism, apply and verify the matching mitigation, and trace it in pinned source.

*(Effort: 50m instruction, 25m practice; source trace shared, 1.5h total)*

---

### Lesson 25.5 — Learning-Rate Schedules: Cosine, Warmup-Stable-Decay, and Decay to Zero

**Engineering Question:**
Which schedule lets you stop, extend, or compare runs without retraining from scratch?

**Concepts & Definitions:**
- **Warmup**: linear increase from ~0 to peak. Longer warmup reduces LR sensitivity (**O**, CLM-021).
- **Cosine**: $\eta(t)=\eta_{min}+(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))/2$, tied to a known length $T$. Overshooting $T$ by more than 25% degrades performance (**O**, CLM-004).
- **Warmup-Stable-Decay (WSD)**: warmup, constant peak, then a decreasing decay function over a short final phase. MiniCPM found 10% decay sufficient in its setup and uses WSD to reuse checkpoints for continued training and scaling studies (**O**, CLM-032). Hägele et al. report constant LR plus cooldown matches cosine, gains plateau near 20% cooldown, a (1-sqrt) shape beat linear, and reuse cuts scaling-sweep runs by about an order of magnitude (**O**, CLM-031).
- **Why loss drops in the decay**: the river-valley conjecture says the high-LR phase progresses along a valley floor while oscillating across it, and the decay removes the oscillation (**O**, CLM-033; FRONTIER). A convex-optimization bound reproduces the cooldown benefit (**O**, CLM-036; FRONTIER).
- **Decay floor**: Bergsma et al. report linear decay to zero beat cosine to 10% at compute-optimal data sizes, with a 610M model at 80 tokens/parameter beating 10×-decay at 200 tokens/parameter (**O**, CLM-034; abstract-level). OLMo 2 decays linearly to zero in mid-training (**O**, CLM-046).
- **Schedule-aware loss law**: $L(s)=L_0+A\,S_1^{-\alpha}-C\,S_2$, with $S_1$ the LR area and $S_2$ an annealing area (**O**, CLM-035; FRONTIER).

**Mechanism Explanation — Pinned Schedule Trace (O, CLM-037):**
At the same OLMo-core commit, `src/olmo_core/optim/scheduler.py`:
- `WSD.get_lr` returns `_linear_warmup` up to `warmup`, `initial_lr` in the stable phase, and `_linear_decay(initial_lr, t_max - current, decay, decay_min_lr)` in the last `decay` steps (default `decay_fraction=0.1`, `decay_min_lr=0.0`).
- `CosWithWarmup.get_lr` warms up linearly, then follows the cosine above with `eta_min = initial_lr * alpha_f` and `alpha_f` defaulting to 0.1, i.e. it decays to 10% of peak, not to zero.

A cosine checkpoint at step $t<T$ is a model at an intermediate LR, so its loss is not the loss of a $t$-step run. A WSD trunk checkpoint has never decayed, so its loss is also not final. Only a decayed checkpoint is a valid scaling-curve point. WSD makes producing one cheap: branch a short decay off the trunk.

**Quantitative Model / Derivation (D, CLM-038):**
Branch cost for budgets $T_1<\dots<T_k$ with decay fraction $f$: WSD costs $T_k(1-f)+\sum_i fT_i$ steps (the trunk plus each decay), versus $\sum_i T_i$ for independent cosine runs.

**Worked Example (synthetic run; peak $3\times10^{-4}$, 2,000 warmup steps, $T=100{,}000$):**
1. *LR at step 50,000*: cosine to 10% gives $3\times10^{-5}+2.7\times10^{-4}(1+\cos(\pi\cdot48{,}000/98{,}000))/2\approx1.69\times10^{-4}$. WSD gives $3\times10^{-4}$.
2. *LR area $S_1$*: summed over all steps, ≈16.5 (cosine) versus ≈28.2 (WSD with 10% linear decay).
3. *Budgets 20k, 60k, 100k*: three cosine runs cost 180k steps. One WSD trunk to 90k plus decays of 2k (from 18k), 6k (from 54k), and 10k (from 90k) costs 108k steps, a 40% saving.
4. *Interpretation and limits*: a larger $S_1$ does not by itself mean lower loss; the annealing term matters (CLM-035). The 40% depends on the budget set. Whether a WSD branch matches a matched cosine run within seed noise on your model is the LAB D hypothesis (**H**, CLM-049).

**Knowledge Check:**
1. Why is a WSD trunk checkpoint not a valid point on a scaling curve, and how do you make one?
2. What happens to a cosine run if you decide at 80% of $T$ to train 50% longer?
3. Why might decaying to zero beat decaying to 10% of peak?

**Guided Practice:**
Implement cosine and WSD from the formulas, check your values against `WSD.get_lr` and `CosWithWarmup.get_lr` semantics at five steps, and plan a WSD branching layout for four budgets with total-step cost and the decay length of each branch.

**Feedback Contract:**
- *Expected Evidence*: LR tables matching the pinned semantics (including the 10% cosine floor), the branching plan with cost, and which checkpoints are valid scaling points.
- *Common Failure*: Fitting a scaling law to undecayed checkpoints, or assuming the library cosine decays to zero.
- *Diagnostic Hint*: What was the LR at the checkpoint you are comparing, and what was it at the end of the competing run?
- *Concept to Revisit*: Schedule confound (Lesson 25.1).

**Learning Outcome:**
Choose and implement a schedule, plan budget-agnostic training with WSD, and identify which checkpoints are comparable.

*(Effort: 50m instruction, 25m practice; source trace shared with Lesson 25.4)*

---

### Lesson 25.6 — Data Mixtures and Checkpoint Averaging

**Engineering Question:**
How do you choose domain weights with small runs, and how do you know the choice survives scaling?

**Concepts & Definitions:**
- **Data mixture**: domain weights $h$ over sources, applied after Module 08's lineage and dedup. It is a hyperparameter of the run, so it must be held fixed for any scaling forecast (Lesson 25.2).
- **Proxy mixture search**:
  - DoReMi: a 280M proxy with Group DRO over domains sets weights for an 8B model; on The Pile it reports +6.5 points average few-shot accuracy and 2.6× fewer steps to baseline accuracy (**O**, CLM-039).
  - RegMix: 512 models of 1M parameters for 1B tokens fit a mixture-to-performance regression; the chosen mixture trained a 1B model for 25B tokens that was best among 64 candidates (**O**, CLM-041).
- **Mixing laws**: Ye et al. predict loss from proportions and nest this with step and size laws; a 1B/100B-token run matched default-mixture performance at 48% more steps (**O**, CLM-040). Shukor et al. predict loss from $(N, D, h)$ and report extrapolation from small runs (**O**, CLM-042). The functional forms were not read (`TODO_VERIFY`).
- **Transfer risk**: AutoScale reports that mixtures that win at small scale may not keep their advantage at larger scale (**O**, CLM-043; contradicts the scale-invariance assumed in CLM-041).
- **Checkpoint averaging**:
  - LAWA: averaging trajectory checkpoints helps early in training, more so at high LR and with wider spacing (**O**, CLM-045).
  - Pre-trained Model Averaging: averaging constant-LR-phase checkpoints under WSD can match an annealed model and predict annealing, with optimal intervals scaling with model size (**O**, CLM-044; FRONTIER).
  - Souping: OLMo 2 averages checkpoints annealed on different permutations of the same mix and reports equal or better performance than any single run (**O**, CLM-046).
  - Same-basin precondition: model soups average fine-tuned models from one pretrained initialization that often lie in a single low-error basin (**O**, CLM-047).

**Mechanism Explanation:**
A mixture changes which loss the model is minimizing. The target metric (a weighted held-out loss or task suite) then depends on per-domain losses, each with its own scaling. If domains scale at different rates, the optimal mixture moves with $N$ and $D$, which is the AutoScale failure. Averaging works for the same reason decay works in the river-valley picture: checkpoints oscillating across the valley average to a point nearer its floor. Averaging points from different basins can land on a ridge, so the precondition is a shared trajectory or initialization.

**Quantitative Model / Derivation (D, CLM-048):**
- Mixture objective: $L_{tgt}(h)=\sum_d w_d L_d(h_d)$ subject to $\sum_d h_d=1$.
- Averaging in a toy valley $L=f(x)+5y^2$: averaging $y$ reduces the mountain term; averaging $x$ lags the river progress.

**Worked Example (synthetic):**
1. *Mixture*: $L_{web}(h)=2.0+0.30h^{-0.5}$, $L_{code}(h)=1.2+0.10h^{-0.5}$, target $=0.7L_{web}+0.3L_{code}$. Grid over code share: 0.15 → 2.0652, 0.20 → 2.0619, 0.25 → 2.0625, 0.30 → 2.0658. The optimum is ≈0.20, but 0.15–0.30 all lie within 0.004. If seed spread is 0.01, these mixtures cannot be ranked.
2. *Averaging*: three checkpoints at $(x,y)=(1.0,+0.30),(1.1,-0.25),(1.2,+0.20)$. The last has mountain term $5\cdot0.20^2=0.20$. The average $(1.10, 0.083)$ has $0.035$ but sits 0.10 behind on the river.
3. *Interpretation and limits*: the flat region means mixture tuning is mostly about avoiding bad corners, not finding a sharp optimum. Averaging trades river progress for a smaller oscillation term, which is why it pays most at high LR. Real domains transfer to each other; this toy does not.

**Knowledge Check:**
1. Why must a mixture be fixed before fitting a size-scaling law?
2. What evidence would show a proxy-optimal mixture did not transfer?
3. Why can averaging two independently pretrained models fail when averaging two checkpoints of one run works?

**Guided Practice:**
With per-domain loss curves from proxy runs at two sizes (provided), fit a per-domain power law in $h$, choose a mixture at each size, and test whether the rank of three candidate mixtures is preserved beyond seed spread. Then average the last $k\in\{2,4,8\}$ trunk checkpoints of a WSD run and compare with the decayed branch.

**Feedback Contract:**
- *Expected Evidence*: Fitted per-domain curves, the chosen mixture per scale, rank-preservation test with seed spread, and averaged versus decayed losses.
- *Common Failure*: Declaring a mixture winner from a gap smaller than seed noise, or averaging checkpoints across a learning-rate decay boundary without noting it.
- *Diagnostic Hint*: Is your mixture gap larger than twice the seed spread at *both* scales?
- *Concept to Revisit*: Data Lineage and Deduplication (Module 08, Lessons 8.1, 8.4).

**Learning Outcome:**
Optimize a data mixture with proxies, test its transfer across scale, and apply checkpoint averaging with a stated same-basin assumption.

*(Effort: 50m instruction, 25m practice)*

---

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020; $6N$ rule and power-law forms. Its allocation exponent is LEGACY.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022; parametric loss and Chinchilla allocation.
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) — Besiroglu et al., 2024; fit-procedure sensitivity.
- [Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907) — Pearce and Song, TMLR 2024.
- [Resolving Discrepancies in Compute-Optimal Scaling](https://arxiv.org/abs/2406.19146) — Porian et al., NeurIPS 2024.
- [Scaling Vision Transformers to 22B](https://arxiv.org/abs/2302.05442) — Dehghani et al., 2023; origin of QK-norm at scale.
- [PaLM](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022; z-loss and spike rollback.
- [DoReMi](https://arxiv.org/abs/2305.10429) — Xie et al., NeurIPS 2023; [Model soups](https://arxiv.org/abs/2203.05482) — Wortsman et al., ICML 2022 (cross-reference only).

**CURRENT DEFAULT** (scoped as engineering baselines, not claims of universal adoption): inference-aware allocation ([Sardana et al.](https://arxiv.org/abs/2401.00448), ICML 2024); repetition-aware data planning ([Muennighoff et al.](https://arxiv.org/abs/2305.16264), NeurIPS 2023); forecast tolerances from [Choshen et al.](https://arxiv.org/abs/2410.11840); width-wise HP transfer ([Tensor Programs V](https://arxiv.org/abs/2203.03466)); QK-norm, z-loss, and small Adam epsilon ([Wortsman et al.](https://arxiv.org/abs/2309.14322), 2023; [OLMo 2](https://arxiv.org/abs/2501.00656)); WSD/constant-plus-cooldown schedules ([MiniCPM](https://arxiv.org/abs/2404.06395); [Hägele et al.](https://arxiv.org/abs/2405.18392), NeurIPS 2024); annealing plus souping (OLMo 2).

**WORKLOAD-DEPENDENT:** logit soft-capping ([Gemma 2](https://arxiv.org/abs/2408.00118)); embedding-scale spike prevention ([Spike No More](https://arxiv.org/abs/2312.16903), COLM 2025); trajectory averaging ([LAWA](https://arxiv.org/abs/2306.03241), workshop).

**FRONTIER:** per-layer SP transfer and Adam-atan2 ([Everett et al.](https://arxiv.org/abs/2407.05872), ICML 2024); depth transfer ([CompleteP](https://arxiv.org/abs/2505.01618), NeurIPS 2025); QK-Clip ([Kimi K2](https://arxiv.org/abs/2507.20534), 2025); downstream unpredictability ([Lourie et al.](https://arxiv.org/abs/2507.00885), EMNLP Findings 2025); river-valley WSD theory ([Wen et al.](https://arxiv.org/abs/2410.05192)); convex-bound schedule theory ([Schaipp et al.](https://arxiv.org/abs/2501.18965), 2025); decay to zero ([Straight to Zero](https://arxiv.org/abs/2502.15938), ICLR 2025); LR-annealing loss law ([Tissue et al.](https://arxiv.org/abs/2408.11029)); mixing laws ([Ye et al.](https://arxiv.org/abs/2403.16952), ICLR 2025; [Shukor et al.](https://arxiv.org/abs/2507.09404), 2025); regression mixtures ([RegMix](https://arxiv.org/abs/2407.01492), ICLR 2025); scale-dependent mixtures ([AutoScale](https://arxiv.org/abs/2407.20177), COLM 2025); pretraining model merging ([Li et al.](https://arxiv.org/abs/2505.12082), 2025).

**LEGACY / INSUFFICIENT:** Kaplan's $C^{0.73}$ allocation as a planning rule; fitting scaling laws to intermediate checkpoints of a single long cosine run; a global SP learning rate copied from a narrow proxy; logging only loss and grad norm; fixed Adam $\epsilon=10^{-8}$ at scale without checking gradient RMS; mixture winners declared inside seed noise.

**PRODUCTION SOURCE TRACE**
- Repository: `allenai/OLMo-core`
- Revision: `5f6f58a133e7ef577d596295f2c8db4651c27857` (committed 2026-10-03)
- Verified: 2026-10-12; static inspection only.
- Files/symbols: `src/olmo_core/nn/functional/cross_entropy_loss.py::cross_entropy_loss`; `src/olmo_core/optim/skip_step_optimizer.py::SkipStepOptimizer.get_step_factor`; `src/olmo_core/optim/scheduler.py::{WSD.get_lr, CosWithWarmup.get_lr, _linear_warmup, _linear_decay}`; `src/olmo_core/nn/transformer/config.py::TransformerConfig.olmo2_7B`; `src/scripts/train/OLMo2/OLMo2-7B.py::build_train_module_config`.
- Not traced: the attention module's QK-norm application and the train-module call path into `cross_entropy_loss`.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` and record a run manifest:
- model shape (width, depth, heads), total and non-embedding parameter counts;
- parametrization (SP or μP, with per-group LR and init table);
- optimizer with $\beta_1,\beta_2,\epsilon$, weight decay, and clipping;
- schedule type, warmup, decay length and floor;
- data snapshot ID, mixture weights, tokenizer, unique tokens and epochs;
- stability options (QK-norm, z-loss weight, soft-cap);
- seeds and hardware tier.

Labs use public text or labeled synthetic data. Hardware tiers are `CPU`, `COLAB-GPU`, or `TRACE`; no local GPU is assumed. Lab code is not provided in this revision.

### LAB A — Scaling-Law Forecast and Its Failure
- **Hardware**: `CPU` for fitting on provided loss logs; `COLAB-GPU` optional for generating proxy runs (≤ ~20M parameters).
- **Objective**: Fit $E + A/N^\alpha$ and $E+A/N^\alpha+B/D^\beta$ on ≥5 proxy runs with schedules matched to each budget, forecast a held-out larger run with a bootstrap prediction interval, and compute compute-optimal allocations under both published parameter sets.
- **Pre-Registered Hypothesis**: CLM-014, with the tolerance and invariants declared before the held-out run is revealed.
- **Independent Variables**: Functional form (pure power law vs. with $E$), number of proxy runs, parameter-counting convention, schedule matching (matched vs. intermediate checkpoints of one long cosine run), unique-token count.
- **Dependent Variables**: Fitted parameters with intervals, ARE at the held-out scale, allocation exponent $a$, effective tokens $D'$.
- **Break & Falsify**: Refit using intermediate checkpoints of one long cosine run and report the change in $a$. Refit with non-embedding counts. Repeat a 1/4-size unique-data subset for 4 and 16 epochs and test whether loss follows $D'$. The forecast hypothesis is falsified if, with invariants held, the held-out loss lies outside the prediction interval.
- **Alignment**: Lessons 25.1–25.2.
- **Effort Estimate**: 3h total.

### LAB B — SP versus μP Learning-Rate Transfer
- **Hardware**: `COLAB-GPU`.
- **Objective**: Train 4-layer decoders at widths 128, 256, 512, 1024 under SP and μP over a ≥7-point log-spaced LR grid with 2 seeds, locate the LR optimum per width, and measure the loss lost by transferring the smallest width's LR.
- **Pre-Registered Hypothesis**: CLM-019.
- **Independent Variables**: Parametrization, width, LR, Adam $\epsilon$ ($10^{-8}$ vs. $10^{-15}$); depth (4 vs. 16) as a secondary axis.
- **Dependent Variables**: Final validation loss per (width, LR), fitted argmin LR, transfer loss gap, coordinate-check activation RMS change.
- **Break & Falsify**: Remove only the output multiplier from μP and show the optimum drifts again. Change depth at fixed width and test whether μP's optimum moves (it is not designed for depth; CLM-018). The hypothesis is falsified if μP's argmin moves more than one grid step across width, or SP's does not move, with seed spread below the shift.
- **Alignment**: Lesson 25.3.
- **Effort Estimate**: 3.5h total.

### LAB C — Inducing and Fixing Logit-Growth Instabilities
- **Hardware**: `COLAB-GPU` for training; `TRACE` for the OLMo-core source trace.
- **Objective**: Train a small decoder without weight decay across an LR grid spanning three orders of magnitude, log max attention logit, attention entropy, mean $\log Z$, loss, and grad norm, and compare baseline, QK-norm, z-loss ($10^{-4}$ and $10^{-5}$), and both. Implement a skip-step rule matching `SkipStepOptimizer.get_step_factor` and complete the pinned source trace.
- **Pre-Registered Hypothesis**: CLM-030.
- **Independent Variables**: LR, QK-norm on/off, z-loss weight, weight decay on/off, warmup length, Adam $\epsilon$.
- **Dependent Variables**: LR sensitivity, leading indicator and its lead time before divergence, final loss, number of skipped steps.
- **Break & Falsify**: Inject a batch of a repeated n-gram sequence at a fixed step and test whether it triggers a spike, whether skip-step catches it, and whether a rollback-and-skip removes it. The hypothesis is falsified if divergence occurs without preceding logit or $\log Z$ growth, or a mitigation does not reduce LR sensitivity beyond seed spread.
- **Alignment**: Lesson 25.4 and Incident 25.1.
- **Effort Estimate**: 3.5h total (source trace counted separately, 1.5h).

### LAB D — Schedules, Mixtures, and Checkpoint Averaging
- **Hardware**: `COLAB-GPU`.
- **Objective**: Compare cosine runs at three budgets against one WSD trunk with three branched decays; average trunk checkpoints and compare with the decayed branch; choose a three-domain mixture at the smallest scale and test rank preservation at 8× compute.
- **Pre-Registered Hypothesis**: CLM-049.
- **Independent Variables**: Schedule (cosine to 10%, cosine to 0, WSD linear, WSD 1-sqrt), decay fraction, averaging window and spacing, mixture weights, scale.
- **Dependent Variables**: Final validation loss per budget with seed spread, compute spent, averaged-checkpoint loss, per-domain and target loss per mixture and scale.
- **Break & Falsify**: Run a cosine schedule for 30% past its $T$ and show the degradation. Average checkpoints from two different seeds and show the failure. The hypothesis is falsified if WSD is worse than cosine beyond seed spread at a budget, averaging does not beat the last trunk checkpoint, or a mixture gap larger than 2× seed spread reverses between scales.
- **Alignment**: Lessons 25.5–25.6.
- **Effort Estimate**: 3h total.

---

## 07 Break / Incident Scenarios

### Incident 25.1 — The Forecast That Missed

- **Incident Symptoms**:
  - A team forecast a 7B run's final loss from six proxy runs (25M–400M) and committed compute. At 60% of tokens, loss is 6% above the forecast curve.
  - Two loss spikes occurred at about 35% and 52% of training. The first recovered within ~200 steps; the second recovered more slowly. Grad norm rose sharply at both.
  - Since the proxies were run, the team (a) switched the mixture to add 20% more code, (b) moved the tokenizer to a larger vocabulary, (c) changed from cosine to WSD for the target only, (d) disabled z-loss "for speed", and (e) widened the model from the proxy's aspect ratio without μP.
  - Proxy losses were taken at intermediate checkpoints of each proxy's cosine schedule.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*:
     - forecast invalid because proxies were not decayed (schedule confound);
     - forecast invalid because the target is in WSD's stable phase, which runs above a decayed curve by design;
     - invariant change: mixture or tokenizer;
     - learning rate not transferred (SP, wider target);
     - output-logit drift after z-loss removal;
     - attention-logit growth;
     - data-triggered spikes;
     - genuine extrapolation error within seed noise.
  2. *Rank Initial Plausibility*: Use the change list, but note that (c) alone predicts a gap above a decayed forecast without any defect, and that several changes interact.
  3. *Identify Missing Evidence*: Proxy and target manifests; LR at each compared checkpoint; per-domain loss; max attention logit and $\log Z$ histories; data positions of spike batches; seed spread at the largest proxy; whether the forecast curve represents decayed or undecayed loss.
  4. *Design Discriminating Tests*:
     - Branch a short decay off the current trunk and compare the decayed loss with the forecast.
     - Refit the proxies using decayed endpoints only.
     - Re-run the two largest proxies with the new tokenizer and mixture.
     - Plot $\log Z$ and max attention logit around both spikes; replay the spike batches from the pre-spike checkpoint with and without z-loss.
     - Run a short LR sweep at a mid-size width to locate the target's LR optimum under SP.
  5. *Execute Causal Diagnosis*: Report how much of the 6% each test explains, which causes interact, and what remains unexplained.
  6. *Prescribe Mitigation and Prevention*: Re-enable z-loss and add QK-norm if logits drifted; rollback-and-skip for data-triggered spikes; refit with decayed, invariant-matched proxies; adopt μP or per-layer LRs for future proxy work; manifest checks that block invariant changes between proxy and target.
  7. *Remeasure*: Decayed-branch loss versus the refit forecast with its interval, $\log Z$ and attention-logit histories, and the spike count over the next 10% of tokens.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Planning a Domain-Specialized Base Model

A company will pretrain a base model for code and technical documentation.

Constraints:
- **Compute.** $10^{23}$ training FLOPs.
- **Data.** 600B unique tokens after Module 08 dedup and decontamination: 250B code, 350B web/technical text.
- **Serving.** Expected 2×10^12 inference tokens over the model's lifetime.
- **Proxy budget.** 3% of the target compute for all proxy work.
- **Architecture.** Standard attention (not MLA); the team uses AdamW.
- **Reporting.** Leadership wants the forecast loss "with a guarantee."

**Required Deliverables**:
1. Allocation under both the Hoffmann and Besiroglu fits, then an inference-aware revision, with every assumption stated. Reference check: at $10^{23}$ the fits give ≈14.6B/1.14T and ≈29.4B/0.57T respectively.
2. Repetition plan: epochs per domain implied by your token count and the effective-token penalty from the data-constrained law.
3. Proxy plan within the 3% budget: sizes, seeds, matched schedules (or WSD branches), invariants held, and the forecast tolerance with its basis.
4. Parametrization and LR plan: μP or SP with per-layer LRs, the proxy sweep, and Adam $\epsilon$.
5. Stability stack and monitoring: QK-norm, z-loss weight, warmup, the signals logged, skip-step or rollback policy, with the pinned OLMo-core trace as reference.
6. Schedule: cosine or WSD, decay length and floor, and which checkpoints are valid forecast points.
7. Mixture: proxy search method, rank-preservation test across scale, and the decision rule if the ranking does not transfer.
8. A response to "guarantee": the forecast interval, the conditions that void it, and the monitoring that would detect a miss early.
9. Diagnosis and remediation for Incident 25.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace z-loss computation, skip-step decision, the OLMo 2 model config's QK-norm and norm-placement defaults, the OLMo 2 7B training script's z-loss weight and schedule, and `WSD.get_lr` / `CosWithWarmup.get_lr` at the pinned OLMo-core commit. State what was statically inspected and what, if anything, was executed, and list the call paths not traced.

### Rubric Dimensions
- **Allocation Reasoning**: *Insufficient* quotes a tokens-per-parameter rule. *Competent* derives allocation from a stated loss model and explains at least two confounds. *Strong* compares fits, adds inference cost, and shows how flat the loss is near the optimum.
- **Forecast Validity**: *Insufficient* reports an in-range fit. *Competent* forecasts with an interval from matched-schedule proxies. *Strong* lists invariants, quantifies seed noise, accounts for repetition, and names the observation that would void the forecast.
- **Parametrization**: *Insufficient* copies a proxy LR. *Competent* implements μP with a coordinate check. *Strong* shows SP drift, explains depth limits, and justifies μP versus per-layer SP.
- **Stability Diagnosis**: *Insufficient* lowers the LR after divergence. *Competent* logs logit signals and applies a matching mitigation. *Strong* identifies the leading indicator, separates logit-growth from data-triggered spikes, and ties mitigations to pinned source.
- **Schedules, Mixtures, Averaging**: *Insufficient* compares undecayed checkpoints or declares mixture winners inside noise. *Competent* uses decayed comparison points and seed spread. *Strong* plans WSD branches by cost, tests mixture transfer across scale, and states the basin assumption for averaging.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Compute-optimal allocation | 25.1 | LAB A | Deliverable 1 | Allocation table under both fits; inference-aware revision |
| Scaling-law forecast and validity | 25.2 | LAB A | Deliverables 2, 3, 8; Incident steps 1–4 | Fits, prediction interval, invariant list, repetition analysis |
| HP transfer (SP vs μP) | 25.3 | LAB B | Deliverable 4; Incident step 4 | LR-optimum plots per width; coordinate check |
| Instability diagnosis and source trace | 25.4 | LAB C | Deliverable 5; Section 09 artifact; Incident steps 3–7 | Logit/log Z traces, LR sensitivity, OLMo-core trace |
| Schedule design | 25.5 | LAB D | Deliverable 6; Incident step 4 | Branch plan with cost; cosine vs WSD losses |
| Mixture optimization and averaging | 25.6 | LAB D | Deliverable 7 | Rank-preservation test; averaged vs decayed losses |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 25 must be able to:
1. Allocate a compute budget between parameters and tokens under a stated loss model, including inference demand, and explain why Kaplan and Chinchilla disagree.
2. Fit a scaling law on proxies, forecast a held-out run with an interval, and state the conditions that invalidate the forecast, including data repetition.
3. Implement μP, verify it, and predict when a learning rate will not transfer under SP.
4. Instrument, diagnose, and mitigate attention-logit growth, output-logit drift, and loss spikes, and trace the mitigations in pinned source.
5. Choose cosine or WSD with its budget consequences and identify which checkpoints are comparable.
6. Optimize a data mixture on proxies, test its transfer across scale, and use checkpoint averaging with a stated basin assumption.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: A forecast from small runs is valid only while the procedure that produced the small runs (counting, schedule, parametrization, data, tokenizer, stability stack) is the procedure of the large run. Every published disagreement in this module is a broken invariant.
- **The Planning Path**: budget and data → allocation under a fitted loss → proxy sweep with matched schedules and μP → forecast with interval and voiding conditions → stable run with logit monitoring → decay and averaging → compare decayed loss with forecast.
- Instabilities are logit growth until shown otherwise, but data-triggered spikes are a separate class. Schedules decide which checkpoints mean anything. Mixtures are hyperparameters that may not transfer. Module 26 takes the chosen configuration and makes it fit and run across many devices; Module 19 adapts the result.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
