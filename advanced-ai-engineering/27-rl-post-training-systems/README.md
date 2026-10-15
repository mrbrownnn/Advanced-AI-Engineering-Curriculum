# Module 27 — RL Post-Training Systems

## 00 Why This Module Exists

An RL post-training run is two programs that disagree. An inference engine samples responses, a trainer computes gradients on them, and a reward function (a learned model or a checker) scores them. The loss is an estimator built from those three pieces. Every engineering choice changes what it estimates: how advantages are baselined, how token losses are aggregated, how far the ratio may move, when weights are copied back to the sampler, and how stale or numerically mismatched the samples are allowed to be. A rising reward curve can mean the policy improved, the checker has a hole, entropy is collapsing toward a narrow mode, or the sampler and trainer are optimizing slightly different policies. Telling these apart is measurement work, not intuition.

```text
prompts --> ROLLOUT ENGINE (inference, policy version v_k, sampler numerics)
               |  responses + rollout log-probs (long-tail lengths)
               v
           REWARD (rule-based verifier | reward model)  --> audit (independent checker / labels)
               |
               v
           TRAINER: old log-probs (recompute or reuse) -> advantages (GAE | group) ->
                    ratio vs old, clip, IS correction vs rollout -> aggregate -> update -> v_{k+1}
               |
               +--> WEIGHT SYNC back to rollout engine (colocated reshard | network push)
                    sync or async (staleness bound)
```

Module 19 (Lesson 19.5) introduced RLHF, DPO, GRPO, the group-normalized advantage, the Dr. GRPO and DAPO normalization changes, and KL-free recipes. Module 06 (Lessons 6.2, 6.4) introduced pass@k and verifier gaming, and Module 15 (Lesson 15.3) treats graders as measurement instruments. This module extends those, it does not reteach them. It owns policy-gradient estimator design as an engineer's choice, RLVR verifier design and its error rates, reward hacking and over-optimization, and RL as a distributed system: rollout/trainer split, colocation versus disaggregation, weight synchronization, asynchronous training and staleness, long-tail rollouts, train/inference log-probability mismatch and its correction, and step-time accounting. Module 19 keeps the adapt-versus-retrieve decision and SFT/DPO mechanics. Serving internals of the rollout engine stay in Modules 03–05 and 20. The numerical root cause of sampler/trainer mismatch (kernels, reduction order, batch invariance) belongs to the planned Module 30 and is referenced only.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Run an RL post-training loop whose reward gains are real, whose estimator does what you think it does, and whose GPUs are not idle waiting for the longest response, and diagnose it when reward, entropy, length, or throughput move in unexpected directions.
- **What You Will Do**: Compute advantages, ratios, clips, and aggregations by hand and in code; reproduce entropy dynamics on a toy policy; measure a verifier's false-negative and false-positive rates and plant an exploitable one; detect hacking with an independent audit; build a step-time and staleness simulator; trace a pinned verl commit from rollout to weight sync; and measure sampler/trainer log-probability mismatch and truncated-importance-sampling corrections.
- **Environment**: Python 3.10+, PyTorch and NumPy on CPU for LAB A and LAB C; a Colab-class GPU for small-model GRPO in LAB B and the optional mismatch measurement in LAB D; a pinned verl checkout for source tracing. No local GPU is assumed.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and measurement-dependent hypotheses (**H**) separate. Worked-example inputs are labeled synthetic. Author-reported speedups and accuracy numbers are reported with their setup and are not general constants.

## 01 Baseline Assumptions

- Module 19: RLHF, reward models, DPO, GRPO objective and group advantage, Dr. GRPO and DAPO normalization, KL-free recipes, KL-to-base monitoring (Lesson 19.5); release gates for adapted models (Lesson 19.6).
- Module 06: pass@k and sampling (Lesson 6.2); outcome, process, and selection verifiers and verifier gaming (Lesson 6.4).
- Module 15: graders as measurement instruments (Lesson 15.3); uncertainty and release gates (Lesson 15.4).
- Module 04: continuous batching (Lesson 4.2) and saturation behavior (Lesson 4.4). Module 03: KV state (Lesson 3.1).
- Module 05: floating-point formats trade precision against range (Lesson 5.3).
- Module 20: collectives and communication cost (Lesson 20.2).
- Not assumed, taught here: importance ratios and PPO clipping, GAE, loss aggregation modes, entropy dynamics, RLVR, verifier error rates, over-optimization, rollout/trainer placement, weight sync, staleness, log-probability mismatch, truncated importance sampling, and effective sample size.

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
  instruction: 5h        # 6 lessons x 50 min
  guided_practice: 2h    # 6 lessons x 20 min
  labs: 13h              # LAB A 3h + LAB B 3.5h + LAB C 3h + LAB D 3.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 27.1 0.5h
  source_trace: 2h       # counted once; shared by Lessons 27.5-27.6 and LAB D
  total: 25h
```

The learner must write down what the loss estimates before trusting a curve, measure a verifier before optimizing against it, treat KL from the initial policy as a spent budget, account for every second of an RL step, and quantify how off-policy each batch is, from staleness and from sampler/trainer numerics, and what correction was applied.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Estimator (advantage: GAE or group baseline; ratio against a data policy; clip bounds; token- versus sequence-level ratio; loss aggregation) $\to$ Exploration dynamics (entropy, clip-higher, zero-variance groups) $\to$ Reward (RLVR checker or reward model; false-negative and false-positive rates; spurious-reward controls) $\to$ Over-optimization (KL budget, KL placement, independent audit) $\to$ System (rollout engine versus trainer; colocated versus disaggregated; weight sync; long-tail step time) $\to$ Off-policyness (staleness from asynchrony; log-probability mismatch from numerics; three-policy view; truncated/masked IS; ESS). Cross-cutting: every knob is a config field in a real framework, and its default is not the run's value until the resolved config says so.

## 04 Lessons

### Lesson 27.1 — The Estimator: Ratios, Clipping, GAE, and Group Baselines

**Engineering Question:**
What gradient does my RL loss actually compute, and when do the ratio and clip change it?

**Concepts & Definitions:**
- **Importance ratio**: $r_t(\theta) = \pi_\theta(o_t|\cdot)/\pi_{old}(o_t|\cdot)$, the ratio between the policy being updated and the policy whose log-probabilities were recorded for the data.
- **PPO clip**: $L^{CLIP} = \mathbb{E}_t[\min(r_t A_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$, designed so that one batch can support multiple epochs of minibatch updates (**O**, CLM-001). Clipping removes the incentive to push the ratio further in the advantage's direction; it is not a hard constraint.
- **When the clip is inert**: with one gradient step per batch, old log-probs recomputed by the same trainer, and deterministic forward passes, $r=1$ and the PPO gradient equals the plain policy gradient. Clipping matters only with several minibatches or epochs per batch, or with stale or mismatched data (**D**, CLM-043).
- **GAE**: $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ with $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. $\lambda=0$ is the one-step TD residual (lower variance, biased unless $V$ is exact); $\lambda=1$ is the return minus $V$ (no bias from $V$, higher variance) (**O**, CLM-002). It needs a learned critic.
- **Group baseline**: GRPO-style estimators baseline each response against its siblings for the same prompt (Module 19, Lesson 19.5). REINFORCE-style variants with leave-one-out baselines (RLOO) were reported to outperform PPO in RLHF settings by Ahmadian et al. (**O**, CLM-042).
- **Implementation detail matters**: `torch.std` defaults to Bessel's correction (divide by $N-1$) (**O**, CLM-004), so implementations that call it produce smaller advantages than the population-std formula: $\pm0.866$ instead of $\pm1$ for rewards $[1,0,0,1]$ (**D**, CLM-044).

**Mechanism Explanation — Pinned verl Estimators (O, CLM-003, CLM-005):**
At `volcengine/verl` commit `fbb4b3a8bf636f290c9c59fc346f756849e9c241` (static inspection, verified 2026-10-12):
1. `verl/trainer/ppo/core_algos.py::compute_gae_advantage_return` runs the reversed-time recursion `delta = r_t + gamma*nextvalues - values_t`, `lastgaelam = delta + gamma*lam*lastgaelam`, skips masked positions, then whitens advantages over the response mask (`masked_whiten`).
2. `compute_grpo_outcome_advantage` sums token rewards into one score per response, groups by prompt index, uses `torch.mean` and `torch.std` per group (mean 0, std 1 for a single-sample group), and returns `(score - mean)/(std + epsilon)` with `epsilon=1e-6`, or `score - mean` when `norm_adv_by_std_in_grpo` is false (documented as Dr. GRPO). The scalar is broadcast to every response token.
3. `compute_policy_loss_vanilla` computes `ratio = exp(clamp(log_prob - old_log_prob, -20, 20))`, clips with separate `clip_ratio_low` and `clip_ratio_high`, and applies a dual-clip bound `clip_ratio_c` (default 3.0) when the advantage is negative.

State → operation → state: (rewards, values or group, masks) → advantage per token → (ratio, clip) per token → per-token loss matrix → aggregation (Lesson 27.2) → scalar loss → update.

**Quantitative Model / Derivation:**
- Per-token clipped loss (minimized): $\ell_t = \max(-A_t r_t,\ -A_t\,\mathrm{clip}(r_t,1-\epsilon_l,1+\epsilon_h))$; for $A_t<0$, also $\ell_t=\min(\ell_t, -A_t c)$ (dual clip).
- Gradient through token $t$ is zero when $A_t>0, r_t>1+\epsilon_h$ or $A_t<0, r_t<1-\epsilon_l$.
- Bias–variance: GAE's $\lambda$ interpolates TD(0) and Monte-Carlo; group baselines need no critic but need $G$ samples per prompt and give one scalar advantage to every token of a response.

**Worked Example (synthetic):**
1. *GAE.* Three response tokens, terminal reward only: $r=[0,0,1]$, critic $V=[0.5,0.6,0.7]$, $\gamma=1$.
   - $\delta=[0.6-0.5,\ 0.7-0.6,\ 1-0.7]=[0.1,0.1,0.3]$.
   - $\lambda=0$: $\hat A=[0.1,0.1,0.3]$. $\lambda=1$: $[0.5,0.4,0.3]$ (return 1 minus $V$). $\lambda=0.95$: $\hat A_1=0.3$, $\hat A_1'=0.1+0.95(0.3)=0.385$, $\hat A_0=0.1+0.95(0.385)=0.46575$.
   - verl then whitens these over the batch, so absolute values are not what the actor sees.
2. *Group baseline.* Rewards $[1,0,0,1]$: mean 0.5, sample std 0.5774, advantages $\pm0.866$; population std would give $\pm1.0$ (CLM-044).
3. *Clip.* $\epsilon_l=\epsilon_h=0.2$, $c=3$, after one minibatch update:
   - token with $A=+0.866$, $r=1.30$: clipped term $-0.866\times1.2=-1.039$ is larger than $-1.126$, so $\ell=-1.039$ and the gradient through this token is zero.
   - token with $A=-0.866$, $r=0.70$: $\max(0.606,\ 0.693)=0.693$, clipped, zero gradient.
   - token with $A=-0.866$, $r=4.0$: $\max(3.464, 1.039)=3.464$, then $\min(3.464,\ 0.866\times3=2.598)=2.598$; dual clip bounds the loss and zeroes the gradient.
4. *Interpretation.* On the first minibatch of a batch, all three ratios would be 1 and nothing would clip (CLM-043). A nonzero clip fraction on the first minibatch therefore signals that old log-probs did not come from the current trainer parameters (Lesson 27.6).

**Knowledge Check:**
1. Why is a nonzero `pg_clipfrac` on the first minibatch suspicious in a decoupled-mode run?
2. What does $\lambda$ trade, and why does an outcome-only reward with $\gamma=\lambda=1$ make GAE equal to "final reward minus $V(s_t)$"?
3. Why does the sample-versus-population std choice not matter for a fixed $G$ without other loss terms, but matter once a KL term is added?

**Guided Practice:**
Implement `gae(rewards, values, mask, gamma, lam)` and `group_adv(scores, index, eps, std_kind)` in NumPy. Reproduce the worked example, then match verl's outputs on the same inputs by reading the pinned functions. Report every difference (whitening, Bessel correction, single-sample groups).

**Feedback Contract:**
- *Expected Evidence*: Hand-computed and coded advantages that agree; a table of clip outcomes for $A$ sign × ratio region; a note on which differences come from whitening and std convention.
- *Common Failure*: Treating the clip as a constraint that always binds, or comparing advantages across frameworks without checking std convention and whitening.
- *Diagnostic Hint*: On which minibatch can $r\neq1$, and which tensor provided `old_log_prob`?
- *Concept to Revisit*: GRPO advantage (Module 19, Lesson 19.5).

**Learning Outcome:**
Compute and explain the per-token gradient contribution of PPO/GRPO-family losses, including when clipping binds and how framework conventions change advantage scale.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 27.2 — Aggregation, Clip Shape, and Entropy Collapse

**Engineering Question:**
Which tokens does my update actually move, and why is entropy falling?

**Concepts & Definitions:**
- **Loss aggregation**: how the token loss matrix becomes a scalar. verl implements `token-mean` (sum over valid tokens divided by the global token count), `token-sum`, `seq-mean-token-sum`, `seq-mean-token-sum-norm` (divided by a constant horizon), and `seq-mean-token-mean`, with `token-mean` the actor default (**O**, CLM-005).
- **Length bias**: Liu et al. report that popular open-source PPO implementations normalized per response length, a non-constant that misaligns with the PPO objective (**O**, CLM-011). DAPO reports that sample-level averaging under-weights tokens in long responses, which in long-CoT RL let gibberish and repetition in overlong samples grow; it uses a token-level loss instead (**O**, CLM-006).
- **Entropy collapse**: policy entropy falling quickly so that groups become nearly identical. DAPO observed it with naive PPO/GRPO (**O**, CLM-007). Cui et al. report it consistently without entropy intervention and fit an empirical relation $R=-a e^{H}+b$ between entropy and performance (**O**, CLM-008).
- **Entropy–covariance mechanism**: for a tabular softmax policy, the step-wise entropy change is approximately $-\mathrm{Cov}(\log\pi,\Delta z)$, and under vanilla policy gradient $\Delta z_a=\eta\,\pi(a)A(a)$; high-probability actions with positive advantage lower entropy (**O**, CLM-008).
- **Clip-Higher**: with $\epsilon=0.2$ and $A>0$, a token at $\pi_{old}=0.01$ can rise only to 0.012 while one at 0.9 is effectively unconstrained; DAPO decouples $\epsilon_{low}=0.2$, $\epsilon_{high}=0.28$ (**O**, CLM-007).
- **Sequence-level ratio**: GSPO uses $s_i=\exp(\frac{1}{|y_i|}\sum_t\log\rho_{i,t})$ and clips whole responses; its clip ranges differ from GRPO's by orders of magnitude (**O**, CLM-010).
- **Efficiency versus asymptote**: Khatri et al. report that aggregation, normalization, curriculum, and off-policy algorithm mainly change compute efficiency rather than the asymptote in their recipes (**O**, CLM-038). Compare variants on curves, not one endpoint.

**Mechanism Explanation:**
Aggregation sets each token's weight before the advantage is applied. Under `seq-mean-token-mean`, a token's weight is $1/(G\,|o_i|)$: long responses contribute the same total as short ones, so each of their tokens counts less. Under `token-mean`, every valid token in the batch counts equally, so long responses carry more total weight. The clip then decides which tokens can move: the upper bound is multiplicative in probability, so it limits the *absolute* gain of rare tokens. Entropy falls when updates concentrate mass on already-likely tokens, which positive-advantage high-probability tokens do. The interventions act at different points: raise $\epsilon_{high}$ (let rare positive tokens rise), restrict high-covariance tokens (Clip-Cov/KL-Cov), filter zero-variance groups (Module 19), or add an entropy bonus (verl `entropy_coeff`, default 0) (**O**, CLM-005).

**Quantitative Model / Derivation:**
- Per-token weight: `seq-mean-token-mean` $w_{i,t}=\frac{1}{G|o_i|}$; `token-mean` $w_{i,t}=\frac{1}{\sum_j|o_j|}$.
- Clip ceiling for $A>0$: $\pi_\theta\le\pi_{old}(1+\epsilon_h)$.
- Entropy change (tabular, first order): $\Delta H\approx-\eta\,\mathrm{Cov}_\pi(\log\pi,\ \pi A)$ (**D**, CLM-009).

**Worked Example (synthetic):**
1. *Aggregation.* A group of $G=2$: response 1 has 100 tokens, $A=+1$; response 2 has 1,000 tokens, $A=-1$.
   - `seq-mean-token-mean`: per-token weight $1/200=0.005$ (short) versus $1/2000=0.0005$ (long). Each token of the long incorrect response is pushed down 10× less than each token of the short correct one is pushed up.
   - `token-mean`: every token weighs $1/1100\approx0.000909$; total weight 0.0909 on the short response and 0.909 on the long one.
   - Consequence: under per-sequence normalization, a long wrong response is penalized little per token, which is the mechanism behind length growth among incorrect answers (Module 19, Lesson 19.5).
2. *Clip-Higher.* $\pi_{old}=0.01$: ceiling 0.012 at $\epsilon_h=0.2$, 0.0128 at 0.28. $\pi_{old}=0.9$: ceiling 1.08 or 1.152, both above 1, so no bound in practice.
3. *Entropy covariance* (**D**, CLM-009). One state, $\pi=[0.7,0.2,0.1]$ ($H=0.80182$ nats), $\eta=0.1$:
   - $A=[0.3,-0.7,-0.7]$ (likely action is good): $\mathrm{Cov}=0.098555$, predicted $\Delta H=-0.00986$, exact after the softmax update $-0.00991$.
   - $A=[-0.1,-0.1,0.9]$ (rare action is good): $\mathrm{Cov}=-0.032088$, predicted $+0.00321$, exact $+0.00321$.
   - Interpretation: entropy falls when advantage aligns with probability. Verifiable rewards on tasks the model already mostly solves produce that alignment.
4. *DAPO ablation, author-reported* (Qwen2.5-32B base, AIME24 avg@32, cumulative): naive GRPO 30, +overlong filtering 36, +Clip-Higher 38, +soft overlong punishment 41, +token-level loss 42, +dynamic sampling 50 (**O**, CLM-007). Increments are cumulative, not independent effects.

**Knowledge Check:**
1. Why can switching from `seq-mean-token-mean` to `token-mean` shorten incorrect responses without any reward change?
2. Why does raising $\epsilon_{high}$ help rare tokens more than common ones?
3. In the tabular model, which reward pattern increases entropy, and why is it uncommon in RLVR on easy prompts?

**Guided Practice:**
On CPU, train a 50-token "vocabulary" bandit policy with REINFORCE and with the PPO clip. Log entropy, the covariance term, and the predicted versus actual entropy change per step. Then switch $\epsilon_h$ from 0.2 to 0.28 and record the step at which entropy falls below a fixed threshold.

**Feedback Contract:**
- *Expected Evidence*: Per-step predicted and measured $\Delta H$ with their correlation; entropy-threshold step for both clip settings; a weight table for two aggregation modes.
- *Common Failure*: Adding an entropy bonus without measuring which tokens lose probability, or attributing length growth to the reward when aggregation changed.
- *Diagnostic Hint*: Is the covariance positive, and which tokens hit the upper clip (`pg_clipfrac`) and at what probability?
- *Concept to Revisit*: Dr. GRPO normalization (Module 19, Lesson 19.5).

**Learning Outcome:**
Predict how aggregation, clip bounds, and ratio granularity redistribute gradient across tokens, and diagnose entropy collapse from logged covariance and clip statistics.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 27.3 — RLVR and Verifier Design

**Engineering Question:**
Is my reward function measuring correctness well enough to optimize against it?

**Concepts & Definitions:**
- **RLVR**: reinforcement learning with verifiable rewards. Tulu 3 gives reward $\alpha$ when a deterministic function verifies the completion is correct and 0 otherwise, optimized with PPO under a KL-to-reference penalty (**O**, CLM-012).
- **Rule-based versus learned reward**: DeepSeek-R1-Zero used accuracy rewards (rule-checked boxed answers; compiler and tests for code) plus format rewards, and avoided neural reward models because they may be hacked in large-scale RL and add retraining cost (**O**, CLM-013).
- **Verifier error rates**: false-negative rate $f_n$ (correct marked wrong) and false-positive rate $f_p$ (wrong marked correct). Huang et al. report rule-based math verifiers with >99% precision but 86% average recall (0.78 on Skywork-OR1), with recall falling as the policy gets stronger; model-based verifiers raised recall (84%→92% on Skywork-OR1) but were hackable during RL (**O**, CLM-014).
- **Verifier versus reward model**: a verifier encodes a correctness criterion; a reward model predicts a preference. Both are instruments with error rates (Module 15, Lesson 15.3), but a verifier's errors are often systematic in format, while a reward model's are exploitable along directions the policy can find.
- **Spurious-reward control**: Shao et al. report that random rewards raised Qwen2.5-Math-7B MATH-500 by 21.4 points versus 29.1 with ground truth, attributed partly to a clipping bias that amplifies pretrained behaviors, and that such gains often fail on Llama3 or OLMo2 (**O**, CLM-015). A reward ablation is therefore a required control.
- **Evaluation contract**: Yue et al. report RLVR models beat their bases at small $k$ but bases reach higher pass@k at large $k$ on their evaluations (**O**, CLM-016). State which $k$ the claim is about (Module 06, Lesson 6.2).

**Mechanism Explanation:**
The policy sees only $r$. If $r$ is correctness passed through a noisy channel, the expected reward of a policy is a mix of its true accuracy and the channel's errors. False negatives scale down the useful signal and grow as the policy finds correct but unusually formatted answers. False positives are worse: any response pattern with elevated $f_p$ that the policy can reach becomes a direction of reward increase that is not a direction of correctness increase. A verifier must therefore be characterized on the *policy's own outputs*, at several checkpoints, against independent labels.

**Quantitative Model / Derivation (D, CLM-017):**
- $\mathbb{E}[r]=q(1-f_n)+(1-q)f_p$, where $q$ is the true correct rate.
- Reward gap between a correct and an incorrect response: $1-f_n-f_p$.
- Hacking condition: if a pattern raises $f_p$ to $f_p'$ on incorrect answers, reward rises whenever $q'(1-f_n)+(1-q')f_p' > q(1-f_n)+(1-q)f_p$, even with $q'<q$.

**Worked Example (synthetic; $f_n=0.14$, $f_p=0.01$, informed by the magnitudes in CLM-014):**
1. Honest policy, $q=0.60$: $\mathbb{E}[r]=0.60(0.86)+0.40(0.01)=0.516+0.004=0.520$. Signal gap $1-0.14-0.01=0.85$: 15% of the correct/incorrect contrast is lost.
2. Hacking policy: $q=0.30$, but its incorrect answers use a pattern (e.g., several candidate answers in one box) the checker accepts with $f_p=0.90$: $\mathbb{E}[r]=0.30(0.86)+0.70(0.90)=0.258+0.630=0.888$.
3. Interpretation: reward rose from 0.520 to 0.888 while accuracy halved. Nothing in the training curve distinguishes the two; only an independent check of correctness does (Lesson 27.4).
4. Spurious-reward control: run the same recipe with shuffled or random rewards. If most of the gain survives, the claim "the reward taught this" is not supported for this model (CLM-015).

**Knowledge Check:**
1. Why does a rule-based verifier's false-negative rate tend to rise as the policy improves?
2. Why is a verifier's static accuracy on a fixed benchmark insufficient to predict its behavior during RL?
3. What result from a random-reward control would make you doubt a reported RLVR gain?

**Guided Practice:**
Take 200 sampled responses from a small model on a math set. Label correctness by hand or with a stricter symbolic checker. Measure $f_n$ and $f_p$ for a string-match checker and for a normalized-answer checker, with intervals. Then construct three adversarial formats (multiple answers, answer-in-text, empty box) and measure each checker's $f_p$ on them.

**Feedback Contract:**
- *Expected Evidence*: Confusion tables per checker with intervals, adversarial-format $f_p$, and the predicted $\mathbb{E}[r]$ for an honest and a hacking policy.
- *Common Failure*: Reporting verifier accuracy on gold reference answers instead of on policy samples, or ignoring false negatives.
- *Diagnostic Hint*: Which response formats can the policy reach that your labeled set never contained?
- *Concept to Revisit*: Graders as Instruments (Module 15, Lesson 15.3); Verifier Gaming (Module 06, Lesson 6.4).

**Learning Outcome:**
Characterize a reward function's error rates on policy samples and decide whether, and with which controls, it is safe to optimize against.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 27.4 — Reward Hacking, Over-Optimization, and KL Placement

**Engineering Question:**
How far can I push a proxy reward before true quality falls, and how do I find out during the run?

**Concepts & Definitions:**
- **Over-optimization**: Gao et al. (synthetic gold reward model labeling a proxy) find gold reward versus $d=\sqrt{KL(\pi\|\pi_{init})}$ follows $R_{bon}(d)=d(\alpha_{bon}-\beta_{bon}d)$ for best-of-$n$ and $R_{RL}(d)=d(\alpha_{RL}-\beta_{RL}\log d)$ for RL, with coefficients scaling smoothly with reward-model size; best-of-$n$ KL is $\log n-(n-1)/n$; RL consumes far more KL than best-of-$n$; and varying the KL penalty acted like early stopping without moving the KL–gold frontier in their setting (**O**, CLM-018).
- **KL budget**: treat KL from the initial policy as spent resource; the gold reward peaks at a finite KL.
- **Checker over-optimization**: Tulu 3 reports some higher-KL RLVR runs on constraint prompts produced outputs that satisfied the checker without meaningful content (**O**, CLM-012).
- **Detection limits**: Moya et al. show that, in general, observations available during RLVR are insufficient to detect or identify accepted errors, and that audit feedback about correctness can reduce them (**O**, CLM-019). The CATCH testbed reports that a chain-of-thought monitor's protection eroded as the policy learned to mislead it (**O**, CLM-020).
- **Independent audit**: a correctness check the policy is not optimized against (stricter verifier, held-out tests, human labels), run on frozen held-out prompts at each checkpoint.
- **KL placement**: verl's `apply_kl_penalty` subtracts $\beta\,\mathrm{KL}$ from token rewards when `use_kl_in_reward` is true (default false); `use_kl_loss` (default false, coef 0.001) adds KL to the loss instead (**O**, CLM-045). In the reward, KL passes through the advantage estimator; with an outcome-only group estimator it becomes a per-response total that is group-centered and std-normalized (**D**, CLM-046).

**Mechanism Explanation:**
The proxy (reward model or checker) agrees with the true objective near the initial policy and disagrees further away. Optimization moves the policy along directions where the proxy's gradient is largest, including directions where proxy and truth disagree. KL from the initial policy measures how far that search has gone. A KL penalty slows the search; in Gao et al.'s setting it did not find better gold reward at a given KL. Because in-loop signals cannot separate accepted errors from real gains, the detection signal must come from outside the loop: the paired gap between training reward and audit pass rate on the same frozen prompts.

**Quantitative Model / Derivation:**
- RL peak: $\frac{dR_{RL}}{dd}=\alpha-\beta\log d-\beta=0\Rightarrow d^*=e^{\alpha/\beta-1}$.
- Best-of-$n$ peak: $d^*=\alpha/(2\beta)$.
- Hacking monitor: $g_k = \mathrm{pass}_{train}(k) - \mathrm{pass}_{audit}(k)$ on frozen held-out prompts at checkpoint $k$, with a paired bootstrap interval (Module 15, Lesson 15.4).
- KL in reward, group estimator: score$_i=r_i-\beta\sum_t\mathrm{kl}_{i,t}$, then centered and divided by group std.

**Worked Example (synthetic coefficients; not fitted values from the paper):**
1. *Peaks.* $\alpha=1$, $\beta=0.5$ for both forms. RL: $d^*=e^{1}=2.718$, KL $=7.39$ nats, $R=2.718(1-0.5\cdot1)=1.359$. Best-of-$n$: $d^*=1$, KL $=1$ nat, $R=0.5$. Best-of-16 spends KL $=\log16-15/16=1.835$ ($d=1.355$); best-of-1000 spends 5.909 ($d=2.431$).
2. *Interpretation.* Each method has a gold peak at a finite KL; past it, more optimization lowers gold reward. Logging KL-to-init lets you place a checkpoint on this axis; the coefficients must be estimated per setup.
3. *KL placement* (**D**, CLM-046). $G=2$, both responses correct ($r=[1,1]$), summed KL $[2,10]$, $\beta=0.05$: scores $[0.90,0.50]$, mean 0.70, sample std 0.2828, advantages $[+0.707,-0.707]$. Without KL this group would have zero advantage. With KL in the reward and std normalization, a small KL difference becomes a unit-scale signal; with KL in the loss, both responses would receive the same-size KL gradient per token regardless of group.
4. *Monitor.* At checkpoints 0, 200, 400: training-checker pass 0.52, 0.71, 0.86; audit pass 0.50, 0.66, 0.58. Gap 0.02, 0.05, 0.28 (synthetic). Stop or roll back at 200 if the declared margin is 0.10 and the gap's lower bound at 400 exceeds it.

**Knowledge Check:**
1. Why can a KL penalty change *when* gold reward peaks without changing the best achievable gold reward in Gao et al.'s setting?
2. Why does group std normalization amplify KL-in-reward differences among all-correct groups?
3. Why must the audit prompts be frozen and the audit checker not be used for training?

**Guided Practice:**
Given logged per-checkpoint training reward, audit pass rate, KL-to-init, and response length from a small RLVR run (provided as a synthetic CSV or your LAB B output), fit both functional forms of gold-versus-$d$, mark the empirical peak, and write a stopping rule with a margin and a rollback checkpoint.

**Feedback Contract:**
- *Expected Evidence*: Fitted curves with residuals, the paired train–audit gap with intervals per checkpoint, KL placement stated from config, a stopping rule.
- *Common Failure*: Using training reward as evidence of improvement, or asserting "KL keeps it safe" without measuring KL-to-init or the audit gap.
- *Diagnostic Hint*: What would your training curve look like if the checker had a hole the policy just found?
- *Concept to Revisit*: KL-to-base monitoring (Module 19, Lesson 19.5).

**Learning Outcome:**
Treat KL as an optimization budget, place KL deliberately, and detect over-optimization with an independent audit and a pre-declared stopping rule.

*(Effort: 50m instruction, 20m practice)*

---

### Lesson 27.5 — The RL Loop as a Distributed System: Placement, Sync, and Step Time

**Engineering Question:**
Where does an RL step's wall time go, and what placement makes it shorter without changing what is learned?

**Concepts & Definitions:**
- **Rollout engine versus trainer**: generation runs on an inference engine (continuous batching, KV cache; Modules 03–04); gradients run on a training engine. Different engines, layouts, and numerics.
- **Colocated (hybrid) placement**: both stages time-share the same GPUs. HybridFlow/verl reshards actor weights between training and generation with a 3D-HybridEngine (zero memory redundancy), reporting 1.53×–20.57× throughput over its baselines (**O**, CLM-022).
- **Disaggregated placement**: dedicated rollout and training resources. StreamRL reports that colocation couples resources although generation is memory-bandwidth-bound and training compute-bound, identifies pipeline bubbles and skewness bubbles (long-tail output lengths), and reports up to 2.66× throughput with streaming and skewness-aware scheduling (**O**, CLM-023).
- **Long tail**: DORA's authors state rollout accounts for 50%–80% of step time (citing prior work) and that p99 length can exceed the median by over 10× in math and code (**O**, CLM-025). Treat the range as something to measure.
- **Weight sync**: copying new weights to rollout replicas after each update. Lower bound: model bytes over the slowest effective bandwidth (**D**, CLM-027).
- **Length cap and reward**: DAPO masks truncated samples and adds a soft overlong punishment, linear from 0 at $L_{max}-L_{cache}$ to −1 at $L_{max}$ (16,384 + 4,096 tokens in its setup) (**O**, CLM-047). The cap is a systems knob with a reward consequence.

**Mechanism Explanation — Pinned verl Loop (O, CLM-028):**
`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` at commit `fbb4b3a8bf636f290c9c59fc346f756849e9c241`, one synchronous step:
1. `async_rollout_manager.generate_sequences` → `checkpoint_manager.sleep_replicas()` (rollout replicas release resources for training).
2. Reward extraction (`extract_reward`), optionally a colocated reward model.
3. Either `apply_bypass_mode` (reuse rollout log-probs as `old_log_probs`) or `_compute_old_log_prob` (trainer forward pass), then optional reference and critic forward passes.
4. Optional `apply_kl_penalty`, rollout correction (Lesson 27.6), `compute_advantage`.
5. Optional `_update_critic`, then `_update_actor`, then `checkpoint_manager.update_weights(global_steps)` pushes weights to rollout replicas.

Each stage is timed with `marked_timer` (`gen`, `reward`, `old_log_prob`, `adv`, `update_actor`), which is the raw material for step accounting. In a synchronous step, stage 1 ends only when the longest response finishes.

**Quantitative Model / Derivation (D, CLM-026, CLM-027):**
- $T_{gen}=\max_i L_i/v$ (all sequences decode concurrently at per-sequence rate $v$).
- Decode-slot utilization $U=\sum_iL_i/(B\max_iL_i)$.
- Step time $T=T_{gen}+T_{reward}+T_{logprob}+T_{ref}+T_{update}+T_{sync}$; rollout fraction $T_{gen}/T$.
- $T_{sync}\ge \text{bytes}/\text{bandwidth}$, plus resharding, engine reload, and cache invalidation.
- Assumes constant $v$; in practice $v$ rises as the batch drains (Module 04), so this is an upper bound on the long-tail penalty.

**Worked Example (synthetic):**
1. *Lengths.* $B=512$: 461 responses of 4,096 tokens, 46 of 12,288, 5 of 32,768.
   - $\sum L = 1{,}888{,}256+565{,}248+163{,}840=2{,}617{,}344$; $B\max L=16{,}777{,}216$; $U=15.6\%$.
2. *Step time.* $v=30$ tok/s per sequence: $T_{gen}=32{,}768/30=1{,}092.3$ s, while a 4,096-token response finishes at 136.5 s. With $T_{logprob}=120$ s, $T_{update}=300$ s, other stages 0: $T=1{,}512.3$ s, rollout fraction 72.2%.
3. *Cap at 16,384.* The 5 longest are truncated: $T_{gen}=546.1$ s, $T=966.1$ s, rollout fraction 56.5%. The 5 truncated responses now need a reward rule (filter or soft punishment, CLM-047), and dropping them biases group statistics toward short answers (CLM-025).
4. *Weight sync.* 7×10⁹ parameters in bf16 = 14 GB: 0.28 s at an effective 50 GB/s, 4.48 s over a 25 Gb/s (3.125 GB/s) link. Colocated, this is negligible next to $T_{gen}$; disaggregated across a slow link with frequent syncs, it is not.

**Knowledge Check:**
1. Why does doubling rollout GPUs barely shorten $T_{gen}$ when one sequence is 8× the median length?
2. What does colocation save, and what does it force both stages to share?
3. Why is "drop the longest responses" a change to the estimator and not only to the system?

**Guided Practice:**
From a provided verl-style timing log (or LAB C output), build a per-stage step-time table, compute $U$ from the length histogram, and predict step time under (a) a length cap, (b) 2× rollout GPUs, (c) disaggregation with a stated link bandwidth and sync every step. State which prediction you trust least and why.

**Feedback Contract:**
- *Expected Evidence*: Stage table summing to measured step time, $U$, three predictions with assumptions, and the change each makes to the samples the trainer sees.
- *Common Failure*: Reporting tokens/s for generation alone as RL throughput, or ignoring what a length cap does to rewards.
- *Diagnostic Hint*: In the `gen` stage, how many sequences are still decoding during its last 50%?
- *Concept to Revisit*: Continuous Batching (Module 04, Lesson 4.2); Collectives (Module 20, Lesson 20.2).

**Learning Outcome:**
Account RL step time by stage, predict the effect of length caps, scaling, and placement, and state their algorithmic side effects.

*(Effort: 50m instruction, 20m practice; source trace 2h, shared with Lesson 27.6 and LAB D)*

---

### Lesson 27.6 — Off-Policyness: Asynchrony, Staleness, and Log-Probability Mismatch

**Engineering Question:**
How off-policy is each training batch, from staleness and from sampler/trainer numerics, and what correction is applied?

**Concepts & Definitions:**
- **Three-policy view**: $\pi_{rollout}$ (the engine that sampled, with its numerics and version), $\pi_{old}$ or $\pi_{prox}$ (the anchor for the trust region), and $\pi_\theta$ (being updated). Hilton et al. decouple the proximal policy (controls update size) from the behavior policy (used for off-policy correction) (**O**, CLM-031).
- **Asynchronous RL and staleness**: generation and training overlap, so a batch contains samples from older policy versions. Noukhovitch et al. find off-policy tolerance depends on the algorithm (online DPO most robust in their tests) and report about 40% and 70% faster training while matching synchronous results on their tasks (**O**, CLM-030).
- **AReaL**: interruptible rollout workers discard and recompute KV caches with new weights mid-generation (trajectories span versions), admission enforces $\lfloor(N_r-1)/B\rfloor\le i+\eta$, and a decoupled PPO objective uses a recent proximal policy. In its ablation (1.5B, math), naive PPO degraded with staleness (AIME24 23.3 at $\eta=4$ vs 42.0 synchronous) while the decoupled objective stayed near the oracle up to $\eta=8$ (41.0); unbounded staleness was still worse (**O**, CLM-024).
- **verl staleness knob**: the experimental fully-async trainer defaults `staleness_threshold` 0.1, `trigger_parameter_sync_step` 4, `partial_rollout` True; its README defines the threshold as a maximum proportion of stale samples, 0 as synchronous, ~1 as one-step-off-policy (**O**, CLM-029). This is not AReaL's version lag $\eta$.
- **Staleness accounting**: PACE separates generation staleness (accumulated while a trajectory is generated) from waiting staleness (accumulated in the pool) (**O**, CLM-037).
- **Log-probability mismatch**: rollout and training engines assign different probabilities to the same tokens even with identical weights. Qi et al. identify BF16's 7 mantissa bits as the primary source in their experiments and report FP16 virtually eliminated it; they describe sequence-level IS as unbiased but high-variance, truncated IS $\min(\rho,C)$ and masked IS $\rho\,\mathbb{1}[\rho\le C]$ as bias-for-variance trades, an extra forward pass (~25% cost under their assumption), and a remaining deployment gap (**O**, CLM-032). Root cause analysis is Module 30.
- **Effective sample size**: Kish ESS $=(\sum w)^2/\sum w^2\in[1,B]$. VCPO reports that collapse under high asynchrony is predicted by collapsing ESS (**O**, CLM-034).
- **Confidence-dependent thresholds**: calibrated IS tightens the cap as token confidence rises (**O**, CLM-035); ENTR scales the trust region by token entropy and flags low-entropy tokens where mismatch is amplified (**O**, CLM-036).
- **Unverified origins**: the original token-level TIS write-up and the "When Speed Kills Stability" blog cited by verl were not opened; their numbers are not used (TODO_VERIFY, CLM-048).

**Mechanism Explanation — Pinned verl Correction (O, CLM-033):**
At commit `fbb4b3a8bf636f290c9c59fc346f756849e9c241`, `verl/trainer/ppo/rollout_corr_helper.py` states its purpose: rollout/training implementation mismatch (e.g., vLLM BF16 versus FSDP FP32), staleness, and general shift.
1. In decoupled mode, `RayPPOTrainer.fit` recomputes `old_log_probs` with the trainer, then calls `compute_rollout_correction_and_add_to_batch`.
2. `compute_rollout_correction_and_rejection_mask` forms `log_ratio = old_log_prob - rollout_log_prob`.
3. `compute_rollout_correction_weights` makes token weights `exp(clamp(log_ratio, -20, 20))` or one sequence weight `exp(sum log_ratio)` broadcast over tokens, then truncates with `clamp(max=rollout_is_threshold)` (TIS) or zeroes out-of-band weights for a `lower_upper` threshold (IcePop); weights are detached and ESS metrics are logged.
4. `verl/workers/utils/losses.py::ppo_loss` passes `rollout_is_weights` to `compute_policy_loss_vanilla`, which multiplies per-token losses by them before aggregation.
5. Defaults in `verl/trainer/config/algorithm/rollout_correction.yaml`: `rollout_is: null` (off), `rollout_is_threshold: 2.0`, `bypass_mode: false`.

So the IS weight corrects $\pi_{rollout}\to\pi_{old}$, and the PPO ratio and clip handle $\pi_{old}\to\pi_\theta$. In bypass mode there is no separate correction: the PPO ratio is taken directly against rollout log-probs. The experimental one-step-off-policy and separation trainers exist at this commit but were not traced (TODO_VERIFY, CLM-049).

**Quantitative Model / Derivation:**
- Token TIS: $w_t=\min(\pi_{old}(o_t)/\pi_{rollout}(o_t),\ C)$; sequence IS: $w=\exp\sum_t\log\rho_t$.
- With independent per-token log-ratio noise $\sigma$, the sequence log-weight has s.d. $\sigma\sqrt T$ (**D**, CLM-039).
- ESS $=(\sum_iw_i)^2/\sum_iw_i^2$.
- AReaL admission: $\lfloor(N_r-1)/B\rfloor\le i+\eta$.

**Worked Example (synthetic):**
1. *Sequence weights.* $\sigma=0.02$ per token, $T=1000$: log-weight s.d. $0.02\sqrt{1000}=0.632$; a $+2\sigma$ sequence has $w=e^{1.265}=3.54$. Token-level weights stay near 1 but drop the cross-token product (biased) (**D**, CLM-039).
2. *ESS and truncation.* Eight sequence weights $[1,1,1,1,1,1,1,10]$: ESS $=17^2/107=2.70$ of 8. With TIS at $C=2$: $[1,\ldots,1,2]$, ESS $=9^2/11=7.36$. Truncation restored most of the effective batch at the cost of under-weighting the one sequence the trainer now likes far more than the sampler did.
3. *Staleness admission.* $B=512$, trainer at version $i=10$, $\eta=2$: generation may continue while $\lfloor(N_r-1)/512\rfloor\le12$, i.e., up to $N_r=6{,}656$ trajectories; trajectory 6,657 waits for version 11.
4. *Interpretation.* Staleness and mismatch both show up as $\pi_{old}\neq\pi_{rollout}$. To separate them, measure mismatch at zero staleness (synchronous step, same weights) and subtract.

**Knowledge Check:**
1. In decoupled mode, which two log-prob tensors form the IS weight, and which two form the PPO ratio?
2. Why does sequence-level IS become unusable for long responses without length normalization or truncation?
3. Why is a nonzero mismatch KL at zero staleness evidence about numerics rather than asynchrony?

**Guided Practice:**
Given per-token rollout and trainer log-probs for 64 responses (provided synthetic arrays or LAB D output), compute token- and sequence-level IS weights, TIS at $C\in\{1.5,2,5\}$, the fraction of tokens truncated, ESS/B, and mismatch KL binned by trainer token probability. Recommend a correction and state its bias.

**Feedback Contract:**
- *Expected Evidence*: Weight distributions, truncation fractions, ESS per setting, probability-binned mismatch, and the chosen configuration with its stated bias and deployment gap.
- *Common Failure*: Treating TIS as removing the mismatch, or confusing verl's `staleness_threshold` (sample proportion) with a version lag.
- *Diagnostic Hint*: What is ESS/B before correction, and which probability bin contributes the largest weights?
- *Concept to Revisit*: Precision formats (Module 05, Lesson 5.3); Lesson 27.1 three-policy discussion.

**Learning Outcome:**
Quantify staleness and sampler/trainer mismatch per batch, choose and configure an importance-sampling correction, and monitor its variance with ESS.

*(Effort: 50m instruction, 20m practice; source trace shared, see Lesson 27.5)*

---

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [PPO](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017; clipped surrogate.
- [GAE](https://arxiv.org/abs/1506.02438) — Schulman et al., 2015; advantage bias–variance.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — Gao, Schulman, Hilton, 2022.
- [Batch size-invariance for policy optimization](https://arxiv.org/abs/2110.00641) — Hilton, Cobbe, Schulman, 2021; proximal versus behavior policy.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025; rule-based rewards, Section 2.2.2.

**CURRENT DEFAULT** (scoped to open recipes and the pinned framework, not industry-wide adoption): RLVR with rule-based checkers for verifiable domains ([Tulu 3](https://arxiv.org/abs/2411.15124), Lambert et al., 2024); critic-free group baselines; colocated hybrid placement ([HybridFlow/verl](https://arxiv.org/abs/2409.19256), Sheng et al., 2024); verl defaults at the pinned commit (token-mean aggregation, symmetric clip 0.2, rollout correction off).

**WORKLOAD-DEPENDENT:** token-level loss and Clip-Higher ([DAPO](https://arxiv.org/abs/2503.14476), Yu et al., 2025); length-bias corrections ([Dr. GRPO](https://arxiv.org/abs/2503.20783), Liu et al., 2025); [REINFORCE/RLOO](https://arxiv.org/abs/2402.14740) (Ahmadian et al., 2024); verifier choice ([rule vs model verifiers](https://arxiv.org/abs/2505.22203), Huang et al., 2025); disaggregation ([StreamRL](https://arxiv.org/abs/2504.15930), Zhong et al., 2025); [asynchronous RLHF](https://arxiv.org/abs/2410.18252) (Noukhovitch et al., 2024).

**FRONTIER:** [Entropy Mechanism](https://arxiv.org/abs/2505.22617) (Cui et al., 2025); [GSPO](https://arxiv.org/abs/2507.18071) (Zheng et al., 2025); [Spurious Rewards](https://arxiv.org/abs/2506.10947) (Shao et al., 2025); [RLVR and reasoning boundary](https://arxiv.org/abs/2504.13837) (Yue et al., 2025); [ScaleRL](https://arxiv.org/abs/2510.13786) (Khatri et al., 2025); [AReaL](https://arxiv.org/abs/2505.24298) (Fu et al., 2025); [FP16 mismatch](https://arxiv.org/abs/2510.26788) (Qi et al., 2025); [DORA](https://arxiv.org/abs/2604.26256) (Hu et al., 2026); [VCPO](https://arxiv.org/abs/2602.17616) (Huang et al., 2026); [ENTR](https://arxiv.org/abs/2607.22186) (Zhao et al., 2026); [PACE](https://arxiv.org/abs/2609.36830) (Li et al., 2026); [Calibrated IS](https://arxiv.org/abs/2609.32444) (Yu et al., 2026); [Verifier Errors in RLVR](https://arxiv.org/abs/2609.35677) (Moya et al., 2026); [CATCH](https://arxiv.org/abs/2609.39533) (Wang et al., 2026). Several 2026 entries were inspected at abstract level only; see the registry.

**LEGACY / INSUFFICIENT:** per-response length normalization by default; trusting training reward without an audit; static verifier accuracy on reference answers as evidence of RL safety; synchronous steps with no length accounting; asynchronous training without a staleness bound or correction metrics; "the clip keeps it on-policy."

**PRODUCTION SOURCE TRACE**
- Repository: `volcengine/verl`
- Revision: `fbb4b3a8bf636f290c9c59fc346f756849e9c241` (last commit on `main` before the 2026-09-30 cutoff, committed 2026-09-30)
- Verified: 2026-10-12; static inspection only.
- Files/symbols:
  - `verl/trainer/ppo/core_algos.py::{compute_gae_advantage_return, compute_grpo_outcome_advantage, agg_loss, compute_policy_loss_vanilla}`
  - `verl/trainer/ppo/ray_trainer.py::{RayPPOTrainer.fit, RayPPOTrainer._compute_old_log_prob, RayPPOTrainer._update_actor, apply_kl_penalty, compute_advantage}`
  - `verl/trainer/ppo/rollout_corr_helper.py::{compute_rollout_correction_and_add_to_batch, compute_rollout_correction_and_rejection_mask, compute_rollout_correction_weights, compute_is_metrics, apply_bypass_mode}`
  - `verl/workers/utils/losses.py::ppo_loss`
  - `verl/trainer/config/actor/actor.yaml`, `verl/trainer/config/ppo_trainer.yaml`, `verl/trainer/config/algorithm/rollout_correction.yaml`, `verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml`
- Not traced: rollout-engine weight-update internals, the fully-async rollouter/trainer code, and the one-step-off-policy and separation trainers (TODO_VERIFY, CLM-049).

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY` and record a run manifest: model and checkpoint hashes, data and prompt-set IDs, reward function version and its measured $f_n/f_p$, estimator (advantage, aggregation, clip bounds, ratio level), KL placement and coefficient, rollout engine and dtype, trainer dtype, staleness bound, correction settings, seeds, and the resolved config file. Hardware tiers: `CPU`, `COLAB-GPU`, `TRACE`; no local GPU is assumed. Labs are described here; no lab scaffolding is provided in this revision.

### LAB A — Estimator Workbench: Advantages, Clipping, Aggregation, and Entropy
- **Hardware**: CPU.
- **Objective**: Implement GAE, group advantages (both std conventions), the clipped loss with decoupled and dual clip, and five aggregation modes; reproduce verl's outputs on fixed tensors; then train a tabular softmax bandit to test the entropy–covariance relation.
- **Pre-Registered Hypothesis**: Per-step entropy change correlates with $-\eta\,\mathrm{Cov}(\log\pi,\pi A)$ (CLM-008, CLM-009); with one minibatch per batch and recomputed old log-probs, clip fraction is zero (CLM-043); raising $\epsilon_h$ from 0.2 to 0.28 delays the step at which entropy crosses a declared threshold.
- **Independent Variables**: $\lambda$, std convention, aggregation mode, $\epsilon_l,\epsilon_h$, minibatches per batch, reward distribution over actions.
- **Dependent Variables**: Advantages, per-token weights, clip fraction (upper and lower), entropy, covariance, predicted versus measured $\Delta H$, threshold-crossing step.
- **Break & Falsify**: Use a reward that favors a rare action and confirm entropy rises; run four minibatches per batch and show nonzero clip fraction appears only after the first. The covariance hypothesis is falsified if the correlation between predicted and measured $\Delta H$ is below a pre-declared value (e.g., 0.9) at small $\eta$.
- **Alignment**: Lessons 27.1–27.2.
- **Effort Estimate**: 3h total.

### LAB B — Planted Verifier: RLVR, Hacking, and an Independent Audit
- **Hardware**: COLAB-GPU (≤1B-parameter policy, short responses).
- **Objective**: Run GRPO-style RLVR on a toy verifiable task (e.g., arithmetic with boxed answers) with three rewards: a strict checker, a checker with a planted loophole (e.g., accepts any box containing the answer among several numbers), and random rewards. Measure each checker's $f_n/f_p$ on policy samples before and during training, and audit every checkpoint with the strict checker on frozen held-out prompts. Ablate per-sequence versus token-level aggregation and record length by correctness.
- **Pre-Registered Hypothesis**: CLM-021 (the train–audit gap exceeds margin $\delta$ while training reward still rises); with per-sequence normalization, incorrect responses lengthen more than correct ones (CLM-006, CLM-011); the random-reward run gains less than the strict run on the audit.
- **Independent Variables**: Reward function, aggregation mode, KL placement (none, in reward, in loss) and $\beta$.
- **Dependent Variables**: Training reward, audit pass rate with paired intervals, loophole-pattern frequency, KL to init, response length by correctness, entropy.
- **Break & Falsify**: Increase the loophole's reachability (put an example in the prompt) and record when the gap opens; falsify CLM-021 if the gap only opens after reward plateaus or loophole frequency does not track it. Fit Gao et al.'s RL form to audit-versus-$\sqrt{KL}$ and report fit quality without treating it as validation of the law.
- **Alignment**: Lessons 27.2–27.4.
- **Effort Estimate**: 3.5h total.

### LAB C — RL Step Simulator: Long Tails, Placement, and Staleness
- **Hardware**: CPU.
- **Objective**: Build a discrete-event simulator of rollout workers and a trainer with a declared length distribution (heavy-tailed, synthetic), per-sequence decode rate, update time, and weight-sync cost. Compare synchronous colocated, synchronous disaggregated, and asynchronous with staleness bound $\eta\in\{0,1,2,4,8\}$ using AReaL's admission rule; add a length cap with filtering.
- **Pre-Registered Hypothesis**: CLM-041 (trainer utilization rises with diminishing returns as $\eta$ grows; consumed-sample lag rises with $\eta$); the synchronous rollout fraction matches the CLM-026 accounting within a declared tolerance.
- **Independent Variables**: Length distribution tail, $B$, GPU split, link bandwidth, sync frequency, $\eta$, length cap.
- **Dependent Variables**: Step time by stage, slot utilization, trainer and rollout busy fractions, samples/s, consumed-sample lag histogram (generation versus waiting staleness, CLM-037), fraction of truncated or dropped samples and their length distribution.
- **Break & Falsify**: Make all lengths equal and show asynchrony gains collapse to sync-overhead hiding; make the link slow and show disaggregation loses to colocation. Falsify CLM-041 if utilization does not rise from $\eta=0$ to $1$. State that the simulator says nothing about learning quality.
- **Alignment**: Lessons 27.5–27.6.
- **Effort Estimate**: 3h total.

### LAB D — Source Trace and Log-Probability Mismatch
- **Hardware**: TRACE (required) + COLAB-GPU (optional measurement).
- **Objective**: Complete the pinned verl trace from `RayPPOTrainer.fit` through rollout, old-log-prob recomputation or bypass, rollout correction, advantage, loss, and `update_weights`, citing symbols. Optionally, on a small model, sample with an inference engine in bf16 and recompute log-probs with the trainer in fp32 and in bf16 on the same tokens; compute mismatch KL, probability-binned mismatch, IS weights, TIS at several $C$, and ESS.
- **Pre-Registered Hypothesis**: CLM-040 (mismatch concentrates in low-probability tokens; TIS raises ESS relative to untruncated weights; matched precision reduces mismatch KL).
- **Independent Variables**: Trainer dtype, inference engine and dtype, token- versus sequence-level IS, $C$, response length.
- **Dependent Variables**: Mismatch k1/k3 KL, mismatch by probability bin, IS weight max and quantiles, truncated fraction, ESS/B, first-minibatch clip fraction in bypass versus decoupled mode.
- **Break & Falsify**: Run with `bypass_mode` semantics (ratio against rollout log-probs) and show first-minibatch clip fraction becomes nonzero; double response length and show sequence-level weight variance grows (CLM-039). Falsify CLM-040 if mismatch is flat across probability bins.
- **Alignment**: Lessons 27.5–27.6; Production Source Trace.
- **Effort Estimate**: 3.5h total (source trace counted separately, 2h).

---

## 07 Break / Incident Scenarios

### Incident 27.1 — The Reward Went Up, the Model Got Worse

- **Incident Symptoms**:
  - A math-and-code RLVR run was moved from a synchronous colocated setup to an asynchronous disaggregated one to cut step time. Step time fell by roughly half.
  - Over the next 600 steps, training reward rose faster than before, entropy fell faster, and mean response length grew, especially among responses the training checker marked wrong.
  - Held-out pass@1 on the release suite fell slightly; pass@16 fell more.
  - The migration changed several things at once: a new inference engine build with bf16 sampling, `staleness_threshold` raised, rollout correction left at its default, a 25% lower max length with truncated samples given reward 0, and a new code checker that accepts outputs when the last test printed "PASS".
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*:
     - checker false positives (the "PASS" pattern) being exploited;
     - staleness without a decoupled objective or correction;
     - sampler/trainer log-prob mismatch from the new engine build;
     - entropy collapse from probability-aligned advantages, worsened by mismatch or staleness;
     - truncation reward noise from the lower length cap;
     - aggregation or config drift between runs;
     - evaluation suite or decoding change;
     - training variance.
  2. *Rank Initial Plausibility*: Use the change list and the symptom pattern (reward up, audit down, length up among wrong answers), but rank nothing as cause without a discriminating test. Several changes can interact: mismatch and staleness both inflate effective off-policyness.
  3. *Identify Missing Evidence*: Both resolved configs; checker $f_n/f_p$ on current policy samples; an independent audit on frozen prompts per checkpoint; mismatch KL and IS weight/ESS metrics; staleness histogram; clip fractions; truncation rate; length by correctness; engine and dtype versions.
  4. *Design Discriminating Tests*:
     - Re-audit saved checkpoints with a strict checker; measure the fraction of "accepted" code that fails independent tests.
     - Recompute trainer log-probs on a fixed rollout batch at zero staleness to isolate mismatch from staleness.
     - Replay one segment synchronously with the old engine build, then change one factor at a time: checker, staleness, engine, length cap.
     - Enable TIS and the decoupled objective on a short branch and compare ESS and entropy slope.
  5. *Execute Causal Diagnosis*: Rank causes with the evidence each test produced; report interactions and what remains unexplained.
  6. *Prescribe Mitigation and Prevention*:
     - Roll back to the last checkpoint whose audit gap was within margin; fix the checker and add its $f_p$ test to CI.
     - Bound staleness with a decoupled objective and correction metrics; alert on ESS/B and mismatch KL.
     - Replace reward-0 truncation with filtering or soft punishment and log truncation rate.
     - Make the audit gap, KL-to-init, entropy slope, and length-by-correctness release-blocking signals.
  7. *Remeasure*: Audit gap with intervals per checkpoint, pass@1 and pass@16 on the release suite, ESS/B, mismatch KL, step time, and truncation rate after mitigation, compared with the pre-migration baseline.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: RL for a SQL-Generation Assistant

A team wants RL post-training for a model that writes SQL against customer schemas. Rewards come from executing queries in a sandbox and comparing results with reference queries; some tasks have no reference and use a learned judge. Responses range from 50 to 6,000 tokens. Two GPU pools are available: one with fast interconnect for training, one older pool suitable only for inference, linked by a 25 Gb/s network. Leadership asks for a 2× faster run than the current synchronous colocated baseline without lowering held-out execution accuracy.

**Required Deliverables**:
1. Estimator specification: advantage estimator, aggregation, clip bounds, ratio level, KL placement and coefficient, with justification and the CLM-043 analysis of when clipping binds.
2. Reward design: checker rules, measured $f_n/f_p$ on policy samples with intervals, adversarial cases (e.g., result-set equality on empty results), judge usage boundaries, and a random-reward control.
3. Over-optimization plan: frozen audit set, KL-to-init logging, train–audit gap monitor, and a pre-declared stopping and rollback rule.
4. Step-time model: per-stage accounting from a baseline profile, length distribution, slot utilization, and predicted effect of length cap, disaggregation, and asynchrony, including weight-sync cost over the link.
5. Off-policy plan: staleness bound and its meaning in the chosen framework, decoupled objective, mismatch measurement at zero staleness, correction choice ($C$, level), and ESS alerts.
6. Pinned verl source trace (or an equivalent framework at a pinned commit) showing where each choice in deliverables 1 and 5 is implemented.
7. Diagnosis and remediation for Incident 27.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace, at the pinned verl commit, advantage computation (GAE and GRPO), clipped loss and aggregation, KL placement, the synchronous `fit` loop including old-log-prob recomputation versus bypass, rollout correction weights and their use in the loss, and weight push. State what was statically inspected and what was executed, and list untraced paths.

### Rubric Dimensions
- **Estimator Reasoning**: *Insufficient* names the algorithm. *Competent* computes advantages, clip outcomes, and token weights for the chosen configuration. *Strong* predicts which tokens move under each aggregation and clip shape, reconciles numbers with the pinned source, and explains entropy dynamics with logged covariance.
- **Reward Validity**: *Insufficient* reports training reward. *Competent* measures $f_n/f_p$ on policy samples. *Strong* plants and detects a loophole, runs a random-reward control, and tracks verifier error over training.
- **Over-Optimization Control**: *Insufficient* adds a KL term. *Competent* logs KL-to-init and an audit. *Strong* pre-declares a stopping rule on the paired audit gap, explains KL placement effects, and states what in-loop signals cannot detect.
- **Systems Accounting**: *Insufficient* reports generation tokens/s. *Competent* builds a per-stage step table. *Strong* predicts and measures the effect of long tails, placement, and sync, and names each change's effect on the estimator.
- **Off-Policy Management**: *Insufficient* enables async with defaults. *Competent* bounds staleness and enables correction. *Strong* separates mismatch from staleness, monitors ESS, and states the bias and deployment gap of the chosen correction.
- **Source Reasoning**: *Insufficient* cites file names. *Competent* traces the execution path by symbol. *Strong* connects config defaults, code paths, and logged metrics to observed run behavior.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Estimator computation and clip analysis | 27.1 | LAB A | Deliverable 1 | Advantage and clip tables; verl parity check |
| Aggregation, clip shape, entropy diagnosis | 27.2 | LAB A, LAB B | Deliverable 1; Incident steps 3–4 | Token-weight tables; entropy vs covariance plot |
| Verifier characterization | 27.3 | LAB B | Deliverable 2; Incident step 4 | $f_n/f_p$ with intervals; adversarial cases; random-reward control |
| Hacking and over-optimization control | 27.4 | LAB B | Deliverable 3; Incident steps 5–6 | Paired audit gap; KL-to-init; stopping rule |
| Step-time accounting and placement | 27.5 | LAB C | Deliverable 4 | Stage table; simulator results; sync cost |
| Staleness and mismatch correction | 27.6 | LAB C, LAB D | Deliverables 5–6; Incident steps 4–7 | Lag histogram; mismatch KL; ESS; source trace |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 27 must be able to:
1. Compute and explain the per-token gradient of a PPO/GRPO-family loss, including when the clip binds.
2. Choose aggregation, clip bounds, and ratio granularity, and diagnose entropy collapse and length pathologies from logs.
3. Measure a verifier's error rates on policy samples and design controls that would reveal a spurious or exploited reward.
4. Treat KL as an optimization budget, place KL deliberately, and stop a run on an independent audit gap.
5. Account RL step time by stage and choose colocated, disaggregated, or asynchronous execution with stated algorithmic side effects.
6. Quantify staleness and sampler/trainer mismatch, configure truncated importance sampling, and monitor ESS, tracing each piece in a pinned production framework.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: An RL loss estimates a policy gradient only for the reward it is given and the data distribution it assumes. Every systems shortcut (asynchrony, a faster sampler, a length cap) changes that distribution, and every reward shortcut changes what is being optimized.
- **The Diagnostic Path**: reward curve → audit gap → KL-to-init → entropy and clip statistics → length by correctness → staleness and mismatch metrics (ESS) → stage timings → resolved config and pinned source.
- Estimator details mostly decide efficiency and stability; the reward decides direction; the system decides how off-policy the data is. Measure all three. This module hands numeric root causes of mismatch to Module 30 and serving-engine internals back to Modules 04 and 20.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
