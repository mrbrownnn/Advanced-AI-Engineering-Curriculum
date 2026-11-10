# Module 17 — Harness Evolution

## 00 Why This Module Exists

A model harness that passed every test in Module 13 still changes. Providers retire snapshots and deprecate request parameters, aliases move to new weights, prompts are edited, optimizers generate new instructions, and tool or output schemas evolve. Each change can alter behavior without any application code change. Harness evolution is the discipline of changing these components deliberately: know exactly what changed, measure which behaviors regressed, and ship through controlled exposure with a rollback that actually restores a valid state.

```text
harness manifest v_n
   |  change (forced: retirement / parameter deprecation | elective: model, prompt, optimizer, schema)
   v
compatibility check -> paired offline evaluation -> (optional) re-optimization + independent confirmation
   |
shadow traffic -> randomized canary vs concurrent control -> staged ramp -> full rollout
   |                                   |
 migration ledger <------------- guardrail breach -> state-aware rollback
```

Module 13 owns the single-version model I/O boundary. Module 15 owns general evaluation programs and release gates; Module 16 owns counterexample search. This module owns **version-to-version change**: manifests, migration evaluation, prompt optimization governance, and progressive rollout. Economics of routing belongs to Module 22 and production SLO operations to Module 23.

**Research cutoff:** 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Change models, prompts, and harness components on your schedule—or a provider's—without shipping hidden regressions or breaking persisted state.
- **What You Will Do**: Build a harness manifest and change classifier, run paired migration evaluations, run and govern an automated prompt optimizer, quantify selection bias, size a canary, design rollback, trace DSPy optimizer source, and diagnose a forced-migration incident.
- **Environment**: Python 3.10+, access to at least two model versions (hosted or local), a frozen evaluation set with a validated grader, DSPy or an equivalent optimizer, and a traffic-replay or simulated canary harness.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**) separate.

## 01 Baseline Assumptions

- Module 13: canonical versus effective requests, provider adapters, capability negotiation, validation ladders, and replay manifests.
- Module 15: evaluation contracts, experimental units, grader validity, paired inference, and release gates.
- Module 16: falsification contracts and discovery-versus-confirmation separation.
- Module 14: durable state and why routing changes do not undo persisted effects.

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
  research_connection: SELECTIVE

estimated_effort:
  instruction: 4h             # lesson instruction: 35+40+40+45+40+35 = 235 min, rounded
  guided_practice: 2h         # six in-lesson practices: 25+20+20+15+20+25 = 125 min, rounded
  labs: 12h                   # LAB A–D: 3h each; LAB C excludes the source trace
  assessment: 4h              # transfer problem 3h + Incident 17.1 1h
  source_trace: 2h            # Lesson 17.4 / LAB C / Section 09 use one activity, counted once
  total: 24h
```

Each category is counted once. Lab C takes 3h plus the separate 2h source trace; the source trace is not counted again as lesson practice or assessment.

The learner must pin every behavior-bearing component, classify changes, evaluate migrations with paired flips, govern automated prompt optimization against selection bias, size and run a canary against a concurrent control, and plan a rollback that accounts for persisted state.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Harness Manifest $\to$ Change Classification (forced / elective) $\to$ Compatibility Check $\to$ Paired Offline Evaluation (block effects, flips, slices) $\to$ Optional Re-optimization (train / validation / independent confirmation) $\to$ Shadow $\to$ Canary vs. Control $\to$ Staged Ramp $\to$ Full Rollout or State-Aware Rollback $\to$ Migration Ledger. Every arrow is a decision with evidence; no single aggregate score authorizes a migration.

## 04 Lessons

### Lesson 17.1 — The Harness Manifest and the Change Surface

**Engineering Question:**
Which components must be versioned so that any production response can be tied to one reproducible configuration, and which changes arrive without our consent?

**Concepts & Definitions:**
- **Harness manifest**: pinned model identifier or snapshot, provider/runtime and region, request parameters, system and task prompts, few-shot demonstrations, tool and output schemas, parser/validator versions, retry/fallback policy, and routing rules.
- **Change surface**: every manifest field whose change can alter the effective request, the model receiving it, or the interpretation of its output (**D**, CLM-001).
- **Forced change**: imposed by a dependency—e.g., under Anthropic's documented lifecycle, deprecated models get a retirement date and replacement, and requests to retired models fail (**O**, CLM-002).
- **Elective change**: chosen by the team—new model, prompt edit, optimizer output, schema change.
- **Alias versus snapshot**: an alias can move to new weights; a dated snapshot identifier is stable until its retirement. Provider-side serving changes behind a stable identifier may be visible only through behavior.

**Mechanism Explanation:**
A manifest hash is attached to every request log. It proves which manifest identity produced a response: identical canonical manifests have the same digest, and any changed canonical field changes the digest. It does **not** prove why behavior changed. Causal attribution needs a controlled contrast, replay, ablation, or another discriminating intervention; provider-side changes and unlogged environment state can remain unknown. A change classifier diffs two manifests and routes the change: parameter or schema changes first go through **request-validity checks** per target; model changes go through capability negotiation (Module 13) and then behavior evaluation. Forced changes carry a deadline, so the ledger tracks lifecycle dates as dependencies (**D**, CLM-001, CLM-017).

A concrete failure class: provider documentation says `temperature`, `top_p`, and `top_k` return a 400 error when set to non-default values on model generation 4.7 and later, while the Python SDK v1.0+ raises `TypeError` because it removes those parameters (**O**, CLM-003). A harness that sets `temperature=0.2` can therefore be valid on an older target and rejected on its replacement—before any quality question arises.

**Quantitative Model / Derivation:**
If $k$ manifest fields may have changed, a one-factor-at-a-time contrast needs the baseline plus $k$ single-change runs. A full factorial over binary old/new fields needs $2^k$ configurations and can identify interaction contrasts under randomized, stable execution. Neither count fixes unlogged provider state or measurement error. A digest distinguishes the configurations that were logged; it supplies no causal contrast by itself (**D**, CLM-017).

*Why these counts.* Each changed field has two states, old or new. One-at-a-time keeps every field at its old value except one, so the runs are: the all-old baseline (1) plus one run per field ($k$), total $k+1$. A full factorial tries every combination of old/new across the $k$ fields: $2\times2\times\cdots\times2=2^k$. Example: $k=2$ gives $3$ one-at-a-time runs but $2^2=4$ factorial cells; $k=5$ gives $6$ versus $32$. Only the factorial contains the "both changed" cell, which is what an interaction needs.

**Worked Example — synthetic interaction fixture:**
Two fields changed: model A→B and prompt A→B. Replay the same 200 frozen, independently sampled items with the same grader under all four cells:

| Model / prompt | Passed items | Success |
|---|---:|---:|
| A / A (baseline) | 192 | 96% |
| B / A | 190 | 95% |
| A / B | 190 | 95% |
| B / B | 140 | 70% |

*Steps.* Each single change is −2/200 = −1 pp. Adding those main effects predicts −2 pp, but the joint change is −52/200 = −26 pp. The difference-of-differences interaction is $(140-190)-(190-192)=-48$ items, or −24 pp. (Read it as: the model change costs $190-192=-2$ items under prompt A, but $140-190=-50$ items under prompt B; the effect of the model depends on the prompt by $-50-(-2)=-48$ items, and $-48/200=-24$ pp. Check: main effects $-1-1=-2$ pp plus interaction $-24$ pp equals the observed $-26$ pp.) *Result.* Hashes identify four cells, but neither the model hash nor prompt hash alone explains the joint regression. *Interpretation / limits.* The fixture supports an interaction under its stable replay assumptions; it does not identify a deeper mechanism. Without the B/A and A/B ablations, the required report is `CAUSE UNKNOWN OR INTERACTING`, not a forced single-component answer. Provider-side drift can keep the cause unknown even after the factorial replay.

**Knowledge Check:**
1. Why is a moving alias a manifest bug even when behavior currently looks unchanged?
2. Which manifest fields can change output interpretation without changing model output?

**Guided Practice:**
Write the manifest for a production endpoint you know. Mark each field as pinned, aliased, or unlogged, and list which forced changes could arrive in the next six months.

**Feedback Contract:**
- *Expected Output*: Complete field list, pin status, lifecycle dates, validity checks, digest verification, and either a one-change contrast or the four-cell interaction table; `UNKNOWN/INTERACTING` when required contrasts are absent.
- *Typical Error*: Calling a matching digest proof that the changed model or prompt caused the regression.
- *Diagnostic Hint*: Which run changes exactly one field, and where is the missing cell needed for a difference-of-differences?
- *Concept to Revisit*: Identity/provenance versus causal contrast; Canonical vs. Effective Request (Module 13).

**Learning Outcome:**
Pin and log every behavior-bearing component and classify a change before evaluating it.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 17.2 — Paired Migration Evaluation and Regression Flips

**Engineering Question:**
How do we decide whether a new model or prompt is compatible when aggregate accuracy looks the same or better?

**Concepts & Definitions:**
- **Paired evaluation**: old and new manifests run on the same frozen items, with the same grader, and—for stochastic outputs—repeated samples per item.
- **Regression flip** ($b$): item passes under old, fails under new. **Improvement flip** ($c$): the reverse.
- **Behavior drift**: Chen, Zaharia, and Zou reported substantial task-dependent changes between March and June 2023 versions of GPT-3.5/GPT-4, including chain-of-thought responsiveness, refusals, and code formatting (**O**, CLM-004).
- **Prompt transfer**: Sclar et al. report up to 76 accuracy points of spread from meaning-preserving few-shot format changes on LLaMA-2-13B and weak cross-model correlation of format performance (**O**, CLM-005). A prompt tuned for one model carries no guarantee for its replacement.

**Mechanism Explanation:**
Build a 2×2 table per slice (task type, language, customer tier, tool path). Report $b$, $c$, net change, and the regression examples themselves. Apply Module 15 Lesson 15.4's release statistics contract: predeclare the independent block, paired effect, interval, margin, critical slices, look/multiplicity policy, and RELEASE/HOLD/INCONCLUSIVE rule. A migration whose goal is compatibility normally uses noninferiority, not a significance test against zero. Because behavior changes are task-dependent, a migration can improve reasoning while breaking output format; slice tables catch this where the global mean does not (**D**, CLM-006, CLM-018).

**Quantitative Model / Derivation:**
With $n$ independent items, $a$ both-pass, $b$ old-pass/new-fail, $c$ old-fail/new-pass, $d$ both-fail:
$$\Delta\text{acc}=\frac{(a+c)-(a+b)}{n}=\frac{c-b}{n},\qquad r_{reg}=\frac{b}{n}$$
Here each item is one independent block and $d_i\in\{-1,0,+1\}$ is new minus old. Concordant items cancel. Equal accuracy implies only $b=c$ (**D**, CLM-006).

*Symbol note.* The count $d$ (both-fail) and the per-item difference $d_i$ are different quantities; $\bar d$ and $s_d$ always refer to $d_i$. Where: old accuracy $=(a+b)/n$ (passes under old), new accuracy $=(a+c)/n$; $r_{reg}$ is the regression-flip rate.

*Step-by-step from the 2×2 table to $\bar d$ and $s_d$.* (1) Score each item $d_i=\text{new}_i-\text{old}_i$: $+1$ for $c$ items, $-1$ for $b$ items, $0$ for the $a+d$ concordant items. (2) Mean: $\bar d=\frac{(+1)c+(-1)b}{n}=\frac{c-b}{n}=\Delta\text{acc}$. (3) Since $d_i^2=1$ exactly for discordant items, $\sum d_i^2=b+c$, so the sample variance is $s_d^2=\frac{\sum d_i^2-n\bar d^2}{n-1}=\frac{(b+c)-n\bar d^2}{n-1}$. (4) Standard error $SE=s_d/\sqrt n$. (5) Interval $[L,U]=\bar d\pm t\cdot SE$, with $t\approx1.961$ for 95% and large $n$.

A two-sided block t interval is $\bar d\pm t_{1-\alpha/2,n-1}s_d/\sqrt n$; a whole-block bootstrap is an alternative. An exact binomial test on $c$ of $b+c$ discordant independent pairs tests no directional change but does not replace the margin gate. If an item has repeated generations, a session has several items, or a user contributes several tasks, first average within that independent block and resample/analyze blocks—not rows (Module 15 CLM-020–022; Module 17 CLM-018).

**Worked Example — synthetic paired fixture:**
*Predeclared contract.* One independently sampled item per block; two-sided 95% block t interval; primary global noninferiority margin $\tau=0.01$; structured-refund critical slice margin $\tau_s=0.02$; one fixed-horizon look; all gates must pass.

*Input and steps.* Across $n=2{,}000$ items, $b=120$, $c=130$, so $\bar d=10/2000=0.005$, $r_{reg}=6\%$, $s_d=0.3536$, $SE=0.00791$, and the 95% interval is $[-0.0105,0.0205]$. Since $L=-0.0105<-0.01$ and $U>-0.01$, the global noninferiority gate is **undetermined**. In the 300-item structured-refund slice, $b=90,c=10$: $\bar d=-0.2667$, $s_d=0.5129$, $SE=0.02961$, interval $[-0.3249,-0.2084]$; because $U<-0.02$, the slice gate is **refuted**.

*Arithmetic shown (global).* $s_d^2=\frac{(120+130)-2000(0.005)^2}{1999}=\frac{249.95}{1999}=0.12504$, so $s_d=0.3536$; $SE=0.3536/\sqrt{2000}=0.3536/44.72=0.00791$; half-width $1.961\times0.00791=0.0155$; interval $0.005\pm0.0155=[-0.0105,0.0205]$. *Slice.* $\bar d=(10-90)/300=-0.2667$; $s_d^2=\frac{100-300(0.2667)^2}{299}=0.2631$, $s_d=0.5129$; $SE=0.5129/\sqrt{300}=0.02961$; half-width $\approx1.968\times0.02961=0.0583$.

*Gate logic in words.* Noninferiority margin $\tau$ means "the new version may be at most $\tau$ worse." Pass if the whole interval sits above $-\tau$ ($L\ge-\tau$); refute if the whole interval sits below $-\tau$ ($U<-\tau$); otherwise the data cannot tell (undetermined/INCONCLUSIVE).

*Result.* Decision = **HOLD**: a critical gate is refuted even though global accuracy rose 0.5 pp. *Interpretation / limits.* Numbers are synthetic. The interval assumes independent items and an approximately normal mean; repeated generations or user clusters require block effects. Planning to show global noninferiority at a true $\Delta=0.005$ with pilot $s_d=0.3536$, $\tau=0.01$, $\alpha=0.05$, and power 0.80 gives $n\ge((1.960+0.842)0.3536/(0.005+0.01))^2=4{,}362.9\to4{,}363$ independent items. (Why this form: the lower bound is about $\bar d-z_{1-\alpha/2}SE$; for it to clear $-\tau$ with probability $1-\beta$ when the true effect is $\Delta$, the distance $\Delta+\tau$ must cover $(z_{1-\alpha/2}+z_{1-\beta})SE=(z_{1-\alpha/2}+z_{1-\beta})s_d/\sqrt n$; solve for $n$. Here $z_{0.975}=1.960$, $z_{0.80}=0.842$, distance $0.015$: $(2.802\times0.3536/0.015)^2=66.05^2$.) Power is the probability that this predeclared gate passes at that planning effect, not the probability that the new harness is better.

**Knowledge Check:**
1. Why does unpaired comparison of two accuracies lose power relative to the discordant-pair analysis?
2. With stochastic decoding, what must be repeated before a flip is counted?

**Guided Practice:**
Given $b=40$, $c=65$ on $n=1{,}500$, compute $\Delta\text{acc}$ and $r_{reg}$, then state what additional slice and severity data a release gate needs.

**Feedback Contract:**
- *Expected Output* (work: $\bar d=(65-40)/1500$; $s_d^2=(105-1500\bar d^2)/1499=0.06977$; $SE=s_d/\sqrt{1500}$; half-width $1.962\,SE=0.0134$): $\Delta\text{acc}=25/1500\approx1.67$ pp; $r_{reg}=40/1500\approx2.67\%$; $s_d=0.2641$, $SE=0.00682$, 95% block t interval $[0.0033,0.0300]$ under independent items; predeclared margins, slice gates, block/repeat policy, and a three-state decision.
- *Typical Error*: Reporting RELEASE from the positive mean or exact sign-test result without testing a practical margin and critical slices.
- *Diagnostic Hint*: How many independent blocks are there, who experiences the 40 regressions, and does every required gate pass?
- *Concept to Revisit*: Module 15 Lesson 15.4: paired block effects, noninferiority, and RELEASE/HOLD/INCONCLUSIVE.

**Learning Outcome:**
Evaluate a migration with paired flips and slice-level regression gates.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 17.3 — Automated Prompt Optimization as Search

**Engineering Question:**
What does a prompt optimizer actually search over, and what does it need from us to produce a trustworthy candidate?

**Concepts & Definitions:**
- **LM program**: declarative modules whose prompts and demonstrations are parameters; DSPy compiles such programs with optimizers that bootstrap demonstrations and tune prompts to maximize a supplied metric (**O**, CLM-007).
- **MIPRO**: optimizes per-module instructions and demonstrations without module-level labels, using program/data-aware proposals and a surrogate model learned from stochastic minibatch evaluations (**O**, CLM-008).
- **GEPA**: reflective prompt evolution that critiques sampled trajectories in natural language and keeps a Pareto set of candidates; its authors report gains over GRPO and MIPROv2 with fewer rollouts on evaluated tasks (**O**, CLM-009; FRONTIER, author-reported).
- **Metric**: the optimizer's only definition of "good". An invalid grader is optimized as faithfully as a valid one.

**Mechanism Explanation:**
All three are search loops: propose candidates → evaluate on (mini)batches → update proposals or surrogate → keep the best by validation score. The search consumes evaluation calls; its output is the argmax of a noisy score. Therefore: (1) metric validity is a precondition (Module 15); (2) data must be split into train (demonstration source), validation (selection), and independent confirmatory evidence (decision); (3) the result must pass the same paired regression gate as any migration. The confirmatory evidence may be a protected random sample from the same declared population or a separately sampled population-shift test; its conclusion applies only to that population, metric, grader, and version. Reusing it to choose candidates turns it back into validation data.

**Quantitative Model / Derivation:**
Optimization cost $\approx N_{trials}\times B_{eval}\times(\text{calls per example})\times(\text{cost per call})$ plus proposal-model calls and full-validation evaluations. Minibatch evaluation lowers cost per trial but raises score noise, which can increase selection bias (Lesson 17.4).

Where: $N_{trials}$ = number of candidate trials; $B_{eval}$ = examples scored per trial (minibatch size); calls per example = LM calls one program run makes (e.g., one per module); cost per call = average price of one LM call.

**Worked Example — pinned-code accounting:**
At the source snapshot traced in Lesson 17.4, MIPROv2 first evaluates the default program on all validation items. With 40 optimization trials, minibatch 35, full evaluation after every five optimization trials plus the implementation's initial full evaluation, 300 validation items, and three LM calls per program evaluation:
- minibatches: $40\times35=1{,}400$ program-example evaluations, or $4{,}200$ LM calls;
- full validation: $\lfloor40/5\rfloor+1=9$ full evaluations, $9\times300=2{,}700$ program-example evaluations, or $8{,}100$ LM calls;
- total: $4{,}100$ program-example evaluations or $12{,}300$ LM calls, before proposal-model calls.

(Step check: "every five trials" over 40 trials gives full evaluations after trials 5, 10, …, 40, i.e., $\lfloor40/5\rfloor=8$, plus the initial default-program evaluation $=9$. Totals: $1{,}400+2{,}700=4{,}100$; $4{,}100\times3=12{,}300$.)

*Result and limits.* Full validation dominates this fixture's task-model calls. This is a static reconstruction of one implementation at one revision, not a universal optimizer formula; non-divisible schedules and automatic run modes alter the count. Measure retries, cache hits, tool calls, and variable module paths in a real budget.

**Knowledge Check:**
1. Why does an optimizer amplify a grader bias instead of averaging it out?
2. Why must re-optimized prompts for a new model still face the paired regression gate?

**Guided Practice:**
Specify an optimization run: program, metric and its validation evidence, train/validation/independent-confirmation sizes and populations, trial budget, and paired block regression gate. State the hypothesis that re-optimization beats prompt carry-over and its falsifier (**H**, CLM-016).

**Feedback Contract:**
- *Expected Output*: Split roles and populations, candidate count, minibatch/full-evaluation schedule, $4{,}100$ program evaluations/$12{,}300$ LM calls for the fixture, proposal-call exclusion, metric validity evidence, independent confirmation policy, and falsifier.
- *Typical Error*: Counting eight full evaluations while omitting the initial baseline, or reporting the optimizer's best validation score as production gain.
- *Diagnostic Hint*: Which line item pays for the default program, and which data never influenced candidate selection?
- *Concept to Revisit*: Grader Validity and Discovery vs. Confirmation (Modules 15–16).

**Learning Outcome:**
Frame prompt optimization as budgeted search against a validated metric with explicit data roles.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 17.4 — Selection Bias, Holdout Reuse, and the Optimizer Source Trace

**Engineering Question:**
By how much does "best validation score" overstate a candidate's true quality, and how do we keep repeated optimization from consuming the holdout?

**Concepts & Definitions:**
- **Winner's curse**: selecting the max of noisy estimates biases the winner's estimate upward (**D**, CLM-010).
- **Adaptive reuse**: each decision informed by the holdout makes later holdout results less valid; Dwork et al. show this can invalidate guarantees and propose a reusable-holdout mechanism under stated conditions (**O**, CLM-011).
- **Independent confirmatory evidence**: protected data or a separately sampled replication that did not influence candidate generation or selection. Its inference is scoped to its declared population, metric, grader, version, and look policy; no single dataset proves production behavior under population shift.

**Quantitative Model / Derivation — toy, not an optimizer model:**
Assume $K$ candidates have the same true score $\theta$ and independent, equal-variance Gaussian validation noise $\sigma Z_k$, $Z_k\sim N(0,1)$. Then the selected score is $\theta+\sigma\max_k Z_k$:
$$\mathbb{E}[\text{selection bias}]=\sigma\,\mathbb{E}[\max_{k\le K}Z_k]\approx 0.564\sigma,\;1.163\sigma,\;1.539\sigma,\;1.867\sigma\quad(K=2,5,10,20)$$
For accuracy on $m$ independent validation items, $\sigma\approx\sqrt{p(1-p)/m}$.

Where: $K$ = number of candidates compared; $\theta$ = their common true score; $Z_k$ = standardized noise of candidate $k$; $\sigma$ = standard deviation of one validation score; $m$ = validation items; $p$ = true accuracy. ($\rho$ below is the correlation between candidates' errors.)

*Step-by-step.* (1) Observed score of candidate $k$: $\hat\theta_k=\theta+\sigma Z_k$. (2) The optimizer keeps the largest: $\max_k\hat\theta_k=\theta+\sigma\max_k Z_k$ (adding the same $\theta$ and scaling by $\sigma>0$ do not change which one is largest). (3) Bias $=\mathbb E[\max_k\hat\theta_k]-\theta=\sigma\,\mathbb E[\max_k Z_k]$. (4) $\mathbb E[\max_k Z_k]$ is a pure number depending only on $K$: $0$ for $K=1$, and growing slowly (roughly like $\sqrt{2\ln K}$) as $K$ rises. (5) $\sigma$ comes from the binomial variance of a mean of $m$ pass/fail items: $\text{Var}=p(1-p)/m$. Intuition: even when nobody is better, the luckiest of $K$ noisy scores looks better.

These assumptions are intentionally narrow. Real optimizer candidates share prompts, demonstrations, data, proposal history, and surrogate feedback; candidate scores are dependent and adaptively generated, true qualities differ, test scores have sampling and grader uncertainty, and the confirmation population can shift. Therefore a validation–confirmation gap decomposes selection bias, genuine quality differences among selected candidates, both samples' noise, grader drift, implementation changes, and population shift. It cannot be attributed entirely to winner's curse (**D**, CLM-010, CLM-019).

**Worked Example — synthetic toy and sensitivity checks (seed 174):**
*Input.* $m=200$, $p=0.80$, $K=20$, so $\sigma=\sqrt{0.16/200}=0.02828$. Numerical integration gives $E[\max Z_k]=1.867$, hence Gaussian-toy bias $=0.0528$ (5.28 pp). (Steps: $p(1-p)=0.8\times0.2=0.16$; $0.16/200=0.0008$; $\sqrt{0.0008}=0.02828$; $1.867\times0.02828=0.0528$.)

*Monte Carlo.* Reproduction note: the toy means use one stream, `numpy.random.default_rng(174)`, with 1,000,000 replications. The binomial, gap, correlated, and unequal-quality checks use a second `default_rng(174)` stream with 400,000 replications, drawn in the order listed; a different draw order gives slightly different digits. One million independent Gaussian replications give means 0.564, 1.163, 1.539, 1.868 standard deviations for $K=2,5,10,20$ (Monte Carlo SE at $K=20$: 0.00053). A separate 400,000-replication binomial version with 20 independent candidates, 200 items each, and true accuracy 0.80 gives mean selection bias 0.0513. Against an independent 500-item confirmation sample from the same population, the gap has mean 0.0513, SD 0.0226, central 95% simulation range $[0.009,0.097]$, and is nonpositive in 0.97% of replications.

*Break assumptions.* With equicorrelated Gaussian errors $\rho=0.5$, the mean maximum falls to 1.322 standard deviations. When one candidate truly scores 0.83 and nineteen score 0.80, the 200-item binomial simulation selects the better candidate 24.7% of the time; apparent gain is 5.48 pp, real selected-candidate gain is 0.74 pp, and selection bias is 4.74 pp. These are toy diagnostics, not predictions for an optimizer.

*Interpretation / limits.* A large positive gap is compatible with selection bias in this toy, but one observed gap does not identify its cause. The confirmatory sample itself has uncertainty: at $m_{test}=500,p=0.8$, its 95% normal half-width is about 3.51 pp ($1.96\sqrt{0.16/500}=1.96\times0.01789=0.0351$). A shifted confirmation population adds a different estimand. Report candidate dependence, adaptive history, both samples' uncertainty, grader versions, and population definitions before explaining the gap.

**Mechanism Explanation — Pinned DSPy Trace (O, CLM-012):**
At `stanfordnlp/dspy` commit `9c900c7de0a3cc3114c23fe8202ebe48e2206ce1` (static re-inspection, 2026-10-01):
1. `MIPROv2.compile(student, trainset=..., valset=None)` calls `_set_and_validate_datasets`, which—when `valset` is omitted—takes the last `min(1000, max(1, int(0.8*len(trainset))))` training examples as validation and the rest as train.
2. `_optimize_prompt_parameters` first evaluates the default program on all validation examples, then tracks `best_score`/`best_program`; in minibatch mode `_perform_full_evaluation` promotes a candidate only after a full validation evaluation.
3. `BootstrapFewShot.compile` → `_bootstrap` iterates the trainset for up to `max_rounds`, keeping traces that pass `metric`/`metric_threshold` as demonstrations.
4. `Evaluate(devset=..., metric=..., failure_score=0.0)` replaces failed executions returned as `None` with `failure_score`.
Implications: the optimizer's reported score is a selection score; independent confirmatory data are not created for you; and errored examples are scored rather than excluded, so an error-rate change moves the metric.

**Knowledge Check:**
1. Why does a larger $K$ make independent confirmatory evidence more, not less, important under the toy assumptions?
2. How can `failure_score` make a flaky provider look like a worse prompt?

**Guided Practice:**
For $m=500$, $p=0.7$, $K=10$, compute the toy expected bias. Then propose a policy: split sizes and population scopes, maximum optimization rounds per confirmation refresh, candidate/dependence logging, and confirmation uncertainty.

**Feedback Contract:**
- *Expected Output*: $\sigma\approx\sqrt{0.21/500}=0.02049$; equal-quality independent-Gaussian toy bias $1.5388\times0.02049=0.0315$ (3.15 pp); candidate dependence/adaptivity, validation and confirmation uncertainty, population scope, grader version, and reuse ledger.
- *Typical Error*: Calling an observed 3.2 pp validation–confirmation gap 3.2 pp of winner's curse, or treating validation as confirmation after 30 adaptive runs.
- *Diagnostic Hint*: Which gap components would remain if all candidates were equal, and which require a shifted population or genuinely unequal candidates?
- *Concept to Revisit*: Discovery vs. Confirmation (Module 16) and Module 15 Lesson 15.4.

**Learning Outcome:**
Quantify selection bias and govern holdout use for repeated prompt optimization.

*(Effort: 45m instruction, 15m practice; source trace 2h)*

---

### Lesson 17.5 — Shadow, Canary, and Staged Rollout

**Engineering Question:**
How large and how long must a canary be to detect the regression we care about, and why is before/after comparison not enough?

**Concepts & Definitions:**
- **Shadow traffic**: duplicate production inputs to the new version without returning its output; measures validity, latency, cost, and offline-graded quality without user impact. Side-effecting tool calls must be stubbed or blocked.
- **Canary**: the SRE Workbook defines canarying as a partial, time-limited deployment evaluated against a control population, recommends starting from SLIs with few metrics, and warns that before/after comparison is confounded by time (**O**, CLM-013).
- **Guardrail metric**: predeclared metric whose breach aborts rollout—validity/parse rate, refusal rate, error rate, latency tail, cost per request, graded quality on sampled traffic, and user-signal proxies.

**Mechanism Explanation:**
Randomize assignment (by user or session, not by request, when conversations are multi-turn) into control and canary at the same time. Compute guardrails per arm. Fast operational stop rules (errors, parse failures, latency) may abort exposure; they do not authorize a ramp. Slow quality and user-outcome gates may authorize a ramp only after their predeclared fixed horizon or valid sequential boundary. A dashboard may display data continuously, but acting on every unadjusted confidence interval is repeated peeking. One planned fixed-horizon look is the default; $L$ planned looks use $\alpha/L$ per Module 15 Lesson 15.4, or use a separately justified always-valid/group-sequential design (**D**, CLM-018).

**Quantitative Model / Derivation:**
For independent Bernoulli outcomes with control success $p_1$, canary success $p_2=p_1-\delta$, two-sided level $\alpha$, and target probability $1-\beta$ that a zero-drop stop rule fires at the planning drop, the equal-allocation normal approximation is (**D**, CLM-014):
$$n_{arm}\approx\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,[p_1(1-p_1)+p_2(1-p_2)]}{\delta^2}.$$
This is an equal-allocation formula. If the canary supplies $n_c$ graded outcomes and control supplies $n_k=r n_c$, then
$$n_c\approx\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\,[p_2(1-p_2)+p_1(1-p_1)/r]}{\delta^2},\qquad n_k=r n_c.$$

Where: $\delta$ = planning drop in success rate; $z_q$ = standard-normal quantile ($z_{0.975}=1.960$, $z_{0.80}=0.842$); $\beta$ = miss probability, so $1-\beta$ is target detection; $n_c$, $n_k$ = graded outcomes in canary and control (subscript $k$ = control, unrelated to candidate count $K$); $r=n_k/n_c$ (unrelated to $r_{reg}$).

*Step-by-step.* (1) The estimated difference $\hat p_2-\hat p_1$ has variance $\frac{p_2(1-p_2)}{n_c}+\frac{p_1(1-p_1)}{n_k}$. (2) The stop rule fires when the upper bound $(\hat p_2-\hat p_1)+z_{1-\alpha/2}SE$ is below $0$. (3) If the true difference is $-\delta$, this happens with probability $1-\beta$ when $\delta=(z_{1-\alpha/2}+z_{1-\beta})SE$. (4) Square and substitute $SE^2$: with $n_c=n_k=n$, $SE^2=[p_1(1-p_1)+p_2(1-p_2)]/n$, giving $n_{arm}$; with $n_k=rn_c$, $SE^2=[p_2(1-p_2)+p_1(1-p_1)/r]/n_c$, giving $n_c$.
Duration is governed by the slower arm. Randomizing by session makes requests clustered: analyze one effect per independent session/block or, only for planning under equal cluster size $m$ and ICC $\rho$, inflate rows by $DE=1+(m-1)\rho$ (here $m$ is requests per session, not Lesson 17.4's validation size, and $\rho$ is the intra-class correlation—how alike two requests from one session are—not the candidate correlation of Lesson 17.4; $L$ in "$L$ planned looks" above is a count, while $L$ in $L\ge-\tau$ is an interval's lower bound). Intuition: with $\rho=0$ each request is new information ($DE=1$); with $\rho=1$ a session's $m$ requests say the same thing once ($DE=m$). The ramp decision itself follows Module 15's margin contract: for paired/block effect $d=\text{canary}-\text{control}$, a noninferiority gate passes only when $L\ge-\tau$; otherwise RELEASE/HOLD/INCONCLUSIVE applies. A sizing calculation at zero threshold is not a release rule, and target power 0.80 means repeated experiments pass that stated rule about 80% of the time under the planning effect—not certainty and not the probability the design is correct.

**Worked Example — synthetic fixed-horizon design and simulation:**
*Inputs.* Traffic = 40 req/s; 5% assigned to canary and 95% control; 10% of each arm is eligible for grading; $p_1=0.95$, $p_2=0.93$, $\alpha=0.05$, target detection rate 0.80 for a 2-pp drop. Assignment unit = independent request for this fixture; one fixed-horizon look; no clustering, delay, grader error, or traffic drift.

*Equal graded arms.* The formula gives $n=2{,}209.46\to2{,}210$ graded outcomes per arm (steps: $0.95(0.05)+0.93(0.07)=0.0475+0.0651=0.1126$; $(z_{0.975}+z_{0.80})^2=2.8016^2=7.849$ with unrounded quantiles; $7.849\times0.1126/0.02^2=2{,}209.46$; round up). Canary graded rate is $40(0.05)(0.10)=0.2$/s. Rather than grading all sampled control requests, draw a randomized graded-control subsample at 0.2/s—0.5263% of the 38 control req/s—so both arms contribute 2,210. Duration $=2210/0.2=11{,}050$ s = 3.07 h. A 20-minute horizon yields only 240 graded outcomes per arm.

*Natural unequal arms.* If 10% of both traffic arms is graded, $r=3.8/0.2=19$ (control graded rate $40(0.95)(0.10)=3.8$/s). The unequal formula gives (bracket $0.0651+0.0475/19=0.0676$; $7.849\times0.0676/0.0004=1{,}326.46$; $n_k=19\times1{,}327$; duration $1{,}327/0.2$) $n_c=1{,}326.46\to1{,}327$ canary and $n_k=25{,}213$ control outcomes, reached in 6,635 s = 1.84 h. Equal-allocation arithmetic must not be attached to these 5%/95% counts.

*Seeded Monte Carlo check.* The following is synthetic; NumPy `default_rng(175)`, 400,000 replications per cell, Wald two-sided 95% bounds, independent binomial arms. `detection` is the fraction whose upper bound for canary-minus-control is below zero when the canary is truly 2 pp worse. `false_alarm` is the same stop under no drop. `ramp_pass_no_drop` is the noninferiority pass rate for $\tau=0.02$ under no drop; `false_ramp` is its pass rate when the true drop is exactly 2 pp.

```python
import numpy as np
from scipy.stats import norm
rng = np.random.default_rng(175); R, z = 400_000, norm.ppf(0.975)
def rates(n_can, n_ctl, p_can, p_ctl=0.95, tau=0.02):
    xc = rng.binomial(n_can, p_can, R) / n_can
    xk = rng.binomial(n_ctl, p_ctl, R) / n_ctl
    se = np.sqrt(xc*(1-xc)/n_can + xk*(1-xk)/n_ctl)
    lo, hi = xc-xk-z*se, xc-xk+z*se
    return (hi < 0).mean(), (lo >= -tau).mean()
for name, nc, nk in [("equal 20 min",240,240), ("equal target",2210,2210),
                     ("unequal 20 min",240,4560), ("unequal target",1327,25213)]:
    stop_drop, pass_drop = rates(nc, nk, 0.93)
    stop_null, pass_null = rates(nc, nk, 0.95)
    print(name, stop_drop, stop_null, pass_null, pass_drop)
```

| Fixed horizon | Graded canary/control | Detection | False alarm | Ramp pass, no drop | False ramp at −2 pp |
|---|---:|---:|---:|---:|---:|
| Equal, 20 min | 240 / 240 | 0.146 | 0.024 | 0.174 | 0.027 |
| Equal, target | 2,210 / 2,210 | 0.801 | 0.025 | 0.862 | 0.026 |
| Unequal, 20 min | 240 / 4,560 | 0.170 | 0.011 | 0.335 | 0.049 |
| Unequal, target | 1,327 / 25,213 | 0.821 | 0.017 | 0.880 | 0.033 |

Monte Carlo SE is at most 0.00064 for these rates. *Result.* Twenty minutes does not reach the 0.80 target; it still detects some seeded drops, so “cannot detect” is false. Target sizing is approximate, and discreteness/Wald behavior shifts the observed rates. At a true value exactly on the $-\tau$ ramp boundary, a two-sided 95% gate should pass only near its one-sided false-pass rate, not 80%.

*Peeking check.* Ten unadjusted cumulative looks at 221 outcomes per equal arm (seed 175, 200,000 replications) produce detection 0.868 and false-alarm 0.097; one final look gives about 0.801 and 0.025. Bonferroni $\alpha/10$ per look lowers these to 0.580 and 0.012 at the same maximum sample, so a multi-look design needs larger planned sample or another valid sequential method. Ramp stage changes are operational exposure controls, not extra uncounted statistical looks.

*Cluster check.* With four graded requests per session and synthetic ICC 0.3, $DE=1.9$ ($1+(4-1)0.3$). The 2,210-row equal-arm plan contains only 553 sessions per arm ($2{,}210/4=552.5$, rounded up); inflated, $2{,}210\times1.9\approx4{,}200$ rows $=1{,}050$ sessions. A session-level simulation (seed 1750, 40,000 replications) gives detection 0.532 and false alarm 0.025; inflating to 1,050 sessions/4,200 rows per arm gives 0.803 and 0.026. Treating rows as independent instead produces false alarms about 0.078. *Limits.* Real graders, delays, attrition, changing traffic, unequal cluster sizes, and low event rates require pilot estimates and a design-specific method.

**Knowledge Check:**
1. Why is randomization by session preferred for multi-turn assistants, and what does it cost statistically?
2. Why can shadow traffic not validate user-facing quality alone?

**Guided Practice:**
Design a four-stage rollout (shadow → 1% → 10% → 50% → 100%) with per-stage guardrails, sample-size targets, abort thresholds, and who can override.

**Feedback Contract:**
- *Expected Output*: Assignment/block unit; fixed-horizon or valid sequential policy; equal graded-control subsample or unequal variance; $n_c,n_k$, rates, duration, clustering adjustment; stop rules separate from ramp gates; RELEASE/HOLD/INCONCLUSIVE and override record.
- *Typical Error*: Using equal-arm $n$ with 5%/95% graded counts, peeking at nominal intervals, or claiming a short canary has zero detection probability.
- *Diagnostic Hint*: Which arm reaches its required graded count last, how many independent blocks—not rows—exist, and which rule authorizes exposure rather than merely stopping it?
- *Concept to Revisit*: Module 15 Lesson 15.4 and Offline–Online Validity.

**Learning Outcome:**
Design a powered, concurrently controlled rollout with predeclared guardrails.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 17.6 — Compatibility, Rollback State, and the Migration Ledger

**Engineering Question:**
What must remain true for "rollback" to restore a valid system, and how do we meet forced migration deadlines without skipping evidence?

**Concepts & Definitions:**
- **Rollback state**: in-flight sessions, cached prompt prefixes, stored structured outputs, tool-call schemas, and downstream consumers of new-version artifacts (**D**, CLM-015).
- **Dual-read compatibility**: the old version can read artifacts written by the new one (or they are migrated/quarantined).
- **Migration ledger**: per change—manifest diff, change class, lifecycle deadline, compatibility checks, paired offline results, optimizer runs and candidate histories, independent-confirmation result and population, shadow/canary evidence, rollout stage, and rollback decisions.

**Mechanism Explanation:**
Routing controls only future requests. If the new version wrote `order_v4` JSON, the old parser must accept it or the records must be converted. If a conversation started on the new model, rolling back mid-session changes persona and format; pin sessions to a version or define a hand-off. Forced migrations are scheduled backward from the retirement date: compatibility and offline evaluation first, then enough canary time to reach the powered sample size, plus a contingency window. Anthropic documents at least 60 days' retirement notice for publicly released models on its platforms (**O**, CLM-002); partner platforms set their own schedules.

**Quantitative Model / Trade-off Comparison:**
Latest safe start $=T_{retire}-(T_{compat}+T_{offline}+T_{reopt}+\sum_s T_{stage,s}+T_{contingency})$, where each canary stage time comes from Lesson 17.5. Where: $T_{retire}$ = days from notice to retirement; each other $T$ = planned duration of that phase; $\sum_s T_{stage,s}$ adds the durations of all canary stages $s$. In words: count backward from the retirement date by the total work. If the latest safe start is already past, the plan must cut scope (fewer slices, larger $\delta$) explicitly and record the accepted risk.

**Worked Example — synthetic schedule:**
Retirement in 60 days. Compatibility 3 d, offline paired evaluation 5 d, re-optimization and independent confirmation 7 d, four canary stages needing 2, 3, 4, 4 d, contingency 10 d: total 38 d ($3+5+7+(2+3+4+4)+10$). Latest safe start is day 22 after notice ($60-38$). Starting on day 40 leaves only 20 d ($60-40$), an 18 d shortfall ($38-20$). Skipping re-optimization saves 7 d but still leaves an 11 d shortfall ($31-20$); the team must also change exposure/evidence scope, negotiate timing, or record explicit risk acceptance. It cannot imply that a coarser $\delta$ alone closes the schedule.

**Knowledge Check:**
1. Give two artifacts that survive a traffic rollback.
2. Why should forced-migration plans be scheduled backward from the retirement date?

**Guided Practice:**
For an agent that stores tool-call plans in a durable workflow (Module 14), list every artifact the new version writes and the rollback action for each.

**Feedback Contract:**
- *Expected Output*: Artifact inventory, reader compatibility, migration/quarantine steps, session pinning, ledger entries, total 38 d, latest safe start day 22, and the 18 d shortfall for a day-40 start.
- *Typical Error*: Declaring rollback complete when the router flag flips, or saying one omitted 7 d activity resolves an 18 d shortfall.
- *Diagnostic Hint*: What did the new version write that someone else will read tomorrow, and how many calendar days remain after every retained activity?
- *Concept to Revisit*: Durable State and Effects (Module 14); backward scheduling.

**Learning Outcome:**
Plan state-aware rollback and schedule forced migrations with explicit evidence and accepted risk.

*(Effort: 35m instruction, 25m practice)*

---

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [DSPy](https://arxiv.org/abs/2310.03714) — Khattab et al., 2023; declarative LM programs compiled by metric-driven optimizers.
- [How is ChatGPT's behavior changing over time?](https://arxiv.org/abs/2307.09009) — Chen, Zaharia, Zou, 2023; longitudinal behavior change behind stable product names.
- [Quantifying LMs' Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324) — Sclar et al., ICLR 2024; format sensitivity and FormatSpread.
- [The reusable holdout](https://www.science.org/doi/10.1126/science.aaa9375) — Dwork et al., Science 2015; adaptive data analysis.
- [Canarying Releases](https://sre.google/workbook/canarying-releases/) — Google SRE Workbook; canary versus control.

**RECOMMENDED ENGINEERING BASELINE** (module derivation): full-manifest identity; controlled contrasts for causal claims; paired/block migration effects; independent confirmatory evidence for selected optimizer output; concurrent randomized canary; predeclared fixed-horizon or valid sequential gates; state-aware rollback; lifecycle-date tracking. These are recommendations (**D**, CLM-001, CLM-006, CLM-010, CLM-014, CLM-015, CLM-017–019), not defaults of every tool or evidence of industry adoption.

**ONE PROVIDER'S DOCUMENTED BEHAVIOR:** the linked lifecycle page defines status, retirement, notices, platform scope, and parameter handling (**O**, CLM-002–003). Other providers and partner platforms differ.

**ONE IMPLEMENTATION SNAPSHOT:** the DSPy source trace below records split, selection, bootstrap, and failure-score behavior at one commit (**O**, CLM-012). It is not an industry default.

**INDUSTRY PREVALENCE:** not established. No adoption survey for full manifests, paired migration gates, confirmatory prompt-optimizer tests, or statistically planned LLM canaries was opened (`TODO_VERIFY`, CLM-020).

**WORKLOAD-DEPENDENT:** [MIPRO](https://arxiv.org/abs/2406.11695) (Opsahl-Ong et al., EMNLP 2024) and other optimizers; split sizes; independent block; margins; canary $\delta$ and stage durations; session pinning; slice definitions.

**FRONTIER:** [GEPA](https://arxiv.org/abs/2507.19457) (Agrawal et al.; arXiv v2, 2026; accepted ICLR 2026 Oral) uses natural-language reflection over sampled trajectories and a Pareto frontier; abstract re-opened 2026-10-01. Its comparative gains and rollout reductions are author-reported on six evaluated tasks, not reproduced here and not a production default (**O**, CLM-009).

**RECENT MIGRATION EVIDENCE:** [What Aggregate Scores Miss](https://arxiv.org/abs/2608.17719) (Xu and Wu, arXiv v1, 2026) reports coexisting improvements and regressions across nine commercial API migration/benchmark cells, using 900 public items, 50 samples per item, practical-equivalence thresholds, and false-discovery-rate control. Abstract opened 2026-10-01; results are author-reported and model/benchmark-specific (**O**, CLM-021).

**LEGACY / INSUFFICIENT:** floating aliases in production; prompt-only versioning; manifest hash as causal proof; before/after dashboards as canary; optimizer validation score as expected gain; attributing every validation–confirmation gap to winner's curse; rollback by router flag only.

**PRODUCTION SOURCE TRACE**
- Repository: `stanfordnlp/dspy`
- Revision: `9c900c7de0a3cc3114c23fe8202ebe48e2206ce1`
- Verified: 2026-09-27; files and symbols below statically re-inspected at the same revision on 2026-10-01; nothing executed.
- Files/symbols: `dspy/teleprompt/mipro_optimizer_v2.py::{MIPROv2.compile, MIPROv2._set_and_validate_datasets, MIPROv2._optimize_prompt_parameters, MIPROv2._perform_full_evaluation}`, `dspy/teleprompt/bootstrap.py::{BootstrapFewShot.compile, BootstrapFewShot._bootstrap, BootstrapFewShot._bootstrap_one_example}`, `dspy/evaluate/evaluate.py::{Evaluate.__init__, Evaluate.__call__}`.
- Entry paths: `MIPROv2.compile` → dataset validation/split → default full evaluation → trial/minibatch evaluations → periodic full evaluation → best program; `BootstrapFewShot.compile` → `_bootstrap` → `_bootstrap_one_example` → `_train`; `Evaluate.__call__` → `ParallelExecutor.execute` → failed result replacement with `failure_score`.
- Scope: one upstream snapshot. Static inspection establishes code paths, not optimizer quality, runtime cost, or prevalence.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Manifest, Change Classifier, and Compatibility Gate
- **Objective**: Build a manifest schema, per-request manifest hashing, and a classifier that diffs manifests and runs request-validity checks per target.
- **Pre-Registered Hypothesis**: For one-at-a-time seeded changes with stable replay conditions, the changed field plus a controlled baseline/new contrast is sufficient to localize the seeded component; a digest alone is not. For multiple changes, attribution remains unknown or interacting unless ablations discriminate it.
- **Independent Variables**: Manifest completeness, one-at-a-time versus joint changes, change type (model, parameter, prompt, schema, parser), alias versus snapshot.
- **Dependent Variables**: Identity-reconstruction accuracy, causal classification (`single`, `interaction`, `unknown`), invalid-request rate caught pre-deploy, time to classify.
- **Break & Falsify**: (1) Seed an alias move and deprecated parameter one at a time; if the gate passes either, compatibility checking is incomplete. (2) Reproduce Lesson 17.1's A/A, B/A, A/B, B/B interaction; omit one ablation and verify that the report refuses a single-cause claim. (3) Add an unlogged provider-side change; the correct result is `UNKNOWN`, even with a complete local manifest.
- **Alignment**: Lesson 17.1; Mastery Deliverables 1–2; Incident 17.1 steps 1, 3–5.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB B — Paired Migration Report
- **Objective**: Evaluate two model versions on a frozen set with repeated samples and produce block-level slice gates and flip tables.
- **Pre-Registered Hypothesis**: At least one predeclared critical slice refutes its noninferiority margin while the global gate is undetermined or passes.
- **Independent Variables**: Model version, prompt (carried over versus format variants), slice, repeats per item, independent block definition.
- **Dependent Variables**: $a,b,c,d$ per slice, block effects, interval, $\Delta\text{acc}$, $r_{reg}$, gate state, format-variant spread.
- **Statistical Contract**: Reproduce Lesson 17.2's synthetic fixture. Then add five repeats per item or several items per user; report row and whole-block intervals side by side and use only the block result for RELEASE/HOLD/INCONCLUSIVE. Predeclare margins, one look, and all-must-pass gates.
- **Break & Falsify**: Swap in unpaired and row-independent comparisons and show what information/uncertainty they lose; if no gate refutes, report that valid negative result and whether the design reached its target power—never call it proof of no regression.
- **Alignment**: Lesson 17.2; Module 15 Lesson 15.4; Mastery Deliverable 3.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Optimizer Under Holdout Governance
- **Objective**: Run an automated prompt optimizer (DSPy or equivalent) with explicit train/validation/independent-confirmation roles and characterize—not assume the cause of—the validation–confirmation gap.
- **Pre-Registered Hypothesis**: Under the equal-quality independent-noise toy, the mean selected validation optimism follows $\sigma E[\max Z]$; an actual optimizer need not match because candidates are adaptive, dependent, and unequal.
- **Independent Variables**: Candidate count $K$, validation size $m$, candidate dependence/adaptivity, confirmation population, metric version (valid versus deliberately gamed).
- **Dependent Variables**: Validation and confirmation scores with intervals, gap, paired flips versus carried-over prompt, candidate correlation/history, call cost, grader/error rate.
- **Required Tests**: Reproduce Lesson 17.4's seeded toy; add correlated and unequal-quality cases; run a same-population independent confirmation and one declared population-shift confirmation. Attribute no observed gap entirely to winner's curse. Test CLM-016 by paired/block comparison against carry-over.
- **Break & Falsify**: Optimize a gameable metric (e.g., keyword presence); evidence against the pre-registered hypothesis includes no mean toy bias across replications or a better confirmation result for carry-over. One noisy reversal is not proof.
- **Alignment**: Lessons 17.3–17.4; Mastery Deliverables 4 and 7.
- **Effort Estimate**: 2h experiment, 1h analysis (3h total), plus the separately counted 2h source trace.

### LAB D — Powered Canary and State-Aware Rollback
- **Objective**: Simulate shadow and staged canary with a concurrent control, then execute rollback with persisted artifacts.
- **Pre-Registered Hypothesis**: Under Lesson 17.5's independent Bernoulli fixture, the target fixed horizon reaches approximately 0.80 detection while the 20-minute horizon has materially lower—but nonzero—detection; false-alarm rates remain near the predeclared level only when looks and blocks are handled correctly.
- **Independent Variables**: Equal graded-control subsample versus natural 5%/95% graded allocation, horizon, canary fraction, graded fraction, request versus session assignment, seeded $\delta$, fixed versus repeated-look policy.
- **Dependent Variables**: Graded $n_c/n_k$, detection rate, false-alarm rate, noninferiority pass/refute/inconclusive rates, time to decision, rollback completeness.
- **Required Simulation**: Recompute Lesson 17.5 with seed 175 and report Monte Carlo SE; compare short versus target horizons under both allocation designs. Add ten unadjusted looks versus $\alpha/10$, then four-request sessions with ICC 0.3 analyzed by rows and sessions. Keep stop rules separate from ramp gates.
- **Break & Falsify**: Run before/after comparison across a seeded traffic-mix shift; write `v_new` artifacts before rollback and verify the old parser; any unreadable artifact falsifies rollback completeness. A result where the target design misses its declared rate beyond simulation uncertainty falsifies its planning model for that fixture.
- **Alignment**: Lessons 17.5–17.6; Incident 17.1 steps 3–7; Mastery Deliverables 5–6.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

---

## 07 Break / Incident Scenarios

### Incident 17.1 — The Deadline Migration

- **Incident Symptoms**: Two days before a provider retirement date, the team switches the model identifier. Offline accuracy on the regression suite rose 1.2 pp. Within an hour of full rollout: structured-extraction parse failures rise from 0.3% to 4%, refunds processed by a downstream service drop, a subset of requests return HTTP 400, and support reports a changed tone in ongoing conversations. Rolling back is impossible for the retired model; rolling to an intermediate model restores tone but parse failures persist for records written in the last hour.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: deprecated decoding parameter; prompt format sensitivity on the new model; schema drift in outputs; grader/suite not covering the extraction slice; downstream consumer tied to old output quirks; session carry-over across versions; unrelated traffic-mix shift.
  2. *Rank Initial Plausibility*: Use the timing of 400s versus parse failures; do not assume one cause.
  3. *Identify Missing Evidence*: Manifest diffs, per-request manifest hashes, 400 response bodies, paired flips by slice, sample failed parses, artifact versions written, session start versions, canary records (if any).
  4. *Design Discriminating Tests*: Replay failing inputs under old/new manifests with parameter stripped; format-variant spread; parser against stored records; session-level split.
  5. *Execute Causal Diagnosis*: Rank interacting causes and state which alternatives the evidence excludes. Treat manifest hashes as identity/provenance only. If replay/ablation cannot separate model, prompt, schema, and traffic effects, report `UNKNOWN/INTERACTING` rather than inventing one root cause.
  6. *Prescribe Mitigation and Prevention*: Adapter parameter fix, slice-specific prompt or re-optimization with independent confirmation, artifact repair/quarantine, session pinning, backward-scheduled migration plan, powered canary.
  7. *Remeasure*: Parse rate, 400 rate, slice flips, downstream refund reconciliation, and artifact readability.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Multi-Model Estate Under Forced Retirements

You own twelve LLM endpoints across two providers, using five model snapshots. Two snapshots receive retirement notices with 60 and 90 days' notice. Three endpoints persist structured outputs; one is a multi-turn agent with durable workflows. The team wants to use the forced migrations to adopt an automated prompt optimizer.

**Synthetic transfer fixture (exercise assumptions, not provider measurements):**

| Input | Value |
|---|---|
| Endpoint traffic | 30 req/s; one multi-turn endpoint contributes four graded requests/session with pilot ICC 0.3 |
| Assignment / grading | 1%, 10%, 50% canary stages; 5% of each traffic arm eligible for grading; concurrent randomized control |
| Quality planning | control 0.96, planning canary 0.94, $\delta=0.02$, $\alpha=0.05$, target detection 0.80 |
| Parse-failure planning | control 0.3%, unsafe canary 1.3%; fast operational stop metric |
| Release contract | one fixed-horizon look per evidence refresh; quality noninferiority $\tau=0.02$ plus critical-slice $\tau_s$ set before results; every gate must pass |
| Offline blocks | 2,000 independent items plus a critical structured-output slice; repeated generations stay inside item block |
| Optimizer | 30 recorded candidates; 400 independent validation blocks; confirmation is a fresh 600-block sample from a declared target population |
| State | `order_v3` old reader; candidate writes `order_v4`; durable sessions pin manifest at session start |
| Timeline inputs | compatibility 4 d, offline 6 d, optimization/confirmation 8 d, staged rollout 16 d, contingency 12 d |

Every replacement of a fixture value must identify measurement method, model/runtime version, population, and uncertainty. Unknowns stay symbolic; do not convert them into guarantees.

**Required Deliverables**:
1. **Manifest inventory and identity evidence**: pin status, lifecycle dates, per-request digest, unlogged state, and the exact claim a digest can/cannot support.
2. **Change classification and causal plan**: compatibility checks for each endpoint; one-at-a-time seeded tests; four-cell replay/ablation for one joint model+prompt regression; `UNKNOWN/INTERACTING` policy.
3. **Paired release record**: block definition, paired effects, whole-block interval, practical/noninferiority margins, critical slices, multiplicity/look policy, and RELEASE/HOLD/INCONCLUSIVE; include one positive-mean case that is HOLD or INCONCLUSIVE.
4. **Optimizer governance**: train/validation/confirmation populations, 30-candidate call budget, candidate dependence/adaptive history, toy winner's-curse estimate, confirmation uncertainty and shift limits. No full gap attribution to winner's curse.
5. **Rollout design**: shadow plus 1%/10%/50% stages; show equal graded-control subsampling and natural unequal allocation; calculate $n_c,n_k$, duration, block inflation for the multi-turn endpoint, stop/ramp rules, and look adjustment.
6. **State-aware rollback and timeline**: reader/writer compatibility table, session pinning, repair/quarantine, ledger, latest-safe-start arithmetic for both notices, named risk acceptance if evidence cannot fit.
7. **Pinned production source trace**: files, symbols, entry path, revision, static/executed scope, split behavior, candidate promotion, demonstration filtering, error scoring; reconcile the 30-candidate plan with traced behavior.
8. **Incident diagnosis**: complete Incident 17.1 steps 1–7 with competing and interacting causes, missing evidence, discriminating replay/ablation, remediation, and quantitative remeasurement.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Trace the pinned DSPy path from `MIPROv2.compile` through dataset splitting, initial/default evaluation, trial evaluation, and full-validation best-candidate promotion; then trace `BootstrapFewShot` demonstration filtering and `Evaluate` error scoring. State files, symbols, entry path, revision, what was statically inspected, and what was not executed.

### Reference Checks (synthetic fixture only)
- **Quality, independent requests, equal arms**: $n=(1.960+0.842)^2[0.96(0.04)+0.94(0.06)]/0.02^2=1{,}860.18\to1{,}861$ graded outcomes/arm (1,860.18 uses unrounded quantiles, $2.8016^2$; the displayed rounded $2.802^2$ gives 1,860.74; both round up to 1,861). At 1%, 10%, 50% canary with 5% grading, canary rates are 0.015, 0.15, 0.75/s ($30\times$ stage fraction $\times0.05$; duration $=1{,}861/\text{rate}$, e.g., $1{,}861/0.015=124{,}067$ s $=34.46$ h); equal-arm durations are 34.46 h, 3.45 h, 0.689 h. These are stage-specific horizons if each stage starts fresh; cumulative reuse requires a predeclared sequential design.
- **Parse failure, equal arms**: for 0.3% versus 1.3%, the same approximation gives 1,242 outcomes/arm ($\delta=0.01$; bracket $0.003(0.997)+0.013(0.987)=0.01582$; $7.849\times0.01582/0.0001=1{,}241.8$). At 1% canary, all canary traffic—not only 5% quality grading—can supply parse outcomes at 0.3/s, so duration is 69 min ($1{,}242/0.3=4{,}140$ s). If all 99% control traffic is used, unequal-allocation sizing gives about 1,010 canary outcomes and 56.1 min ($r=99$; bracket $0.01283+0.00299/99$; $1{,}010/0.3=3{,}367$ s). This operational metric can stop exposure; it does not make the quality gate pass.
- **Multi-turn blocks**: $DE=1+(4-1)0.3=1.9$ is a planning approximation; quality rows become $1{,}861(1.9)=3{,}536$ per arm, about 884 sessions. Analysis must use session effects, not row-independent intervals.
- **Optimizer toy**: with $m=400,p=0.75,K=30$, $\sigma=\sqrt{0.1875/400}=0.02165$ and $E[\max Z]\approx2.043$, giving 4.42 pp under the equal-quality independent-Gaussian toy only. Confirmation at $m=600,p=0.75$ has a normal 95% half-width about 3.46 pp before pairing/block gains or grader error.
- **Timeline**: $4+6+8+16+12=46$ d. Latest safe start is day 14 for the 60-day notice and day 44 for the 90-day notice ($60-46$, $90-46$).

### Rubric Dimensions
- **Versioning and Change Control**: *Insufficient* versions prompts only or calls a digest causal proof. *Competent* pins manifests, classifies changes, and limits the digest to identity. *Strong* catches forced incompatibilities, uses replay/ablation for joint changes, and reports unknown/interacting causes when evidence cannot discriminate.
- **Migration Evidence**: *Insufficient* compares aggregates or rows as independent. *Competent* reports paired block effects, intervals, margins, slice gates, and a three-state decision. *Strong* demonstrates row-versus-block consequences, declares looks/multiplicity, and distinguishes planning power from certainty.
- **Optimization Governance**: *Insufficient* reports validation gains or attributes the whole gap to winner's curse. *Competent* uses independent confirmatory evidence with population scope and uncertainty. *Strong* records dependence/adaptivity, reproduces and breaks the toy, audits cost/errors, and limits reuse.
- **Rollout and Rollback**: *Insufficient* uses before/after dashboards or equal-arm arithmetic on 5%/95% counts. *Competent* runs a concurrent fixed-horizon canary with correct graded counts and state-aware rollback. *Strong* handles unequal allocation, session blocks, looks, stop versus ramp rules, persisted artifacts, and evidence-limited outcomes.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Manifest identity, change classification, causal limits | 17.1 interaction fixture | LAB A tests 1–3 | Mastery Deliverables 1–2; Incident 17.1 steps 1, 3–5; Versioning rubric | Manifest/digest inventory, compatibility report, four-cell ablation, `UNKNOWN/INTERACTING` disposition |
| Paired block migration gates | 17.2 worked release record | LAB B statistical contract | Mastery Deliverable 3; Incident steps 3–5; Migration Evidence rubric | Flip/slice tables, block effects and interval, margins, look policy, RELEASE/HOLD/INCONCLUSIVE record |
| Prompt optimization as budgeted search | 17.3 pinned-code accounting | LAB C optimizer run | Mastery Deliverable 4; Optimization rubric | Split/population contract, candidate and call ledger, grader/error report |
| Selection bias and confirmation governance | 17.4 toy and sensitivity checks | LAB C required tests; source trace | Mastery Deliverables 4 and 7; Optimization rubric | Seeded toy output, dependence/adaptivity ledger, validation–confirmation intervals and scoped interpretation, pinned trace |
| Fixed-horizon / sequential canary rollout | 17.5 allocation and simulations | LAB D required simulation | Mastery Deliverable 5; Incident steps 3–7; Rollout rubric | $n_c/n_k$, duration, session-block and look analysis, stop/ramp gates, decision record |
| State-aware rollback and migration ledger | 17.6 timeline | LAB D rollback test | Mastery Deliverable 6; Incident steps 3–7; Rollout rubric | Reader/writer compatibility table, repair/quarantine proof, latest-safe-start timeline and ledger |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 17 must be able to:
1. Pin every behavior-bearing component, use hashes as identity evidence, and use contrasts/ablations—or report unknown/interacting causes—for causal attribution.
2. Detect forced incompatibilities (retired models, deprecated parameters) before deployment.
3. Report paired block effects and regression flips by slice; apply predeclared margin, look, and three-state release rules.
4. Run a prompt optimizer with independent confirmatory evidence; quantify a labeled selection-bias toy without attributing the whole observed gap to it.
5. Size and run a concurrently controlled canary with explicit graded allocation, independent blocks, stop/ramp separation, and fixed or valid sequential looks.
6. Execute a rollback that keeps persisted artifacts valid.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: A harness is a versioned configuration; behavior changes whenever any component changes, whether or not you changed it.
- **The Change Path**: manifest diff → validity → paired flips → governed optimization → shadow → powered canary → ramp or state-aware rollback → ledger.
- Aggregate scores approve nothing on their own; paired block gates, independent scoped confirmation, concurrent controls, and rollback-state checks provide complementary—not absolute—evidence.

### Cost & Telemetry Hook
- [Lesson 22.3](../22-ai-economics/README.md): re-price every migration candidate with dated prices; reconcile canary spend against invoice.
- [Lesson 22.6](../22-ai-economics/README.md): compare candidate manifests on a constrained cost/quality frontier and record flip points.
- [Lesson 23.1](../23-observability-reliability/README.md): emit the manifest hash and model ID on every span so canary and control are separable.
- [Lesson 23.4](../23-observability-reliability/README.md): gate canary ramp on offered-outcome SLI burn rate, not before/after dashboards.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
