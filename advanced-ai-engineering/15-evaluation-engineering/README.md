# Module 15 — Evaluation Engineering

## 00 Why This Module Exists

An evaluation score is not evidence until its decision, population, unit, system boundary, data, grader, uncertainty, and failure policy are explicit. Evaluation engineering builds that chain and keeps it reproducible as models, prompts, tools, traffic, and judges change.

```text
decision + risk tolerance
          |
population -> sampled/versioned items -> system under test -> outcomes
          |                                  |                 |
       slices                            attempts/effects    failures too
          \__________________________________|_________________/
                                             |
                          graders -> estimates -> uncertainty
                                             |
                            gate -> shadow/canary/A-B -> decision
                                             ^
                         incidents, drift, fresh data, user outcomes
```

This module owns the evaluation program: contracts, datasets, grader validation, human and LLM judging, statistical comparison, regression gates, benchmark freshness, end-to-end outcome accounting, and offline-to-online validity. Module 00 owns general evidence foundations; Module 07 owns behavioral calibration; Module 12 evaluates agent-loop behavior but owns loop mechanics; Module 16 deepens falsification and adversarial test design; Module 17 reuses the release statistics defined in Lesson 15.4; Module 23 owns production observability and SLO operations.

**Research cutoff:** 2026-09-26 for claims unchanged since registry 1.0.0. Claims added or changed in registry 1.1.0 were checked against sources opened on 2026-10-01, with a landscape cutoff of 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Produce decision-relevant evidence whose population, measurement instrument, uncertainty, and release rule remain reproducible as the system changes.
- **What You Will Do**: Specify an evaluation contract, build a versioned sliced dataset, calibrate graders, run paired dependence-aware comparisons, account for all terminal outcomes, trace the evaluation harness, and defend an offline-to-online release gate.
- **Environment**: Python 3.10+ with NumPy and SciPy, immutable dataset/run manifests, deterministic and model-based graders, optional blinded human labels, and a statistical notebook or test harness.
- **Evidence Rule**: **O** is a source observation, **D** is a derivation under stated assumptions, and **H** is an engineering hypothesis requiring telemetry. `CLM-xxx` identifiers point to `research-registry/15-evaluation-engineering.yaml`. Every dataset in this module's worked examples is **synthetic** and is labeled so; none is a measurement of a real system.

## 01 Baseline Assumptions

- Module 00, Lesson 0.3: experimental unit, blocking, and paired block differences $d_i=Y_{B,i}-Y_{A,i}$. Lesson 0.5: the t interval for a mean, the half-width planning formula, and practical significance.
- Module 00 does **not** teach resampling intervals, noninferiority margins, power for a minimum effect, or multiplicity. Lesson 15.4 teaches each at first use.
- Modules 04 and 07: latency/goodput and behavioral uncertainty/calibration.
- Modules 08–11: lineage, retrieval evidence, context, and memory state.
- Modules 12–14: trajectories, attempt identity, external effects, and durable execution.

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
  economics: REQUIRED
  architecture_tradeoff: REQUIRED
  research_connection: REQUIRED

estimated_effort:
  instruction: 4h       # lesson instruction: 35+45+35+60+35+30 min
  guided_practice: 3h   # lesson practice: 25+30+25+40+30+30 min
  labs: 12h             # LAB A 3h + LAB B 3h + LAB C 3h + LAB D 3h
  assessment: 3h        # Mastery transfer problem 2.5h + Incident 15.1 0.5h
  source_trace: 2h      # Section 09 Production Source Trace artifact, counted once
  total: 24h
```

Each category is counted once. The 2h source trace is the Section 09 artifact; it is not also counted inside Lesson 15.6 practice or LAB D.

The learner must be able to define an evaluation estimand and population; build versioned datasets and slices; select and validate graders; run paired, dependence-aware comparisons; define release gates before seeing results; include failed and unfinished outcomes; connect offline evidence to online outcomes; trace a pinned harness; and diagnose a disagreement among benchmark, judge, system, and user metrics.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Keep four things separate:

1. **Construct:** the property the decision cares about—task success, safety, user utility, cost, or latency.
2. **Observation:** raw output, tool effect, human label, executable test, or production event.
3. **Estimator:** aggregation, weighting, uncertainty, and slice procedure.
4. **Decision rule:** improvement/noninferiority thresholds, hard constraints, escalation, and rollout policy.

A grader observes a proxy. A benchmark samples a population. A confidence interval describes a sampling procedure. None is the product decision by itself.

## 04 Lessons

### Lesson 15.1 — Evaluation Contract and Experimental Unit

**Engineering Question:**
What population quantity is being estimated, at which independent unit, and for which decision?

**Concepts & Definitions:**
- **Construct**: property the decision cares about.
- **Estimand**: precisely defined population quantity.
- **Independent unit (block)**: the unit at which sampling variability is modeled; rows inside one block may be correlated.
- **Design effect**: the factor by which correlated rows inflate the variance of a mean relative to the same number of independent rows.

**Mechanism Explanation:**
Start with the decision: what change could the result authorize, block, or investigate? Then bind the target population and sampling frame, independent unit, system boundary, policy and model versions, outcomes, thresholds, uncertainty procedure, missingness, and exclusions **before** observing results (**D**, CLM-001). Request, turn, conversation, trajectory, user, and session are not interchangeable units (**D**, CLM-003).

HELM is a reference for scenario and metric coverage, not a universal production suite. Its durable lesson is to expose which scenarios and desiderata are measured and which remain absent (**O**, CLM-002).

**Quantitative Model / Derivation:**
When A and B run on the same independent blocks, compare paired differences:

$$d_i=m_B(i)-m_A(i),\qquad \bar d=\frac{1}{n}\sum_i d_i.$$

Suppose $G$ independent blocks each hold $m$ rows with common variance $\sigma^2$ and correlation $\rho$ between rows of the same block. Then

$$\mathrm{Var}(\bar d)=\frac{\sigma^2}{Gm}\bigl[1+(m-1)\rho\bigr],\qquad n_{\text{eff}}=\frac{Gm}{1+(m-1)\rho}.$$

where:
- $i$ indexes paired rows; $n$ is the number of paired rows ($n=Gm$ in the block model);
- $m_A(i), m_B(i)$ are the metric values of systems A and B on row $i$ (here $m$ with a subscript is the metric; plain $m$ below is the number of rows per block);
- $d_i$ is the paired difference, positive when B is better; $\bar d$ is its mean;
- $G$ is the number of independent blocks; $\sigma^2$ is the variance of one $d_i$; $\rho$ is the correlation between two rows of the same block (rows in different blocks are independent);
- $n_{\text{eff}}$ is the number of independent rows that would give the same $\mathrm{Var}(\bar d)$.

The bracket is the design effect. It follows from summing $m$ variances and $m(m-1)$ covariances inside each block (**D**, CLM-003). It assumes equal block sizes and one shared $\rho$; real data need the block-level analysis of Lesson 15.4.

*Step-by-step.*
1. One block's row sum has variance $m\sigma^2$ (the $m$ variances) plus $m(m-1)\rho\sigma^2$ (each ordered pair of distinct rows contributes covariance $\rho\sigma^2$), i.e. $m\sigma^2[1+(m-1)\rho]$.
2. Blocks are independent, so the total over all $G$ blocks has variance $Gm\sigma^2[1+(m-1)\rho]$.
3. $\bar d$ divides that total by $Gm$, which divides the variance by $(Gm)^2$: $\mathrm{Var}(\bar d)=\frac{\sigma^2}{Gm}[1+(m-1)\rho]$.
4. Independent rows would give $\sigma^2/n_{\text{eff}}$. Setting the two equal gives $n_{\text{eff}}=Gm/[1+(m-1)\rho]$.

**Worked Example (synthetic):**
- *Input*: two systems run on 100 shared prompts with five generations each, 500 paired rows. Assume $\rho=0.6$ between generations of one prompt.
- *Steps*: design effect $=1+(5-1)\times0.6=3.4$; $n_{\text{eff}}=500/3.4=147.1$.
- *Result*: the 500 rows carry about as much information about the prompt population as 147 independent rows. At $\rho=1$ it is 100; at $\rho=0$ it is 500.
- *Interpretation / limits*: pair by prompt and resample prompts, not rows. Repeats estimate conditional generation variability; they do not add prompts. $\rho=0.6$ is an exercise assumption, not a measured value.

**Knowledge Check:**
1. Why are turns from one conversation not automatically independent?
2. Which decision changes if the system boundary includes retries and retrieval?

**Guided Practice:**
(a) A sample has 40 conversations with 6 turns each and assumed $\rho=0.3$. Compute the design effect and $n_{\text{eff}}$. (b) Write an evaluation contract for one release decision and identify every unit that could be confused with the independent block.

**Feedback Contract:**
- *Expected Output*: (a) design effect $=1+5\times0.3=2.5$; $n_{\text{eff}}=240/2.5=96$. (b) A contract naming population, frame, independent unit, boundary, estimand, versions, thresholds, and missingness. Knowledge check 1: turns share a user, topic, and earlier context, so their errors move together.
- *Typical Error*: answering $n_{\text{eff}}=240$, or treating repeated samples as new independent prompts.
- *Diagnostic Hint*: What process generated the independent blocks? If you doubled rows per block, which new blocks did you sample?
- *Concept to Revisit*: evaluation estimand and experimental unit (Module 00, Lesson 0.3).

**Learning Outcome:**
State exactly what quantity a result estimates and which decision it can support.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 15.2 — Dataset Lineage, Slices, Freshness, and Contamination

**Engineering Question:**
How can a dataset support reproducible estimates without becoming an invisible development target or hiding critical slices?

**Concepts & Definitions:**
- **Lineage**: identity, origin, transformations, labels/rubrics, and exclusions.
- **Tag**: an attribute of an item, such as language or prompt length. One item can carry several tags.
- **Partition**: a set of slices in which every item belongs to exactly one slice (disjoint and exhaustive).
- **Cell**: one unique combination of tags. Cells always form a partition.
- **Critical slice**: a predeclared subgroup with its own pass/fail gate. Critical slices may overlap each other.
- **Protected holdout**: data withheld from routine inspection and optimization.

**Mechanism Explanation:**
Every item needs stable identity, origin/license, capture time, sampling probability or intended weight, split, version, deduplication lineage, tags, expected answer or rubric provenance, and exclusion history (**D**, CLM-004). Preserve a protected final holdout; routine regression sets become development data once teams inspect and optimize against them.

Choose tags and slices from risks and workload structure before results: task, language, locale, prompt/output length, user cohort, retrieval/tool path, policy class, traffic source, difficulty, and failure mode. Tiny slices need uncertainty, not confident rankings.

LiveBench is a design example: recent sources, frequent refresh, objective grading, and diverse tasks limit contamination (**O**, CLM-012). “Contamination-limited” is the defensible claim. Fresh data does not prove absence of private leakage, tuning feedback, benchmark-specific adaptation, or production relevance.

Two different questions use slices, and they need different machinery:

1. *What is the expected score on the target population?* This is a prevalence-weighted estimate. It needs a partition.
2. *Is any protected subgroup unacceptably bad?* This is a set of critical-slice gates. It needs no partition and no prevalence weight.

**Quantitative Model / Derivation:**

*Partition rule.* Let slices $S_1,\dots,S_K$ be disjoint and exhaustive, with target prevalences $w_s=P(S_s)$ and $\sum_s w_s=1$. By the law of total probability the population mean is $\sum_s P(S_s)\,E[m\mid S_s]$, so

$$\hat m=\sum_{s=1}^{K} w_s\,\hat m_s$$

estimates it (**D**, CLM-019). Here $K$ is the number of slices, $m$ is the per-item metric, $\hat m_s$ is the sample mean of $m$ over the items in $S_s$, and $w_s$ is the target share of slice $s$ (not its sample share). The identity fails if an item belongs to two slices, because that item's score enters two terms, or to none, because it enters no term.

*Item-level weighting.* When tags overlap, weight items instead of slices. Give item $i$ the weight $\omega_i=(\text{target share of its cell})/(\text{sample share of its cell})$ and compute

$$\hat m=\frac{\sum_i \omega_i m_i}{\sum_i \omega_i}.$$

where $m_i$ is item $i$'s score and $\omega_i$ its weight. Each item is counted once, however many tags it carries. When every item of a cell shares one weight, this equals the partition formula applied to cells (**D**, CLM-019).

*Step-by-step (why the two forms agree).* Let cell $c$ have $n_c$ of the $N$ sampled items, sample share $n_c/N$, target share $w_c$, and cell mean $\hat m_c$. Every item in $c$ gets $\omega_i=w_c/(n_c/N)=Nw_c/n_c$.
1. Numerator: $\sum_i\omega_i m_i=\sum_c \frac{Nw_c}{n_c}\sum_{i\in c}m_i=\sum_c Nw_c\,\hat m_c$.
2. Denominator: $\sum_i\omega_i=\sum_c n_c\cdot\frac{Nw_c}{n_c}=N\sum_c w_c=N$.
3. Ratio: $\sum_c w_c\,\hat m_c$, the partition formula over cells. Both forms assume the target shares are known and every target cell has sampled items; an empty cell cannot be reweighted.

*Critical-slice gates.* A gate has the form $\hat m_s\ge f_s$ (or the interval form of Lesson 15.4) and is evaluated on the members of $S_s$ only. A weight $w_s$ never enters a gate, so a small prevalence cannot buy back a failed gate (**D**, CLM-011).

**Worked Example A — overlap double-counts (synthetic):**
- *Input*: 20 graded items with two tags, L (long context) and N (non-English). Pass counts by cell:

| Cell | Items | Passes | Cell score |
|---|---:|---:|---:|
| neither | 10 | 9 | 0.90 |
| L only | 4 | 3 | 0.75 |
| N only | 2 | 1 | 0.50 |
| L and N | 4 | 1 | 0.25 |

- *Steps (wrong)*: treat "L", "N", and "neither" as slices. $\hat m_L=(3+1)/8=0.50$, $\hat m_N=(1+1)/6=0.333$, $\hat m_{\text{neither}}=0.90$. Sample shares are $8/20=0.4$, $6/20=0.3$, and $0.5$, which sum to 1.2. The sum $\sum w_s\hat m_s=0.20+0.10+0.45=0.75$. Dividing by 1.2 gives 0.625.
- *Steps (right)*: count each item once. $(9+3+1+1)/20=0.70$.
- *Result*: the true sample mean is 0.70. The overlapping-slice sum gives 0.75 and its normalized form gives 0.625. The four "L and N" items were counted twice: 24 item-counts for 20 items.
- *Reweighting to a target*: suppose the target cell shares are 0.6, 0.2, 0.1, 0.1. The partition formula over cells gives $0.6\times0.90+0.2\times0.75+0.1\times0.50+0.1\times0.25=0.765$. The item weights are 1.2, 1.0, 1.0, 0.5 (target share / sample share: $0.6/0.5$, $0.2/0.2$, $0.1/0.1$, $0.1/0.2$), and $\sum\omega_i m_i/\sum\omega_i=(10.8+3+1+0.5)/(12+4+2+2)=15.3/20=0.765$. The two forms agree.
- *Interpretation / limits*: neither wrong number is biased in a predictable direction; the error depends on how the overlap cell scores. The target shares here are exercise assumptions.

**Worked Example B — a gate is not a weight (synthetic):**
- *Input*: a partition with weights 0.9 and 0.1 and slice scores 0.90 and 0.40. The second slice is critical, with a predeclared floor of 0.70.
- *Steps*: weighted score $=0.9\times0.90+0.1\times0.40=0.85$. Gate: $0.40\ge0.70$ is false.
- *Result*: the aggregate is 0.85 and the release still fails.
- *Interpretation / limits*: report both numbers. With a small slice the gate must use an interval (Lesson 15.4), not the point estimate.

**Knowledge Check:**
1. Why does frequent refresh limit rather than eliminate contamination?
2. When does a regression set become development data?
3. Two critical slices overlap. Does that break the gates? Does it break the weighted estimate?

**Guided Practice:**
(a) 25 graded items carry tags V (voice) and T (tool path). Cells: neither 15 items with 12 passes; V only 5 with 4; T only 3 with 1; V and T 2 with 1. Compute the correct sample mean, the overlapping-slice sum, and its normalized form. Then reweight to target cell shares 0.5, 0.2, 0.2, 0.1. A floor of 0.60 applies to all T-tagged items; evaluate it. (b) Build a manifest, deduplication lineage, and freshness policy for a stratified sample.

**Feedback Contract:**
- *Expected Output*: (a) correct mean $18/25=0.72$; overlapping-slice sum $0.76$ with weights summing to 1.08; normalized $0.704$; reweighted $0.677$ by both forms; the T gate uses $(1+1)/5=0.40<0.60$ and fails. Knowledge check 3: overlap does not affect gates, because each gate reads only its own members; it does break $\sum w_s\hat m_s$. (b) Reproducible membership, weights, leakage audit, uncertainty by slice, and protected split policy.
- *Typical Error*: using tag frequencies as weights when tags overlap, post-result exclusions, or reporting an aggregate that hides a failed gate.
- *Diagnostic Hint*: Do your weights sum to one? Sum the per-slice item counts: do they equal the number of items?
- *Concept to Revisit*: partition versus tag; critical-slice gate versus prevalence weight.

**Learning Outcome:**
Reproduce dataset membership, compute a weighted estimate that counts each item once, and keep critical-slice gates separate from that estimate.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 15.3 — Graders Are Measurement Instruments

**Engineering Question:**
How should deterministic, model-based, and human graders be calibrated before they influence a release gate?

**Concepts & Definitions:**
- **Oracle**: observation procedure valid for a declared construct.
- **Meta-evaluation**: measuring grader error against independent adjudicated evidence.
- **Order sensitivity**: judgment changes caused by presentation order rather than answer quality.
- **Swap consistency** (this module's definition): the fraction of pairs whose verdict is unchanged when the two answers are shown in the opposite order.

**Mechanism Explanation:**

Use the strongest valid oracle for each construct (**D**, CLM-005):

| Grader | Strong use | Failure to test |
|---|---|---|
| Exact/normalized match | Canonical closed answers | Valid equivalents; normalization bugs |
| Executable test | Observable program/tool invariant | Incomplete tests; sandbox nondeterminism |
| Semantic/rule checker | Explicit domain invariants | Coverage and implementation errors |
| LLM judge | Open-ended rubric at scale | Position, verbosity, self/preference, prompt and model drift |
| Human/domain expert | Normative, ambiguous, high-risk judgment | Rater interpretation, fatigue, identity cues, disagreement |

The MT-Bench/Chatbot Arena paper reports useful judge–human agreement in its settings and documents position, verbosity, self-enhancement, and reasoning limitations (**O**, CLM-006). A later study reports that position bias varies by judge and task (**O**, CLM-007). Therefore validate a judge against blinded, independently adjudicated labels; fix judge model, prompt, decoding, parser, and rubric; randomize and swap answer order; allow ties/abstention; and report confusion, consistency, disagreement, and slice behavior (**D**, CLM-008).

Human protocols require rubric examples and counterexamples, qualification, randomization/blinding where feasible, repeated labels, tie/abstain, disagreement analysis, adjudication, privacy, and worker well-being (**D**, CLM-009). Do not erase meaningful plural judgments by forcing consensus.

**Quantitative Model / Derivation:**
For $n$ pairwise judgments, the descriptive tie-adjusted preference rate is

$$\hat p=\frac{W_B+0.5T}{n}.$$

where $W_B$ is the number of pairs B wins, $T$ the number of ties, and $W_A$ the number A wins, with $n=W_A+W_B+T$. Each tie counts as half a win for each side, so $\hat p$ for B and the matching rate $(W_A+0.5T)/n$ for A sum to 1, and $\hat p=0.5$ means no net preference. Swap consistency $=(\text{verdicts unchanged after the swap})/n$.

It is conditional on sampled items, presentation, judge/rater population, and protocol—not context-free model quality. Swapping order detects sensitivity but does not manufacture ground truth.

**Worked Example (synthetic):**
- *Input*: 20 order-balanced pairs: B wins 9, A wins 7, 4 tie. The same 20 pairs are judged again with the order swapped; 14 verdicts are unchanged.
- *Steps*: $\hat p=(9+0.5\times4)/20=0.55$. Swap consistency $=14/20=0.70$.
- *Result*: tie-adjusted preference for B is 0.55; 30% of verdicts depend on presentation order.
- *Interpretation / limits*: with six order-dependent verdicts, a 0.55 preference is not evidence that B is better. Neither number says which answers are correct; that needs independent labels.

**Knowledge Check:**
1. Why does swapping order detect sensitivity but not create ground truth?
2. What independent evidence is needed before using a judge in a gate?

**Guided Practice:**
(a) 30 pairs: B wins 14, A wins 10, 6 tie; 24 verdicts survive an order swap. Compute both statistics. (b) Calibrate exact/executable, semantic, two judge, and blinded human paths on one stratified sample; report disagreement rather than forcing consensus.

**Feedback Contract:**
- *Expected Output*: (a) $\hat p=(14+3)/30=0.567$; swap consistency $=24/30=0.80$. (b) Fixed grader versions/protocol, order assignment, ties, reference labels, slice error, uncertainty, latency, and cost. Knowledge check 2: blinded labels adjudicated without seeing the judge's output.
- *Typical Error*: dropping ties from the denominator ($14/24=0.583$), self-judging, or forced choice without calibration.
- *Diagnostic Hint*: Does disagreement track answer position, length, or slice?
- *Concept to Revisit*: grader validity and order sensitivity.

**Learning Outcome:**
Quantify what every grader gets wrong before trusting it in a gate.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 15.4 — Uncertainty, Multiple Comparisons, and Release Gates

**Engineering Question:**
How can paired evidence, dependence, practical thresholds, and repeated looks be combined into an auditable release decision?

**Concepts & Definitions:**
- **Paired effect**: within-row difference between systems on the same row, signed so that positive means B is better.
- **Block effect**: the mean paired effect inside one independent block.
- **Practical threshold** $\epsilon$: the smallest improvement worth releasing for.
- **Noninferiority margin** $\tau$: the largest loss that is still acceptable.
- **Gate**: one predeclared comparison of an interval bound with a threshold.
- **Multiplicity policy**: how error rates are controlled across several gates, slices, metrics, and looks.
- **Look**: one analysis of accumulating data that can trigger a decision.
- **Power**: the probability that a gate passes when the true effect equals a stated planning value.

**Mechanism Explanation:**
Predeclare, before any result is seen: metric and direction, the independent block, the interval procedure and level, $\epsilon$ and $\tau$, the critical slices and their margins, the multiplicity and look policy, and the three-state decision rule (**D**, CLM-011). Then compute paired effects, reduce them to block effects, build the interval at block level, evaluate every gate, and record the decision with its inputs. Thresholds are product and risk judgments; statistics cannot choose them. Choices made after seeing results belong in a new, separately sampled confirmation.

A scalar $S=\sum_k w_k z_k$ is meaningful only with fixed directions, scales, transforms, and weights. Preserve the metric vector, Pareto frontier, and hard safety/reliability constraints so a gain cannot compensate for an unacceptable harm.

**Quantitative Model / Derivation:**

The seven items below are this module's **release statistics contract**. Module 17 reuses it.

*1. Paired effect.* For row $j$ in block $g$, with both systems run on that same row, $d_{g,j}=m_B(g,j)-m_A(g,j)$. The block effect is $\bar d_g=\frac{1}{n_g}\sum_j d_{g,j}$. The estimand is the mean block effect $\Delta=E[\bar d_g]$, estimated by $\hat\Delta=\frac1G\sum_g\bar d_g$. If block sizes differ, the row-weighted mean $\sum_{g,j}d_{g,j}/\sum_g n_g$ is a different estimand; declare which one the decision uses (**D**, CLM-010).

where $g=1,\dots,G$ indexes blocks, $j=1,\dots,n_g$ indexes rows inside block $g$, $m_A(g,j)$ and $m_B(g,j)$ are the metric values, $\Delta$ is the population mean block effect, and $\hat\Delta$ its estimate.

*2. Interval.* Let $s_b$ be the sample standard deviation of the $G$ block effects, $s_b=\sqrt{\frac{1}{G-1}\sum_g(\bar d_g-\hat\Delta)^2}$. Let $\alpha$ be the two-sided error level (0.05 for a 95% interval) and $t_{1-\alpha/2,\,G-1}$ the $1-\alpha/2$ quantile of the t distribution with $G-1$ degrees of freedom.

- Block t interval: $\hat\Delta\pm t_{1-\alpha/2,\,G-1}\,s_b/\sqrt G$. This is Module 00's paired interval applied to block effects.
- Block bootstrap: draw $G$ blocks with replacement, keep every row of each drawn block, recompute $\hat\Delta$, repeat, and take percentiles. The standard deviation of that bootstrap distribution is exactly $\sqrt{(G-1)/G}\;s_b/\sqrt G$. Step-by-step: each resample draws $G$ block effects independently from the $G$ observed values, whose variance with divisor $G$ is $\frac{G-1}{G}s_b^2$; the mean of $G$ such draws has variance $\frac{G-1}{G}s_b^2/G$; take the square root. Example (R-15 below): $\sqrt{9/10}\times0.1491=0.1414$.
- **Block resampling is required whenever an independent unit contributes more than one row**: repeated generations of one prompt, turns of one conversation, tasks of one user or session, questions about one document, or translations of one question. Row resampling is valid only when each block has one row or rows are uncorrelated. Under positive within-block correlation it understates the standard error by about the square root of the design effect of Lesson 15.1 (**D**, CLM-020).
- No interval here is a guarantee. With few blocks, the percentile bootstrap undercovers; the simulation below shows it. The t interval assumes roughly normal block effects.

The same adjustment appears in the evaluation literature as clustered standard errors, together with the recommendation to analyze question-level paired differences (**O**, CLM-023).

*3. Gates.* Let $[L,U]$ be the two-sided $1-\alpha$ interval for a quantity signed so that larger is better.

- Improvement gate: **passes** if $L\ge\epsilon$; **refuted** if $U<\epsilon$.
- Noninferiority gate: **passes** if $L\ge-\tau$; **refuted** if $U<-\tau$.
- Otherwise the gate is **undetermined**.

Here $L$ and $U$ are the lower and upper interval bounds (not the look count $L$ of item 5). Because $L<U$, a gate cannot both pass and be refuted.

For a harm metric where lower is better, with $\Delta_h=h_B-h_A$, the noninferiority gate is $UCB(\Delta_h)\le\tau$. Here $h_A,h_B$ are the harm rates of A and B and $UCB$ is the upper confidence bound (the $U$ of the harm interval). Flipping the sign, $-\Delta_h$ is larger-is-better, and $L(-\Delta_h)\ge-\tau$ is the same condition as $UCB(\Delta_h)\le\tau$. A pass at a two-sided 95% interval is a one-sided 97.5% statement, so a gate whose true value sits exactly at its threshold passes at most about 2.5% of the time (**D**, CLM-021). Reason: a two-sided $1-\alpha$ interval leaves $\alpha/2$ in each tail, so when the true value equals the threshold, $L\ge$ threshold happens with probability about $\alpha/2=0.025$.

*4. Critical-slice gates.* Each critical slice gets its own gate, computed from the rows in that slice with the same block structure and its own margin $\tau_s$. Prevalence weights do not enter. Hard constraints, such as zero unauthorized effects, are deterministic checks and need no interval.

*5. Multiplicity and looks.*

- *All gates must pass.* If release requires every gate to pass, no adjustment is needed to control false release. Reason: if any one requirement is truly violated, release needs that gate to pass, which happens with probability at most its own level (**D**, CLM-021). The price is lower power.
- *Any of $K$ can win.* If any one of $K$ metrics or slices could justify a claim, test each at $\alpha/K$ (Bonferroni). This also applies to exploratory slices scanned for regressions.
- *Looks.* The default is one planned look. With $L$ planned looks, use $\alpha/L$ at each. This is conservative and valid. An unplanned look or rerun is treated as a new experiment on a fresh protected sample; its data are never silently pooled.
- *Rerun-until-pass.* With $k$ independent reruns of a gate whose true value sits at its threshold, the chance that at least one passes is $1-(1-\alpha/2)^k$. Step-by-step: each rerun falsely passes with probability $\alpha/2$ (item 3), so it fails with $1-\alpha/2$; all $k$ independent reruns fail with $(1-\alpha/2)^k$; at least one passes with the complement. With $\alpha=0.05$, $k=5$: $1-0.975^5=0.119$.

*6. Decision.*

- **RELEASE**: every gate passes and no hard constraint is violated.
- **HOLD**: at least one gate is refuted or a hard constraint is violated.
- **INCONCLUSIVE**: anything else. Do not release. Either extend the sample at a planned look or escalate for an explicit, recorded risk acceptance by a named owner. Risk acceptance is not a statistical pass.

*7. Planning.* For an improvement gate with planning effect $\delta>\epsilon$, pilot block standard deviation $s_b$, two-sided level $\alpha$, and power $1-\beta$, the normal approximation gives

$$G\ \ge\ \left(\frac{(z_{1-\alpha/2}+z_{1-\beta})\,s_b}{\delta-\epsilon}\right)^2 .$$

where $z_q$ is the $q$ quantile of the standard normal ($z_{0.975}=1.960$, $z_{0.80}=0.842$), $\beta$ is the planned miss probability, and $\delta$ is the true effect assumed for planning.

*Step-by-step (normal approximation, $SE=s_b/\sqrt G$).*
1. The gate passes when $L=\hat\Delta-z_{1-\alpha/2}SE\ge\epsilon$, i.e. $\hat\Delta\ge\epsilon+z_{1-\alpha/2}SE$.
2. If the true effect is $\delta$, then $\hat\Delta\approx\delta+SE\cdot Z$ with $Z$ standard normal. Requiring a pass with probability $1-\beta$ means $\delta-z_{1-\beta}SE\ge\epsilon+z_{1-\alpha/2}SE$.
3. Rearrange: $(z_{1-\alpha/2}+z_{1-\beta})\,s_b/\sqrt G\le\delta-\epsilon$. Solve for $G$ and square to get the formula above; round up.
4. The noninferiority gate passes when $L\ge-\tau$, so the distance to the threshold is $\delta-(-\tau)=\delta+\tau$.

For a noninferiority gate replace $\delta-\epsilon$ by $\delta+\tau$. Power here is the probability that the gate passes when the true effect is $\delta$. It is not the probability that B is better, and it is not defined at a true effect equal to the threshold, where the pass rate is the false-pass rate (**D**, CLM-022). $G$ counts blocks, not rows.

**Worked Example — fixture R-15 (synthetic):**

*Predeclared contract.* Metric: task success (1 or 0), higher is better. Block: user. Ten users were sampled independently; each ran the same four tasks on A and on B. Primary gate: improvement with $\epsilon=0.05$. Critical slice: the effectful refund task T4, noninferiority with $\tau=0.05$. Interval: two-sided 95% block t. One look. All gates must pass.

*Input.* Paired effects $d=\text{success}_B-\text{success}_A$:

| User | T1 | T2 | T3 | T4 (critical) | Block effect $\bar d_g$ |
|---|---:|---:|---:|---:|---:|
| U1 | +1 | +1 | +1 | +1 | 1.00 |
| U2 | +1 | +1 | +1 | 0 | 0.75 |
| U3 | +1 | +1 | +1 | 0 | 0.75 |
| U4 | +1 | +1 | 0 | 0 | 0.50 |
| U5 | 0 | 0 | 0 | 0 | 0.00 |
| U6 | 0 | 0 | 0 | 0 | 0.00 |
| U7 | 0 | 0 | 0 | 0 | 0.00 |
| U8 | 0 | 0 | 0 | 0 | 0.00 |
| U9 | +1 | 0 | 0 | −1 | 0.00 |
| U10 | 0 | 0 | −1 | −1 | −0.50 |

There are 13 rows at +1, 24 at 0, and 3 at −1.

*Step 1 — estimate.* $\hat\Delta=(1+0.75+0.75+0.5+0+0+0+0+0-0.5)/10=0.25$. Block sizes are equal, so the row mean is also $10/40=0.25$.

*Step 2 — the wrong interval (rows as independent).* $n=40$, $s=0.5883$, $SE=0.5883/\sqrt{40}=0.0930$, $t_{0.975,39}=2.023$. Interval $[0.062,\,0.438]$. A row bootstrap (10,000 resamples, seed 15) gives the percentile interval $[0.075,\,0.425]$.

*Step 3 — the block interval.* Squared deviations of the block effects from 0.25 sum to 2.0, so $s_b=\sqrt{2.0/9}=0.4714$ and $SE=0.4714/\sqrt{10}=0.1491$. With $t_{0.975,9}=2.262$ the half-width is $2.262\times0.1491=0.337$ and the interval is $0.25\pm0.337=[-0.087,\,0.587]$. A block bootstrap (10,000 resamples, seed 15) gives $[-0.025,\,0.525]$. The exact bootstrap standard deviations are $0.1414$ for blocks and $0.0919$ for rows.

*Step 4 — why they differ.* The block standard error is $0.1491/0.0930=1.60$ times the row standard error, a variance ratio of 2.57 ($1.60^2$). Read through Lesson 15.1, this ratio plays the role of an observed design effect: $1+(4-1)\rho=2.57$ would mean $\rho\approx0.52$. That is only an analogy, since the 15.1 model assumes one shared $\rho$. Users are either helped on most tasks or not at all, so four rows from one user are far from four independent rows.

*Step 5 — primary gate.* Block interval: $L=-0.087<0.05$, so the gate does not pass; $U=0.587\ge0.05$, so it is not refuted. **Undetermined.** The row interval would have passed it ($0.062\ge0.05$).

*Step 6 — critical-slice gate.* T4 has one row per user: $(+1,0,0,0,0,0,0,0,-1,-1)$. Mean $-0.10$, $s=\sqrt{2.9/9}=0.5676$, $SE=0.1795$, interval $[-0.506,\,0.306]$. $L<-0.05$ and $U\ge-0.05$. **Undetermined.** Cross-check: the discordant pairs are 1 positive and 2 negative, and an exact two-sided sign test gives $p=1.0$. (Ties, the seven zeros, are dropped; with 3 discordant pairs under a fair coin, the outcome 1-vs-2 is the most likely split, so every possible outcome is at least as extreme and $p=1$.)

*Step 7 — decision.* No gate is refuted and not every gate passes. **INCONCLUSIVE**, with a mean of +0.25.

*Look and multiplicity arithmetic.* If two looks had been planned, each would use $\alpha/2$: $t_{0.9875,9}=2.685$ and the interval widens to $[-0.150,\,0.650]$. If five exploratory slices were scanned, each would use $t_{0.995,9}=3.250$, a half-width 1.44 times larger ($3.250/2.262$). Five independent reruns of a gate sitting at its threshold pass at least once with probability $1-0.975^5=0.119$; ten reruns, 0.224.

*Planning calculation.* Treat $s_b=0.4714$ as a pilot value and plan for $\delta=0.25$, $\epsilon=0.05$, $\alpha=0.05$, power 0.80:

$$G\ge\left(\frac{(1.960+0.842)\times0.4714}{0.25-0.05}\right)^2=43.6\ \Rightarrow\ 44\text{ users}.$$

Step-by-step: $1.960+0.842=2.802$; $2.802\times0.4714=1.321$; $1.321/0.20=6.604$; $6.604^2=43.6$; round up to 44.

The answer is sensitive to the pilot: $s_b=0.35$ gives 25 users and $s_b=0.60$ gives 71. A planning effect of 0.15 gives 175.

*Planning simulation (synthetic; seed 154; 20,000 replications; block effects drawn from a normal distribution with mean $\delta$ and standard deviation 0.4714; block t gate).* The primary gate passes with probability 0.223 at $G=10$, 0.788 at $G=44$, and 0.801 at $G=46$. With the true effect set to $\epsilon$, it passes with probability 0.024 to 0.025 at every $G$ tried. So the ten-user sample would have been inconclusive about three times in four even if the true effect were 0.25.

*Coverage simulation (synthetic; seed 151; 5,000 replications; 4 rows per block; block effects normal with mean 0.25 and standard deviation 0.4; row noise normal with standard deviation 0.4, so $\rho=0.5$; 1,000 resamples per bootstrap).* Fraction of nominal 95% intervals that contain the true mean:

| Blocks | Row bootstrap | Block bootstrap (percentile) | Block t |
|---:|---:|---:|---:|
| 10 | 0.756 | 0.900 | 0.953 |
| 40 | 0.774 | 0.931 | 0.946 |

Monte Carlo error is about 0.003 near 0.95 and 0.006 near 0.76 (**D**, CLM-024).

*Interpretation / limits.* A positive mean is not a release. Row resampling misses the truth about one time in four in this model, and more blocks do not repair it. The block bootstrap is the right resampling unit but still undercovers at ten blocks. The block t interval is near 95% here because the simulation draws normal block effects; R-15's block effects are bounded and discrete, so its interval is approximate. All numbers describe synthetic data and one simulation model.

**Knowledge Check:**
1. Why can row bootstrap understate uncertainty for user-clustered data?
2. What happens to nominal error rates under rerun-until-pass behavior?
3. Release requires three gates to all pass. Why is no Bonferroni correction needed for false release, and what does the requirement cost?

**Guided Practice:**
Fixture P-15 (synthetic) uses the same contract as R-15, with twelve users. Paired effects for T1–T4:

`U1 (+1,+1,+1,−1)`, `U2 (+1,+1,+1,−1)`, `U3 (+1,+1,+1,0)`, `U4 (+1,+1,0,−1)`, `U5 (+1,+1,0,0)`, `U6 (+1,0,+1,−1)`, `U7 (+1,+1,0,−1)`, `U8 (0,+1,+1,0)`, `U9 (+1,0,0,−1)`, `U10 (+1,+1,0,0)`, `U11 (0,+1,+1,−1)`, `U12 (+1,0,+1,0)`.

Compute the block effects, the block t interval, both gates, and the decision. Then compute the row-level interval and explain how it compares.

**Feedback Contract:**
- *Expected Output*: block effects $(0.5,0.5,0.75,0.25,0.5,0.25,0.25,0.5,0,0.5,0.25,0.5)$; $\hat\Delta=4.75/12=0.396$; $s_b=0.198$; $SE=0.0572$; $t_{0.975,11}=2.201$; interval $[0.270,\,0.522]$, so the primary gate **passes**. T4 has seven −1 and five 0: mean $-0.583$, $s=0.515$, $SE=0.149$, interval $[-0.911,\,-0.256]$; $U<-0.05$, so the critical-slice gate is **refuted**. Decision: **HOLD**, with a positive mean and a passing primary gate. The row interval is $[0.182,\,0.610]$, wider than the block interval, because here rows inside a user are negatively related (gains on T1–T3, losses on T4). Block analysis is the correct unit, not the wider one. Knowledge check 3: release needs the violated gate to pass, which has probability at most its own level; the cost is lower power.
- *Typical Error*: reporting RELEASE because the mean is positive or because the primary gate passes; using $n=48$ rows in the t formula; averaging the T4 loss into the aggregate.
- *Diagnostic Hint*: How many independent users are there? Which gate reads only T4? Which choices were made after seeing results?
- *Concept to Revisit*: block effect and block interval; critical-slice gate; the three decision states.

**Learning Outcome:**
Make release decisions auditable under practical and statistical uncertainty.

*(Effort: 60m instruction, 40m practice)*

---

### Lesson 15.5 — End-to-End and Offline-to-Online Validity

**Engineering Question:**
When does an offline quality gain fail to improve offered-user utility under real latency, failures, effects, and traffic?

**Concepts & Definitions:**
- **Offered denominator**: every eligible request/session, including failure and unfinished states.
- **Completion-conditioned quality**: quality computed only over requests that completed.
- **Goodput**: useful accepted outcomes meeting declared constraints per time.
- **Online validity**: causal evidence that proxy changes transfer to user/business outcomes.

**Mechanism Explanation:**
Score offered work, not only completed successes. Keep timeouts, refusals, invalid output, retrieval/tool failure, retries, abandonment, and unfinished trajectories in explicit terminal classes (**D**, CLM-014). Conditioning quality on completion permits a weak system to improve by dropping hard cases.

For agents, preserve turn and trajectory success, attempts, tool effects, recovery, intervention, irreversible harm, and final state. A correct final answer reached after unauthorized or duplicate effects is not a successful trajectory.

Offline scores are proxies. Shadowing, canaries, randomized A/B tests, or other causally credible designs must test whether they predict user and business outcomes under real traffic, latency, interaction, and feedback (**D**, CLM-013). State the assignment and exposure units, and check sample-ratio mismatch, interference, novelty, selection, and delayed outcomes.

**Quantitative Model / Derivation:**
Join task quality with safety, latency, cost, and completion. A thresholded offered-request goodput is

$$G=\frac{1}{T}\sum_i \mathbf{1}[q_i\ge q^*,\ l_i\le l^*,\ c_i\le c^*,\ terminal_i=accepted],$$

with units of useful accepted requests per time.

where $T$ is the observation window length, $i$ ranges over all offered requests in that window, $q_i,l_i,c_i$ are request $i$'s quality, latency, and cost, $q^*,l^*,c^*$ are the declared thresholds, $terminal_i$ is its terminal class, and $\mathbf{1}[\cdot]$ is 1 if every condition holds and 0 otherwise. Here $G$ means goodput, not the block count $G$ of Lesson 15.4, and $T$ means time, not the tie count of Lesson 15.3. Dividing the same count by the number of offered requests instead of $T$ gives the offered success rate. Thresholds are workload decisions; show sensitivity and the quality–latency–cost frontier.

**Worked Example (synthetic):**
- *Input*: in 100 s, 50 requests are offered. 30 accepted outcomes satisfy the quality, latency, and cost thresholds; 8 time out; 4 refuse; 3 fail tools; 5 remain unfinished.
- *Steps*: goodput $=30/100=0.30$ useful requests/s. Offered success rate $=30/50=0.60$.
- *Second input*: system A completes 40 of 50 offered requests and 30 of those are good. System B completes 30 of 50 and 27 are good.
- *Steps*: completion-conditioned quality is $30/40=0.75$ for A and $27/30=0.90$ for B. Offered success is $30/50=0.60$ for A and $27/50=0.54$ for B.
- *Result*: B looks better by 15 points on completed work and is worse by 6 points on offered work.
- *Interpretation / limits*: B improved the conditional number by completing fewer requests. Which denominator the decision uses must be stated; some decisions need both.

**Knowledge Check:**
1. How can completion-conditioned quality improve while user utility falls?
2. Why can user-level assignment be required for an interactive product?

**Guided Practice:**
(a) In 200 s, 80 requests are offered: 52 accepted and within thresholds, 10 accepted but late, 6 timeouts, 5 refusals, 4 tool failures, 3 unfinished. Compute goodput, offered success, and quality conditioned on the 62 accepted outcomes. (b) Build an offered-outcome ledger and an online validation plan specifying assignment, exposure, guardrails, sample-ratio checks, interference, and delayed effects.

**Feedback Contract:**
- *Expected Output*: (a) goodput $=52/200=0.26$/s; offered success $=52/80=0.65$; conditional $=52/62=0.839$. The terminal classes sum to 80. (b) All terminal classes, trajectory effects, thresholds, sensitivity, assignment integrity, and offline/online deltas by slice. Knowledge check 2: one user's sessions share state and learning, so request-level assignment mixes treatments inside a user.
- *Typical Error*: counting only final successes, reporting 0.839 as the success rate, or treating an offline proxy as causal proof.
- *Diagnostic Hint*: Which offered users disappeared from the denominator?
- *Concept to Revisit*: offered denominator and offline-to-online validity.

**Learning Outcome:**
Prevent a benchmark improvement from silently reducing production utility.

*(Effort: 35m instruction, 30m practice)*

---

### Lesson 15.6 — Evaluation Operations and Harness Trace

**Engineering Question:**
How should evaluation run as versioned infrastructure whose cost, coverage, drift, and escaped defects are observable?

**Concepts & Definitions:**
- **Evaluation funnel**: staged checks from deterministic contracts through controlled online evidence.
- **Run manifest**: immutable identity and lineage for data, system, grader, attempts, outcomes, and environment.
- **Unique defect yield**: adjudicated failures found by a stage and by no other stage.

**Mechanism Explanation:**
Use a staged funnel: deterministic unit/contract checks, sampled regression suites, adversarial and slice suites, grader/human audits, shadow/canary, and controlled online evidence. Measure each stage's cost, latency, rerun variance, unique defects, overlap, false blocks, and escaped incidents. The claim that staging lowers cost and escapes is a hypothesis to test, not a guarantee (**H**, CLM-016).

An immutable run manifest includes dataset/slice hashes, item/root request IDs, model/system/policy/harness/grader revisions, prompts and generation settings, all attempts, raw and normalized outputs, grades, timing, token/cost data, exclusions, terminal states, and environment.

**Production source trace.** At EleutherAI `lm-evaluation-harness` revision `d6de81643928d653435c431bae19945d41d32520`, `simple_evaluate` loads tasks through `TaskManager.load` and delegates to `evaluate`. `evaluate` calls `task.build_all_requests`, dispatches each request type with `getattr(lm, reqtype)`, calls `task.process_results` per document, and hands the per-document metrics to `_process_results` for aggregation (**O**, CLM-015). The task API exposes `build_all_requests`, `construct_requests`, `process_results`, `aggregation`, and `higher_is_better`. This is static inspection of one pinned implementation, not the definition of evaluation engineering.

**One tool's default is not this module's interval.** At the same revision, `simple_evaluate` and `evaluate` default to `bootstrap_iters=100000`. `stderr_for_metric` maps the `mean` aggregation to `mean_stderr`, the sample standard deviation over per-document values divided by $\sqrt n$. For a listed set of other metrics it uses `bootstrap_stderr`, which resamples per-document values with `rnd.choices(xs, k=len(xs))` (**O**, CLM-018). Both treat documents as independent and unpaired; there is no block key. A harness-reported `_stderr` therefore answers a different question from Lesson 15.4. Export per-document results and compute paired block effects yourself.

JudgeArena (2026) is frontier work on making judge, benchmark, prompt, inference backend, and metadata swappable and logged (**O**, CLM-017; abstract-level). It does not remove the need to validate the judge or reproduce performance claims.

**Quantitative Model / Trade-off Comparison:**
A stage is not justified by case count alone. For each stage report cost per run, defects found, unique defects, severity of the unique defects, false blocks, and escaped incidents. Compare cost per unique consequential defect under a predeclared decision rule. Early detection has separate value: a defect caught by a cheap early stage avoids the cost of running later stages on a broken build.

**Worked Example (synthetic):**
- *Input*:

| Stage | Cost per run | Defects found | Unique defects | Severe unique |
|---|---:|---:|---:|---:|
| Contract checks | \$2 | 6 | 0 | 0 |
| Slice suite | \$40 | 5 | 2 | 2 |
| Judge audit | \$120 | 4 | 1 | 0 |

- *Steps*: cost per unique defect is $40/2=\$20$ for the slice suite and $120/1=\$120$ for the judge audit. The contract stage has no unique defects; all six are also found downstream.
- *Result*: ranking by case count puts contract checks first. Ranking by severe unique yield puts the slice suite first.
- *Interpretation / limits*: zero unique yield does not make the contract stage worthless. It finds six defects for \$2 before \$160 of later stages run. Remove a stage only after measuring escapes without it. The costs are exercise assumptions.

**Knowledge Check:**
1. Which manifest fields localize a grader change from a model change?
2. Why is a pinned source path not the definition of evaluation engineering?
3. A harness reports `acc_stderr`. What unit does it treat as independent?

**Independent Practice:**
Run one immutable manifest through task loading, request construction, dispatch, per-document processing, aggregation, and metadata capture; inject one change at a time. Then recompute the uncertainty of one metric at block level and compare it with the harness-reported standard error.

**Feedback Contract:**
- *Expected Output*: a versioned run, complete attempts, stage costs, unique/overlap defects, false blocks, escapes, and the source trace. A two-line comparison of harness standard error and block-level standard error with the block key named. Knowledge check 1: grader revision, judge prompt hash, and parser version differ while model and dataset hashes match. Knowledge check 3: the document.
- *Typical Error*: changing data or graders invisibly between runs, or quoting a harness standard error as a paired release interval.
- *Diagnostic Hint*: What is the earliest differing manifest field? Which column of your per-document export is the block key?
- *Concept to Revisit*: run manifest; independent unit (Lesson 15.1).

**Learning Outcome:**
Operate evaluation as versioned production infrastructure with known cost and failure coverage.

*(Effort: 30m instruction, 30m practice)*

---

## 05 Literature & Production Source Map

Entries say what was opened and when. "Abstract-level" means only the abstract page was read in this revision; section pointers recorded in the registry from the 2026-09 pass were not re-checked.

**REFERENCE / BASELINE**

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Liang et al., 2022. Abstract re-opened 2026-10-01.
  - *Scope*: scenario and metric coverage with stated omissions (**O**, CLM-002). Its scenarios and rankings are a benchmark design, not a production workload.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023. Abstract re-opened 2026-10-01.
  - *Scope*: judge utility and the named biases in its settings (**O**, CLM-006). Its agreement figure is author-reported for its judges and data and is not used here as a general rate.
- [Judging the Judges](https://arxiv.org/abs/2406.07791) — Shi et al., 2024. Abstract re-opened 2026-10-01.
  - *Scope*: position bias varies by judge and task; three separate measurements (**O**, CLM-007).
- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) — Miller, 2024. arXiv HTML v1 opened 2026-10-01: §2.2 and Eq. 4 (clustered standard errors), §4.2 and Eq. 7 (paired differences), §5 and Eq. 9 (sample-size formula).
  - *Scope*: the cluster adjustment, paired analysis, and power planning used in Lesson 15.4 (**O**, CLM-023). Its worked numbers are the author's and are not reused. It does not define this module's three-state decision rule.

**RECOMMENDED ENGINEERING BASELINE** (this module's derivation; earlier revisions labeled this list "CURRENT DEFAULT")

Explicit evaluation contract; immutable dataset and run lineage; paired comparisons; block-level uncertainty; validated grader portfolio; offered-work denominators; critical-slice gates separate from weighted aggregates; cost/latency/quality reporting; staged regression and online validation.

These follow from stated assumptions (**D**, CLM-001, CLM-003, CLM-004, CLM-008, CLM-010, CLM-011, CLM-013, CLM-014, CLM-019, CLM-020, CLM-021). They are a recommendation. They are not the default of any tool, and this module has not measured how widely they are adopted.

**ONE IMPLEMENTATION'S DEFAULT** (observed, scoped to the named source)

- `lm-evaluation-harness` at the pinned commit: `bootstrap_iters=100000` by default; standard errors are computed over per-document values, unpaired and without a block key (**O**, CLM-018). This is one harness at one revision.

**INDUSTRY PREVALENCE:** not established by this module. No adoption survey of paired, block-aware, or predeclared release gating was opened (`TODO_VERIFY`, CLM-025).

**WORKLOAD-DEPENDENT:** grader mix, rubric, sample/weighting, repeats, block unit, thresholds and margins, risk slices, judge model, refresh cadence, online design, and acceptable evaluation cost.

**FRONTIER** (each scoped to what was read)

- [LiveBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e4a46394ba5378b3f9a186a5b4c650d1-Abstract-Conference.html) — White et al., ICLR 2025. Abstract re-opened 2026-10-01. Frequently updated questions from recent sources, objective ground-truth scoring, and varied tasks (**O**, CLM-012). No result figure from it is used here.
- [JudgeArena](https://arxiv.org/abs/2608.02620) — Lushtaku et al., 2026, arXiv v1. Abstract-level, opened 2026-10-01. One interface over several judge benchmarks with swappable judges and metadata logging (**O**, CLM-017). Its claims about tuned open judges are author-reported and not reproduced; its adoption is unknown.
- The wider 2025–2026 landscape of evaluation-statistics tooling and sequential release testing was not surveyed (`TODO_VERIFY`, CLM-025).

Neither frontier source proves zero contamination or universally valid automated judging.

**LEGACY / INSUFFICIENT:** one static benchmark as product quality; one mean without uncertainty; success-only scoring; independent tests on paired items; rows-as-independent for repeated turns; tag frequencies used as weights for overlapping slices; judge self-scoring without calibration; forced-choice without ties; changing prompts/graders invisibly; tuning repeatedly on the test set; one weighted score that offsets critical harm.

**PRODUCTION SOURCE TRACE**

- Repository: `EleutherAI/lm-evaluation-harness`
- Revision: `d6de81643928d653435c431bae19945d41d32520`
- Verified: 2026-09-26; files and symbols below re-read at this revision on 2026-10-01. Static inspection only; nothing was executed.
- Files/symbols: `lm_eval/evaluator.py::{simple_evaluate,evaluate}`; `lm_eval/api/task.py::Task.{build_all_requests,construct_requests,process_results,aggregation,higher_is_better}`; `lm_eval/evaluator_utils.py::_compute_task_aggregations`; `lm_eval/api/metrics.py::{stderr_for_metric,mean_stderr,bootstrap_stderr}`.
- Path: task loading/configuration → instance construction → request dispatch → per-document results → aggregation, standard error, and run metadata.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`. Seeded or generated lab data are synthetic and must be labeled so in every artifact.

### LAB A — Evaluation Contract and Dataset

- **Objective**: Build a reproducible contract and sampled dataset for one production decision.
- **Pre-Registered Hypothesis**: Preregistered units, weights, slices, and missingness will reveal at least one ranking or uncertainty change hidden by naive row-level scoring.
- **Independent Variables**: Sampling frame, split/freshness, weighting, slice definition, and deduplication.
- **Dependent Variables**: Coverage, leakage, weighted/unweighted effects, slice uncertainty, and exclusions.

- Define one production decision, population, independent unit, system boundary, estimands, thresholds, and missingness policy.
- Build an immutable dataset manifest with provenance, sampling/weights, deduplication, protected split, freshness, tags, and critical slices.
- Give at least two tags that overlap. Compute the weighted estimate by cells and by item weights, show they agree, and show the overlapping-slice sum for comparison.
- Break it with duplicates, near-duplicates, stale questions, slice imbalance, label leakage, repeated user/session rows, and post-result exclusions.
- Artifact: coverage map, leakage audit, weighted/unweighted estimates with the weight check ($\sum w=1$, item counts equal items), and limitations register.
- **Break & Falsify**: Inject duplicates, stale items, imbalance, leakage, clustered rows, and post-result exclusion; survival of a biased estimate falsifies the controls.
- **Alignment**: Lessons 15.1–15.2.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB B — Grader Meta-Evaluation

- **Objective**: Calibrate executable, semantic, judge, and human paths on the same stratified sample.
- **Pre-Registered Hypothesis**: Order balancing and independent labels will expose measurable judge error or sensitivity on at least one preregistered stress slice.
- **Independent Variables**: Grader/protocol, answer order/length/cue, rubric, judge model/prompt, and slice.
- **Dependent Variables**: Confusion/agreement, ties/abstention, swap consistency, slice error, stability, latency, and cost.

- Implement exact/executable, semantic, two LLM-judge, and blinded human/adjudication paths on the same stratified sample.
- Randomize and swap pairwise order; vary answer length, identity cues, rubric, judge prompt/model, and ambiguous cases.
- Measure confusion/agreement, ties/abstention, repeat stability, swap consistency, slice errors, latency, and full attempt cost.
- Artifact: grader card and a justified policy for automatic grade, dual grade, abstain, or expert escalation.
- **Break & Falsify**: Include ambiguous cases and deliberately invalid cues; a judge whose errors cannot be bounded must not control the gate.
- **Alignment**: Lesson 15.3.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Paired Regression Gate

- **Objective**: Implement the Lesson 15.4 release statistics contract with protected slice constraints.
- **Pre-Registered Hypothesis**: Block-level paired analysis will produce different uncertainty than row-level analysis on seeded clustered data, and at least one seeded scenario with a positive mean will not be a RELEASE.
- **Independent Variables**: System revision, block/repeat structure, within-block correlation, gate type, threshold, multiplicity policy, and number of blocks.
- **Dependent Variables**: Paired effect, interval width, simulated coverage, decision state, slice gate results, and rerun stability.

- Compare two system revisions on shared independent blocks with block effects, a block t interval, and a block bootstrap.
- Add stochastic repeats without pretending they are new items; exercise improvement, noninferiority, critical-slice gates, and the all-must-pass rule.
- **Required test**: (i) reproduce the R-15 numbers in Lesson 15.4 from the table; (ii) on seeded clustered data, report row and block intervals side by side; (iii) produce one positive-mean case that is INCONCLUSIVE and one that is HOLD; (iv) run a planning calculation and check it with a seeded simulation.
- Break the gate with unpaired analysis, tiny slices, metric shopping, rerun-until-pass, aggregate compensation, and success-only deletion.
- Artifact: **Release Decision Record** — predeclared contract, block effects, intervals with method and seed, each gate's state, the decision, the planning inputs, and the look log.
- **Break & Falsify**: Use tiny slices, metric shopping, rerun-until-pass, aggregate compensation, and success-only deletion; any undeclared selection that passes exposes an invalid gate. If row and block intervals agree on the seeded clustered data, the hypothesis is falsified for that seed and the correlation must be reported.
- **Alignment**: Lesson 15.4.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — End-to-End Evaluation Pipeline

- **Objective**: Run a versioned staged evaluation from deterministic checks through simulated/shadow online validation.
- **Pre-Registered Hypothesis**: Complete offered-outcome accounting will expose at least one regression hidden by accepted-success-only scoring in the injected set.
- **Independent Variables**: Failure class, trajectory effect, traffic slice, stage, assignment/exposure design, and system version.
- **Dependent Variables**: Quality, completion, safety, latency, cost, goodput, stage yield, false blocks, escapes, and online outcome.

- Build a staged suite and persist a full manifest through the pinned harness path.
- Evaluate request, turn, and trajectory outcomes; inject timeout, refusal, invalid output, tool/retrieval failure, retry, duplicate effect, and abandoned session.
- Shadow or simulate an online validation and compare offline deltas with completion, latency, cost, safety, and user outcome.
- Measure stage cost, unique defect yield, false blocks, rerun instability, escaped regressions, and detection time.
- Artifact: evaluation DAG, offered-outcome ledger, offline-online validity report, rollout/rollback gate, and TODO_VERIFY list.
- **Break & Falsify**: Inject timeout/refusal/invalid output/tool failure/retry/duplicate effect/abandonment and experiment-integrity faults; any invisible offered outcome falsifies completeness.
- **Alignment**: Lessons 15.5–15.6 and Incident 15.1.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total). The source trace is a separate 2h artifact (Section 09).

## 07 Break / Incident Scenarios

### Incident 15.1 — The Judge Says Better; Users Say Worse

- **Incident Symptoms**: A release wins offline pairwise evaluation and an aggregate gate, yet rollout increases abandonment and support escalation while the quality dashboard remains positive.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: Judge bias/drift, broken randomization, traffic mismatch, contamination, missing terminal outcomes, success-only retries, latency/tool/retrieval regressions, hidden harmful effects, metric selection, rows treated as independent, or invalid online measurement.
  2. *Rank Initial Plausibility*: Use timing, slice, and assignment evidence without assuming offline or online metrics are ground truth.
  3. *Identify Missing Evidence*: Recover manifests, weights, versions, paired outputs, block keys, order assignment, calibration labels, all attempts/states, latency/cost/effects, traffic slices, assignment/exposure logs, and preregistered gates.
  4. *Design Discriminating Tests*: Blind re-grade with independent oracles/humans, swap order, include offered work, recompute the offline interval at block level, replay both systems on identical items, and verify assignment/sample ratios; state falsifiers.
  5. *Execute Causal Diagnosis*: Decompose offline and online deltas by slice and rank supported interacting explanations.
  6. *Prescribe Mitigation and Prevention*: Pause/roll back if guardrails require it, then repair the earliest invalid measurement or system boundary.
  7. *Remeasure*: Offline effect, judge error, completion, latency, cost, safety, user outcomes, and experiment integrity.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Evidence-Gated LLM Release

Design and operate the release evaluation for a system change serving heterogeneous interactive and automated workflows.

**Constraints** (declare any value you assume and label it synthetic):
- Interactive traffic arrives as multi-turn conversations, so one conversation contributes several rows.
- Two tags overlap (for example, non-English and long-context).
- One critical slice has fewer than 30 independent blocks.
- The judge returns no verdict for some items; those items stay in the offered denominator.
- One interim look is planned before the final analysis.

**Required Deliverables**:
1. Decision contract, population, independent unit, boundary, estimands, and missingness.
2. Immutable dataset/slice manifest with leakage, freshness, and weighting controls; a weighted estimate that counts each item once.
3. Executable, judge, and human grader cards with meta-evaluation.
4. Release Decision Record: paired block effects, interval method, $\epsilon$ and $\tau$, critical-slice gates, multiplicity and look policy, planning calculation, and a RELEASE/HOLD/INCONCLUSIVE decision.
5. Offered-request quality/latency/cost/effect accounting.
6. Pinned harness trace and staged cost/coverage/escape telemetry.
7. Online validation, guardrail, rollout, and rollback plan.
8. Evidence-backed diagnosis of Incident 15.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit a pinned trace that maps:
1. Entry through `simple_evaluate` and task loading.
2. Request construction through `build_all_requests`/`construct_requests`.
3. Dispatch of each request type to the model.
4. Per-document `process_results` and metric aggregation.
5. Where the standard error is computed, which unit it treats as independent, and how that differs from a paired block interval.

Separate observed implementation behavior from general evaluation requirements, and state what was read statically versus executed.

### Required Artifact: Release Decision Record

One record per release decision, with the seven contract items of Lesson 15.4, the look log, and the named owner of any risk acceptance. Reviewers recompute every interval from the block effects in the record.

### Reference Checks (fixtures in this README only)

Reviewers use these to check arithmetic. A submission on other data is checked against its own inputs.

- R-15: $\hat\Delta=0.25$; block t interval $[-0.087,\,0.587]$; T4 interval $[-0.506,\,0.306]$; INCONCLUSIVE.
- P-15: $\hat\Delta=0.396$; block t interval $[0.270,\,0.522]$; T4 interval $[-0.911,\,-0.256]$; HOLD.
- Lesson 15.2 Example A: 0.70 correct; 0.75 and 0.625 by overlapping slices; 0.765 reweighted.

### Rubric Dimensions

- **Validity and Population** (Deliverables 1–2): *Insufficient* reports a score, or weights overlapping slices. *Competent* defines construct, population, unit, estimator, and decision, and counts each item once. *Strong* proves lineage, weighting, dependence, slices, and shift limitations.
- **Graders and Statistics** (Deliverables 3–4): *Insufficient* trusts a judge or a mean, resamples rows, or releases on a positive mean. *Competent* calibrates graders, reports block-level paired intervals, and applies the three-state rule with predeclared thresholds. *Strong* also handles multiplicity and looks, shows planning sensitivity, and states where the interval is approximate.
- **Accounting and Transfer** (Deliverables 5, 7): *Insufficient* scores successes only. *Competent* retains failures/cost/latency/effects and plans guarded online validation. *Strong* verifies assignment integrity and explains offline-online divergence.
- **Operations and Diagnosis** (Deliverables 6, 8): *Insufficient* changes fixtures invisibly. *Competent* versions the pipeline and source trace. *Strong* measures unique yield, false blocks, escapes, and discriminating remeasurement.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Evaluation contract and independent unit | Lesson 15.1 | 15.1 Guided Practice (a)–(b); LAB A | Mastery Deliverable 1; Incident 15.1 steps 1, 3 | LAB A contract with the block key named; $n_{\text{eff}}$ calculation |
| Dataset lineage and slice weighting | Lesson 15.2 | 15.2 Guided Practice (a)–(b); LAB A | Mastery Deliverable 2; rubric *Validity and Population* | LAB A manifest, leakage audit, weight check, cell and item-weight estimates |
| Grader and human protocol validity | Lesson 15.3 | 15.3 Guided Practice (a)–(b); LAB B | Mastery Deliverable 3; Incident 15.1 step 4 (blind re-grade, order swap) | LAB B grader card with confusion, swap consistency, and escalation policy |
| Paired block statistics and release gates | Lesson 15.4 | 15.4 Guided Practice (P-15); LAB C required test (i)–(iv) | Mastery Deliverable 4; Section 09 Reference Checks; rubric *Graders and Statistics* | Release Decision Record |
| End-to-end and online validity | Lesson 15.5 | 15.5 Guided Practice (a)–(b); LAB D | Mastery Deliverables 5, 7; Incident 15.1 steps 5–7 | LAB D offered-outcome ledger and offline-online validity report |
| Evaluation operations | Lesson 15.6 | 15.6 Independent Practice; LAB D | Mastery Deliverable 6; rubric *Operations and Diagnosis* | Run manifest and funnel metrics (cost, unique yield, false blocks, escapes) |
| Pinned harness source trace | Lesson 15.6 (source trace and tool-default paragraphs) | 15.6 Independent Practice (block-level recomputation) | Section 09 Production Source Trace items 1–5; Mastery Deliverable 6 | Trace pinned to commit `d6de8164…` |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner successfully completing Module 15 must be able to:
1. Name the decision, population, independent unit, boundary, and estimand.
2. Reproduce dataset, system, harness, and grader versions.
3. Expose missingness, failures, retries, unfinished work, latency, cost, and effects.
4. Validate automated/human graders and quantify paired effects with a block-level interval.
5. Compute a weighted estimate that counts each item once and prevent aggregate compensation of critical regressions.
6. Reach and defend a RELEASE, HOLD, or INCONCLUSIVE decision from predeclared gates.
7. Test rather than assume offline-to-online transfer.
8. Trace a pinned harness without generalizing it.

### Module Wrap-Up (Final Mental Model Reconstruction)

- **The Core Invariant**: Evaluation is a versioned measurement-and-decision system.
- **The Evidence Path**: decision/population → sampled observations → graders → block-level estimates → explicit gates → guarded online validation.
- Credibility comes from population fit, valid observations, explicit value judgments, complete denominators, and falsifiable links to production outcomes—not from a precise-looking score.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
