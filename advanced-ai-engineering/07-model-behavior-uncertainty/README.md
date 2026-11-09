# Module 07 — Model Behavior & Uncertainty

## 00 Why This Module Exists

An LLM can be fluent and wrong, uncertain and correct, confidently wrong on one slice, or well calibrated on average while failing a subgroup. “Hallucination,” “confidence,” and “reliability” are not measurements until the engineer defines the event, evidence boundary, population, sampling unit, and consequence.

This module builds the operational loop:

$$
\text{behavior contract} \to \text{labeled events and slices}
\to \text{uncertainty signal} \to \text{calibration}
\to \text{accept/abstain/escalate} \to \text{regression monitoring}.
$$

The module covers failure taxonomy, factuality/truthfulness measurement, calibration, distribution shift, selective prediction, uncertainty signals for free-form generation, conformal factuality, and behavior regression. Test-time search and verifiers belong to Module 06; general evaluation-program design belongs to Module 15; retrieval quality to Modules 09–10; security behavior to Module 18.

**Research cutoff:** 2026-09-27 for the original revision. Sources added in the 2026-09-30 revision (conformal construction, frontier items) were searched through 2026-09-30. Claims not listed as changed in the registry were not re-verified.

**Module Orientation**

- **Engineering problem:** determine when a model output should be trusted, checked, abstained, or escalated under a declared deployment contract.
- **What you will do:** annotate failures at claim level; construct and calibrate confidence signals; measure proper scores and risk-coverage; break uncertainty methods under shift and correlated errors; trace generation scores in current source; and diagnose a release regression hidden by aggregates.
- **Environment:** Python 3.10+ with numerical/statistical tooling, a reproducible model endpoint or local model, and an annotation store that preserves blinded labels and adjudication. Pin model, prompt, tokenizer, evaluator, and calibration data revisions.
- **Evidence rule:** preserve source observations (**O**), assumption-backed derivations (**D**), and telemetry-dependent hypotheses (**H**). A detector score is not a factuality label, and a marginal guarantee is not a per-request promise.

## 01 Baseline Assumptions

- **Module 00:** measurands, units, sampling frames, uncertainty intervals, preregistration, leakage, multiple comparisons, provenance, and falsification.
- **Module 01, Lesson 1.2:** logits, softmax with temperature, top-k/top-p sampling, the sampling contract, and autoregressive factorization over token IDs with sequence log-likelihood. Module 01 does not teach tokenization itself. The few tokenization facts this module needs (one string can map to different numbers of tokens, and a sequence score is a sum over tokens) are taught at first use in Lesson 7.2.
- **Module 06:** candidate sampling, verifier fallibility, oracle-versus-selected accuracy, and test-time cost. This module uses these mechanisms only as uncertainty inputs.
- **Mathematics:** conditional expectation, Bernoulli loss, binning, logarithms, and empirical distributions.
- **Product contract:** an answer event, cost of wrong answer, cost of abstention/escalation, evidence authority, knowledge date, and target population must be declared.

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
  instruction: 5h
  guided_practice: 3h
  labs: 20h
  assessment: 3h
  source_trace: 2h
  total: 33h
```

The learner must be able to:

1. replace “hallucination” with an auditable, task-relative failure taxonomy;
2. define confidence as a forecast of a specific event on a specific population;
3. measure calibration without relying on ECE alone;
4. distinguish likelihood, self-evaluation, sample disagreement, semantic uncertainty, and learned scores;
5. select abstention thresholds using risk-coverage and application utility;
6. implement a split-conformal claim filter and state exactly what its guarantee covers and which assumptions sustain it;
7. detect slice and shift regressions across model, prompt, runtime, retrieval, tool, and evaluator versions;
8. diagnose apparent behavior changes through label audit and matched reruns.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
task + evidence + time + output obligations
                    |
                    v
             behavior contract
                    |
          output -> events/claims -> labels + severity + slices
                    |
     +--------------+-----------------------------+
     |              |              |              |
 likelihood   self-evaluation  sample/semantic  external score
     +--------------+--------------+--------------+
                    |
             held-out calibration
                    |
       proper scores + reliability + ranking
                    |
      threshold -> answer / abstain / check / escalate
                    |
       coverage + conditional risk + utility + capacity
                    |
        versioned regression and shift monitoring
```

Never collapse the following:

- correctness, factual support, truthfulness, usefulness, instruction compliance, and safety;
- token likelihood, answer correctness probability, and confidence expressed in natural language;
- calibration, discrimination/ranking, and raw task accuracy;
- low risk at low coverage and reliability for the whole population;
- marginal conformal coverage and per-prompt correctness.

---

## 04 Lessons

### Lesson 7.1 — Behavior Contracts and Failure Taxonomy

**Engineering Question:**
What exactly failed, relative to which evidence and obligation?

**Concepts & Definitions:**

A behavior contract declares:

- task, user population, language and time window;
- allowed context, retrieval/tool outputs, authoritative sources, and knowledge cutoff;
- required answer, refusal, citation, format, and uncertainty behavior;
- unit of analysis: response, claim, citation, step, session, or user outcome;
- label rules, ambiguity/adjudication, severity, and downstream consequence.

**Mechanism Explanation:**

A useful taxonomy may include unsupported fabrication, contradiction with supplied evidence, stale fact, wrong attribution/citation, reasoning error, omission, instruction violation, invalid refusal, format failure, and evaluator failure. Categories can interact and are not universal: an unsupported claim in closed-book QA differs from a contradiction in grounded summarization.

TruthfulQA isolates susceptibility to selected human misconceptions (**O**, CLM-008). FActScore decomposes long-form output into atomic facts and measures supported-fact precision (**O**, CLM-009). Neither is “the hallucination metric”: TruthfulQA is a fixed behavior slice, while FActScore depends on claim decomposition, retrieval, evidence authority, entailment, and time. Factual precision also omits required-fact recall and usefulness.

Kalai and Vempala prove a lower bound for a defined class of arbitrary facts under their calibration assumptions (**O**, CLM-013). The theorem matters because its scope is narrow: it does not make systematic facts, grounded answers, or all production errors inevitable. A 2025 follow-up by Kalai, Nachum, Vempala, and Zhang argues that binary-graded evaluations reward guessing over abstention. That is an argument about evaluation incentives, read at abstract level here, and it motivates scoring abstention explicitly as in Lesson 7.3. It is not a measured production result (**O**, CLM-019).

**Worked Example:**
A response can contain four supported claims, one stale claim, and one required omission. A response-level “hallucinated” label hides which intervention is relevant; a claim/evidence/time table exposes both stale-fact precision and missing-required-fact recall.

**Knowledge Check:**
1. Why are factual support and usefulness different events?
2. Which timestamp and authority are needed to label a time-sensitive claim?

**Guided Practice:**
Annotate a small blinded set at response and claim level with two raters. Record ambiguity, disagreement, adjudication, severity, and evaluator revision.

**Feedback Contract:**
- *Expected Evidence*: Versioned guide, independent labels, agreement/disagreement counts, adjudication reasons, and intervention mapping.
- *Common Failure*: Using an automatic detector as its own ground truth.
- *Diagnostic Hint*: Can two trained raters reproduce the event boundary?
- *Concept to Revisit*: Behavior Contract and Label Provenance.

**Learning Outcome:**
Create failure labels that identify an intervention instead of hiding distinct causes under one rate.

*(Effort: 40m instruction, 25m practice)*

---

### Lesson 7.2 — Confidence, Calibration, and Proper Measurement

**Engineering Question:**
Probability of what, for whom, and under which sampling process?

**Concepts & Definitions:**

- **Forecast event $Y_i$:** a binary event fixed by the behavior contract, for example "the delivered answer to item $i$ is labeled correct under guide v3". Changing the label guide changes $Y$.
- **Confidence $q_i\in[0,1]$:** a number the system emits *before* seeing $Y_i$, intended as a probability of that event. A logit, a log-likelihood, or a verbal phrase is not yet a confidence until it is mapped to $[0,1]$ and validated.
- **Calibration:** agreement between $q$ and the empirical frequency of $Y$ among items given that $q$, on a declared population (**D**, CLM-001).
- **Discrimination (ranking):** whether higher $q$ goes with $Y=1$ more often, for example AUROC. It is unaffected by any strictly increasing relabeling of $q$, so it can be good while calibration is bad.
- **Sharpness:** how far forecasts move away from the base rate. Always forecasting the base rate is calibrated but has no sharpness.
- **Proper scoring rule:** a loss that is minimized in expectation by reporting the true probability; Brier and NLL are proper, and binned ECE is not a scoring rule at all (**D**, CLM-002).
- **Reliability table/diagram:** per-bin count, mean confidence, and observed frequency under a declared binning.

**Quantitative Model / Derivation:**
For binary event $Y$ and forecast $Q$, population calibration targets:

$$
\mathbb{E}[Y\mid Q=q]=q.
$$

where:
- $Q$ is the confidence as a random variable over the declared population, and $q$ is one particular value it takes;
- $Y\in\{0,1\}$ is the forecast event, so $\mathbb{E}[Y\mid Q=q]=P(Y=1\mid Q=q)$ is the true frequency of the event among items that received confidence $q$.

Read it as: "of all items stamped with confidence 0.7, a fraction 0.7 actually have $Y=1$." Example: if 100 items get $q=0.7$ and 58 of them are correct, the observed frequency is $0.58$, so on that group the forecaster is overconfident by $0.12$ (up to sampling noise).

This is an estimand, not a finite-sample test result. Change correctness rules, prompts, languages, time, or subgroups and the estimand changes. Aggregate calibration can coexist with severe subgroup miscalibration.

For confidence bins $B_b$, a common estimator is:

$$
ECE=\sum_b\frac{n_b}{n}|acc(B_b)-conf(B_b)|.
$$

where:
- $B_b$ is the set of items whose confidence falls in bin $b$ (bin edges are declared below), $n_b=|B_b|$, and $n$ is the total item count, so the weights $n_b/n$ sum to 1;
- $acc(B_b)=\frac{1}{n_b}\sum_{i\in B_b}y_i$ is the observed event frequency in the bin (written $\bar y$ in the worked example);
- $conf(B_b)=\frac{1}{n_b}\sum_{i\in B_b}q_i$ is the mean confidence in the bin (written $\bar q$);
- $|\cdot|$ is the absolute gap, so over- and under-confidence both count as error.

Step-by-step: (1) assign each item to a bin by its $q_i$; (2) per non-empty bin compute $\bar y$ and $\bar q$; (3) take $|\bar y-\bar q|$; (4) weight by $n_b/n$ and sum. ECE is a binned plug-in estimate of $\mathbb{E}\,|\mathbb{E}[Y\mid Q]-Q|$, which is zero under perfect calibration.

ECE depends on binning and finite-sample noise. Report bin definitions/counts and uncertainty. Complement it with accuracy and ranking plus proper scores such as:

$$
Brier=\frac1n\sum_i(q_i-y_i)^2,
$$

$$
NLL=-\frac1n\sum_i[y_i\log q_i+(1-y_i)\log(1-q_i)].
$$

where $y_i\in\{0,1\}$ is item $i$'s label, $q_i$ its confidence, $n$ the item count, and $\log$ is the natural log (see conventions below).

Per item, the NLL bracket keeps only one term because $y_i$ is 0 or 1: if $y_i=1$ the loss is $-\ln q_i$; if $y_i=0$ it is $-\ln(1-q_i)$. In words, NLL charges $-\ln(\text{probability given to what actually happened})$. Brier is the squared distance between the forecast and the 0/1 outcome, so it lies in $[0,1]$ per item; NLL is unbounded as the probability given to the realized outcome goes to 0.

*Why these are proper (step-by-step).* Suppose the true probability of $Y=1$ is $p$ and the forecaster reports $q$.
1. Expected Brier loss: $p(1-q)^2+(1-p)q^2$. Its derivative in $q$ is $-2p(1-q)+2(1-p)q=2(q-p)$, which is zero only at $q=p$, and the function is convex, so honest reporting minimizes it.
2. Expected NLL: $-p\ln q-(1-p)\ln(1-q)$. Its derivative is $-p/q+(1-p)/(1-q)$; setting it to zero gives $p(1-q)=(1-p)q$, i.e. $q=p$.
3. Example with $p=0.7$: reporting $q=0.7$ gives expected Brier $0.7(0.09)+0.3(0.49)=0.210$; exaggerating to $q=0.9$ gives $0.7(0.01)+0.3(0.81)=0.250$, which is worse.

These scores combine calibration and sharpness; they do not localize the failure cause.

*Conventions used in this module:* NLL uses the natural logarithm (units: nats). A forecast of exactly $0$ or $1$ on the wrong outcome gives infinite NLL; report that as $\infty$ with the offending item IDs instead of silently clipping. If a pipeline must clip, declare $\epsilon$ and report both clipped NLL and the count of clipped items. ECE uses equal-width bins $[0,0.2),[0.2,0.4),\dots,[0.8,1.0]$ (the last bin closed), weights each non-empty bin by $n_b/n$, and skips empty bins. Every ECE report states its bin edges.

Scalar temperature scaling uses:

$$
p_T(y\mid x)=softmax(z(x)/T)_y,\quad T>0.
$$

where:
- $x$ is the input, $y$ a candidate class/answer option (here $y$ is a class index, not the 0/1 label $y_i$ above);
- $z(x)$ is the model's logit vector, with entry $z_y$ for class $y$;
- $softmax(v)_y=e^{v_y}/\sum_{j}e^{v_j}$;
- $T$ is the single fitted temperature ($T=1$ leaves the model unchanged; $T>1$ flattens, $T<1$ sharpens). Do not confuse $T$ with the threshold $t$ of Lesson 7.3.

*Why argmax is preserved.* For $T>0$, $z_a>z_b \Rightarrow z_a/T>z_b/T \Rightarrow e^{z_a/T}>e^{z_b/T}$, and all classes share the same denominator, so the class order never changes; only the gaps between probabilities do.

*Example.* Logits $z=(2,1,0)$ give probabilities $(0.665, 0.245, 0.090)$ at $T=1$, $(0.506, 0.307, 0.186)$ at $T=2$, and $(0.867, 0.117, 0.016)$ at $T=0.5$. Class 1 stays the argmax in all three; its probability, which is the confidence being calibrated, moves from 0.506 to 0.867. In practice $T$ is chosen to minimize NLL on the calibration split.

It preserves argmax but changes probability sharpness (**O**, CLM-003). Fit $T$ only on calibration data, freeze it before testing, and repeat after shift. One scalar cannot generally correct task-, class-, language-, or subgroup-conditional errors.

For free-form generation, declare how a scalar is constructed: selected-token log probability, length normalization, answer-option probability, prompted P(True), learned probe, verifier, or another signal. Jiang et al. show calibration problems in studied generative QA models (**O**, CLM-006); Kadavath et al. show promising but task-dependent P(True)/P(IK) behavior (**O**, CLM-007). Neither licenses universal self-confidence semantics.

*Tokenization facts needed here (taught at first use).* A tokenizer maps text to a sequence of token IDs. The same answer can be expressed with different numbers of tokens, and one string can have more than one tokenization. A sequence log-likelihood is the sum of per-token log-probabilities (Module 01, Lesson 1.2), so it is a score for a *token sequence under one tokenizer*, not for an answer's correctness. Synthetic illustration: the answer "Paris" generated as 2 tokens with probabilities $0.9$ and $0.8$ has sequence probability $0.72$ ($-0.329$ nats). The equally correct "The capital is Paris" generated as 4 tokens with probabilities $0.9, 0.8, 0.7, 0.9$ has $0.4536$ ($-0.790$ nats). Length normalization (mean log-probability) gives $-0.164$ and $-0.198$ nats, which is closer but still different. (Arithmetic: $\ln 0.9=-0.1054$, $\ln 0.8=-0.2231$, $\ln 0.7=-0.3567$. Two tokens: $-0.1054-0.2231=-0.3285$, mean $-0.3285/2=-0.164$. Four tokens: $-0.1054-0.2231-0.3567-0.1054=-0.7906$, mean $-0.7906/4=-0.198$. Summing logs is the same as multiplying probabilities: $e^{-0.3285}=0.72$.) Two consequences follow: raw and length-normalized likelihoods rank equally correct answers differently, and a tokenizer or chat-template revision changes these scores without any change in correctness. Pin the tokenizer revision with the confidence definition.

**Worked Example (synthetic exercise data):**
*Input.* One model answered 10 held-out questions; $y_i=1$ means the answer was labeled correct, so the answer accuracy is $6/10$ for both forecasters. Two confidence constructions, A and B, forecast the same events:

| item | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| $y$ | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| $q^A$ | 0.95 | 0.85 | 0.75 | 0.65 | 0.65 | 0.55 | 0.45 | 0.35 | 0.25 | 0.05 |
| $q^B$ | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |

*Steps.*

1. Brier for A: squared errors are $0.0025, 0.0225, 0.0625, 0.4225, 0.1225, 0.3025, 0.3025, 0.1225, 0.5625, 0.0025$; the sum is $1.925$, so $Brier_A=0.1925$. For B, five items contribute $0.16$ and five contribute $0.36$, so $Brier_B=2.2/10=0.2200$.
2. NLL (nats) for A: the per-item losses are $-\ln q_i$ when $y_i=1$ and $-\ln(1-q_i)$ when $y_i=0$; for example item 9 contributes $-\ln 0.25=1.3863$. The mean is $NLL_A=0.5447$. For B, $NLL_B=0.6325$. (All ten A losses: $0.0513, 0.1625, 0.2877, 1.0498, 0.4308, 0.7985, 0.7985, 0.4308, 1.3863, 0.0513$, sum $5.4475$. For B every item gives either $-\ln0.6=0.5108$ (items 1, 2, 3, 5, 6, 8, 10) or $-\ln0.4=0.9163$ (items 4, 7, 9), so $NLL_B=(7\cdot0.5108+3\cdot0.9163)/10=0.6325$.)
3. Five-bin ECE for A: bin members are $[0,0.2)$: item 10; $[0.2,0.4)$: items 8, 9; $[0.4,0.6)$: items 6, 7; $[0.6,0.8)$: items 3, 4, 5; $[0.8,1.0]$: items 1, 2. These bins hold $(n, \bar y, \bar q)$ = $(1, 0, 0.05)$, $(2, 0.5, 0.30)$, $(2, 0.5, 0.50)$, $(3, 0.667, 0.683)$, $(2, 1.0, 0.90)$. Weighted gaps: $0.1(0.05)+0.2(0.20)+0.2(0)+0.3(0.0167)+0.2(0.10)=0.0700$. For B only two bins are non-empty: $(5,0.4,0.4)$ and $(5,0.8,0.6)$, giving $0.5(0)+0.5(0.2)=0.1000$.
4. Two-bin ECE (edges $[0,0.5),[0.5,1.0]$): A has bins $(4,0.5,0.275)$ and $(6,0.667,0.733)$, so $0.4(0.225)+0.6(0.0667)=0.1300$. B is unchanged at $0.1000$.
5. Ranking: counting positive–negative pairs with ties as one half, AUROC is $18.5/24=0.771$ for A and $17/24=0.708$ for B. AUROC is the probability that a random correct item gets a higher $q$ than a random wrong item; there are $6\times4=24$ (positive, negative) pairs. For A, the positives $0.95,0.85,0.75$ each beat all 4 negatives ($12$); $0.65$ beats $0.55,0.35,0.05$ and ties $0.65$ ($3.5$); $0.45$ beats 2 ($2$); $0.25$ beats 1 ($1$): total $18.5$. For B, the four positives at $0.60$ each beat three negatives and tie one ($4\times3.5=14$), and the two positives at $0.40$ tie three negatives ($2\times1.5=3$): total $17$.

*Result.* A beats B on Brier, NLL, AUROC, and five-bin ECE. With two bins, however, B has the *lower* ECE ($0.100<0.130$). The ECE ranking depends on the binning choice, while both proper scores give the same ordering.

*Interpretation and limits.* ECE is a binned estimator, not a property of the forecaster, which is why the bin edges must be reported (**D**, CLM-002). Ten items cannot support a claim that A is better calibrated in general; with $n=10$ the uncertainty is larger than every gap shown here. The example teaches the computation only. It is not evidence about any real model.

**Knowledge Check:**
1. Does temperature scaling change the argmax class?
2. Can aggregate calibration coexist with subgroup miscalibration?

**Guided Practice:**
(a) By hand, for $y=(1,0,1,1)$ and $q=(0.9,0.3,0.6,0.8)$, compute Brier, NLL in nats, and two-bin ECE with the edges above. (b) Fit temperature on a calibration split, freeze it, and evaluate raw and scaled scores on in-distribution and shifted test sets. Report bins, counts, uncertainty, proper scores, ranking, and slice results.

**Feedback Contract:**
- *Expected Evidence*: (a) Brier $=(0.01+0.09+0.16+0.04)/4=0.0750$; NLL $=(0.1054+0.3567+0.5108+0.2231)/4=0.2990$ nats (item 2 has $y=0$, so it contributes $-\ln(1-0.3)$); two-bin ECE $=0.25$, because the low bin holds only item 2 ($|0-0.3|$, weight 0.25) and the high bin holds items 1, 3, and 4 with $\bar y=1$ and $\bar q=0.767$ (gap $0.233$, weight 0.75). The sum is $0.075+0.175=0.25$. (b) Exact event and population, split lineage, fitted $T$, raw predictions, reproducible binning, and held-out results.
- *Common Failure*: Using $\log_{10}$ or clipping without saying so, which makes NLL incomparable. Fitting $T$ or choosing bins on the reported test set.
- *Diagnostic Hint*: If your ECE differs, list each bin's members; an item at exactly $0.5$ belongs to the upper bin under these edges. For (b), ask which examples determined the transformation.
- *Concept to Revisit*: Calibration Versus Discrimination; Proper Scoring Rule.

**Learning Outcome:**
Turn “confidence” into a reproducible forecast and prove its meaning on held-out data.

*(Effort: 65m instruction, 30m practice)*

---

### Lesson 7.3 — Selective Prediction and Abstention

**Engineering Question:**
Which outputs should the system deliver, and what happens to the rest?

**Concepts & Definitions:**

- **Acceptance rule:** deliver the answer when its score is at least $t$ ($s_i\ge t$). Otherwise route the item to a fallback action.
- **Coverage:** the fraction of the target population that is answered directly.
- **Selective (conditional) risk:** the expected loss among *accepted* items only (**D**, CLM-005).
- **Rejected-set outcome:** what happens to non-accepted items, for example fallback success, delay, abandonment, and whether the model's answer would have been correct.
- **Utility:** a declared per-item value that combines accepted-correct gain, accepted-wrong harm (severity-weighted), and fallback cost. It is a product decision, not a statistical property.
- **Operating point:** a threshold chosen on data other than the reporting test set, together with its coverage, risk, utility, subgroup coverage, and fallback load.

**Quantitative Model / Derivation:**
For acceptance event $A_t$ at threshold $t$ and loss $L$:

$$
coverage(t)=P(A_t),\qquad risk(t)=\mathbb{E}[L\mid A_t].
$$

where:
- $s_i$ is the acceptance score of item $i$ (in the example below, $s_i=q^A_i$) and $t$ is the threshold;
- $A_t$ is the event "the item is accepted", i.e. $s_i\ge t$;
- $L$ is the per-item loss of the delivered answer; with 0/1 loss ($L=1$ if wrong), $risk(t)$ is the error rate among accepted items.

Empirical versions on $n$ items: $coverage(t)=\#\{i: s_i\ge t\}/n$ and $risk(t)=\#\{\text{accepted and wrong}\}/\#\{\text{accepted}\}$. The risk denominator is the accepted count, not $n$, which is why risk is undefined when nothing is accepted. Raising $t$ can only shrink the accepted set, so coverage never increases with $t$; risk usually falls but is not guaranteed to be monotone.

Plot risk against coverage and report prespecified operating points. A system can achieve near-zero answered risk by rejecting almost everything. Therefore record abstention/retry/escalation cost, user abandonment, queue/capacity impact, subgroup coverage, and risk among rejected cases when labels are obtainable.

Calibration and ranking answer different questions. A monotone score may rank risk well but be numerically miscalibrated; a globally calibrated score can rank poorly inside a narrow operating range. Thresholds selected on the test set leak outcomes. Thresholds selected once can fail after model, prompt, traffic, or knowledge shift.

Possible actions are not only “answer” or “refuse”: retrieve evidence, invoke a deterministic tool, sample/reason again, ask a clarifying question, return a bounded uncertainty set, or escalate. Each action needs its own cost and failure contract.

**Worked Example (synthetic exercise data and costs):**
*Input.* Use the 10 items and score $q^A$ from Lesson 7.2 as the acceptance score. The exercise utility is $+1$ for an accepted correct answer, $-4$ for an accepted wrong answer, and $-0.3$ for each item routed to escalation, assuming escalation always resolves the item. The escalation queue can absorb at most 5 of every 10 items.

*Steps.* For each threshold, count accepted items ($q^A\ge t$), wrong accepted items, and rejected items. Then compute total utility as $1\cdot\text{right}-4\cdot\text{wrong}-0.3\cdot\text{rejected}$, and divide by 10 for utility per item. Example row $t=0.6$: accepted items 1–5 ($q^A\ge0.6$), of which item 4 is wrong, so right $=4$, wrong $=1$, rejected $=5$; coverage $5/10=0.5$, risk $1/5=0.20$, utility $(4-4-1.5)/10=-0.15$. Row $t=0.7$: accepted items 1–3, all right, so $(3-0-2.1)/10=+0.09$.

| $t$ | coverage | accepted wrong | selective risk | rejected (of which model was right) | utility / item | within escalation capacity? |
|---|---|---|---|---|---|---|
| 0.0 | 1.0 | 4 | 0.40 | 0 (0) | −1.00 | yes |
| 0.5 | 0.6 | 2 | 0.33 | 4 (2) | −0.52 | yes |
| 0.6 | 0.5 | 1 | 0.20 | 5 (2) | −0.15 | yes |
| 0.7 | 0.3 | 0 | 0.00 | 7 (3) | +0.09 | **no** |
| 0.9 | 0.1 | 0 | 0.00 | 9 (5) | −0.17 | **no** |

*Result.* Selective risk reaches zero at both $t=0.7$ and $t=0.9$. However, $t=0.9$ throws away five answers that were correct and has lower utility than $t=0.7$. Utility alone would pick $t=0.7$, but that rejects 7 items against a capacity of 5. The best feasible threshold in this table is $t=0.6$: coverage 0.5, risk 0.20, utility −0.15 per item.

*Interpretation and limits.* The selected operating point depends on the cost weights and the capacity constraint as much as on the score. Change the $-4$ harm weight or add fallback failure, and the choice moves. Ten items give no usable uncertainty, so a real sweep needs a separate threshold-selection split, intervals on risk and coverage, and subgroup coverage. The table also shows that "risk 0" can hide rejected users whose answers were correct.

**Knowledge Check:**
1. How can a system drive selective risk toward zero without becoming useful?
2. Why must fallback queueing and failure be included in utility?

**Guided Practice:**
(a) Repeat the table with $q^B$ from Lesson 7.2 at $t\in\{0.5,0.61\}$, using the same costs. (b) Sweep thresholds on a selection split, freeze the choice, then report coverage, conditional risk, false-accept severity, fallback volume, end-to-end success, cost, and subgroup effects on a separate test split.

**Feedback Contract:**
- *Expected Evidence*: (a) With $t=0.5$, B accepts items 1–5: coverage $0.5$, 1 wrong, risk $0.20$, 5 rejected (2 of which the model got right), utility $(4-4-1.5)/10=-0.15$ per item. With $t=0.61$, B accepts nothing: coverage 0, risk undefined, utility $-0.30$ per item, and 10 rejections exceed capacity. (b) Frozen threshold-selection process, risk–coverage curve with uncertainty, fallback outcomes, and capacity impact.
- *Common Failure*: Reporting risk without coverage or rejected outcomes, or reporting "risk 0" for an empty accepted set instead of "undefined".
- *Diagnostic Hint*: What happened to every case the model did not answer, and could the fallback absorb them?
- *Concept to Revisit*: Selective Utility Contract.

**Learning Outcome:**
Defend an operating point by coverage, conditional risk, utility, subgroup impact, and downstream capacity.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 7.4 — Uncertainty for Free-Form Generation

**Engineering Question:**
How can we estimate uncertainty when many strings express the same answer?

**Concepts & Definitions:**

Useful signal families include:

- token/sequence likelihood from an accessible model;
- prompted or trained self-evaluation such as P(True);
- disagreement across stochastic samples;
- semantic clustering and entropy over meanings;
- hidden-state probes or independent learned detectors;
- retrieval/evidence support and entailment scores.

**Mechanism Explanation:**

SelfCheckGPT uses black-box sample inconsistency as a hallucination signal (**O**, CLM-010). Semantic entropy groups generations by meaning before computing uncertainty (**O**, CLM-011). Semantic-entropy probes approximate that quantity from one generation's hidden states to avoid multi-sample cost. This is frontier work, and its reported gains are scoped to the authors' models and tasks (**O**, CLM-018). These are valuable mechanisms, not oracles. A model can repeat one correlated falsehood with low disagreement. Surface variation can be factual paraphrase. Semantic clustering can merge contradiction or split equivalents. Sampling adds cost and changes with temperature.

Evaluation must use an independent correctness label—not the same detector being assessed—and report discrimination, calibration, risk-coverage, cost, and performance by failure type. Test counterexamples: stable misconceptions, ambiguous questions, multiple correct answers, paraphrases, entity aliases, stale facts, and adversarially persuasive false statements.

**Worked Example (synthetic samples; simplified estimator):**
*Estimator used here.* Sample $m=5$ answers, group them into meaning clusters $c$ with an equivalence judge, and compute the discrete entropy $H=-\sum_c \hat p_c\ln\hat p_c$ in nats, where $\hat p_c$ is the fraction of samples in cluster $c$. This frequency-based estimate is a teaching simplification. Published estimators may weight clusters differently, for example by sequence likelihood, and this toy does not reproduce any paper's estimator.

Symbols: $m$ is the sample count; $c$ ranges over clusters; $\hat p_c=(\text{samples in }c)/m$, so the $\hat p_c$ sum to 1. $H=0$ when all samples fall in one cluster ($1\cdot\ln1=0$); with $m=5$ the maximum is $\ln5=1.609$ nats, reached when all five samples are in different clusters. The symbol $H$ here is unrelated to the **H** (hypothesis) evidence tag.

Case 3, step by step: $\hat p=(0.4,0.4,0.2)$; $-\ln0.4=0.9163$ and $-\ln0.2=1.6094$; $H=2(0.4)(0.9163)+0.2(1.6094)=0.7330+0.3219=1.055$ nats. Case 2: $0.6(0.5108)+0.4(0.9163)=0.3065+0.3665=0.673$.

| case | five samples (paraphrased) | clusters | $H$ (nats) | independent label | what the signal says |
|---|---|---|---|---|---|
| 1. Correlated falsehood | "It opened in 1912", "1912", "The year was 1912", "Opened: 1912", "In 1912" | {1912}×5 | $0$ | all wrong (evidence: 1914) | maximally certain, and wrong |
| 2. Alias split by a bad judge | "11 Nov 1918" ×3, "Armistice Day, 1918" ×2 | judge splits into 3 + 2 | $-0.6\ln0.6-0.4\ln0.4=0.673$ | all correct | "uncertain" although every sample is right |
| 2′. Same samples, audited merge | same | {Nov 1918}×5 | $0$ | all correct | certain and right |
| 3. Genuine ambiguity | "Paris" ×2, "Lyon" ×2, "Nice" ×1 | 2 + 2 + 1 | $1.055$ | question underspecified | high uncertainty is appropriate |

*Result.* Case 1 has zero semantic entropy but is labeled wrong, which falsifies "low semantic entropy implies correct" on this item. Case 2 shows that the entropy value depends on the equivalence judge. The same five correct samples score 0.673 or 0 depending on one merge decision.

*Interpretation and limits.* The detector can be evaluated only against labels that come from evidence outside the sample pool. The cluster audit (merge/split errors) is part of the measurement, not an optional clean-up step. Three cases are counterexamples that falsify universal claims, not estimates of how often these failures occur.

**Knowledge Check:**
1. Why can high surface diversity coexist with one semantic answer?
2. Which label must remain independent of the uncertainty method?

**Guided Practice:**
(a) Five samples are "Canberra" ×3, "Sydney" ×1, "canberra, ACT" ×1, and a string-match judge treats the last one as distinct. Compute $H$ as clustered, then after an audited merge. (b) Cluster repeated answers, audit merge/split errors, and compare likelihood, self-report, surface disagreement, and semantic uncertainty under fixed sample count and decoding.

**Feedback Contract:**
- *Expected Evidence*: (a) As clustered, $(3,1,1)/5$ gives $H=-0.6\ln0.6-2(0.2\ln0.2)=0.950$ nats. After merging, $(4,1)/5$ gives $H=0.500$ nats. The majority answer's correctness still requires an external label. (b) Raw samples, clustering version, cluster audit, independent labels, cost, and failure-type slices.
- *Common Failure*: Treating consensus as correctness, or reporting entropy without the judge version.
- *Diagnostic Hint*: Verify the most stable answer against evidence rather than the sample pool, and re-run the entropy with the audited clusters.
- *Concept to Revisit*: Consistency Illusion.

**Learning Outcome:**
Choose the granularity and signal that match the event, then actively search for consistency illusions.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 7.5 — Conformal Guarantees Without Guarantee Inflation

**Engineering Question:**
What does a finite-sample coverage statement actually promise?

**Concepts & Definitions:**

- **Exchangeability:** the joint distribution of the $n$ calibration examples and the one test example is unchanged by any reordering. I.i.d. sampling implies it. Temporal drift, adaptive selection, or reuse of the test stream for tuning can break it.
- **Nonconformity score $r_i$:** a scalar computed from a *labeled* calibration example, where larger means "harder to cover". It must be computed the same way for the test example.
- **Split-conformal quantile:** with $k=\lceil (n+1)(1-\alpha)\rceil$, set $\hat q$ to the $k$-th smallest calibration score. Under exchangeability, $P(r_{n+1}\le\hat q)\ge 1-\alpha$, marginally over the calibration draw and the test draw (**O**, CLM-016). Symbols: $n$ is the calibration-set size; $\alpha\in(0,1)$ is the allowed miscoverage rate (target coverage $1-\alpha$); $\lceil\cdot\rceil$ rounds up to an integer; $r_{n+1}$ is the score of the new test example; $\hat q$ is the calibrated threshold (a score value, unrelated to the confidence $q_i$ of Lesson 7.2).
- *Why $k=\lceil (n+1)(1-\alpha)\rceil$ (sketch).* Under exchangeability, and ignoring ties, the test score $r_{n+1}$ is equally likely to occupy each of the $n+1$ ranks among $r_1,\dots,r_{n+1}$. The event $r_{n+1}\le\hat q$ (the $k$-th smallest calibration score) happens whenever the test score's rank among all $n+1$ is at most $k$, so its probability is at least $k/(n+1)$. Requiring $k/(n+1)\ge1-\alpha$ gives the smallest valid $k=\lceil (n+1)(1-\alpha)\rceil$. Ties only push more mass onto "$\le$", so the lower bound still holds. Using $\lceil n(1-\alpha)\rceil$ instead gives $k/(n+1)<1-\alpha$ in general, which is why the $+1$ matters. Example: $n=9$, $\alpha=0.2$ gives $k=\lceil8\rceil=8$ and $8/10=0.8\ge0.8$ (the uncorrected $\lceil7.2\rceil=8$ happens to agree). At $n=10$, $\alpha=0.2$ they differ: the correct $k=\lceil8.8\rceil=9$ gives $9/11=0.818\ge0.8$, while the uncorrected $\lceil8.0\rceil=8$ gives only $8/11=0.727<0.8$. The two often coincide, so unit-test the formula rather than a few convenient cases.
- *Reading the bound.* Rearranging $k\le n$ gives $(n+1)(1-\alpha)\le n$, i.e. $\alpha\ge1/(n+1)$; this is where the admissible range $\alpha\in[1/(n+1),1]$ and the rule "$n\ge1/\alpha-1$" in the worked example come from.
- **Conformal factuality (Mohri and Hashimoto):** split an LM output into sub-claims, score each sub-claim, and back off by removing low-scoring claims at a threshold calibrated this way. The covered event is "the reference entails the backed-off output". The authors state a lower bound $\ge1-\alpha$ under exchangeability for $\alpha\in[1/(n+1),1]$ (**O**, CLM-012).

**Executable construction taught in this module — conformal claim filter (D, CLM-017):**

*Inputs.* A frozen decomposer that turns an output into claims, a frozen claim scorer $s$ (higher means more trusted), a frozen label guide, a miscoverage target $\alpha$, and $n$ labeled calibration outputs drawn from the deployment population. No calibration output may appear in the test data.

1. **Label.** For every calibration output $i$, label each claim true or false under the label guide, with independent raters.
2. **Score.** Set $r_i=\max\{s(c): c \text{ in output } i,\ c \text{ false}\}$, or $r_i=-\infty$ if output $i$ has no false claim. So $r_i$ is the score of the most-trusted false claim.
3. **Rank.** Compute $k=\lceil (n+1)(1-\alpha)\rceil$ with exact rational arithmetic; floating-point arithmetic can land just above an integer and bump the ceiling (in IEEE double, $10\times(1-0.7)$ evaluates to $3.0000000000000004$, so a float `ceil` returns 4 instead of 3). If $k\le n$, $\hat q$ is the $k$-th smallest $r_i$, counting tied values with multiplicity. If $k>n$, set $\hat q=+\infty$.
4. **Output rule.** For a new output, keep exactly the claims with $s(c)>\hat q$ (strict inequality), and drop every claim with $s(c)\le\hat q$. If nothing survives, return an abstention or escalation, never the unfiltered output. $\hat q=+\infty$ therefore means "always abstain".
5. **Why it is valid.** Every kept claim is true $\iff$ no false claim has $s>\hat q$ $\iff$ $r_{n+1}\le\hat q$. That last event is exactly the split-conformal event above, so under exchangeability $P(\text{all kept claims true})\ge1-\alpha$.
6. **Ties.** The strict output rule makes the equivalence in step 5 exact when a test claim's score equals $\hat q$: that claim is dropped whether it is true or false. The lower bound does not need tie-free scores. The companion upper bound $1-\alpha+1/(n+1)$ holds only without ties or with randomized tie-breaking (CLM-016), and this module does not claim it.

*Correspondence to the paper (D, CLM-017).* This filter uses the same back-off and quantile idea as conformal factuality, but it is **not** a reproduction of that paper's implementation. The paper uses its own decomposer, scorers, merge step, annotation scheme, and entailment operator. Our covered event, "every kept claim is individually labeled true", matches the paper's event only when entailment of the kept set reduces to per-claim truth. Report results as "conformal claim filter (this module's construction)", not as "conformal factuality".

**Mechanism Explanation:**

Audit every guarantee:

- prediction object and event covered;
- marginal versus conditional/per-group statement;
- calibration sample and exchangeability assumption;
- score and entailment/evidence semantics;
- finite-sample procedure and tie/randomization details;
- what happens after prompt, traffic, temporal, model, or evaluator shift;
- coverage cost: less-specific output, larger sets, or more abstention.

A nominal marginal guarantee does not imply every prompt is correct, all subgroups meet the rate, the answer is complete/useful, or a changed pipeline remains covered. Monitor realized coverage with confidence intervals, but do not “repair” a violation by repeatedly tuning on the test stream without accounting for adaptivity.

**Worked Example (synthetic calibration data):**
*Input.* There are $n=9$ calibration outputs. Claims are listed as (score, label), with 1 = true and 0 = false:

| $i$ | claims | $r_i$ |
|---|---|---|
| 1 | (0.91,1) (0.62,1) (0.40,0) | 0.40 |
| 2 | (0.88,1) (0.71,0) (0.30,1) | 0.71 |
| 3 | (0.95,1) (0.80,1) | $-\infty$ |
| 4 | (0.84,1) (0.52,0) (0.45,0) | 0.52 |
| 5 | (0.93,1) (0.77,1) (0.66,0) | 0.66 |
| 6 | (0.79,0) (0.60,1) | 0.79 |
| 7 | (0.90,1) (0.58,1) (0.35,0) | 0.35 |
| 8 | (0.86,1) (0.66,0) (0.20,1) | 0.66 |
| 9 | (0.97,1) (0.83,1) (0.50,1) | $-\infty$ |

*Steps.* The sorted scores are $-\infty,-\infty,0.35,0.40,0.52,0.66,0.66,0.71,0.79$.

- $\alpha=0.2$: $k=\lceil10\times0.8\rceil=8$, so $\hat q=0.71$.
- $\alpha=0.3$: $k=\lceil 7\rceil=7$, so $\hat q=0.66$. The tied value 0.66 occupies ranks 6 and 7.
- $\alpha=0.05$: $k=\lceil9.5\rceil=10>9$, so $\hat q=+\infty$ and the filter always abstains.

Test output claims: (0.92, true), (0.79, true), (0.66, **false**), (0.55, true).

- With $\hat q=0.71$: keep 0.92 and 0.79. All kept claims are true, and 2 of 4 claims are retained.
- With $\hat q=0.66$: the false claim at *exactly* 0.66 is dropped by the strict rule, so the output again keeps 0.92 and 0.79. A rule of $s\ge\hat q$ would have kept the false claim.

*Result.* The procedure gives an auditable threshold, and the tie example shows why the inequality direction is part of the specification. With only 9 calibration outputs, $\alpha=0.05$ is not achievable except by abstaining. Smaller $\alpha$ requires $n\ge 1/\alpha-1$.

*Simulation check (synthetic, not a benchmark).* We ran 20,000 trials of a generator that draws exchangeable outputs with 1–4 claims, scores rounded to 0.1 (so ties are frequent), and a claim is true with probability $0.3+0.6s$ (seed 7). The realized rate of "all kept claims true" was $0.835$ at $\alpha=0.2, n=9$ and $0.926$ at $\alpha=0.1, n=49$, both above $1-\alpha$. When test labels were redrawn as true with probability 0.3 regardless of score (a post-calibration shift), the rate fell to $0.561$. That breaks the target, as expected once exchangeability no longer holds.

*Interpretation and limits.* The guarantee is marginal over calibration and test draws. It does not say that this particular test output has a 0.8 probability of being fully true, that each language slice attains 0.8, or that the retained 2 of 4 claims are useful or complete. It also inherits every labeling error, because labels define the event.

**Knowledge Check:**
1. What changes when calibration examples are no longer exchangeable with deployment examples?
2. Does larger or less-specific output preserve usefulness automatically?
3. Why is the output rule $s>\hat q$ and not $s\ge\hat q$ under this module's definition of $r_i$?

**Guided Practice:**
(a) Using the nine $r_i$ above, compute $\hat q$ for $\alpha=0.4$, then filter the test claims (0.70, false), (0.52, true), (0.51, false). (b) Implement the filter, then run the required test below on your own labeled calibration data and on a controlled temporal/domain-shifted split.

**Required test (part of Lab D):** (i) unit-test $k$ and $\hat q$ against hand-computed cases, including $k>n$, ties at $\hat q$, and all $r_i=-\infty$; (ii) show that $\hat q$ never increases as $\alpha$ increases; (iii) run a seeded synthetic exchangeable simulation and check that the realized coverage is at least $1-\alpha$ minus Monte Carlo error; (iv) report realized coverage with a binomial interval on held-out labeled data, marginally and by slice, before and after shift.

**Feedback Contract:**
- *Expected Evidence*: (a) $k=\lceil10\times0.6\rceil=6$, so $\hat q=0.66$. The filter keeps (0.70, false) and drops both 0.52 and 0.51, so this output **violates** the event. That is allowed: the guarantee is marginal, and a single miss does not falsify it. (b) Calibration-set lineage, scorer and decomposer versions, $r_i$ table, $k$, $\hat q$, retained fraction, realized coverage with an interval, slice results, and a scope statement.
- *Common Failure*: Using the $\lceil n(1-\alpha)\rceil$ empirical quantile, which drops the $+1$ correction; using $\ge$ at the threshold; tuning $\alpha$ or the scorer on the test stream; calling the result "conformal factuality".
- *Diagnostic Hint*: Write the covered event as an inequality on $r_{n+1}$. If you cannot, the guarantee does not apply to what you built.
- *Concept to Revisit*: Marginal Coverage Semantics; Split-Conformal Quantile.

**Learning Outcome:**
Implement a split-conformal claim filter with an exact score, rank, and output rule, verify it with the required test, and state its guarantee without per-item, subgroup, or post-shift inflation.

*(Effort: 65m instruction, 45m practice)*

---

### Lesson 7.6 — Behavior Regression and Shift Diagnosis

**Engineering Question:**
Did the model change, or did the population, evidence, evaluator, or pipeline change?

**Concepts & Definitions:**

A regression suite combines:

- stable anchor items for paired release comparison;
- time-sensitive items with explicit as-of dates;
- deployment-language, domain, and subgroup slices;
- adversarial and known-severity failures;
- sampled production cases with provenance and privacy controls;
- repeated samples for stochastic behavior;
- calibration, proper score, and risk-coverage outputs—not accuracy alone.

**Mechanism Explanation:**

Version model weights, system/developer/user prompts, decoding, runtime, retrieval corpus/index, tools, evaluator, label policy, and sampling frame. When a slice moves, consider model behavior, prompt interaction, runtime/tokenizer change, retrieval/tool failure, judge drift, label error, contamination, and traffic shift.

Treat regression claims as hypotheses to be tested (**H**, CLM-015); uncertainty signals validated before a release need revalidation after it (**O**, CLM-004). Use paired deltas where the same items are valid across releases; use independent-population methods for live samples. Predeclare practically important margins and high-severity zero/near-zero tolerance rules. Exploratory slices found after seeing results require held-out replication.

**Worked Example:**
Rescore identical old/new outputs with both old and new judges. If only the judge swap changes the metric, model regression is weakened; if matched raw outputs also worsen under blinded adjudication, evaluator drift alone is insufficient.

**Knowledge Check:**
1. When is a paired release comparison invalid?
2. Why must exploratory slice findings be replicated?

**Guided Practice:**
Build a two-by-two model-release/evaluator-release replay and one matched traffic slice. Predeclare severity and practical-effect gates before inspecting deltas.

**Feedback Contract:**
- *Expected Evidence*: Full component manifest, paired outputs where valid, evaluator cross-score, label audit, uncertainty, and temporal/slice results.
- *Common Failure*: Attributing a metric change to weights while prompts, retrieval, or judge also changed.
- *Diagnostic Hint*: Hold all but one versioned component fixed.
- *Concept to Revisit*: Matched Regression Diagnosis.

**Learning Outcome:**
Diagnose behavior changes causally enough to select rollback, recalibration, routing, retrieval, prompt, or model interventions.

*(Effort: 40m instruction, 30m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo et al. (2017): calibration metrics and temperature scaling.
- [Can You Trust Your Model's Uncertainty?](https://arxiv.org/abs/1906.02530) — Ovadia et al. (2019): uncertainty under dataset shift; original domain is image classification.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) — Geifman and El-Yaniv (2017): coverage and selective risk.
- [How Can We Know When Language Models Know?](https://arxiv.org/abs/2012.00955) — Jiang et al. (TACL 2021): generative QA calibration.
- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) — Kadavath et al. (2022): P(True) and P(IK), including transfer limitations.
- [TruthfulQA](https://arxiv.org/abs/2109.07958) — Lin, Hilton, and Evans (ACL 2022): imitative-falsehood slice.
- [FActScore](https://arxiv.org/abs/2305.14251) — Min et al. (EMNLP 2023): atomic-fact factual precision.

**CURRENT DEFAULT**

- Scope of this label: these are the module's *recommended baseline* practices, derived from the cited definitions (CLM-001, CLM-005, CLM-015). They are not a measured survey of industry adoption or of any single vendor's default.
- Explicit event/population definitions, held-out calibration, risk-coverage reporting, slice-based regression, and versioned evaluator provenance are baseline practices.
- No one uncertainty signal, ECE binning, threshold, or “hallucination metric” is a universal default.

**WORKLOAD-DEPENDENT**

- Temperature scaling, verbalized confidence, learned probes, sample disagreement, semantic entropy, tool/retrieval checks, abstention, and escalation.
- The correct operating point depends on costs, severity, user workflow, coverage target, and available fallback capacity.

**FRONTIER**

- [Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2302.09664) — Kuhn, Gal, and Farquhar (ICLR 2023).
- [Language Models with Conformal Factuality Guarantees](https://arxiv.org/abs/2402.10978) — Mohri and Hashimoto (arXiv v1, February 2024). Reading pointers: §3 (entailment sets and conformal setup), Algorithm 1, Theorem 4.1, and Assumption 5.1. Lesson 7.5 teaches a related construction, not their implementation (CLM-012, CLM-017).
- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511) — Angelopoulos and Bates (arXiv v6, December 2022). Reading pointers: §1.1 quantile rule and Theorem 1, with the tie-breaking footnote for the upper bound (CLM-016).
- [Calibrated Language Models Must Hallucinate](https://arxiv.org/abs/2311.14648) — Kalai and Vempala (STOC 2024): theoretical result with explicit fact/calibration scope.
- [Semantic Entropy Probes](https://arxiv.org/abs/2406.15927) — Kossen et al. (arXiv, June 2024): approximate semantic entropy from one generation's hidden states. Read at abstract level; results are author-reported on their models and tasks (CLM-018).
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) — Kalai, Nachum, Vempala, and Zhang (arXiv, September 2025): argues that evaluation grading that penalizes abstention sustains guessing. Read at abstract level; this is an argument, not a deployment measurement (CLM-019).
- `TODO_VERIFY` (CLM-020): the wider 2025–2026 landscape of efficient uncertainty estimators and reasoning-trace or trajectory-based detectors was **not** surveyed in this revision. No specific paper from that landscape is claimed here, and none is a production default.

**LEGACY / INSUFFICIENT WHEN USED ALONE**

- Treating raw token probability as answer correctness probability.
- Reporting accuracy without calibration, or ECE without accuracy/ranking/bin details.
- Reporting selective risk without coverage and abstention cost.
- Treating one fixed benchmark as a complete behavior contract.
- Using an LLM judge or detector score as its own ground truth.

**PRODUCTION SOURCE TRACE**

- Repository: `huggingface/transformers`
- Revision: `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`
- Verified: 2026-09-26 by static inspection; not executed here.
- File: `src/transformers/generation/utils.py`.
- Symbols: `GenerateDecoderOnlyOutput`, `GenerationMixin.generate`, `GenerationMixin.compute_transition_scores`.
- Observed behavior: generation outputs can retain processed per-step scores and raw logits when requested; `compute_transition_scores` stacks/gathers selected-token scores and optionally applies `log_softmax`.
- Generalization (**O**, CLM-014): these are generation-path scores at a pinned upstream revision. The 2026-09-26 static read was not repeated in the 2026-09-30 revision. They do not become calibrated answer-correctness probabilities without an explicit event, aggregation, labels, and validation.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow:

$$
\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}
\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}.
$$

### LAB A — Claim-Level Failure Taxonomy

- **Objective:** annotate long-form outputs under an explicit evidence/time contract and compare response-, claim-, and severity-weighted metrics.
- **Pre-Registered Hypothesis:** claim-level labels will expose mixed and actionable failures hidden by one response-level label on the declared sample.
- **Independent Variables:** prompt family, evidence availability, knowledge date, answer length, domain, and model/prompt revision.
- **Dependent Variables:** atomic-claim precision and required-claim recall, contradiction, stale/unsupported attribution, omission, refusal, instruction compliance, severity, inter-annotator agreement, and adjudication.
- **Break & Falsify:** include ambiguous claims, multiple valid phrasings, changing facts, unanswerable prompts, correct but uncited statements, and retrieved evidence conflicts. Show a case where one response-level label hides mixed claim behavior.
- **Artifact:** versioned annotation guide, raw labels, adjudication log, evaluator error analysis, and intervention mapping.
- **Alignment:** Lesson 7.1.
- **Effort Estimate:** 4h.

### LAB B — Calibration Under Controlled Shift

- **Objective:** construct answer-confidence signals and compare raw, temperature-scaled, self-reported, and optional learned confidence.
- **Pre-Registered Hypothesis:** held-out temperature scaling will improve at least one declared proper score in-distribution but need not preserve that gain under controlled shift.
- **Split:** separate fit, calibration, in-distribution test, and shifted test by the deployment unit; no threshold or temperature tuning on test.
- **Independent Variables:** confidence construction, calibration method, prompt/domain/language/time/model shift, and subgroup.
- **Dependent Variables:** accuracy, Brier, NLL, declared-bin ECE with uncertainty, reliability plot, ranking metric, subgroup calibration, and risk-coverage.
- **Break & Falsify:** shift prompt format, domain, language, time, and model version; change length normalization; find equal-ECE models with different decision utility or an aggregate-calibrated model with subgroup error.
- **Artifact:** confidence semantics, source trace, calibration code/tests, raw predictions, bins/counts, shift matrix, and recalibration trigger.
- **Alignment:** Lesson 7.2.
- **Effort Estimate:** 5h.

### LAB C — Abstention and Consistency Illusions

- **Objective:** compare likelihood, P(True), black-box disagreement, and semantic uncertainty as routing signals.
- **Pre-Registered Hypothesis:** semantic grouping will reduce false disagreement from paraphrases on the chosen slice but correlated falsehoods will remain a counterexample to consistency-based confidence.
- **Independent Variables:** sample count, temperature, equivalence method, threshold, task ambiguity, and fallback action.
- **Dependent Variables:** detector discrimination/calibration, coverage-risk, sample cost, latency, false accept/reject severity, subgroup coverage, fallback success, and escalation load.
- **Break & Falsify:** stable false misconceptions, diverse correct paraphrases, aliases, multiple correct answers, and adversarially confident false claims. Use independent labels, not consensus, as correctness.
- **Artifact:** raw generations, semantic clusters with audits, routing frontier, counterexample set, and selected operating point with rejected alternatives.
- **Alignment:** Lessons 7.3 and 7.4.
- **Effort Estimate:** 5h.

### LAB D — Release Regression and Conformal Scope

- **Objective:** compare two pinned releases on anchor/time/slice suites, then implement the Lesson 7.5 conformal claim filter (required) and compare it with a prespecified abstention gate on the same calibration and test splits.
- **Pre-Registered Hypothesis:** matched slice and evaluator-crossed analysis will expose at least one regression or guarantee-scope failure that aggregate accuracy alone cannot localize.
- **Independent Variables:** model/prompt/evaluator release, temporal/domain shift, retrieval state, and conformal or abstention policy.
- **Dependent Variables:** paired behavior deltas, severity, calibration/proper scores, marginal and subgroup coverage, risk-coverage, temporal validity, evaluator agreement, and system cost.
- **Break & Falsify:** alter exchangeability through temporal/domain shift, change evaluator revision, inject a retrieval outage, and inspect a slice whose regression is hidden by the mean. Verify whether the stated guarantee still applies before interpreting coverage.
- **Required conformal test:** Lesson 7.5 items (i)–(iv). Item (iv) must be run on the in-distribution split and on the temporally or domain-shifted split; the shifted run is expected to be able to fail.
- **Artifact:** release manifest, frozen thresholds/calibration set, regression report, **Conformal Filter Record** (calibration IDs and lineage, decomposer/scorer/label-guide versions, per-output $r_i$, $n$, $\alpha$, $k$, $\hat q$, retained-claim fraction, abstention rate, realized coverage with a binomial interval marginally and by slice, required-test output, and the scope statement), assumption audit, incident decision, and rollback/recalibration criteria.
- **Alignment:** Lessons 7.5 and 7.6.
- **Effort Estimate:** 6h (2h conformal filter and required test, 4h release regression and evaluator crossing).

## 07 Break / Incident Scenarios

### Incident 07.1 — The Aggregate Dashboard Is Green

**Incident Symptoms:**
A new model/prompt release has unchanged overall exact-match accuracy and lower reported ECE. Production complaints show confidently wrong time-sensitive answers in one language, more abstentions for a high-value subgroup, and citations that look plausible but do not support the claims. The automatic judge was also upgraded.

**Diagnostic Protocol (Task):**
The learner must:

1. **Compete hypotheses:** true model regression; changed confidence semantics; prompt/decoding shift; temporal knowledge drift; retrieval/index outage; judge drift; label-policy change; traffic mix; answer-normalization bug; or subgroup threshold mismatch.
2. **Recover evidence:** paired old/new raw outputs, prompt/model/runtime/retrieval/judge versions, claim/evidence records, as-of dates, confidence and bins, coverage-risk by slice, abstention destination, sampled human adjudication, and queue/cost effects.
3. **Discriminate:** rescore both releases with both evaluator versions and an audited sample; replay the same prompts/evidence; freeze retrieval; inspect score distributions; recompute calibration and coverage on matched slices; vary only the suspected component.
4. **Rank causes:** use matched deltas, temporal order, label audit, and uncertainty. Aggregate stability and lower ECE are not exculpatory.
5. **Intervene:** rollback/gate the affected route, repair evidence or evaluator, recalibrate thresholds, refresh time-sensitive sources, or change fallback capacity based on evidence.
6. **Remeasure:** repeat the identical slice, calibration, coverage, severity, citation-support, and operational contract with preregistered promotion criteria.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Build a Trustworthy Answering Gate

Design an uncertainty and regression system for a multilingual assistant answering both stable and time-sensitive factual questions. It can answer, retrieve/check, abstain, or escalate.

**Required Deliverables:**

1. behavior contracts and a non-overlapping-enough failure taxonomy with evidence, time, unit, severity, and ambiguity rules;
2. a representative sampling/split plan and label/adjudication protocol;
3. at least three distinct confidence signals with exact semantics and cost;
4. calibration equations, proper scores, binning rules, uncertainty, ranking, and subgroup analyses;
5. risk-coverage and utility for every fallback, including rejected-user and capacity costs;
6. shift tests across prompt, language, time, domain, retrieval state, and model version;
7. a consistency/semantic-uncertainty counterexample analysis using independent labels;
8. a conformal claim filter for the long-form answer path: $r_i$ definition, $k$ and $\hat q$ at the chosen $\alpha$ with exact arithmetic, strict output rule, abstention rule when nothing survives, required-test results (Lesson 7.5 (i)–(iv)), realized coverage with an interval by slice before and after one shift, and the guarantee copied into system terms with every assumption and non-guarantee;
9. a pinned runtime source trace explaining why generation scores are not correctness probabilities;
10. versioned release gates, canary, rollback, evaluator audit, and post-intervention remeasurement.

## 09 Required Evidence & Rubric

### Required Artifact: Behavior and Uncertainty Record

Each record must join prompt/task and all component revisions, allowed evidence and date, raw/sample output, atomic claims and labels, category/severity/adjudication, every confidence signal and transformation, acceptance/fallback action, evaluator provenance, and observed downstream outcome where lawful and available. The long-form path additionally submits the Conformal Filter Record defined in LAB D.

### Rubric Dimensions

For every dimension, **Insufficient** reports an unlabeled score or aggregate; **Competent** satisfies the stated dimension on a pinned population with held-out evidence; **Strong** adds slice/shift counterexamples, independent adjudication, scoped guarantees, and post-intervention remeasurement.

- **Semantic precision:** events, evidence, time, population, and units are explicit.
- **Calibration discipline:** no score is called probability of correctness without held-out evidence; bins and proper scores are reproducible.
- **Decision quality:** risk, coverage, abstention/escalation utility, subgroup impact, and capacity are co-reported.
- **Adversarial reasoning:** stable falsehoods, paraphrases, ambiguous/multiple answers, shift, and judge failure are tested.
- **Guarantee discipline:** *Insufficient* calls a score or threshold "guaranteed" or omits $k$/$\hat q$ arithmetic. *Competent* implements the Lesson 7.5 filter with exact $k$, a strict output rule and abstention, passes required tests (i)–(iii), and reports marginal realized coverage with an interval. *Strong* also reports slice and post-shift coverage from test (iv), explains any violation through exchangeability rather than by retuning on the test stream, and keeps marginal, conditional, per-item, and post-shift claims distinct.
- **Diagnosis:** competing model/pipeline/evaluator/population causes are separated by matched experiments.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Behavior contract and taxonomy | Lesson 7.1 | 7.1 Guided Practice; LAB A | Mastery deliverables 1–2; Incident 07.1 step 2 | LAB A annotation guide, raw labels, adjudication log; claim/evidence rows in the Behavior and Uncertainty Record |
| Confidence and calibration | Lesson 7.2 | 7.2 Guided Practice (a) hand computation, (b) temperature fit; LAB B | Mastery deliverables 3–4; Incident 07.1 step 3 (recompute calibration on matched slices) | LAB B raw predictions, bin table with edges/counts, Brier/NLL (nats), AUROC, shift matrix |
| Generation-score source trace | Lesson 7.2 (free-form scalar construction); §05 Production Source Trace | LAB B source-trace artifact | Mastery deliverable 9 | Pinned `compute_transition_scores` trace and the documented mapping from score to event |
| Selective prediction | Lesson 7.3 | 7.3 Guided Practice (a) threshold table, (b) frozen sweep; LAB C | Mastery deliverable 5; Incident 07.1 step 5 (threshold/fallback intervention) | LAB C routing frontier with coverage, risk, utility, capacity, rejected-set outcomes |
| Free-form uncertainty | Lesson 7.4 | 7.4 Guided Practice (a) entropy recomputation, (b) cluster audit; LAB C | Mastery deliverables 3, 7 | LAB C semantic clusters with merge/split audit, counterexample set with independent labels |
| Conformal claim filter and guarantee scope | Lesson 7.5 | 7.5 Guided Practice (a) $\hat q$ and filtering, (b) implementation; LAB D required conformal test | Mastery deliverable 8; rubric dimension *Guarantee discipline* | LAB D Conformal Filter Record, including required-test output (i)–(iv) and shifted-split coverage |
| Behavior regression | Lesson 7.6 | 7.6 Guided Practice (2×2 replay); LAB D | Mastery deliverables 6, 10; Incident 07.1 steps 1–6 | LAB D release manifest, evaluator cross-score, paired deltas, remeasurement report |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can:

1. define a failure without relying on the word hallucination;
2. distinguish factuality, truthfulness, correctness, compliance, usefulness, and safety;
3. derive calibration, ECE, Brier/NLL, and risk-coverage with assumptions;
4. show why likelihood and verbal confidence need event-specific calibration;
5. select and defend abstention/escalation thresholds under shift and subgroup constraints;
6. break sample-consistency and semantic-uncertainty methods with realistic counterexamples;
7. implement and test a split-conformal claim filter, then state its guarantee without per-item, subgroup, or post-shift inflation;
8. trace generation-score source code at a pinned revision;
9. diagnose a hidden behavior regression and remeasure after intervention.

### Module Wrap-Up (Final Mental Model Reconstruction)

Model uncertainty is not a property emitted by one number. It is a validated relationship between a score, an event, a population, and a decision. Reliability comes from maintaining that relationship under version and distribution change—and refusing to expand its scope beyond the evidence.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
