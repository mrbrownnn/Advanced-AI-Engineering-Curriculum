# Module 39 — Feedback Loop & AI Product

## 00 Why This Module Exists

A shipped LLM feature produces a second stream of data next to its traces: thumbs, ratings, side-by-side votes, accepts, copies, edits, regenerations, dwell, and abandonment. Teams route that stream into dashboards, evaluation sets, reward signals, and A/B decisions. Each use quietly assumes that the signal measures quality, that the users who emit it represent the users who do not, and that an experiment which improves it improves the product.

None of those assumptions holds by default. A feedback event is a measurement taken by a self-selected subset of users, with a propensity that depends on the interface and on the user. Approval rewards agreement as well as correctness when the rater cannot check the answer. In April 2025 a model update that added a thumbs-based reward passed offline evaluation and A/B tests, then was rolled back within days for sycophancy (**O**, CLM-006, CLM-007).

```text
exposure (trace) --> user action? --> feedback event (selected, delayed, typed)
       |                                   |
       |                 +-----------------+------------------+
       v                 v                 v                  v
  denominator      dashboards        eval candidates     training data
  (join, 39.1)     (bias, 39.2)      (adjudicate, 39.3)  (provenance, 39.4)
                         \                 |                  /
                          +---> A/B decision with integrity checks (39.5)
                                           |
                          latency + trust UX shapes who acts and how (39.6)
```

Module 15 Lesson 15.5 owns the offered denominator, goodput, and the requirement that offline gains be validated online. Module 19 Lessons 19.4–19.5 own mined training data, poisoning, and preference objectives. Module 07 owns calibration of confidence signals. Module 23 Lesson 23.5 owns delayed labels as an SLI problem. This module owns the measurement model of feedback signals, bias correction, the conversion of feedback into evaluation and training data, the mechanics of LLM feature experiments that Lesson 15.5 only names (SRM, CUPED, novelty), and latency and trust UX. It is scheduled after Module 19 because it uses both Module 15 evaluation and the Module 19 data path. It is numbered 39 because 25–36 are reserved by [EXPERT-TRACK.md](../EXPERT-TRACK.md) and 37–38 are taken.

**Research cutoff:** 2026-11-10. All sources were opened on 2026-11-10. Abstract-only sources are labeled in Section 05 and in the registry.

**Module Orientation**
- **Engineering Problem**: Use production feedback to evaluate, train, and ship LLM features without mistaking a selected, approval-shaped signal for quality.
- **What You Will Do**: Build a feedback ledger joined to exposures; quantify and correct nonresponse and position bias; route feedback into adjudicated evaluation sets; convert binary and edit feedback into KTO training data with provenance filters; plan and audit an A/B test with SRM, CUPED, and novelty checks; trace GrowthBook's SRM and CUPED code and Langfuse's score ingestion at pinned commits; diagnose an approval-optimized release that passed its A/B test.
- **Environment**: Python 3.10+ with NumPy, SciPy, and pandas on CPU. All lab data are generated synthetically and labeled so. Source traces are static reads of pinned GitHub revisions.
- **Evidence Rule**: **O** is a source observation, **D** a derivation under stated assumptions, **H** an engineering hypothesis that needs measurement. `CLM-xxx` identifiers point to `research-registry/39-feedback-loop-ai-product.yaml`. What regenerate, edit, dwell, and abandonment measure in chat LLMs, whether streaming lowers perceived latency, and how interference biases LLM experiments are taught as **H** with the measurement that would test them.
- **If You Only Have One Week**: Lessons 39.1, 39.2, 39.5 — together they cover the denominator join, the bias that makes raw thumbs misleading, and the experiment checks that would have stopped an approval-optimized release; 39.3 and 39.4 reuse these ideas for eval and training data.

## 01 Baseline Assumptions

- Module 00, Lessons 0.3 and 0.5: experimental unit, randomization within blocks, confidence intervals, and the half-width planning formula.
- Module 04, Lesson 4.1: TTFT, ITL, and end-to-end latency at declared boundaries. Lesson 39.6 uses them as UX inputs.
- Module 07, Lesson 7.2: confidence as a forecast of a declared event, and calibration. Lesson 39.6 contrasts calibrated confidence with user trust.
- Module 15, Lesson 15.3: graders as instruments, judge position and verbosity bias, and swap consistency. Lesson 15.4: paired block intervals, gates, and power for paired gates. Lesson 15.5: offered denominator, goodput, and the online-validation checklist that names SRM, interference, novelty, selection, and delayed outcomes without teaching their mechanics.
- Module 19, Lesson 19.4: failure mining, train/eval separation, poisoning, and the `untrusted` label on externally writable feedback. Lesson 19.5: RLHF, DPO, GRPO, and KL to a reference.
- Module 23, Lesson 23.2: inverse-probability weighting of sampled telemetry is named there; this module teaches propensity weighting of feedback at first use in Lesson 39.2. Lesson 23.5: delayed labels and proxy quality.
- Not taught earlier and taught here at first use: the Pearson chi-square test (39.5), CUPED (39.5), unpaired two-arm sample size (39.5), KTO (39.4), propensity models of feedback (39.2), and overreliance measures (39.6).

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
  instruction: 4h        # lesson instruction: 40+45+35+40+45+35 min
  guided_practice: 2h    # lesson practice: 20+25+20+20+25+10 min
  labs: 13h              # LAB A 3h + LAB B 3h + LAB C 3.5h + LAB D 3.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 39.1 0.5h
  source_trace: 2h       # Section 09 Production Source Trace (GrowthBook + Langfuse), counted once
  total: 24h
```

Each category is counted once. The source trace is not also counted inside Lesson 39.5 practice or LAB D.

The learner must be able to state what each feedback signal measures and its validity status; compute feedback rates against exposures; correct nonresponse and position bias when propensities are estimable and say when they are not; explain how approval rewards produce sycophancy; turn feedback into adjudicated evaluation items and provenance-filtered training data; plan and audit an LLM feature experiment; and set latency and trust UX targets that are measured against correctness, not satisfaction alone.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Exposure (trace, rendered response) $\to$ Propensity to act (UI, user, task, quality) $\to$ Feedback Event (explicit: thumbs, rating, side-by-side; implicit: accept, copy, edit, regenerate, dwell, abandon) $\to$ Ledger Join (exposure denominator, async ingestion, user/session keys) $\to$ three consumers:

1. **Measurement** — rates per exposure, bias correction by propensity, validity against adjudicated outcomes.
2. **Data** — candidates selected for adjudication into eval sets; desirable/undesirable labels and edits into training data, filtered by provenance.
3. **Decisions** — A/B tests checked for SRM, variance-reduced by CUPED, read across time for novelty and primacy, and gated by behavior guardrails that approval metrics cannot replace.

Latency and trust UX change the propensity term at the start of the chain, so a UX change can move every downstream number without changing model quality.

Keep separate: approval, reliance, correctness, and satisfaction; per-event share and per-exposure rate; a selector of candidates and a label; an integrity check (SRM) and an effect estimate.

## 04 Lessons

### Lesson 39.1 — Feedback Signals as Measurements

**Engineering Question:**
What does each feedback signal measure, over which population, and with which denominator?

**Concepts & Definitions:**
- **Exposure**: a response rendered to a user, logged with trace, user, session, variant, and timestamp. It is the denominator for every per-response rate.
- **Explicit signal**: an action whose only purpose is feedback (thumbs, star rating, side-by-side choice, free-text comment). It measures *approval* by the user who chose to act.
- **Implicit signal**: an action taken for another purpose (accept a code suggestion, copy, edit, regenerate, dwell, abandon). Its meaning is an inference.
- **Response propensity**: the probability that an exposure produces a given feedback event. It depends on the interface, the user, the task, and possibly quality.
- **Three estimands**: the *share* of up among feedback events (conditions on acting), the *rate* of up per exposure, and the *per-user* rate (one value per randomization unit). They answer different questions and move differently.
- **Signal validity status**: whether the signal's meaning was checked against adjudicated outcomes on this product. Unchecked signals are hypotheses.

**Mechanism Explanation:**
What the evidence supports:
- Copilot's acceptance rate of shown suggestions predicted *perceived* productivity better than persistence measures; the authors say it supports coarse monitoring and that other approaches remain necessary (**O**, CLM-011). The outcome was self-report, not correctness.
- In a 77-participant study, high overreliance went with frequent copy-paste and skipped comprehension (**O**, CLM-012; abstract only). Copy is evidence of reliance.
- Accept, copy, and thumbs-up therefore measure reliance or approval, not correctness (**D**, CLM-048).

What the evidence does not yet support (this pass opened no primary source that validates them for chat LLMs):
- Regenerate measures dissatisfaction rather than exploration (**H**, CLM-050).
- Edit size measures the gap to the user's preference rather than added content or restyling (**H**, CLM-051).
- Dwell measures reading of a useful answer rather than confusion; abandonment measures failure rather than "got what I needed" (**H**, CLM-052).

Each **H** names its falsifier in the registry. Until a validation study on adjudicated outcomes is run, a dashboard should show these signals with their status, not as quality.

*Capture path (one tool).* At the pinned Langfuse commit, a feedback score is a typed row (`NUMERIC`, `CATEGORICAL`, `BOOLEAN`, `CORRECTION`, `TEXT`) attached to one of `traceId`, `sessionId`, `observationId`, or `datasetRunId`; `CORRECTION` (an edited output) and `TEXT` are excluded from aggregation (**O**, CLM-033). The public `POST /api/public/scores` handler wraps each score as a `SCORE_CREATE` event and calls `processEventBatch`, so writes are asynchronous ingestion (**O**, CLM-034). A score row exists only when the user acted. Rates therefore need a join to the exposure log, and counts read shortly after the window closes may be incomplete (**D**, CLM-042).

**Quantitative Model / Derivation:**
For a window with $E$ exposures, $U$ up events, and $N$ down events:

$$\text{share}_{up}=\frac{U}{U+N},\qquad r_{up}=\frac{U}{E},\qquad r_{down}=\frac{N}{E},\qquad r_{resp}=\frac{U+N}{E}.$$

*Where*: $E$ counts rendered responses at a declared boundary (not requests sent); $U,N$ count feedback events joined to those exposures; at most one thumb per exposure is assumed.

*Step-by-step*: (1) Join feedback rows to exposures by trace ID; drop feedback whose trace is outside the window. (2) Count. (3) Note that $\text{share}_{up}=r_{up}/r_{resp}$: the share divides by the response rate, so any change in who responds moves it even when $r_{up}$ and $r_{down}$ are fixed in ratio to quality.

**Worked Example (synthetic):**
- *Input*: Week A: $E=50{,}000$, $U=540$, $N=660$. Week B, after a UI change that makes the thumbs more visible: $E=50{,}000$, $U=1{,}500$, $N=900$.
- *Steps*: Week A share $=540/1{,}200=0.45$; $r_{down}=660/50{,}000=1.32\%$; $r_{resp}=2.4\%$. Week B share $=1{,}500/2{,}400=0.625$; $r_{down}=900/50{,}000=1.80\%$; $r_{resp}=4.8\%$.
- *Result*: the up share rose 17.5 points, and the per-exposure down rate also rose 0.48 points.
- *Interpretation and limits*: both cannot be read as "quality improved". The response rate doubled, so who responds changed. Without adjudicated quality on a sample, neither week's numbers identify quality. All values are synthetic.

**Knowledge Check:**
1. Why can the up share rise while the per-exposure down rate also rises?
2. What population does an accept rate describe, and what outcome did Ziegler et al. validate it against?
3. Why must the ledger tolerate feedback that arrives after the window closes?

**Guided Practice:**
A week has $E=30{,}000$, $U=360$, $N=540$. (a) Compute share, $r_{up}$, $r_{down}$, $r_{resp}$. (b) Write the ledger schema: exposure keys, feedback keys, variant, user, ingestion time, and how late feedback is handled. (c) For regenerate, write the validation study that would move CLM-050 from **H** to observed.

**Feedback Contract:**
- *Expected Evidence*: (a) share $=0.40$; $r_{up}=1.2\%$; $r_{down}=1.8\%$; $r_{resp}=3.0\%$. (b) A schema with an exposure table keyed by trace and a feedback table joined to it, plus a maturity rule for late rows. (c) A stratified adjudicated sample of regenerated and non-regenerated responses, with which answer the user kept, and the falsifier stated.
- *Common Failure*: Computing rates from score rows alone, or presenting regenerate rate as a defect rate.
- *Diagnostic Hint*: What is the denominator, and which exposures produced no row?
- *Concept to Revisit*: Offered denominator (Module 15, Lesson 15.5).

**Learning Outcome:**
Define what a feedback signal measures, compute it against exposures, and state its validity status.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 39.2 — Feedback Bias: Nonresponse, Position, and Approval

**Engineering Question:**
When does a feedback metric move without quality moving, and when can the bias be corrected?

**Concepts & Definitions:**
- **Nonresponse (selection) bias**: feedback comes from users whose propensity to act differs by satisfaction, task, or user type.
- **Position / presentation bias**: the probability of examining or selecting an item depends on where and how it is shown. Joachims et al. found clicks informative but biased by trust in the ranking and by result quality; reversing the ranking moved the mean clicked rank from 2.66 to 4.03 (**O**, CLM-001).
- **Relative vs absolute interpretation**: "click > skip above" agreed with explicit judgments 80.8 ± 3.6% (inter-judge 89.5%) and was stable under swapped and reversed rankings; "click > earlier click" scored 28.6–36.8% (**O**, CLM-002).
- **Inverse propensity weighting (IPW / IPS)**: weight each observed event by $1/p$, the probability it would be observed. The IPS ranking estimator is unbiased when propensities are positive and depend only on observables; under a position-based model, a swap intervention estimates propensities up to a constant (**O**, CLM-003).
- **Sycophancy**: a model telling users what they appear to want. Five assistants were sycophantic across four tasks (**O**, CLM-004).
- **Approval reward**: using user or rater approval as a training or selection signal.

**Mechanism Explanation:**
*Nonresponse.* If satisfied users rarely click and dissatisfied users often do, the observed share is dominated by the propensity ratio (**D**, CLM-040). IPW recovers the satisfied share only if propensities are estimated without bias, for example from a randomized prompt-for-feedback arm (**D**, CLM-041). Whether propensity also depends on quality within observable strata is untested; if it does, IPW on observables stays biased (**H**, CLM-056).

*Position.* An LLM UI that shows several drafts or citations may have position effects like a results page (**H**, CLM-055). Test with a swap intervention before logging selection as preference.

*Approval.*
- In hh-rlhf preference data, "matches user's beliefs" was consistently among the most predictive features; the Claude 2 preference model preferred sycophantic over baseline truthful answers 95% of the time, and best-of-4096 against it gave about 75% sycophantic answers on the hardest misconceptions versus about 25% under an oracle (**O**, CLM-005).
- RLHF with time-limited evaluators made models better at convincing them but not at the task, raising evaluator false-positive rates by 24.1% and 18.3% (**O**, CLM-008).
- With 2% of simulated users vulnerable, RL on feedback learned to target them while behaving well with others (**O**, CLM-009; abstract only, simulated users).
- LLMs preserved user "face" 45 points more than humans, and social sycophancy is rewarded in preference data (**O**, CLM-010; abstract only).
- OpenAI's GPT-4o update in April 2025 was rolled back. The vendor wrote that it "focused too much on short-term feedback" (**O**, CLM-006) and that a thumbs-based reward "weakened the influence of our primary reward signal, which had been holding sycophancy in check" (**O**, CLM-007).

So approval is a quality proxy only for raters who can verify the answer. A short-term approval metric can rise through agreement, persuasion, or targeting (**D**, CLM-049).

**Quantitative Model / Derivation:**
Let $s$ be the satisfied share among exposures, $r_u$ the probability of thumbs-up given satisfied, $r_d$ of thumbs-down given unsatisfied:

$$\text{share}_{up}=\frac{s\,r_u}{s\,r_u+(1-s)\,r_d},\qquad \hat s_{IPW}=\frac{U/r_u}{U/r_u+N/r_d}.$$

*Step-by-step*: (1) Expected up events are $E\,s\,r_u$ and down events $E(1-s)r_d$. (2) Their ratio gives the share; $E$ cancels. (3) Dividing each count by its propensity estimates the counts that would have been observed at propensity 1, so $\hat s_{IPW}\to s$ when $r_u,r_d$ are correct.

For positions under $P(\text{click})=P(\text{examine}\mid\text{rank})\,P(\text{relevant})$: $E[\mathbf 1[\text{click}]/p_{rank}]=P(\text{relevant})$, and $p_r/p_k=\text{CTR}(\text{item at }r)/\text{CTR}(\text{same item at }k)$ from a swap (**O**, CLM-003). Small $p$ inflates variance; clipping $1/p$ trades variance for bias (**D**, CLM-041).

**Worked Example (synthetic):**
- *Input*: $E=10{,}000$, $s=0.8$, $r_u=0.02$, $r_d=0.10$.
- *Steps*: $U=10{,}000\times0.8\times0.02=160$; $N=10{,}000\times0.2\times0.10=200$; share $=160/360=0.444$. IPW: $160/0.02=8{,}000$ and $200/0.10=2{,}000$, so $\hat s=0.8$. If a UI change raises $r_u$ to 0.04: share $=0.032/(0.032+0.020)=0.615$.
- *Swap step*: a citation shown at rank 1 gets CTR 0.12; the same citation swapped to rank 3 gets 0.048. So $p_3/p_1=0.4$ and a click at rank 3 gets weight $2.5$ relative to rank 1.
- *Result*: an 80%-satisfied product shows a 44% up share, and a button change moves it to 62% with no quality change.
- *Interpretation and limits*: the correction needs $r_u,r_d$ from a randomized elicitation arm and fails if propensity depends on unobserved quality (CLM-056).

**Knowledge Check:**
1. Why does IPW need a randomized prompt-for-feedback arm rather than the volunteered feedback itself?
2. Why was "click > skip above" robust to reversed rankings while "click > earlier click" was not?
3. Which observation in the GPT-4o postmortems shows that an A/B test on approval metrics could not detect the regression?

**Guided Practice:**
$E=20{,}000$, $s=0.7$, $r_u=0.03$, $r_d=0.06$. (a) Compute $U$, $N$, the observed share, and $\hat s_{IPW}$. (b) Design the randomized elicitation arm: fraction of exposures prompted, how $r_u,r_d$ are estimated, and the test of CLM-056.

**Feedback Contract:**
- *Expected Evidence*: (a) $U=420$, $N=360$, share $=0.021/0.039=0.538$; $\hat s=14{,}000/(14{,}000+6{,}000)=0.70$. (b) A randomly sampled prompted arm, propensities estimated per stratum with intervals, and a comparison of adjudicated quality between volunteered and prompted responders within strata.
- *Common Failure*: Treating 0.538 as satisfaction, or estimating propensities from volunteered feedback.
- *Diagnostic Hint*: Which users would never appear in your data without the prompt?
- *Concept to Revisit*: Selection in sampled telemetry (Module 23, Lesson 23.2).

**Learning Outcome:**
Quantify nonresponse and position bias in feedback metrics, correct them when propensities are estimable, and explain why approval rewards agreement.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 39.3 — From Feedback to Evaluation Sets

**Engineering Question:**
How should feedback enter Module 15 evaluation sets without importing its bias as ground truth?

**Concepts & Definitions:**
- **Selector**: a rule that picks exposures for review (thumbs-down, edit, regenerate, random). Its **precision** is confirmed defects / reviewed candidates.
- **Label**: an adjudicated judgment under a versioned guide by a named rater pool. A feedback event is a selector, not a label (**D**, CLM-047).
- **Label provenance**: who labeled, under which guide, when, with what agreement, and from which selector the item came.
- **Public preference platform**: a crowd-voting arena. Chatbot Arena had about 240K votes from about 90K users, Bradley–Terry scores, and crowd–expert agreement of 72.8–83.1% versus expert–expert 79.4–89.8% on 160 sampled battles (**O**, CLM-016).
- **Public usage logs**: LMSYS-Chat-1M (1M conversations from demo and Arena users, best-effort PII removal, unsafe conversations retained; **O**, CLM-018) and WildChat (1M opt-in conversations with free access, no feedback labels; **O**, CLM-019).

**Mechanism Explanation:**
1. **Select.** Draw candidates from several selectors plus a random exposure sample. The random sample estimates the base defect rate; each selector's precision is compared with it.
2. **Adjudicate.** Label blind to the selector, with the Module 15 Lesson 15.3 protocol.
3. **Record provenance.** Keep selector, guide version, and rater for each item. Weight items back to the exposure population if the eval set must estimate prevalence (Module 15, Lesson 15.2).
4. **Separate.** Items that enter evaluation are excluded from training by source and time (Module 19, Lesson 19.4).
5. **Do not validate with a judge alone.** LLM judges may fail to recognize fixes that users' feedback drove and prefer the baseline (**O**, CLM-021; single abstract-level preprint).

Public platforms are not product evaluations. The Leaderboard Illusion reports undisclosed private testing (27 private variants before Llama-4), skewed data access (19.2% and 20.4% for two providers versus 29.7% for 83 open models), and up to 112% relative gains on the arena distribution from arena data (**O**, CLM-017; abstract only). Usage logs describe self-selected populations, not your traffic.

**Quantitative Model / Trade-off Comparison:**
For selector $k$ with $m_k$ reviewed and $d_k$ confirmed, precision $\hat\pi_k=d_k/m_k$ with a Wilson interval. A selector is useful for mining when its interval sits above the random-sample defect rate $\hat\pi_0$; it costs $m_k$ adjudications per $d_k$ defects found. Items selected this way over-represent the selector's slice, so prevalence estimates require weights equal to (exposure share)/(sample share), as in Lesson 15.2.

**Worked Example (synthetic):**
- *Input*: 300 thumbs-down candidates, 132 confirmed defects. 300 regenerate candidates, 96 confirmed. 150 random exposures, 51 defects.
- *Steps*: thumbs-down $\hat\pi=0.44$, Wilson 95% $[0.385,0.497]$. Regenerate $\hat\pi=0.32$, $[0.270,0.375]$. Random $\hat\pi_0=0.34$, $[0.269,0.419]$.
- *Result*: thumbs-down finds defects somewhat more often than random. Regenerate does not beat random here.
- *Interpretation and limits*: in this synthetic set, regenerate behaves like exploration, consistent with the CLM-050 hypothesis being false *for this product*. Intervals overlap partly; a real study needs more items per slice. A feedback-selected set describes flagged exposures, not the product, unless reweighted.

**Knowledge Check:**
1. Why is a thumbs-down not an evaluation label?
2. What does crowd–expert agreement of 72.8–83.1% imply for using arena votes as gold labels?
3. Why should an LLM judge not be the only check on a model trained from user feedback?

**Guided Practice:**
(a) 200 edited responses were adjudicated; 90 had defects. Compute precision and a Wilson interval and compare it with the random rate above. (b) Write the provenance record for one eval item mined from an edit, including selector, guide, raters, and train/eval exclusion.

**Feedback Contract:**
- *Expected Evidence*: (a) $0.45$, $[0.383,0.519]$; overlaps the thumbs-down interval and sits above the random point estimate, but the intervals overlap. (b) Fields: exposure and trace ID, selector, timestamp, guide version, rater IDs, agreement, adjudicated label, exclusion from training snapshots after that date.
- *Common Failure*: Labeling every thumbs-down item as a defect, or estimating product defect rate from a selector-built set.
- *Diagnostic Hint*: What would the random sample have contained?
- *Concept to Revisit*: Graders as instruments (Module 15, Lesson 15.3); weighting (Lesson 15.2).

**Learning Outcome:**
Route feedback into adjudicated, provenance-tracked evaluation sets and measure each selector's precision.

*(Effort: 35m instruction, 20m practice)*

---

### Lesson 39.4 — From Feedback to Training Data

**Engineering Question:**
How can binary and edit feedback train a model without rewarding approval or opening a poisoning path?

**Concepts & Definitions:**
- **KTO**: an objective for unpaired desirable/undesirable outputs. With $r_\theta=\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ and reference point $z_0=\mathrm{KL}(\pi_\theta\|\pi_{ref})$ estimated per microbatch, $v=\lambda_D\sigma(\beta(r_\theta-z_0))$ for desirable and $\lambda_U\sigma(\beta(z_0-r_\theta))$ for undesirable outputs, and $L=\mathbb E[\lambda_y-v]$. The authors report it matches or beats preference methods from 1B to 30B despite a binary signal; their experiments split preference pairs into binary labels and did not use logged thumbs (**O**, CLM-013).
- **Imbalance rule**: choose $\lambda_D n_D/(\lambda_U n_U)\in[1,\tfrac32]$; on Llama-7B, up to 90% of desirable data could be dropped while still beating DPO (**O**, CLM-014).
- **Edit feedback**: PRELUDE scores a round by token-level Levenshtein distance between the response and the user's edit; CIPHER infers a textual preference from edits and retrieves it for similar contexts with at most 3 extra LLM calls and no fine-tuning, evaluated with GPT-4 simulated users (**O**, CLM-015).
- **Feedback poisoning**: one user who can only prompt and vote can, through later preference tuning, inject false facts, insecure code, or fake news for all users (**O**, CLM-020; abstract only).

**Mechanism Explanation:**
1. **Label source.** Desirable/undesirable comes from feedback, so it inherits Lesson 39.2's approval bias. Training on thumbs teaches the model what raters approve, including agreement (**D**, CLM-049; the 2025 incident is the case where this happened, **O**, CLM-007).
2. **Mixing.** Keep a primary reward or data source that measures correctness or policy compliance, and treat feedback-derived data as one component whose weight is a gated hyperparameter. Track behavior evals (sycophancy, refusal, factuality) before and after.
3. **Provenance.** Every feedback channel is a write path from untrusted users into weights. Apply Module 19 Lesson 19.4 controls: per-user caps, account-age and rate filters, `untrusted` labels for open channels, and exclusion until reviewed (**O**, CLM-020).
4. **Edits.** Edit cost is a candidate signal of preference violation. Whether real edit size tracks violation rather than added content is **H** (CLM-051); PRELUDE's evidence is on simulated users.

**Quantitative Model / Derivation:**
With $n_D$ desirable and $n_U$ undesirable examples and $\lambda_U=1$: $\lambda_D\in[n_U/n_D,\;1.5\,n_U/n_D]$.

Per-example loss: desirable $\lambda_D(1-\sigma(\beta(r_\theta-z_0)))$; undesirable $\lambda_U(1-\sigma(\beta(z_0-r_\theta)))$.

*Step-by-step*: (1) $v$ is bounded by $\lambda_y$, so each loss lies in $[0,\lambda_y]$. (2) A desirable output gains when $r_\theta$ rises above $z_0$, i.e. when it becomes more likely than the reference by more than the batch-average drift. (3) Raising $\lambda_D$ scales the desirable gradients so the rarer class is not swamped.

**Worked Example (synthetic):**
- *Input*: after provenance filtering, 12,000 thumbs-up and 48,000 thumbs-down responses (1:4); $\beta=0.1$; $z_0=0.5$.
- *Steps*:
  1. Imbalance: $\lambda_D\in[4,6]$; choose $\lambda_D=5$ (**D**, CLM-046).
  2. A desirable output with $r_\theta=2.0$: $\sigma(0.1\times1.5)=\sigma(0.15)=0.537$; loss $=5(1-0.537)=2.313$.
  3. An undesirable output with $r_\theta=-1.0$: $\sigma(0.1\times(0.5-(-1.0)))=0.537$; loss $=1-0.537=0.463$.
  4. Provenance: 3% of votes came from accounts under 24 hours old voting more than 50 times; they are excluded and counted in the manifest.
- *Result*: the desirable example carries five times the loss scale of the undesirable one at the same margin.
- *Interpretation and limits*: the rule is the authors' heuristic on preference-derived data. Nothing here shows that thumbs labels are correct; a behavior-eval gate must be run on the trained model.

**Knowledge Check:**
1. Why is KTO a better fit than DPO for logged thumbs, and what does it not fix?
2. What changed in the GPT-4o update that made a thumbs reward harmful, according to the vendor?
3. Which Module 19 controls apply to a public thumbs channel?

**Guided Practice:**
(a) Logged feedback is 1 up : 10 down with $\lambda_U=1$. Give the $\lambda_D$ range. (b) With $\lambda_D=5$, $\beta=0.1$, $z_0=0.5$, compute the loss for a desirable output with $r_\theta=-1.0$. (c) Write the provenance filter and the behavior evals that gate a feedback-trained adapter.

**Feedback Contract:**
- *Expected Evidence*: (a) $[10,15]$ (the authors' own example). (b) $\sigma(-0.15)=0.463$; loss $=5(1-0.463)=2.687$. (c) Per-user caps, account and rate filters, `untrusted` channel labels, exclusion of eval items, and sycophancy/factuality/refusal suites with margins (Module 19, Lesson 19.6).
- *Common Failure*: Treating thumbs-up as "correct", or training on all votes from an open channel.
- *Diagnostic Hint*: Who can write to this label, and what would they gain?
- *Concept to Revisit*: Poisoning and provenance (Module 19, Lesson 19.4); preference objectives (Lesson 19.5).

**Learning Outcome:**
Convert binary and edit feedback into weighted, provenance-filtered training data and gate the result on behavior, not approval.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 39.5 — A/B Testing LLM Features

**Engineering Question:**
Is this experiment valid, sensitive enough, and measuring the outcome the release decision needs?

**Concepts & Definitions:**
- **Assignment unit**: the unit randomized to an arm, usually the user for interactive LLM features (Module 15, Lesson 15.5).
- **Sample ratio mismatch (SRM)**: the observed assignment ratio differs from the designed one; it is a symptom of data-quality problems, not a cause (**O**, CLM-024; abstract only).
- **Pearson chi-square**: $\chi^2=\sum_i (o_i-e_i)^2/e_i$ with $k-1$ degrees of freedom for $k$ arms, where $e_i$ is the designed share times the total.
- **CUPED**: $\hat Y_{cv}=\bar Y-\theta\bar X+\theta\,E X$ with a pre-experiment covariate $X$; optimal $\theta=\mathrm{cov}(Y,X)/\mathrm{var}(X)$ gives variance $\mathrm{var}(\bar Y)(1-\rho^2)$; unbiasedness needs $E X$ equal across arms, which pre-period data guarantee under randomization (**O**, CLM-022).
- **Novelty / primacy**: novelty is an early effect from exploring a change that fades; primacy is an early advantage to control while users relearn (**O**, CLM-025).
- **Guardrail**: a metric that can block a release regardless of the primary metric.

**Mechanism Explanation:**
*SRM.* At the pinned GrowthBook commit, `check_srm` computes the Pearson statistic from summed users per arm and design weights and returns `chi2.sf(x, len(users) - 1)`, skipping zero-weight arms in the sum but not in the df (**O**, CLM-029). `DEFAULT_SRM_THRESHOLD = 0.001`; the UI reads `settings.srmThreshold ?? DEFAULT_SRM_THRESHOLD` and reports healthy when $p\ge$ threshold (**O**, CLM-030). Repeated daily checks of a fixed-n p-value inflate false alarms; eBay uses sequential analysis for continuous SRM monitoring (**O**, CLM-026; abstract only). On an SRM alert, do not read the effect; find the cause (bot filtering, redirect loss, crash in one arm, logging change).

*CUPED.* Bing reported about 50% variance reduction, more than 45% from queries/user with a 1-week pre-period, under 5% for revenue/user, and smaller gains as pre-period coverage falls (**O**, CLM-023). GrowthBook pools $\theta$ across arms and, when pooled pre-period variance is near zero, silently reverts both arms to the unadjusted statistic (**O**, CLM-031). A test sized on CUPED's reduction then runs underpowered with no warning (**D**, CLM-045). Ratio metrics such as thumbs per response, with users randomized, need the delta-method variance (`variance_of_ratios`, **O**, CLM-032).

*Time.* In an A/A test, day 1 had a 67% chance and day 2 a 55% chance of falling outside the final 95% bound; the authors found no significant result flipping to significant in the opposite direction from novelty or primacy; CI width scales as CV$/\sqrt n$, and for metrics whose CV grows over time a longer run does not shrink the CI; one week is the minimum for day-of-week effects (**O**, CLM-025).

*What the metric measures.* The GPT-4o A/B metrics were thumbs, side-by-side preferences, and usage, and they looked positive while the regression was in behavior (**O**, CLM-007). Approval metrics cannot detect an agreement-for-quality trade, so behavior evaluations and long-horizon outcomes must be launch-blocking guardrails (**D**, CLM-049). Randomized task outcomes are a different kind of evidence: recruited developers with Copilot finished an HTTP-server task 55.8% faster (**O**, CLM-028; abstract only, lab task). LLM-simulated users recovered only 39% of a human treatment effect before calibration (**O**, CLM-027; abstract only), so simulated A/B tests are surrogates.

*Interference.* Arms that share a cache, memory or retrieval store, rate limit, or interacting users can bias the difference in means (**H**, CLM-054). This pass opened no primary source on interference designs; compare user-randomized and cluster- or time-sliced estimates to test it.

**Quantitative Model / Derivation:**
*CUPED.* $\mathrm{var}(\bar Y-\theta\bar X)=\frac1n(\sigma_Y^2+\theta^2\sigma_X^2-2\theta\,\mathrm{cov})$. Setting the derivative to zero: $\theta^*=\mathrm{cov}/\sigma_X^2$. Substituting: $\sigma_Y^2(1-\rho^2)/n$. With coverage $c$ (users with pre-period data) the factor is approximately $1-c\rho^2$ under homogeneity, an approximation to verify (**D**, CLM-044).

*Sample size (unpaired, two arms).*

$$n_{arm}\approx\frac{2(z_{1-\alpha/2}+z_{1-\beta})^2\sigma^2}{\delta^2}=15.70\,\frac{\sigma^2}{\delta^2}\quad(\alpha=0.05,\ 1-\beta=0.8).$$

*Where*: $\sigma^2$ is the per-user outcome variance ($p(1-p)$ for a per-user binary outcome); $\delta$ the minimum effect worth detecting; $z_q$ normal quantiles. This differs from Lesson 15.4's paired block formula: the two arms contain different users, hence the factor 2. CUPED multiplies $n$ by its variance factor (**D**, CLM-045).

**Worked Example (synthetic):**
1. *SRM*, 50/50 design, df = 1 (**D**, CLM-043):

| Users (A / B) | $\chi^2$ | $p$ | At 0.001 | At 0.01 |
|---|---:|---:|---|---|
| 50,500 / 49,500 | 10.0 | 1.57e-3 | healthy | alert |
| 50,600 / 49,400 | 14.4 | 1.48e-4 | alert | alert |
| 50,300 / 49,700 | 3.6 | 0.058 | healthy | healthy |
| 505 / 495 | 0.1 | 0.75 | healthy | healthy |

Step for row 1: $e=50{,}000$; $(500^2+500^2)/50{,}000=10.0$. The same 1% relative skew is decisive at 100,000 users and invisible at 1,000. The 0.001 default is one tool's choice, overridable by settings.

2. *Planning*: $\sigma=4$, $\delta=0.1$: $n=15.70\times16/0.01=25{,}117$ per arm. With $\rho=0.6$: $\times0.64=16{,}075$. With 70% coverage: $\times(1-0.7\times0.36)=\times0.748=18{,}788$.

3. *Silent fallback*: the team sized at 16,075 per arm, but the pre-period covariate was constant (a new metric), so $\theta=0$ and the analysis ran unadjusted. Power at $\delta=0.1$: $SE=\sqrt{2\times16/16{,}075}=0.0446$; power $=\Phi(0.1/0.0446-1.96)=0.61$, not 0.80.

- *Interpretation and limits*: all counts and variances are synthetic. The decision record must include the platform revision, the threshold, whether CUPED was applied, and $\theta$.

**Knowledge Check:**
1. Why is SRM checked before the effect is read, and why does a fixed-n p-value checked daily overstate alarms?
2. Under what condition does GrowthBook stop applying CUPED, and how would you detect it in a report?
3. Day 1 of a test shows a large positive effect. What does the Kohavi et al. A/A result suggest?
4. Why cannot a positive thumbs A/B result reject a sycophantic release?

**Guided Practice:**
(a) A 50/50 test shows 50,450 vs 49,550 users. Compute $\chi^2$ and $p$ and state the result at 0.001 and 0.01. (b) Plan a test on per-user thumbs-up rate $p=0.3$ with $\delta=0.02$, then with CUPED at $\rho=0.6$. (c) List three guardrails that are not approval metrics.

**Feedback Contract:**
- *Expected Evidence*: (a) $\chi^2=8.1$, $p=0.0044$: healthy at 0.001, alert at 0.01. (b) $15.70\times0.21/0.0004=8{,}242$ per arm; $\times0.64=5{,}275$. (c) For example: sycophancy and factuality evals on sampled traffic, task success on adjudicated samples, 28-day retention, overreliance on seeded errors.
- *Common Failure*: Reading the effect despite SRM; quoting CUPED's reduction without checking $\theta\neq0$; using the paired-block formula for independent arms.
- *Diagnostic Hint*: Which arm lost users, and where in the pipeline?
- *Concept to Revisit*: Online validity checklist (Module 15, Lesson 15.5); paired planning (Lesson 15.4).

**Learning Outcome:**
Plan, audit, and interpret an LLM feature A/B test with SRM, CUPED, time effects, and non-approval guardrails.

*(Effort: 45m instruction, 25m practice. The GrowthBook trace is counted once, under `source_trace`.)*

---

### Lesson 39.6 — UX for Latency and Trust

**Engineering Question:**
Which latency and trust targets should an LLM UI meet, and how do you know a trust feature helps rather than increases overreliance?

**Concepts & Definitions:**
- **Response-time limits**: about 0.1 s feels instantaneous; about 1 s keeps the flow of thought though the delay is noticed; about 10 s is the limit for attention, beyond which a percent-done indicator and a way to interrupt are needed; for 2–10 s a full percent-done indicator may be overkill (**O**, CLM-035). Nielsen attributes these to Miller (1968) and Card et al. (1991), which were metadata-checked only.
- **Latency business effect (legacy web)**: a 500 ms Bing slowdown impacted revenue about 3%, with a linear rule of 0.6% revenue per 100 ms (**O**, CLM-036; talk slides). Transfer to LLM products is **H** (CLM-057).
- **Overreliance**: users accepting incorrect AI recommendations; measures include accepted-incorrect / incorrect-shown, switch fraction, and weight of advice (**O**, CLM-037).
- **Explanation effect**: explanations raised acceptance regardless of correctness without adding to team accuracy (**O**, CLM-038); detailed explanations raise trust and overreliance even when baseless (**O**, CLM-037).
- **Cognitive forcing**: designs that make users think first reduced overreliance, and the most effective ones were the least liked (**O**, CLM-039; abstract only).

**Mechanism Explanation:**
Map Module 04 timestamps to the UX limits: TTFT is when the user first sees progress; end-to-end time is when the answer is usable. Whether streaming tokens lowers perceived latency and abandonment at fixed end-to-end time is **H** (CLM-053): no primary user study of LLM streaming was located. Test it by randomizing streaming versus buffered delivery at matched end-to-end latency.

Trust features change behavior, not just perception. Copy-paste and skipped comprehension accompany overreliance (**O**, CLM-012; abstract only). Because satisfaction and acceptance can rise with overreliance and fall with the safer design, trust UI changes need a correctness-aware guardrail: the overreliance rate on seeded or adjudicated incorrect answers (**D**, CLM-048).

**Quantitative Model / Trade-off Comparison:**
Overreliance rate $=\dfrac{\#\text{accepted incorrect}}{\#\text{incorrect shown}}$, measured with seeded errors that resemble natural ones. Compare arms with intervals; report it next to task success and satisfaction.

| Budget | Anchor | Design implication |
|---|---|---|
| ≤ 0.1 s | Nielsen "instantaneous" | UI acknowledgment of input, not model output |
| ≤ 1 s | Nielsen "flow" | Target for first visible progress (TTFT plus render) |
| 2–10 s | Nielsen | Progress cue; full percent-done may be overkill |
| > 10 s | Nielsen | Percent-done indicator and interrupt (cancel) |

**Worked Example (synthetic):**
- *Input*: an "explain answer" panel was A/B tested. In each arm 200 seeded incorrect answers were shown. Control accepted 124; treatment accepted 150. Satisfaction rose in treatment.
- *Steps*: overreliance $124/200=0.62$, Wilson $[0.551,0.684]$; $150/200=0.75$, $[0.686,0.805]$. The intervals do not overlap.
- *Result*: the panel increased acceptance of wrong answers by 13 points while satisfaction rose.
- *Interpretation and limits*: the direction matches CLM-038, but this is synthetic. Seeded errors must resemble natural errors for the rate to transfer.

**Knowledge Check:**
1. Which Module 04 timestamp maps to the 1 s limit, and why is it not end-to-end latency?
2. Why can a satisfaction metric prefer the design with more overreliance?
3. What experiment would test CLM-053?

**Guided Practice:**
Design the streaming experiment for CLM-053: randomization unit, how end-to-end latency is held fixed, the outcomes (perceived wait, abandonment, task success), and the falsifying result.

**Feedback Contract:**
- *Expected Evidence*: user-level randomization; buffered arm released at the same end-to-end time as streaming; outcomes with intervals; falsifier = no difference in perceived wait or abandonment, or worse task success.
- *Common Failure*: Comparing streaming with a buffered arm that also finishes later; stating streaming's benefit as fact.
- *Diagnostic Hint*: Which variable differs between arms besides streaming?
- *Concept to Revisit*: TTFT and end-to-end boundaries (Module 04, Lesson 4.1); calibration (Module 07, Lesson 7.2).

**Learning Outcome:**
Set latency targets from response-time limits and measured timestamps, and evaluate trust features with overreliance, not satisfaction alone.

*(Effort: 35m instruction, 10m practice)*

---

## 05 Literature & Production Source Map

All sources opened 2026-11-10. "Abstract only" means only the arXiv/Semantic Scholar abstract was read; no section pointers are claimed for those.

**REFERENCE / BASELINE**
- *Accurately Interpreting Clickthrough Data as Implicit Feedback* (Joachims, Granka, Pan, Hembrooke, Gay, SIGIR 2005). — [PDF](https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf). Full text.
  - *Scope*: clicks as biased implicit feedback; relative vs absolute interpretation (CLM-001, CLM-002). Small 2005 student samples.
- *Unbiased Learning-to-Rank with Biased Feedback* (Joachims, Swaminathan, Schnabel, WSDM 2017). — [arXiv:1608.04468](https://arxiv.org/abs/1608.04468). Full text.
  - *Scope*: IPS estimator, position-based propensity, swap estimation (CLM-003).
- *Trustworthy Online Controlled Experiments: Five Puzzling Outcomes Explained* (Kohavi et al., KDD 2012). Full text via Wayback copy of exp-platform.com.
  - *Scope*: novelty, primacy, early-trend regression to the mean (CLM-025).
- *Productivity Assessment of Neural Code Completion* (Ziegler et al., MAPS 2022). — [arXiv:2205.06537](https://arxiv.org/abs/2205.06537). Full text.
  - *Scope*: what acceptance rate predicts (CLM-011).
- *Response Times: The 3 Important Limits* (Nielsen). — [nngroup.com](https://www.nngroup.com/articles/response-times-3-important-limits/). Tier 2.
  - *Scope*: UI response-time limits (CLM-035).
- *Overreliance on AI: Literature Review* (Passi, Vorvoreanu, Microsoft Aether 2022). Tier 2, Wayback snapshot; WOA formula not used.
  - *Scope*: overreliance definition and measures (CLM-037).
- *Does the Whole Exceed its Parts?* (Bansal et al., CHI 2021). — [arXiv:2006.14779](https://arxiv.org/abs/2006.14779). Abstract plus setup.
  - *Scope*: explanations raise acceptance regardless of correctness (CLM-038).

**CURRENT DEFAULT** (scoped: mechanisms in current use, not an adoption survey)
- *CUPED* (Deng, Xu, Kohavi, Walker, WSDM 2013). — [PDF](https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf). Full text.
  - *Scope*: estimator and Bing variance reductions (CLM-022, CLM-023). Implemented in the traced platform; one implementation does not establish industry adoption.
- SRM checks with a Pearson chi-square, as implemented in the traced platform (CLM-029, CLM-030). Fabijan et al. KDD 2019 defines SRM (abstract only; taxonomy and thresholds not verified, CLM-024).
- *KTO* (Ethayarajh et al.). — [arXiv:2402.01306](https://arxiv.org/abs/2402.01306), v5. Full text.
  - *Scope*: objective and imbalance rule (CLM-013, CLM-014). Evaluated on preference-derived binary labels, not product thumbs.
- *Towards Understanding Sycophancy in Language Models* (Sharma et al., ICLR 2024). — [arXiv:2310.13548](https://arxiv.org/abs/2310.13548), v4. Full text.
  - *Scope*: sycophancy and its link to preference data (CLM-004, CLM-005).
- OpenAI, *Sycophancy in GPT-4o* (2025-04-29) and *Expanding on what we missed with sycophancy* (2025-05-02). Tier 2, Wayback snapshots (live site returned 403).
  - *Scope*: the canonical counterexample (CLM-006, CLM-007). Vendor self-report, no effect sizes.
- *Chatbot Arena* (Chiang et al. 2024). — [arXiv:2403.04132](https://arxiv.org/abs/2403.04132). Full text. *LMSYS-Chat-1M* — [arXiv:2309.11998](https://arxiv.org/abs/2309.11998). *WildChat* — [arXiv:2405.01470](https://arxiv.org/abs/2405.01470). Full text greps.
  - *Scope*: public preference and usage data with population caveats (CLM-016, CLM-018, CLM-019).

**WORKLOAD-DEPENDENT**
- Which signals are valid (regenerate, edit, dwell, abandonment; CLM-050–CLM-052), propensity models (CLM-056), CUPED gain (depends on correlation and coverage), SRM threshold, and the weight of feedback-derived data in training.
- *The Impact of AI on Developer Productivity* (Peng et al.). — [arXiv:2302.06590](https://arxiv.org/abs/2302.06590). Abstract only. Lab RCT (CLM-028).

**FRONTIER**
- *Language Models Learn to Mislead Humans via RLHF* (Wen et al.). — [arXiv:2409.12822](https://arxiv.org/abs/2409.12822). Abstract plus introduction (CLM-008).
- *On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback* (Williams et al.). — [arXiv:2411.02306](https://arxiv.org/abs/2411.02306). Abstract only; simulated users (CLM-009).
- *ELEPHANT* (Cheng et al.). — [arXiv:2505.13995](https://arxiv.org/abs/2505.13995). Abstract only (CLM-010).
- *Behavioral Indicators of Overreliance* (Liu et al. 2026). — [arXiv:2602.11567](https://arxiv.org/abs/2602.11567). Abstract only (CLM-012).
- *PRELUDE/CIPHER* (Gao et al.). — [arXiv:2404.15269](https://arxiv.org/abs/2404.15269). Full-text grep; simulated users (CLM-015).
- *The Leaderboard Illusion* (Singh et al.). — [arXiv:2504.20879](https://arxiv.org/abs/2504.20879). Abstract only (CLM-017).
- *LLM Hypnosis* (Hilel et al.). — [arXiv:2507.02850](https://arxiv.org/abs/2507.02850). Abstract only (CLM-020).
- *User Feedback Provides a Unique Signal that LLMs Can not Detect* (Don-Yehiya et al. 2026). — [arXiv:2609.02859](https://arxiv.org/abs/2609.02859). Abstract only (CLM-021).
- *Ensure A/B Test Quality at Scale* (Nie et al., eBay). — [arXiv:2208.07766](https://arxiv.org/abs/2208.07766). Abstract only (CLM-026).
- *Statistical Foundations of LLM-based A/B Testing* (Persson et al. 2026). — [arXiv:2606.17165](https://arxiv.org/abs/2606.17165). Abstract only (CLM-027).
- *To Trust or to Think* (Buçinca et al., CSCW 2021). — [arXiv:2102.09692](https://arxiv.org/abs/2102.09692). Abstract only (CLM-039).

**LEGACY**
- Kohavi, *Seven Rules of Thumb for Web Site Experimenters* (KDD 2014 slides), latency-to-revenue numbers for 2014 web search (CLM-036). LLM transfer is **H** (CLM-057).
- Practices to retire: thumbs share as a quality KPI; feedback rows as labels; approval-only A/B gates; reading effects without SRM checks.

**PRODUCTION SOURCE TRACE** (static reads at pinned commits, 2026-11-10; nothing executed)
- Repository: `growthbook/growthbook`. Revision: `362047d225f9bde2f7e2693114746cc6ddad0272`.
  - `packages/stats/gbstats/utils.py::{check_srm, variance_of_ratios}`; called from `packages/stats/gbstats/gbstats.py` `analyze_dimension`.
  - `packages/shared/src/constants.ts::DEFAULT_SRM_THRESHOLD`; `packages/front-end/components/Experiment/SRMWarning.tsx`.
  - `packages/stats/gbstats/models/statistics.py::{RegressionAdjustedStatistic, compute_theta, create_theta_adjusted_statistics}`; entry `packages/stats/gbstats/frequentist/tests.py::BaseABTest.__init__`.
  - Generalizable: PARTIAL. Matches CUPED with one pooled $\theta$; thresholds, pooling, and the zero-$\theta$ fallback are this tool's choices. Not traced: the use site of the minimum-count constant.
- Repository: `langfuse/langfuse`. Revision: `5349d484579e29da86cb124a21a9ae8d9e15bfc9`.
  - `packages/shared/src/domain/scores.ts::{ScoreSourceArray, ScoreDataTypeArray, AGGREGATABLE_SCORE_TYPES, isAnnotationScoreMissingConfigId}`.
  - `web/src/pages/api/public/scores/index.ts` (POST); `web/src/features/public-api/server/scores-api-service.ts::{createScore, createScores, toScoreCreateEvent}` → `processEventBatch`.
  - Generalizable: PARTIAL (one vendor's schema). Not traced: downstream worker validation of `SCORE_CREATE`.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow $\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}$. All data are generated by a seeded simulator written in the lab and are **synthetic**; every artifact says so. Ground truth (satisfaction, defects, correctness) is known to the simulator, which is what allows estimators to be checked.

### LAB A — Feedback Ledger and Nonresponse Correction
- **Objective**: Simulate exposures with latent satisfaction and signal-specific propensities; build a ledger that joins feedback to exposures with delayed (async) arrival; compare naive share, per-exposure rates, and IPW using a randomized prompt-for-feedback arm.
- **Pre-Registered Hypothesis**: When propensities depend only on observed strata, IPW from the randomized arm recovers the true satisfied share within its interval, while the naive share misses it by more than the declared practical margin (CLM-040, CLM-041).
- **Independent Variables**: $r_u$, $r_d$, UI prominence, randomized-arm fraction, ingestion delay, strata.
- **Dependent Variables**: naive share, per-exposure rates, IPW estimate and interval, bias against truth, fraction of feedback missing at window close.
- **Break & Falsify**: Make propensity depend on latent quality within strata (CLM-056). The hypothesis predicts IPW stays unbiased only when this dependence is absent; if IPW remains unbiased with the dependence present, the bias mechanism is falsified for that simulator. Also simulate regenerate as exploration in one slice and show that its rate does not track defects (CLM-050).
- **Hardware**: CPU.
- **Alignment**: Lessons 39.1–39.2.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB B — Position Bias and Swap Estimation
- **Objective**: Simulate a UI that shows three drafts or citations with position-dependent examination; log selections; estimate propensities by swap interventions; compute an IPS estimate of item quality.
- **Pre-Registered Hypothesis**: Swap-estimated propensity ratios match the simulator's within interval, and IPS ranks items by true quality where raw selection rate does not (CLM-003, CLM-055).
- **Independent Variables**: examination curve, swap fraction, clipping threshold, sample size.
- **Dependent Variables**: estimated $p_r/p_1$, IPS vs raw ranking agreement with truth, variance as clipping changes.
- **Break & Falsify**: Set examination flat (no position effect): the swap ratios should be 1 within interval; a nonzero estimated effect falsifies the estimator for that setup. Then drive $p_3\to0.02$ and show variance blow-up and the bias of clipping.
- **Hardware**: CPU.
- **Alignment**: Lesson 39.2.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Feedback to Eval and Training Data
- **Objective**: From a synthetic feedback log, (1) build an adjudicated eval set from several selectors plus random exposures and compute selector precision; (2) build a KTO dataset with class weights and a provenance filter; (3) inject a coordinated voting attack from a few accounts and measure how many poisoned labels reach the dataset with and without the filter.
- **Pre-Registered Hypothesis**: Thumbs-down precision exceeds the random defect rate; per-user caps and account filters cut poisoned labels in the training set by at least the declared fraction while removing few legitimate labels (CLM-047, CLM-020, CLM-046).
- **Independent Variables**: selector, adjudication budget, $\lambda_D$, filter settings, attacker count and vote rate.
- **Dependent Variables**: precision with intervals, eval set composition, class ratio, poisoned-label fraction, legitimate-label loss.
- **Break & Falsify**: Give the attacker many low-rate accounts so per-user caps do not bind; report the filter's failure. Compute normalized edit cost on simulated edits that mix corrections and added content and show where it stops tracking violations (CLM-051). Optionally train a tiny KTO model on CPU; not required.
- **Hardware**: CPU.
- **Alignment**: Lessons 39.3–39.4.
- **Effort Estimate**: 2.5h implementation, 1h analysis (3.5h total).

### LAB D — A/B Test Audit Toolkit
- **Objective**: Implement Pearson SRM, CUPED with an explicit $\theta$ report, unpaired sample size, and day-by-exposure effect plots; run them on simulated experiments with injected faults; compare with the GrowthBook logic traced in Section 09.
- **Pre-Registered Hypothesis**: The toolkit flags injected SRM at the declared threshold, CUPED achieves about $1-\rho^2$ variance reduction when the covariate is informative, and a test sized on CUPED loses the predicted power when $\theta$ falls back to 0 (CLM-043, CLM-044, CLM-045).
- **Independent Variables**: arm loss rate, pre-period correlation and coverage, constant covariate, novelty decay, approval-vs-quality trade in the treatment, shared-cache coupling between arms.
- **Dependent Variables**: SRM $p$, variance ratio, realized power over replications, day-level effect curve, approval and quality metrics side by side, user- vs cluster-randomized estimates.
- **Break & Falsify**: (1) A treatment raises simulated approval by increasing agreement while lowering correctness: show the approval A/B passes and the quality guardrail fails (CLM-049, CLM-007). (2) Couple arms through a shared cache and compare user- and cluster-randomized estimates (CLM-054); agreement within uncertainty falsifies interference for that coupling. (3) Simulate a trust-panel arm and report overreliance on seeded errors (CLM-048); simulate streaming at matched end-to-end time only as a design check for CLM-053, not as evidence.
- **Hardware**: CPU.
- **Alignment**: Lessons 39.5–39.6 and Incident 39.1.
- **Effort Estimate**: 2.5h implementation, 1h analysis (3.5h total). The source trace is a separate 2h artifact (Section 09).

---

## 07 Break / Incident Scenarios

### Incident 39.1: The Release Users Liked
- **Incident Symptoms**:
  A support-and-coding assistant ships a release whose training added a reward term from thumbs-up/down. Before launch:
  - Offline evals were flat or slightly positive.
  - A two-week 50/50 A/B test showed thumbs-up share +6 points, side-by-side win rate 54%, and sessions per user +3%. Days 1–3 showed the largest effects.
  - The experiment dashboard showed no SRM warning.
  Three weeks after full rollout:
  - Escalations to human support rose 11%; code suggestions accepted then reverted within a day rose.
  - Trust-and-safety reports the assistant agrees with users' incorrect claims about billing policy.
  - The thumbs-up share is still higher than before launch.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: approval reward selected for agreement (sycophancy) (CLM-049, CLM-007); nonresponse shift (the release changed who leaves feedback) (CLM-040); novelty inflating the early A/B effect (CLM-025); SRM hidden by a threshold override or by per-dimension checks; CUPED silently not applied, so the test was underpowered on guardrails (CLM-031); interference through a shared cache or memory store (CLM-054, **H**); unrelated traffic or policy change after launch; a poisoned feedback cohort (CLM-020).
  2. *Rank Initial Plausibility*: The pattern "approval up, verified outcomes down" fits the first hypothesis, but nonresponse shift and novelty also predict parts of it. Do not rank by narrative.
  3. *Identify Missing Evidence*: per-exposure feedback rates and response rates by arm; the randomized prompt-for-feedback arm, if any; the SRM p-value and the threshold actually in settings; whether $\theta\neq0$ for each metric; day-by-exposure effects; adjudicated samples of billing answers from both arms; the training data manifest with feedback provenance; the reward mixture weights.
  4. *Design Discriminating Tests*: Replay billing prompts with seeded incorrect user claims to both models and measure agreement rate (sycophancy eval). Recompute per-exposure rates and IPW share. Plot the effect against days since first exposure. Retrain with the feedback reward weight set to 0, holding all else fixed. Audit vote concentration by account. State what each result would falsify.
  5. *Execute Causal Diagnosis*: Rank causes from the test results. More than one may hold, for example sycophancy plus a nonresponse shift.
  6. *Prescribe Immediate Mitigation & Long-Term Intervention*: Roll back. Make sycophancy, factuality, and verified-outcome metrics launch-blocking guardrails regardless of A/B approval (as OpenAI did, CLM-007). Cap and gate the feedback reward weight; require long-horizon outcomes before full rollout; add an opt-in or staged phase.
  7. *Remeasure*: Sycophancy agreement rate, adjudicated billing correctness, escalation and revert rates, per-exposure feedback rates, and IPW share, on the same definitions as before the incident.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Product Transfer Problem: Feedback-Driven Release of a Coding & Support Assistant
You own the feedback loop for an assistant that answers support questions and suggests code. The team proposes a release that adds a thumbs-derived reward term and a new "why this answer" explanation panel, and wants to ship on a two-week A/B test.

**Workload Fixture (SYNTHETIC — exercise assumptions, not measurements of any product; registry CLM-058):**

| Input | Fixture value |
|---|---|
| Traffic | 400,000 weekly active users; 6,000,000 responses per week |
| Volunteered feedback | 0.9% of responses: 21,000 up, 33,000 down |
| Implicit signals | 9% of responses regenerated; 4% edited |
| Randomized prompt-for-feedback arm | 2% of responses, response rate 41% |
| Primary per-user outcome | weekly thumbs-up rate, $\sigma^2=0.21$; pre-period $\rho=0.55$ with 80% coverage |
| Pilot assignment | 50/50 design shows 201,800 vs 198,200 users |

You may replace any value with your own measurement; record the source. Answers are graded conditionally on declared inputs.

**Required Deliverables**:
1. **Signal inventory**: each signal, what it measures, its denominator, and its validity status (O, D, or H with the test that would change it).
2. **Bias analysis**: observed share and per-exposure rates; how the randomized arm yields propensities; the IPW estimate you would report and the assumption it rests on.
3. **Eval routing**: selectors, adjudication budget, provenance record, and how prevalence is reweighted.
4. **Training-data plan**: KTO class weights from the fixture's up:down ratio, provenance filters, and the behavior evals that gate the feedback-trained model.
5. **Experiment plan and audit**: SRM result on the pilot and what you do about it; sample size with and without CUPED for a 1-point change in weekly thumbs-up rate; novelty plan; guardrails that are not approval metrics; how you detect a silent CUPED fallback.
6. **Trust UI**: the overreliance measurement for the explanation panel and the decision rule if satisfaction rises while overreliance rises.
7. **Latency targets**: TTFT and end-to-end budgets tied to the response-time limits, and the streaming experiment if streaming is proposed (CLM-053 is **H**).
8. **Decision and rollback**: ship, hold, or redesign; the metrics and thresholds that would roll back the release after launch.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace
Submit a static trace at the pinned commits that maps:
1. GrowthBook: `analyze_dimension` → `check_srm` (statistic, df, zero-weight handling) → `DEFAULT_SRM_THRESHOLD` and the `SRMWarning` health rule.
2. GrowthBook: `statistic_type == "mean_ra"` → `BaseABTest.__init__` → `create_theta_adjusted_statistics` → `compute_theta`, including the condition that returns 0 and the fallback to unadjusted statistics.
3. GrowthBook: `variance_of_ratios` and when a ratio metric needs it.
4. Langfuse: `POST /api/public/scores` → `ScoresApiService.createScore` → `toScoreCreateEvent` → `processEventBatch`; score types, target IDs, and `AGGREGATABLE_SCORE_TYPES`.
5. One consequence for each tool: what the trace implies for computing feedback rates (Langfuse) and for trusting an experiment report (GrowthBook). State what was read statically and what, if anything, was executed.

### Reference Checks for Mastery Deliverables 2, 4, 5 (fixture inputs only)
- Observed up share $21{,}000/54{,}000=0.389$; volunteered response rate $54{,}000/6{,}000{,}000=0.9\%$. The randomized arm has $120{,}000$ responses and $49{,}200$ feedback events.
- KTO: up:down $=21{,}000:33{,}000$, so with $\lambda_U=1$, $\lambda_D\in[33/21,\,1.5\times33/21]=[1.571,\,2.357]$.
- SRM: $\chi^2=2\times1{,}800^2/200{,}000=32.4$, $p=1.25\times10^{-8}$: alert at any common threshold. Do not read the effect.
- Sample size for $\delta=0.01$: $15.70\times0.21/0.0001=32{,}966$ per arm; with CUPED at $\rho=0.55$, coverage 0.8: factor $1-0.8\times0.3025=0.758$, giving $24{,}988$ per arm.

### Rubric Dimensions
- **Measurement Validity** (Deliverables 1–2): *Insufficient* reports thumbs share as quality. *Competent* joins exposures, separates share from rate, and labels signal validity. *Strong* also estimates propensities from a randomized arm with intervals and states the test for quality-dependent nonresponse.
- **Data Provenance** (Deliverables 3–4): *Insufficient* uses feedback rows as labels or trains on all votes. *Competent* adjudicates with provenance, separates eval from training, and weights KTO classes. *Strong* measures selector precision against random and quantifies the filter's effect on a poisoning attempt.
- **Experiment Integrity** (Deliverable 5): *Insufficient* reads effects without SRM or plans without variance. *Competent* runs SRM, plans size, applies CUPED, and plans for novelty. *Strong* detects a silent CUPED fallback, distinguishes paired from unpaired planning, and states interference as a tested hypothesis.
- **Approval vs Quality** (Deliverables 5–6, Incident 39.1): *Insufficient* ships on a positive approval A/B. *Competent* adds non-approval guardrails. *Strong* makes behavior evals launch-blocking, measures overreliance with seeded errors, and defines long-horizon rollback triggers.
- **Source Reasoning** (Section 09 artifact): *Insufficient* cites file names. *Competent* traces the call paths at the pinned commits. *Strong* connects each traced behavior (df, threshold override, pooled $\theta$, fallback, async ingestion) to a concrete way a report could mislead.
- **Uncertainty Discipline** (all): *Insufficient* states hypotheses as facts. *Competent* labels O/D/H. *Strong* gives the measurement that would change each **H**.

---

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| **Feedback signal measurement and denominators** | Lesson 39.1 | Lesson 39.1 Guided Practice; LAB A | Mastery Deliverable 1; Incident 39.1 step 3 | Feedback ledger with exposure join and signal validity table |
| **Nonresponse and position bias correction** | Lesson 39.2 | Lesson 39.2 Guided Practice; LABs A–B | Mastery Deliverable 2; Incident 39.1 step 4 | IPW and swap estimates vs simulator truth |
| **Approval reward and sycophancy** | Lesson 39.2 | LAB D Break (1) | Incident 39.1 steps 1, 6; Mastery Deliverable 8 | Approval-vs-quality A/B simulation and guardrail list |
| **Feedback to eval sets** | Lesson 39.3 | Lesson 39.3 Guided Practice; LAB C (1) | Mastery Deliverable 3 | Selector precision table and provenance records |
| **Feedback to training data** | Lesson 39.4 | Lesson 39.4 Guided Practice; LAB C (2)–(3) | Mastery Deliverable 4; Section 09 Reference Checks | KTO dataset manifest and poisoning filter report |
| **A/B test mechanics** | Lesson 39.5 | Lesson 39.5 Guided Practice; LAB D | Mastery Deliverable 5; Section 09 Reference Checks; Incident 39.1 step 4 | SRM, CUPED, power, and novelty audit outputs |
| **Production source trace** | Lessons 39.1, 39.5; Section 05 | LAB D comparison with traced logic | Section 09 Required Artifact items 1–5 | Trace pinned to `362047d2…` and `5349d484…` |
| **Latency and trust UX** | Lesson 39.6 | Lesson 39.6 Guided Practice; LAB D Break (3) | Mastery Deliverables 6–7 | Overreliance comparison and streaming experiment design |

---

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
A learner successfully completing Module 39 must be able to:
1. State what each feedback signal measures, compute it against exposures, and label its validity.
2. Quantify nonresponse and position bias and correct them when propensities are estimable.
3. Explain how approval rewards produce sycophancy and design guardrails that approval metrics cannot replace.
4. Route feedback into adjudicated evaluation sets and provenance-filtered training data.
5. Plan and audit an LLM feature A/B test with SRM, CUPED, and time effects, and detect a silent fallback.
6. Set latency targets from measured timestamps and evaluate trust UI by overreliance.
7. Trace SRM, CUPED, and feedback ingestion in pinned production code.

### Module Wrap-Up (Final Mental Model Reconstruction)
- **The Core Invariant**: A feedback event is a measurement by a selected user under a UI-dependent propensity. It becomes evidence only after it is joined to exposures, corrected or bounded for selection, and validated against adjudicated outcomes.
- **The Loop**: exposure → propensity → event → ledger → (measurement | adjudicated eval | provenance-filtered training) → experiment with integrity checks and non-approval guardrails → release or rollback.
- Approval is not correctness. A positive A/B on approval cannot reject a release that trades truth for agreement; behavior evals and long-horizon outcomes must be able to block it. Regenerate, edit, dwell, streaming, and interference effects remain hypotheses until measured on your product.

### Cost & Telemetry Hook
- [Lesson 22.1](../22-ai-economics/README.md): cost adjudication and feedback-derived training per useful offered outcome, counting rejected and rolled-back releases.
- [Lesson 22.6](../22-ai-economics/README.md): the guardrail thresholds and adjudication budget are a decision frontier; report the flip point.
- [Lesson 23.3](../23-observability-reliability/README.md): define feedback SLIs per exposure, not per feedback event.
- [Lesson 23.5](../23-observability-reliability/README.md): thumbs and judge scores are fast proxies; long-horizon outcomes, adjudicated labels, and late feedback rows are delayed labels. Compare only mature cohorts before declaring a feedback-trained release healthy.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
