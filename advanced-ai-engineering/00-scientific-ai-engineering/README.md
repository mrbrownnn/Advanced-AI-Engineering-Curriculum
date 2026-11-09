# Module 00 — Scientific AI Engineering

## 00 Why This Module Exists

An advanced AI system can produce convincing benchmark numbers while measuring the wrong population, hiding failures, confounding the treatment with machine state, or applying a valid formula to dependent observations. This module establishes the evidence discipline used by every later module.

The goal is not to turn every engineer into a statistician. The goal is to make engineering claims auditable: define the quantity, preserve the observations, state assumptions, design a comparison that can fail, and connect the result to a decision threshold.

**Module Orientation**

- **Engineering Problem**: Decide whether a measured change is real, practically important, and likely to transfer to the intended workload.
- **What You Will Do**: Build a timestamped benchmark harness, expose warmup and order bias, compare randomized paired treatments, quantify uncertainty without overstating it, break a load generator with coordinated omission, trace MLPerf LoadGen, and package a repeatable artifact.
- **Environment**: Python 3.10+ for harness and analysis work; a GPU is optional. Hardware-specific experiments must record the accelerator, driver, runtime, clocks/power policy, and isolation state.
- **Research Cutoff**: evidence registry 2026-09-25; WP-F1 review 2026-09-30 re-read the pinned LoadGen completion and percentile code (CLM-014) and re-checked the presence of the other pinned symbols. Canonical statistical methods remain relevant; implementation claims remain pinned to the revisions in §05.

## 01 Baseline Assumptions

Prerequisites are basic algebra, basic probability, Python, and command-line use. This module introduces the curriculum-wide research contract:

- **O — Source observation**: what a source or experiment actually reports, scoped to its setup.
- **D — Derivation**: what follows from stated premises and assumptions.
- **H — Engineering hypothesis**: a causal explanation with a prediction, discriminating measurement, and falsifier.

This module teaches general measurement and experimental design. Domain-specific model evaluation belongs primarily to Module 15; GPU execution and profiler semantics to Module 02; serving queueing models to Module 04.

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
  instruction: 5h
  guided_practice: 3h
  labs: 10h
  assessment: 3h
  source_trace: 2h
  total: 23h
```

*Effort reconciliation*: lesson instruction sums to 300 min and lesson practice to 180 min. Labs are A 3h + B 3h + C 3h + D's 1h boundary experiment = 10h; LAB D's 2h source trace is counted once, under `source_trace`.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
Decision and target population
        ↓
Atomic claim + practical threshold
        ↓
Measurand, unit, boundary, success/failure semantics
        ↓
Workload + treatment + controls + experimental unit
        ↓
Randomized/blocked execution with state and provenance capture
        ↓
Raw event-level observations, failures, censoring, and metadata
        ↓
Estimator + uncertainty model + assumption checks
        ↓
Competing explanations + discriminating intervention
        ↓
Decision, rollback rule, and same-boundary remeasurement
```

Feedback paths matter. A slow trial can heat the device, change frequency, grow a queue, alter caching, or cause a response-coupled load generator to issue fewer requests. The measurement process can therefore change the state it intends to observe. Treat those paths as hypotheses and instrument them.

## 04 Lessons

### Lesson 0.1 — Claims, Estimands, and Evidence Types

**Engineering Question:** What exactly must be true for the result to justify a decision?

**Concepts & Definitions:**

A benchmark begins with a decision and an **estimand**: the population quantity the experiment aims to estimate. “Latency improved” is not an estimand. A usable statement might be:

> For successful requests from workload version W under offered load A, configuration B changes client-observed P95 end-to-end latency relative to A by an amount that matters operationally, while preserving the declared quality and failure criteria.

Define:

- target population and workload version;
- treatment and baseline;
- request and response boundaries;
- units and aggregation level;
- success, timeout, cancellation, retry, and exclusion semantics;
- practical threshold that would change the decision.

Keep O, D, and H separate (**D**, CLM-013). A paper's measured speedup is O. A FLOP or memory equation is D. “The speedup came from higher cache hit rate” is H until aligned telemetry and an intervention support it. A statistically detectable effect is not yet a decision: it must be compared with a practical threshold (**O**, CLM-012).

**Worked Example (synthetic instantiation):**
- *Input*: the vague claim “runtime B is faster than A.”
- *Steps*: (1) population: requests replayed from support-chat trace version W3; (2) load: open-loop arrivals at 8 requests/s; (3) boundary: client send to last byte received; (4) estimand: difference in P95 end-to-end latency over all issued requests, nearest-rank convention, timeouts counted at the 10 s timeout value; (5) guard conditions: failure rate not higher by more than 0.2 percentage points and the declared quality check unchanged; (6) practical threshold: adopt B only if P95 falls by at least 10%.
- *Result*: “For W3 at 8 requests/s open-loop, B changes client-observed P95 end-to-end latency relative to A by at least −10% with failure and quality guards satisfied” is one atomic, falsifiable claim. Its status is H until measured, and it becomes O for this setup only.
- *Interpretation / limits*: none of the six choices is universal. A different load model or timeout treatment defines a different estimand, and a result for W3 says nothing about other workloads without new evidence.

**Knowledge Check:**
1. Which part of “B is faster because cache reuse increased” is O, D, or H?
2. Why does a statistically detectable effect not automatically justify deployment?

**Guided Practice:** Rewrite these claims atomically and label each O, D, or H:

1. “The new runtime is 20% faster.”
2. “P99 improved, so users will prefer it.”
3. “The kernel is memory-bound because GPU utilization is low.”

**Feedback Contract:**
- *Expected Evidence*: each rewritten claim names population, workload, boundary, estimator, baseline, uncertainty, and limits. Claim 1 is O only with a named setup; claim 2 joins an O (“P99 improved”) to an unsupported H (“users will prefer it”); claim 3 is an H whose stated evidence (low utilization) does not discriminate it.
- *Common Failure*: labeling a causal explanation O because a number accompanies it.
- *Diagnostic Hint*: for each sentence ask, “which observation would prove this wrong?” If none exists, it is not yet a testable claim.
- *Concept to Revisit*: estimands and the O/D/H distinction.

**Learning Outcome:** Write atomic, decision-linked claims while preserving the distinction among observation, derivation, and hypothesis.

*(Effort: 45m instruction, 25m practice)*

---

### Lesson 0.2 — Harness Mechanics and Measurement Boundaries

**Engineering Question:** Which timestamps and state transitions produce the reported value?

**Concepts & Definitions:** A measurement boundary names the clocks, lifecycle states, population, and inclusion rules that give an observed duration meaning. A number without a declared measurand, boundary, unit, procedure, and uncertainty is not a complete engineering claim (**O**, CLM-001).

**Mechanism Explanation:**

A harness should make its lifecycle observable:

1. pin code, data, configuration, dependencies, and hardware identity;
2. initialize the system and record initialization separately;
3. warm to a declared state or intentionally measure cold start;
4. schedule work according to a declared arrival model;
5. synchronize asynchronous execution at the observation boundary;
6. record intended issue, actual issue, start, completion, failure, and cancellation events;
7. preserve raw event data before aggregation.

**Quantitative Model / Derivation:**
For one observation, the measured interval is

$$Y_i=t_{end,i}-t_{start,i}.$$

where: $i$ indexes one observation (e.g., one request); $t_{start,i}$ and $t_{end,i}$ are the timestamps of the two declared boundary events, read from the *same* clock; $Y_i$ is the measured interval in that clock's units. Changing which events $t_{start,i}$ and $t_{end,i}$ denote changes the measurand, as the worked example below shows.

This identity is exact only for the declared clocks and boundaries. It does not prove that $Y_i$ is GPU execution time, user-visible latency, or service time. Timer resolution, clock domain, asynchronous queues, batching, buffering, and instrumentation overhead can change the meaning.

**Warmup is part of the question.** Cold-start and steady-state performance are different estimands. Do not discard early samples merely because they are slower; declare which state matters and report transition behavior when it matters operationally.

**Coordinated omission:** If the generator waits for a response before issuing the next intended request, a long stall suppresses arrivals and therefore suppresses latency samples (**O**, CLM-009). Record an independent intended-arrival schedule when the production question assumes arrivals independent of completions.

**Worked Example (synthetic timestamps, one synchronized clock):**
- *Input*: intended issue 0 ms; actual issue 20 ms; server receives 22 ms; device work 35–45 ms; server sends 55 ms; client receives 60 ms.
- *Steps*: issue lag $=20-0=20$ ms; latency from intended arrival $=60-0=60$ ms; latency from actual issue $=60-20=40$ ms; server residence $=55-22=33$ ms; device time $=45-35=10$ ms.
- *Result*: five different numbers from one request: 20, 60, 40, 33, and 10 ms.
- *Interpretation / limits*: a response-coupled generator would report 40 ms and hide the 20 ms the request waited to be issued. Reporting the 10 ms device interval as “latency” omits 50 ms of what the user experienced. If client and server clocks are not synchronized, the 33 ms and 10 ms intervals remain valid within their own clocks, but cross-clock differences do not.

**Knowledge Check:** Why can a host timer around an asynchronous launch understate execution time, and which event reveals coordinated omission?

**Guided Practice:** Draw timestamp boundaries for client-observed latency, server residence, device execution, and post-processing. Identify which pairs of timestamps share a clock.

**Feedback Contract:**
- *Expected Evidence*: a timeline with every event, its clock, and the four named intervals; which timestamp pairs share a clock; the inclusion rule for failures and cancellations; intended and actual issue recorded separately. Knowledge check: an asynchronous launch returns before completion, and a growing intended-minus-actual issue lag reveals coordinated omission.
- *Common Failure*: subtracting timestamps from different clocks, or measuring from actual issue in a response-coupled loop.
- *Diagnostic Hint*: during an injected 2-second stall, how many requests did your generator intend to issue, and how many latency samples did it record?
- *Concept to Revisit*: measurement boundaries and coordinated omission.

**Learning Outcome:** Implement and defend a lifecycle-aware timing contract that preserves cold state, failures, and intended arrivals.

*(Effort: 40m instruction, 25m practice)*

---

### Lesson 0.3 — Experimental Units, Randomization, Blocking, and Pairing

**Engineering Question:** What is actually independent, and what nuisance factors can reverse the comparison?

**Concepts & Definitions:**

The **experimental unit** is the smallest unit independently assigned to a treatment. Ten thousand requests inside one process are not ten thousand independent process-level replications (**O**, CLM-004). Setup details that look irrelevant can bias or even reverse a comparison (**O**, CLM-002), which is why treatments are randomized within blocks of comparable nuisance conditions (**O**, CLM-003). Generalization across seeds, model loads, machines, or days requires repetition at those levels.

Use:

- **control** to hold an important factor fixed;
- **blocking** to compare treatments within a nuisance-factor level;
- **randomization** to avoid systematically aligning treatment with uncontrolled drift;
- **pairing** to analyze within-block differences when observations are meaningfully matched.

**Quantitative Model / Derivation:**
For paired block differences $d_i=Y_{B,i}-Y_{A,i}$:

$$\bar d=\frac{1}{n}\sum_i d_i,\qquad SE(\bar d)=\frac{s_d}{\sqrt n}.$$

An approximate classical interval is

$$\bar d\pm t_{1-\alpha/2,n-1}\frac{s_d}{\sqrt n},$$

under the stated independence and distribution assumptions for the block-level differences.

where:
- $i=1,\dots,n$ indexes blocks, and $n$ is the number of independent blocks (not requests);
- $Y_{A,i}$, $Y_{B,i}$ are the block-level results (e.g., block mean latency) for baseline A and treatment B in block $i$;
- $\bar d$ is the mean paired difference; a positive $\bar d$ means B is larger (slower, for latency);
- $s_d=\sqrt{\frac{1}{n-1}\sum_i (d_i-\bar d)^2}$ is the sample standard deviation of the differences ($n-1$ in the denominator);
- $SE(\bar d)$ is the standard error of $\bar d$;
- $\alpha$ is one minus the confidence level ($\alpha=0.05$ for 95%), and $t_{1-\alpha/2,n-1}$ is the $1-\alpha/2$ quantile of Student's t distribution with $n-1$ degrees of freedom.

Why pairing helps: the nuisance level shared by A and B inside block $i$ appears in both $Y_{A,i}$ and $Y_{B,i}$, so it cancels in $d_i$. The interval is then built from the spread of the differences, $s_d$, rather than from the larger block-to-block spread of the raw results.

**Worked Example (synthetic differences):**
- *Input*: four independent fresh-process blocks, each running A and B in random order, give paired differences $d=[1.2,0.9,1.1,0.8]$ ms.
- *Steps*: $\bar d=4.0/4=1.0$ ms; squared deviations $0.04+0.01+0.01+0.04=0.10$, so $s_d=\sqrt{0.10/3}\approx0.183$ ms; $SE=0.183/\sqrt4\approx0.091$ ms; $t_{0.975,3}=3.182$.
- *Result*: $1.0\pm3.182\times0.091=[0.71,1.29]$ ms.
- *Interpretation / limits*: the interval excludes zero under the model, but $n=4$ is the number of blocks, not the number of requests inside them, and four blocks on one machine on one day say little about other machines or days. If each block had instead been 10,000 requests from a single process, using $n=40{,}000$ would shrink the interval by more than 100-fold without adding any process-level evidence.

**Knowledge Check:** Why are many requests in one process not necessarily independent process replications, and when is pairing invalid?

**Independent Practice:** Run A then B repeatedly without randomized order; then randomize order within fresh-process blocks and compare conclusions. CPU/GPU temperature, frequency, caches, allocator state, and background work can become treatment labels.

**Feedback Contract:**
- *Expected Evidence*: the assignment unit and why it is independent; the block factor; the randomization record (seed and realized order); paired differences; a carryover check (A-then-B versus B-then-A); and the population the result can transfer to. In the fixed-order run, the learner shows how drift is attributed to the treatment.
- *Common Failure*: computing $s/\sqrt n$ with $n$ equal to the number of requests when treatments were assigned per process.
- *Diagnostic Hint*: what was randomly assigned—requests, processes, or machines? That is your $n$.
- *Concept to Revisit*: experimental unit and pseudoreplication.

**Learning Outcome:** Design randomized, blocked, or paired comparisons at the level that supports the intended claim.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 0.4 — Distributions, Quantiles, Failures, and Outliers

**Engineering Question:** Which summary preserves the user-visible behavior relevant to the decision?

**Concepts & Definitions:**

Mean, median, quantiles, maximum, throughput, and failure rate answer different questions. A mean cannot guarantee P99. A P99 does not describe the worst case. A percentile is incomplete without:

- population and observation boundary;
- quantile convention;
- sample count and dependence structure;
- treatment of timeouts, errors, censoring, and dropped requests;
- uncertainty or stability across independent runs.

**Quantitative Model / Derivation:**
For an independent event with probability $q$, the chance of observing it at least once in $n$ trials is

$$P(\text{at least one})=1-(1-q)^n.$$

If $q=0.01$, achieving a 95% chance of seeing at least one such event requires

$$n\ge\frac{\ln(0.05)}{\ln(0.99)},$$

so $n=299$ after rounding upward ($\ln0.05/\ln0.99=298.07$).

where: $q$ is the per-trial probability of the event (here, landing in the slowest 1%); $n$ is the number of independent trials; $(1-q)^n$ is the probability that all $n$ trials miss the event.

*Step-by-step*:
1. Require $1-(1-q)^n\ge0.95$, i.e. $(1-q)^n\le0.05$.
2. Take natural logs of both sides (log is increasing, so the inequality keeps its direction): $n\ln(1-q)\le\ln0.05$.
3. Divide by $\ln(1-q)$, which is **negative** for $0<q<1$, so the inequality flips: $n\ge\ln0.05/\ln(1-q)$.
4. With $q=0.01$: $\ln0.05\approx-2.9957$ and $\ln0.99\approx-0.010050$, giving $n\ge298.07$; $n$ must be an integer, so $n=299$.
5. Check: $0.99^{298}\approx0.0500$ (just above 0.05, fails) and $0.99^{299}\approx0.0495$, so $P(\text{at least one})\approx0.9505\ge0.95$.

This is only an exposure check (**D**, CLM-007). It does not produce a precise confidence interval for P99.

**Worked Example (synthetic run; nearest-rank quantile, rank $=\lceil qn\rceil$):**
- *Convention*: sort the $n$ observations ascending; the nearest-rank $q$-quantile is the observation at 1-based rank $\lceil qn\rceil$ (round $qn$ up to an integer). Here $q$ is the quantile level (0.95 for P95), not the event probability of the previous formula.
- *Input*: 100 issued requests: 94 successes and 6 timeouts at a 2,000 ms timeout. Among the successes, the 90th smallest latency is 240 ms.
- *Steps*: success-only P95 uses $n=94$: rank $\lceil0.95\times94\rceil=90$, so 240 ms. All-issued P95 uses $n=100$ with timeouts ordered last: rank 95 falls among the 6 timeouts (ranks 95–100).
- *Result*: success-only P95 $=240$ ms; all-issued P95 $\ge2{,}000$ ms (censored at the timeout); failure rate $=6\%$.
- *Interpretation / limits*: both numbers are correct for their populations and they answer different questions. Reporting 240 ms without the 6% and the population silently removes the slowest experiences. With only 100 requests, either P95 is also unstable across runs.

**Outliers are evidence until explained** (**O**, CLM-008). A slow observation may be clock corruption, a GC pause, thermal throttling, a retry, a real failure mode, or a valid heavy-tail sample. Keep raw data. Apply only predeclared rules, record reasons, and report sensitivity with and without exclusions or with robust estimators.

**Knowledge Check:** Why can a mean improve while P99 worsens, and what changes when timeouts disappear from the denominator?

**Guided Practice:** Given a run with 2,000 successes, 40 timeouts, and 10 client cancellations, define two defensible metrics for different decisions. Explain why silently dropping the 50 incomplete requests changes the population.

**Feedback Contract:**
- *Expected Evidence*: two metrics with populations, e.g. (a) latency quantiles over all 2,050 issued requests with the 40 timeouts censored at the timeout and the 10 cancellations reported separately, for a user-experience decision; (b) success-only latency over 2,000 requests *together with* the 2.4% incomplete rate (50/2,050), for a capacity decision. Dropping the 50 changes the population from “issued” to “succeeded” and removes exactly the slowest outcomes.
- *Common Failure*: a single “P99” with no denominator, quantile convention, or failure treatment.
- *Diagnostic Hint*: what is $n$ in your quantile, and where do timeouts sit in the sorted list?
- *Concept to Revisit*: population, censoring, and quantile conventions.

**Learning Outcome:** Select summaries and failure semantics that preserve the behavior relevant to the decision.

*(Effort: 40m instruction, 20m practice)*

---

### Lesson 0.5 — Confidence Intervals, Sample Size, and Practical Significance

**Engineering Question:** How uncertain is the estimate, and is the plausible effect large enough to matter?

**Concepts & Definitions:** An interval procedure, effect size, minimum practically important effect, and decision rule answer different questions.

**Quantitative Model / Derivation:**

For independent observations from a normal population with unknown variance, the classical interval for a mean is

$$\bar x\pm t_{1-\alpha/2,n-1}\frac{s}{\sqrt n}.$$

where: $n$ is the number of independent observations; $\bar x$ is their sample mean; $s$ is their sample standard deviation (denominator $n-1$); $s/\sqrt n$ is the standard error of $\bar x$; $\alpha$ and $t_{1-\alpha/2,n-1}$ are as in Lesson 0.3. The term $t\cdot s/\sqrt n$ is the **half-width**: the interval extends that far on each side of $\bar x$.

A 95% confidence procedure has approximately 95% long-run coverage under its assumptions. It does not mean there is a 95% posterior probability that the fixed parameter lies inside this realized interval (**O**, CLM-005).

**Worked Example (synthetic summary statistics):**
- *Input*: $n=25$ independent process-level means, $\bar x=10$ ms, $s=2$ ms, $t_{0.975,24}\approx2.064$.
- *Steps*: $SE=2/\sqrt{25}=0.4$ ms; half-width $=2.064\times0.4=0.826$ ms.
- *Result*: approximately $[9.17,10.83]$ ms, a half-width of 8.3% of the mean.
- *Interpretation / limits*: conditional on independence and the t model. If the practical threshold is a 0.5 ms change, this interval is too wide to decide; if it is 3 ms, it is ample. It says nothing about P99.

For known $\sigma$ and target absolute half-width $E$:

$$n\ge\left(\frac{z_{1-\alpha/2}\sigma}{E}\right)^2.$$

where: $\sigma$ is the (assumed known) population standard deviation; $E$ is the largest acceptable half-width, in the same units as $\sigma$; $z_{1-\alpha/2}$ is the standard normal quantile ($1.96$ for 95%).

*Step-by-step*: with known $\sigma$ the half-width is $z_{1-\alpha/2}\,\sigma/\sqrt n$. Require it to be at most $E$: $z\sigma/\sqrt n\le E \iff \sqrt n\ge z\sigma/E \iff n\ge(z\sigma/E)^2$ (both sides are positive, so squaring keeps the direction). Round up, because $n$ is an integer and rounding down would make the half-width exceed $E$. Note that halving $E$ multiplies $n$ by 4.

With $\sigma=2$ ms, $E=0.5$ ms, and 95% confidence, the normal approximation gives $n\ge(1.96\times2/0.5)^2=61.47$, hence 62 (**D**, CLM-006). If $\sigma$ came from a small pilot, 62 is a planning approximation. This formula does not size P99, ratios, clustered data, multiple comparisons, or statistical power for a minimum effect.

Report effect size and uncertainty against a **minimum practically important effect**. A small p-value does not measure effect size, practical value, or the probability that a hypothesis is true.

When observations are dependent or hierarchical, analyze at the correct level or use a model/resampling plan that preserves the dependency structure. Naively bootstrapping individual requests from one process does not create independent process replications.

**Knowledge Check:** What does a 95% frequentist procedure claim, and why can a narrow interval still be operationally irrelevant?

**Guided Practice:** Choose a minimum practically important effect and use pilot variance only as a planning input. Explain what changes when observations are clustered by process.

**Feedback Contract:**
- *Expected Evidence*: the minimum practically important effect in units; the pilot variance and its source; the planning $n$ labeled as an approximation; the experimental unit; and a statement that with observations clustered by process, $n$ counts processes (or a hierarchical/resampling analysis preserves the clustering). Knowledge check: 95% describes the long-run coverage of the procedure; a narrow interval around an effect smaller than the practical threshold changes no decision.
- *Common Failure*: “there is a 95% probability the true mean is in this interval,” or sizing a P99 experiment with the mean formula.
- *Diagnostic Hint*: if you doubled requests per process but kept the number of processes, what new process-level information did you gain?
- *Concept to Revisit*: interval coverage, experimental unit, and practical significance.

**Learning Outcome:** Quantify uncertainty without overstating probability, independence, tail coverage, or practical significance.

*(Effort: 50m instruction, 25m practice)*

---

### Lesson 0.6 — Provenance, Repeatability, and Source Tracing

**Engineering Question:** Could another engineer understand, rerun, and challenge the result?

**Concepts & Definitions:** Provenance connects a metric to code, data, configuration, environment, raw events, and transformation steps.

**Mechanism Explanation:**

The artifact should include:

- code revision and dirty-tree state;
- dependency lock or image identity;
- model, tokenizer, dataset, prompt, and workload versions;
- hardware, driver, firmware, clocks/power policy, topology, and resource isolation;
- full configuration and random seeds;
- raw event data, logs, failures, and checksums;
- analysis code and generated tables/plots;
- exact commands and known limitations.

This module adopts ACM's current terminology and cites it explicitly (**O**, CLM-010):

- **repeatability**: same team and setup;
- **reproducibility**: different team using the same setup/artifacts;
- **replicability**: different team with an independently developed setup.

Other communities have used the last two terms differently, so never rely on the word alone.

**Worked Example (static source reading):**
- *Input*: MLCommons Inference commit `3fbc329939999c13d0a7b5e67fb2092287e06047`, question: “what interval is a LoadGen sample latency, and how is its percentile chosen?”
- *Steps*: (1) `loadgen/loadgen.cc::StartTest` through sanitized settings and scenario/mode dispatch; (2) `IssueQueries` to `loadgen/issue_query_controller.cc::IssueQueryController::StartIssueQueries` (**O**, CLM-011); (3) SUT completion through `loadgen/loadgen.cc::QuerySamplesComplete`, which reads `PerfClock::now()` once on entry and passes that timestamp to each sample's `SampleComplete`; (4) in `ResponseDelegateDetailed::SampleComplete` the recorded latency is `sched.delta(complete_begin_time)`, where `sched` is built from `query->scheduled_time`; (5) `loadgen/results.cc::PerformanceSummary::ProcessLatencies` sorts the latencies and indexes `sample_latencies[sample_count * percentile]`.
- *Result*: at this revision a sample latency runs from the query's *scheduled* time to the completion-callback timestamp, and a percentile is the sorted element at the truncated index $\lfloor n\cdot p\rfloor$ (0-based) (**O**, CLM-014). For example, with $n=1{,}000$ sorted latencies and $p=0.99$, the index is $\lfloor990\rfloor=990$, i.e. the 991st smallest value; the nearest-rank rule of Lesson 0.4 would instead take rank $\lceil0.99\times1000\rceil=990$, the 990th smallest. Here $n$ is the sample count, $p$ the percentile as a fraction, and $\lfloor\cdot\rfloor$ rounds down.
- *Interpretation / limits*: measuring from scheduled rather than actual issue time is the property that keeps issue delay inside the latency (Lesson 0.2). The index rule is one quantile convention among several, so the same data can give a slightly different “P99” in another tool. This is static inspection of one upstream snapshot; it is not an executed run, and MLPerf's workload contract is not the universal definition of inference performance.

**Knowledge Check:** Why is a commit hash insufficient when generated artifacts or the worktree differ, and which path establishes completion timing?

**Independent Practice:** Produce the pinned trace, provenance manifest, and exact rerun commands. Mark unexecuted branches `TODO_VERIFY`.

**Feedback Contract:**
- *Expected Evidence*: repository, revision, dirty-tree state, exact files and symbols, the execution path, environment, raw output, checksums, rerun commands, and static-versus-executed status for every step. Knowledge check: a commit hash does not identify uncommitted changes, generated files, data, or dependencies; `QuerySamplesComplete` establishes completion timing.
- *Common Failure*: citing a branch name or “latest,” or presenting a statically read path as executed.
- *Diagnostic Hint*: could a colleague with only your manifest rebuild the same binary and regenerate the same table?
- *Concept to Revisit*: provenance and repeatability/reproducibility/replicability.

**Learning Outcome:** Package a result so another engineer can audit its lineage, rerun it, and identify unresolved verification.

*(Effort: 40m instruction, 30m practice, source-trace integration)*

---

### Lesson 0.7 — Diagnostic Reasoning and Falsification

**Engineering Question:** What evidence would make the favored explanation wrong?

**Concepts & Definitions:** A useful diagnosis is a ranked causal explanation with alternatives, missing evidence, predicted observations, and a measurement or intervention that can weaken it.

**Mechanism Explanation:**

Use this loop:

```text
SYMPTOM
  → COMPETING HYPOTHESES
  → MISSING EVIDENCE
  → DISCRIMINATING MEASUREMENT OR INTERVENTION
  → RANKED EXPLANATION
  → CHANGE
  → SAME-BOUNDARY REMEASUREMENT
```

Example symptom: treatment B improves mean latency but worsens P99.

Competing hypotheses include cache warmup, batching changes, thermal drift, retries, rare long inputs, load-generator omission, or measurement corruption. GPU utilization, one profiler screenshot, or one correlation is necessary in some investigations but insufficient to select one mechanism. Predict what each hypothesis should change, intervene on one causal link, and preserve alternatives that remain observationally equivalent.

**Worked Example (synthetic results):**
- *Input*: fixed order A→B shows mean latency 100 ms → 90 ms and P99 400 ms → 520 ms. Hypotheses: H1 order/warm-up drift, H2 a real mean improvement from B, H3 a B-specific tail mechanism (e.g., rare long batches).
- *Steps*: predictions under randomized order within fresh-process blocks: H1 predicts the mean difference vanishes; H2 predicts it persists; H3 predicts the P99 gap persists regardless of order. The rerun gives mean 95 ms versus 95 ms and P99 405 ms versus 515 ms.
- *Result*: H1 is strengthened and H2 weakened for the mean; the P99 regression survives randomization, so H3 (or another B-specific tail cause) remains live.
- *Interpretation / limits*: the intervention discriminated H1 from H2 but did not identify the tail mechanism. The next measurement must target the tail (per-request batch composition, retries, input length) rather than repeat the mean comparison. Blocks here are synthetic; a real rerun also needs uncertainty on both differences.

**Knowledge Check:** What observation weakens thermal drift, and why is lower GPU utilization not a unique cause?

**Guided Practice:** Build a matrix of hypotheses, predicted telemetry, discriminating interventions, falsifiers, and unresolved equivalence for the example symptom.

**Feedback Contract:**
- *Expected Evidence*: a matrix with at least three hypotheses, the telemetry each predicts, one intervention that separates at least two of them, the falsifying observation for each, and the pairs that remain observationally equivalent. Knowledge check: stable temperature/frequency telemetry, or an effect that survives randomized order, weakens thermal drift; lower utilization is also produced by host gaps, smaller batches, or a faster kernel.
- *Common Failure*: listing alternatives but choosing an intervention every hypothesis predicts identically.
- *Diagnostic Hint*: for your chosen measurement, write the expected result under each hypothesis. If the column is constant, pick another measurement.
- *Concept to Revisit*: discriminating measurements and falsifiers.

**Learning Outcome:** Produce an evidence-backed explanation that states what was ruled out, what remains uncertain, and what next measurement would change the decision.

*(Effort: 40m instruction, 25m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- NIST/SEMATECH Engineering Statistics Handbook: [confidence limits for the mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm), [randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm), and [outlier detection](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm) (CLM-003, CLM-005, CLM-008).
- NIST, [*Metrological Traceability*](https://www.nist.gov/metrology/metrological-traceability): measurand, procedure, and uncertainty (CLM-001).
- Mytkowicz et al. (ASPLOS 2009), [*Producing Wrong Data Without Doing Anything Obviously Wrong*](https://research.ibm.com/publications/producing-wrong-data-without-doing-anything-obviously-wrong): setup-induced measurement bias (CLM-002).
- Georges, Buytaert, and Eeckhout (OOPSLA 2007), [*Statistically Rigorous Java Performance Evaluation*](https://doi.org/10.1145/1297027.1297033): startup, steady state, and repeated execution (CLM-004).
- Kalibera and Jones (ISMM 2013), [*Rigorous Benchmarking in Reasonable Time*](https://kar.kent.ac.uk/33611/): hierarchical variance and experiment dimensioning (CLM-004).
- Wasserstein and Lazar (2016), [*The ASA's Statement on p-Values*](https://doi.org/10.1080/00031305.2016.1154108): limits of p-value interpretation (CLM-012).
- ACM, [*Artifact Review and Badging – Current*](https://www.acm.org/publications/policies/artifact-review-and-badging-current): repeatability, reproducibility, replicability (CLM-010).

These references were accessed 2026-09-25 and were not re-read in the WP-F1 review. They are canonical methods sources, not evidence about 2025–2026 practice; this module makes no frontier or current-default claim about benchmarking practice.

**CURRENT IMPLEMENTATION SNAPSHOTS**

- MLCommons Inference LoadGen commit `3fbc329939999c13d0a7b5e67fb2092287e06047`: call path statically verified 2026-09-25 (CLM-011); completion timestamp, latency origin, and percentile indexing re-read 2026-09-30 (CLM-014). Not executed.
- HdrHistogram commit `de84b0a7de2378abfc405da503bf4898e84ea98e`, `recordValueWithExpectedInterval`: statically verified 2026-09-25; symbol presence re-checked 2026-09-30 (CLM-009).

**WORKLOAD-DEPENDENT**

- Warmup duration, number of repetitions, load model, percentile estimator, block factors, and practical thresholds.
- Whether a closed-loop load generator is representative of the real producer/client system.

**LEGACY / REJECT**

- Best-of-N reporting without a declared estimand.
- Deleting slow observations solely because they look anomalous.
- Treating thousands of correlated requests as independent machine-level replications.
- Claiming a universal sample count for “95% confidence and 5% error.”

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

Every lab follows:

$$\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}$$

### LAB A — Event-Level Benchmark Harness

- **Objective**: Build a harness around a controllable noisy operation with event-level timestamps and raw JSONL/CSV output.
- **Pre-Registered Hypothesis**: A response-coupled generator will under-observe latency during an injected stall relative to an independent intended-arrival schedule.
- **Independent Variables**: Load-generation mode, cold/warm state, stall injection, and observation boundary.
- **Dependent Variables**: Intended/actual issue lag, latency, throughput, failures, and declared quantiles.
- **Required controls**: cold versus warm state, timer and clock identity, synchronization boundary, intended and actual issue times, success/failure status, machine metadata, and immutable configuration.
- **Break & Falsify**: Introduce a 2-second pause while comparing response-coupled and independent-arrival generators.
- **Evidence**: raw trace, aggregation code, mean/median/quantiles/failures, quantile convention, and an explanation of coordinated omission.
- **Alignment**: Lessons 0.1, 0.2, and 0.4.
- **Effort Estimate**: 2h implementation, 1h analysis.

### LAB B — Randomized Paired Benchmark

- **Objective**: Compare baseline A and treatment B across independent blocks such as fresh processes or time windows.
- **Independent Variables**: Treatment and randomized within-block order; nuisance-factor levels are block metadata.
- **Dependent Variables**: Paired differences, uncertainty, carryover/order effects, and threshold decisions.
- **Design**: randomize A/B order within each block, record nuisance factors, retain paired differences, and test an A-then-B order as a deliberately biased comparator.
- **Pre-Registered Hypothesis**: State the mechanism, predicted telemetry, practical effect threshold, and falsifier before running.
- **Break & Falsify**: Add a monotonic background load or thermal drift and show when fixed order reverses or exaggerates the conclusion.
- **Alignment**: Lessons 0.3 and 0.5.
- **Effort Estimate**: 2h implementation, 1h analysis.

### LAB C — Uncertainty and Tail Audit

- **Objective**: Compare a classical mean interval, a correctly structured resampling or hierarchical analysis, and a tail-exposure calculation.
- **Pre-Registered Hypothesis**: Independent process starts support a process-level claim more directly than additional correlated requests within one process.
- **Independent Variables**: Requests per process, process starts, resampling unit, and synthetic tail-event probability.
- **Dependent Variables**: Interval behavior, effective sample structure, and tail-exposure probability.
- **Design**: vary requests per process and number of independent process starts while holding total request count similar.
- **Break & Falsify**: Demonstrate that more within-process requests can narrow a naive interval without adding process-level evidence.
- **Evidence**: assumptions, effective experimental unit, interval method, sensitivity analysis, and no percentile guarantee derived from a mean formula.
- **Alignment**: Lessons 0.3–0.5.
- **Effort Estimate**: 2h analysis, 1h report.

### LAB D — Reproducible LoadGen Source Trace

- **Objective**: Build or inspect the pinned MLPerf LoadGen revision and produce a source trace for one scenario.
- **Pre-Registered Hypothesis**: Changing an issue, completion, or failure boundary can change the metric while SUT work is held fixed.
- **Independent Variables**: Scenario, boundary/failure policy, and instrumented versus uninstrumented execution.
- **Dependent Variables**: Issued/completed counts, latency population, aggregate metrics, and trace completeness.
- **Required trace**: settings sanitation, scenario dispatch, request issue, completion timestamp, result processing, and relevant output artifacts.
- **Break & Falsify**: Change one boundary or failure policy and show how the reported metric changes even when the SUT is unchanged.
- **Artifact**: exact commands, patches if any, raw logs, code revision, build environment, and a `TODO_VERIFY` for any path not executed.
- **Alignment**: Lessons 0.2 and 0.6.
- **Effort Estimate**: 2h source trace (counted under `source_trace`), 1h boundary experiment (3h total).

## 07 Break / Incident Scenarios

### Incident 00.1: The 18% Optimization That Disappeared

A team reports an 18% latency improvement after replacing a runtime component. The baseline always ran first, treatment second. Only successful requests were logged. Each configuration processed 50,000 requests in one long process. A rerun on another machine shows no gain, and production P99 worsens.

**Diagnostic Protocol (Task):**
The learner must:

1. enumerate competing explanations: warmup, thermal/frequency drift, cache state, retry/failure filtering, process-level pseudoreplication, workload mismatch, or a real machine interaction;
2. identify missing raw data and provenance;
3. design a randomized blocked rerun with explicit success/failure semantics;
4. choose estimands and practical thresholds before seeing the new result;
5. use aligned state telemetry to rank explanations;
6. preserve a result that contradicts the original claim;
7. propose a rollback and same-boundary production remeasurement.

The final response must separate immediate rollback from long-term measurement correction and define quantitative evidence that would allow reconsideration.

No single metric is declared the root cause in advance.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Defend a Decision-Ready Benchmark

Given two AI inference configurations and a noisy heterogeneous workload, produce a decision-ready benchmark package.

**Required deliverables**

1. Atomic decision claim, target population, O/D/H labels, and minimum practical effect.
2. Measurement contract with timestamp boundaries, units, workload, failures, retries, censoring, and exclusions.
3. Experimental-unit argument and randomized blocked design.
4. Raw event-level data and provenance manifest.
5. Mean and distribution summaries with justified uncertainty methods.
6. A tail audit that states the quantile convention and why the sample supports—or does not support—the tail claim.
7. At least three competing causal explanations and one discriminating intervention.
8. Reanalysis after intentionally introducing order bias or coordinated omission.
9. Pinned production source trace.
10. Decision, limitations, rollback threshold, and reproduction instructions.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit the pinned MLPerf LoadGen trace from Lesson 0.6/Lab D with repository, revision, verification date, exact files and symbols, execution path, static-versus-executed status, commands, and `TODO_VERIFY` markers.

### Rubric Dimensions

A submission is **Competent** overall when every dimension is at least Competent. Strong is not required on any dimension to pass, and Strong on one dimension does not offset Insufficient on another.

| Dimension (evidence) | Insufficient | Competent | Strong |
|---|---|---|---|
| **Claim discipline** (D1) | Claims lack population, boundary, or baseline; a causal explanation is stated as fact | Atomic claims with O/D/H labels, scope, and a practical threshold set before measurement | Also states what each claim does *not* support and the observation that would falsify each H |
| **Measurement validity** (D2, D4; Lab A) | Only aggregates or only successes are available; clock or boundary undeclared; intended arrivals not recorded | Event-level raw data with declared clocks, boundaries, intended and actual issue, and all outcomes including failures | Demonstrates how a boundary or load-model change alters the metric and justifies the chosen one for the decision |
| **Design validity** (D3; Lab B) | Wrong or unstated experimental unit; fixed A-then-B order; no blocking | Correct assignment unit; randomized order within declared blocks; nuisance factors recorded | Carryover/order effect tested; replication chosen at the level the claim generalizes to |
| **Statistical reasoning** (D5–D6; Lab C) | Formula applied to the wrong unit (e.g., $n$ = requests when processes were assigned); tail claim from a mean interval | Effect size with an interval at the correct unit; assumptions stated; quantile convention and tail-exposure check reported | Sensitivity to exclusions/estimators shown; hierarchical or resampling analysis matches the data structure |
| **Diagnosis** (D7–D8; Incident 00.1) | One explanation asserted; no alternative could have been ruled out | At least three alternatives and one intervention that discriminates two of them; unexplained results retained | Hypotheses ranked by evidence; reanalysis under injected order bias or coordinated omission reproduces the predicted distortion |
| **Reproducibility** (D4, D9–D10; Lab D; required artifact) | Results cannot be regenerated; revision, data, or commands missing | Pinned code/data/config/environment, raw data, analysis code, exact commands, static-versus-executed status | A second person reruns from the manifest; `TODO_VERIFY` items and limitations are explicit |

*Calibration cases for reviewers*:
- A submission computes the t interval correctly but uses 50,000 requests from one process per configuration as $n$. It is **Insufficient** on Design validity and Statistical reasoning, because correct arithmetic on the wrong unit does not support the claim; other dimensions are graded on their own evidence.
- A submission contains only an aggregated summary log (means and percentiles, no per-request events). It is **Insufficient** on Measurement validity and Reproducibility, because no reviewer can recompute or re-slice the result, even if its design description is sound.
- A submission meets every Competent descriptor and no Strong descriptor. It passes as **Competent**.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Atomic O/D/H claims | 0.1 | 0.1 Guided Practice; Lab B pre-registration | Mastery D1 | Claim table with labels and threshold; rubric: Claim discipline |
| Measurement boundaries | 0.2 | 0.2 Guided Practice; Lab A; Lab D | Mastery D2; Incident steps 2–3 | Event schema, timestamp diagram, raw trace; rubric: Measurement validity |
| Randomized blocked design | 0.3 | 0.3 Independent Practice; Lab B | Mastery D3; Incident step 3 | Randomization record and paired differences; rubric: Design validity |
| Distribution and tail reasoning | 0.4 | 0.4 Guided Practice; Labs A and C | Mastery D5–D6; Incident step 4 | Raw outcomes, quantile convention, denominator audit; rubric: Statistical reasoning |
| Conditional uncertainty | 0.5 | 0.5 Guided Practice; Lab C | Mastery D5 | Interval at the correct unit with assumptions; rubric: Statistical reasoning |
| Artifact provenance and source trace | 0.6 | 0.6 Independent Practice; Lab D | Required source trace; Mastery D4, D9–D10 | Manifest, pinned LoadGen trace, rerun log; rubric: Reproducibility |
| Falsification diagnosis | 0.7 | 0.7 Guided Practice; Break steps of Labs A–D | Mastery D7–D8; Incident steps 1, 5–7 | Competing-hypothesis matrix and reanalysis; rubric: Diagnosis |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner can exit Module 00 when they can:

1. define what was measured and what decision population it represents;
2. identify the true experimental unit and major nuisance factors;
3. implement a harness that records intended arrivals, actual timing, outcomes, and provenance;
4. explain warmup, state drift, pseudoreplication, tail uncertainty, and coordinated omission;
5. calculate and correctly interpret a conditional mean interval and sample-size approximation;
6. preserve failures and anomalies instead of optimizing the dataset for a desired result;
7. falsify a favored explanation with a controlled intervention;
8. hand another engineer an artifact they can audit and rerun.

### Module Wrap-Up (Final Mental Model Reconstruction)

The final invariant is simple: **a precise number is not strong evidence unless the measurement contract, design, assumptions, and failure semantics make it answer the intended question.**

## 12 Competency Targets

```yaml
competency:
  sfia: 4-5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
