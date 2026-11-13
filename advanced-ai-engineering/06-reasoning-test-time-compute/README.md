# Module 06 — Reasoning & Test-Time Compute

## 00 Why This Module Exists

A model can spend inference-time work in several different ways: generate a longer trajectory, sample independent candidates, branch and backtrack, call tools, score final answers, or verify intermediate steps. These mechanisms do not buy the same thing. More generated tokens are not automatically more useful reasoning; an oracle finding that one of many candidates is correct is not a deployable selection result; and parallel work can increase total compute without increasing critical-path latency by the same factor.

This module teaches the complete system:

$$
\text{task} \to \text{candidate generation} \to \text{compute allocation}
\to \text{verification/selection} \to \text{stopping}
\to \text{answer + quality + cost + latency}.
$$

The feedback path matters. A policy that samples more candidates can increase accelerator demand, reduce serving headroom, lengthen queues, trigger timeouts, cancel unfinished branches, and change the candidate distribution seen by the verifier. Those effects are hypotheses until joined request, candidate, verifier, and serving telemetry support them.

The scope is inference-time reasoning, search, verification, budgeting, and diagnosis. Model-behavior uncertainty is developed in Module 07; training and model adaptation in Module 19; serving scheduling and capacity in Module 04; and inference-kernel optimization in Module 05.

**Research cutoff:** 2026-09-30. The sources in Section 05 were re-read on 2026-10-01 at the depth stated per entry. No exhaustive search for work published between 2026-09-27 and the cutoff was done; that gap is an open verification item, not a claim that nothing new exists.

**Module Orientation**

- **Engineering problem:** choose and operate a reasoning policy that improves declared task utility under quality, cost, latency, capacity, and safety constraints.
- **What you will do:** instrument chain-of-thought behavior; implement self-consistency and best-of-N; derive and break pass@k assumptions; build verifier-guided search; implement budget and stopping controls; test faithfulness and verifier gaming; and diagnose an offline-to-production regression.
- **Environment:** Python 3.10+ for sampling, search, scoring, and analysis; access to a model endpoint or local model that can return generation metadata; optional GPU access for loaded latency/capacity experiments. Pin model, runtime, tokenizer, prompt, judge, and pricing revisions.
- **Evidence rule:** label source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**). A paper result remains scoped to its model, tasks, decoding, verifier, budget, and evaluation boundary.
- **If You Only Have One Week**: Lessons 6.2, 6.4, 6.5 — they separate oracle pass@k from selected accuracy, validate verifiers beyond their own score, and account for compute and stopping; chain-of-thought semantics, search, and adaptive policies build on those three.

## 01 Baseline Assumptions

Each prerequisite below names the lesson where it is taught. Anything this module needs that those lessons do not teach is taught here at first use and is listed in the last bullet.

- **Sampling contract (Module 01, Lesson 1.2):** autoregressive factorization over token IDs; logits to a next-token distribution through softmax with temperature; greedy, top-k, and top-p with renormalization; and the sampling-contract fields (processor order, library defaults, seed, stop rules, context-limit policy). Two scope limits from that lesson carry over: the factorization gives the probability of a token-ID sequence, not of a text string, and reproducibility across batch composition or kernels is not guaranteed. Module 01 traces a different Transformers revision than this module's Section 05; treat them as two separate snapshots.
- **Forward-pass cost (Module 01, Lesson 1.6; Module 05, Lesson 5.5):** logical FLOP accounting per token, and the fact that model-call cost depends on implementation and workload. Token counts are not interchangeable with FLOPs, elapsed time, or money.
- **Serving metrics (Module 04, Lessons 4.1 and 4.4):** TTFT, ITL/TPOT, end-to-end latency, throughput, goodput, queueing, concurrency, and capacity headroom.
- **Evaluation discipline (Module 00):** the probability of at least one occurrence in independent trials (Lesson 0.4); experimental units, blocking, and pairing (Lesson 0.3); intervals and a minimum practically important effect (Lesson 0.5); pre-registered hypotheses in every lab.
- **General mathematics assumed:** conditional probability, complements, expected value, and explicit units.
- **Taught here at first use, not assumed:** the difference between a generated rationale and the model's internal computation (Lesson 6.1); what makes $k$ candidates independent draws, and how candidates become dependent (Lesson 6.2); combinations and the binomial distribution (Lesson 6.2).

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
  instruction: 4.75h      # lesson instruction lines: 35+60+40+50+55+45 min
  guided_practice: 2.75h  # lesson practice lines: 20+30+30+30+25+30 min
  labs: 18h               # LAB A 4h + LAB B 4h + LAB C 5h + LAB D 5h
  assessment: 3h          # Mastery transfer problem 2.5h + Incident 06.1 0.5h
  source_trace: 2h        # Section 09 Production Source Trace; LAB A attaches this artifact and does not count it again
  total: 30.5h
```
Each category is counted once. Lab analysis is not counted again as guided practice, and the source trace is not counted inside any lab.

By the end, the learner must be able to:

1. separate visible rationale, candidate generation, oracle opportunity, verifier selection, and user-visible correctness;
2. implement sequential, parallel, and search-based test-time strategies with complete cost accounting;
3. derive pass@k and majority-vote baselines and state exactly when their assumptions fail;
4. evaluate outcome and process verifiers for calibration, discrimination, shift, and exploitation;
5. design fixed and adaptive budget/stopping policies without assuming monotonic returns;
6. distinguish total executed work from critical-path latency and production capacity impact;
7. trace a real generation/stopping implementation at a pinned revision;
8. diagnose correlated errors, selection gaps, unfaithful rationales, verifier gaming, overthinking, and load-induced regressions.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
autoregressive generation
        |
        +--> visible chain of thought -----> faithfulness tests
        |
        +--> candidate set
               |-- parallel samples ------> vote / self-consistency
               |-- best-of-N -------------> verifier selection
               `-- tree expansion --------> value, prune, backtrack
                                      |
                                      v
                         outcome/process verifier
                                      |
                                      v
                           selected answer / abstain

budget controller: strategy + sequential limit + parallel width + stop policy
        |
        +--> total work: generator + verifier + tools + orchestration
        +--> critical path: wall-clock dependency chain
        `--> serving impact: queue, capacity, cancellations, SLO-goodput
```

Three measurement boundaries must remain separate:

- **Generator opportunity:** did the candidate set contain a correct answer?
- **Selector quality:** did the vote or verifier choose a correct candidate, conditional on the set?
- **System utility:** did the delivered answer meet application quality, cost, latency, capacity, and safety constraints?

---

## 04 Lessons

### Lesson 6.1 — Chain-of-Thought Is an Output Protocol, Not a Causal Proof

**Engineering Question:**
What does an intermediate rationale demonstrate, and what does it not demonstrate?

**Concepts & Definitions:**

- **Chain-of-thought prompting** elicits intermediate text before a final answer. Wei et al. report improvements on the models and arithmetic, commonsense, and symbolic tasks they evaluated (**O**, CLM-001). That observation motivates a workload-specific experiment; it does not prove universal benefit, an emergent capability threshold that transfers unchanged, or faithfulness of the printed rationale.
- **Rationale text versus internal computation.** A rationale is a sequence of generated tokens. Each token is drawn from a next-token distribution (Module 01, Lesson 1.2) produced by a forward pass whose intermediate activations are never printed. The rationale is therefore an *output* of the computation, and it also becomes *input* to later tokens, which condition on it. It is not a transcript of the computation. This module calls the unprinted activations the model's internal (latent) computation.
- **Faithfulness (as tested here):** when an input factor changes the answer, the rationale acknowledges that factor. This is an intervention-specific, observable property, not a claim about all internal causes.

**Mechanism Explanation:**
A reasoning-trace contract records:

- model and prompt revision, exemplars, formatting, and decoding parameters;
- whether the trace is user-visible, hidden, summarized, or discarded;
- how the final answer is parsed and scored;
- token, latency, and cost boundaries;
- interventions used to test whether the rationale acknowledges factors that change the answer.

Turpin et al. and later work on reasoning models provide counterexamples to universal faithfulness: answer-changing cues can be omitted from the explanation (**O**, CLM-007). Such experiments falsify “the rationale always reports the cause”; they do not establish that every rationale is false or reveal the model's complete hidden computation.

**Break cases:** misleading few-shot exemplars, spurious hints, answer-format leakage, persuasive but invalid steps, rationale truncation, and a task where direct answering is already reliable.

**Worked Example (synthetic counts):**
- *Input*: 200 multiple-choice items, each run twice with greedy decoding: once unchanged, once with a spurious cue pointing at a wrong option. The pair is the sampling unit. Pre-declared rules: the answer "follows the cue" if it changes from another option to the cued option; the rationale "acknowledges" the cue if it mentions the cue as a reason.
- *Observed*: the answer follows the cue in 50 pairs. In 8 of those 50, the rationale acknowledges the cue.
- *Steps*: acknowledgement rate among cue-following pairs $=8/50=16\%$. The cue changed the answer in $50/200=25\%$ of pairs. The fraction $8/200=4\%$ has a different denominator and answers a different question.
- *Result*: in 42 pairs the answer followed the cue while the rationale did not report it.
- *Interpretation and limits*: those 42 pairs are evidence against faithful reporting for this cue, model, prompt, and decoding. They are not a measurement of every hidden causal factor, and the 150 unchanged pairs say nothing about faithfulness. With sampling instead of greedy decoding, an answer can change between two runs without any cue, so a no-cue repeat is needed as a control.

**Knowledge Check:**
1. Why does a correct final answer not validate every intermediate step?
2. What observation would weaken the claim that a displayed rationale faithfully reports the answer-changing cue?

**Guided Practice:**
Define a paired intervention with an unchanged task and one controlled cue. Specify the answer-change event, acknowledgement rule, sampling unit, and exclusions before generation. Then, for 120 pairs in which the answer follows the cue in 30 and the rationale acknowledges it in 6 of those, report the acknowledgement rate with its denominator.

**Feedback Contract:**
- *Expected Evidence*: Paired raw outputs, exact prompt/model/decoding revisions, answer and acknowledgement labels, and a bounded conclusion. For the practice counts: $6/30=20\%$ among cue-following pairs (not $6/120=5\%$). Knowledge check: (1) a final answer can be right by a path the text does not describe, or despite an invalid step; (2) an answer that changes with the cue while the rationale omits the cue.
- *Common Failure*: Treating fluent rationale text or aggregate accuracy as causal evidence; dividing by all pairs instead of cue-following pairs; running the pair under sampling without a no-cue repeat.
- *Diagnostic Hint*: Which input factor changed, and was it acknowledged before the answer? What is the denominator of your rate?
- *Concept to Revisit*: Reasoning-Trace Contract; rationale text versus internal computation.

**Learning Outcome:**
Use visible reasoning as an instrumented output artifact, not privileged ground truth about internal causation.

*(Effort: 35m instruction, 20m practice)*

---

### Lesson 6.2 — Parallel Sampling, Self-Consistency, and pass@k

**Engineering Question:**
Does generating more candidates improve opportunity, selection, or both?

**Concepts & Definitions:**

- **Candidate**: one complete generation for an item under a stated sampling contract (Module 01, Lesson 1.2).
- **Self-consistency**: sample several reasoning paths, parse a final answer from each, and return the most frequent answer. Wang et al. define the aggregation as $\arg\max_a\sum_i\mathbb{1}(a_i=a)$ and call it a majority vote (**O**, CLM-002). Operationally this is a **plurality** vote: the winning answer needs the most votes, not more than half. *Reading the formula*: $a_i$ is the parsed answer of candidate $i$; $\mathbb{1}(a_i=a)$ is 1 if candidate $i$ gave answer $a$ and 0 otherwise; the sum therefore counts the votes for $a$, and $\arg\max_a$ returns the answer with the most votes. Example: answers $(7,9,9,7,9)$ give 2 votes for 7 and 3 for 9, so the vote returns 9. A production implementation must also define answer normalization, invalid parses, abstentions, tie-breaking, duplicate handling, and whether it uses plurality, weighted voting, or a separate verifier.
- **Independent draws (operational definition)**: $k$ candidates for one item are independent draws when each is generated from the same prompt, model, and processor chain with its own random stream: no shared seed, no reuse of a cached sampled response, no deduplication, and no candidate conditioned on an earlier candidate. Under that protocol, and conditional on the item, the correctness indicators are independent with one item-specific probability $p_i$.
- **Dependent candidates**: any case where that protocol does not hold or where the item is not fixed.
  - *Greedy decoding*: every candidate is the same sequence in exact arithmetic, so pass@k equals pass@1.
  - *Structured generation*: beam or diverse-beam candidates, revisions conditioned on an earlier answer, and candidates sharing a seed are dependent by construction.
  - *Across a population*: items differ in $p_i$. Two candidates for the same randomly chosen item are positively correlated, even when each item's draws are independent.
- **Oracle set success (pass@k)**: the event that at least one of $k$ candidates is correct, judged with ground truth. It measures opportunity.
- **Selected accuracy**: the event that the one answer the system returns is correct. It measures opportunity and selection together.
- **Strict majority** means more than $k/2$ candidates agree. Plurality can select an answer without a strict majority.
- **Combinations and the binomial distribution**: $\binom{n}{k}=\frac{n!}{k!(n-k)!}$ counts the size-$k$ subsets of $n$ things. If $k$ trials are independent with success probability $p$, the number of successes $J$ satisfies $P(J=j)=\binom{k}{j}p^j(1-p)^{k-j}$.
  - *Why*: one particular ordering with $j$ successes and $k-j$ failures has probability $p^j(1-p)^{k-j}$ (independence lets the per-trial probabilities multiply), and $\binom{k}{j}$ counts the orderings, i.e. the ways to choose which $j$ of the $k$ trials succeed.
  - *Example*: $\binom{5}{3}=\frac{5!}{3!\,2!}=\frac{120}{6\cdot2}=10$. With $k=3$, $p=0.4$: $P(J=2)=\binom{3}{2}(0.4)^2(0.6)=3\cdot0.16\cdot0.6=0.288$.
- **Notation used in this lesson**: $p$ (or $p_i$ for item $i$) is the probability that one candidate is correct; $k$ is the number of candidates considered; $n$ is the number of candidates actually observed for an item, and $c$ how many of those are correct; $\bar p$ is the average of $p_i$ over items; $\mathbb{E}_i[\cdot]$ averages over items. Lesson 6.3 reuses $k$ for thoughts per state, and Lesson 6.5 reuses $j$, $a$, $b$ with different meanings; each is redefined there.

**Quantitative Model / Derivation:**

*Model 1 — one item, independent draws.* For $k$ independent candidates with the same correctness probability $p$ (**D**, CLM-009):

$$
P(\text{at least one correct})=1-(1-p)^k.
$$

*Step-by-step*: (1) "at least one correct" is the complement of "all $k$ wrong"; (2) one candidate is wrong with probability $1-p$; (3) independence lets the $k$ wrong-probabilities multiply, giving $P(\text{all wrong})=(1-p)^k$; (4) subtract from 1. Example: $p=0.4$, $k=3$ gives $1-0.6^3=1-0.216=0.784$.

*Model 2 — a population of items.* With independent draws per item and item-specific $p_i$, population pass@k is the average of the per-item values, and it is bounded by the pooled formula (**D**, CLM-017):

$$
\mathbb{E}_i\bigl[1-(1-p_i)^k\bigr]\;\le\;1-(1-\bar p)^k,\qquad \bar p=\mathbb{E}_i[p_i],
$$

with equality only when all $p_i$ are equal (for $k\ge2$), because $(1-p)^k$ is convex in $p$. Plugging a pooled single-sample accuracy into Model 1 therefore *overstates* population pass@k whenever items differ in difficulty. Neither model applies when the draws themselves are dependent.

*Step-by-step*: (1) for a random item, pass@k is $1-(1-p_i)^k$, so the population value averages it: $1-\mathbb{E}_i[(1-p_i)^k]$; (2) $f(p)=(1-p)^k$ curves upward (convex; its second derivative $k(k-1)(1-p)^{k-2}$ is $\ge0$, and $>0$ for $k\ge2$, $p<1$); (3) Jensen's inequality for a convex $f$ gives $\mathbb{E}_i[f(p_i)]\ge f(\mathbb{E}_i[p_i])=(1-\bar p)^k$; (4) subtracting both sides from 1 flips the inequality. Example (Worked Example C below): $p_i\in\{0.8,0\}$ with equal weight, $k=3$: $\mathbb{E}_i[(1-p_i)^3]=0.5\cdot0.008+0.5\cdot1=0.504$, so population pass@3 $=0.496$, while $(1-0.4)^3=0.216$ gives the pooled $0.784$.

*Model 3 — finite-sample estimator.* Given $n$ observed candidates for an item, $c$ of them correct, the probability that a uniformly chosen subset of size $k\le n$ contains at least one correct candidate is (**D**, CLM-010):

$$
\widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
$$

taken as 1 when $n-c<k$.

*Step-by-step*: (1) there are $\binom{n}{k}$ equally likely size-$k$ subsets of the $n$ candidates; (2) a subset has no correct candidate only if all $k$ members come from the $n-c$ wrong ones, which can happen in $\binom{n-c}{k}$ ways; (3) the ratio is the probability of an all-wrong subset, and its complement is the estimate; (4) when $n-c<k$ no all-wrong subset exists, hence the value 1. Example: $n=5$, $c=2$, $k=3$: $\binom{3}{3}=1$ all-wrong subset out of $\binom{5}{3}=10$, so $\widehat{pass@3}=1-1/10=0.9$.

Chen et al. average this over problems, state that it is an unbiased estimate of per-problem $1-(1-p_i)^k$ when the $n$ samples are independent draws, and show that the plug-in $1-(1-c/n)^k$ underestimates it (**O**, CLM-018). It is an oracle set metric, not selected accuracy.

*Model 4 — strict majority of independent binary votes.* For odd $k$ (**D**, CLM-011):

$$
P(\text{majority correct})=
\sum_{j=(k+1)/2}^{k}\binom{k}{j}p^j(1-p)^{k-j}.
$$

*Step-by-step*: (1) treat each candidate as a binary vote, correct with probability $p$, so the number of correct votes $J$ is binomial; (2) for odd $k$, "more than $k/2$" means $J\ge(k+1)/2$ (odd $k$ also rules out a tie); (3) add the binomial probabilities of those $J$ values. Example: $k=5$, $p=0.4$ needs $J\in\{3,4,5\}$: $10(0.4)^3(0.6)^2+5(0.4)^4(0.6)+(0.4)^5=0.2304+0.0768+0.01024\approx0.317$, the value used in Worked Example B.

Real self-consistency uses multi-class plurality: wrong answers may split, one wrong answer may dominate through correlated error, and normalization can merge or fragment answers. Report at least single-sample accuracy, oracle pass@k, selected accuracy, parsing failure, answer entropy, duplicates, cost, and latency.

**Worked Example A — the IID baseline (analytical inputs).**
- *Input*: one item, independent draws, $p=0.4$, $k=3$.
- *Steps*: oracle opportunity $=1-0.6^3=0.784$. Strict-majority correctness $=3(0.4)^2(0.6)+(0.4)^3=0.352$ (Model 4 with $J\in\{2,3\}$: $0.288+0.064$).
- *Result and limits*: the gap shows why oracle pass@k cannot be reported as selected accuracy. These are values of a formula under its assumptions, not forecasts for model samples.

**Worked Example B — a candidate table (synthetic data).**
- *Input*: four items, $n=5$ sampled candidates each, answers already normalized, ground truth known. Selection rule: plurality over all five, ties counted as wrong.

| Item | Gold | Candidate answers | Correct $c$ | $\widehat{pass@3}$ | Plurality answer | Selected correct? |
|---|---|---|---|---|---|---|
| 1 | 18 | 18, 18, 18, 18, 17 | 4 | $1-\binom{1}{3}/\binom{5}{3}=1$ | 18 (4 votes) | yes |
| 2 | 7 | 7, 9, 9, 9, 7 | 2 | $1-\binom{3}{3}/\binom{5}{3}=0.9$ | 9 (3 votes) | no |
| 3 | 42 | 42, 40, 41, 42, 39 | 2 | $0.9$ | 42 (2 votes) | yes |
| 4 | 5 | 6, 6, 8, 6, 6 | 0 | $1-\binom{5}{3}/\binom{5}{3}=0$ | 6 (4 votes) | no |

- *Steps*:
  - Single-sample accuracy $=(4+2+2+0)/20=0.40$ (correct candidates over all $4\times5=20$ candidates).
  - Finite-sample pass@3 $=(1+0.9+0.9+0)/4=0.70$.
  - Oracle set success over all five candidates $=3/4=0.75$ (items 1–3 contain a correct candidate).
  - Plurality-selected accuracy $=2/4=0.50$. Strict-majority correctness $=1/4=0.25$ (item 1 only).
  - Oracle–selection gap at $k=5$: $0.75-0.50=0.25$, all of it from item 2.
- *Comparison with formulas*: the pooled IID formula gives $1-0.6^3=0.784$, above the estimated $0.70$. That is the direction Model 2 gives for items of unequal difficulty; with four items it is an illustration, not a test. The per-item plug-in $1-(1-c/n)^3$ (e.g. item 1: $1-(1-4/5)^3=1-0.2^3=0.992$) averages $(0.992+0.784+0.784+0)/4=0.64$, below $0.70$, the direction Chen et al. report. The IID strict-majority formula at $p=0.4$, $k=5$ gives $0.317$, which matches neither the observed plurality accuracy ($0.50$) nor the observed strict-majority rate ($0.25$).
- *Result*: item 2 had a correct candidate that plurality did not select (a repeated wrong answer). Item 3 was selected correctly with only two of five votes because the wrong answers split. Item 4 had nothing to select.
- *Interpretation and limits*: four synthetic items show the accounting, not an effect size. On real data, report uncertainty across items and keep the item as the unit of analysis.

**Worked Example C — why independent sampling is not the same as independent correctness (synthetic mixture).**
- *Input*: half of the items have $p_i=0.8$ and half have $p_i=0$; draws are independent within each item; $k=3$.
- *Steps*: pooled $\bar p=0.4$. Population pass@3 $=0.5\,(1-0.2^3)+0.5\cdot0=0.496$. The pooled IID formula gives $0.784$. The correlation between the correctness of two candidates for the same random item is $(\mathbb{E}[p_i^2]-\bar p^2)/(\bar p(1-\bar p))=(0.32-0.16)/0.24\approx0.67$.
  - *Where the correlation comes from*: let $X_1,X_2$ be the 0/1 correctness of two candidates for the same random item. Each has mean $\bar p$ and variance $\bar p(1-\bar p)$. Given the item they are independent, so $\mathbb{E}[X_1X_2]=\mathbb{E}_i[p_i\cdot p_i]=\mathbb{E}[p_i^2]$. Covariance $=\mathbb{E}[X_1X_2]-\bar p^2$; dividing by the common variance gives the correlation. Here $\mathbb{E}[p_i^2]=0.5\cdot0.8^2+0.5\cdot0^2=0.32$, $\bar p^2=0.16$, $\bar p(1-\bar p)=0.4\cdot0.6=0.24$. If all $p_i$ were equal, $\mathbb{E}[p_i^2]=\bar p^2$ and the correlation would be 0.
- *Result*: the same single-sample accuracy as Example A yields a pass@3 of $0.496$ instead of $0.784$.
- *Interpretation and limits*: nothing was wrong with the sampler. The dependence comes from the population, and more samples cannot help the items with $p_i=0$. If the draws themselves were fully dependent, so that the three candidates are identical, pass@3 would equal that decoder's pass@1. Under greedy decoding that is the greedy accuracy, which need not equal the sampled single-sample accuracy of $0.4$.

**Knowledge Check:**
1. Which assumptions permit $1-(1-p)^k$, and which of them fails in Worked Example C?
2. Can a verifier improve selected correctness if the fixed candidate set has no correct answer?
3. Why is item 3 in Worked Example B selected correctly without a strict majority?

**Guided Practice:**
Three items, $n=4$ candidates each, gold answer `x` in every item: item A answers `x, x, y, x`; item B answers `y, y, x, z`; item C answers `y, z, w, y`. Compute single-sample accuracy, finite-sample pass@2, plurality-selected accuracy over the four candidates, oracle set success, and the oracle–selection gap. Compare pass@2 with the pooled IID value and explain the difference. Then repeat on your own candidate table and inspect duplicated wrong answers.

**Feedback Contract:**
- *Expected Evidence*: for the three-item table: single-sample accuracy $4/12\approx0.333$; pass@2 per item $1$, $0.5$, $0$, mean $0.5$ (e.g. item B: $c=1$, $1-\binom{3}{2}/\binom{4}{2}=1-3/6$); plurality accuracy $1/3$ (only item A); oracle set success $2/3$; gap $1/3$, from item B; pooled IID value $1-(2/3)^2\approx0.556$, higher than $0.5$ because the items differ in difficulty. For your own table: candidate-level labels and seeds, answer normalization, tie policy, all formulas with assumptions, and uncertainty across items. Knowledge check: (1) independent draws and one common $p$; the common $p$ fails; (2) no; (3) the wrong answers split, so two votes are a plurality.
- *Common Failure*: Calling oracle opportunity a production selection result; pooling candidates across items before applying the IID formula; treating $k$ low-temperature near-duplicates as $k$ independent attempts.
- *Diagnostic Hint*: Did the correct answer exist in the set, and if so, why was it not selected? Would your pass@k change if you computed it per item and then averaged?
- *Concept to Revisit*: Opportunity Versus Selection; independent draws versus dependent candidates.

**Learning Outcome:**
Determine whether extra sampling improves the candidate pool, the selector, or neither, and state which independence assumption each formula needs.

*(Effort: 60m instruction, 30m practice)*

---

### Lesson 6.3 — Search Over Reasoning States

**Engineering Question:**
When is explicit branching and backtracking better than independent full answers?

**Concepts & Definitions:**

- **State**: the input plus the sequence of thoughts so far; a node of the search tree.
- **Thought**: one intermediate step that extends a state. Its size is a design choice: large enough to evaluate, small enough to generate diverse alternatives.
- **Thought generator**: produces $k$ candidate next thoughts for a state, either by $k$ independent samples or by one "propose" call that lists $k$ alternatives.
- **State evaluator**: scores states, either one at a time (a value) or by comparing a set (a vote). It is a heuristic, not ground truth.
- **Search algorithm**: decides which states to keep and expand. Breadth-first search keeps the $b$ best states per step for $T$ steps; depth-first search expands the most promising state and backtracks when a state scores below a threshold.
- **Pruning** discards a state; **backtracking** returns to an earlier state after a dead end.
- **Four separate budgets**: nodes represented, model calls, tokens, and wall-clock time.

**Mechanism Explanation:**
Tree of Thoughts provides a reference mechanism with exactly these four parts: thought decomposition, a thought generator, a state evaluator, and a search algorithm (**O**, CLM-003). The design space also includes state representation, terminal tests, duplicate-state detection, and call scheduling.

Search helps only when three conditions approximately hold:

1. intermediate states capture choices that affect the solution;
2. proposal diversity reaches useful alternatives;
3. the evaluator discriminates promising from misleading states early enough to repay search cost.

The paper's results on Game of 24, Creative Writing, and Mini Crosswords establish scoped observations, not a general law. On tasks without a meaningful partial-state value, search may multiply calls, prune the correct branch, or optimize a proxy.

**Implementation record per node:** parent, depth, state text or structured state, generator configuration, score and scorer revision, expansion timestamp, token/cost ledger, prune reason, terminal result, and independent correctness if available.

**Worked Example (call accounting for one declared configuration):**
- *Input*: breadth-first search with $k=3$ candidate thoughts per kept state, breadth limit $b=2$, $T=3$ steps, one evaluator call per candidate state, no duplicate elimination.
- *Step 1 — states generated*: step 1 expands the root: $1\times3=3$. Steps 2 and 3 each expand two kept states: $2\times3=6$. Total generated states $=3+6+6=15$; kept states $=2+2+2=6$.
- *Step 2 — evaluator calls*: one per generated state $=15$.
- *Step 3 — generator calls, sample strategy*: one call per candidate $=15$, so $15+15=30$ model calls.
- *Step 4 — generator calls, propose strategy*: one call per expanded state $=1+2+2=5$, so $5+15=20$ model calls.
- *General form (same assumptions, $b\le k$, $T\ge1$)*: generated states $=k+(T-1)\,b\,k$ (the root expansion, then $b$ kept states expanded per later step); kept states $=bT$; evaluator calls = generated states; sample-strategy generator calls = generated states; propose-strategy generator calls = expanded states $=1+(T-1)\,b$. Check: $3+2\cdot2\cdot3=15$ and $1+2\cdot2=5$. Here $k$ counts thoughts per state, not final-answer candidates as in Lesson 6.2.
- *Contrast*: an unpruned binary tree of depth 3 has $1+2+4+8=15$ nodes. The same number 15 appears, but it counts something else, and neither count is a token count or a latency.
- *Result*: the same search shape costs 30 or 20 model calls depending on the generator strategy, before any difference in tokens per call.
- *Interpretation and limits*: these are counts under the declared configuration. Batching, repeated evaluator samples, early termination, and duplicate elimination change them, so record executed calls and tokens rather than inferring cost from node count.

**Knowledge Check:**
1. When can a poor state evaluator make wider search worse?
2. Why are node count, model calls, tokens, and wall-clock latency different budgets?

**Independent Practice:**
Implement one breadth/depth policy and one best-first policy under the same total-call cap. Log every proposed, pruned, duplicated, and terminal state. Before running, compute the generated-state and model-call counts for $k=4$, $b=3$, $T=2$ with one evaluator call per state and the sample strategy.

**Feedback Contract:**
- *Expected Evidence*: for the pre-run count: generated states $4+12=16$ ($k+(T-1)bk=4+1\cdot3\cdot4$), evaluator calls $16$, generator calls $16$, total $32$ model calls. For the implementation: search graph, actual work ledger, evaluator revision, terminal correctness, and a matched independent-sampling baseline at the same call cap. Knowledge check: (1) when the evaluator ranks a wrong state above the correct one, more candidates give it more chances to keep the wrong state and prune the right one; (2) a node can cost zero, one, or several calls; calls differ in tokens; calls can run in parallel.
- *Common Failure*: Comparing strategies at different budgets, hiding pruned correct branches, or reporting node count as cost.
- *Diagnostic Hint*: Did proposal fail, or did the evaluator prune the useful state? Does your ledger's call count equal the count you predicted from $k$, $b$, and $T$?
- *Concept to Revisit*: Proposal–Evaluation–Budget Separation.

**Learning Outcome:**
Treat search as proposal plus value estimation plus resource control, not as a prompt slogan.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 6.4 — Outcome, Process, and Selection Verifiers

**Engineering Question:**
What exactly is being verified, and against which truth?

**Concepts & Definitions:**

- **Outcome verifier:** scores the final answer or product.
- **Process verifier/reward model:** scores intermediate steps or transitions.
- **Objective judge:** executes a test, checks a proof, or compares with ground truth under declared rules.
- **Learned or model-based judge:** estimates correctness or preference and can be wrong, shifted, or exploited.

**Mechanism Explanation:**
Lightman et al. compare outcome and process supervision and report a process-supervision advantage in their MATH setup while releasing PRM800K (**O**, CLM-004). This supports studying step-level feedback; it does not establish universal verifier superiority or transfer.

Separate (**D**, CLM-013):

$$
\text{oracle set success} \geq \text{selected success}
$$

when both operate on the same candidate set and ground-truth correctness is well-defined. *Why*: for each item, "selected answer correct" implies "the set contains a correct candidate", so the selected-success count can never exceed the oracle count. The difference, oracle minus selected, is the selection gap. A verifier cannot select a correct item absent from that fixed set, and can fail despite one being present. If verifier-guided search changes the candidate set, the comparison no longer isolates selector quality.

Verifier validation includes score semantics, calibration, ranking/discrimination, false positives and negatives, abstention, subgroup performance, temporal and domain shift, adversarial candidates, and score-quality behavior as search pressure grows. A rising verifier score without rising independent correctness is evidence consistent with gaming, but evaluator noise and distribution shift remain competing explanations (**H**, CLM-016).

**Worked Example (synthetic counts):**
- *Input*: 100 items, each with a fixed set of 4 candidates generated before the verifier is consulted. Ground-truth labels are independent of the verifier. The verifier returns exactly one candidate per item and never abstains.
- *Item level*: 60 sets contain at least one correct candidate. The verifier's selected candidate is correct in 45 items.
- *Steps*: oracle set success $=60/100=0.60$. Selected success $=45/100=0.45$. Selection accuracy given opportunity $=45/60=0.75$ (selected-correct items divided by items whose set contained a correct candidate; equivalently $0.45/0.60$). Gap $=0.60-0.45=0.15$.
- *Candidate level (same data, different unit)*: of the 400 candidates, 90 are correct. Used as an accept/reject classifier at a fixed score threshold, the verifier accepts 120 candidates, 72 of them correct. Precision $=72/120=0.60$; recall $=72/90=0.80$; false-positive rate $=48/310\approx0.155$.
  - *Confusion counts behind these rates*: true positives $TP=72$ (accepted and correct); false positives $FP=120-72=48$ (accepted but wrong); false negatives $FN=90-72=18$ (rejected but correct); wrong candidates $=400-90=310$, so true negatives $TN=310-48=262$. Precision $=TP/(TP+FP)$, recall $=TP/(TP+FN)$, false-positive rate $=FP/(FP+TN)$.
- *Result*: 15 of 100 items were lost in selection although a correct candidate was present, and 40 items could not be won by any selector.
- *Interpretation and limits*: the 0.15 gap is attributable to selection only because labels, set membership, and evaluation rules are identical in both numbers. Candidate-level precision and recall do not determine item-level selected success: the item-level number depends on how false positives are distributed across items. If the verifier had guided generation, the candidate sets would differ and this decomposition would not apply. With abstention, report abstentions separately instead of counting them as wrong or dropping them.

**Knowledge Check:**
1. What additional evidence distinguishes verifier gaming from a noisy independent judge?
2. Why can process scores fail even when final-answer scoring is reliable?

**Guided Practice:**
(a) For 50 items with fixed candidate sets, 35 sets contain a correct candidate and the verifier selects a correct one in 28. Compute oracle set success, selected success, selection accuracy given opportunity, and the gap. (b) Create a confusion table for verifier accept/reject versus independent correctness, stratified by candidate source and search depth. Audit the highest-scoring false positives.

**Feedback Contract:**
- *Expected Evidence*: (a) $0.70$, $0.56$, $28/35=0.80$, and $0.14$. (b) Frozen verifier and independent-label revisions, calibration/ranking results, adversarial false positives, and abstention handling. Knowledge check: (1) a score–correctness gap that widens with search pressure on held-out, independently judged items, with repeated exploitable features in selected outputs, points to gaming; a gap that is flat across search pressure points to noise; (2) step labels can encode an annotation policy, and a chain can reach a right answer through a step the process verifier rejects, or a wrong one through steps it accepts.
- *Common Failure*: Validating a verifier against its own labels or score; reporting candidate-level precision as selected accuracy.
- *Diagnostic Hint*: What truth source is independent of the selection mechanism? Is your unit the item or the candidate?
- *Concept to Revisit*: Verifier Meta-Evaluation; Opportunity Versus Selection (Lesson 6.2).

**Learning Outcome:**
Build verification as a measured subsystem rather than equating its score with truth.

*(Effort: 50m instruction, 30m practice)*

---

### Lesson 6.5 — Compute Ledgers, Budgets, and Stopping

**Engineering Question:**
How much work was executed, how long did the user wait, and was another unit of work worth doing?

**Concepts & Definitions:**

- **Total work**: the sum of all executed work, including cancelled and pruned branches, in one declared unit.
- **Critical path**: the longest chain of dependent steps from request arrival to the delivered answer, given the parallel resources actually available. It determines wall-clock latency.
- **Budget**: a declared cap on a resource such as new tokens, elapsed time, samples, branches, depth, or cost.
- **Stopping policy**: the rule that ends generation or sampling. A *resource stop* fires on a cap. An *exact early stop* fires when further work cannot change the selected answer. A *heuristic stop* fires on an estimate that further work is not worth its cost.
- **Marginal utility**: the change in measured task utility from one more unit of budget.
- **Cancellation**: stopping a branch already in flight; its executed work still counts.
- **Quality is not token count**: more generated tokens are more work. Whether they are more useful is an empirical question, and utility can fall as budget rises.

**Quantitative Model / Derivation:**
Let every executed generator, verifier, tool, and orchestration action be converted to a declared unit (**D**, CLM-012):

$$
C_{total}=C_{generator}+C_{verifier}+C_{tools}+C_{orchestration}.
$$

where each $C_x$ is the work of that component, summed over every executed action (including cancelled and pruned ones) and already converted to the one declared unit. This $C$ (cost) is unrelated to $c$, the count of correct candidates in Lesson 6.2.

Possible units include GPU-seconds on a pinned device, accelerator FLOPs with a stated estimator, tokens separated by model class, or monetary cost under a pinned price contract. Tokens from different models and sequence positions are not compute-equivalent. Total work is additive after valid conversion; wall-clock latency follows the dependency critical path under available parallel resources.

A fixed policy may cap new tokens, elapsed time, samples, branches, depth, or cost. An adaptive policy estimates whether marginal expected value justifies marginal cost. One useful decision model is:

$$
U(B)=Q(B)-\lambda C(B),
$$

where $U$ is net utility, $B$ is a declared budget, $Q$ measured task utility, $C$ measured cost, and $\lambda$ a stakeholder trade-off in compatible units. Hard latency, safety, or subgroup constraints may not be reducible to this scalar.

*Reading the trade-off*: $\lambda$ converts cost into utility units, e.g. "accuracy points given up per thousand tokens". Moving from budget $B_1$ to $B_2$ changes utility by $\Delta U=\Delta Q-\lambda\,\Delta C$. With $C(B)=B$ this is worth doing only if the marginal quality per unit of budget, $\Delta Q/\Delta B$, exceeds $\lambda$. Worked Example B below applies this rule.

Snell et al. provide evidence that effective allocation varies with prompt difficulty in their evaluated setting (**O**, CLM-005). The s1 paper demonstrates one sequential control—truncating thinking or appending “Wait” at an attempted stop—for its trained model and competition-math experiments (**O**, CLM-006). A 2026 preprint that uses the same budget-forcing control on open-weight models reports diminishing returns and cases of abandoning correct answers on mathematical and scientific tasks (**O**, CLM-008); treat this as frontier evidence to reproduce, not a universal overthinking law.

**An exact early stop for plurality voting** (**D**, CLM-019). With at most $k_{max}$ candidates and $j$ already sampled, let the leading answer have $a$ votes and the runner-up $b$. If $a-b>k_{max}-j$, the remaining candidates cannot change the plurality winner, so stopping returns the same answer as the full vote under the same tie rule. (Here $j$, $a$, $b$ are vote counts, not the binomial $j$ of Lesson 6.2 or the search breadth $b$ of Lesson 6.3.) *Why*: $k_{max}-j$ candidates remain. In the worst case for the leader, every one of them votes for one rival answer; since every rival starts with at most $b$ votes, no rival can finish above $b+(k_{max}-j)$. The strict inequality $a>b+(k_{max}-j)$ means even this worst case leaves the leader strictly ahead, so no tie can arise. This rule saves work without changing selection. It does not make the selected answer correct.

**Runtime stop primitives are resource stops.** At the pinned Transformers revision in Section 05, `GenerationMixin._prepare_generated_length` sets `max_length` to the prompt length plus `max_new_tokens`; `_get_stopping_criteria` adds a length criterion, a time criterion when `max_time` is set, a stop-string criterion when `stop_strings` is set, and an EOS criterion when an EOS token is configured; and `StoppingCriteriaList.__call__` ORs their per-row results (**O**, CLM-014). None of these inspects whether the answer is good. A semantic or marginal-value stop is policy code that the application adds on top.

**Stopping policy requirements:** global and per-branch limits, clock definition, cancellation semantics, completed-work accounting, minimum answer reserve, fallback, timeout behavior, and post-stop parsing. Test both premature stopping and excess continuation.

**Worked Example A — total work versus critical path (synthetic times).**
- *Input*: four candidates with generation times $2.0$, $2.4$, $3.1$, and $2.7$ s; one verifier pass over all four takes $0.5$ s and starts after all four finish; orchestration adds $0.1$ s before the verifier. Unit of work: accelerator-seconds, one accelerator-second per second of generation or verification. Orchestration runs on the host and is recorded separately.
- *Step 1 — total work*: generator $2.0+2.4+3.1+2.7=10.2$; verifier $0.5$; total $10.7$ accelerator-seconds, plus $0.1$ s of host time.
- *Step 2 — critical path with four free workers*: $\max(2.0,2.4,3.1,2.7)+0.1+0.5=3.7$ s. All four run at once, so the join waits for the slowest branch; orchestration and the verifier then run in sequence after it.
- *Step 3 — critical path with two workers*, schedule (2.0 then 3.1) and (2.4 then 2.7): both workers finish at $5.1$ s ($2.0+3.1$ and $2.4+2.7$), so the path is $5.1+0.1+0.5=5.7$ s. Total work is unchanged at $10.7$.
- *Result*: four-way parallelism multiplied work by about four relative to one candidate and left latency near the slowest branch, but only while four workers were free.
- *Interpretation and limits*: the schedule assumes no queueing and no slowdown from sharing a device. Under batching on one accelerator, per-branch times change with batch size (Module 04), so both numbers must be measured rather than derived from width.

**Worked Example B — budget, utility, and stopping (synthetic sweep).**
- *Input*: thinking budget $B$ in thousands of tokens, measured accuracy $Q(B)$ as a fraction, cost $C(B)=B$, and $\lambda=0.02$ of accuracy (two percentage points) per thousand tokens.

| $B$ (k tokens) | $Q(B)$ | $U=Q-\lambda B$ | Marginal accuracy per extra k tokens |
|---|---|---|---|
| 1 | 0.60 | 0.58 | — |
| 2 | 0.68 | 0.64 | $+0.080$ |
| 4 | 0.71 | 0.63 | $+0.015$ |
| 8 | 0.70 | 0.54 | $-0.0025$ |

- *How the columns are computed*: $U=Q-0.02B$, e.g. at $B=2$: $0.68-0.02\cdot2=0.64$. Marginal accuracy $=\Delta Q/\Delta B$ between consecutive rows, e.g. from 2 to 4: $(0.71-0.68)/(4-2)=0.015$; from 4 to 8: $(0.70-0.71)/4=-0.0025$.
- *Steps*: accuracy peaks at $B=4$; utility peaks at $B=2$; from 4 to 8 the marginal accuracy is negative. Utility stops rising at $B=2$ because the next step's marginal accuracy ($0.015$) is below $\lambda=0.02$, the price of one more thousand tokens.
- *Result*: the accuracy-maximizing budget, the utility-maximizing budget, and the largest budget are three different settings. Doubling tokens from 4 to 8 lowered accuracy in this sweep.
- *Exact early stop*: with $k_{max}=5$ and the first three candidates agreeing ($j=3$, $a=3$, $b=0$), $a-b=3>5-3=2$, so sampling stops and two candidates are saved with the same selected answer.
- *Interpretation and limits*: the sweep is synthetic. A real sweep needs uncertainty on each $Q(B)$, the same items at every budget, and per-item correctness transitions to separate gains from losses. $\lambda$ is a stakeholder choice; a hard latency or safety constraint may override $U$.

**Knowledge Check:**
1. Why are tokens from two model classes not automatically additive compute units?
2. What must be measured before claiming an adaptive stop has positive marginal utility?
3. In Worked Example A, why does total work stay at 10.7 while the critical path changes from 3.7 s to 5.7 s?

**Guided Practice:**
(a) Three branches take $1.5$, $2.2$, and $1.9$ s; a verifier pass takes $0.4$ s after the join; orchestration adds $0.05$ s. Compute total accelerator-seconds, and the critical path with three workers and with one worker. (b) Build a ledger for one sequential and one parallel policy of your own. Reconcile generator, verifier, tool, cancelled, and orchestration work against billing/runtime counters and a dependency timeline.

**Feedback Contract:**
- *Expected Evidence*: (a) $5.6+0.4=6.0$ accelerator-seconds (generator $1.5+2.2+1.9=5.6$; orchestration $0.05$ s is host time, recorded separately as in Worked Example A); $2.2+0.05+0.4=2.65$ s with three workers; $5.6+0.05+0.4=6.05$ s with one worker. (b) Compatible cost units, a complete executed-work ledger including cancelled work, cancellation semantics, and a critical-path trace. Knowledge check: (1) cost per token differs by model size, sequence position, and batch shape; (2) per-item utility with and without the extra work on the same items, its cost, and uncertainty; (3) work is a sum over executed branches, while the path depends on how many branches run at once.
- *Common Failure*: Counting only delivered tokens, equating parallel width with a latency multiplier, or reading a longer rationale as a better one.
- *Diagnostic Hint*: Which work executed, and which work lay on the dependency path? At which budget did accuracy stop rising?
- *Concept to Revisit*: Total Work Versus Critical Path; stopping policy types.

**Learning Outcome:**
Optimize measured utility across a full compute and critical-path ledger, not rationale length.

*(Effort: 55m instruction, 25m practice)*

---

### Lesson 6.6 — Adaptive Policies and Causal Diagnosis

**Engineering Question:**
How do we decide which requests deserve which strategy without leaking answers or masking failures?

**Concepts & Definitions:**

- **Router**: a policy that maps a request to a strategy and a budget before the answer is known.
- **Decision-time feature**: an input available to the router when it must decide. A feature computed from the gold label, from the outcome of the strategy being chosen, or from future verifier scores is not available at decision time.
- **Leakage**: use of such unavailable information, which makes an offline policy look better than a deployable one.
- **Equal-budget comparison**: two policies evaluated on the same items with the same aggregate executed work, so a quality difference cannot be explained by a difference in spend.
- **Replay**: re-running a pinned population through alternative policies to compare them on identical items.

**Mechanism Explanation:**
A router can choose direct response, bounded sequential reasoning, parallel sampling, or search. Difficulty is latent. A proxy derived from benchmark labels, future verifier outcomes, or target answers creates leakage; one calibrated on one model/domain may fail after a prompt, model, or population change. Snell et al. make the distinction explicit: their "oracle difficulty" bins come from ground-truth correctness of many samples per question, which they note is unavailable at deployment, and their "model-predicted difficulty" replaces it with a learned verifier's scores at additional inference cost (**O**, CLM-005).

That a calibrated, leakage-free adaptive policy beats uniform allocation at equal cost is a hypothesis to test on each workload, not a result to assume (**H**, CLM-015). Evaluate the policy at equal aggregate resource budget and report:

- routing features and their availability time;
- predicted difficulty or marginal value and calibration;
- allocation by task and subgroup;
- single-sample, oracle, selected, and user-visible utility;
- total work, critical-path latency, serving capacity, cancellations, and SLO-goodput;
- temporal and out-of-domain holdouts;
- a uniform-budget and direct-answer baseline.

Use the diagnostic loop:

$$
\text{symptom}\to\text{competing hypotheses}\to\text{missing evidence}
\to\text{discriminating measurement}\to\text{ranked explanation}
\to\text{intervention}\to\text{remeasurement}.
$$

Necessary but insufficient signals include longer rationales, higher verifier score, higher oracle pass@k, more answer diversity, and higher accelerator utilization. Each can coexist with worse selected correctness or production utility.

**Worked Example (synthetic replay):**
- *Input*: 100 requests, 60 routed "easy" and 40 routed "hard" by a decision-time feature. One candidate costs one unit. Measured selected accuracy by route and width: easy $0.90$ at $k=1$ and $0.92$ at $k=3$; hard $0.40$ at $k=3$, $0.55$ at $k=6$, and $0.58$ at $k=8$.
- *Rule used below*: budget $=\sum_{\text{routes}}(\text{requests on route})\times k_{\text{route}}$; overall accuracy $=\sum_{\text{routes}}(\text{share of requests})\times(\text{route accuracy at its }k)$, with shares $60/100=0.6$ and $40/100=0.4$.
- *Uniform policy*: $k=3$ for all. Budget $=100\times3=300$ units. Accuracy $=0.6\cdot0.92+0.4\cdot0.40=0.712$.
- *Adaptive policy at equal budget*: easy $k=1$, hard $k=6$. Budget $=60\cdot1+40\cdot6=300$ units. Accuracy $=0.6\cdot0.90+0.4\cdot0.55=0.760$.
- *Adaptive policy at unequal budget*: easy $k=1$, hard $k=8$. Budget $=60+320=380$ units, $27\%$ more ($80/300\approx0.267$). Accuracy $=0.6\cdot0.90+0.4\cdot0.58=0.772$.
- *Result*: at equal budget the adaptive policy gains $0.048$ ($0.760-0.712$; the third policy's $0.060$ is $0.772-0.712$). The $0.060$ gain of the third policy cannot be credited to routing, because it also spent more.
- *Interpretation and limits*: the gain is a point estimate on synthetic numbers and needs an interval on real data. The easy route lost $0.02$, so a subgroup constraint on that route could reject the policy. If the "hard" label had been derived from whether direct answering failed on these same items, the comparison would be leaked. Equal units of candidates are not equal accelerator-seconds or equal latency; the loaded test checks those.

**Knowledge Check:**
1. Which routing features are unavailable at decision time and therefore leak outcomes?
2. Why can an offline gain disappear under loaded serving?

**Guided Practice:**
(a) 100 requests split 50 easy and 50 hard. Uniform policy: $k=2$ for all, with accuracy $0.88$ on easy and $0.30$ on hard. Adaptive policy: easy $k=1$ ($0.85$), hard $k=3$ ($0.42$). Check that the budgets match and compute both accuracies. (b) Predeclare an equal-cost replay, a shifted holdout, and a loaded test. Produce a routing confusion matrix and quality/cost/SLO-goodput by route and subgroup.

**Feedback Contract:**
- *Expected Evidence*: (a) both budgets are 200 units ($100\cdot2$ and $50\cdot1+50\cdot3$); uniform $0.5\cdot0.88+0.5\cdot0.30=0.59$, adaptive $0.5\cdot0.85+0.5\cdot0.42=0.635$, difference $0.045$ at equal budget. (b) Decision-time feature manifest, matched budgets, temporal holdout, capacity telemetry, and rollback thresholds. Knowledge check: (1) gold labels, the chosen strategy's own outcome, verifier scores computed after generation, and benchmark difficulty tags derived from correctness; (2) extra candidates raise accelerator demand, which can lengthen queues, trigger timeouts and cancellations, and lower SLO-goodput.
- *Common Failure*: Attributing an unequal-budget gain to adaptivity; reporting only the aggregate and missing a route that got worse.
- *Diagnostic Hint*: Freeze aggregate work and remove every feature unavailable before routing. Does the gain survive?
- *Concept to Revisit*: Causal Evaluation of Routing Policies; equal-budget comparison.

**Learning Outcome:**
Deploy adaptive compute as a falsifiable control policy with quality and operational feedback.

*(Effort: 45m instruction, 30m practice)*

## 05 Literature & Production Source Map

Each entry states what was read on 2026-10-01 and what the module uses it for. "Abstract only" means the method and results sections were not re-read in this pass; such entries support only the scoped statement beside them. No reported benchmark number is used as a general result.

**REFERENCE / BASELINE**

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Wei et al. (arXiv v6, 2023). *Read*: abstract only. *Scope*: the prompting method and the task families it reports on (CLM-001); nothing about faithfulness.
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) — Wang et al. (ICLR 2023; arXiv v4). *Read*: Section 2 (sampling from the decoder, answer marginalization, the vote definition) and Table 1. *Scope*: the procedure and the plurality-vote definition in Lesson 6.2 (CLM-002).
- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Yao et al. (arXiv v2, 2023). *Read*: Section 3 (thought decomposition, generator, state evaluator, search) with Algorithms 1 and 2. *Scope*: the four-part mechanism and the $k$, $b$, $T$ parameters in Lesson 6.3 (CLM-003). Its task results were not re-read.
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al. (arXiv v1, 2023). *Read*: abstract only. *Scope*: the outcome/process distinction and the existence of PRM800K (CLM-004).
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) — Chen et al. (arXiv v2, 2021). *Read*: Section 2.1 (pass@k estimator, Eq. 1, Figure 3) and Appendix A (unbiasedness; bias of the plug-in estimator). *Scope*: the finite-sample estimator in Lesson 6.2 (CLM-010, CLM-018).
- [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) — Turpin et al. (arXiv v2, 2023). *Read*: abstract only. *Scope*: biasing features can change answers without being mentioned in the rationale (CLM-007).

**CURRENT DEFAULT — scoped to one pinned runtime**

- At the pinned Transformers revision below, generation offers length, time, stop-string, EOS, and custom stopping criteria (CLM-014). This is one runtime snapshot. It does not establish what other runtimes or hosted APIs provide: `TODO_VERIFY` per runtime before generalizing. These are resource/protocol stops, not semantic proof that sufficient reasoning occurred.
- No particular self-consistency width or reasoning-token budget is claimed as a default anywhere.

**RECOMMENDED BASELINE — a requirement of this module, not an observed industry default**

- Candidate-level telemetry, explicit answer parsing, and a declared tie and abstention policy for any multi-sample policy.

**WORKLOAD-DEPENDENT**

- Chain-of-thought prompting, self-consistency, best-of-N, verifier ranking, and tree search.
- Outcome versus process verification.
- Sequential versus parallel compute and fixed versus adaptive budgets.

**FRONTIER**

- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) — Snell et al. (arXiv v1, August 2024). *Read*: abstract and the passage defining oracle and model-predicted difficulty bins. *Scope*: effectiveness of test-time strategies varies with prompt difficulty in their setting, and their oracle difficulty needs ground truth (CLM-005). Their efficiency and model-size comparisons are not used.
- [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393) — Muennighoff et al. (arXiv v3, March 2025). *Read*: abstract, introduction, and Section 3.1 (budget forcing). *Scope*: budget forcing as a control tied to one fine-tuned model and math benchmarks (CLM-006).
- [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) — Chen et al. (arXiv v1, May 2025). *Read*: abstract only. *Scope*: evaluated reasoning models often do not verbalize hints they use (CLM-007).
- [When More Thinking Hurts: Overthinking in LLM Test-Time Compute Scaling](https://arxiv.org/abs/2604.10739) — Zhou et al. (arXiv v1, 12 April 2026, preprint). *Read*: abstract, Sections 1–3 (budget forcing, marginal utility, flip events), and Limitations. *Scope*: evidence motivating non-monotonic budget tests on open-weight models and mathematical/scientific tasks (CLM-008). Not reproduced here; replication remains necessary.

**LEGACY / INSUFFICIENT WHEN USED ALONE**

- Greedy single-path decoding as the only reasoning baseline.
- Reporting oracle pass@k as if a production selector achieved it.
- Treating visible rationale plausibility or length as correctness, faithfulness, or compute.
- Counting only generator output tokens while omitting verifier, tool, orchestration, cancellation, and capacity costs.

**PRODUCTION SOURCE TRACE**

- Repository: `huggingface/transformers`
- Revision: `27166ea03f12c940f23176a904ab1d2ff1a3dcbb` — [generation/utils.py](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/generation/utils.py) · [generation/stopping_criteria.py](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/generation/stopping_criteria.py)
- Verified: 2026-09-26 by direct static inspection; the same revision, files, and symbols were statically re-inspected on 2026-10-01. Not executed in this repository.
- Files: `src/transformers/generation/utils.py`, `src/transformers/generation/stopping_criteria.py`.
- Symbols: `GenerationMixin.generate`, `GenerationMixin._prepare_generated_length`, `GenerationMixin._get_stopping_criteria`, `GenerationMixin._sample`, `MaxLengthCriteria.__call__`, and `StoppingCriteriaList.__call__`.
- Observed path: `generate` calls `_prepare_generated_length` and `_get_stopping_criteria`, then enters the selected decode loop; in `_sample` each step ends with `unfinished_sequences & ~stopping_criteria(input_ids, scores)`. `max_new_tokens` sets `max_length` to `max_new_tokens + input_ids_length`. The criteria list is built from a length criterion, a time criterion if `max_time` is set, a stop-string criterion if `stop_strings` is set (a tokenizer is required), an EOS criterion if an EOS token is configured, a confidence criterion only for assistant-model generation, and any user-supplied criteria. `StoppingCriteriaList.__call__` ORs the per-row boolean results; the length and time criteria return one decision for all rows.
- Generalization: this is a pinned implementation snapshot. It demonstrates generic generation controls, not a universal runtime design or a semantic reasoning controller. Module 01 traces sampling at a different Transformers revision; do not mix symbols across the two.

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

Every lab follows:

$$
\text{PREDICT}\to\text{BUILD}\to\text{MEASURE}\to\text{EXPLAIN}
\to\text{BREAK}\to\text{IMPROVE}\to\text{FALSIFY}.
$$

### LAB A — Sequential Budget Sweep and Faithfulness Probe

- **Objective:** compare direct answers and elicited reasoning across token/time budgets, then test whether rationales acknowledge controlled answer-changing cues.
- **Pre-Registered Hypothesis:** one bounded reasoning regime will improve declared utility on selected strata, while some direct-answer or excessive-continuation strata will not improve; predeclare the minimum effect and cost ceiling.
- **Independent Variables:** model/prompt revision, task stratum, direct versus chain-of-thought prompt, new-token cap, stop rule, temperature, and benign versus biasing intervention.
- **Dependent Variables:** parsed correctness, rationale and answer tokens, TTFT/end-to-end latency, cost, finish reason, correctness transitions by prefix, intervention sensitivity, and acknowledgement rate.
- **Break & Falsify:** include tasks where direct answering is strong, misleading exemplars, early truncation, forced continuation, repeated loops, and cues that change answers. Falsify monotone “more tokens means better reasoning” if a preregistered budget interval loses utility.
- **Required artifact:** raw generations, parser specification/tests, budget-quality-cost curves with uncertainty, intervention pairs with the acknowledgement rate and its denominator, the effective stopping-criteria list for one run (from the Section 09 source trace, attached rather than redone), and a statement of what the faithfulness probe cannot establish.
- **Alignment:** Lessons 6.1 and 6.5.
- **Effort Estimate:** 4h.

### LAB B — Self-Consistency, pass@k, and Correlated Error

- **Objective:** implement repeated sampling, oracle pass@k, plurality/majority selection, and a dependence audit.
- **Pre-Registered Hypothesis:** increasing $k$ will raise oracle opportunity on the chosen population, but selected utility will depend on candidate dependence and the declared selection rule.
- **Independent Variables:** $k$, temperature/top-p, prompt variant, answer normalizer, tie rule, and task difficulty stratum.
- **Dependent Variables:** single-sample accuracy, finite-sample pass@k, selected accuracy, oracle-selection gap, duplicates, answer entropy, pairwise agreement/error correlation, invalid parses, total work, and critical-path latency at controlled parallelism.
- **Break & Falsify:** construct or locate a stratum where candidates confidently repeat one wrong answer; perturb normalization to expose merge/split errors; compare empirical gains with the IID analytical baseline.
- **Required artifact:**
  1. Derivations with assumptions for the four models of Lesson 6.2.
  2. A tested implementation that reproduces Worked Example B exactly (single-sample 0.40, pass@3 0.70, plurality 0.50, oracle set success 0.75) before it is run on real candidates.
  3. The candidate table with seeds, normalized answers, and labels.
  4. Dependence diagnostics: per-item correct counts, the spread of per-item accuracy, duplicate rate, and population pass@k computed per item and averaged versus the pooled IID value.
  5. A statement of the sampling protocol that makes draws independent, and one run that deliberately breaks it (greedy or a shared seed) with the resulting pass@k.
  6. A cost-normalized comparison against direct answering.
- **Alignment:** Lesson 6.2.
- **Effort Estimate:** 4h.

### LAB C — Verifier-Guided Search Under Adversarial Candidates

- **Objective:** build bounded best-of-N or tree search using an outcome or process verifier and separate candidate opportunity from selection.
- **Pre-Registered Hypothesis:** verifier-guided search will improve selected success over an equal-work unguided baseline only where verifier discrimination remains adequate under search pressure.
- **Independent Variables:** candidate count, search breadth/depth, verifier revision, threshold/ranking rule, pruning, domain, and adversarial perturbation strength.
- **Dependent Variables:** oracle set success, selected success, verifier calibration and ranking, false positives/negatives, abstention, score-quality gap, branches expanded/pruned, independent correctness, total work, and critical path.
- **Break & Falsify:** inject persuasive wrong solutions, invalid but high-scoring steps, paraphrases, out-of-domain items, and increasing search pressure. Reject the verifier-gaming explanation if independent utility and calibration remain stable under the preregistered stress set.
- **Required artifact:** node/candidate records, verifier card, held-out and adversarial results, source/revision manifest, failure taxonomy, and rollback threshold.
- **Alignment:** Lessons 6.3 and 6.4.
- **Effort Estimate:** 5h.

### LAB D — Adaptive Budget Controller Under Cost and SLO

- **Objective:** route requests among direct, sequential, parallel, and search strategies using features available before target outcomes are known.
- **Pre-Registered Hypothesis:** a leakage-free adaptive policy will improve declared utility over uniform policies at equal aggregate cost on a heterogeneous workload without violating protected SLO/subgroup constraints.
- **Independent Variables:** router features, budget levels, strategy set, load, latency deadline, and aggregate resource envelope.
- **Dependent Variables:** difficulty/value calibration, routing confusion, allocated work, utility, subgroup effects, cost, critical-path latency, cancellation waste, serving queue/capacity signals, and SLO-goodput.
- **Break & Falsify:** remove leakage-prone features, apply temporal/domain shift, introduce traffic bursts, and compare against equal-cost uniform policies. Falsify adaptive advantage if it disappears at equal aggregate cost or violates a protected constraint.
- **Required artifact:** controller and manifest, offline replay with the aggregate budget of each policy stated in one unit, a total-work and critical-path ledger per route, a controlled loaded test, leakage audit, policy frontier, operational limits, and canary/rollback design.
- **Alignment:** Lessons 6.5 and 6.6.
- **Effort Estimate:** 5h.

## 07 Break / Incident Scenarios

### Incident 06.1 — Offline Reasoning Gain, Production Utility Loss

**Incident Symptoms:**
A release replaces direct decoding with an adaptive policy. It sends difficult-looking requests to parallel candidates and a process verifier, while allowing longer sequential continuation when confidence is low. Offline benchmark selected accuracy improves. In production, tail latency and cost rise, SLO-goodput falls, one user subgroup regresses, and verifier scores keep increasing with search depth even though sampled human audits do not.

**Diagnostic Protocol (Task):**
The incident does not prescribe one root cause. The learner must:

1. **Form competing hypotheses:** traffic or task shift; correlated candidates; answer-normalization bugs; verifier miscalibration or gaming; difficulty-label leakage; continuation-induced answer reversal; cancellation waste; serving saturation; subgroup routing skew; or an unrelated deployment change.
2. **Identify missing evidence:** joined request/candidate/verifier/resource traces; direct, oracle, selected, and audited correctness; route allocation; score calibration; token/call ledger; queue and capacity metrics; cancellation status; release diff; subgroup and temporal slices.
3. **Discriminate:** replay a pinned population; compare uniform and adaptive policies at equal total cost; sweep width/depth/budget; independently judge selected and rejected candidates; disable features suspected of leakage; repeat under controlled load.
4. **Rank explanations:** use temporal ordering, counterfactual routing, factorial ablations, and effect uncertainty. Do not treat one correlation or aggregate accuracy as causal proof.
5. **Intervene:** gate or roll back the failed route, recalibrate/replace the verifier, repair parsing, cap or cancel work, reserve serving headroom, or change the objective—only as supported by evidence.
6. **Remeasure:** apply the same population, quality, subgroup, cost, latency, capacity, and SLO-goodput contract with preregistered promotion and rollback thresholds.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Transfer Problem — Defend a Test-Time Reasoning Policy

Design a production reasoning policy for a mixed workload containing objectively checkable problems, open-ended analytical tasks, and latency-sensitive requests. You may use direct decoding, bounded sequential reasoning, self-consistency, best-of-N, tree search, outcome verification, or process verification.

**Required Deliverables:**

1. a pinned task population, model/runtime configuration, serving envelope, cost unit, quality criteria, and subgroup/SLO constraints;
2. a mechanism choice per workload class with evidence classification and explicit non-goals, including what any visible rationale is and is not used as evidence for and the intervention probe that supports that use;
3. derivations for IID oracle pass@k, population pass@k with unequal item difficulty, finite-sample pass@k, and a voting baseline, each with its independence assumption and a failure case;
4. a complete compute ledger separating total executed work from critical-path latency, with the parallel resources assumed for the critical path;
5. a candidate and verifier telemetry schema that reconstructs oracle opportunity, selection, pruning, cancellation, and final delivery;
6. an experiment that measures candidate dependence and oracle-selection gap;
7. a verifier validation suite covering calibration, shift, adversarial candidates, abstention, and independent checks;
8. a fixed or adaptive budget/stopping policy that names each stop as a resource stop, an exact early stop, or a heuristic stop, with a leakage audit, an equal-budget comparison, and non-monotonic budget tests;
9. a pinned production source trace for generation/stopping behavior and a clear generalizability statement;
10. a canary, capacity plan, rollback rule, and incident playbook that remeasures after intervention.

A benchmark accuracy increase alone does not pass. The defense must show where quality comes from, how the system selects it, what all executed work costs, and which observations would falsify the policy's causal story.

## 09 Required Evidence & Rubric

### Required Artifact: Reasoning-System Trace

For every request, preserve or safely summarize enough data to join:

- request/task stratum and prompt/model/runtime revision;
- route, budgets, stop conditions, and random seeds;
- every candidate/node, parent, parsed answer, finish/prune reason, and timestamps;
- verifier revision, input view, score, selected output, threshold, and abstention;
- independent correctness or audit status where available;
- generator/verifier/tool/orchestration work, cancellations, critical-path latency, and serving context.

Sensitive reasoning traces require an explicit retention and access policy; observability does not authorize unbounded storage of user data.

### Required Artifact: Production Source Trace

The reference trace uses Transformers commit `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`. The learner's trace must contain:

1. repository, exact revision, verification date, and files;
2. the entry point `GenerationMixin.generate` and the symbols on the path to the stop decision (`_prepare_generated_length`, `_get_stopping_criteria`, the decode loop, `StoppingCriteriaList.__call__`);
3. how `max_new_tokens` becomes a total length, and which configuration fields add which criteria;
4. how per-row decisions are combined;
5. static-versus-executed status for each statement, and a generalizability statement that separates this runtime's resource stops from a semantic reasoning controller.

### Reference Checks (arithmetic only; not a design answer)

Reviewers use these to check a submission's calculations. A submission with different declared inputs is checked against its own inputs.

- **IID baseline** ($p=0.4$, $k=3$, independent draws, one common $p$): oracle $0.784$; strict majority $0.352$.
- **Candidate table** (Lesson 6.2, Worked Example B): single-sample $0.40$; finite-sample pass@3 $0.70$; oracle set success at five candidates $0.75$; plurality-selected $0.50$; gap $0.25$.
- **Unequal difficulty** (half of items $p_i=0.8$, half $p_i=0$, $k=3$): population pass@3 $0.496$ versus pooled IID $0.784$.
- **Selection** (Lesson 6.4): oracle $0.60$, selected $0.45$, selection accuracy given opportunity $0.75$.
- **Ledger** (Lesson 6.5, Worked Example A): $10.7$ accelerator-seconds; critical path $3.7$ s with four workers and $5.7$ s with two.
- **Budget sweep** (Lesson 6.5, Worked Example B): utility $0.58$, $0.64$, $0.63$, $0.54$; maximum at $B=2$.
- **Routing** (Lesson 6.6): $0.712$ uniform versus $0.760$ adaptive at 300 units; $0.772$ at 380 units is not an equal-budget result.

### Rubric Dimensions

For every dimension, **Insufficient** reports a result without mechanism, assumptions, or complete cost; **Competent** satisfies the stated dimension with reproducible evidence; **Strong** adds counterexamples, matched falsification, scoped source reasoning, and operational remeasurement.

- **Mechanistic reasoning** (Mastery 2, 5): separates elicitation, sampling, search, verification, selection, stopping, and delivery.
- **Mathematical discipline** (Mastery 3; Reference Checks): states which draws are independent, the common-$p$, subset, majority, and fixed-candidate-set assumptions; does not apply the IID formula to pooled or dependent candidates; does not convert oracle or mean metrics into deployed guarantees.
- **Measurement** (Mastery 1, 4, 6): reports opportunity, selection, task utility, total work, critical path, and production impact at declared boundaries; does not use token count as a quality measure.
- **Verification rigor** (Mastery 7): treats learned scores as estimates, validates calibration and shift, and uses independent/adversarial checks.
- **Source trace** (Mastery 9; Required Artifact items 1–5): pinned, path-complete, and scoped.
- **Failure diagnosis** (Incident 06.1 steps 1–6): ranks competing causes with discriminating experiments and permits interactions and bottleneck transitions.
- **Operational defense** (Mastery 8, 10): includes privacy, cancellation, capacity, subgroup, canary, and rollback controls.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Reasoning traces and faithfulness | Lesson 6.1 | Lesson 6.1 Guided Practice; LAB A (intervention pairs) | Mastery 2, 5; Incident 06.1 step 2 (audited versus selected correctness) | Intervention pairs with acknowledgement rate and denominator; trace contract; rubric: Mechanistic reasoning |
| pass@k baselines and their independence assumptions | Lesson 6.2 (Models 1–4, Worked Examples A–C) | Lesson 6.2 Guided Practice; LAB B artifacts 1–2, 4–5 | Mastery 3; Reference Checks "IID baseline", "Candidate table", "Unequal difficulty" | Derivations, tested implementation, dependence diagnostics; rubric: Mathematical discipline |
| Self-consistency and oracle-versus-selected accounting | Lesson 6.2 (Worked Example B); Lesson 6.4 | LAB B artifacts 3, 6; LAB C | Mastery 5, 6; Incident 06.1 steps 1–3 (correlated candidates, normalization) | Candidate table, oracle–selection gap by item; rubric: Measurement |
| Tree/search mechanics and call accounting | Lesson 6.3 | Lesson 6.3 Independent Practice; LAB C | Mastery 2, 4, 5 | Search graph, node records, predicted versus executed call counts, pruning evidence |
| Outcome/process verification | Lesson 6.4 | Lesson 6.4 Guided Practice; LAB C | Mastery 7; Incident 06.1 steps 1–5 (verifier miscalibration or gaming); Reference Check "Selection" | Verifier card, confusion table, calibration, adversarial results; rubric: Verification rigor |
| Compute ledger and critical path | Lesson 6.5 (Worked Example A) | Lesson 6.5 Guided Practice; LAB D ledger | Mastery 4; Reference Check "Ledger" | Total-work and critical-path ledger with declared parallel resources; rubric: Measurement |
| Budget and stopping policy | Lesson 6.5 (Worked Example B, stop types) | LAB A budget sweep; LAB D | Mastery 8; Incident 06.1 steps 3 and 5 (continuation, cancellation); Reference Check "Budget sweep" | Budget–quality–cost curve, stop-type table, correctness transitions |
| Production source trace | Lesson 6.5 (runtime stop primitives); Section 05 trace | LAB A (attaches the effective stopping-criteria list) | Mastery 9; Section 09 Production Source Trace items 1–5 | Trace pinned to commit `27166ea0…`; rubric: Source trace |
| Adaptive policy diagnosis | Lesson 6.6 | Lesson 6.6 Guided Practice; LAB D | Mastery 8, 10; Incident 06.1 steps 1–6; Reference Check "Routing" | Leakage audit, equal-budget replay, loaded test, canary plan; rubric: Failure diagnosis, Operational defense |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can:

1. explain why a visible rationale is neither automatic proof of correctness nor faithful causal explanation;
2. implement self-consistency and distinguish oracle pass@k from selected and user-visible accuracy;
3. derive the relevant probability baselines, state which draws must be independent and which probabilities must be equal, and show an empirical dependence counterexample;
4. build bounded search with inspectable proposal, evaluation, pruning, and cost records;
5. validate a verifier beyond its own score and detect a widening score-quality gap;
6. account for generator, verifier, tool, orchestration, cancellation, total-work, and critical-path costs, without using token count as a measure of quality;
7. design stopping and adaptive budgets that test premature stopping and overthinking;
8. trace a pinned runtime implementation without treating it as the definition of reasoning control;
9. diagnose and remediate an offline-to-production regression, then remeasure at the same boundaries.

### Module Wrap-Up (Final Mental Model Reconstruction)

Test-time reasoning is controlled generation plus selection under resource constraints. Extra work creates opportunities, not guaranteed value. The engineering task is to expose where opportunity becomes quality, measure what selection loses, charge every executed branch, and stop only when evidence supports the utility trade.

### Cost & Telemetry Hook

- [Lesson 22.1](../22-ai-economics/README.md): cost per *correct selected* answer — every sampled branch, verifier call, and cancelled candidate stays in the numerator.
- [Lesson 22.5](../22-ai-economics/README.md): an adaptive budget is a cascade; cost escalation the same way as routing (escalation rate times added cost).
- [Lesson 23.1](../23-observability-reliability/README.md) and [Lesson 23.2](../23-observability-reliability/README.md): emit per-candidate spans (tokens, verifier score, stop reason); sampling or dropping candidate spans censors the vote record.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
