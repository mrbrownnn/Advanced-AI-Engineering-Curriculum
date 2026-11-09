# Module 16 — Falsification Engineering

## 00 Why This Module Exists

Evaluation estimates how a system performs on a declared population. Falsification engineering asks a different question: **what controlled search could prove our current requirement, oracle, or causal story wrong?** A passing suite is bounded negative evidence; a valid minimized counterexample is a concrete defect and a new regression asset.

```text
requirement / hypothesis
          |
 valid domain + oracle + severity
          |
 examples | properties | metamorphic | differential | mutation | faults
          |
 search + stochastic repetitions + complete lineage
          |
 candidate failure
          |
 validate -> adjudicate -> minimize -> deduplicate -> attribute
          |
 fix / accept risk -> promote regression -> remeasure escapes
```

Module 15 owns evaluation programs, estimates, graders, and release decisions. This module owns systematic counterexample search and test adequacy. Module 18 owns the security threat model and controls; Module 23 owns production reliability operations. Security red teams and chaos experiments use this module's methods but keep their domain policy in those modules.

**Research cutoff:** 2026-09-26 for claims unchanged since registry 1.0.0. Claims added or changed in registry 1.1.0 were checked against sources opened on 2026-10-01, with a landscape cutoff of 2026-09-30.

**Module Orientation**
- **Engineering Problem**: Search systematically for counterexamples without confusing test volume, coverage, agreement, or suite survival with correctness.
- **What You Will Do**: Specify falsification contracts, build generators/state machines/shrinkers, validate metamorphic relations, run differential and mutation tests, inject bounded faults, trace Hypothesis, and operate a counterexample lifecycle.
- **Environment**: Python 3.10+, Hypothesis or an equivalent property-testing engine, isolated model/tool mocks, reproducible seeds, a fault injector, and disposable sandboxes for effectful tests.
- **Evidence Rule**: Keep source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**) separate. `CLM-xxx` identifiers point to `research-registry/16-falsification-engineering.yaml`.
- **Running fixture**: Lessons 16.2–16.6 follow one **synthetic** case, fixture F-16 (timeout after commit), from generation through shrinking, metamorphic adjudication, differential testing, mutation, replay, and promotion. Its traces and tables come from a small deterministic toy model that was executed for this module (**D**, CLM-019). The model is not in the repository and is not a production system; its rules are stated in Lesson 16.2 so the learner can rebuild it in LAB A.

## 01 Baseline Assumptions

- Module 00: claims, assumptions, measurement validity, and uncertainty.
- Module 09, Lessons 9.1, 9.3, and 9.6: BM25 scoring, exact versus approximate nearest-neighbor search, and Recall@k. Lesson 16.3 uses them without re-teaching them.
- Modules 12–14: state transitions, attempts, effects, recovery, and fault boundaries. Module 14, Lesson 14.2 teaches idempotency keys and delivery semantics.
- Module 15: evaluation contracts, experimental units, grader validity, and release gates.

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
  instruction: 4h       # lesson instruction: 30+50+45+45+35+35 min
  guided_practice: 3h   # lesson practice: 25+35+30+30+30+30 min
  labs: 12h             # LAB A 3h + LAB B 3h + LAB C 3h + LAB D 3h
  assessment: 3h        # Mastery transfer problem 2.5h + Incident 16.1 0.5h
  source_trace: 2h      # Section 09 Production Source Trace artifact, counted once
  total: 24h
```

Each category is counted once. The 2h source trace is the Section 09 artifact; it is not also counted inside LAB A.

The learner must specify properties and oracles; build valid generators and shrinkers; test stateful protocols; justify metamorphic relations; use differential and mutation testing without oracle illusions; quantify stochastic evidence; separate adaptive discovery from estimation; inject bounded faults; operate a counterexample lifecycle; and trace a pinned property-testing engine.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

A falsification contract records:

- requirement/hypothesis and owner;
- system boundary, valid domain, preconditions, and severity;
- predicate or relation and independent adjudication path;
- generator/search/fault policy, seed, budget, and stopping rule;
- stochastic repetition and denominator policy;
- system, model, prompt, tool, data, environment, and oracle versions;
- what failure means and what survival cannot prove.

Coverage is a search signal. Disagreement is a triage signal. Mutation score is suite-sensitivity evidence. None is correctness.

## 04 Lessons

### Lesson 16.1 — From Requirement to Falsifiable Contract

**Engineering Question:**
How can a vague robustness claim become a bounded predicate whose failure and non-claims are explicit?

**Concepts & Definitions:**
- **Valid domain**: inputs/states for which the requirement is intended.
- **Oracle**: procedure that adjudicates the property.
- **Generator support**: the inputs a generator can produce with nonzero probability.
- **Non-claim**: behavior the search budget or generator cannot establish.

**Mechanism Explanation:**
Turn “the agent is robust” into observable contracts: schema is always valid; unauthorized effects never occur; adding irrelevant evidence should not change a cited answer; retry preserves one logical effect; cancellation prevents later effects; a locale transformation preserves a task-specific relation. Mark preconditions and acceptable nondeterminism (**D**, CLM-001).

CheckList's minimum-functionality, invariance, and directional-expectation tests are a useful behavioral vocabulary (**O**, CLM-006). Build a capability × test-type × risk-slice matrix, but do not treat filled cells as proof of completeness. Structural coverage records what was exercised, not whether the oracle checked it correctly (**D**, CLM-013).

**Quantitative Model / Derivation:**
For generator support $S_G$ and valid domain $D$, a property test can falsify $P$ only on sampled $x\in S_G\cap D$. Passing provides no direct evidence about $D\setminus S_G$.

where:
- $P$ is the property (predicate) under test; $x$ is one generated input or action sequence;
- $S_G$ is the set of inputs the generator can produce with nonzero probability; $D$ is the valid domain;
- $S_G\cap D$ ("intersection") is the inputs that are both reachable and valid — the only place a test can find a real counterexample;
- $D\setminus S_G$ ("set difference") is the valid inputs the generator can never produce — the non-claims.

*Tiny example (synthetic):* the domain allows amounts 1–100, but a generator draws amounts 1–50. Then $S_G\cap D$ covers amounts 1–50 and $D\setminus S_G$ is amounts 51–100. A million passing cases say nothing about amount 75, because amount 75 was never possible. Inputs in $S_G\setminus D$ (invalid ones the generator *can* produce) are the opposite problem: their failures are harness noise, not counterexamples.

**Worked Example (synthetic):**
- *Input*: the claim “the payment agent is robust to retries.”
- *Steps*: (1) Predicate: for an authorized transfer intent in the declared schema, the receiver records at most one effect per logical intent. (2) Domain: opened account, amount 1–100, unique intent id, at most two attempts. (3) Oracle: the receiver-side ledger, not the client's response. (4) Generator: action sequences with a fault choice per attempt from {none, request lost, acknowledgement lost}. (5) Budget and seed. (6) Severity: critical, because money moves.
- *Result*: a contract that one observation can falsify: two ledger entries for one intent.
- *Interpretation / limits*: non-claims are part of the result. This generator never crashes the receiver between its own steps and never uses a third attempt, so survival says nothing about either. Fixture F-16 in Lesson 16.2 runs this contract.

**Knowledge Check:**
1. Why is a filled capability matrix not completeness proof?
2. Which precondition makes an authorization property meaningful?

**Guided Practice:**
Write a falsification contract for one invariant and list three inputs outside generator support.

**Feedback Contract:**
- *Expected Output*: requirement, domain/preconditions, oracle, generator/search budget, stopping rule, severity, and at least three named non-claims. Knowledge check 1: cells record that a test exists, not that its oracle is right or that the cell's inputs were reached. Knowledge check 2: the identity and permission state under which the action is attempted.
- *Typical Error*: using case count or coverage as correctness, or an oracle that reads only the client's response.
- *Diagnostic Hint*: What exact observation falsifies the predicate? Who records it?
- *Concept to Revisit*: bounded negative evidence; generator support.

**Learning Outcome:**
State a predicate precise enough to fail and humble enough not to overclaim.

*(Effort: 30m instruction, 25m practice)*

---

### Lesson 16.2 — Property-Based and Stateful Testing

**Engineering Question:**
How do generators, state machines, and shrinkers find small valid failures without changing their causal identity?

**Concepts & Definitions:**
- **Generator**: constructive sampler over structured valid inputs/actions.
- **Stateful model**: commands, preconditions, transitions, invariants, and postconditions.
- **Failure signature**: the tuple that identifies *why* a case fails. In this module it is (violated invariant, causal boundary, receiver-side effect count).
- **Shrinker**: ordered reduction process. A shrink step is **valid** only if the candidate is inside the domain and fails with the same signature.

**Mechanism Explanation:**
QuickCheck established the generator-plus-property pattern (**O**, CLM-002). In AI systems, generate structured prompts, schemas, tool responses, event streams, model outcomes, and action sequences—not arbitrary invalid bytes unless parser rejection is the property.

Measure generator support proxies: category/slice frequencies, boundary-value reach, filter/rejection rate, state/action transitions, and failure yield. Use constructive generators instead of filtering when possible (**D**, CLM-003). “Smallest” is relative to the shrinker's representation and order.

Stateful tests model commands, preconditions, transitions, invariants, and postconditions. Vary retries, cancellation, duplicate/out-of-order events, stale versions, timeout boundaries, and recovery. Record the whole action/observation/effect trace.

**Production source trace.** At Hypothesis revision `9c55f97e507eae21677457e84737b386fd06e271`, `given` wraps a test, and `find` and `get_state_machine_test` (called by `run_state_machine_as_test`) each build a `@given` test. `StateForActualGivenExecution.run_engine` constructs a `ConjectureRunner` and calls `run`. `ConjectureRunner._run` goes through reuse, generate, and shrink phases; `shrink_interesting_test_cases` calls `Shrinker.shrink`, which runs `initial_coarse_reduction` and `greedy_shrink` (**O**, CLM-014). Static source inspection only.

**What "same failure" means in that engine.** At the same revision, the shrink predicate keeps a candidate only if it is `INTERESTING` with the same `interesting_origin`. `InterestingOrigin` holds the exception type, file name, line number, exception context, and group elements. When `report_multiple_bugs` is disabled the predicate accepts any `INTERESTING` result. Candidates are ordered by `sort_key`: shorter choice sequences first, then smaller choices (**O**, CLM-017).

That is weaker than this module's signature. Two invariants checked by one `assert` on one line share an origin, so the engine may slide from one to the other. To make the engine preserve the signature, raise each invariant from its own line or exception type, and assert the receiver-side effect count inside the test (**D**, CLM-018).

**Quantitative Model / Trade-off Comparison:**
Report slice/boundary/transition reach, rejection rate, unique failure clusters, shrink steps and ratio, predicate calls, replay rate, and execution cost. None alone establishes oracle adequacy.

**Worked Example — fixture F-16, generation and shrinking (synthetic, executed toy model):**

*Model rules.*
- Receiver: holds opened accounts, a ledger of committed effects, and a dedupe table keyed by `effect_id`. It commits, then acknowledges.
- Client: `transfer(intent, account, amount)` makes at most two attempts and retries after a timeout. **Seeded defect**: it mints a new `effect_id` for every attempt (`i1-a1`, `i1-a2`).
- Faults, one per attempt: `none`; `req_lost` (lost before the receiver); `ack_lost` (the receiver commits, the acknowledgement is lost, and the client sees a timeout).
- Domain: the account is opened before any transfer to it; intent ids are unique; amount is 1–100.
- Receiver-side invariants: `INV-ONE-EFFECT` (at most one ledger entry per intent); `INV-ACKED-IMPLIES-EFFECT`; `INV-TERMINAL-KNOWN` (the client does not end in `UNKNOWN`).

*Input — original trace.* Generator seed 16, second generated case, 8 steps:

```text
1 open(B)
2 read(A)
3 note(y)
4 read(A)
5 note(x)
6 transfer(i1, B, 59, faults=(ack_lost, ack_lost))
7 transfer(i2, B, 29, faults=(req_lost, req_lost))
8 transfer(i3, B, 54, faults=(none, req_lost))
```

Client results: `i1 UNKNOWN`, `i2 UNKNOWN`, `i3 OK`. Receiver ledger: `(i1, i1-a1, 59)`, `(i1, i1-a2, 59)`, `(i3, i3-a1, 54)`.

Target signature: `(INV-ONE-EFFECT, ack_lost_after_commit, receiver_effects(i1)=2)`.

*Steps — signature-preserving shrink.* The shrinker tries, in order: delete a step (last first), replace a fault by a simpler one (`none` < `req_lost` < `ack_lost`), set the amount to 1. It restarts after each accepted step. Selected candidates:

| Candidate | In domain | Outcome | Receiver effects (i1) | Accepted |
|---|---|---|---:|---|
| delete step 8 `transfer(i3)` | yes | same signature | 2 | yes |
| delete step 7 `transfer(i2)` | yes | same signature | 2 | yes |
| delete step 6 `transfer(i1)` | yes | passes | 0 | no: no failure |
| delete `note`, `read` (4 steps) | yes | same signature | 2 | yes |
| delete step 1 `open(B)` | **no** | harness error `NO_ACCOUNT` | 0 | no: out of domain |
| fault 1 `ack_lost → none` | yes | passes | 1 | no: no failure |
| fault 1 `ack_lost → req_lost` | yes | `INV-TERMINAL-KNOWN` | 1 | no: different signature |
| fault 2 `ack_lost → none` | yes | same signature | 2 | yes |
| amount `59 → 1` | yes | same signature | 2 | yes |

The run made 24 predicate calls and accepted 8 steps.

*Counting the accepted steps:* deleting steps 8 and 7 (2), deleting the four `note`/`read` steps one at a time (4), simplifying fault 2 (1), and lowering the amount (1) give $2+4+1+1=8$. A **predicate call** is one execution of a candidate plus its oracle check; the table shows only selected candidates, so rejected calls outnumber the visible "no" rows.

*Result — minimized trace.* 2 steps, a shrink ratio of 2/8 (minimized length ÷ original length $=0.25$; lower means more reduction):

```text
1 open(B)
2 transfer(i1, B, 1, faults=(ack_lost, none))

attempt 1: send effect_id=i1-a1 -> receiver commits -> ack lost -> client TIMEOUT
attempt 2: send effect_id=i1-a2 -> receiver commits -> ACK      -> client OK
receiver ledger: (i1, i1-a1, 1), (i1, i1-a2, 1)
```

Same signature, receiver effects 2. The timeout, the retry, and the duplicate effect are all still present.

*An invalid shrink.* A naive shrinker that accepts **any** failure follows the same deletions, then also deletes `open(B)`, because the harness error counts as a failure. From there every simplification "still fails". It ends at one step:

```text
1 transfer(i1, B, 1, faults=(none, none))      -> harness error NO_ACCOUNT, receiver effects 0
```

This trace is smaller, outside the domain, has no timeout, no retry, and no effect. It is not a counterexample to the contract.

*Replay.* The fault is scripted, so replay is deterministic in this model: 20 of 20 replays of the minimized trace give the same signature and the same two ledger entries. Lesson 16.5 treats the stochastic case.

*Interpretation / limits.* The shrink is minimal for this shrinker's order and representation, not in any absolute sense. The fixture shows a mechanism in a toy model; it is not evidence about any real payment system.

**Knowledge Check:**
1. When is constructive generation preferable to filtering?
2. Why is the smallest serialized input not always the smallest causal trace?
3. In the pinned engine, which fields decide that a shrunk case is "the same failure"?

**Guided Practice:**
(a) Under the F-16 model, the trace `open(B)`, `read(A)`, `transfer(i1, B, 40, (req_lost, ack_lost))`, `transfer(i2, B, 7, (ack_lost, none))` fails. State its signature. Then classify each candidate as accepted or rejected, with the reason: delete `read`; delete `transfer(i2)`; delete `transfer(i1)`; delete `open`; set i2's amount to 1; change i2's first fault to `req_lost`. Give the minimized trace. (b) Generate action sequences with cancellation, retries, stale versions, and unknown effects; implement a signature-preserving shrinker.

**Feedback Contract:**
- *Expected Output*: (a) signature `(INV-ONE-EFFECT, ack_lost_after_commit, 2)` on i2 (i1 ends `UNKNOWN` with one effect). Delete `read`: accepted. Delete `transfer(i2)`: rejected, different signature (`INV-TERMINAL-KNOWN`, one effect). Delete `transfer(i1)`: accepted. Delete `open`: rejected, out of domain. Amount to 1: accepted. First fault to `req_lost`: rejected, the duplicate disappears. Minimized: `open(B)`, `transfer(i2, B, 1, (ack_lost, none))`. (b) Support proxies, preconditions, full trace, seed, shrink log with rejection reasons, failure signature, and replay count. Knowledge check 3: exception type, file, line, context, and group elements.
- *Typical Error*: accepting "delete `transfer(i2)`" because the trace still fails; shrinking out the boundary that caused the defect; an oracle that reads the client response.
- *Diagnostic Hint*: Does the minimized case fail for the same adjudicated reason, with the same receiver-side effects?
- *Concept to Revisit*: failure signature and valid shrinking.

**Learning Outcome:**
Produce a valid, small, replayable counterexample rather than an untriageable random transcript.

*(Effort: 50m instruction, 35m practice)*

---

### Lesson 16.3 — Behavioral and Metamorphic Relations

**Engineering Question:**
When is a transformation semantics-preserving enough that output inconsistency is evidence of a defect?

**Concepts & Definitions:**
- **Input relation** $R_i$: declared relation between source and follow-up inputs, including every precondition.
- **Output relation** $R_o$: invariant or directional expectation on outcomes.
- **Input-relation validator**: a check, run before $R_o$, that reads inputs and run configuration only and decides whether $R_i$ holds.
- **Verdicts**: `NOT_APPLICABLE` ($R_i$ fails), `PASS`, or `VIOLATION` ($R_i$ holds and $R_o$ fails).

**Mechanism Explanation:**
Metamorphic testing replaces a missing single-input oracle with two obligations (**O**, CLM-004):

$$R_i(x,x')\Rightarrow R_o(f(x),f(x')).$$

where $x$ is the source input, $x'$ the follow-up input, $f$ the system under test, and $\Rightarrow$ reads "if … then". In words: *if* the pair of inputs satisfies the input relation, *then* the pair of outputs must satisfy the output relation. A violation needs both halves: $R_i$ true **and** $R_o$ false. If $R_i$ is false, the implication says nothing, which is why the verdict is `NOT_APPLICABLE` rather than `PASS`.

Both $x,x'$ must remain in-domain; $R_i$ must preserve or deliberately change the relevant semantics; $R_o$ must encode the expected invariant or direction. Candidate relations include permutation invariance only when order is irrelevant, citation preservation under irrelevant distractors, and equivariance under a semantics-preserving locale transform. Each is a claim to be proved for the task, not a default.

A 2025 study collected 191 relations for NLP tasks, ran 36 of them on three LLMs, and manually analyzed 937 reported violations. The authors report that roughly 60% were true positives (**O**, CLM-005). That figure is author-reported for their four tasks and relations and does not transfer. Its lesson does: a large share of raw violations can come from the relation, not the system. Never assume synonym replacement, paraphrase, formatting, answer-order swap, or added context is semantically neutral for every task.

For free-form outputs, separate transformation validity from output-relation grading. Use executable/domain checks where possible and audit any semantic judge independently.

**Quantitative Model / Derivation:**

*An invalid relation: "recall is monotonic when the eligible corpus grows."* With $Recall@k=|Rel\cap Top_k|/|Rel|$ and fixed $k$, this is false. Counterexample: $k=1$, one relevant document $r$ with score 0.8, so $Recall@1=1$. Add one irrelevant document with score 0.9. It takes rank 1 and $Recall@1=0$ (**D**, CLM-016).

where $Rel$ is the set of relevant documents, $Top_k$ the set of the first $k$ returned documents, $|\cdot|$ set size, and $k$ the cutoff (here $k$ is a rank cutoff, not the reproduction count $k$ of Lesson 16.5). The numerator can only count relevant documents that fit inside the $k$ slots, so a new higher-scoring irrelevant document pushes a relevant one out.

Corpus growth has no general direction at all. With a corpus-dependent scorer such as BM25, adding irrelevant documents changes the IDF of query terms and can reorder the *old* documents. Example with $tf=1$, no length normalization, and $IDF=\ln\bigl(1+\frac{N-n+0.5}{n+0.5}\bigr)$: corpus $d_1=\{a\}$, $d_2=\{b\}$, $d_3=\{b\}$, $d_4=\{c\}$, query $\{a,b\}$. Scores are $d_1=1.204$ and $d_2=d_3=0.693$, so $d_1$ ranks first. Add three irrelevant documents containing only $a$. Now $IDF(a)=0.575$ and $IDF(b)=1.163$, and $d_2$ ranks first. If $Rel=\{d_2\}$, $Recall@1$ rose from 0 to 1; if $Rel=\{d_1\}$, it fell from 1 to 0.

*Step-by-step for the BM25 example.* $N$ is the number of documents in the corpus and $n$ the number containing the term. With $tf=1$ and no length normalization, a document's score is the sum of $IDF$ over query terms it contains.
1. Before: $N=4$. Term $a$: $n=1$, $IDF(a)=\ln\bigl(1+\frac{4-1+0.5}{1+0.5}\bigr)=\ln(1+2.333)=\ln 3.333=1.204$. Term $b$: $n=2$, $IDF(b)=\ln\bigl(1+\frac{2.5}{2.5}\bigr)=\ln 2=0.693$.
2. Scores: $d_1$ contains only $a$, so $1.204$; $d_2,d_3$ contain only $b$, so $0.693$. Rank 1 is $d_1$.
3. After adding three $\{a\}$ documents: $N=7$. Term $a$: $n=4$, $IDF(a)=\ln\bigl(1+\frac{3.5}{4.5}\bigr)=\ln 1.778=0.575$. Term $b$: $n=2$, $IDF(b)=\ln\bigl(1+\frac{5.5}{2.5}\bigr)=\ln 3.2=1.163$.
4. Scores: $d_1$ (and each new document) $0.575$; $d_2,d_3$ $1.163$. Rank 1 is now $d_2$ (by tie-break over $d_3$). Term $a$ became common, so it is worth less; $b$ became relatively rarer, so it is worth more.

*A valid relation, MR-K: grow $k$ on a fixed exact ranking.*

Preconditions ($R_i$):
1. same query;
2. same corpus snapshot;
3. same scorer version;
4. retrieval is exact: the result is a prefix of one deterministic total order with a declared tie-break;
5. the relevance set $Rel$ is the same and non-empty;
6. $k'\ge k$.

Output relation ($R_o$): $Recall@k'\ge Recall@k$.

Proof: under 1–4, $Top_k$ is the first $k$ entries of the same order as $Top_{k'}$, so $Top_k\subseteq Top_{k'}$ and $|Rel\cap Top_k|\le|Rel\cap Top_{k'}|$. Under 5 the denominator is unchanged (**D**, CLM-016).

An approximate index does not satisfy precondition 4. With HNSW-style search the candidate set depends on the search parameters, so the top-2 of one run need not be a prefix of the top-3 of another (Module 09, Lesson 9.3). A changing ANN ranking must not be treated as an exact ranking.

*Input-relation validator.* It reads the run manifest, never the outputs:

```python
def recall_at_k(ranking, relevant, k):
    return len(set(ranking[:k]) & relevant) / len(relevant)

def validate_input_relation(src, fol):
    """R_i for MR-K. Checks run-manifest fields only, never outputs."""
    problems = []
    for field in ("query", "corpus_hash", "scorer_version", "tie_break"):
        if src[field] != fol[field]:
            problems.append(f"{field} differs")
    if src["mode"] != "exact" or fol["mode"] != "exact":
        problems.append("retrieval mode is not exact")
    if not src["relevant"] or src["relevant"] != fol["relevant"]:
        problems.append("relevance universe is empty or differs")
    if fol["k"] < src["k"]:
        problems.append("k' < k")
    return problems

def check_mr_k(src, fol):
    problems = validate_input_relation(src, fol)
    r_src = recall_at_k(src["ranking"], src["relevant"], src["k"])
    r_fol = recall_at_k(fol["ranking"], fol["relevant"], fol["k"])
    if problems:
        return "NOT_APPLICABLE", problems, r_src, r_fol
    nested = fol["ranking"][:src["k"]] == src["ranking"][:src["k"]]
    verdict = "PASS" if r_fol >= r_src else "VIOLATION"
    return verdict, [f"nested_prefix={nested}"], r_src, r_fol
```

`nested_prefix` is a diagnostic on outputs. It is reported with the verdict and is not part of $R_i$; putting it in $R_i$ would make the relation unable to fail.

**Worked Example A — MR-K and its counterexamples (synthetic; the code above was executed on these inputs):**
- *Input*: exact ranking `d1, d3, d2, d4` with $Rel=\{d_1,d_2\}$, so $Recall@k$ for $k=1..4$ is 0.5, 0.5, 1.0, 1.0.

| Case | Source → follow-up | Validator | Recall | Verdict |
|---|---|---|---|---|
| A | exact, $k=2\to3$, same manifest | no problems | 0.5 → 1.0 | `PASS` |
| B | corpus grows, $k=1$, `[d1]` → `[x9]`, $Rel=\{d_1\}$ | `corpus_hash differs` | 1.0 → 0.0 | `NOT_APPLICABLE` |
| C | `mode=ann`, $k=2$ `[d1,d2]` → $k=3$ `[d1,d3,d4]` | `retrieval mode is not exact` | 1.0 → 0.5 | `NOT_APPLICABLE` |
| D | manifest says exact, same lists as C | no problems; `nested_prefix=False` | 1.0 → 0.5 | `VIOLATION` |
| E | a new document is judged relevant between runs | `relevance universe … differs` | 0.5 → 0.667 | `NOT_APPLICABLE` |

- *Interpretation / limits*: B is the counterexample to the corpus-growth relation; the drop is real and is not a defect under MR-K. C is expected behavior of an approximate index; measure it against exact search with ANN recall (Module 09), not with MR-K. D is a defect candidate: either the manifest is wrong or the "exact" path is not deterministic, and `nested_prefix=False` says where to look. The validator trusts the manifest's `mode` field; confirm it on a sample by brute-force recomputation.

**Worked Example B — metamorphic adjudication of fixture F-16 (synthetic, executed toy model):**

| Relation | $R_i$ | $R_o$ | Observed | Adjudication |
|---|---|---|---|---|
| MR-1, receiver boundary | the same request, same `effect_id`, delivered twice | ledger has one entry | 1 entry; second delivery answered `ACK_DUP` | $R_i$ valid; `PASS`. The receiver dedupes correctly. |
| MR-2, client-intent boundary | source: one intent, no fault. Follow-up: same intent, `ack_lost` on attempt 1 | `receiver_effects` of the follow-up ≤ that of the source | source 1, follow-up 2 | $R_i$ valid: same intent, permitted fault, retry allowed. Oracle is the receiver ledger. `VIOLATION`: real defect, the F-16 signature. |
| MR-X, invalid | "two transfers with the same amount and receiver are duplicates" | one ledger entry | 2 entries, one per intent | $R_i$ **invalid**: two intents are two requests. `NOT_APPLICABLE`; reporting it as a defect would be a false positive. |

With the client fixed to reuse one `effect_id` per intent, MR-2 gives 1 and 1. MR-1 alone would never have found the defect, because the receiver was never wrong.

**Knowledge Check:**
1. Why is paraphrase not universally semantics-preserving?
2. Can an LLM judge validate both transformation and output relation without circularity?
3. Why must the validator not read the returned rankings?

**Guided Practice:**
(a) An exact ranking is `d2, d5, d1, d7, d3` with $Rel=\{d_1,d_3\}$. Compute $Recall@k$ for $k=2,3,5$ and state the MR-K verdict for $k=2\to3$ and $k=3\to5$. Then give the verdict for: the same $k$ change after the index was rebuilt from a larger crawl; the same $k$ change on an ANN index. (b) Define five relations with preconditions, including one deliberately invalid transformation, and adjudicate both stages independently.

**Feedback Contract:**
- *Expected Output*: (a) $Recall@2=0$, $Recall@3=0.5$, $Recall@5=1.0$; both steps `PASS`. Rebuilt index: `NOT_APPLICABLE` (`corpus_hash differs`). ANN index: `NOT_APPLICABLE` (`retrieval mode is not exact`). (b) Relation cards with domain proof, independent output oracle, stochastic repetitions, and a false-positive taxonomy. Knowledge check 3: a relation whose precondition is checked on the outputs it constrains cannot be violated.
- *Typical Error*: treating any changed answer as model failure; filing an ANN recall drop or a corpus-growth drop as an MR violation; asserting recall rises with corpus size.
- *Diagnostic Hint*: Did the transformation preserve the relevant semantics? Which manifest field changed between the two runs?
- *Concept to Revisit*: input-relation validity; exact versus approximate ranking (Module 09, Lesson 9.3).

**Learning Outcome:**
Detect inconsistent behavior without inventing false invariants.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 16.4 — Differential and Mutation Testing

**Engineering Question:**
How can disagreement and controlled mutants expose blind spots without pretending either one identifies ground truth automatically?

**Concepts & Definitions:**
- **Differential candidate**: shared input with divergent outcomes across comparable systems.
- **Common-mode miss**: comparable systems agree because they share a defect or a dependency, so no candidate appears.
- **Mutation operator**: controlled injected fault representing a declared defect class.
- **Equivalent mutant**: a mutation that does not change required behavior. **Invalid (stillborn) mutant**: one that cannot run. Both are excluded from the score.
- **Subsumed mutant**: one killed by every test that kills another mutant; it adds no information.

**Mechanism Explanation:**
Differential testing runs comparable implementations or revisions on the same input. DeepXplore demonstrates this family for neural systems (**O**, CLM-007). A disagreement locates a candidate boundary; adjudication decides whether A, B, both, or neither satisfy the contract. Independence matters: related models, shared retrieval, shared prompts, a shared mock, or the same judge can fail together.

Mutation testing injects controlled faults into code, configuration, prompt, policy, data, tool response, or workflow guard (**O**, CLM-008). Report operator distribution, equivalence review, subsumption, execution cost, and which test killed which mutant. Include realistic mutants such as removed authorization, wrong timeout unit, missing citation binding, stale version, dropped terminal event, retry after unknown commit, and changed system prompt.

**Quantitative Model / Derivation:**
With generated mutants $M$, killed mutants $K$, and adjudicated equivalent or invalid mutants $E$:

$$MS=\frac{K}{M-E}.$$

where $MS$ is the mutation score, $M$ the count of generated mutants, $K$ the count killed (at least one test fails on the mutant), and $E$ the count adjudicated equivalent or invalid. The denominator $M-E$ is the number of mutants a test *could* kill: an equivalent mutant behaves correctly, so no correct test should fail on it, and an invalid mutant never runs. Leaving them in the denominator would penalize the suite for something no test can do. (Here $K$ is a count of mutants; it is unrelated to the cutoff $k$ of Lesson 16.3.)

Example: 20 generated, 12 killed, 3 equivalent, 1 invalid gives $MS=12/(20-3-1)=0.75$. Step by step: $E=3+1=4$; killable mutants $=20-4=16$; $MS=12/16=0.75$. The naive $12/20=0.60$ understates sensitivity. The score measures sensitivity to the selected operators—not production defect prevalence or correctness. One surviving critical mutant matters more than the score.

**Worked Example — fixture F-16, differential miss and mutation kill table (synthetic, executed toy model):**

*Differential.* Input: the minimized trace of Lesson 16.2. Targets: client A (new `effect_id` per attempt, two attempts), client B (same id policy, three attempts), client C (one `effect_id` per intent).

| Receiver used | A | B | C | Disagreement | Receiver-side oracle |
|---|---|---|---|---|---|
| Shared mock: commit and acknowledgement are one atomic call, so `ack_lost` cannot be expressed | OK, 1 effect | OK, 1 effect | OK, 1 effect | none | passes for all three |
| Faithful receiver | OK, 2 effects | OK, 2 effects | OK, 1 effect | A and B agree; C differs | A and B violate `INV-ONE-EFFECT`; C passes |

Two common-mode misses appear. First, the shared mock hides the defect from every target. Second, on the faithful receiver A and B still agree with each other, because they share the id policy. Only a target with an independent design (C) or an oracle that does not depend on agreement exposes the defect. Even then, disagreement alone does not say which side is wrong; the ledger invariant does.

*Mutation.* Base system: the fixed client (one `effect_id` per intent) and the deduping receiver. Tests:
- T1: response-only; one request is lost, the retry must end `OK`.
- T2: ledger invariant with no fault.
- T3: the promoted minimized F-16 trace with the receiver-side oracle.
- T4: MR-1 at the receiver boundary.

All four pass on the base system.

| Mutant | T1 | T2 | T3 | T4 | Status with T1, T2, T4 only | Status after T3 is promoted |
|---|---|---|---|---|---|---|
| M1 client mints a new `effect_id` per attempt | pass | pass | **kill** | pass | survives | killed |
| M2 receiver dedupe disabled | pass | pass | **kill** | **kill** | killed | killed |
| M3 client does not retry after a timeout | **kill** | pass | **kill** | pass | killed | killed |
| M4 receiver dedupe key is (`effect_id`, attempt number) | pass | pass | **kill** | pass | survives | killed |
| M5 dedupe store refactored, same behavior | pass | pass | pass | pass | equivalent | equivalent |
| M6 receiver acknowledges before it commits | pass | pass | pass | pass | survives | **survives** |

- *Steps*: $M=6$, $E=1$ (M5), so $M-E=5$ killable mutants. Before promotion only M2 and M3 are killed by T1, T2, or T4: $K=2$ and $MS=2/5=0.40$. Promoting T3 adds kills of M1 and M4. After promotion $K=4$ and $MS=4/5=0.80$.
- *Result*: M1 is the incident's defect, and the suite could not see it until T3 existed. M4 is a second defect class that only T3 kills. M6 survives a suite scoring 0.80.
- *Why M6 is not equivalent*: a probe outside the suite crashes the receiver between its two steps. The base system then ends with one effect after the retry; M6 ends with the client told `OK` and zero effects. The suite has no fault at that boundary, which is exactly the non-claim recorded in Lesson 16.1.
- *Interpretation / limits*: the score rose from 0.40 to 0.80 and the most severe survivor is unchanged. Six hand-written mutants are not an operator distribution; a real registry needs systematic operators and an equivalence review.

**Knowledge Check:**
1. Why does disagreement require independent adjudication?
2. How do equivalent mutants distort a naive score?
3. A and B agree on every input. Name two reasons this can happen when both are wrong.

**Guided Practice:**
(a) A registry has 15 generated mutants: 9 killed, 2 equivalent, 1 stillborn. Compute $MS$. One survivor removes an authorization check. State what you report. (b) Build a kill matrix across code, configuration, prompt, policy, retrieval, event-order, retry, and stopping mutants.

**Feedback Contract:**
- *Expected Output*: (a) $MS=9/(15-3)=0.75$; report the score **and** the surviving authorization mutant as a critical gap with the missing test named. (b) Operator distribution, equivalence review, adjudicated disagreement, common dependencies, kill matrix, and cost. Knowledge check 3: a shared defect in both targets, or a shared dependency (mock, retrieval, prompt, judge) that hides the difference.
- *Typical Error*: dividing by all 15 mutants (0.60); declaring one disagreeing implementation wrong by identity; reporting a high score while a critical mutant survives.
- *Diagnostic Hint*: Which oracle or dependency do all targets share? Which test would kill the survivor?
- *Concept to Revisit*: suite sensitivity and common-mode failure.

**Learning Outcome:**
Expose shared assumptions and measure whether the suite notices controlled breakage.

*(Effort: 45m instruction, 30m practice)*

---

### Lesson 16.5 — Stochastic, Adaptive, and Fault-Injection Evidence

**Engineering Question:**
What probability claim is justified after stochastic search, adaptive discovery, or controlled fault injection?

**Concepts & Definitions:**
- **Trial unit**: sample, request, trajectory, user/session, or environment seed.
- **Reproduction rate**: the fraction of replays of one fixed case that fail with the same signature.
- **Adaptive discovery**: search distribution changes in response to prior results.
- **Confirmatory sample**: protected evidence not selected by the discovery process.

**Mechanism Explanation:**
Define the trial unit and preserve failed attempts. Repetitions estimate conditional stochastic behavior of one case; broader generators estimate input variation.

Fuzzers and red teams adapt toward weakness. Their finds are discovery evidence, not unbiased prevalence samples (**D**, CLM-010). Freeze promoted regressions and use a separate protected confirmatory sample or an analysis that models adaptive selection.

Fault injection names target boundary, fault, timing, expected invariant, blast radius, abort/cleanup, and recovery oracle (**D**, CLM-011). Inject timeout, malformed/partial response, cancellation race, quota, stale cache, duplicate event, tool failure, worker crash, or lost acknowledgement at controlled boundaries. Synthetic faults may miss correlated provider or regional incidents.

**Quantitative Model / Derivation:**
Under IID Bernoulli trials with a fixed failure predicate and zero observed failures in $n$ trials, the exact one-sided upper bound at confidence $1-\alpha$ is

$$p_U=1-\alpha^{1/n}.$$

With $n=100$ and $\alpha=0.05$, $p_U\approx0.0295$ (**D**, CLM-009). By symmetry, $n$ reproductions in $n$ replays give a one-sided lower bound $\alpha^{1/n}$ on the reproduction rate. For $k$ reproductions in $n$ replays with $0<k<n$, report $k/n$ with an exact binomial (Clopper–Pearson) interval. All three assume independent, identically distributed trials. They do not cover unseen prompts and fail under correlated retries, shifting systems, adaptive search, or hidden exclusions.

where $p$ is the unknown per-trial failure probability, $n$ the number of trials, $\alpha$ the error rate we accept (confidence $1-\alpha$, e.g. 95% for $\alpha=0.05$), $p_U$ the upper bound on $p$, and $k$ the number of reproductions (not the rank cutoff of Lesson 16.3). "IID Bernoulli" means each trial is an independent yes/no draw with the same $p$.

*Step-by-step for $p_U$.*
1. If the true rate is $p$, the chance that $n$ independent trials all pass is $(1-p)^n$.
2. The bound is the largest $p$ that is still not "too surprising" given zero failures: set $(1-p_U)^n=\alpha$. Any $p>p_U$ would make a clean run less likely than $\alpha$.
3. Take the $n$-th root: $1-p_U=\alpha^{1/n}$, so $p_U=1-\alpha^{1/n}$.
4. Numbers for $n=100$: $\ln 0.05=-2.996$; $-2.996/100=-0.02996$; $e^{-0.02996}=0.9705$; $p_U=1-0.9705=0.0295$. Check: $(1-0.0295)^{100}\approx0.050$.

Plain reading: 100 clean trials do not show the failure rate is zero; they show it is probably below about 3%. Because $-\ln 0.05\approx3$, $p_U\approx 3/n$ for large $n$, so halving the bound needs twice the trials.

*The same rule as a search budget.* To claim $p_U\le p^*$ from a clean run, solve $(1-p^*)^n\le\alpha$ for $n$: $n\ge\ln\alpha/\ln(1-p^*)$. For $p^*=0.01$ and $\alpha=0.05$: $n\ge-2.996/-0.01005=298.1$, so 299 clean IID trials. A smaller budget cannot support that claim, whatever the result.

*Step-by-step for the reproduction lower bound.* Let $q$ be the reproduction rate. All $n$ replays reproduce with probability $q^n$. Setting $q_L^{\,n}=\alpha$ gives $q_L=\alpha^{1/n}$ — the same algebra with "pass" and "fail" swapped.

*Clopper–Pearson in plain language.* The interval contains every rate $p$ under which the observed $k$ of $n$ would not be in the extreme $\alpha/2$ tail on either side. It is "exact" because it uses the binomial distribution directly instead of a normal approximation, so it stays valid for small $n$ and for $k$ near 0 or $n$. Its width shrinks roughly like $1/\sqrt{n}$: in the worked example below, going from 20 to 200 replays narrows the interval from width 0.438 to 0.132.

**Worked Example — fixture F-16 under replay (synthetic, executed toy model):**
- *Input 1, scripted fault*: the minimized trace with `ack_lost` forced on attempt 1. 20 replays.
- *Result 1*: 20 of 20 reproduce the signature with two receiver effects. One-sided 95% lower bound on the reproduction rate: $0.05^{1/20}=0.861$. The model is deterministic, so the bound is slack; it shows what 20 clean replays can and cannot support.
- *Input 2, timing race*: no scripted fault. The acknowledgement latency of attempt 1 is drawn from a log-normal distribution with median 80 ms and $\sigma=0.5$; the client timeout is 100 ms; the duplicate occurs when latency exceeds the timeout. Replay seed 163.
- *Result 2*: 7 of 20 replays reproduce the signature: rate 0.35, exact 95% interval $[0.154,\,0.592]$. With 200 replays: 60 reproduce, rate 0.30, interval $[0.237,\,0.369]$. In this model the true probability is 0.328.
- *Where 0.328 comes from*: log-normal latency $L$ means $\ln L$ is normal with mean $\ln 80$ and standard deviation $\sigma=0.5$ (σ is on the log scale, not in ms). A duplicate needs $L>100$. Standardize: $z=\frac{\ln 100-\ln 80}{0.5}=\frac{\ln 1.25}{0.5}=\frac{0.223}{0.5}=0.446$. Then $P(L>100)=1-\Phi(0.446)=0.328$, where $\Phi$ is the standard normal CDF. Both intervals contain 0.328; the point estimates 0.35 and 0.30 do not equal it, which is ordinary sampling noise.
- *Interpretation / limits*: one failing run does not make the case deterministic, and 13 passing replays do not make it fixed. Record the count and interval, and keep the scripted-fault version as the regression test. In every reproducing replay the signature and both ledger entries match; a replay that fails for another reason is counted separately, not as a reproduction. The latency distribution is an exercise assumption.
- *Contrast*: 100 independent draws on one fixed prompt match the IID model. 100 retries on ten prompts chosen adaptively after near failures do not. Neither covers unseen prompt domains.

**Knowledge Check:**
1. Why do repeated generations estimate conditional rather than broad input variation?
2. Which safety controls must accompany fault injection?

**Guided Practice:**
(a) A fixed case shows 0 failures in 50 independent trials. Give the 95% upper bound. A flaky counterexample reproduces in 12 of 30 replays; give the rate and exact 95% interval. (b) Separate adaptive discovery, protected confirmation, and workload estimation; specify fault boundary, blast radius, abort, cleanup, and recovery oracle.

**Feedback Contract:**
- *Expected Output*: (a) $1-0.05^{1/50}=0.058$; $12/30=0.40$ with interval $[0.227,\,0.594]$. (b) Trial unit, independence assumptions, denominator, selection history, system version, fault timing, and cleanup. Knowledge check 2: bounded blast radius, an abort condition, cleanup, and a recovery oracle.
- *Typical Error*: reporting red-team finds as prevalence, applying an IID bound to adaptive trials, or deleting a flaky case instead of measuring its reproduction rate.
- *Diagnostic Hint*: Did later trials depend on earlier outcomes? Is the denominator every replay or only the ones that failed?
- *Concept to Revisit*: discovery versus estimation; reproduction rate.

**Learning Outcome:**
Quantify evidence without converting search success into population rates.

*(Effort: 35m instruction, 30m practice)*

---

### Lesson 16.6 — Counterexample Lifecycle and Portfolio Economics

**Engineering Question:**
How should discoveries become durable regression evidence, and which mix of methods is worth its operational cost?

**Concepts & Definitions:**
- **Failure cluster**: cases grouped by adjudicated causal signature.
- **Promotion**: versioned minimized counterexample enters regression protection.
- **Portfolio evidence**: unique consequential yield, overlap, false positives, cost, latency, and escapes.

**Mechanism Explanation:**
Every candidate moves through `VALIDATE → ADJUDICATE → MINIMIZE → DEDUPLICATE → ATTRIBUTE → FIX/ACCEPT → PROMOTE → RETIRE/REFRESH` (**D**, CLM-012). Preserve original and minimized cases, transformation/shrink trace, system/oracle versions, raw attempts, effect evidence, root cause, owner, severity, and replay status.

Cluster by causal signature, not surface text alone. A flaky failure remains a stochastic test with a measured reproduction rate; it is not silently deleted. Review obsolete fixtures when product requirements change.

The claim that a risk-weighted portfolio beats one coverage objective is a hypothesis; test it against a preregistered baseline (**H**, CLM-015).

**Quantitative Model / Trade-off Comparison:**
For each method report unique adjudicated clusters and severity alongside compute/judge/human/triage cost, overlap, suite latency, false blocks, replay rate, and escaped incidents. One summary is

$$\text{yield}=\frac{\sum_{\text{unique clusters}}\text{severity weight}}{\text{total cost}},$$

where the weights are risk policy, not measurements.

The numerator sums one severity weight per **unique adjudicated cluster** the method found — not per case — so 20 duplicates of one cluster count once. The denominator is the method's total declared cost (compute, judge, human, triage) in one unit. Yield is therefore "risk-weighted distinct problems found per unit of cost".

**Worked Example A — the F-16 ledger entry (synthetic):**

| Field | Value |
|---|---|
| Original case | seed 16, case 2, 8 steps (Lesson 16.2) |
| Minimized case | `open(B)`, `transfer(i1, B, 1, (ack_lost, none))` |
| Shrink log | 24 predicate calls, 8 accepted; rejected: out of domain, no failure, different signature |
| Signature | `(INV-ONE-EFFECT, ack_lost_after_commit, receiver_effects=2)` |
| Oracle | receiver ledger |
| Adjudication | MR-2 `VIOLATION` with valid $R_i$ (Lesson 16.3) |
| Replay | scripted fault 20/20; timing race 7/20, interval $[0.154,\,0.592]$ (Lesson 16.5) |
| Root cause | client mints a new `effect_id` per attempt |
| Promotion | T3; kills M1 and M4, which the earlier suite missed (Lesson 16.4) |
| Open gap | M6 survives; no fault between acknowledgement and commit |
| Retire/refresh | review when the retry policy or receiver protocol changes |

**Worked Example B — yield (synthetic):**
- *Input*: Method A finds 20 cases that collapse to two low-severity clusters at cost 1. Method B finds four cases across three clusters, one critical, at cost 2. Weights: low 1, critical 10.
- *Steps*: A yields $(1+1)/1=2.0$ per unit cost. B yields $(1+1+10)/2=6.0$.
- *Result*: case count ranks A first (20 against 4). Weighted unique yield ranks B first.
- *Interpretation / limits*: the ranking depends on the weight given to a critical cluster. With weight 2 instead of 10, B yields 2.0 and the methods tie. State the weights before the campaign.

**Knowledge Check:**
1. Why should flaky failures be measured rather than silently deleted?
2. When should a promoted case be retired or refreshed?

**Guided Practice:**
(a) Method C finds 12 cases in four clusters, three low and one high (weight 5), at cost 3. Compute its yield and rank it against A and B. (b) Run `validate → adjudicate → minimize → deduplicate → attribute → fix/accept → promote`, then compare a risk-weighted portfolio with one coverage-maximizing baseline.

**Feedback Contract:**
- *Expected Output*: (a) $(3+5)/3=2.67$; order B (6.0), C (2.67), A (2.0). (b) A ledger with every field of Worked Example A, causal cluster, owner/severity, replay rate, cost, overlap, and escape tracking. Knowledge check 2: when the requirement, protocol, or oracle it encodes changes, or when another case covers the same signature.
- *Typical Error*: ranking by case count; accumulating unreproducible fixtures with no owner; dropping the original trace after minimizing.
- *Diagnostic Hint*: Which unique risk does this test continue to cover?
- *Concept to Revisit*: counterexample lifecycle.

**Learning Outcome:**
Turn discoveries into durable evidence without building a counterexample cemetery.

*(Effort: 35m instruction, 30m practice)*

---

## 05 Literature & Production Source Map

Entries say what was opened and when. Where no date is given, the source was not re-opened in this revision and its registry entry stands from the 2026-09 pass.

**REFERENCE / BASELINE**

- [QuickCheck](https://doi.org/10.1145/351240.351266) — Claessen and Hughes, 2000; properties and generators (**O**, CLM-002). Not re-opened.
- [Metamorphic Testing](https://www.cse.ust.hk/faculty/scc/publ/CS98-01-metamorphictesting.pdf) — Chen, Cheung, and Yiu, 1998; source/follow-up relations (**O**, CLM-004). Not re-opened.
- [DeepXplore](https://arxiv.org/abs/1705.06640) — Pei et al., 2017. Abstract re-opened 2026-10-01; differential testing with similar systems as cross-referencing oracles (**O**, CLM-007). Its reported finding rates are specific to its models and datasets and are not used here.
- [CheckList](https://aclanthology.org/2020.acl-main.442/) — Ribeiro et al., ACL 2020; capability/test-type matrix (**O**, CLM-006). Not re-opened.
- [Mutation Testing Survey](https://doi.org/10.1109/TSE.2010.62) — Jia and Harman, 2011 (**O**, CLM-008). Not re-opened.

**RECOMMENDED ENGINEERING BASELINE** (this module's derivation; earlier revisions labeled this list "CURRENT DEFAULT")

Explicit falsification contracts; valid structured generation; stateful invariants; domain- and signature-preserving shrink; input-relation validation before output adjudication; independent adjudication; discovery/confirmation separation; complete attempt lineage; minimized versioned regression cases; cost and escape accounting.

These follow from stated assumptions (**D**, CLM-001, CLM-003, CLM-009, CLM-010, CLM-011, CLM-012, CLM-013, CLM-016, CLM-018). They are a recommendation. They are not the default of any tool, and this module has not measured how widely they are adopted.

**ONE IMPLEMENTATION'S DEFAULT** (observed, scoped to the named source)

- Hypothesis at the pinned commit: the default settings profile has `max_examples=100`, `stateful_step_count=50`, `derandomize=False`, and a 200 ms deadline; `run_state_machine_as_test` without settings uses `deadline=None`. Shrinking keeps a candidate with the same `interesting_origin` (**O**, CLM-017). This is one library at one revision. A 100-example default is not a recommended budget for an effectful property.

**INDUSTRY PREVALENCE:** not established by this module. No adoption survey of property-based, metamorphic, or mutation testing for AI systems was opened (`TODO_VERIFY`, CLM-020).

**WORKLOAD-DEPENDENT:** generator distributions, metamorphic relations, oracle portfolio, repetitions, mutants, fault model, severity, shrink order, coverage signals, and red-team budget.

**FRONTIER** (scoped to what was read)

- [Metamorphic Testing of Large Language Models for Natural Language Processing](https://arxiv.org/abs/2511.02108) — Cho, Ruberto, Terragni, arXiv v1, 2025. Abstract, introduction, and Section IV opened 2026-10-01 (**O**, CLM-005). Author-reported and not reproduced; its relations and rates need task-specific validation.
- Other 2025–2026 work on automated relation discovery, LLM-specific mutation operators, and agent fault injection was not surveyed (`TODO_VERIFY`, CLM-020).

**LEGACY / INSUFFICIENT:** random prompts without a domain model; “no failures means safe”; agreement as truth; disagreement as proof one named system is wrong; coverage as correctness; mutation score without equivalent-mutant policy; red-team findings as prevalence; shrinking that changes the failure; "recall can only rise when the corpus grows"; chaos without blast-radius controls.

**PRODUCTION SOURCE TRACE**

- Repository: `HypothesisWorks/hypothesis`
- Revision: `9c55f97e507eae21677457e84737b386fd06e271`
- Verified: 2026-09-26; files and symbols below re-read at this revision on 2026-10-01. Static inspection only; nothing was executed.
- Files/symbols, under `hypothesis/src/hypothesis/`: `core.py::{given,find,StateForActualGivenExecution.run_engine}`; `stateful.py::{RuleBasedStateMachine,get_state_machine_test,run_state_machine_as_test}`; `internal/conjecture/engine.py::ConjectureRunner.{run,_run,shrink_interesting_test_cases,shrink,new_shrinker}`; `internal/conjecture/shrinker.py::{sort_key,Shrinker.shrink}`; `internal/escalation.py::InterestingOrigin`; `_settings.py` (default profile).
- Path: `@given` test or state machine → `run_engine` → `ConjectureRunner.run` → reuse / generate / shrink phases → `Shrinker.shrink` with the same-origin predicate.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`. Seeded or generated lab data are synthetic and must be labeled so in every artifact.

### LAB A — Generators, State Machines, and Shrinking

- **Objective**: Build structured stateful generators and validity-preserving shrinkers for tool-loop invariants.
- **Pre-Registered Hypothesis**: Constructive generation plus signature-aware shrinking will improve valid boundary reach and replayable minimization over filter-heavy generation on the declared domain.
- **Independent Variables**: Generator design, action topology, fault boundary, shrink order, and nondeterminism.
- **Dependent Variables**: Support/transition reach, rejection rate, unique failures, shrink validity/ratio, predicate calls, replay, and cost.

- Specify schema, authorization, cancellation, retry, and effect invariants for a tool-using loop.
- Rebuild the F-16 model from the rules in Lesson 16.2 as a rule-based state machine with a receiver-side oracle; pin and replay seeds.
- **Required test**: (i) find a duplicate-effect failure and record the original trace; (ii) minimize it with a signature-preserving shrinker and log every rejected candidate with its reason; (iii) run a shrinker that accepts any failure and show where it leaves the domain or changes signature; (iv) replay the minimized case 20 times and report the count.
- Break with invalid overgeneration, heavy filtering, rare transitions, correlated choices, nondeterminism, and a shrinker that changes failure identity.
- Artifact: support/transition report, original and minimized trace, shrink log, replay count, and explicit non-claims.
- **Break & Falsify**: Seed a failure that a naive shrinker erases; if the final case changes causal signature, minimization fails.
- **Alignment**: Lessons 16.1–16.2.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total). The source trace is a separate 2h artifact (Section 09).

### LAB B — Metamorphic Oracle Audit

- **Objective**: Validate task-specific input/output relations and measure false positives.
- **Pre-Registered Hypothesis**: Separating transformation validity from output adjudication will reject at least one plausible but invalid relation in the declared set.
- **Independent Variables**: Relation, task slice, transformation, model/judge version, and stochastic repeat.
- **Dependent Variables**: Transformation validity, verdict counts (`NOT_APPLICABLE`, `PASS`, `VIOLATION`), false positives, stability, latency, and cost.

- Define at least five task-specific relations with preconditions and expected output relations.
- Include invariance, directional, equivariant, and deliberately invalid transformations. One must be MR-K with its validator; one must be the corpus-growth relation, shown to be invalid with a counterexample.
- Run MR-K on an exact index and on an approximate index and report the verdict for each.
- Validate transformations and output judgments independently; test prompt/model/judge versions and stochastic repetitions.
- Artifact: relation cards with validator code, false-positive taxonomy, violation examples, and surviving-risk analysis.
- **Break & Falsify**: Include deliberately invalid transformations; acceptance of one as a product defect falsifies oracle discipline.
- **Alignment**: Lesson 16.3.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB C — Differential and Mutation Adequacy

- **Objective**: Adjudicate cross-system disagreement and measure sensitivity to realistic mutants.
- **Pre-Registered Hypothesis**: Mutation will expose at least one common-mode blind spot missed by related differential targets on the seeded fault set.
- **Independent Variables**: Model/backend/harness target, shared dependency, mutation operator, and oracle.
- **Dependent Variables**: Adjudicated disagreement, killed/equivalent/invalid/subsumed mutants, common-mode misses, cost, and latency.

- Cross two model versions, two backends, and two harness revisions on identical inputs; independently adjudicate disagreements.
- Run the same targets against a shared mock and against a faithful dependency, and record which defects the mock hides.
- Mutate authorization, timeout, prompt, schema, retrieval evidence, event order, retry, and stopping logic.
- Measure killed/equivalent/invalid/subsumed mutants, common-mode misses, method overlap, cost, and latency. Report the score before and after promoting one counterexample.
- Artifact: differential matrix, mutation operator registry, kill matrix, and gaps ranked by hazard.
- **Break & Falsify**: Mutate authorization, timeout, prompt, schema, retrieval, event order, retry, and stop logic; a high score with critical surviving mutant falsifies aggregate adequacy.
- **Alignment**: Lesson 16.4.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

### LAB D — Stochastic Search and Fault Campaign

- **Objective**: Separate adaptive discovery, protected confirmation, workload estimation, and bounded fault injection.
- **Pre-Registered Hypothesis**: A risk-weighted portfolio will improve unique consequential yield per declared cost over one coverage-maximizing baseline on the seeded campaign.
- **Independent Variables**: Corpus role, search policy, trial unit, fault/timing, method portfolio, and budget.
- **Dependent Variables**: Conditional failure estimates, reproduction rates with intervals, confirmed clusters, overlap, false positives, cost, latency, recovery, and escapes.

- Separate adaptive discovery corpus from a protected confirmatory sample and a workload sample.
- Inject boundary-specific transport, tool, state, and worker faults with abort and cleanup controls. Include a fault between acknowledgement and commit.
- Estimate conditional failure probabilities only where assumptions hold; preserve all attempts and effects. For a timing-dependent counterexample report $k/n$ and an exact interval.
- Run the full counterexample lifecycle and compare a risk-weighted portfolio against one coverage-maximizing baseline.
- Artifact: campaign manifest, fault matrix, counterexample ledger, cost/yield/escape report, and TODO_VERIFY list.
- **Break & Falsify**: Inject correlated/adaptive trials and unsafe cleanup assumptions; misuse of IID bounds or failed containment invalidates the campaign.
- **Alignment**: Lessons 16.5–16.6 and Incident 16.1.
- **Effort Estimate**: 2h implementation, 1h analysis (3h total).

## 07 Break / Incident Scenarios

### Incident 16.1 — Ten Million Passing Tests, One Duplicate Payment

- **Incident Symptoms**: A suite reports enormous case count and high transition coverage, yet production duplicates a payment after timeout; generation, mocks, shrinking, differential targets, and mutants all excluded the critical acknowledgement window.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: Invalid property, unreachable branch, correlated choices, mock mismatch, missing fault boundary, response-only oracle, retry drift, stale fixture, or telemetry loss.
  2. *Rank Initial Plausibility*: Treat test volume/coverage as search signals, not proof against the observed effect.
  3. *Identify Missing Evidence*: Recover generator distributions/rejections, seeds/shrink trace, state/action reach, contracts, attempt/effect IDs, fault timing, mutation matrix, revisions, and external postconditions.
  4. *Design Discriminating Tests*: Reproduce with a receiver-side oracle, acknowledgement-loss fault, and realistic mutant; state outcomes that falsify each leading hypothesis.
  5. *Execute Causal Diagnosis*: Rank the earliest invalid domain, oracle, mock, shrink, or fault-model boundary.
  6. *Prescribe Mitigation and Prevention*: Fix the boundary, promote the valid minimized trace, diversify common dependencies, and add ownership/retirement policy.
  7. *Remeasure*: Reproduction count with interval, shrink validity, mutant kill, unique failure yield, false positives, cost, and escaped incidents.

Fixture F-16 is a worked instance of steps 4–7 in a toy model. The incident answer must use the learner's own evidence.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Falsification Program for an Effectful AI System

Build a falsification program for a stochastic tool-using system with governed external effects and incomplete oracles.

**Constraints** (declare any value you assume and label it synthetic):
- At least one external effect cannot be undone.
- One dependency is available only as a mock in test.
- Retrieval uses an approximate index.
- One known failure reproduces only some of the time.

**Required Deliverables**:
1. Falsification contracts, valid domains, oracles, severity, budgets, and non-claims.
2. Property/stateful generators, support evidence, and a valid shrink with original trace, minimized trace, and shrink log.
3. Behavioral/metamorphic relation cards with input-relation validators and independent adjudication, including one rejected relation.
4. Differential matrix with shared dependencies named, and a mutation operator/kill registry with surviving mutants ranked by hazard.
5. IID-qualified stochastic analysis, reproduction counts with intervals, and discovery/confirmation separation.
6. Safe fault campaign and versioned counterexample ledger.
7. Pinned Hypothesis trace plus portfolio cost/yield/escape evidence.
8. Diagnosis and promoted regression for Incident 16.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit a pinned trace that maps:
1. Entry from a `@given` test or a state machine into `run_engine`.
2. The reuse, generate, and shrink phases of `ConjectureRunner`.
3. How `Shrinker` decides that a candidate is simpler and that it is the same failure.
4. Which default settings bound examples and stateful steps.
5. What that definition of "same failure" does not preserve, and how your test code closes the gap.

State which engine paths were statically inspected and which stochastic/runtime behaviors were not executed.

### Required Artifact: Counterexample Ledger

One entry per promoted case with every field of Lesson 16.6 Worked Example A. Reviewers replay the minimized case and compare the signature and effect evidence.

### Reference Checks (fixtures in this README only)

Reviewers use these to check a rebuilt F-16 model. A submission on another system is checked against its own contract.

- Minimized trace: `open(B)`, `transfer(i1, B, 1, (ack_lost, none))`; signature `(INV-ONE-EFFECT, ack_lost_after_commit, 2)`.
- MR-K cases A–E: `PASS`, `NOT_APPLICABLE`, `NOT_APPLICABLE`, `VIOLATION`, `NOT_APPLICABLE`.
- Kill table: $MS=0.40$ before T3 and $0.80$ after; M6 survives.

### Rubric Dimensions

- **Contract and Generation** (Deliverables 1–2): *Insufficient* sends random prompts or shrinks to a case with a different signature. *Competent* defines domain/property/oracle, measures support, and logs a signature-preserving shrink. *Strong* produces valid replayable minimized counterexamples with explicit non-claims and shows what a naive shrinker would have done.
- **Oracles and Adequacy** (Deliverables 3–4): *Insufficient* treats disagreement/coverage as truth or files an invalid relation as a defect. *Competent* validates input relations before output relations and reviews mutants. *Strong* exposes common-mode and equivalent-mutant blind spots through independent adjudication and ranks survivors by hazard.
- **Statistics and Fault Safety** (Deliverables 5–6): *Insufficient* reports prevalence from adaptive cases, calls a once-seen failure deterministic, or injects uncontrolled faults. *Competent* states trial assumptions, reports reproduction counts with intervals, and contains faults. *Strong* separates discovery/confirmation and proves abort, cleanup, and postconditions.
- **Lifecycle and Economics** (Deliverables 6–8): *Insufficient* accumulates cases. *Competent* owns, clusters, promotes, and reviews them. *Strong* optimizes unique consequential yield, overlap, latency, triage cost, false blocks, and escapes.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Falsification contract and oracle | Lesson 16.1 | 16.1 Guided Practice; LAB A | Mastery Deliverable 1; Incident 16.1 steps 1–2 | LAB A contract with receiver-side oracle and named non-claims |
| Property/stateful generation and shrinking | Lesson 16.2 | 16.2 Guided Practice (a)–(b); LAB A required test (i)–(iv) | Mastery Deliverable 2; Incident 16.1 steps 3–4; Section 09 Reference Checks | Original and minimized trace, shrink log, replay count |
| Metamorphic relations and input-relation validation | Lesson 16.3 | 16.3 Guided Practice (a)–(b); LAB B | Mastery Deliverable 3; rubric *Oracles and Adequacy* | LAB B relation cards with validator code and verdict counts |
| Differential and mutation testing | Lesson 16.4 | 16.4 Guided Practice (a)–(b); LAB C | Mastery Deliverable 4; Incident 16.1 step 4 (realistic mutant) | LAB C differential matrix, kill matrix before/after promotion, ranked survivors |
| Stochastic/adaptive/fault evidence | Lesson 16.5 | 16.5 Guided Practice (a)–(b); LAB D | Mastery Deliverables 5–6; Incident 16.1 step 7 | LAB D campaign manifest, fault matrix, reproduction counts with intervals |
| Counterexample lifecycle and portfolio | Lesson 16.6 | 16.6 Guided Practice (a)–(b); LAB D | Mastery Deliverables 6, 8; Incident 16.1 steps 5–6 | Counterexample Ledger; yield/cost/escape report |
| Pinned property-testing engine source trace | Lesson 16.2 (source trace and same-failure paragraphs) | LAB A (state machine on the engine); 16.2 Knowledge Check 3 | Section 09 Production Source Trace items 1–5; Mastery Deliverable 7 | Trace pinned to commit `9c55f97e…` |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner successfully completing Module 16 must be able to:
1. Falsify a meaningful system claim with a valid minimized case that keeps its failure signature.
2. Defend the domain, oracle, transformation, and non-claims.
3. Reject an invalid metamorphic relation and validate the preconditions of a valid one.
4. Measure generator reach, shrink validity, and stochastic assumptions.
5. Expose differential common-mode and mutation blind spots.
6. Inject one bounded fault with abort, cleanup, and recovery evidence.
7. Separate adaptive discovery from confirmatory estimation.
8. Promote and replay counterexamples, with reproduction counts, while measuring portfolio cost and escapes.

### Module Wrap-Up (Final Mental Model Reconstruction)

- **The Core Invariant**: Falsification engineering is disciplined search for disconfirming evidence.
- **The Search Path**: claim/domain/oracle → generated transformations/faults → candidate failure → validate/adjudicate/minimize → fix or accept risk → regression.
- Its output is not a giant test count but a trustworthy counterexample—or a precisely bounded statement about where none was found.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
