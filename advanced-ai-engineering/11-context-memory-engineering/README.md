# Module 11 — Context & Memory Engineering

## 00 Why This Module Exists

A long context window and durable memory are different systems. The context window is the finite serialized input available to one inference. Application memory is persisted state that must be written, updated, selected, authorized, compacted, and reintroduced into a later context. Neither is reliable merely because it is large.

```text
events/messages/tool results
        |
        +--> immutable history -----------+
        +--> canonical structured state --+--> read/select policy
        +--> summaries/reflections --------+         |
        +--> episodic/semantic records ----+         v
                                            context allocator
system + tools + current input + selected memory + evidence + output reserve
                                            |
                                            v
                                     model inference
                                            |
                              outcome + memory write decision
```

This module owns model-aware context budgeting, effective-context testing, memory types, write/read/update/delete policy, summarization and compression, and memory-stage diagnosis. Module 03 owns KV-cache internals, Module 09 retrieval algorithms, Module 10 RAG evidence/context assembly, Module 12 agent-loop control, and Module 14 durable execution/checkpoint semantics.

**Research cutoff:** 2026-09-30. Sources re-opened on that date are marked in the registry; other entries keep their original access dates.

**Module Orientation**

- **Engineering problem:** retain and expose the minimum sufficient, valid, authorized state for the next decision under finite tokens, latency, cost, privacy, and correctness constraints.
- **What you will do:** account for serialized context; map effective-context limits; design typed/versioned memory; implement write/read/correct/delete and conflict policies; compare compaction strategies; inject stale, poisoned, fragmented, and cross-tenant state; trace current helpers; and diagnose the earliest failing memory stage.
- **Environment:** Python 3.10+ with the target tokenizer or server usage accounting, a versioned state store/search fixture, and a reproducible model endpoint. Pin serialization protocol, model, prompt, memory schema, selectors, summarizers, and evaluator revisions.
- **Evidence rule:** distinguish source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**). Cognitive terms are software metaphors unless an implementation contract defines them.

## 01 Baseline Assumptions

- Module 00: measurands, experimental units, paired comparisons, and falsification.
- Module 01: the model input is a token-ID sequence of shape $[B,S]$ produced by one fixed tokenizer, and a probability is assigned to that ID sequence, not to a text string (Lessons 1.1–1.2); causal attention (Lesson 1.3); positional mechanisms (Lesson 1.4). Module 01 does not teach how a tokenizer splits text or how to count tokens. Lesson 11.1 teaches the token-counting facts this module needs at first use.
- Module 03: physical KV state is not application memory.
- Module 07: uncertainty, abstention, and behavioral regression.
- Modules 08–10: versioned data, retrieval contracts, RAG context assembly, citations, and freshness.
- Module 04: latency distributions, goodput, and capacity effects of longer prompts.

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
  security: REQUIRED
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 5h        # lesson instruction: 55+40+45+60+45+55 min = 300 min
  guided_practice: 3h    # lesson practice: 30+30+25+35+30+30 min = 180 min
  labs: 18h              # LAB A 4h + LAB B 4.5h + LAB C 5h + LAB D 4.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 11.1 0.5h
  source_trace: 2h       # Section 05 LangChain trace practice and Mastery Deliverable 8, counted once
  total: 31h
```
Each category is counted once. Lab analysis is not also counted as guided practice, and the source trace is not also counted as assessment time.

The learner must be able to account for every context token; reserve output safely; measure effective context rather than advertised length; design typed, versioned memory; implement selective write/read/update/delete paths; quantify compaction loss and break-even; inject stale, poisoned, contradictory, and fragmented records; and attribute failure across memory stages.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
                 application state across requests
       +----------------+----------------+----------------+
       |                |                |                |
  raw events       canonical facts    summaries      procedures
  immutable        versioned state     derived/lossy  reviewed/reusable
       \                |                |                /
        +---------------+----------------+---------------+
                            |
                  eligibility + retrieval
                            |
                 context-budget allocator
                            |
        instructions | tools | recent turns | memory | evidence
                            |
                     output reserve
```

Keep these boundaries explicit:

1. **Stored:** was the needed state correctly extracted and persisted?
2. **Valid:** is it current, authorized, and not superseded?
3. **Selected:** did the read policy retrieve it for this request?
4. **Placed:** did it survive context budgeting and ordering?
5. **Used:** did the model apply it correctly?

## 04 Lessons

### Lesson 11.1 — Context Accounting and Admission

**Engineering Question:**
Will the exact serialized request and reserved output fit without breaking protocol invariants?

**Concepts & Definitions:**

Token-counting facts needed here (taught at first use; Module 01 only establishes that the model consumes token IDs):

- A **tokenizer** maps text to a sequence of integer token IDs from a model-specific vocabulary. The count of IDs, not characters or words, is what the context limit measures. The same text can produce different counts under different tokenizers, and counts per character vary with language, code, and numbers.
- A chat request is **serialized** before tokenization. Role markers, message delimiters, tool schemas, tool-call IDs, and tool results are turned into tokens by the model's chat template or by the server. A UI may hide them; the limit still counts them.
- An **exact count** comes from running the target tokenizer on the exact serialized request, or from the usage figures the server returns. A **heuristic count** (for example a fixed characters-per-token ratio) is an estimate with an error that has to be measured (**D**, CLM-002).

For a declared model, tokenizer, message protocol, and endpoint (**D**, CLM-001):

$$
T_{system}+T_{tools}+T_{history}+T_{evidence}+T_{state}+T_{input}
+T_{framing}+T_{output\ reserve}\le T_{limit}.
$$

where every $T_{\cdot}$ is a token count under the declared tokenizer and protocol:

- $T_{system}$: system instructions; $T_{tools}$: serialized tool schemas;
- $T_{history}$: earlier turns, including tool calls and tool results; $T_{evidence}$: retrieved evidence; $T_{state}$: canonical state and selected memory;
- $T_{input}$: the current user input; $T_{framing}$: role markers, delimiters, and other tokens the chat template or server adds;
- $T_{output\ reserve}$: tokens kept free for the answer; $T_{limit}$: the context limit of the endpoint.

In plain terms: add up everything that will be sent, add the room you promise for the answer, and the sum must not exceed the limit. If any term is a heuristic estimate rather than an exact count, the worked example below adds a measured margin to the left side. If the interface has separate input and output limits, check the input terms against the input limit and the reserve against the output limit.

The inequality is an admission invariant, not a quality guarantee. Tool schemas, role/name fields, tool-call IDs, images, hidden delimiters, and provider framing can consume tokens even when a UI hides them. Input/output limits may be combined or separate. Pin the interface version and record the serialized request when policy permits.

Use the target tokenizer or server-reported usage for exact accounting where possible. If a hot path uses a heuristic, measure signed and absolute error by language, modality, tool shape, and message mix; reserve a calibrated margin. Define priority classes and atomic groups so truncation never leaves a tool result without its call or strips the instruction that interprets state.

**Worked Example — category token ledger with reserve and overflow** (synthetic request, registry CLM-020; the limit and all counts are exercise inputs, not properties of a real model).

Inputs: a combined input-plus-output limit $T_{limit}=32{,}000$ tokens. The output reserve is fixed at 2,000 and may not be reduced by the allocator. Every category is counted with the target tokenizer except evidence, which is counted by a heuristic whose measured worst-case under-count on this workload is 5%. Priority P0 is never dropped; P3 is dropped first.

| Category | Priority | Cap | Requested | Placed | Action |
|---|---|---:|---:|---:|---|
| system instructions | P0 | 1,500 | 1,200 | 1,200 | keep |
| tool schemas | P0 | 4,000 | 3,400 | 3,400 | keep |
| current input | P0 | — | 900 | 900 | keep |
| framing (measured) | P0 | — | 180 | 180 | keep |
| canonical state | P1 | 1,000 | 600 | 600 | keep |
| selected memory | P2 | 3,000 | 2,800 | 2,800 | keep |
| retrieved evidence | P2 | 12,000 | 14,000 | 12,000 | drop the 4 lowest-ranked 500-token chunks, whole chunks only |
| history | P3 | 8,000 | 11,500 | 7,900 | drop the oldest atomic groups |
| output reserve | fixed | 2,000 | 2,000 | 2,000 | never reduced |
| **total** | | | **36,580** | **30,980** | |

History consists of six atomic groups, oldest first. A group is a unit that must be kept or dropped whole:

| Group | Content | Tokens |
|---|---|---:|
| g1 | user turn | 400 |
| g2 | assistant tool call (300) + its tool result (2,900) | 3,200 |
| g3 | user turn | 350 |
| g4 | assistant turn | 1,050 |
| g5 | assistant tool call (200) + its tool result (3,300) | 3,500 |
| g6 | latest user turn, assistant turn, tool call, tool result | 3,000 |

Steps:

1. Requested total: $36{,}580$. Overflow against the limit: $36{,}580-32{,}000=4{,}580$. The request cannot be sent as is.
2. Enforce caps from the lowest priority upward. History: dropping g1 leaves 11,100; dropping g2 leaves 7,900, which is within the 8,000 cap. Evidence: 14,000 → 12,000 by dropping four whole chunks.
   - History arithmetic: $11{,}500-400=11{,}100>8{,}000$, so keep dropping; $11{,}100-3{,}200=7{,}900\le8{,}000$, so stop. Dropping g2 removes its call *and* its result.
   - Evidence arithmetic: $14{,}000-12{,}000=2{,}000$ over the cap, and $2{,}000/500=4$ whole chunks.
3. Placed total: $30{,}980$. Slack: $32{,}000-30{,}980=1{,}020$.
   - Check: $1{,}200+3{,}400+900+180+600+2{,}800+12{,}000+7{,}900+2{,}000=30{,}980$. Equivalently, requested minus dropped: $36{,}580-2{,}000-3{,}600=30{,}980$.
4. Estimator margin: 5% of the 12,000 heuristic-counted evidence tokens is 600. Admission check: $30{,}980+600=31{,}580\le32{,}000$. **Admit**, with 420 tokens unallocated.
   - Why only evidence gets a margin: every other category is an exact count. The heuristic may under-count evidence by up to 5%, so the true evidence size could be as large as $12{,}000+0.05\times12{,}000=12{,}600$. The margin reserves room for that worst case.
   - Unallocated: $32{,}000-31{,}580=420$.
5. Reconcile: placed input 28,980 + reserve 2,000 + margin 600 + unallocated 420 = 32,000. (Placed input is the placed total without the reserve: $30{,}980-2{,}000=28{,}980$.) The ledger records the dropped IDs (g1, g2, four chunk IDs) so the request can be replayed.

Why groups matter: suppose the history cap were 10,900 and the allocator dropped *messages* oldest-first. It drops g1 (11,100 left), then g2's tool call (10,800 left) and stops. The tool result of g2 is now in the prompt without the call that produced it. That request is protocol-invalid for tool-calling chat formats, or it leaves the model reading a result with no question. Dropping by group removes g1 and all of g2 and never produces an orphan.

Interpretation and limits: the ledger proves the request fits and is well-formed. It says nothing about whether the model can *use* 28,980 tokens of input well; that is Lesson 11.2. If the caps had not been enough, the next steps are a declared compaction (Lesson 11.5) or rejection. Reducing the output reserve to make room is not an allowed step, because it changes the output contract without telling the caller.

**Knowledge Check:**
1. Why can visible text undercount serialized tool/message tokens?
2. In the ledger above, the evidence heuristic is found to under-count by up to 9%. Is the request still admissible?

**Guided Practice:** Compare heuristic and exact/server counts across languages and tool schemas; test exact-boundary, over-limit, and atomic tool-call cases. Then redo the ledger with $T_{limit}=30{,}000$.

**Feedback Contract:**
- *Expected Output*: an estimator-error distribution, pinned protocol/tokenizer, a category ledger that reconciles to the limit, invariant tests, and a fallback. For Knowledge Check 2: the margin becomes $0.09\times12{,}000=1{,}080$, and $30{,}980+1{,}080=32{,}060>32{,}000$, so the request is **not** admissible; drop one more evidence chunk (500 tokens) or count evidence exactly. For the 30,000 limit: placed 30,980 already exceeds it by 980 before any margin, so caps alone are not enough; with the 600 margin the allocator must free about 1,580 tokens, for example by dropping g3 and g4 (1,400) and one more evidence chunk (500), which gives $29{,}080$ placed plus a $575$ margin $=29{,}655\le30{,}000$; or by compacting. (Arithmetic: $30{,}980+600-30{,}000=1{,}580$ to free; $30{,}980-1{,}400-500=29{,}080$; evidence is now $11{,}500$, so the margin is $0.05\times11{,}500=575$; unallocated $30{,}000-29{,}655=345$.) Rejecting is also a valid answer if the fallback says so.
- *Common Failure*: truncating individual messages without preserving call/result pairs; or shrinking the output reserve until the request fits.
- *Diagnostic Hint*: add up your "placed" column, the reserve, the margin, and the unallocated tokens. If the sum is not exactly the limit, a category is missing or counted twice. Then scan the placed history for a tool result whose call ID is absent.
- *Concept to Revisit*: admission invariant; atomic groups; output reserve.

**Learning Outcome:** Implement admission with explicit category budgets, valid-message invariants, fallback, and output reserve.

*(Effort: 55m instruction, 30m practice)*

### Lesson 11.2 — Advertised Window vs Effective Context

**Engineering Question:**
Can the model reliably use the required information across position, noise, multiplicity, and task complexity?

**Concepts & Definitions:**

Maximum accepted length asks “can the request be processed?” Effective context asks “can the model reliably use the required information for this task?” Measure a phase surface across:

- input length and evidence position;
- relevant-item count, distractor count, and similarity;
- exact lookup, multi-hop tracing, aggregation, long-dialogue, and code/state tasks;
- prompt/template and model revision;
- answer format, reasoning budget, latency, and cost.

A "phase surface" here is a table (or plot) of task success over a grid of these variables, so that each cell holds one combination such as (length, position, task type). The variables must be *crossed*: every length is tested at every position, so a change in success can be traced to one variable at a time.

RULER shows that simple needle retrieval can conceal failures on multi-needle, tracing, aggregation, and QA tasks as length grows (**O**, CLM-003). Lost in the Middle shows position sensitivity in evaluated models (**O**, CLM-004). Neither implies a universal degradation curve. A model upgrade can improve one slice and regress another.

**Worked Example:** A model can accept 128k tokens and retrieve one unique string near the end while failing multi-fact aggregation at a shorter length. Acceptance and one needle task therefore do not define the same envelope.

**Knowledge Check:** Which crossed variables are required to distinguish length from position effects?

**Guided Practice:** Sweep length, position, distractors, relevant count, hops, and model/prompt revision with paired seeds and no-answer controls.

**Feedback Contract:**
- *Expected Output*: a table or plot of task success by (length × position × task type), with the number of trials and an interval per cell, plus latency and cost per cell. The no-answer control should show the rate at which the model invents an answer.
- *Common Failure*: reporting maximum accepted length as usable memory; or varying length and position together, so a drop cannot be assigned to either.
- *Diagnostic Hint*: pick two cells that differ in only one variable. If you cannot find such a pair for length and for position, the design is not crossed.
- *Concept to Revisit*: accepted length versus effective context; crossed experimental variables (Module 00).

**Learning Outcome:** Publish workload-specific effective-context envelopes, not a single model-card number.

*(Effort: 40m instruction, 30m practice)*

### Lesson 11.3 — Memory Types and Source of Truth

**Engineering Question:**
Which record is authoritative state, which is immutable evidence, and which is a lossy derived view?

**Concepts & Definitions:**

Use types to encode different semantics:

- **immutable events:** what happened, with source, sequence, and time;
- **working/recent context:** a bounded recency window for the current task;
- **canonical state:** current preferences, decisions, entities, constraints, and versions;
- **episodic records:** selected prior interactions or outcomes;
- **semantic records:** durable facts with source and validity;
- **procedural artifacts:** reviewed strategies, plans, examples, or code;
- **summaries/reflections:** derived, lossy views that never silently replace raw provenance.

MemGPT is a reference for moving information between bounded in-context and external tiers (**O**, CLM-006). Generative Agents is a reference for an event stream, retrieval, and reflection (**O**, CLM-007). These are mechanism families, not proof of infinite, lossless, safe, or human-like memory.

**Worked Example — typed memory records** (synthetic records for one user `u1` in tenant `t1`).

Every record carries the same envelope: `id`, `type`, `owner`, `key`, `value`, `version`, `valid_from`/`valid_to`, `tx_time` (when the store accepted it), `source`, `derived_from`, `status`.

| id | type | key → value | version | valid from | tx time | source / derived from | status |
|---|---|---|---:|---|---|---|---|
| E-17 | immutable event | "use tabs in this repo" (user message) | — | 05-01 | 05-01 10:00 | session s3, turn 12 | live |
| E-19 | immutable event | "actually, spaces; the linter needs it" | — | 05-03 | 05-03 09:00 | session s5, turn 4 | live |
| C-2 | canonical state | `pref:indent` → `tabs` | 3 | 05-01 | 05-01 10:00 | derived from E-17 | superseded by C-2 v4 |
| C-2 | canonical state | `pref:indent` → `spaces` | 4 | 05-03 | 05-03 09:00 | derived from E-19 | live |
| S-8 | summary | "User prefers tabs and dislikes linters" | 1 | — | 05-02 02:00 | derived from E-17 only | stale: built before E-19 |
| F-4 | semantic record | `project:runtime` → `Python 3.11` | 2 | 04-10 | 04-10 | `pyproject.toml` at a recorded commit | live |
| P-1 | procedural artifact | release checklist | 5 | 04-20 | 04-20 | reviewed by owner | live |

Steps, for the question "what indentation should I use?" asked on 05-04:

1. The read policy filters by owner and tenant, then by status and validity. C-2 v3 is superseded and S-8 is stale; both are excluded *before* any similarity ranking.
2. C-2 v4 is the canonical answer: `spaces`. Its lineage points to E-19, which can be shown as the reason.
3. S-8 is closer in wording to the question than C-2 is, so a similarity-only read would pick it and answer `tabs`, and would add an invented claim about linters.
4. S-8 is rebuilt from E-17 and E-19, becoming version 2. E-17 is not deleted: it is still a true record of what was said on 05-01.

Interpretation and limits: the types differ in which fields carry the meaning. Events have no version because they never change. Canonical state has one live version per key. A summary is valid only relative to the events it was derived from. The schema is one workable design, not a standard; what matters is that each type has an explicit source-of-truth rule.

**Knowledge Check:**
1. Why must summaries retain derived-from lineage?
2. In the table, which field shows that S-8 is unsafe to use, without reading its text?

**Guided Practice:** Classify an event log, current preference, prior episode, procedure, and summary; define owner, authority, times, supersession, and deletion behavior. Then add the record that results when the user says "forget my indentation preference" on 05-05.

**Feedback Contract:**
- *Expected Output*: typed schemas and explicit source-of-truth rules. For the "forget" request: a new event E-20; C-2 becomes a tombstone at version 5 with a deletion reason; S-8 must be rebuilt without the preference or tombstoned; whether E-17 and E-19 are erased or retained depends on the declared deletion policy (a user-requested erasure normally reaches the raw events too, and then the summary cannot be rebuilt from them). Knowledge Check 2: `derived_from` lists only E-17 while a later event E-19 exists for the same key, and its `tx_time` precedes E-19.
- *Common Failure*: allowing a reflection or summary to silently overwrite canonical state; or deleting the canonical record but leaving the summary that repeats it.
- *Diagnostic Hint*: for each record ask "if this were wrong, which other record would I use to prove it?" If the answer is "none", the record has no source of truth.
- *Concept to Revisit*: canonical state versus derived view; valid time versus transaction time.

**Learning Outcome:** Define record schemas, authority, ownership, valid/transaction time, and correction paths for every memory type.

*(Effort: 45m instruction, 25m practice)*

### Lesson 11.4 — Write, Read, Update, and Forget

**Engineering Question:**
How does memory converge under retries, concurrent corrections, expiry, deletion, and access constraints?

**Concepts & Definitions:**

A write policy asks whether an event is eligible, novel, durable, attributable, consented, and safe to retain. Store stable identity, source, confidence, privacy class, valid time, transaction time, TTL, supersession, and derived-from lineage. Append-only growth can retain prompt injection, transient mood, wrong tool output, or another user's data (**D**, CLM-008).

A read policy first applies hard access, validity, deletion, and task constraints; only then rank by workload-tested relevance, recency, importance, authority, confidence, and prior utility. Similarity alone does not prove applicability (**D**, CLM-009). Selected records need lineage in the request trace.

In plain terms, reading is two steps. Step 1 is a set of yes/no gates (right owner and tenant, not deleted, not superseded, valid now, allowed for this task); a record that fails any gate is removed. Step 2 orders only the survivors by score. Because the gates run first, no ranking score, however high, can bring back a record the gates removed. Lesson 11.3's S-8 is the example: it is the closest match in wording but fails the validity gate.

Concurrent or multi-writer updates need an explicit compare/version rule. Conditional writes, monotonic sequence or transaction versions, and a domain conflict policy make lost updates observable; last-writer-wins is valid only when its clock and product semantics are declared. Backfills, TTL expiry, tombstones, indexes, replicas, and caches must converge before physical garbage collection removes recovery evidence.

Forgetting is a capability: expiry, correction, user deletion, policy deletion, invalidation after failure, and capacity-based eviction differ. Tombstones and supersession must reach all indexes/caches. A 2025 study reports experience-following, error propagation, and misaligned replay in its evaluated agents—use this as a failure family, not a universal rate.

**Worked Example — concurrent correction and tombstone trace** (synthetic; continues the `pref:indent` record of Lesson 11.3, stored at version 4 = `spaces`).

Write rule for canonical state (**D**, CLM-018): every write carries an operation ID and the base version the writer read. The store, in one atomic step per key, (a) returns the stored outcome if the operation ID is already recorded, (b) accepts if the base version equals the stored version, writing version + 1, and (c) rejects as stale otherwise. A delete is a write that produces a tombstone with its own version.

In plain terms: the *base version* is the version the writer saw when it last read the key. "Accept only if base = stored" means "accept only if nobody else has written since you read". The operation ID catches the same write arriving twice; the base version catches a write built on an out-of-date read.

| # | Actor | Operation (op ID, base) | Stored before | Decision | Stored after |
|---|---|---|---|---|---|
| 1 | device A | read | v4 `spaces` | — | v4 |
| 2 | device B | read | v4 `spaces` | — | v4 |
| 3 | device A | set `2-spaces` (a1, base 4) | v4 | ACCEPT | v5 `2-spaces` |
| 4 | device B | set `tabs` (b1, base 4) | v5 | REJECT: stale base | v5 |
| 5 | device A, network retry | set `2-spaces` (a1, base 4) | v5 | DUPLICATE: a1 already recorded; return its outcome | v5 |
| 6 | user | delete (d1, base 5) | v5 | ACCEPT | tombstone v6, reason: user request |
| 7 | summarizer job that read v5 before step 6 | write "prefers 2-spaces" (j1, base 5) | tombstone v6 | REJECT: stale base | tombstone v6 |
| 8 | queue redelivery of step 3 | set `2-spaces` (a1, base 4) | tombstone v6 | DUPLICATE if a1 is still recorded; otherwise REJECT: stale base | tombstone v6 |
| 9 | memory search index still holds the v5 embedding | read for a new request | tombstone v6 | selected hit is checked against the canonical store and excluded | tombstone v6 |
| 10 | garbage collector | remove tombstone | tombstone v6 | allowed only after every index, replica, and cache reports version ≥ 6 and the redelivery window has passed | — |

Interpretation:

- Step 4 is a lost update under an unconditional write: B would overwrite A with no trace. Here B is told, re-reads v5, and either asks the user or applies a declared domain rule. Which value should win is a product decision; that the conflict is *visible* is the engineering requirement.
- Steps 7 and 8 are the delayed writes that would bring a deleted preference back. Both are rejected by the base-version check, and step 8 has a second guard in the operation ID. Under last-writer-wins, step 7 resurrects the record.
- Step 9 shows that deletion is not finished when the canonical store has the tombstone. Until the index converges, the read path must validate hits against canonical state.
- Step 10: collecting the tombstone early reopens steps 7 and 8, because a write with base "none" would be accepted as a new record.

Limits: the rule needs an atomic compare-and-write per key; two-step check-then-write lets A and B both pass at step 3–4. The trace is a design walk-through, not an executed test. Module 10 Lesson 10.2 applies the same idea to index events ordered by a source version, and Module 14 Lesson 14.2 to external effects.

**Knowledge Check:**
1. Why must access/validity filtering precede similarity ranking?
2. At step 8, which of the two guards still works if operation IDs are retained for only one hour and the redelivery arrives after two?

**Guided Practice:** Inject duplicate, reordered, concurrent, corrected, expired, deleted, and cross-tenant records; verify convergence across primary store, index, replica, cache, and context trace. Then extend the trace: after step 6 the user sets a new preference `tabs` (op n1). Which base version must that write carry, and what must happen to the tombstone?

**Feedback Contract:**
- *Expected Output*: version/conflict logs, lineage, authorization checks, tombstone/GC state, and convergence probes. Knowledge Check 2: the base-version check; base 4 does not equal 6, so the write is rejected as stale. Extension: the new write reads the tombstone and carries base 6; the store accepts it as version 7, a live record that supersedes the tombstone. A write with base "none" must be rejected while the tombstone exists, otherwise step 7 could use the same path.
- *Common Failure*: deleting only the primary record; or deduplicating by operation ID alone and then expiring the IDs.
- *Diagnostic Hint*: for each rejected or duplicate row, name the field that the store compared. For each read, name the check that stops a stale index hit from reaching the context.
- *Concept to Revisit*: conditional write with base version; tombstone and garbage-collection boundary; validity filter before ranking.

**Learning Outcome:** Survive duplicated/out-of-order writes, corrections, deletes, poisoning, and distribution shift without replaying invalid state.

*(Effort: 60m instruction, 35m practice)*

### Lesson 11.5 — Truncation, Summarization, and Compression

**Engineering Question:**
Which compaction policy preserves required semantics and lowers end-to-end cost for the target workload?

**Concepts & Definitions:**

Compaction options have different failure modes:

- sliding windows preserve recent verbatim turns but forget older dependencies;
- deterministic selection preserves chosen fields but fails on an incomplete schema;
- summaries save tokens but can omit qualifiers, time, negation, disagreement, and source;
- retrieval externalizes history but can miss or misrank it;
- learned/token compression adds another model, overhead, and corruption surface.

Treat summaries as versioned derived artifacts. Link each claim to source events, allow rebuild, and periodically compare with raw history. Recursive summarization can turn an interpretation into an apparent fact (**D**, CLM-010).

Define:

$$
r=\frac{T_{retained}}{T_{original}},\qquad
\Delta L_{e2e}=L_{compress}+L_{model,compressed}-L_{model,original}.
$$

where:

- $T_{original}$: tokens of the content before compaction; $T_{retained}$: tokens kept after compaction. $r$ is between 0 and 1 for a compaction that removes tokens; smaller $r$ means more compression. Example: 10,000 tokens compacted to 2,500 gives $r=2{,}500/10{,}000=0.25$.
- $L_{compress}$: time spent by the compaction step itself (preprocessing, a summarizer or compressor call).
- $L_{model,original}$ and $L_{model,compressed}$: model time for the request with the original and with the compacted content.
- $\Delta L_{e2e}$: change in end-to-end latency caused by compacting. Negative means faster; positive means slower. ($L$ is time; $T$ is tokens.)

Break-even, step by step: compaction is faster only when $\Delta L_{e2e}<0$, that is
$L_{compress}+L_{model,compressed}<L_{model,original}$, which rearranges to
$L_{compress}<L_{model,original}-L_{model,compressed}$.
The compaction step must cost less time than it saves in model time. Note that $r$ does not appear in $\Delta L_{e2e}$: a smaller $r$ shrinks the right-hand side only to the extent it actually reduces measured model time.

Token reduction does not imply latency reduction. A 2026 study found operating regions where compression helped and regions where preprocessing erased the benefit. Measure quality, protocol validity, memory, queueing, and end-to-end latency at the target model/hardware/load.

**Worked Example:** A compressor that saves 500 ms of model time but adds 700 ms preprocessing has $\Delta L_{e2e}=+200$ ms and is slower at that operating point despite reducing tokens. Step by step: $L_{model,compressed}-L_{model,original}=-500$ ms and $L_{compress}=700$ ms, so $\Delta L_{e2e}=700+(-500)=+200$ ms. Break-even check: $700<500$ is false.

**Knowledge Check:** Why can recursive summary claims become false canonical facts?

**Guided Practice:** Compare full, windowed, structured, source-linked summary, retrieval, and learned compression across ratio, depth, hardware, and load.

**Feedback Contract:**
- *Expected Output*: for each compaction method, the ratio $r$, semantic preservation by slice (negation, corrections, long-range dependencies), protocol validity, the three timing terms of $\Delta L_{e2e}$ measured separately, cost, and a raw-replay check. A method is reported as faster only where $\Delta L_{e2e}<0$ at the stated model, hardware, and load.
- *Common Failure*: reporting compression ratio as latency speedup. In the worked example the ratio improves and the request is 200 ms slower.
- *Diagnostic Hint*: which of the three terms did you measure, and which did you infer from token counts? An inferred model-time saving is a hypothesis until it is timed under the same load.
- *Concept to Revisit*: end-to-end latency delta; summaries as versioned derived artifacts (Lesson 11.3).

**Learning Outcome:** Choose compaction from a semantic-loss and end-to-end frontier, with raw-state recovery.

*(Effort: 45m instruction, 30m practice)*

### Lesson 11.6 — Memory Evaluation and Diagnosis

**Engineering Question:**
At which stored → valid → selected → placed → used boundary did the required state disappear or become harmful?

**Concepts & Definitions:**

Static long-context recall is not a complete memory test. Include:

- write/extraction precision and recall at event time;
- persistence, correction, deletion, TTL, authorization, and version convergence;
- valid-record retrieval and false-memory selection at probe time;
- context placement/order and survival after compaction;
- utilization given the required record is present;
- downstream task outcome, calibration/abstention, privacy leakage, latency, and cost;
- long-range understanding, test-time experience reuse, and selective forgetting.

MemoryAgentBench explicitly expands evaluation toward incremental interaction and four competencies. IFCMemoryBench (2026, one professional domain) separates ingestion, retrieval, and utilization and reports that the evaluated general-purpose memory systems performed poorly in its setup. Its abstract (re-read 2026-11-09) says those systems often retrieve topically relevant context but store project knowledge as incomplete or fragmented facts; full methods and results were not re-audited. Do not turn either benchmark into a universal leaderboard.

Diagnosis follows the earliest discriminated boundary:

```text
wrong response
 -> fact never written?
 -> written incorrectly or stale?
 -> valid fact not selected?
 -> selected but compacted/evicted/misordered?
 -> present but model ignored/misused it?
```

**Quantitative Model / Derivation:**
For each declared probe population, report stage opportunity conditionally: valid given stored, selected given valid, placed given selected, and correctly used given placed. Multiplying stage rates is justified only when the denominators and conditional chain match; retain raw counts and end-to-end success (**D**, CLM-019).

Notation: let $N_0$ be the number of probes and $N_1,\dots,N_5$ the counts that reach stored, valid, selected, placed, and used. Each stage rate is $p_k=N_k/N_{k-1}$ ("of those that reached the previous stage, the fraction that also passed this one"). Then

$$
p_1p_2p_3p_4p_5=\frac{N_1}{N_0}\cdot\frac{N_2}{N_1}\cdot\frac{N_3}{N_2}\cdot\frac{N_4}{N_3}\cdot\frac{N_5}{N_4}=\frac{N_5}{N_0},
$$

because each numerator cancels the next denominator. This cancellation is why the product is valid only when every $N_k$ is a subset of $N_{k-1}$ on the same probes.

**Worked Example — one cohort through five stages** (synthetic counts, registry CLM-020).

Cohort: 200 probes. Each probe needs exactly one specific memory record to be answered correctly at probe time. Each count is a subset of the row above it.

| Stage | Count | Conditional rate | Denominator |
|---|---:|---:|---|
| probes | 200 | — | — |
| stored: the needed record was written correctly | 180 | $180/200=0.900$ | all probes |
| valid: the stored record is current and authorized at probe time | 162 | $162/180=0.900$ | stored |
| selected: the read policy returned the valid record | 130 | $130/162\approx0.802$ | valid |
| placed: the selected record is in the serialized prompt | 117 | $117/130=0.900$ | selected |
| used: the answer applies the placed record correctly | 94 | $94/117\approx0.803$ | placed |

Steps:

1. Because each denominator is the previous numerator, the product telescopes: $0.900\times0.900\times0.802\times0.900\times0.803=94/200=0.47$. The chain rate equals the fraction of probes that succeeded *through the memory path*.
2. The largest single loss in counts is selection: $162-130=32$ probes. The next is use: $117-94=23$. Storage, validity, and placement lose 20, 18, and 13.
3. End-to-end correctness is a different number. Suppose 6 of the 106 probes that failed somewhere in the chain were still answered correctly, because the model guessed or knew the fact from elsewhere. End-to-end correct is then $(94+6)/200=0.50$, not 0.47.
4. A mismatched denominator breaks the product. Suppose the read policy returned 150 records in total for these probes, of which 130 were the valid needed ones. "Selection precision" is $130/150\approx0.867$. Substituting it for the selected-given-valid rate gives $0.9\times0.9\times0.867\times0.9\times0.803\approx0.508$, which matches nothing that happened. Precision answers "of what was selected, how much was right"; the chain needs "of what was valid, how much was selected".

Interpretation and limits: conditional rates locate the loss; the end-to-end rate measures the outcome; they agree only when no success bypasses the chain. The 6 bypass successes are not credit for the memory system, and a memory-disabled replay is how they are identified. If a fact is valid and selected but absent from the serialized prompt, retrieval tuning cannot fix the placement loss; replaying the same request with oracle placement discriminates that stage. Counts are synthetic and have no uncertainty attached; with 117 placed probes, a rate near 0.80 has a standard error of about 0.037. (Binomial standard error $\sqrt{p(1-p)/n}$ with $p=94/117\approx0.803$ and $n=117$: $\sqrt{0.803\times0.197/117}\approx\sqrt{0.00135}\approx0.037$.)

**Knowledge Check:**
1. Which oracle replay separates selection failure from model non-use?
2. An oracle-selection replay on the same cohort (every valid record is selected) gives 146 placed and 117 used. What are the new chain rate and the placed-given-selected rate, and what do they tell you?

**Guided Practice:** Run baseline plus oracle stored-state, oracle selection, oracle placement, and memory-disabled replays on stale, fragmented, corrected, and poisoned cases.

**Feedback Contract:**
- *Expected Output*: stage timestamps/identities, conditional counts on one declared cohort, privacy/deletion checks, downstream outcome, and remeasurement. Knowledge Check 2: chain rate $117/200=0.585$; placed-given-selected $146/162\approx0.901$; used-given-placed $117/146\approx0.801$. Fixing selection alone would raise the chain rate from 0.47 to about 0.585, and use becomes the largest remaining loss (29 probes). (Derivation: oracle selection makes selected $=162$, so the stage rates are $0.9,\ 0.9,\ 162/162=1,\ 146/162,\ 117/146$, and the product telescopes to $117/200$. Losses per stage: stored 20, valid 18, selected 0, placed $162-146=16$, used $146-117=29$.)
- *Common Failure*: one end-to-end score with no stage lineage; or multiplying rates whose denominators come from different populations.
- *Diagnostic Hint*: write the denominator next to every rate. Each one must be the numerator of the stage before it, on the same probes.
- *Concept to Revisit*: conditional stage rates versus end-to-end rate; oracle-stage replay.

**Learning Outcome:** Run oracle-stage replays and assign interventions to the failing stage, then remeasure.

*(Effort: 55m instruction, 30m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Liu et al. (TACL 2023).
- [RULER](https://arxiv.org/abs/2404.06654) — Hsieh et al. (COLM 2024).
- [MemGPT](https://arxiv.org/abs/2310.08560) — Packer et al. (2024 revision).
- [Generative Agents](https://arxiv.org/abs/2304.03442) — Park et al. (2023).

*Scope:* each entry supports only its evaluated models, tasks, and prompts. These four were not re-read in this revision and keep their earlier access dates.

**RECOMMENDED BASELINE (course position, not a surveyed industry default):** explicit context budgets/output reserve; model/protocol token accounting; valid-message boundaries; typed versioned records with conditional writes and tombstones; selective authorized write/read/update/delete; recoverable summary provenance; stage-attributed evaluation. These follow from the derivations in Lessons 11.1, 11.3, 11.4, and 11.6 (**D**, CLM-001, CLM-005, CLM-015, CLM-017, CLM-018, CLM-019). This module has no evidence about how widely they are deployed; registry entries labelled "CURRENT DEFAULT" mean this recommended baseline.

**WORKLOAD-DEPENDENT:** recency window, memory taxonomy detail, scoring weights, summary cadence, token compressor, context order, TTL, reflection, and effective-context envelope. On compression: [Prompt Compression in the Wild](https://arxiv.org/abs/2604.02985) (Kummer et al., arXiv v1, 2026-04-03) reports in its abstract that end-to-end gains appear only when prompt length, compression ratio, and hardware are matched, and that outside that window the compression step cancels the gain (**O**, CLM-011). Only the abstract was read; the paper's figures are not quoted here.

**FRONTIER (each item is scoped to the source named; abstracts read on 2026-09-30, full methods and results not re-audited):**

- Incremental memory-agent evaluation: [MemoryAgentBench](https://arxiv.org/abs/2507.05257) (Hu, Wang, McAuley; arXiv v4, 2026-06-28) names four competencies (accurate retrieval, test-time learning, long-range understanding, selective forgetting) and reports that the evaluated methods do not master all four (**O**, CLM-012).
- Experience reuse and its failure modes: [Xiong et al.](https://arxiv.org/abs/2505.16067) (arXiv v2, 2025-10-10) describe experience-following, error propagation, and misaligned experience replay (**O**, CLM-013).
- Domain-specific memory: [IFCMemoryBench](https://arxiv.org/abs/2607.26072) (Du et al., arXiv v1, 2026-07-13, a workshop paper) measures ingestion, retrieval, and utilization in one professional domain and argues for domain-aware memory representations (**O**, CLM-014).
- `TODO_VERIFY`: structured or graph memory architectures and learned selective forgetting were named in the earlier version of this list without a primary source in the registry. No source for them was opened, so the module makes no claim about them.

No universal architecture or benchmark winner is established by these sources.

**LEGACY / INSUFFICIENT:** replay the entire transcript; keep the last N messages without token/protocol checks; store everything forever; retrieve by similarity alone; treat summaries as facts; equate context length with usable memory; report compression ratio as speedup.

**PRODUCTION SOURCE TRACE**

- Repository: `langchain-ai/langchain`
- Revision: `80b74090515a594cc8421fb2d82e98f4e256c29f`
- Verified: 2026-09-30 (both files re-read at this revision), static inspection only; nothing was executed.
- Files/symbols: `libs/core/langchain_core/messages/utils.py::{trim_messages,count_tokens_approximately}` and `libs/core/langchain_core/chat_history.py::InMemoryChatMessageHistory.{add_message,clear}`.
- Observed (**O**, CLM-016):
  - `trim_messages` takes `strategy` (`"first"` or `"last"`, default `"last"`), `allow_partial`, `start_on`, `end_on`, and `include_system`. Its docstring says the *caller* should configure these so the result is a valid history, and notes that a `ToolMessage` can generally only appear after an `AIMessage` with a tool call.
  - `count_tokens_approximately` has defaults `chars_per_token=4.0` and `extra_tokens_per_message=3.0`, and its docstring says it may not match model-specific counts.
  - `InMemoryChatMessageHistory.add_message` appends to a list on the object, and `clear` replaces it with an empty list.
- Not verified: whether every `trim_messages` configuration keeps tool call/result pairs together. That was not established by reading and needs an executed test (`TODO_VERIFY`).
- Scope: pinned helpers, not a universal memory policy and not durable persistence.
- **Trace practice (the 2h `source_trace` effort):** at the pinned revision, read `trim_messages` and the helper it calls for `strategy="last"`. Answer in writing: (1) which arguments would you set to avoid a history that starts with a tool result, and where in the code is that enforced? (2) with the default approximate counter, what does a 4,000-character tool result count as, and what Lesson 11.1 ledger entry must cover the difference from the exact count? Expected answers: (1) `start_on` (for example `"human"`), applied in the last-strategy path; report the function and line you found, and say that you have not shown it holds for every configuration unless you ran it; (2) the function adds the content length, the tool-call ID length, and the role string length, divides by `chars_per_token` rounding up, and adds `extra_tokens_per_message`; for a 20-character call ID and the role `tool` that is $\lceil(4000+20+4)/4\rceil+3=1{,}009$ (step by step: $4000+20+4=4024$ characters; $4024/4=1006$, already whole so the ceiling $\lceil\cdot\rceil$ leaves it at 1006; $1006+3=1009$), and the difference from the real count must be covered by the measured estimator margin. A common error is to treat the helper's default as an exact count, or to assume trimming is protocol-safe without a test.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Context Accountant and Admission Gate

- **Objective:** build a serializer-aware context ledger and admission gate with output reserve and message invariants.
- **Pre-Registered Hypothesis:** heuristic count error will vary by language/protocol shape, requiring measured safety margin rather than one universal constant.
- **Independent Variables:** counter, language, tool/JSON/image shape, protocol/tokenizer revision, input length, and reserve.
- **Dependent Variables:** signed/absolute count error, admission errors, protocol validity, fallback, latency, and unused margin.
- **Break & Falsify:** underestimate framing, orphan tool messages, over-reserve output, change tokenizer, and test exact boundaries.
- **Alignment:** Lesson 11.1.
- **Effort Estimate:** 4h.

### LAB B — Effective-Context Phase Surface

- **Objective:** measure a workload-specific effective-context surface beyond accepted-length and needle baselines.
- **Pre-Registered Hypothesis:** position, distractor, multiplicity, and task complexity will interact with length, so no one accepted-window number predicts all outcomes.
- **Independent Variables:** length, position, relevant/distractor count/similarity, facts/hops/aggregation, task, and model/prompt revision.
- **Dependent Variables:** accuracy/support, utilization, abstention, latency, tokens, cost, and protocol failures.
- **Break & Falsify:** include long dialogue, structured state, code, multi-fact aggregation, and no-answer cases.
- **Alignment:** Lesson 11.2.
- **Effort Estimate:** 4.5h.

### LAB C — Versioned Memory Lifecycle

- **Objective:** implement typed/versioned records and prove write/read/correct/delete/conflict convergence.
- **Pre-Registered Hypothesis:** similarity-only, unconditional-write, or primary-store-only deletion baselines will fail at least one validity, privacy, lost-update, or convergence test.
- **Independent Variables:** memory type, duplicate/order/concurrency, correction/delete/TTL, ownership, poison, and replica/cache state.
- **Dependent Variables:** write precision/recall, conflicts, convergence, valid selection, false-memory/utilization, privacy violations, and recovery.
- **Break & Falsify:** replay the Lesson 11.4 trace as an executed test, then inject stale-base concurrent writes, redelivery after operation-ID expiry, cross-tenant facts, adversarial memory, fragmented knowledge, lagging indexes, and GC before convergence. Any run in which a deleted record is selected or restored falsifies the design. Records are synthetic unless their source is stated.
- **Alignment:** Lessons 11.3 and 11.4.
- **Effort Estimate:** 5h.

### LAB D — Compaction and Memory Regression

- **Objective:** compare compaction policies and attribute memory failure by stored/valid/selected/placed/used stage.
- **Pre-Registered Hypothesis:** no compaction method will preserve every semantic slice, and preprocessing/load will produce at least one point where fewer tokens do not reduce end-to-end latency.
- **Independent Variables:** method, ratio/depth, source age, dependency, protocol shape, hardware/load, and oracle-stage replay.
- **Dependent Variables:** semantic preservation, stage opportunity, downstream outcome, preprocessing/model/queue latency, memory, cost, privacy, and recovery.
- **Break & Falsify:** use negation, corrections, long-range dependencies, recursive summaries, tool protocols, poisoning, and loaded queues.
- **Alignment:** Lessons 11.5 and 11.6.
- **Effort Estimate:** 4.5h.

## 07 Break / Incident Scenarios

### Incident 11.1 — Helpful Memory Release Causes Repeated Wrong Actions

**Incident Symptoms:**
A memory release lowers average prompt tokens and improves repeat-task success, but some users see old preferences after correction, another tenant's detail appears in a response, tool-call histories become invalid after trimming, and one previously successful but wrong execution is repeatedly replayed.

**Diagnostic Protocol (Task):**
1. **Form Competing Hypotheses:** tokenizer/framing undercount, reserve exhaustion, invalid trimming, extraction error, ownership-filter failure, lost/out-of-order update, incomplete deletion/GC, summary drift, stale selection, experience replay, position loss, or model non-use.
2. **Identify Missing Evidence:** raw events, versions/owners/valid and transaction times, conflict/tombstone/index/cache state, compaction lineage, exact prompt, counter/version/error, selected order, protocol structure, reserve, model/prompt, and stage timings.
3. **Design Discriminating Measurements:** replay with oracle stored/valid state, oracle selection, oracle placement/uncompressed context, changed position, and memory disabled; probe every replica/cache.
4. **Rank Explanations:** locate the earliest stored → valid → selected → placed → used failure and retain interacting privacy/protocol causes.
5. **Intervene:** gate the route, repair ownership/version/conflict/delete propagation, rebuild summaries/indexes, change compaction/placement, or disable harmful experience reuse.
6. **Remeasure:** repeat stage opportunity, corrections/deletes, privacy, protocol validity, task utility, abstention, latency, cost, and rollback checks.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem — Multi-Session Technical Assistant

Design context and memory for a multi-session technical assistant that uses tools, honors concurrent corrections/deletion, remembers project decisions, and operates under strict context, latency, privacy, and cost constraints.

**Constraints (SYNTHETIC fixture — exercise inputs, not properties of a real model or product).** A submission may replace a value with its own measurement if it says so; it is then graded against its own declared inputs.

| Item | Fixture value |
|---|---|
| Context limit | 24,000 tokens, input and output combined |
| Output reserve | 1,500 tokens, fixed |
| Fixed input | system 1,000; tool schemas 3,000; framing 150 (all exact counts) |
| Typical request before allocation | current input 800; canonical state 500; selected memory 3,500; evidence 9,000 (heuristic count, measured worst-case under-count 4%); history 9,400 in atomic groups of 600, 2,800 (tool call + result), 400, 1,200, 3,400 (tool call + result), 1,000, oldest first |
| Caps | memory 3,000; evidence 8,000 in 400-token chunks; history 6,000 |
| Memory writers | two user devices and one background summarizer; redelivery window 30 minutes |
| Deletion (hard) | a user-deleted record is never placed in context after the delete is accepted, and is never resurrected |
| Tenancy (hard) | zero cross-tenant records selected |
| Latency | p95 added latency from memory read plus compaction $\le300$ ms |

**Required Deliverables:**
1. serializer-aware category ledger for the fixture request: requested, placed, dropped IDs, margin, and reconciliation to the limit, with output reserve, admission decision, and protocol invariants;
2. effective-context phase surface across length, position, noise, multiplicity, and task complexity;
3. typed/versioned schemas with authority, ownership, valid/transaction time, lineage, and source of truth;
4. write/read/correct/conflict/forget policy with deletion, TTL, tombstone, index/cache convergence, and GC boundary, including a decision trace for two concurrent corrections, a duplicate redelivery, a delete, and a delayed summarizer write;
5. poisoned, stale, fragmented, concurrent-update, protocol, and cross-tenant tests;
6. compaction semantic-loss and end-to-end performance frontier with raw recovery, reporting $\Delta L_{e2e}$ from its three measured terms;
7. stored/valid/selected/placed/used counts on one declared cohort, the chain rate, the end-to-end rate, and oracle-stage replays;
8. source trace of a pinned context/memory helper with explicit non-durability/generalizability boundary;
9. canary, rollback, and Incident 11.1 diagnosis.

## 09 Required Evidence & Rubric

### Required Artifact: Context and Memory Lifecycle Trace

Submit an exact serialized request ledger, typed record/change history, conflict/delete convergence probes, selected-memory lineage, compaction comparison, oracle-stage diagnosis, source trace, and release/rollback decision.

### Reference Checks for Mastery Deliverables 1, 4, and 7 (fixture inputs only)

Reviewers use these to check arithmetic and decisions. A submission with different declared inputs is checked against its own inputs.

- **Ledger (Deliverable 1):** requested total 28,850, which is 4,850 over the limit. After caps: memory 3,000; evidence 8,000; history 6,000 (the 600 and 2,800 groups are dropped whole, leaving 400 + 1,200 + 3,400 + 1,000). Placed total 23,950, slack 50. The evidence margin is $0.04\times8{,}000=320>50$, so the request is **not yet admissible**. Dropping one more 400-token evidence chunk gives placed 23,550 and margin 304: $23{,}854\le24{,}000$, admitted with 146 unallocated. Equivalent answers (exact evidence counting, or a declared compaction) are accepted if they reconcile. An answer that lowers the 1,500 reserve is not.
  - Arithmetic: requested $1{,}000+3{,}000+150+800+500+3{,}500+9{,}000+9{,}400+1{,}500=28{,}850$, and $28{,}850-24{,}000=4{,}850$. History: $9{,}400-600=8{,}800>6{,}000$, then $8{,}800-2{,}800=6{,}000\le6{,}000$. Placed: $28{,}850-500-1{,}000-3{,}400=23{,}950$; slack $24{,}000-23{,}950=50$. After one more chunk: evidence $7{,}600$, margin $0.04\times7{,}600=304$, $23{,}550+304=23{,}854$, unallocated $24{,}000-23{,}854=146$.
- **Protocol check (Deliverable 1):** no placed tool result without its call; the 2,800 group is absent as a whole.
- **Mutation trace (Deliverable 4):** the expected decisions follow Lesson 11.4: first correction ACCEPT, second correction from the same base REJECT as stale, redelivery DUPLICATE, delete ACCEPT as tombstone, delayed summarizer write REJECT as stale, tombstone kept at least 30 minutes and until indexes and caches converge. A design in which the summarizer write or a redelivery restores the deleted value fails the hard deletion constraint.
- **Stage counts (Deliverable 7):** each stage count must be a subset of the previous one on the same cohort; the product of the conditional rates must equal used/probes; the end-to-end rate is reported separately with the memory-disabled replay that explains any difference.

### Rubric Dimensions

- **Budget:** *Insufficient* counts visible text only, or makes the request fit by shrinking the output reserve or by cutting inside a tool call/result pair. *Competent* pins tokenizer/protocol/model and reserve, and produces a ledger that reconciles to the limit with whole-group drops. *Strong* measures estimator error, multimodal/tool framing, invariants, and fallback.
- **Effective Context:** *Insufficient* quotes window size. *Competent* crosses length/position/noise. *Strong* maps multiplicity/complexity/revisions with uncertainty and operational cost.
- **Memory Lifecycle:** *Insufficient* stores untyped text, or uses unconditional last-writer-wins so that a delayed write restores a deleted record. *Competent* versions owner/time/TTL/update/delete and gives the correct decision for each row of the mutation trace. *Strong* proves concurrent conflict, index/cache convergence, GC safety, and lineage with executed tests.
- **Selection/Compaction:** *Insufficient* uses similarity or token ratio alone. *Competent* applies hard validity/access and measures semantic loss. *Strong* includes recursive drift, protocol validity, preprocessing, load, and recovery.
- **Diagnosis:** *Insufficient* blames memory generally, or multiplies rates with mismatched denominators. *Competent* separates stages with conditional counts on one cohort and keeps the end-to-end rate separate. *Strong* uses oracle replays to locate earliest failure and tests competing causes.
- **Operations:** *Insufficient* omits privacy/deletion. *Competent* demonstrates them. *Strong* includes poisoning, rollback, raw-event recovery, and scoped source behavior.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Context accounting and admission | Lesson 11.1 token-counting facts and category ledger | Lesson 11.1 Guided Practice (30,000-token ledger); LAB A | Mastery Deliverable 1; Incident 11.1 steps 1–3 (undercount, reserve, invalid trimming); Budget rubric row | Reconciled ledger with dropped IDs; estimator-error report; Section 09 Reference Checks |
| Effective-context measurement | Lesson 11.2 | Lesson 11.2 Guided Practice; LAB B | Mastery Deliverable 2; Effective Context rubric row | Length × position × task table with intervals |
| Typed memory records and source of truth | Lesson 11.3 record table | Lesson 11.3 Guided Practice ("forget" request) | Mastery Deliverable 3; Memory Lifecycle rubric row | Typed schemas with lineage and source-of-truth rule |
| Write/read/correct/forget convergence | Lesson 11.4 mutation trace | Lesson 11.4 Guided Practice (write after tombstone); LAB C | Mastery Deliverables 4–5; Incident 11.1 steps 1–5 (lost update, incomplete deletion, cross-tenant); Memory Lifecycle and Operations rubric rows | Decision trace per write; tombstone/GC state; convergence probes |
| Compaction choice | Lesson 11.5 (200 ms example) | Lesson 11.5 Guided Practice; LAB D | Mastery Deliverable 6; Selection/Compaction rubric row | Semantic-loss table and three-term latency delta |
| Stage attribution and diagnosis | Lesson 11.6 cohort table | Lesson 11.6 Knowledge Check 2 and Guided Practice; LAB D | Mastery Deliverable 7; Incident 11.1 steps 3–6; Diagnosis rubric row | Conditional counts on one cohort; oracle and memory-disabled replays |
| Production source trace | Section 05 LangChain trace | Section 05 Trace practice, questions 1–2 | Mastery Deliverable 8 | Pinned trace (revision, file, symbol) with stated unverified points |
| Release and rollback | Lessons 11.4 and 11.6 | LAB C recovery; LAB D regression | Mastery Deliverable 9; Operations rubric row | Canary/rollback record and incident diagnosis |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can account for serialized context and output reserve; distinguish accepted from effective length; reject human-memory analogies without software semantics; survive concurrent update/delete/TTL/privacy/index/cache and replay failures; prove summaries are recoverable derived state; identify compression break-even rather than assume it; trace current source without generalizing it; and locate failures across stored, valid, selected, placed, and used boundaries.

### Module Wrap-Up (Final Mental Model Reconstruction)

Context is scarce execution input; memory is a governed state system. Reliability comes from budgeting, typed/versioned records, explicit conflict and deletion convergence, selective authorized movement into context, recoverable compaction, and stage-level falsification—not from replaying more text.

### Cost & Telemetry Hook

- [Lesson 22.3](../22-ai-economics/README.md): record per-call input tokens from the provider count, not the approximate counter; reconcile against the context-budget ledger.
- [Lesson 22.2](../22-ai-economics/README.md): price compaction/summarization calls and memory storage as fixed plus marginal cost against the replayed-context cost they replace.
- [Lesson 23.1](../23-observability-reliability/README.md): emit budget, reserved output, trimmed count, and memory record IDs/versions selected per call; [Lesson 23.6](../23-observability-reliability/README.md) governs retention and redaction of memory content in telemetry.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
