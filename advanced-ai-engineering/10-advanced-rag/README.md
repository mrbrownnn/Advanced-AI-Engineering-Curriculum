# Module 10 — Advanced RAG

## 00 Why This Module Exists

RAG is not “retrieve top-k and paste it into a prompt.” It is a versioned evidence system whose answer depends on source state, index visibility, query transformations, candidate retrieval, context selection and order, generator behavior, citation binding, and conflict policy.

```text
source change -> parse/chunk/embed -> candidate index -> query-visible revision
                                                       |
query -> preserve/rewrite/decompose/route -> retrieve/rerank
      -> select/deduplicate/order context -> generate claims
      -> bind citations -> verify/support/abstain -> response
```

A wrong answer can therefore arise because evidence did not exist, was not yet visible, was not retrieved, was omitted from context, was displaced by order or noise, contradicted another version, was ignored by the model, or was cited incorrectly. Those failures require different fixes.

This module owns query orchestration, incremental-index freshness and version consistency, context assembly, contradiction handling, citations/grounding, staged RAG diagnosis, and RAG-versus-long-context decisions. Module 08 owns upstream data lineage, Module 09 owns retrieval/index/fusion/reranking mechanics, Module 11 owns general context and memory engineering, and Module 15 owns the organization-wide evaluation program.

**Research cutoff:** 2026-09-30. Sources marked as re-opened on that date are listed in the registry; other entries keep their original access dates.

**Module Orientation**

- **Engineering problem:** deliver evidence-backed answers from a mutable corpus under quality, freshness, consistency, latency, cost, access, and abstention constraints.
- **What you will do:** define a versioned RAG request contract; implement replay-safe updates and deletes; measure query-visible freshness; compare query and context policies; resolve temporal/authority conflicts; audit claim-level citations; test RAG against long-context and hybrid routes; and diagnose stale unsupported answers.
- **Environment:** Python 3.10+ with a versioned document store/search fixture and reproducible generation endpoint; optional distributed index and load generator for replica/cache/failure tests. Pin source, index, query transformation, prompt, model, evaluator, and cache revisions.
- **Evidence rule:** keep source observations (**O**), derivations (**D**), and telemetry-dependent hypotheses (**H**) distinct. A paper result remains scoped to its model, corpus, retriever, prompt, task, and metric.

## 01 Baseline Assumptions

- Module 00: measurement boundaries, paired tests, uncertainty, and falsification.
- Module 07: behavioral failures, uncertainty, and selective prediction.
- Module 08: stable identity, provenance, dataset versions, and deletion/rollback.
- Module 09: lexical/dense retrieval, exact/ANN separation, fusion, reranking, and retrieval metrics.
- Module 04: latency distributions, queueing, goodput, overload, and capacity.
- Mathematical prerequisites: sets, conditional probability intuition, constrained optimization, critical paths, timestamps, and empirical distributions.

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
  economics: REQUIRED
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 5h        # lesson instruction: 35+65+45+50+55+50 min = 300 min
  guided_practice: 3h    # lesson practice: 30+35+30+25+30+30 min = 180 min
  labs: 18h              # LAB A 4.5h + LAB B 4.5h + LAB C 4.5h + LAB D 4.5h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 10.1 0.5h
  source_trace: 2h       # Section 05 Haystack trace and Mastery Deliverable 8, counted once
  total: 31h
```
Each category is counted once. Lab analysis is not also counted as guided practice, and the source trace is not also counted as assessment time.

The learner must be able to define an end-to-end RAG contract; build idempotent incremental updates and deletes; prove query visibility and snapshot lineage; test query rewrites and context order; resolve or expose contradictions; validate citations at claim level; attribute failures by stage; and choose RAG, long context, or a hybrid route from matched evidence.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
                    mutable source-of-truth
                             |
             stable ID + version + event time
                             v
      parse -> chunk -> embed -> write -> refresh/commit
                             |
                   visible high-watermark
                             |
request contract -> query transform -> retrieve/rerank
                             |
            evidence eligibility + context budget
                 /           |            \
          authority       conflict       ordering
                 \           |            /
                    assembled context
                             |
                  generator claims
                             |
            citation binding + verification
                             |
                  answer or abstention
```

Every request must preserve enough lineage to answer: which source versions were eligible, which were actually retrieved, what entered the prompt and in what order, which claim cites which immutable evidence span, and which policy resolved a conflict.

## 04 Lessons

### Lesson 10.1 — The RAG Request and Evidence Contract

**Engineering Question:**
Can a delivered answer be reconstructed from an immutable request and evidence boundary?

**Concepts & Definitions:**

Pin the request's query intent and time, tenant/access scope, corpus or index revision, original and transformed queries, retrieval/context budgets, model and prompt versions, answer format, citation semantics, and abstention policy. “Latest” is not a version. A response built from unrecorded mutable state cannot be reproduced or audited.

The original RAG architecture is a reference: generation is conditioned on retrieved non-parametric evidence (**O**, CLM-001). Production systems add many lossy boundaries. Keep three values separate:

- **available:** the required evidence exists in the authorized source snapshot;
- **opportunity:** the required evidence reaches candidates or assembled context;
- **use/support:** the generator uses it correctly and citations entail the claims.

**Worked Example:** Replay one request against its recorded snapshot and against “latest.” Different answers under latest show why a mutable alias is not sufficient lineage.

**Knowledge Check:** Distinguish available evidence, candidate/context opportunity, and correct use/support.

**Guided Practice:** Specify and validate a request trace containing all revisions, transformations, candidates, context order, claims, citations, and finish state.

**Feedback Contract:**
- *Expected Output*: a trace record whose replay against the recorded snapshot reproduces the same eligible set, candidates, context order, and citation targets. Any dependency that cannot be replayed (for example, a hosted model revision that is no longer served) is listed explicitly as unavailable.
- *Common Failure*: storing only the final prompt and answer. The prompt shows what the model saw. It does not show which evidence was eligible but not retrieved, or which query transformation ran.
- *Diagnostic Hint*: remove one field from your trace. Can you still tell whether a wrong answer came from a missing source, an invisible version, or a retrieval miss? If you can, that field was not doing any work.
- *Concept to Revisit*: available versus opportunity versus use/support.

**Learning Outcome:** Capture a complete request trace rather than only the final prompt and answer.

*(Effort: 35m instruction, 30m practice)*

### Lesson 10.2 — Incremental Indexing, Visibility, and Consistency

**Engineering Question:**
When is a source update or deletion actually visible to production-equivalent reads?

**Concepts & Definitions:**

An update lifecycle is not complete when a worker reports success:

```text
source event -> detected -> parsed -> chunked -> embedded -> written
             -> refreshed/committed -> replicated/routed -> cache-visible
             -> retrieved by a production-equivalent probe
```

Define the freshness measurand:

$$
L_{visible}=t_{query\ visible}-t_{source\ event}.
$$

where:

- $t_{source\ event}$: the time the change happened at the source of truth (the declared start endpoint);
- $t_{query\ visible}$: the first time a production-equivalent query (same route, replica, and cache path a user would hit) returns the new version;
- $L_{visible}$: the visibility lag in seconds for one event on one read path. Each event and path gives one sample; the SLO is stated over the distribution of samples (for example its p95).

A useful split, used in the worked example below, puts the write acknowledgement time $t_w$ between the two endpoints:

$$
L_{visible}=\underbrace{(t_w-t_{source\ event})}_{\text{before the write}}+\underbrace{(t_{query\ visible}-t_w)}_{\text{after the write}}.
$$

This is an identity (add and subtract $t_w$), so it always holds for one event. What it does not give is a per-stage breakdown inside each bracket, because stages can overlap.

The relation is exact for declared endpoints; decomposing it into stage times requires the actual dependency graph because work may overlap. Record event time semantics and clock skew. Report a distribution and stale-read fraction by source/slice, not only a mean. In Elasticsearch, for example, a write can return before it is searchable, and refresh trades visibility latency against indexing and search work (**O**, CLM-008).

**Stable IDs and versions are inputs to idempotency; they do not create it.** A retry is safe only if the *store* applies an acceptance rule to every write, and applies it atomically. Each logical source item has a stable `doc_id`. The source assigns each change a monotonically increasing `version` (an event sequence number or source revision, never a worker's wall clock). The store keeps one record per `doc_id`: `(version_s, content_hash_s, state_s ∈ {live, tombstone})`. An incoming event `(op, doc_id, version_e, hash_e)` is decided as follows (**D**, CLM-018):

| Stored state | Incoming event | Decision | Store after | Acknowledge source as |
|---|---|---|---|---|
| no record | upsert / create | ACCEPT | live, `version_e` | applied |
| no record | delete | ACCEPT as tombstone | tombstone, `version_e` | applied |
| any | `version_e > version_s` | ACCEPT (upsert → live, delete → tombstone) | new state, `version_e` | applied |
| any | `version_e = version_s`, same op, same hash | DUPLICATE (no write) | unchanged | applied (return the stored outcome) |
| any | `version_e = version_s`, different op or hash | CONFLICT (no write, alert) | unchanged | rejected; the same key was reused with a different intent |
| any | `version_e < version_s` | STALE (no write) | unchanged | superseded |

Four conditions make this rule an idempotency guarantee rather than a hope:

1. **Atomicity.** Compare and write must be one atomic step per `doc_id`: a conditional write, a compare-and-set, or a single-writer partition. "Read the stored version, then write" in two steps lets two workers both pass the check. Elasticsearch implements such conditions: `version_type=external` fails the write with a 409 conflict when the supplied version is less than or equal to the stored one, `if_seq_no`/`if_primary_term` make a write conditional on the last modification, and `op_type=create` fails if the ID already exists (**O**, CLM-019). Those are one store's semantics, not a definition for every vector store.
2. **Same rule in every representation.** Text, chunks, embeddings, metadata, and citation-target stores each apply the rule with the *source* version, not a local counter. A tombstone that reaches lexical search but not vector search is not a completed delete.
3. **Tombstones outlive the replay window.** A delete must leave a versioned tombstone. If it is physically removed (or the delete was unversioned) before every delayed or retried event for that `doc_id` has arrived, a late upsert meets "no record" and is accepted. The deleted document comes back.
4. **Acknowledge after the decision is durable.** The ingestion worker acknowledges a source event only after the ACCEPT/DUPLICATE/STALE/CONFLICT outcome is durably recorded. That acknowledgement means "decided", not "query-visible".

This is the same contract Module 14 (Lesson 14.2) develops for external effects: the receiver stores identity, canonical intent (here, version plus content hash), and outcome atomically, and rejects reuse of a key with different intent. Module 10 applies it to index writes. Durable workflow recovery stays in Module 14.

**Solved traces for one `doc_id`** (synthetic events; hashes `hA ≠ hB`):

| Trace | Events in arrival order | Decisions | Final state | What last-writer-wins would do |
|---|---|---|---|---|
| Reordered retry | upsert v2 hA; upsert v1 hB; upsert v2 hA | ACCEPT; STALE; DUPLICATE | live v2 hA | ends at v2, but v1 content is served between the second and third event |
| Same order, different tail | upsert v2 hA; upsert v2 hA; upsert v1 hB | ACCEPT; DUPLICATE; STALE | live v2 hA | ends at **v1**: an older version wins permanently |
| Duplicate create | create v1 hA; create v1 hA; create v1 hB | ACCEPT; DUPLICATE; CONFLICT | live v1 hA | second create overwrites or errors; a changed body under the same version is silently accepted |
| Delete then delayed upsert | upsert v3; delete v4; *(retry of)* upsert v3 | ACCEPT; ACCEPT (tombstone v4); STALE | tombstone v4 | the deleted document is live again |
| Same, tombstone collected before the retry arrives | upsert v3; delete v4; *GC*; upsert v3 | ACCEPT; ACCEPT; —; ACCEPT (no record) | **live v3: resurrected** | same failure |

The last row is the reason for condition 3. The fix is a tombstone retention period longer than the maximum event delay plus replay window, and a convergence check across all representations before garbage collection.

**Progress watermark versus pinned read snapshot.** These are different objects (**D**, CLM-021):

- A **progress watermark** $H_r$ for representation $r$ is the largest source sequence number $s$ such that every event $\le s$ has been decided and is visible in $r$. It is a monotone progress report. By itself it guarantees nothing about what one request reads.
- A **pinned read snapshot** $S$ is chosen once per request, and *every* read for that request is served at $S$: lexical, vector, metadata, citation-target fetch, and cache. Enforcement needs one of: a store-level point-in-time or generation/alias read; or an application filter that drops any hit with version newer than $S$ and refuses (or waits for) representations with $H_r<S$. If a representation cannot serve $S$, the response is marked partial or the request abstains.

Notation: lowercase $s$ is any source sequence number; uppercase $S$ is the one sequence number a request is pinned to. Small example of the watermark definition: if events 100, 101, and 103 are decided and visible in $r$ but 102 is not, then $H_r=101$, not 103, because "every event $\le s$" fails at $s=102$. A request pinned at $S$ can be served by $r$ only when $H_r\ge S$.

Deletions obligated by policy or law are a separate hard constraint. A snapshot older than a delete must still exclude the deleted item, typically through a tombstone overlay checked at read time.

**Mixed-version read fixture** (synthetic). Event 101 updates `A` from v1 to v2; event 102 deletes `B`. At request time $H_{lex}=102$, $H_{vec}=100$, and a cached candidate list was built at 100.

| Read policy | What the request sees | Decision |
|---|---|---|
| No pinning | lexical returns `A` v2; vector and cache return `A` v1 chunks and `B` | **Mixed-version answer.** Context contains both versions of `A` and a deleted document. Not acceptable. |
| Pin $S=\min_r H_r=100$, no overlay (needs lexical to still serve generation 100) | `A` v1 and `B` everywhere | consistent but stale, *and* serves a deleted document. Fails the delete constraint. |
| Pin $S=100$ plus tombstone overlay through 102 (same requirement) | `A` v1, `B` excluded | consistent; answer marked stale by the lag of event 101. Acceptable if the freshness SLO allows it. |
| Pin $S=102$ | lexical ready; vector must catch up | wait up to the deadline for $H_{vec}\ge102$; otherwise answer from lexical only, marked partial, or abstain. |
| Stores keep only the latest version (no generation 100 on lexical): per-document version agreement | lexical has `A` v2, vector has `A` v1; `B` excluded by overlay | drop `A` from this request's candidates or wait for vector; never combine the two. The answer is marked partial if `A` was needed. |

A read-time filter can only *drop* hits newer than $S$. It cannot bring back an older version that the store has already overwritten. Pinning to an older snapshot therefore needs a store that retains that generation; without one, the last row is the only consistent option.

**Cache keys.** Including $S$ (or the index generation), the transformation/prompt versions, and the access scope in the cache key is one strategy. It stops a response built at one snapshot from being served to a request pinned at another. It does not remove the need for delete overlays and for validating the version of each cached citation target, because a key built at an old $S$ can still be requested legitimately. Timeout and retry handling must expose whether a response is stale, partial, failed, or a safe abstention. A fallback that silently serves an older snapshot changes the consistency contract.

**Worked Example — freshness ledger with two replicas and a cache** (synthetic values, registry CLM-022; times in seconds after the source event at $t=0$):

| Stage | Time | Note |
|---|---:|---|
| source event (document `P` updated v7→v8) | 0.0 | declared endpoint $t_{event}$ |
| change detected, parsed, chunked, embedded | 9.0 | |
| write acknowledged by primary | 11.0 | $t_w$ |
| refresh makes v8 searchable on replica R1 | 12.5 | |
| replica R2 applies and refreshes | 19.0 | R2 is lagging |
| result cache entry for the popular query, built at | 5.0 | TTL 60 s, holds v7 |
| delete of document `Q` (same event batch): lexical tombstone applied | 12.0 | |
| delete of `Q`: vector delete fails, retried and applied | 40.0 | branch failure, not lag |

Steps:

1. Per-path visibility: $L_{visible,R1}=12.5$ s and $L_{visible,R2}=19.0$ s. The post-write portion $t_v-t_w$ is 1.5 s on R1 and 8.0 s on R2. Reporting only $t_v-t_w$ would hide the 11 s before the write. (Here $t_v$ is short for $t_{query\ visible}$ on that replica. Check with the split above: R1 $=11.0+1.5=12.5$ s; R2 $=11.0+8.0=19.0$ s.)
2. Probes run once per second from $t=0$ to $t=30$ (31 probes), alternating R1 (even seconds) and R2 (odd seconds), bypassing the cache. Stale probes: 7 on R1 ($t=0,2,\dots,12$) and 9 on R2 ($t=1,3,\dots,17$), so $16/31\approx0.516$ of probes in the window are stale. Counting: a probe is stale when it runs before its replica's visibility time. R1 has 16 even-second probes and the ones with $t<12.5$ are $0,2,\dots,12$, that is $12/2+1=7$. R2 has 15 odd-second probes and the ones with $t<19.0$ are $1,3,\dots,17$, that is $(17-1)/2+1=9$. Stale fraction $=(7+9)/(16+15)=16/31$. The fraction depends on the window length chosen, so report the window with it.
3. The cache is independent of both replicas. With no invalidation, the cached v7 answer is served until $t=65$. With invalidation at $t=12.0$ and a cache miss refilled from R2 at $t=12.2$, the refill stores v7 again and serves it until $t=72.2$. Invalidation without a version check made the cache *later* than either replica. A cache key or entry check that requires version ≥ v8 prevents that refill.
4. For `Q`, the lexical path stops returning it at 12 s, but the vector path still returns it until 40 s. That is an incomplete delete, not refresh lag.

Interpretation: an "index refresh lag" hypothesis predicts that direct replica probes and cached answers become fresh at the same time. At $t=30$ both direct replica probes return v8 while the cached path returns v7, so index lag is **refuted** as the cause of staleness after 19 s. The remaining causes are the cache and, for `Q`, the failed vector delete. Limits: the numbers are synthetic, the probe schedule is deterministic, and a real measurement needs clock-skew bounds and a distribution over many events, not one document.

**Knowledge Check:**
1. Why does worker success not establish cache or replica visibility?
2. In the trace `upsert v2; delete v3; upsert v2 (retry)`, what is the final state, and what extra condition makes it safe?
3. Why does a progress watermark not stop a request from reading two versions of one document?

**Guided Practice:** Inject duplicate, reordered, delayed, failed, and timed-out events across text/vector/cache representations; probe high-watermarks and deletion convergence. Then decide these two traces with the acceptance table: (a) `create v1 hA; upsert v3 hB; upsert v2 hC; delete v3`; (b) `upsert v5; delete v6; GC after 30 s; upsert v5 retry arriving at 45 s`.

**Feedback Contract:**
- *Expected Output*: (a) ACCEPT, ACCEPT, STALE, CONFLICT: the delete reuses version 3 with a different op, so it is rejected and alerted, and the item stays live at v3 hB. The source must issue the delete as v4. (b) The retry meets "no record" and is accepted, resurrecting the document; the design fails unless tombstone retention exceeds 45 s. Your ledger should also report visibility as a distribution per representation, the stale/partial rate, and a rollback proof. Knowledge Check 2: the final state is tombstone v3, because the retry is STALE; it is safe only while the tombstone is retained. Knowledge Check 3: a watermark says how far each representation has progressed, and two representations can have different watermarks at the moment one request reads both.
- *Common Failure*: averaging only successful updates; or treating stable IDs plus versions as idempotent without an atomic conditional write.
- *Diagnostic Hint*: for each event ask "who compares the version, and can two workers both pass that comparison?" For each stale read ask "was the replica behind, or did the cache refill from a replica that was behind?"
- *Concept to Revisit*: conditional acceptance rule; progress watermark versus pinned read snapshot; Module 14 Lesson 14.2 (atomic idempotency).

**Learning Outcome:** Prove update/delete visibility and rollback end to end under reordered and duplicated events.

*(Effort: 65m instruction, 35m practice)*

### Lesson 10.3 — Query Orchestration and Context Assembly

**Engineering Question:**
Which query transformation and context packing choices improve evidence opportunity without changing intent?

**Concepts & Definitions:**

Preserving the original query is a valid route. Rewriting, expansion, decomposition, and multi-step retrieval are interventions that can improve evidence opportunity or silently alter identifiers, negation, scope, entity, and time. Log every transformation and compare it with the original using paired slices (**H**, CLM-009).

Context assembly is constrained selection, not concatenation. Candidate evidence differs in relevance, authority, freshness, length, redundancy, conflict, and access eligibility. Deduplicate by stable identity/version, retain source boundaries, budget tokens, and vary order deliberately (**D**, CLM-010).

Fusion-in-Decoder shows that a model can aggregate multiple retrieved passages in a scoped QA setup (**O**, CLM-002). Lost in the Middle (**O**, CLM-003) and FreshLLMs (**O**, CLM-004) show why “more context” is not a universal rule: evidence position, amount, and order can matter. Test depth/order matrices, relevant-plus-distractor mixtures, duplicate amplification, and truncation for the target model and prompt.

**Worked Example — packing five candidates into a 1,000-token evidence budget** (synthetic candidates, registry CLM-022).

| Rank | Chunk | Document / version | Tokens | Note |
|---:|---|---|---:|---|
| 1 | k1 | D1 v3 | 450 | relevant |
| 2 | k2 | D1 v3 | 400 | overlaps k1 almost entirely |
| 3 | k3 | D2 v1 | 300 | superseded by D2 v2 |
| 4 | k4 | D3 v1 | 350 | the only chunk with the qualifier the answer needs |
| 5 | k5 | D2 v2 | 250 | current version of D2 |

Steps:

Both policies below walk the list in rank order and add a chunk only if the running total stays within the budget $B=1{,}000$ tokens, i.e. chunk $k$ is added when $\sum_{j\in\text{context}} tokens_j + tokens_k \le B$. They differ only in which chunks are eligible before that walk starts.

1. *Concatenate by rank:* k1 (450) + k2 (400) = 850 tokens. k3 does not fit (1,150). The context holds one document twice and does not contain k4.
2. *Constrained selection:* drop k2 as a duplicate of k1 by document/version and overlap; drop k3 because a newer version is eligible. Remaining in rank order: k1 (450), k4 (350), k5 (250). k1 + k4 = 800 fits; adding k5 gives 1,050, which does not. Context = {k1, k4}, 800 tokens, 200 unused.
3. *Result:* evidence opportunity for the needed qualifier is 0 in the first context and 1 in the second, with 50 fewer tokens ($850-800=50$).

Interpretation and limits: the gain here comes from selection, with retrieval unchanged. Whether the model then *uses* k4 is a separate measurement. To test placement, hold the context {k1, k4} fixed and move k4 from first to last, then add distractors; a quality change under that design isolates placement sensitivity more directly than changing retrieval and order together. The table does not predict which order is better for a given model.

**Knowledge Check:** Why can rewriting improve recall while corrupting an identifier or negation?

**Guided Practice:** Cross original/rewrite/decompose with depth, order, duplicates, distractors, and token budget; preserve candidate/context lineage.

**Feedback Contract:**
- *Expected Output*: a paired table, one row per (query route × depth × order) cell at a matched token budget, with intent fidelity, evidence opportunity, use/support, latency, and cost, plus the identifier/negation slice reported separately. At least one cell should show a transformation that raises opportunity while lowering intent fidelity, or you should state that none was found and how many cases were tried.
- *Common Failure*: changing the rewrite, the depth, and the order in one run. The quality change then cannot be assigned to any of them.
- *Diagnostic Hint*: for a failing case, was the supporting passage in the candidates before the rewrite? Was it in the assembled context? If it was in both, the loss is in order or use, not in retrieval.
- *Concept to Revisit*: context assembly as constrained selection; Lesson 10.1 opportunity versus use.

**Learning Outcome:** Demonstrate which query and packing policy improves answer/citation quality at matched latency and cost, including slices where it fails.

*(Effort: 45m instruction, 30m practice)*

### Lesson 10.4 — Contradiction Is a Data-and-Policy Problem

**Engineering Question:**
Which evidence applies when sources disagree across authority, scope, and time?

**Concepts & Definitions:**

Contradiction can occur among source documents, revisions of one source, retrieved context and model memory, or the answer and its cited passage. Normalize the contested claim before choosing a winner:

```text
(subject, predicate, object, valid_time, transaction/version,
 source_identity, authority, scope)
```

Two passages may be temporally different rather than contradictory. Ten mirrors of a stale article do not outrank one authoritative update. “Newest” is unsafe when publication time differs from effective time. Model confidence does not establish source authority (**D**, CLM-012).

Real-document experiments report that models can retain incorrect parametric answers despite corrective context (**O**, CLM-011). Therefore test both directions: trusted context correcting model memory, and untrusted context attempting to override a valid prior. If authority/time/scope cannot resolve the evidence, surface the conflict or abstain.

**Worked Example — authority and valid-time conflict table** (synthetic sources).

Question: "What is the daily meal reimbursement cap for an employee in region EU?" asked with query time 2026-03-15. All five sources are retrieved and all pass the access check.

| ID | Source (authority tier) | Scope | Published | Valid from | Supersedes | Value |
|---|---|---|---|---|---|---|
| S1 | Travel policy v3 (A: policy owner) | global | 2026-01-10 | 2026-01-01 | v2 | 100 EUR |
| S2 | Travel policy v4 (A) | global | 2026-02-20 | 2026-04-01 | v3 | 120 EUR |
| S3 | EU addendum (A) | EU | 2025-11-01 | 2025-12-01 | — | 90 EUR |
| S4 | Intranet wiki article, 10 mirrored copies (C: unreviewed) | unstated | 2026-03-01 | unstated | — | 150 EUR |
| S5 | HR FAQ (B: reviewed summary) | EU | 2026-01-15 | unstated | — | 100 EUR |

Declared policy, applied in order. This is a product decision made for the exercise, not a universal rule:

1. Collapse copies that share a source identity or content hash. S4 counts once.
2. Keep sources whose scope covers the question and whose validity interval contains the query time. A source with no stated validity is kept but cannot outrank one that states it.
3. Prefer the highest authority tier present.
4. Within that tier, a more specific scope overrides a general one, unless a later-valid source lists it under *Supersedes*.
5. Within one source lineage, the latest valid version wins.
6. If two or more distinct values remain, the state is UNRESOLVED: show both with their sources, or abstain.

Steps for query time 2026-03-15:

- Rule 1: S4 becomes one tier-C source.
- Rule 2: S2 is dropped, because 2026-04-01 is after the query time. S1, S3, S4, and S5 remain.
- Rule 3: tier A leaves S1 and S3.
- Rule 4: S3 has EU scope, and S1 does not list the addendum as superseded, so S3 overrides S1.
- Result: **90 EUR, cited to S3.** S5 (100 EUR) is reported as a lower-tier source that disagrees and may be out of date.

For comparison, the two shortcuts give a different answer on the same table:

- "Newest publication wins" picks S4 (2026-03-01): 150 EUR.
- "Majority by document count" counts ten copies of S4: 150 EUR.

Now move the query time to 2026-04-15. S2 is valid and replaces S1 by rule 5. S2 is global and S3 is EU, and S2's *Supersedes* field lists only v3. Rule 4 as written keeps S3, so this policy returns 90 EUR. Whether a new global policy was *meant* to replace a regional addendum cannot be read from these fields. A stricter rule 4 is also defensible: when a general source becomes valid *after* the specific one, require an explicit record saying the specific one is still in force. Under that variant the output is UNRESOLVED, with both 120 EUR (S2) and 90 EUR (S3) shown and a request for a supersession record. The two variants disagree on this input, so the choice between them must be declared before the system is evaluated.

Interpretation and limits: S1 and S2 are temporal versions, not a contradiction. S1 and S3 are a scope difference, not a contradiction. Only the S2/S3 case at the later date can be a real unresolved conflict, and it is a missing-metadata problem, not a model problem. A different organization could reasonably order rules 3 and 4 the other way; the table and the trace are what make the decision auditable.

**Knowledge Check:**
1. When are two different values temporal versions rather than contradictions?
2. In the table above, which single metadata field would resolve the 2026-04-15 case, and who is allowed to set it?

**Guided Practice:** Resolve a set containing stale mirrors, future-effective updates, scope-specific rules, and equal-authority conflict; preserve an unresolved state. Start from the table above with one change: S5 is reclassified to tier A with valid-from 2026-02-01.

**Feedback Contract:**
- *Expected Output*: a normalized claim table and a rule-by-rule trace. For the modified table at query time 2026-03-15: rule 3 keeps S1, S3, and S5; rule 4 removes S1; S3 (90 EUR) and S5 (100 EUR) are two tier-A EU sources from different lineages with different values, so the result is UNRESOLVED with both shown.
- *Common Failure*: majority vote by document count, or "newest publication wins". A second one is choosing S5 because its valid-from date is later, which applies rule 5 across two different source lineages.
- *Diagnostic Hint*: write each source as (subject, predicate, value, scope, valid interval, lineage, tier). Which of those fields actually differ between the two candidates you are comparing?
- *Concept to Revisit*: valid time versus publication time; normalized claim tuple; deliberate unresolved state.

**Learning Outcome:** Implement an auditable authority/temporal policy and a deliberate unresolved-conflict state.

*(Effort: 50m instruction, 25m practice)*

### Lesson 10.5 — Citations, Grounding, and Stage Attribution

**Engineering Question:**
Does each support-required claim bind to adequate evidence, and where was opportunity lost?

**Concepts & Definitions:**

A citation has at least two independent properties. This module uses the following definitions (**D**, CLM-020); they are course definitions, and a published metric with a similar name may differ:

$$
CitationCorrectness=\frac{supported\ emitted\ citations}{checked\ emitted\ citations}
$$

$$
SupportCompleteness=\frac{support\text{-}required\ claims\ whose\ cited\ evidence\ entails\ them}
{support\text{-}required\ claims}.
$$

A third quantity is purely syntactic and must not be confused with either:

$$
CitationCoverage=\frac{support\text{-}required\ claims\ with\ at\ least\ one\ citation\ marker}
{support\text{-}required\ claims}.
$$

Short notation for one answer:

- $e$: emitted citations that were checked; $s$: those judged supported (rule below). $CitationCorrectness=s/e$.
- $n$: support-required claims; $k$: claims whose cited evidence entails them; $m$: claims with at least one citation marker. $SupportCompleteness=k/n$ and $CitationCoverage=m/n$.

A claim can only have entailing cited evidence if it has a citation, so $k\le m\le n$, and therefore completeness $\le$ coverage always. A gap between them counts the claims that carry a marker but are not supported. Correctness has a different denominator ($e$, not $n$), so it is not bounded by either.

The units differ. Correctness counts *citations*. Completeness and coverage count *claims*. A claim is "support-required" when the answer policy says it needs evidence; greetings, hedges, and restatements of the question are excluded by a declared rule.

**Multi-citation rule.** When one claim carries several citations:

- The claim counts toward completeness if the *concatenation* of its cited passages entails it. Two passages may each be insufficient and jointly sufficient.
- A citation counts as supported if it entails the claim alone, or if removing it breaks the joint support. A citation that does not support the claim alone, and whose removal leaves the claim still supported, is unsupported (irrelevant).

ALCE uses the same structure: its citation recall asks whether the concatenation of a statement's cited passages entails the statement, and its citation precision flags a citation as irrelevant under the removal test above (**O**, CLM-013; §3.3 and Figure 3 of the paper). ALCE scores precision as 0 for every citation of a statement whose recall is 0. That detail belongs to ALCE; state your own rule for it.

**Zero-denominator rule.** If an answer emits no citations, correctness is *undefined* for that answer: report it as "n/a, 0 citations", not as 1 and not as 0. If an answer has no support-required claims, completeness and coverage are undefined. When aggregating, say whether you pool counts over answers (micro) or average per-answer ratios (macro), exclude undefined answers from a macro average, and report how many were excluded. An abstention is scored by the abstention policy, not by these ratios.

For answers $a=1,\dots,A$ with counts $s_a$ and $e_a$ (same pattern for $k_a/n_a$ and $m_a/n_a$):

$$
Correctness_{micro}=\frac{\sum_a s_a}{\sum_a e_a},\qquad
Correctness_{macro}=\frac{1}{|A'|}\sum_{a\in A'}\frac{s_a}{e_a},
$$

where $A'$ is the set of answers with $e_a>0$. Micro weights each citation equally, so answers with many citations dominate; macro weights each answer equally.

Declare claim segmentation, support threshold, citation granularity, and the rules above before scoring. A syntactically valid `[3]` can point to a real but irrelevant document. ALCE separates answer correctness and citation quality; source inspection of Haystack at the pinned revision likewise shows numeric reference parsing/binding, not semantic entailment verification.

Diagnose an incorrect claim in order:

```text
source absent?
  -> version invisible?
  -> not retrieved?
  -> retrieved but not selected?
  -> selected but ignored/misread?
  -> correct claim but wrong/incomplete citation?
```

Each stage admits a different intervention, so attribute the failure to the earliest failing stage rather than to one answer score (**D**, CLM-014). Record evaluator identity and uncertainty. Automated NLI or LLM judges are measurements with errors, not ground truth.

**Worked Example — one claim-citation table, three metrics** (synthetic answer; entailment labels are assumed to come from an adjudicated human check).

Answer 1 has four support-required claims and emits at most one citation per claim:

| Claim | Emitted citation | Does the cited span entail the claim? | Citation supported? | Claim supported? |
|---|---|---|---|---|
| C1 | [d1] | yes | yes | yes |
| C2 | [d2] | yes | yes | yes |
| C3 | [d3] | no: d3 is a real document on another topic | no | no |
| C4 | none | — | — | no |

Steps, all from this one table:

- Emitted citations checked: 3. Supported: 2. $CitationCorrectness=2/3\approx0.667$.
- Support-required claims: 4. Claims whose cited evidence entails them: 2 (C1, C2). $SupportCompleteness=2/4=0.5$.
- Claims with a citation marker: 3. $CitationCoverage=3/4=0.75$.

Interpretation: 3/4 is the number a marker-counting script reports. It is coverage, and it overstates support, because C3 has a marker but no support. Reporting "completeness 3/4" for this answer would count C3 as supported while the correctness figure counts the same citation as unsupported. That contradiction is the error to avoid: both ratios must come from the same rows.

Answer 2 shows the multi-citation rule. It has two claims:

| Claim | Citations | Entailment | Citations supported | Claim supported? |
|---|---|---|---|---|
| C5 | [d4][d5] | neither alone; d4+d5 together entail C5 | d4 yes, d5 yes (removing either breaks support) | yes |
| C6 | [d6][d7] | d6 alone entails C6; d7 is unrelated | d6 yes, d7 no (removing d7 changes nothing) | yes |

Answer 2: correctness $3/4$, completeness $2/2$, coverage $2/2$. (Counting: $e=4$ citations, of which d4, d5, d6 are supported, so $s=3$; $n=2$ claims, both supported and both marked, so $k=m=2$.)

Aggregating the two answers:

| | Micro (pooled counts) | Macro (mean of per-answer ratios) |
|---|---|---|
| Correctness | $(2+3)/(3+4)=5/7\approx0.714$ | $(2/3+3/4)/2=17/24\approx0.708$ |
| Completeness | $(2+2)/(4+2)=4/6\approx0.667$ | $(1/2+1)/2=0.75$ |

Step-by-step for correctness: micro pools numerators and denominators, $s_1+s_2=2+3=5$ over $e_1+e_2=3+4=7$. Macro averages the two ratios, $(2/3+3/4)/2=(8/12+9/12)/2=17/24$. Micro is slightly higher because Answer 2, with the better ratio, also has more citations and so more weight.

Micro and macro differ, so the report must say which one it uses. A third answer with zero citations would add nothing to the micro correctness denominator and would be excluded from the macro correctness average, with the exclusion count reported. Limits: the entailment labels are given here. In practice they come from an NLI model or an LLM judge with its own error rate, and that error must be estimated against human adjudication before the ratios are trusted.

**Knowledge Check:**
1. Why does a valid reference index not establish semantic support?
2. An answer has 5 support-required claims, 5 citation markers, and 5 supported citations, all on the same 3 claims. What are correctness, completeness, and coverage?

**Guided Practice:** Label claims and evidence spans, then replay with oracle availability, candidates, context, and citation binding to isolate the earliest failing stage.

**Feedback Contract:**
- *Expected Output*: a claim-evidence table with one row per (claim, citation) pair, evaluator/human agreement, a stage trace, and uncertainty. For Knowledge Check 2: correctness $5/5=1$, completeness $3/5$, coverage $3/5$ ($e=s=5$; $n=5$; the markers sit on 3 claims, so $m=3$, and those 3 are supported, so $k=3$). High correctness with low completeness means the citations that exist are good and two claims have none.
- *Common Failure*: one aggregate "groundedness" score; or computing completeness from marker counts, which gives coverage.
- *Diagnostic Hint*: point to the row in your table that each numerator counts. If correctness and completeness cannot be traced to the same rows, they were computed from different data.
- *Concept to Revisit*: citation versus claim as the counted unit; multi-citation rule; zero-denominator rule.

**Learning Outcome:** Produce a claim-evidence table and assign failures to the earliest discriminated stage.

*(Effort: 55m instruction, 30m practice)*

### Lesson 10.6 — RAG, Long Context, and Adaptive Routes

**Engineering Question:**
Which route maximizes declared utility while preserving hard evidence and access constraints?

**Concepts & Definitions:**

Compare architectures with the same corpus snapshot, eligible evidence, base model where possible, prompt objective, answer set, and load. Include retrieval/index work, input tokens, cached prefixes, latency tails, failures, and evaluator costs. A large advertised context window does not prove robust evidence use; retrieval may miss evidence that full context contains; full context may dilute or misorder evidence that retrieval isolates.

A useful engineering decision is constrained rather than ideological:

$$
U(route)=Q(route)-\lambda_L L(route)-\lambda_C C(route),
$$

subject to hard access, safety, freshness, and consistency constraints.

where $Q$ is a unitless quality score, $L$ is a latency (seconds), $C$ is a cost (USD), and $\lambda_L$ (per second) and $\lambda_C$ (per USD) convert latency and cost into quality units, so every term of $U$ is unitless. $\lambda_L$ answers "how much quality would we give up to save one second?" Note: in this lesson $L$ means latency, not the visibility lag $L_{visible}$ of Lesson 10.2, and $Q$ means quality, not document `Q` of the Lesson 10.2 ledger.

This is a product-specific heuristic: quality definition and weights are not universal. Published comparisons through 2025 report different task-dependent outcomes, which is evidence against a universal winner (**O**, CLM-015).

Self-RAG, corrective RAG, and hybrid routing demonstrate adaptive mechanism families. They add evaluator/router errors, correlated self-judgment, extra model or search calls, new source-quality risks, latency, and cost. Treat them as workload-dependent/frontier until calibrated on the target service (**O**, CLM-016).

**Worked Example — route decision with hard constraints and declared weights** (all values synthetic, invented for this exercise; they are not measurements of any model or product).

Setup: four routes evaluated on the same 400 questions, the same source snapshot, the same generator, and the same load.

- $Q$: fraction of answers that are correct *and* have complete support (unitless, 0–1).
- $L$: p95 end-to-end latency in seconds, including retrieval, router calls, and queueing.
- $C$: mean cost per query in USD, including index, router, and generation work.
- Hard constraints (pass/fail, not traded off): access violations $=0$; stale-answer rate $\le1\%$; $L\le4.0$ s; $Q\ge0.80$.
- Utility among feasible routes: $U=Q-\lambda_L L-\lambda_C C$ with $\lambda_L=0.02\ \text{s}^{-1}$ and $\lambda_C=5\ \text{USD}^{-1}$. In words: one second of p95 latency is worth 0.02 of $Q$, and 0.01 USD per query is worth 0.05 of $Q$. These weights are a product choice.

| Route | $Q$ | $L$ (s) | $C$ (USD) | Stale rate | Access violations | Feasible? | $U$ |
|---|---:|---:|---:|---:|---:|---|---:|
| RAG | 0.82 | 2.1 | 0.004 | 0.4% | 0 | yes | $0.82-0.042-0.020=0.758$ |
| Long context (LC) | 0.88 | 6.5 | 0.030 | 0.4% | 0 | no: $L>4.0$ | (0.600, not compared) |
| Hybrid router | 0.86 | 3.6 | 0.012 | 0.4% | 0 | yes | $0.86-0.072-0.060=0.728$ |
| LC with cached prefix | 0.87 | 3.2 | 0.010 | 2.5% | 0 | no: stale $>1\%$ | (0.756, not compared) |

Steps:

How a $U$ cell is computed (RAG row): $\lambda_L L=0.02\times2.1=0.042$; $\lambda_C C=5\times0.004=0.020$; $U=0.82-0.042-0.020=0.758$. The other rows follow the same pattern.

1. Apply hard constraints first. LC fails latency. Cached LC fails freshness, even though its utility would be 0.756. A route that fails a hard constraint is rejected whatever its utility.
2. Compare utilities of the feasible routes: RAG 0.758, Hybrid 0.728. **RAG is selected**, although Hybrid has the higher quality.
3. Sensitivity: Hybrid ties RAG when $0.778-0.004\lambda_C=0.788-0.012\lambda_C$, that is at $\lambda_C=1.25\ \text{USD}^{-1}$. Derivation: hold $\lambda_L=0.02$ fixed and leave $\lambda_C$ free. RAG: $Q-\lambda_L L=0.82-0.042=0.778$, so $U_{RAG}=0.778-0.004\lambda_C$. Hybrid: $0.86-0.072=0.788$, so $U_{Hyb}=0.788-0.012\lambda_C$. Setting them equal gives $(0.012-0.004)\lambda_C=0.788-0.778$, so $\lambda_C=0.010/0.008=1.25$. If the product values cost at less than 1.25 per USD, Hybrid wins. The decision depends on a weight, so the weight must be recorded with the decision.
4. Uncertainty: with 400 questions, the standard error of a proportion near 0.82 is about $\sqrt{0.82\times0.18/400}\approx0.019$. The 0.04 gap between RAG and Hybrid needs a paired interval (Module 15) before it is treated as real. (Formula: for a proportion $\hat p$ measured on $n$ questions, $SE=\sqrt{\hat p(1-\hat p)/n}$; here $\hat p=0.82$, $n=400$. For scale only: if the two routes were scored on independent question sets, the SE of the difference would be about $\sqrt{0.0192^2+0.0173^2}\approx0.026$, so the gap is roughly 1.5 SE. Because both routes answer the same 400 questions, the paired interval uses the per-question differences instead, which is why it is the right test.)

Interpretation and limits: the example shows the order of operations (constraints, then utility, then sensitivity). It does not show that RAG beats hybrid routing in general. Retrieval misses weaken RAG; evidence dilution or position sensitivity can weaken full context; a router adds its own errors. Per-slice results (no-answer, temporal, access-restricted, long noisy documents) can reverse the aggregate ranking, so the selected route still needs a fallback.

**Knowledge Check:**
1. Why is advertised context length insufficient evidence for long-context quality?
2. In the table, why is "LC with cached prefix" rejected although its utility is higher than Hybrid's?

**Guided Practice:** Run matched RAG, full-context, and hybrid routes under equal eligibility and load; test no-answer, noisy-long-document, temporal, and access slices. Then recompute the table decision with the latency constraint relaxed to $L\le7.0$ s.

**Feedback Contract:**
- *Expected Output*: a matched frontier, route calibration/errors, full cost and latency, hard-constraint checks, fallback, and rollback. For the relaxed constraint: LC becomes feasible with $U=0.88-0.130-0.150=0.600$ ($0.02\times6.5=0.130$, $5\times0.030=0.150$), which is still the lowest feasible utility, so RAG remains selected. Feasible is not the same as preferred.
- *Common Failure*: comparing unequal corpora or models; or folding a hard constraint into the utility, so that a large quality gain "pays for" stale or unauthorized answers.
- *Diagnostic Hint*: list every route you rejected and the single reason. If the reason is a utility number, check that the route passed all hard constraints first.
- *Concept to Revisit*: hard constraints versus weighted utility; matched comparison conditions.

**Learning Outcome:** Build a measured route policy with fallback and a falsifiable decision log.

*(Effort: 50m instruction, 30m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al. (NeurIPS 2020).
- [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) — Izacard and Grave (2021).
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Liu et al. (TACL 2023).
- [Enabling Large Language Models to Generate Text with Citations](https://arxiv.org/abs/2305.14627) — Gao et al. (EMNLP 2023).
- [FreshLLMs](https://arxiv.org/abs/2310.03214) — Vu et al. (2023).

*Scope:* each paper supports only its own models, corpora, retrievers, prompts, and metrics. For ALCE, §3.3 and Figure 3 (arXiv v2) were read in full on 2026-09-30 for the citation recall/precision definitions used in Lesson 10.5. The other four entries were not re-read in this revision and keep their earlier access dates.

**RECOMMENDED BASELINE (course position, not a surveyed industry default):** stable document/version identity with an atomic acceptance rule, query-visible freshness probes, versioned tombstones, request/context/citation lineage, explicit conflict policy, claim-level support checks, staged diagnosis, and matched end-to-end evaluation. These follow from the derivations in Lessons 10.1–10.5 (**D**, CLM-005, CLM-006, CLM-007, CLM-018). This module has no evidence about how widely they are deployed; registry entries that carry the label "CURRENT DEFAULT" mean this recommended baseline and nothing more.

**ONE RUNTIME'S DOCUMENTED BEHAVIOR:** Elasticsearch — [refresh parameter](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/refresh-parameter), [optimistic concurrency control](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/optimistic-concurrency-control), and the [index API](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-index) (external versioning, `if_seq_no`/`if_primary_term`, `op_type=create`). The last two were opened on 2026-09-30. They describe one store and are not a default for vector databases in general.

**WORKLOAD-DEPENDENT:** query rewrite/decomposition, context depth and order, compression, authority rules, citation granularity, long-context use, retrieval frequency, and route thresholds. Evidence that the RAG-versus-long-context outcome depends on task and setup: [Li et al. 2024](https://arxiv.org/abs/2407.16833) (EMNLP 2024 industry track) and [Li et al., arXiv 2501.01880](https://arxiv.org/abs/2501.01880) (submitted 2024-12-27). Both were checked at abstract level on 2026-09-30; their methods and result tables were not re-audited.

**FRONTIER:** learned or self-reflective retrieval decisions ([Self-RAG](https://arxiv.org/abs/2310.11511)), corrective retrieval with an evidence evaluator ([CRAG](https://arxiv.org/abs/2401.15884)), and self-routing between RAG and long context (Self-Route in Li et al. 2024). A published mechanism is not an industry default. `TODO_VERIFY`: no 2025–2026 primary source on task-specific context orchestration was opened for this revision, so the module makes no claim about the state of that work in 2026.

**LEGACY / INSUFFICIENT:** one-shot static top-k stuffing; “indexed” equated with query-visible; newest/frequent source automatically trusted; citation marker treated as support; answer score used as the only RAG metric; RAG or long context declared universally superior.

**PRODUCTION SOURCE TRACE**

- Repository: `deepset-ai/haystack`
- Revision: `8a5406eea71a0fc19e94c4b9a5cd96df2158a45a`
- Verified: 2026-09-26, static inspection only. On 2026-09-30, `answer_builder.py` and `document_writer.py` were re-read at the same revision (reference-pattern parsing and the `referenced` metadata flag in `AnswerBuilder.run`; `DuplicatePolicy` passed to the document store in `DocumentWriter.run`). `pipeline.py` and `prompt_builder.py` were not re-read, so the registry date for the whole trace is unchanged.
- Files/symbols: `Pipeline.run/_run_component`, `PromptBuilder.run`, `DocumentWriter.run`, and `AnswerBuilder.run` in the registry-recorded paths.
- Execution: pipeline dispatch invokes component `run`; prompt variables are rendered; document writes delegate to the configured store; answer building parses reference indices and attaches document copies. No semantic citation-entailment validation was observed in that path (**O**, CLM-017).
- Relation to Lesson 10.2: `DocumentWriter` exposes an ID-based `DuplicatePolicy` (`NONE`, `SKIP`, `OVERWRITE`, `FAIL`) and delegates to `document_store.write_documents`. No version comparison appears in this component. `OVERWRITE` is therefore last-writer-wins by ID at this layer, and `SKIP` keeps the first writer; the version acceptance rule has to come from the document store or from code the learner adds. Whether a given store backend adds version checks was not inspected.
- Scope: pinned Haystack behavior, not a general definition of RAG orchestration or consistency.
- **Trace practice (the 2h `source_trace` effort):** at the pinned revision, follow `DocumentWriter.run` to the `write_documents` call and `AnswerBuilder.run` to the point where the `referenced` flag is set. Answer in writing: (1) which component decides what happens on a duplicate ID, and does it see a version? (2) what input would make `AnswerBuilder` mark a document as referenced although it does not support the sentence? Expected answers: (1) the policy value is passed through and the store decides; this component compares no version; (2) any reply that contains an in-range reference number matching the pattern; the flag is set per reply from the pattern match alone, with no claim-to-document mapping and no entailment check. A common error is to conclude that the framework "handles deduplication" or "validates citations" from the names of the parameters.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Incremental Index Visibility and Delete Convergence

- **Objective:** build a versioned change ledger and prove query-visible update/delete convergence across representations, replicas, and caches.
- **Pre-Registered Hypothesis:** worker acknowledgement will precede full read visibility in at least one injected delay/failure case; an atomic version-acceptance rule with retained tombstones will converge under every injected order, while a check-then-write or last-writer-wins baseline will fail at least one trace.
- **Independent Variables:** event order/duplication/delay/failure, representation, refresh, replica/cache state, and rollback.
- **Dependent Variables:** visibility distributions, stale/partial reads, version monotonicity, access violations, idempotence, and recovery time.
- **Break & Falsify:** replay the Lesson 10.2 traces (`v2,v1,v2`; duplicate create; delete then delayed upsert), race two concurrent duplicates, collect a tombstone before a delayed upsert arrives, lag caches/replicas, refill a cache from a lagging replica, time out one branch, and verify fallback state is explicit. Any final state that differs from the acceptance table, or any answer that mixes two versions of one document, falsifies the design. Data is a synthetic fixture unless you state its source.
- **Alignment:** Lessons 10.1 and 10.2.
- **Effort Estimate:** 4.5h.

### LAB B — Query and Context Phase Diagram

- **Objective:** compare query transformations and context packing at matched evidence and resource budgets.
- **Pre-Registered Hypothesis:** transformation/packing gains will vary by slice, with identifier/negation and distractor/order counterexamples.
- **Independent Variables:** query route, depth, order, duplicates, distractors, conflicts, compression, and token budget.
- **Dependent Variables:** intent fidelity, evidence opportunity, use/support, answer/citation quality, latency, tokens, and cost.
- **Break & Falsify:** test identifiers, negation, temporal, multi-hop, no-answer, middle-position, and noisy-context cases.
- **Alignment:** Lesson 10.3.
- **Effort Estimate:** 4.5h.

### LAB C — Contradiction and Citation Audit

- **Objective:** implement conflict policy and claim-level citation correctness/completeness with independent adjudication.
- **Pre-Registered Hypothesis:** document-count/newest-only policies and citation syntax will fail at least one authority/time/support case.
- **Independent Variables:** authority, valid/publication/transaction time, scope, mirrors, context order, reference binding, and evaluator.
- **Dependent Variables:** resolved/unresolved accuracy, citation correctness/completeness, evaluator agreement, severity, and abstention.
- **Break & Falsify:** use stale mirrors, future-effective rules, equal-authority conflicts, deletes, partial/multi-source support, and broken numeric references.
- **Alignment:** Lessons 10.4 and 10.5.
- **Effort Estimate:** 4.5h.

### LAB D — Matched RAG/Long-Context/Hybrid Routing

- **Objective:** compare RAG, long-context, and hybrid routing on identical evidence eligibility and loaded conditions.
- **Pre-Registered Hypothesis:** no route will dominate every prespecified slice after full work, freshness, failure, and SLO constraints are counted.
- **Independent Variables:** route, evidence length/order/noise, query slice, temporal state, router threshold, load, and cache.
- **Dependent Variables:** answer/support quality, tokens, full work, stage/end-to-end tails, cost, freshness, abstention, route error, and SLO-goodput.
- **Break & Falsify:** include answerable-without-context, shuffled/no evidence, temporal updates, long noisy documents, router shift, and bursts.
- **Alignment:** Lesson 10.6.
- **Effort Estimate:** 4.5h.

## 07 Break / Incident Scenarios

### Incident 10.1 — Fresh Answer Metric, Stale and Unsupported Production Answers

**Incident Symptoms:**
After an incremental-index release, average answer accuracy improves, but some tenants receive deleted policies, recent updates are absent for minutes, contradictory versions are blended, and citations open valid documents that do not support the attached claims.

**Diagnostic Protocol (Task):**
1. **Form Competing Hypotheses:** source polling lag, parse/embed failure, unrefreshed write, replica/alias/cache skew, retry reordering, incomplete tombstone fanout, rewrite drift, retrieval loss, context dilution, parametric bias, generator misuse, reference binding, or evaluator error.
2. **Identify Missing Evidence:** event/version logs, representation state/high-watermarks, route/cache/replica identity, original/transformed queries, candidates, exact context/order, prompt/model, claims/citations, timeout/fallback state, and stage timing.
3. **Design Discriminating Measurements:** probe each representation and replica, replay pinned snapshots, disable transformations/caches one at a time, and substitute oracle candidates/context/bindings.
4. **Rank Explanations:** locate the earliest available → visible → retrieved → selected → used → supported divergence and preserve interacting causes.
5. **Intervene:** gate/rollback the smallest supported stage, converge deletes/replicas/caches, repair transformation/context/binding, or abstain on unresolved partial state.
6. **Remeasure:** repeat freshness/stale/partial distributions, access/deletion invariants, claim support/citation completeness, latency, abstention, and user utility.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem — Versioned Evidence Platform

Design Advanced RAG for mutable technical and policy documentation with tenant ACLs, frequent updates/deletes, contradictory versions, citations, and a latency/cost SLO.

**Workload and constraints (SYNTHETIC fixture — exercise inputs, not measurements).** A submission may replace any value with its own measurement if it says so; it is then graded against its own declared inputs.

| Item | Fixture value |
|---|---|
| Corpus | 120,000 documents, 3 tenants, per-document ACL |
| Change rate | 2,000 updates and 150 deletes per day, in bursts of up to 200 events per minute |
| Representations | lexical index, vector index, metadata store, result cache (TTL 300 s), 2 search replicas |
| Freshness SLO | p95 query-visible lag $\le60$ s on every route for updates |
| Delete constraint (hard) | a deleted document is returned by no route later than 300 s after the source event, and is never resurrected |
| Consistency (hard) | one answer never mixes two versions of the same document |
| Access (hard) | zero cross-tenant evidence in context or citations |
| Latency and cost | p95 end-to-end $\le4.0$ s; mean cost $\le0.015$ USD per query |
| Quality floor | correct-with-complete-support rate $\ge0.80$ on the answerable set; abstain on the no-answer set |
| Maximum event delay / replay window | 15 minutes |

**Source-change fixture.** The design must give the expected decision for each event and the final state per document. Events are listed in *arrival* order; versions are assigned by the source.

| # | Arrival (s) | Event |
|---|---:|---|
| 1 | 0 | upsert `doc-7` v4 hash `h4` |
| 2 | 3 | upsert `doc-7` v5 hash `h5` |
| 3 | 4 | upsert `doc-7` v4 hash `h4` (retry of 1) |
| 4 | 10 | delete `doc-9` v8 |
| 5 | 12 | create `doc-11` v1 hash `hA` |
| 6 | 12 | create `doc-11` v1 hash `hA` (duplicate delivery) |
| 7 | 400 | upsert `doc-9` v7 hash `h7` (delayed) |
| 8 | 410 | upsert `doc-11` v1 hash `hB` |
| 9 | 415 | upsert `doc-7` v6 hash `h6` |

Two further conditions apply to the same timeline:

- The vector-index application of event 4 fails at 20 s and succeeds on retry at 200 s. The lexical index applies it at 11 s.
- A query arrives at 420 s. At that moment the lexical index has applied events 1–9 and the vector index has applied events 1–8. Neither store retains older generations.

**Conflict fixture.** Use the five-source table of Lesson 10.4 at both query times.

**Required Deliverables:**
1. request/read contract with eligible snapshot, transformations, budgets, and abstention;
2. incremental event ledger with the acceptance rule, its atomicity mechanism, tombstone retention, watermarks, and the decision for each of events 1–9;
3. query-visible freshness and stale/partial/failure measurements per route, including the cache and both replicas, the delete-convergence check for `doc-9`, and the read decision for the query at 420 s;
4. query/context depth/order/noise ablations with complete lineage;
5. temporal/scope/authority contradiction policy, its trace on the conflict fixture, and the unresolved state;
6. claim-citation table with correctness, support completeness, and coverage computed from the same rows, the multi-citation and zero-denominator rules, and calibration against blinded adjudication;
7. matched RAG/long-context/hybrid comparison with hard constraints applied before any utility, declared weights, and a sensitivity statement;
8. source trace of a pinned RAG framework revision with generalizability boundary;
9. canary, fallback, rollback, and Incident 10.1 diagnosis separating source, visibility, retrieval, selection, use, and support.

## 09 Required Evidence & Rubric

### Required Artifact: End-to-End Evidence Lineage Trace

Submit one replayable source-event-to-answer trace, visibility/deletion tests, query/context checkpoints, claim-evidence table, evaluator audit, matched architecture results, source trace, and release/rollback decision.

### Reference Checks for Mastery Deliverables 2, 3, 5, and 6 (fixture inputs only)

Reviewers use these to check decisions and arithmetic. A submission with different declared inputs is checked against its own inputs.

- **Events 1–9** under the Lesson 10.2 acceptance rule: 1 ACCEPT (`doc-7` live v4); 2 ACCEPT (v5); 3 STALE; 4 ACCEPT as tombstone v8, whatever lower version was stored; 5 ACCEPT; 6 DUPLICATE; 7 STALE, because v7 < tombstone v8; 8 CONFLICT, same version with a different hash, so `doc-11` stays at `hA` and an alert is raised; 9 ACCEPT (`doc-7` live v6).
- **Tombstone retention:** event 7 arrives 390 s after the delete, and the declared replay window is 900 s. A design that collects tombstones at the 300 s delete deadline resurrects `doc-9`. Retention must be at least 900 s plus the convergence check.
- **Delete convergence for `doc-9`:** the lexical path stops returning it at 11 s and the vector path at 200 s. Both are within the 300 s constraint, but between 11 s and 200 s the routes disagree. The submission must either apply a tombstone overlay at read time or report that window as a partial delete. Calling it "refresh lag" is wrong: the cause is a failed branch.
- **Query at 420 s:** the representations disagree only on `doc-7` (lexical v6, vector v5). With no retained generations, pinning to the older state is impossible. Acceptable decisions: wait for the vector index within the deadline; or exclude `doc-7` and mark the answer partial if it was needed; or abstain. Combining v5 chunks with v6 text violates the consistency constraint.
- **Conflict fixture:** 90 EUR cited to S3 at 2026-03-15. At 2026-04-15 the answer is 90 EUR under rule 4 as written, or UNRESOLVED (120 EUR and 90 EUR) under the stricter variant. Either is accepted if the variant is declared; 150 EUR is not.
- **Citation arithmetic:** for a table shaped like Lesson 10.5 Answer 1, correctness $2/3$, support completeness $2/4$, coverage $3/4$.

### Rubric Dimensions

- **Idempotency and Read Consistency** (Deliverables 2–3): *Insufficient* says stable IDs and versions make retries safe, or uses a watermark as if it were a read guarantee. *Competent* states the acceptance rule, names the atomic mechanism, decides events 1–9 correctly, keeps tombstones past the replay window, and gives a consistent decision for the 420 s query. *Strong* also tests concurrent duplicates and collection-before-replay, and shows which store capability (conditional write, retained generation) each guarantee depends on.
- **Lineage:** *Insufficient* stores final prompt only. *Competent* preserves stable IDs/versions and transformations. *Strong* reproduces eligibility, order, claims, citations, caches, and failures.
- **Freshness:** *Insufficient* equates write success with visibility. *Competent* measures declared endpoints. *Strong* tests clocks, high-watermarks, retries, partial state, deletes, and rollback.
- **Context:** *Insufficient* changes top-k only. *Competent* measures opportunity/order/noise/truncation. *Strong* uses matched stage ablations and workload slices.
- **Contradiction/Citations:** *Insufficient* trusts newest or reference syntax. *Competent* defines time/authority and support. *Strong* preserves unresolved conflicts and calibrates correctness/completeness judgments.
- **Architecture:** *Insufficient* declares a universal winner. *Competent* matches evidence/model/objective. *Strong* includes all work, router error, load, freshness, and hard constraints.
- **Diagnosis:** *Insufficient* guesses. *Competent* follows the diagnostic sequence. *Strong* identifies earliest divergence, interacting causes, intervention, and remeasurement.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Request and evidence contract | Lesson 10.1 | Lesson 10.1 Guided Practice; LAB A | Mastery Deliverable 1; Lineage rubric row | Replayable request trace with listed unavailable dependencies |
| Idempotent update/delete acceptance | Lesson 10.2 acceptance table and solved traces | Lesson 10.2 Guided Practice traces (a)–(b); LAB A Break & Falsify | Mastery Deliverable 2 (events 1–9); Incident 10.1 steps 1–3; Idempotency and Read Consistency rubric row | Event ledger with per-event decision; Section 09 Reference Checks |
| Query-visible freshness and read consistency | Lesson 10.2 freshness ledger and mixed-version fixture | LAB A | Mastery Deliverable 3 (420 s query, `doc-9` convergence); Incident 10.1 steps 3–6; Freshness rubric row | Per-route visibility distribution, stale/partial rate, read-decision record |
| Query and context orchestration | Lesson 10.3 | Lesson 10.3 Guided Practice; LAB B | Mastery Deliverable 4; Context rubric row | Paired phase table and context lineage |
| Contradiction policy | Lesson 10.4 conflict table | Lesson 10.4 Guided Practice (modified table); LAB C | Mastery Deliverable 5 (both query times); Incident 10.1 step 4 | Normalized claim table and rule trace |
| Claim-level citations and stage attribution | Lesson 10.5 claim-citation table | Lesson 10.5 Guided Practice; LAB C | Mastery Deliverable 6; Incident 10.1 steps 3–4 and 6; Contradiction/Citations rubric row | Claim-evidence table with three ratios from the same rows; adjudication agreement |
| RAG vs long context route decision | Lesson 10.6 route table | Lesson 10.6 Guided Practice (relaxed constraint); LAB D | Mastery Deliverable 7; Architecture rubric row | Matched comparison with constraint checks, weights, and sensitivity |
| Production source trace | Section 05 Haystack trace | Section 05 Trace practice, questions 1–2 | Mastery Deliverable 8 | Pinned trace (revision, file, symbol, entry path) with generalizability boundary |
| Release, rollback, and diagnosis | Lessons 10.2 and 10.5 diagnostic order | LAB A rollback; LAB C audit | Mastery Deliverable 9; Incident 10.1 steps 1–6; Diagnosis rubric row | Canary/rollback record and staged diagnosis |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can demonstrate query-visible freshness rather than ingestion success; survive duplicate/reordered updates, partial failure, cache/replica lag, and deletes; reproduce an answer from immutable lineage; identify rewrite and context-order failures; preserve unresolved contradictions; distinguish citation syntax, correctness, and completeness; trace a current implementation without generalizing it; and defend a workload-specific route under quality, latency, cost, freshness, access, and abstention constraints.

### Module Wrap-Up (Final Mental Model Reconstruction)

RAG is a versioned evidence-control loop. Preserve identity across every transformation, measure visibility at the query boundary, treat cache/fallback state as part of the read contract, treat context as constrained selection, keep contradictions explicit, require claim-level support, and choose architecture from matched evidence rather than fashion.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
