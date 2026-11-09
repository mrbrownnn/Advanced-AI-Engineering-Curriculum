# Module 08 — AI Data Engineering

## 00 Why This Module Exists

Model behavior is bounded by the data pipeline that created and evaluates it. A corpus can be syntactically valid yet semantically wrong, reproducible in name yet mutable in bytes, “deduplicated” under one detector yet contaminated under another, or improved on average while losing rare but important tails.

This module engineers the lifecycle:

$$
\text{source}\to\text{immutable intake}\to\text{contract and lineage}
\to\text{validate/quarantine}\to\text{transform/deduplicate/split}
\to\text{label or synthesize}\to\text{release}\to\text{monitor/rollback}.
$$

It covers provenance, validation, drift, leakage, deduplication, contamination, synthetic data, and active learning. Retrieval indexing belongs to Module 09, model adaptation to Module 19, evaluation-program architecture to Module 15, and operational telemetry to Module 23.

**Research cutoff:** 2026-09-27 for the original revision. Sources added in the 2026-09-30 revision (shift definitions, contamination frontier) were searched through 2026-09-30. Claims not listed as changed in the registry were not re-verified.

**Module Orientation**

- **Engineering problem:** release data that is fit, traceable, replayable, leakage-controlled, measurable under shift, and safely replaceable.
- **What you will do:** build immutable manifests and lineage; implement layered validation and quarantine; diagnose drift and training-serving skew; red-team leakage and decontamination; control synthetic mixtures; operate a human-label and active-learning loop; and defend promotion, deletion, and rollback.
- **Environment:** Python 3.10+ with dataframe, hashing, validation, and statistical tooling; an object-store or filesystem snapshot fixture; optional distributed processing for scale tests. Pin source snapshots, schemas, transforms, tokenizers, detectors, label policy, and execution environment.
- **Evidence rule:** distinguish source observation (**O**), assumption-backed derivation (**D**), and telemetry-dependent hypothesis (**H**). A passed check is evidence only for the invariant that check actually measures.

## 01 Baseline Assumptions

- Module 00: experimental units, provenance, leakage, sampling, uncertainty, and falsification.
- Module 07: behavior events, calibration, distribution shift, slices, and regression.
- Data fundamentals: schemas, joins, hashes, partitions, snapshots, transactions, and access controls.
- ML fundamentals: train/calibration/test boundaries, features/labels, fitting, and target-population risk.

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
  instruction: 5h
  guided_practice: 3h
  labs: 18h
  assessment: 3h
  source_trace: 2h
  total: 31h
```

The learner must build replayable manifests, layered validation, drift experiments, leak-resistant splits, auditable dedup/decontamination, controlled synthetic mixtures, cost-aware active learning, and safe promotion/rollback. They must also preserve consent/license/authority and retention/deletion obligations as executable release constraints rather than documentation-only fields.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
source bytes + authority + collection time
                 |
          immutable identity
                 |
     contract + datasheet + lineage DAG
                 |
  structural -> statistical -> relational -> semantic -> task validation
                 |
           quarantine / approve
                 |
 normalize -> deduplicate -> split -> label/synthesize
                 |
       versioned release + consumer compatibility
                 |
     drift + delayed labels + incident feedback
                 `----> new candidate release, never silent mutation
```

## 04 Lessons

### Lesson 8.1 — Data Contracts, Identity, and Lineage

**Engineering Question:**
Can this exact dataset be explained and rebuilt?

**Concepts & Definitions:**

Datasheets document motivation, composition, collection, processing, use, distribution, and maintenance. An executable manifest adds source snapshot/hash, stable record IDs, parent-child lineage, transform revision/configuration/environment, schema, ordering, seeds, filters, split assignments, label provenance, quality results, and consumer compatibility.

**Quantitative Model / Derivation:**
A dataset display name or mutable URI is not identity. If output is $D=f(S,C,E,R)$ for source bytes $S$, code/config $C$, execution semantics $E$, and randomness/external results $R$, replay requires those arguments or an explicit statement that replay is impossible. Documentation can be stale; hashes do not establish consent, license, authority, retention permission, quality, or fitness. Keep both, and propagate deletion/expiry obligations through derived releases.

*Reading the formula.* $f$ is the whole build pipeline, treated as a function. Replay means calling $f$ again and getting the same $D$, which is only possible when every argument is the same:

- $S$ fixed: the same source **bytes** (a snapshot plus its hash), not "the same URI", because a URI can later point to different bytes.
- $C$ fixed: the same code revision and configuration (filters, thresholds, tokenizer version, normalizer version).
- $E$ fixed: the same execution semantics (library versions, file ordering, parallelism), because these can change results even with identical $S$ and $C$.
- $R$ fixed: the same seeds and the same answers from external calls (API responses, generator outputs). If an external call cannot be repeated, its outputs must be snapshotted, or the manifest must say replay is impossible.

If any one argument is unpinned, two builds may differ and nothing in the manifest can say why. Symbol note: here $D$ is the output dataset; in the worked example below, `D` names the dedup transform, and in evidence tags **D** means "derivation". These are three unrelated uses.

**Break cases:** upstream overwrite, nondeterministic file order, unpinned tokenizer, mutable API response, undocumented manual edit, changed normalization, and a seed that does not cover distributed nondeterminism.

Datasheets complement but do not replace executable lineage (**O**, CLM-001; **D**, CLM-002).

**Worked Example (synthetic lineage ledger):**
*Input.* Two sources feed one support-answer dataset. Content IDs are the first 12 hex digits of SHA-256 over UTF-8 text. `N` is the lowercase and whitespace-collapse normalizer at version `norm-v1`. Dedup `D` keeps the lowest record ID in each normalized-hash cluster. A synthetic paraphraser `G` (a pinned generator revision) creates `syn1` from `s1/r3`.

| record | text (abridged) | raw hash | normalized hash | parents | R1 status |
|---|---|---|---|---|---|
| s1/r1 | "Reset the router … 10 s." | d16627232a7b | 758177e67701 | — | canonical of cluster 758177e67701 |
| s1/r2 | "Warranty lasts 12 months." | bcaa60ff9b25 | 4d5f899c2390 | — | kept |
| s1/r3 | "Call Jane Doe at 555-0100 …" | 322d74ff50c2 | 258806f2d970 | — | kept (contains personal data) |
| s2/r4 | "reset the router … 10 s." | 758177e67701 | 758177e67701 | — | removed as duplicate of s1/r1 |
| syn1 | paraphrase of s1/r3 | (generator output) | — | s1/r3, G@rev | kept, marked synthetic |

Release R1 = {s1/r1, s1/r2, s1/r3, syn1}: 4 records from 5 inputs. The ledger stores each transform edge (N, D, and G with its version) and the split assignment.

*Step 1: correction.* The source owner corrects s1/r2 to "Warranty lasts 24 months." Its new raw hash is 4b135b6cd43c, a different record version. Build R2 as a new release in which s1/r2@v2 replaces s1/r2@v1, and record the correction reason. R1 stays immutable, so anything trained on R1 is still reproducible and auditable.

*Step 2: deletion.* A deletion request arrives for s1/r3. Walking the lineage graph gives the affected set {s1/r3, syn1}: syn1 is a descendant even though its text differs. R3 = {s1/r1, s1/r2@v2}. Consumers that trained on R1 or R2 are notified. Deleting from the data does **not** delete from trained weights; what happens to those models is a separate policy decision (Module 19).

*Result.* The release sizes are 4 → 4 → 2 (R1: 5 inputs − 1 duplicate = 4; R2: one record swapped for its new version, still 4; R3: 4 − |{s1/r3, syn1}| = 2). A hash-only view would miss syn1, because no hash links it to s1/r3; only the parent edge does.

*Interpretation and limits.* Hashes prove byte identity only. Eligibility, meaning consent, license, and retention, lives in separate metadata that can change while bytes stay the same. Replay of `G` is exact only if the generator is deterministic under the recorded seed, or if its outputs were snapshotted.

**Knowledge Check:**
1. Why does a content hash not prove that a record may be retained or trained on?
2. Which inputs besides source bytes must be pinned for replay?

**Guided Practice:**
(a) Using the ledger above, process a deletion of **s1/r1** instead of s1/r3. State the new release contents and which ledger rows change. (b) Build a manifest for one real release, replay it twice, compare bytes, counts, and order, then issue one source deletion and trace every affected derivative and consumer.

**Feedback Contract:**
- *Expected Evidence*: (a) s1/r1 leaves, but s2/r4 is still eligible and was removed *only* as its duplicate. Dedup must be re-run, so s2/r4 becomes the canonical member of cluster 758177e67701. The new release is {s2/r4, s1/r2@v2, s1/r3, syn1} (or without s1/r3 and syn1 if the earlier deletion also stands). Without re-promotion, eligible content disappears silently. (b) Immutable IDs, complete inputs and environment, lineage edges, replay diff, eligibility metadata, and deletion acknowledgements.
- *Common Failure*: Deleting the canonical record and its whole duplicate cluster, missing synthetic descendants, or calling a mutable URI or a seed alone "reproducible".
- *Diagnostic Hint*: Follow parent edges, not hashes. For each removed record, ask whether it was removed as a *duplicate* of something that is now gone.
- *Concept to Revisit*: Dataset Identity and Lineage Closure.

**Learning Outcome:**
Produce a manifest that supports diff, rollback, deletion propagation, and incident reconstruction.

*(Effort: 45m instruction, 30m practice)*

### Lesson 8.2 — Layered Validation and Quarantine

**Engineering Question:**
What can a passed validation gate actually rule out?

**Concepts & Definitions:**

- **Invariant:** a property the data must satisfy for the release contract to hold, for example "`order_id` is unique", "timestamps are epoch seconds within the collection window", or "a left join preserves the left row count".
- **Check:** executable code that tests one invariant on a batch. A check can only rule out the violations it tests for.
- **Anomaly:** a recorded check failure with reason code, feature or slice, observed value, and threshold.
- **Quarantine:** holding a failed batch out of every release while keeping it for diagnosis.
- **Override:** a time-limited, owned, scoped decision to release despite an anomaly, followed by revalidation.
- **False pass / false fail:** the check passes on a batch that violates the invariant it was meant to protect, or fails on a valid batch. Both are measured by injecting defects and by auditing samples.
- **Schema-green:** passes all type, presence, range, and domain checks. This says nothing about units, join multiplicity, or meaning (**O**, CLM-003).

**Mechanism Explanation:**
Validation layers include:

1. byte/container readability and checksums;
2. schema, type, presence, range, enumeration, and cardinality;
3. statistical distributions and slice counts;
4. relational keys, join multiplicity, referential and temporal integrity;
5. semantic units, language, encoding, evidence authority, and label meaning;
6. duplicates, leakage, policy/privacy and task-level canaries.

Infer schemas for exploration, then review and version them. A bad baseline can make inferred anomalies look normal. Failed batches enter quarantine with reason codes; overrides require owner, justification, scope, expiry, and revalidation. A schema-green batch may still contain milliseconds interpreted as seconds or a many-to-many join explosion.

**Source trace:** the pinned TFDV snapshot exposes schema/statistics validation and drift/skew records (**O**, CLM-013). This is one implementation, not the definition of data quality.

**Worked Example (synthetic; a schema-green join failure):**
*Input.* `orders(order_id int, user_id str, amount_cents int)` has 4 rows: (101, u1, 1250), (102, u2, 4000), (103, u1, 999), (104, u3, 1500). `profiles(user_id str, country str)` has 4 rows, but u1 appears twice (VN and SG) because an upstream change started emitting one row per profile *version*. The feature job runs `orders LEFT JOIN profiles USING (user_id)`.

*Schema checks.* Types match, no nulls, `amount_cents` lies in $[0, 10^6]$, and `country` is in the allowed enumeration. Every check passes.

*Steps.* Orders 101 and 103 each match two profile rows, so the output has $4+2=6$ rows. Summed revenue becomes $1250\cdot2+4000+999\cdot2+1500=9998$ cents instead of $7749$, a 29.0% inflation. Every row is individually valid.

Step by step: a left join emits, for each left row, one output row per matching right row (or one row with nulls if there is no match). So

$$|out|=\sum_{\text{orders } o}\max\bigl(1,\,m(o)\bigr),$$

where $m(o)$ is the number of `profiles` rows with the same `user_id` as order $o$. Here $m=2$ for u1 and $m=1$ for u2 and u3, so $|out|=2+1+2+1=6$. Each order's amount is counted $m(o)$ times: the correct total is $1250+4000+999+1500=7749$; the extra copies of orders 101 and 103 add $1250+999=2249$, giving $7749+2249=9998$; and $2249/7749\approx0.290$, the 29.0% inflation.

*Invariant checks that catch it.* (1) Key uniqueness on the dimension side: `profiles.user_id` is unique fails, with u1 appearing twice. (2) Row conservation for a many-to-one left join: $|out|=|orders|$ fails, since $6\ne4$. (3) An additive total reconciled against the source ledger: $9998\ne7749$ fails.

*Second defect, units.* The event timestamp `1727700000000` is epoch **milliseconds**. It passes an "int64, non-negative" check. Read as seconds, it lies about 54,748 years after 1970 ($1.7277\times10^{12}\,\text{s}\div(365.2425\cdot86400\,\text{s/year})\approx54{,}748$ years). Read correctly as milliseconds, it is $1{,}727{,}700{,}000$ s after 1970, which is 2024-09-30 UTC. A declared-unit check or a collection-window range check (for example, a 2020–2030 window) catches it; a type check cannot.

*Result.* Schema validation passed both defective batches. Only relational invariants and semantic range checks rejected them.

*Interpretation and limits.* Each added check rules out one more class of bad batch. None proves the batch is task-valid. A many-to-many join where both sides are legitimately non-unique needs an explicit contract saying which row wins.

**Knowledge Check:**
1. Why is a schema-green batch not necessarily task-valid?
2. What governance must accompany a manual override?

**Guided Practice:**
(a) The same orders are joined to a `profiles` table in which u2 has *three* rows and u1 has one. Compute the output row count and summed revenue, and name the first invariant that fails. (b) Inject one defect at each validation layer and record whether the gate detects, quarantines, overrides, and revalidates it.

**Feedback Contract:**
- *Expected Evidence*: (a) 6 rows ($1+3+1+1$); revenue $1250+3\cdot4000+999+1500=15749$ cents instead of 7749. Dimension-key uniqueness fails first, then row conservation and the revenue reconciliation. (b) Check-to-invariant matrix, quarantined samples, false pass/fail audit, owner, expiry, and rollback action.
- *Common Failure*: Treating inferred schema or aggregate distributions as semantic truth, or "fixing" the output by `DISTINCT`, which hides which profile row won.
- *Diagnostic Hint*: State exactly which bad worlds remain possible after the check passes. Compare the output row count with the left input row count first.
- *Concept to Revisit*: Layered Validation Boundaries.

**Learning Outcome:**
Map every check to its invariant, blind spots, action, owner, and rollback.

*(Effort: 50m instruction, 30m practice)*

### Lesson 8.3 — Drift, Skew, and Alert Evidence

**Engineering Question:**
What changed, and does the change harm the target?

**Concepts & Definitions:**

Let $P_S$ be the reference (training) distribution and $P_T$ the current (serving) distribution over inputs $X$ and targets $Y$. Every joint factorizes two ways:

$$P(X,Y)=P(Y|X)P(X)=P(X|Y)P(Y).$$

Both equalities are the definition of conditional probability, $P(Y\mid X)=P(X,Y)/P(X)$ and $P(X\mid Y)=P(X,Y)/P(Y)$, each multiplied back out. Read them as two recipes for generating one data point: "draw an input, then its label" or "draw a label, then an input that fits it". A shift type says which ingredient of a recipe changed between $P_S$ and $P_T$.

A named shift type is a claim about **which factor changes and which stays invariant**. The invariant is the assumption that makes correction methods work, so it must be stated:

| shift type | what changes | what is assumed invariant | what unlabeled $X$ monitoring can see |
|---|---|---|---|
| covariate shift | $P(X)$ | $P(Y\mid X)$ (**O**, CLM-015) | the $P(X)$ change, but not whether $P(Y\mid X)$ really stayed fixed |
| label (prior) shift | $P(Y)$ | $P(X\mid Y)$ (**O**, CLM-016) | a change in $P(X)$ induced through $P(Y)$, but it cannot confirm that $P(X\mid Y)$ is fixed |
| concept shift | $P(Y\mid X)$ | nothing needs to be fixed; $P(X)$ may be unchanged | possibly nothing, because $P(X)$ can be identical |
| mixed shift | several factors | none of the above | a marginal change it cannot attribute |
| training–serving skew | the *pipeline* that computes $X$ (or $Y$) differs | — | feature-level differences, which may look exactly like population drift |

Under pure covariate shift, a model of $P(Y\mid X)$ that was correct on $P_S$ stays correct pointwise; only which regions of $X$ matter changes. Under pure label shift, the class-conditional input distributions are fixed, which is what lets black-box estimators recover $P_T(Y)$ from unlabeled predictions under the stated conditions (**O**, CLM-016). **An unlabeled drift alarm can show that $P(X)$ moved. It cannot prove that the shift is *pure* covariate shift, because a mixed shift can produce exactly the same $P_T(X)$ (**D**, CLM-004).**

**Quantitative Model / Derivation:**
Drift signals include missingness/cardinality, quantiles, category shares, distances/tests, embedding or classifier two-sample methods, and slice/task canaries. High-dimensional projections may miss rare slices; many tests create false alerts; huge samples make tiny differences statistically detectable.

For $B$ bins with current proportions $p_i$ and reference proportions $q_i$, both over identical, exhaustive bins with **every $p_i>0$ and $q_i>0$**:

$$PSI=\sum_{i=1}^{B}(p_i-q_i)\ln\frac{p_i}{q_i}.$$

where:

- $B$ is the number of bins (fixed edges, shared by both histograms);
- $p_i$ is the fraction of **current** values in bin $i$ and $q_i$ the fraction of **reference** values, so $\sum_i p_i=\sum_i q_i=1$;
- $\ln$ is the natural log.

Each term is non-negative, because $p_i-q_i$ and $\ln(p_i/q_i)$ have the same sign. (If $p_i>q_i$ then $p_i/q_i>1$ and the log is positive; if $p_i<q_i$ both factors are negative; if they are equal the term is 0.) So PSI is 0 only when every bin share is unchanged. The log makes a small bin that empties out expensive: $q_i=0.1$ falling to $p_i=0.001$ contributes $(-0.099)\ln(0.01)\approx0.456$, while $q_i=0.5$ falling to $p_i=0.4$ contributes only $(-0.1)\ln(0.8)\approx0.022$, although both lose about 0.1 of share. A zero bin makes a term undefined, so a pseudocount or bin-merging policy is part of the metric definition (**D**, CLM-005). PSI changes with bins and smoothing. There is no universal cutoff or quality mapping. Calibrate alerts against incidents, downstream labels, lead time, operator cost, and simpler baselines (**H**, CLM-014).

**Worked Example 1 (synthetic mixed-shift joint distribution):**
*Input.* $X\in\{a,b\}$ (for example, query language) and $Y\in\{0,1\}$ (answer needs escalation). The deployed rule predicts $\hat Y=0$ for $a$ and $\hat Y=1$ for $b$.

| joint $P(X,Y)$ | $(a,0)$ | $(a,1)$ | $(b,0)$ | $(b,1)$ | $P(X=a)$ | $P(Y=1\mid a)$ | $P(Y=1\mid b)$ | $P(Y=1)$ | error of rule |
|---|---|---|---|---|---|---|---|---|---|
| $S$ reference | 0.36 | 0.04 | 0.18 | 0.42 | 0.40 | 0.10 | 0.70 | 0.46 | 0.22 |
| $T_1$ pure covariate | 0.63 | 0.07 | 0.09 | 0.21 | 0.70 | 0.10 | 0.70 | 0.28 | 0.16 |
| $T_2$ mixed | 0.42 | 0.28 | 0.09 | 0.21 | 0.70 | **0.40** | 0.70 | 0.49 | **0.37** |
| $T_3$ pure label | 0.533 | 0.017 | 0.267 | 0.183 | 0.551 | 0.032 | 0.406 | 0.20 | 0.284 |

*Steps.* $T_1$ keeps $P(Y\mid X)$ from $S$ and sets $P(X=a)=0.7$. $T_2$ has the same $P(X)$ but raises $P(Y=1\mid a)$ from 0.1 to 0.4, so $P(Y\mid X)$ changed. It is also not a label shift, because $P(X=a\mid Y=1)$ moves from $0.087$ to $0.571$. $T_3$ sets $P(Y=1)=0.2$ and keeps $P(X\mid Y)$ from $S$; for example $P(X=a\mid Y=0)=0.667$ in both $S$ and $T_3$. In $T_3$, $P(Y\mid X)$ changes as a *consequence*, which is why label-shift correction must not assume $P(Y\mid X)$ is fixed. The error is $P(a,1)+P(b,0)$.

How each row is read and built (all values from the table):

- Marginal and conditionals from a joint row: $P(X=a)=P(a,0)+P(a,1)$ and $P(Y=1\mid a)=P(a,1)/P(X=a)$. For $S$: $0.36+0.04=0.40$ and $0.04/0.40=0.10$; $P(Y=1\mid b)=0.42/0.60=0.70$; $P(Y=1)=0.04+0.42=0.46$.
- Error of the rule: it predicts 0 on $a$, so it is wrong on $(a,1)$; it predicts 1 on $b$, so it is wrong on $(b,0)$. For $S$: $0.04+0.18=0.22$.
- $T_1$ (keep $P(Y\mid X)$, change $P(X)$): $P(a,1)=0.7\cdot0.1=0.07$, $P(a,0)=0.63$, $P(b,1)=0.3\cdot0.7=0.21$, $P(b,0)=0.09$.
- $T_2$ (same $P(X)$ as $T_1$, change $P(Y\mid a)$): $P(a,1)=0.7\cdot0.4=0.28$, $P(a,0)=0.42$; the $b$ cells are as in $T_1$. Label-shift check: $P(X=a\mid Y=1)=P(a,1)/P(Y=1)$ is $0.04/0.46=0.087$ in $S$ and $0.28/0.49=0.571$ in $T_2$.
- $T_3$ (keep $P(X\mid Y)$, change $P(Y)$): from $S$, $P(a\mid Y=0)=0.36/0.54=0.667$ and $P(a\mid Y=1)=0.04/0.46=0.087$. Multiply by the new priors $P(Y=0)=0.8$, $P(Y=1)=0.2$: $P(a,0)=0.8\cdot0.667=0.533$, $P(a,1)=0.2\cdot0.087=0.017$, $P(b,0)=0.8\cdot0.333=0.267$, $P(b,1)=0.2\cdot0.913=0.183$. Then $P(Y=1\mid a)=0.017/0.551=0.032$, no longer 0.10.

*Result.* $T_1$ and $T_2$ have **identical** $P(X)$, so any unlabeled $X$ monitor gives the same alarm for both. PSI on $X$ is $(0.7-0.4)\ln(0.7/0.4)+(0.3-0.6)\ln(0.3/0.6)=0.376$ in each case (term by term: $0.3\cdot0.560=0.168$ and $(-0.3)(-0.693)=0.208$). Yet the rule's error *falls* from 0.22 to 0.16 under $T_1$ and *rises* to 0.37 under $T_2$.

*Interpretation and limits.* The alarm is necessary evidence that something moved, but it cannot tell $T_1$ from $T_2$, and so cannot tell harmless from harmful. Discriminating them needs delayed labels, a labeled canary slice, or a justified causal proxy. This is a four-cell toy; real drift involves high-dimensional $X$ and uncertain estimates.

**Worked Example 2 (synthetic PSI with zero bins):**
*Input.* Four fixed bins of one feature. Reference counts are $(120,380,400,100)$ and current counts are $(300,350,350,0)$, with $n=1000$ each.

*Steps.* The last current bin is empty, so the raw PSI term $(0-0.1)\ln(0/0.1)$ is undefined. Policy A adds a pseudocount $\epsilon=0.5$ to every bin of both histograms and renormalizes, so each proportion becomes $(\text{count}_i+\epsilon)/(n+B\epsilon)$ with $n=1000$, $B=4$ (denominator 1002). The empty bin becomes $p_4=0.5/1002=0.0005$ against $q_4=100.5/1002=0.1003$, so its term is $(0.0005-0.1003)\ln(0.0005/0.1003)=(-0.0998)(-5.30)=0.529$. All four terms are $0.1642, 0.0025, 0.0067, 0.5293$, so $PSI=0.702$. Policy B uses $\epsilon=5$, giving $PSI=0.465$. Policy C merges bins 3 and 4, giving proportions $(0.30,0.35,0.35)$ vs $(0.12,0.38,0.50)$ and $PSI=0.221$.

Why the policies disagree: most of the PSI comes from the emptied bin, and its log-ratio is set by $\epsilon$. A smaller pseudocount pushes $p_4$ closer to 0 and the log more negative, so PSI rises; merging removes the empty bin entirely.

*Result.* The same data gives PSI of 0.70, 0.46, or 0.22 depending on a policy choice. Under a hypothetical cut-off of 0.25, the alert fires or not depending on the smoothing rule.

*Interpretation and limits.* The zero-bin policy, bin edges, and $n$ belong in the alert definition. None of these numbers says whether model quality changed.

**Worked Example 3 (synthetic 30-day alert cost table):**
*Exercise assumptions.* Each alert costs 1.5 engineer-hours at 80 per hour. Each missed incident costs 3,000. The window contains 5 labeled incidents.

Formulas used in the table, with $A$ = alerts fired and $k$ = true incidents flagged:

- precision $=k/A$ (share of alerts that were real); recall $=k/5$ (share of incidents caught);
- investigation cost $=A\cdot1.5\cdot80=120A$; missed-incident cost $=(5-k)\cdot3000$;
- total = investigation + missed. Row 1: $120\cdot40=4800$ plus $(5-4)\cdot3000=3000$, total 7,800.

| alert rule | alerts | true incidents flagged | precision | recall | investigation cost | missed-incident cost | total |
|---|---|---|---|---|---|---|---|
| PSI > 0.10 on any feature | 40 | 4 | 0.10 | 0.80 | 4,800 | 3,000 | 7,800 |
| PSI > 0.25 **and** labeled canary-slice drop | 8 | 3 | 0.375 | 0.60 | 960 | 6,000 | 6,960 |
| labeled canary only | 3 | 2 | 0.67 | 0.40 | 360 | 9,000 | 9,360 |

*Result.* Under these costs the combined rule has the lowest total, even though neither its precision nor its recall is the best. Doubling the miss cost to 6,000 changes the totals to 10,800 / 12,960 / 18,360, so the broad PSI rule becomes cheapest. (Only the missed term doubles: $4800+1\cdot6000$, $960+2\cdot6000$, $360+3\cdot6000$.)

*Interpretation and limits.* The ranking of alert rules depends on the cost ratio, which is a product decision. Five incidents give very wide uncertainty on recall, so evaluate on held-out windows before adopting a rule (**H**, CLM-014).

**Knowledge Check:**
1. Can unlabeled $X$ monitoring identify $P(Y\mid X)$ change in general?
2. Under label shift, which conditional is assumed fixed, and does $P(Y\mid X)$ stay fixed?
3. Why does a universal PSI threshold lack a stable quality meaning?

**Guided Practice:**
(a) Construct a $T_4$ with $P(X=a)=0.4$ (unchanged from $S$) but $P(Y=1\mid a)=0.4$ and $P(Y=1\mid b)=0.7$. Compute PSI on $X$, the rule's error, and name the shift. (b) Create controlled covariate, label, concept, and pipeline-skew cases, and cross alert signals with delayed labels and rare slices.

**Feedback Contract:**
- *Expected Evidence*: (a) $P(X)$ is unchanged, so PSI on $X$ is $0$. The joint is $(0.24,0.16,0.18,0.42)$ and the error is $0.16+0.18=0.34$, up from 0.22. This is concept shift, fully invisible to an $X$-only monitor. (b) Population and window, bins and smoothing, multiple-test policy, task labels where available, and alert-to-incident utility.
- *Common Failure*: Calling an $X$-drift alarm "covariate shift" as though $P(Y\mid X)$ had been checked, or naming every distribution change "concept drift".
- *Diagnostic Hint*: Which factor of $P(X,Y)$ did you actually observe, and which invariant are you *assuming*?
- *Concept to Revisit*: Shift Identifiability; Conditional Invariance.

**Learning Outcome:**
Report symptom → competing shift hypotheses → missing labels/evidence → discriminating measurement → action → remeasurement.

*(Effort: 60m instruction, 30m practice)*

### Lesson 8.4 — Leakage, Deduplication, and Contamination

**Engineering Question:**
Did unavailable or evaluation information cross the learning boundary?

**Concepts & Definitions:**

Leakage is information unavailable at the intended prediction boundary that influences fitting, preprocessing, selection, or evaluation (**O**, CLM-006). It can enter through future timestamps, same entity/session across splits, relatives/derivatives, fitted preprocessing before splitting, label-derived features, target-aware filtering, repeated prompt templates, benchmark items in pre/post-training, or selection on the test set.

**Mechanism Explanation:**
Split by the deployment unit and time before fitting transformations. Then audit exact and near relations across all splits and training stages. Exact hash, normalized text, n-gram, MinHash, embedding, and semantic matchers each define different equivalence relations. Record thresholds, clusters, canonical policy, removals, false-merge audits, and residual risk.

Lee et al. provide scoped evidence that deduplication reduced memorized output and overlap in studied LM corpora (**O**, CLM-007). Dedup is not monotonically beneficial: legitimate templates, quotations, repeated events, and minority patterns can be removed. A clean result means “no match under this detector on these accessible snapshots,” not “never seen” (**D**, CLM-008).

**Worked Example (synthetic labeled pair audit):**
*Input.* Six benchmark–corpus pairs were labeled by hand as derived from the benchmark item (1) or not (0). Each pair has an embedding similarity score.

| pair | relation | label | exact hash match | normalized hash match | embedding similarity |
|---|---|---|---|---|---|
| P1 | verbatim copy | 1 | yes | yes | 1.00 |
| P2 | case/whitespace variant | 1 | no | yes | 0.99 |
| P3 | paraphrase | 1 | no | no | 0.88 |
| P4 | translation | 1 | no | no | 0.71 |
| P5 | same topic, different content | 0 | no | no | 0.84 |
| P6 | shared boilerplate template | 0 | no | no | 0.62 |

*Steps.* Precision = flagged pairs that are truly derived ÷ flagged pairs. Recall = flagged derived pairs ÷ 4 derived pairs. In symbols, with $TP$ = flagged pairs labeled 1 and $FP$ = flagged pairs labeled 0: precision $=TP/(TP+FP)$ and recall $=TP/4$. For example, embedding ≥ 0.80 flags P1, P2, P3 ($TP=3$) and P5 ($FP=1$): precision $3/4=0.75$, recall $3/4=0.75$. A threshold detector flags every pair scoring at or above the cut, so lowering the cut only adds flags: recall can only rise, and precision falls whenever an added pair is non-derived.

| detector | flags | precision | recall |
|---|---|---|---|
| exact hash | P1 | 1.00 | 0.25 |
| normalized hash | P1, P2 | 1.00 | 0.50 |
| embedding ≥ 0.85 | P1, P2, P3 | 1.00 | 0.75 |
| embedding ≥ 0.80 | P1, P2, P3, P5 | 0.75 | 0.75 |
| embedding ≥ 0.70 | P1–P5 | 0.80 | 1.00 |

*Result.* Exact matching reports "one contaminated item" while three derived items remain. Sorting pairs by score shows why: P1 1.00, P2 0.99, P3 0.88, **P5 0.84**, P4 0.71, P6 0.62. A non-derived pair sits between two derived ones, so no threshold on this score separates the translation (0.71) from the unrelated same-topic document (0.84), so full recall costs a false merge.

*Interpretation and limits.* A "clean" report is a statement about one detector on one accessible snapshot. Six pairs are far too few to estimate a detector's rates, so a real audit needs a labeled sample sized for the decision. Removing P5 as a "duplicate" deletes legitimate training content.

**Knowledge Check:**
1. Why must grouped/time splits precede fitting transforms?
2. What does a clean detector result fail to prove?

**Guided Practice:**
(a) In the table above, compute precision and recall for embedding ≥ 0.60. (b) Construct exact, normalized, paraphrase, translation, entity-relative, and unrelated-hard-negative pairs. Compare matchers and inspect false merges/misses.

**Feedback Contract:**
- *Expected Evidence*: (a) All six pairs are flagged: precision $4/6=0.667$, recall $4/4=1.0$. Lowering the threshold from 0.70 to 0.60 adds only a false merge (P6). (b) Equivalence definition, thresholds, labeled audit sample, clusters, removals, downstream deltas, and residual-risk statement.
- *Common Failure*: Treating one matcher as complete or deduplication as monotonically beneficial.
- *Diagnostic Hint*: Which transformation preserves benchmark information while defeating the detector?
- *Concept to Revisit*: Detector-Relative Contamination.

**Learning Outcome:**
Defend evaluation independence without claiming perfect contamination detection.

*(Effort: 45m instruction, 30m practice)*

### Lesson 8.5 — Synthetic Data as a Versioned Dependency

**Engineering Question:**
Does generated data add coverage or amplify the generator's blind spots?

**Concepts & Definitions:**

- **Synthetic record:** a training or evaluation record whose content was produced by a model rather than observed. It remains a derived artifact with parents, never a source.
- **Generator:** the pinned model, prompt, decoding settings, seed, and tools that produced the record.
- **Filter / judge:** the pinned rule or model that accepts, rejects, or labels generated records. If it shares a model family with the generator, the two can share errors.
- **Mixture weight:** the fraction of training work (tokens or examples, declared) drawn from synthetic records. "30% synthetic" is ambiguous until the unit is stated. Example: 7,000 real examples of 300 tokens and 3,000 synthetic examples of 900 tokens are 30% synthetic by examples ($3000/10000$) but 56.25% synthetic by tokens ($2.7\text{M}/4.8\text{M}$).
- **Recursion round:** training generation $g+1$ on data that includes outputs of generation $g$. Model-collapse results concern this regime (**O**, CLM-009).
- **Real-data anchor:** a fixed, independently collected set that every synthetic mixture is evaluated against, and ideally trained alongside.

**Mechanism Explanation:**
Each synthetic record needs seed/parent links, generator and tokenizer revision, system/user prompt, decoding/seed, tools/evidence, filters, judge/labeler, policy decisions, and generation time. Preserve real versus synthetic origin and mixture weights (**D**, CLM-010).

Shumailov et al. show model-collapse behavior in recursive regimes; Seddik et al. analyze conditions for synthetic-only versus mixed regimes. These results reject “synthetic data is free real data,” not all augmentation. Test generator families, real-data anchors, repeated generations, tail/subgroup retention, novelty, privacy/memorization, judge correlation, and downstream utility at equal training budget.

**Break cases:** same model generates and judges; easy-example amplification; rare-mode loss; benchmark leakage through prompts; synthetic duplicates; error feedback; and quality filters that narrow style/diversity.

**Worked Example (synthetic exercise numbers, single run each):**
*Input.* Four training mixtures at equal training tokens: real-only and 10%, 30%, and 60% synthetic by tokens. Each is evaluated on a real-data anchor holdout (mean accuracy) and a rare-language slice.

| mixture | anchor mean | rare slice | synthetic records rejected by filter |
|---|---|---|---|
| real-only | 0.700 | 0.600 | — |
| 10% synthetic | 0.712 | 0.598 | 18% |
| 30% synthetic | 0.731 | 0.552 | 11% |
| 60% synthetic | 0.735 | 0.470 | 6% |

*Steps.* Mean gain vs real-only is $+0.012, +0.031, +0.035$; rare-slice change is $-0.002, -0.048, -0.130$ (each is the row value minus the real-only value, e.g. $0.731-0.700=+0.031$ and $0.552-0.600=-0.048$). At equal training tokens, a synthetic share $w$ leaves a fraction $1-w$ of the budget for real tokens: at 60% synthetic, real tokens, rare-language ones included, drop to 40% of the real-only run. The filter rejection rate falls as the synthetic share rises, which is worth checking: either the generator is improving or the filter is increasingly agreeing with it.

*Result.* Beyond 10%, the mean rises while the rare slice falls. The "best mean" choice (60%) costs 13 points on the rare slice.

*Interpretation and limits.* These are single runs, so first repeat with different seeds to size run-to-run variance. Competing explanations include correlated generator/judge error, fewer real rare-language tokens at equal budget, and anchor contamination through the prompts. Ablate each before concluding that synthetic data "caused" the slice loss.

**Knowledge Check:**
1. Why is synthetic ancestry required for rollback?
2. What risk arises when the same model family generates and filters examples?

**Guided Practice:**
Run a controlled mixture ablation with a real-data anchor, independent audit labels, duplicate/novelty checks, and tail slices.

**Feedback Contract:**
- *Expected Evidence*: Complete generation provenance, mixture weights with their unit, equal-budget comparison, repeated seeds, independent audit, tail results, and removable lineage. In the table above, a correct write-up names the 10% mixture as the only one without a material rare-slice loss, and does not declare a winner before measuring variance.
- *Common Failure*: Discarding origin or selecting only by the generator's own judge.
- *Diagnostic Hint*: Which errors are shared by generator, filter, and downstream model?
- *Concept to Revisit*: Synthetic Feedback Correlation.

**Learning Outcome:**
Ship synthetic data only as an ablated, lineage-complete mixture with removal and rollback.

*(Effort: 45m instruction, 30m practice)*

### Lesson 8.6 — Active Learning and Human Label Systems

**Engineering Question:**
Which example is worth labeling next, at what cost?

**Concepts & Definitions:**

- **Unlabeled pool:** the candidate items available for annotation. It is usually *not* the target distribution.
- **Query strategy:** the rule that selects the next batch, for example by uncertainty, diversity, density, disagreement, expected error reduction, or a hybrid (**O**, CLM-011).
- **Annotation cost:** the measured time and money per item, including expertise level, adjudication, and abstentions. Item count is not cost.
- **Adjudication:** resolving rater disagreement under a versioned guide. Persistent disagreement can mean the guide is ambiguous rather than that a rater failed.
- **Representative holdout:** a fixed evaluation set drawn from the target population that the query policy never sees (**D**, CLM-012).
- **Label-efficiency curve:** holdout utility plotted against cumulative annotation *cost*.

**Mechanism Explanation:**
The loop scores a target pool, selects a batch, labels/adjudicates it, updates the model/data, and evaluates on an independent representative holdout. Query strategies can use uncertainty, expected change/error reduction, diversity, density, disagreement, coverage, or hybrids. The label system also versions guidelines, qualification, blind assignment, abstention, disagreement, adjudication, reviewer drift, privacy controls, and correction propagation.

Uncertainty-only batches can be redundant, out-of-distribution, adversarial noise, or artifacts of early miscalibration. Batch selection needs diversity and representativeness; the objective needs annotation time, expertise, disagreement, abstention, and delay—not item count alone.

Plot representative holdout utility against cumulative annotation cost. Keep initialization, model-update schedule, annotator policy, and test set fixed when comparing strategies. The adaptively queried pool is selection-biased and is not the target test distribution without a justified estimator.

**Worked Example (synthetic exercise costs and gains):**
*Input.* Three strategies each run one acquisition round from the same initial model, using the same annotator pool rules. Gains are measured on the same representative holdout.

| strategy | items | minutes/item | rater rate (per hour) | annotation hours | cost | holdout gain (points) | gain per 100 items | cost per point |
|---|---|---|---|---|---|---|---|---|
| random | 400 | 1.5 | 30 | 10.0 | 300 | 3.0 | 0.75 | 100 |
| uncertainty | 250 | 4.0 | 60 | 16.7 | 1,000 | 3.2 | 1.28 | 312.5 |
| hybrid (uncertainty + diversity) | 300 | 2.5 | 60 | 12.5 | 750 | 3.4 | 1.13 | 220.6 |

*Steps.* Cost = items × minutes/60 × rate. Gain per 100 items = gain ÷ items × 100. Cost per point = cost ÷ gain. For uncertainty: $250\times4.0/60=16.7$ hours, $16.7\times60=1000$; $3.2/250\times100=1.28$; $1000/3.2=312.5$. Uncertainty sampling picks harder items that need an expert rate and take longer per item.

*Result.* Per item, uncertainty sampling looks best (1.28 points per 100 items). Per unit cost, random sampling is 3.1× cheaper per point (100 vs 312.5; $312.5/100=3.125$). The ranking reverses when the objective changes from items to cost.

*Interpretation and limits.* These numbers are exercise assumptions from one round with no uncertainty; real comparisons need repeated rounds and intervals on the gains. The uncertainty strategy's items also cannot be used as a test set, because they were selected by the model.

**Knowledge Check:**
1. Why is the adaptively selected pool not a representative test set?
2. Which disagreement indicates ambiguous guidance rather than annotator failure?

**Guided Practice:**
Compare random, uncertainty, and diversity-aware batches under equal annotation minutes. Blind a shared subset, adjudicate disagreements, and measure representative holdout utility.

**Feedback Contract:**
- *Expected Evidence*: Selection logs, guideline/annotator revisions, time and disagreement, adjudication, independent holdout, and cost-normalized curve. For the table above, a correct answer computes cost per point for each strategy and states that the preferred strategy depends on whether annotation budget or item count is the binding constraint.
- *Common Failure*: Evaluating on the selected pool or ignoring expert time.
- *Diagnostic Hint*: Hold the evaluation population fixed while acquisition changes.
- *Concept to Revisit*: Adaptive Sampling and Label-System QA.

**Learning Outcome:**
Operate a query-label-audit loop that improves target utility rather than its own queue metric.

*(Effort: 55m instruction, 30m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Gebru et al. (CACM 2021).
- [Data Validation for Machine Learning](https://proceedings.mlsys.org/paper_files/paper/2019/file/928f1160e52192e3e0017fb63ab65391-Paper.pdf) — Breck et al. (MLSys 2019).
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) — Lee et al. (ACL 2022).
- [A Survey of Active Learning for Natural Language Processing](https://arxiv.org/abs/2210.10109) — Zhang, Strubell, and Hovy (EMNLP 2022).
- [A One-step Approach to Covariate Shift Adaptation](https://proceedings.mlr.press/v129/zhang20a.html) — Zhang, Yamane, Lu, and Sugiyama (ACML 2020, PMLR 129). The abstract defines covariate shift as a changed input distribution with an unchanged conditional distribution of output given input. Abstract-level reading (CLM-015).
- [Detecting and Correcting for Label Shift with Black Box Predictors](https://proceedings.mlr.press/v80/lipton18a.html) — Lipton, Wang, and Smola (ICML 2018, PMLR 80). The abstract defines label shift as a changed $p(y)$ with an unchanged $p(x\mid y)$, and introduces BBSE. Abstract-level reading (CLM-016).

**CURRENT DEFAULT** (the module's *recommended baseline*, derived from CLM-002, CLM-003, CLM-006, and CLM-010; not a measured survey of industry adoption): immutable versions, lineage, layered gates, split-before-fit, cross-split duplicate audits, separately identifiable synthetic/feedback data, quarantine, canary, and rollback.

**WORKLOAD-DEPENDENT:** drift metric/threshold, semantic match rule, synthetic mixture/filter, active-learning query strategy, and human adjudication policy.

**FRONTIER:**

- Contamination in RL post-training: [Tao et al., *Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models*](https://arxiv.org/abs/2510.09259) (arXiv v1 October 2025, v2 March 2026; the arXiv comment states ICLR 2026 acceptance, which was not independently checked). The authors report that earlier detectors are near chance for RL-phase contamination and propose an entropy-based probe with a new benchmark. Author-reported, abstract-level reading (CLM-017).
- Dynamic benchmarks as a contamination response: [Chen et al., *Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation*](https://arxiv.org/abs/2502.17521) (arXiv v1 February 2025, v2 September 2025). A survey that argues standardized criteria for evaluating dynamic benchmarks are missing. Abstract-level reading (CLM-018).
- `TODO_VERIFY` (CLM-019): semantic decontamination, automated data valuation, learned quality filters, and adaptive synthetic mixtures were **not** re-surveyed from 2025–2026 primary sources in this revision. No specific result is claimed, and none is a default.

**LEGACY:** one mutable “latest” dataset; schema-only quality; a universal PSI threshold; random row split for grouped/time data; exact-match-only decontamination; synthetic origin discarded.

**PRODUCTION SOURCE TRACE**

- Repository: `tensorflow/data-validation`
- Revision: `aaa0ae1902262cac7b306358588d61e8521f3b25`
- Verified: 2026-09-26, static inspection only.
- Files/symbols: `api/validation_api.py::validate_statistics`, `statistics/stats_options.py::StatsOptions`, `utils/display_util.py::get_drift_skew_dataframe`.
- Scope (**O**, CLM-013): proves behavior of this pinned upstream snapshot, not every TFX pipeline or universal validation semantics. The 2026-09-26 static read was not repeated in the 2026-09-30 revision.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Replayable Release and Layered Gate

- **Objective:** ingest two sources into immutable snapshots and build lineage, layered gates, quarantine, controlled overrides, deletion propagation, and a consumer canary.
- **Pre-Registered Hypothesis:** replay with all declared inputs pinned will reproduce the release, while at least one schema-green semantic defect will require a deeper gate.
- **Independent Variables:** source version/order, transform environment, defect type, override, and deletion event.
- **Dependent Variables:** byte/count/order equality, gate detection, false pass/fail, quarantine, replay time, deletion convergence, and canary outcome.
- **Break & Falsify:** inject unit, encoding, join multiplicity, missing-ID, mutable-source, eligibility, and schema-green semantic failures.
- **Alignment:** Lessons 8.1 and 8.2.
- **Effort Estimate:** 4.5h.

### LAB B — Drift Without a Single-Score Story

- **Objective:** create controlled shift families and connect alert evidence to delayed task consequences.
- **Pre-Registered Hypothesis:** no one marginal drift score will correctly identify every injected cause or harm state.
- **Independent Variables:** covariate/label/concept/pipeline shift, magnitude, binning/smoothing, sample size, and slice prevalence.
- **Dependent Variables:** distribution signals, task quality, alert precision/lead time, false alerts, and slice harm.
- **Break & Falsify:** include high-signal/no-harm and low-aggregate-signal/rare-severe-harm counterexamples, plus a pure-covariate and a mixed shift with *identical* $P(X)$ (as in Lesson 8.3, Worked Example 1). For each injected family, state which conditional was held invariant and show that the unlabeled alarm cannot tell the two apart.
- **Alignment:** Lesson 8.3.
- **Effort Estimate:** 4h.

### LAB C — Leakage and Decontamination Red Team

- **Objective:** construct leakage and related-item attacks, compare detectors, rebuild clean splits, and bound residual risk.
- **Pre-Registered Hypothesis:** exact matching will miss declared semantic derivatives, while looser detectors will introduce false merges.
- **Independent Variables:** relation type, detector, threshold, split unit/time, and accessible corpus scope.
- **Dependent Variables:** detector precision/recall on audited pairs, clusters/removals, score change, tail loss, and residual uncertainty.
- **Break & Falsify:** include entity/time leakage, fit-before-split, exact/near/paraphrase/translation derivatives, legitimate templates, and minority patterns.
- **Alignment:** Lesson 8.4.
- **Effort Estimate:** 4.5h.

### LAB D — Synthetic Mixtures and Active Labeling

- **Objective:** compare controlled synthetic mixtures and active-label strategies at equal training and annotation budgets.
- **Pre-Registered Hypothesis:** mixture and query-policy gains will be workload-dependent and at least one correlated-error/rare-mode stress will reverse an aggregate gain.
- **Independent Variables:** real/synthetic ratio, recursive round, generator/judge, query policy, annotation budget, and shift.
- **Dependent Variables:** representative holdout and slice utility, novelty/duplicates, ancestry, audit agreement, annotation time/disagreement, and cost-normalized learning curve.
- **Break & Falsify:** inject correlated generator/judge error, rare modes, early miscalibration, redundant uncertain cases, and temporal shift.
- **Alignment:** Lessons 8.5 and 8.6.
- **Effort Estimate:** 5h.

## 07 Break / Incident Scenarios

### Incident 08.1 — The “Clean” Dataset Release Regresses Production

**Incident Symptoms:**
A new release passes schema checks, removes more duplicates, adds synthetic rare-class examples, and reports low aggregate drift. After training, one language slice regresses, benchmark score rises suspiciously, online labels arrive with changed semantics, and one deletion request is absent from a derived release.

**Diagnostic Protocol (Task):**
1. **Form Competing Hypotheses:** join/unit error, over-deduplication, leakage/contamination, synthetic feedback/judge bias, label-policy change, deletion-lineage gap, concept shift, evaluator change, or unrelated training variance.
2. **Identify Missing Evidence:** manifests, hashes, transform/environment and detector revisions, duplicate clusters, split/entity/time lineage, synthetic ancestry, eligibility/deletion ledger, label guides, slice statistics, and paired outputs.
3. **Design Discriminating Measurements:** replay the release, audit units/joins and labels, rescore old/new outputs with fixed evaluators, inspect removed clusters, and rebuild factorial candidates varying one component.
4. **Rank Explanations:** use matched deltas, temporal ordering, audited labels/matches, repeated training where needed, and uncertainty; permit interacting causes.
5. **Intervene:** quarantine or roll back the smallest supported component, repair lineage/labels/detectors, restore independent anchors, and propagate required deletion.
6. **Remeasure:** repeat byte/count replay, label/slice quality, contamination audit, deletion convergence, downstream utility, and rollback criteria under the same contract.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem — Govern a Multilingual AI Data Platform

Design a data platform for a multilingual assistant with pretraining documents, SFT examples, preferences, evaluation anchors, retrieval content, production feedback, and synthetic augmentation.

**Required Deliverables:**
1. contracts, immutable identities, source authority/consent/license/retention fields, and lineage with deletion propagation;
2. replay proof and a declared gap for every non-replayable dependency;
3. structural, statistical, relational, semantic, policy, and task gates with quarantine/override lifecycle;
4. drift and delayed-label telemetry with competing hypotheses and calibrated alerts;
5. leak-resistant split, deduplication, and contamination audits with false-merge/miss evidence;
6. synthetic ancestry, equal-budget mixture ablation, and correlated generator/judge stress;
7. label guide, qualification, blind assignment, disagreement/adjudication, correction, and active-learning economics;
8. pinned production source trace and generalizability boundary;
9. promotion, canary, rollback, deletion, and incident procedures;
10. an experiment separating data, training variance, model, and evaluator causes.

## 09 Required Evidence & Rubric

### Required Artifact: Versioned Data Release Dossier

Submit manifests and lineage, replay and deletion-convergence output, validation/quarantine records, split/dedup/contamination ledger, label/adjudication log, synthetic/active-learning ancestry, source trace, and promotion/rollback decision.

### Rubric Dimensions

- **Provenance:** *Insufficient* uses names/URIs only. *Competent* identifies content and transformations. *Strong* proves replay, eligibility, deletion propagation, and explicit gaps.
- **Validation:** *Insufficient* reports schema pass. *Competent* maps each layered gate to action. *Strong* measures blind spots, overrides, false decisions, and consumer canaries.
- **Quantitative Discipline:** *Insufficient* quotes drift/dedup scores. *Competent* states units/bins/populations/uncertainty. *Strong* ties signals to delayed outcomes and decision cost.
- **Independence:** *Insufficient* splits rows casually. *Competent* aligns split/fit/dedup boundaries with deployment. *Strong* red-teams semantic relatives and residual risk.
- **Feedback Safety:** *Insufficient* mixes synthetic/feedback data irreversibly. *Competent* preserves ancestry. *Strong* isolates correlated error, tail effects, human-label QA, and removal.
- **Diagnosis:** *Insufficient* blames “bad data.” *Competent* uses diffs/audits/ablations. *Strong* ranks interacting causes and remeasures after a narrow intervention.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Contracts, lineage, correction, deletion | 8.1 | 8.1 Guided Practice (a) canonical-deletion ledger; LAB A | Mastery deliverables 1–2, 9; Incident 08.1 steps 2, 5 (deletion-lineage gap) | LAB A manifest, lineage graph, replay diff, deletion-convergence output |
| Validation and quarantine | 8.2 | 8.2 Guided Practice (a) join computation, (b) defect injection; LAB A | Mastery deliverable 3; Incident 08.1 step 3 (audit units/joins) | LAB A check-to-invariant matrix, quarantine records, false pass/fail audit |
| Pinned validation source trace | 8.2 (source trace note); §05 Production Source Trace | LAB A/B source-trace reading | Mastery deliverable 8 | TFDV `validate_statistics` / `get_drift_skew_dataframe` trace with a generalizability boundary |
| Shift taxonomy and drift alerts | 8.3 | 8.3 Guided Practice (a) concept-shift toy, (b) controlled shifts; LAB B | Mastery deliverable 4; Incident 08.1 steps 1, 3 (concept shift vs label-policy change) | LAB B shift-family matrix with invariants stated, PSI with bin/zero policy, alert cost table on held-out windows |
| Leakage, dedup, contamination | 8.4 | 8.4 Guided Practice; LAB C | Mastery deliverable 5; Incident 08.1 step 3 (inspect removed clusters) | LAB C detector precision/recall on audited pairs, contamination ledger, residual-risk statement |
| Synthetic data | 8.5 | 8.5 Guided Practice; LAB D | Mastery deliverables 6, 10; Incident 08.1 steps 1, 4 | LAB D ancestry records, equal-budget mixture ablation with repeated seeds, rare-slice results |
| Active learning and label systems | 8.6 | 8.6 Guided Practice; LAB D | Mastery deliverable 7 | LAB D cost-normalized holdout curve, label/adjudication log |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can replay a release, break schema-green data, distinguish shift hypotheses, expose leakage beyond exact matches, quantify dedup false merges, isolate synthetic ancestry, operate a reproducible label/adjudication loop, evaluate active learning on a representative holdout, propagate deletion, trace pinned validation code, and defend promotion/rollback.

### Module Wrap-Up (Final Mental Model Reconstruction)

Data engineering is controlled evidence transformation. Preserve identity, authority, eligibility, and ancestry; validate progressively; keep evaluation independent; measure drift against consequences; and never let feedback data erase the distinction between observation and model-generated belief.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
