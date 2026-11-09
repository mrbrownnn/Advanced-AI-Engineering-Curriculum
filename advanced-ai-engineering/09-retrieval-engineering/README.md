# Module 09 — Retrieval Engineering

## 00 Why This Module Exists

Retrieval is not “vector search.” It is a staged decision system that converts a query into a bounded candidate set under corpus, relevance, filter, latency, and capacity constraints. A relevant document can be lost by analysis, representation, approximate search, filtering, fusion, truncation, or reranking; the same missing result can therefore have several competing causes.

$$
\text{query}\to\text{analyze/embed}\to\text{candidate branches}
\to\text{filter/fuse}\to\text{rerank}\to\text{top-}k.
$$

*Notation note.* The letter $k$ is overloaded in retrieval. In this module, $k$ alone (top-$k$, Recall@$k$, nDCG@$k$) is the result-list cutoff; $k_1$ is the BM25 saturation parameter; the RRF paper's constant (also written $k$ there) is renamed $c$ here; and Lucene's "per-leaf $k$" is the number of hits requested from one index segment. Likewise $c$ in Lesson 9.3 (the popped HNSW candidate) is unrelated to the RRF constant $c$ in Lesson 9.4.

This module covers lexical and dense retrieval, exact and approximate nearest neighbors, HNSW, hybrid fusion, reranking, late interaction, metrics, and latency–quality trade-offs. Document ingestion and lineage belong to Module 08. Query rewriting, generation, citations, freshness orchestration, and RAG failure handling belong to Module 10.

**Research cutoff:** 2026-09-27 for the original revision. Sources added in the 2026-09-30 revision (IDF conventions, HNSW Algorithm 2, MaxSim, RRF constants, frontier items) were searched through 2026-09-30, and the pinned Lucene files were re-read on 2026-09-30. Claims not listed as changed in the registry were not re-verified.

**Module Orientation**

- **Engineering problem:** maximize useful relevance under filter correctness, tail latency, memory, update, and capacity constraints.
- **What you will do:** implement lexical and exact-dense baselines; quantify representation and approximation loss separately; tune HNSW under selective filters; build hybrid fusion and bounded reranking; measure judgment and service trade-offs; trace Lucene source; and diagnose a loaded retrieval regression.
- **Environment:** Python 3.10+ for experiment orchestration and analysis, plus a pinned search engine or ANN implementation and a versioned corpus/query/judgment set. Record index hardware, build/search parameters, concurrency, filters, and cache state.
- **Evidence rule:** separate source observations (**O**), derivations (**D**), and telemetry-dependent hypotheses (**H**). Paper gains remain scoped to model, corpus, queries, judgments, index, hardware, and baseline.

## 01 Baseline Assumptions

- Module 00: experimental units, paired comparisons, uncertainty, and causal diagnosis.
- Module 02/04: performance measurement, latency distributions, queueing, throughput, and goodput.
- Module 08: immutable corpus/chunk lineage, validation, contamination, and release rollback.
- Mathematical prerequisites: vectors, dot product, cosine/L2 distance, logarithms, sets, ranks, and empirical distributions.

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
  labs: 17h
  assessment: 3h
  source_trace: 2h
  total: 30h
```

The learner must be able to implement strong lexical/dense baselines, isolate encoder from ANN loss, tune HNSW under filters and load, construct hybrid fusion, rerank bounded candidates, select metrics from a relevance contract, and diagnose offline-to-production regressions.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
query + corpus snapshot + relevance/filter contract
        |                         |
        | lexical analyzer       | query encoder
        v                         v
   BM25 candidate list      vector candidate list
                                  |
                         exact kNN or ANN/HNSW
        \_________________________/
                    |
              identity + fusion
                    |
             reranker/late interaction
                    |
        ranked results + complete stage trace
```

Keep three loss boundaries separate:

1. **Representation/relevance loss:** even exact search ranks the wrong items.
2. **Approximation/filter loss:** ANN differs from exact search under the same vectors and filters.
3. **Selection/system loss:** fusion, reranking, truncation, latency, or capacity changes the delivered list.

## 04 Lessons

### Lesson 9.1 — Lexical Retrieval and BM25

**Engineering Question:**
Which lexical analysis and scoring choices make an exact term retrievable and reproducible?

**Concepts & Definitions:**

- **Analyzer:** the pipeline that turns text into indexed terms, including tokenization, lowercasing, Unicode normalization, stemming, stopwords, and synonyms. Query and index text must use compatible analyzers.
- **Document frequency $n_t$:** the number of documents in the field that contain term $t$. $N$ is the number of documents that have the field.
- **IDF convention:** the formula that turns $n_t$ and $N$ into a weight. Two conventions matter here:
  - Robertson–Zaragoza / RSJ without relevance information: $\ln\frac{N-n_t+0.5}{n_t+0.5}$. This is **zero at $n_t=N/2$ and negative above it**.
  - Lucene at the pinned commit: $\ln\!\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right)$, which is always positive (**O**, CLM-013).
- **Term-frequency saturation:** controlled by $k_1$. **Length normalization:** controlled by $b\in[0,1]$ with document length $|d|$ and average length $avgdl$.

BM25 is a strong, interpretable baseline for exact identifiers, names, rare terms, and lexical intent (**O**, CLM-001). The term contribution used in this module is:

$$
IDF(t)\cdot\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/avgdl)}.
$$

where:

- $t$ is one analyzed query term and $d$ is one document; the document score is the sum of this contribution over the query terms $t$.
- $f(t,d)$ is the term frequency: how many times $t$ occurs in $d$ after analysis.
- $IDF(t)$ is one of the two IDF conventions above (this module states which one each example uses).
- $|d|$ is the document length in analyzed tokens, and $avgdl$ is the mean $|d|$ over the $N$ documents in the field.
- $k_1\ge0$ and $b\in[0,1]$ are the tuning constants ($k_1$ is not the top-$k$ cutoff).

*Step-by-step reading of the formula.*

1. Write $K=k_1(1-b+b\,|d|/avgdl)$, the "length-adjusted saturation constant". The contribution becomes $IDF(t)\cdot\frac{f(k_1+1)}{f+K}$.
2. *Length normalization.* If $b=0$, then $K=k_1$ for every document (no length effect). If $b=1$, then $K=k_1|d|/avgdl$ (full proportional penalty). An average-length document ($|d|=avgdl$) gets $K=k_1$ for any $b$; longer documents get a larger $K$ and therefore a smaller contribution.
3. *Saturation.* For fixed $K$, $\frac{f(k_1+1)}{f+K}$ is $0$ at $f=0$, increases with $f$, and approaches $k_1+1$ as $f\to\infty$. So repeated occurrences help with diminishing returns; with $k_1=0$ the term part is $1$ for any $f\ge1$ (pure presence/absence).
4. *Why the RSJ IDF can be zero or negative.* $\ln x=0$ when $x=1$, i.e. $N-n_t+0.5=n_t+0.5$, i.e. $n_t=N/2$. For $n_t>N/2$ the ratio is below 1 and the logarithm is negative. Lucene adds 1 inside the logarithm, so its argument is always greater than 1 and its IDF is always positive.
5. *Why dropping $(k_1+1)$ preserves ranking.* With fixed $k_1$, every term contribution of every document is multiplied by the same positive constant $k_1+1$, so all document scores are scaled by the same factor and their order is unchanged; only the absolute values differ.

Small check with this module's constants ($k_1=1.2$, $b=0.75$) at $|d|=avgdl$, so $K=1.2$: $f=1$ gives $2.2/2.2=1.0$, $f=2$ gives $4.4/3.2=1.375$, $f=10$ gives $22/11.2=1.964$, all below the ceiling $k_1+1=2.2$.

At the pinned commit, Lucene's scorer computes $\text{boost}\cdot IDF\cdot \frac{f}{f+k_1(1-b+b\,|d|/avgdl)}$, without the $(k_1+1)$ factor. That constant does not change rankings for a fixed $k_1$, but it changes absolute scores. Lucene also stores $|d|$ in a lossy one-byte norm, so its scores can differ slightly from a hand calculation (**O**, CLM-013).

**Mechanism Explanation:**

The formula is incomplete without analyzer, token positions/overlaps, fields/boosts, corpus statistics, query construction, $k_1$, and $b$. Stemming, stopwords, Unicode normalization, synonyms, and index/query analyzer mismatch can dominate tuning.

**Break cases:** product codes split incorrectly; language-specific tokenization; a corpus refresh changes IDF; synonyms expand only one side; long templated documents receive unintended length penalties.

**Worked Example (synthetic four-document fixture):**
*Input.* The query is `xj-200 firmware`, with $k_1=1.2$ and $b=0.75$, natural logarithms, and the $(k_1+1)$ form above.

| doc | text |
|---|---|
| d1 | reset router xj-200 firmware |
| d2 | router firmware update guide for home router |
| d3 | xj-200 warranty |
| d4 | printer 200 dpi setup guide |

*Steps with analyzer K (lowercase, split on whitespace only).* Lengths are $(4,7,2,5)$, so $avgdl=4.5$. The query terms are `xj-200` ($n=2$) and `firmware` ($n=2$). Lucene IDF is $\ln(1+2.5/2.5)=\ln2=0.6931$ for both. For d1: norm $=1.2(0.25+0.75\cdot4/4.5)=1.1$, tf part $=1\cdot2.2/(1+1.1)=1.0476$, so each term contributes $0.7262$ and the total is $1.4523$. Likewise d3 $=0.8970$ (norm 0.70), d2 $=0.5648$ (norm 1.70), and d4 $=0$.
Here "norm" is $K=k_1(1-b+b|d|/avgdl)$ and $1-b=0.25$. Expanded: d3 has only `xj-200` with $f=1$, norm $=1.2(0.25+0.75\cdot2/4.5)=0.70$, tf part $=2.2/1.70=1.2941$, score $=0.6931\cdot1.2941=0.8970$. d2 has only `firmware` with $f=1$, norm $=1.2(0.25+0.75\cdot7/4.5)=1.70$, tf part $=2.2/2.70=0.8148$, score $=0.6931\cdot0.8148=0.5648$. d4 contains neither query term, so it scores 0. The shorter d3 beats the longer d2 because both match one term with the same IDF, and only the length normalization differs.
Ranking: **d1 > d3 > d2 > d4**.

*Steps with analyzer S (also split on `-`).* Now `xj-200` becomes `xj` and `200`. Lengths become $(5,7,3,5)$, so $avgdl=5.0$. The term `200` now also matches d4 ("200 dpi"), so $n_{200}=3$ and Lucene IDF $=\ln(1+1.5/3.5)=0.3567$. Totals: d1 $=1.7430$, d3 $=1.2552$, d2 $=0.5957$, d4 $=0.3567$.
d4 now scores through a coincidental match on `200`.

*Same analyzer S, RSJ IDF instead of Lucene's.* With $N=4$, $n=2$ gives $\ln(2.5/2.5)=0$ and $n=3$ gives $\ln(1.5/3.5)=-0.8473$. Matching `200` now **lowers** a score. Totals: d2 $=0$, d1 $=-0.8473$, d4 $=-0.8473$, d3 $=-1.0131$.
Ranking: **d2 first**, the only document that does *not* contain the product code.

*Result.* The analyzer changed the term inventory, $avgdl$, and the candidate set. The IDF convention reversed the ranking even with $k_1$ and $b$ held fixed.

*Interpretation and limits.* $N=4$ exaggerates the zero and negative RSJ weights, but $n_t>N/2$ is realistic for common tokens in large corpora. The point stands: a "BM25 score" is reproducible only together with the analyzer, the IDF convention, the field statistics, and the engine revision.

**Knowledge Check:** Why is $k_1,b$ tuning unable to repair a term removed by analysis?

**Guided Practice:** (a) With analyzer K and Lucene IDF, add document d5 = "xj-200 xj-200 manual" and recompute $n$, $avgdl$, IDF, and d1's score. (b) Reproduce one score from a real pinned index (term frequency, stored norm, corpus statistics, fields, analyzer output), then change one factor.

**Feedback Contract:**
- *Expected Evidence*: (a) $N=5$ and $avgdl=(4+7+2+5+3)/5=4.2$. $n_{xj\text{-}200}=3$, so IDF $=\ln(1+2.5/3.5)=0.5390$; $n_{firmware}=2$, so IDF $=\ln(1+3.5/2.5)=0.8755$. d1 norm $=1.2(0.25+0.75\cdot4/4.2)=1.1571$, tf part $=2.2/2.1571=1.0199$, so d1 $=1.0199(0.5390+0.8755)=1.4426$. (b) A pinned analyzer/index manifest and a score decomposition that matches the engine's explanation output within its norm quantization.
- *Common Failure*: Comparing BM25 parameters across different analyzers, mixing the RSJ and Lucene IDF forms, or ignoring Lucene's missing $(k_1+1)$ factor when matching absolute scores.
- *Diagnostic Hint*: Print the analyzed tokens for the query and for the missing document before touching $k_1$ or $b$.
- *Concept to Revisit*: IDF Convention; Analyzer as Semantic Input.

**Learning Outcome:** Reproduce scores from the pinned index contract and explain which input changed.

*(Effort: 50m instruction, 30m practice)*

### Lesson 9.2 — Dense Dual Encoders and Similarity Semantics

**Engineering Question:**
Did the representation fail, or did approximate search lose a useful vector result?

**Concepts & Definitions:**

Dense retrieval encodes query $q$ and document $d$ separately, then scores inner product, cosine, or distance. DPR establishes this mechanism for open-domain QA, not universal superiority over BM25 (**O**, CLM-002). Training positives/negatives, encoder version, pooling, normalization, dimension, chunking, domain, and similarity must match index and query paths.

**Mechanism Explanation:**

Cosine and inner product are equivalent in ranking only under compatible unit normalization. Under unit vectors, $\|q-d\|^2=2-2\,q\cdot d$, so L2 ranking is the reverse of inner-product ranking; without normalization, no such identity holds. Do not switch metrics by name.

where $q,d\in\mathbb{R}^m$ are the query and document embeddings ($m$ is the embedding dimension), $q\cdot d=\sum_i q_i d_i$ is the inner product, $\|x\|=\sqrt{x\cdot x}$ is the L2 norm, and $\cos(q,d)=\frac{q\cdot d}{\|q\|\,\|d\|}$.

*Step-by-step.*

1. Expand the squared distance: $\|q-d\|^2=(q-d)\cdot(q-d)=\|q\|^2+\|d\|^2-2\,q\cdot d$.
2. With unit vectors, $\|q\|=\|d\|=1$, so $\|q-d\|^2=2-2\,q\cdot d$, and also $\cos(q,d)=q\cdot d$.
3. $2-2x$ decreases as $x$ increases, and $\sqrt{\cdot}$ is increasing, so the smallest L2 distance is the largest inner product, which is the largest cosine: all three rankings coincide.
4. Without normalization, step 1 keeps the term $\|d\|^2$, which varies by document. Inner product then rewards large $\|d\|$, L2 penalizes it, and cosine ignores it, so the three rankings can disagree (see the example check in the Feedback Contract below).
5. For a fixed query, $\|q\|$ is the same for every document, so normalizing only the query never changes an inner-product or L2 ranking; what matters is whether the *document* vectors are normalized. First run exact vector search on a representative subset: if exact search misses relevance, increasing ANN effort cannot repair the encoder.

**Worked Example:**
Run exact search over the same normalized vectors and metric. If the relevant item is absent from exact top-$k$, raising HNSW search effort cannot repair that representation ranking.

**Knowledge Check:** Under which normalization condition do cosine and inner-product rankings coincide?

**Guided Practice:** Cross normalized/unnormalized embeddings with cosine, inner product, and L2; record ranking changes and invalid combinations.

**Feedback Contract:**
- *Expected Evidence*: A pinned encoder/tokenizer/pooling/metric manifest plus exact ranks. Example check: query $(1,0)$, documents $a=(2,2)$ and $b=(0.9,0.1)$. Inner product ranks $a$ first ($2$ vs $0.9$). Cosine ranks $b$ first ($0.994$ vs $0.707$), and so does L2 ($0.141$ vs $2.236$). After unit-normalizing both documents, all three metrics rank $b$ first. Step-by-step: $\|a\|=\sqrt{8}=2.828$ and $\|b\|=\sqrt{0.82}=0.906$. Cosines are $2/2.828=0.707$ and $0.9/0.906=0.994$. L2 distances are $\sqrt{(2-1)^2+2^2}=\sqrt5=2.236$ and $\sqrt{0.1^2+0.1^2}=0.141$. Unit-normalized, $\hat a=(0.707,0.707)$ and $\hat b=(0.994,0.110)$; inner products equal the cosines ($0.707$, $0.994$), and $\|q-\hat d\|^2=2-2\,q\cdot\hat d$ gives $0.586$ and $0.012$ (distances $0.765$ and $0.111$), so $b$ wins under every metric.
- *Common Failure*: Blaming ANN before exact comparison, or indexing normalized vectors while querying with unnormalized ones.
- *Diagnostic Hint*: Is the missing item in the *exact* top-$k$ under the served metric? If not, the problem is upstream of the index.
- *Concept to Revisit*: Representation vs Approximation Loss.

**Learning Outcome:** Distinguish semantic representation failure from index approximation.

*(Effort: 40m instruction, 25m practice)*

### Lesson 9.3 — Exact kNN, HNSW, and Filtered ANN

**Engineering Question:**
How do approximation, filters, updates, and graph-search budgets change delivered candidates?

**Concepts & Definitions:**

HNSW builds randomized hierarchical proximity graphs and searches from sparse upper layers toward a denser base (**O**, CLM-003). Construction connectivity/effort, search effort, vector distribution, distance, deletion/update policy, memory layout, and implementation produce a recall–latency–memory surface.

The paper's layer search, Algorithm 2 `SEARCH-LAYER(q, ep, ef, l)`, keeps three structures (**O**, CLM-015):

- **visited $v$:** every node whose distance has been computed;
- **candidates $C$:** a frontier of nodes still to expand, nearest first;
- **results $W$:** the best $ef$ nodes found so far.

At each step it pops the nearest candidate $c$. If $c$ is farther than the farthest node in $W$, it **stops**. Otherwise it scores each unvisited neighbor $e$ and adds $e$ to both $C$ and $W$ if $e$ is closer than the farthest node in $W$ or $|W|<ef$, trimming $W$ back to $ef$. Upper layers run with $ef=1$ to find an entry point, and the base layer runs with the requested $ef$.

**Quantitative Model / Derivation:**

For exact top-$k$ set $E_k$ and approximate set $A_k$ under identical vectors, metric, filter, snapshot, and ties:

$$
Recall^{ANN}@k=\frac{|A_k\cap E_k|}{|E_k|}.
$$

where $E_k$ is the set of $k$ items returned by exact (brute-force) search, $A_k$ is the set of $k$ items returned by the ANN index for the same query, and $|\cdot|$ counts set elements. Normally $|E_k|=k$, so the value is "how many of the true $k$ nearest neighbors the ANN index found, divided by $k$"; it ranges from 0 to 1, and order inside the top-$k$ is ignored. The search effort $ef$ in the algorithm above is a separate knob: $ef\ge k$ is needed to return $k$ results from the base layer.

This measures approximation, not relevance (**D**, CLM-004). A system can have perfect ANN recall and poor retrieval relevance.

Filters complicate search. Prefiltering, integrated graph filtering, oversampling then filtering, and exact fallback behave differently with selectivity and vector/filter correlation (**H**, CLM-012). Lucene at the pinned commit illustrates one implementation (**O**, CLM-014):

- Filtered-out nodes are still expanded as graph *candidates* but are not *collected* as results.
- Per segment, if the filter admits at most the per-leaf $k$ documents, Lucene runs exact search directly.
- Otherwise it runs approximate search with a visit limit of (filter cost + 1). If that limit is hit, or too few results come back, it falls back to exact search.

Selective filters can therefore turn an ANN query into an exact scan, with a latency cost that depends on the filter. Measure exact filtered ground truth, visited nodes/candidates, fallback, tail latency, and concurrency. ACL correctness is a hard invariant, not a recall trade.

Index lifecycle is part of correctness: insert, update, delete/tombstone, segment merge or graph rebuild, replica visibility, and rollback can temporarily diverge. Track query-visible revision and resource cost; “write accepted” does not establish that every serving replica uses the same candidate universe.

**Worked Example (synthetic single-layer HNSW trace):**
*Input.* Seven 2-D points: A(0,0), B(2,1), C(4,0), D(5,3), E(2,4), F(6,5), G(7,1). Undirected edges: A–B, A–E, B–C, B–E, C–G, D–E, D–F, D–G. The query is $q=(4,2)$, the entry point is A, and the distance is Euclidean. Distances to $q$: D 1.414, C 2.000, B 2.236, E 2.828, G 3.162, F 3.606, A 4.472. Exact top-2 is {D, C}.

*Trace with $ef=1$.*

| step | pop $c$ | neighbors scored | $C$ after | $W$ after | visited |
|---|---|---|---|---|---|
| 0 | — | — | {A} | {A} | {A} |
| 1 | A | B, E | {B} | {B} | A, B, E |
| 2 | B | C | {C} | {C} | A, B, C, E |
| 3 | C | G (3.162 > 2.000, not added) | ∅ | {C} | A, B, C, E, G |

$C$ is empty, so the search ends with {C}, a local minimum. D is reachable only through E or G, and both were rejected because each was farther than the current best.

*Trace with $ef=2$.* $W$ becomes {B, E} after expanding A, then {C, B} after expanding B, and C adds nothing. The next pop is E (2.828), which is farther than the farthest node in $W$ (B, 2.236), so the search **stops** with {C, B}.

*Trace with $ef=3$.* E stays in $W$ long enough to be expanded, which reaches D. The result is {D, C, B} after 7 distance evaluations (all nodes).

*Result.* ANN recall@1 is $0/1$ at $ef=1$. ANN recall@2 is $|\{C,B\}\cap\{D,C\}|/2=0.5$ at $ef=2$ and $1.0$ at $ef=3$. The distance evaluations go 5, 5, 7.

*Interpretation and limits.* A larger $ef$ bought recall with extra work. The failure mode is local minima in a greedy graph walk, not randomness. This is a hand-built graph; real HNSW adds layers, degree limits, and neighbor-selection heuristics, and Lucene orders by similarity score rather than distance.

A second, set-level example: if exact filtered top-10 contains ten eligible targets and ANN returns eight of those plus two other eligible items, ANN recall@10 is 0.8. That says nothing about relevance recall unless the exact set is itself judged relevant. Returning an unauthorized item is a correctness failure regardless of 0.8.

**Knowledge Check:** Why must filter, vectors, metric, snapshot, and ties match when computing ANN recall?

**Guided Practice:** (a) Rerun the trace above with query $q=(6,4)$ and $ef=1$ from entry point A. (b) Sweep search effort and filter selectivity/correlation, then inject update/delete and replica lag. Compare against exact filtered ground truth.

**Feedback Contract:**
- *Expected Evidence*: (a) Distances: D 1.414, F 1.000, G 3.162, E 4.000, C 4.472, B 5.000, A 7.211. The walk is A → E (4.000 beats B 5.000) → D (1.414) → F (1.000), ending at F, the exact nearest. Greedy search succeeds here, so one success does not prove $ef=1$ is safe. (b) Exact/ANN sets, visited work, tail latency, memory, revision, fallback count, and zero ACL violations.
- *Common Failure*: Calling ANN recall "relevance recall", or forgetting that a node rejected from $W$ is also never expanded.
- *Diagnostic Hint*: When ANN misses an exact neighbor, find the graph path to it and ask which node on that path was rejected, and why.
- *Concept to Revisit*: Frontier, Visited Set, and Stop Rule; Approximation vs Relevance.

**Learning Outcome:** Tune ANN against exact search and break it with selective, correlated filters and lifecycle churn.

*(Effort: 65m instruction, 40m practice)*

### Lesson 9.4 — Hybrid Retrieval and Rank Fusion

**Engineering Question:**
Do retrieval branches add complementary opportunity, and does fusion preserve it?

**Concepts & Definitions:**

Lexical and dense branches can be complementary: one captures explicit terms, the other learned semantic similarity. They can also be redundant. Normalize document identity before union so aliases/chunks do not fragment votes.

**Quantitative Model / Derivation:**

Reciprocal rank fusion uses:

$$
RRF(d)=\sum_r\frac{w_r}{c+rank_r(d)}.
$$

where:

- $d$ is a candidate document (after identity normalization) and $r$ ranges over the input rankings (here, the lexical and dense branches).
- $rank_r(d)\in\{1,2,\dots\}$ is $d$'s 1-based position in ranking $r$; if $d$ is absent from ranking $r$, that term is taken as 0 in this module.
- $w_r\ge0$ is the weight of ranking $r$, and $c\ge0$ is the smoothing constant.

*Step-by-step reading.* Each ranking contributes $w_r/(c+rank)$, which is largest at rank 1 and decays slowly. The gap between ranks 1 and 2 is $\frac{1}{c+1}-\frac{1}{c+2}=\frac{1}{(c+1)(c+2)}$: with $c=1$ that is $1/2-1/3=0.167$, with $c=60$ it is only $1/3782=0.00026$. A large $c$ therefore flattens rank differences, so *how many* rankings contain $d$ matters more than *where*. Only ranks enter the formula, never raw scores, which is why branch score scales do not need to be comparable.

In the original paper, $w_r=1$, $c=60$ (the paper calls it $k$) was fixed in a pilot, and each input ranking is a permutation over all documents (**O**, CLM-005). The weights, a document absent from a truncated list (contributing 0 in this module), and duplicate-identity rules are engineering extensions that must be declared. RRF avoids treating incomparable raw score scales as comparable. Fusion cannot recover an item missing from all branches. Compare each branch, union oracle opportunity, fusion, and latency at equal budgets.

**Worked Example (synthetic rankings):**
*Input.* Lexical top-5 = (d3, d1, d7, d2, d9) and dense top-5 = (d2, d5, d1, d8, d3). Weights are 1, $c=60$, a missing document contributes 0, and exact score ties are broken by ascending document ID.

*Steps.*
- d2: $1/(60+4)+1/(60+1)=0.015625+0.016393=0.032018$
- d1: $1/62+1/63=0.016129+0.015873=0.032002$
- d3: $1/61+1/65=0.016393+0.015385=0.031778$
- d5: $1/62=0.016129$
- d7: $1/63=0.015873$
- d8: $1/64=0.015625$
- d9: $1/65=0.015385$

*Result.* Fused order: **d2, d1, d3, d5, d7, d8, d9**. The top three are separated by less than $2.5\times10^{-4}$. With $c=1$ the order becomes d2 (0.700), d3 (0.667), d1 (0.583), and so on, so $c$ decides d1 vs d3.

*Interpretation and limits.* With $c=60$, appearing in both lists dominates rank position within the top 5: every document in both lists outranks every single-list document. Check: the weakest possible two-list score in two top-5 lists is $2/(60+5)=0.0308$, and the strongest single-list score is $1/(60+1)=0.0164$. The union has 7 distinct documents; if the relevant ones are d4 and d6, no fusion setting can find them. That is branch opportunity loss, not fusion loss.

**Knowledge Check:** Can fusion recover an item absent from every branch?

**Guided Practice:** (a) Recompute the fused top 3 above with lexical weight 2 and dense weight 1 at $c=60$. (b) Compare branch results, normalized-identity union opportunity, RRF, and a single-branch baseline at matched depth and latency budget.

**Feedback Contract:**
- *Expected Evidence*: (a) d1 $=2/62+1/63=0.048131$; d3 $=2/61+1/65=0.048172$; d2 $=2/64+1/61=0.047643$; d7 $=2/63=0.031746$. The order is d3, d1, d2, so a weight change reorders the top 3. (b) Branch lineage, duplicate policy, union oracle, fused ranks, and critical-path/total work.
- *Common Failure*: Adding incomparable raw scores, or letting chunk IDs of the same document split its votes.
- *Diagnostic Hint*: Compute the union oracle first. If the relevant document is not in the union, stop tuning fusion.
- *Concept to Revisit*: Branch Opportunity vs Fusion Loss.

**Learning Outcome:** Prove complementarity rather than assume “hybrid is better.”

*(Effort: 45m instruction, 25m practice)*

### Lesson 9.5 — Reranking and Late Interaction

**Engineering Question:**
How much candidate opportunity should be purchased before reranking cost and truncation dominate?

**Concepts & Definitions:**

- **Cross-encoder reranker:** encodes the query and one candidate *together* in a single forward pass and outputs a score, so every query–candidate pair needs a new forward pass. It can reorder only the candidate pool it receives, and its cost grows with candidate count and sequence shapes (**D**, CLM-006). Truncation can remove the only relevant span.
- **Late interaction (ColBERT):** encodes the query and the document *separately* into one contextual vector per token, then scores with MaxSim. Document vectors can therefore be precomputed offline (**O**, CLM-007). It is **not** a cross-encoder, because no query token attends to document tokens inside the encoder.
- **MaxSim:** $S(q,d)=\sum_{i\in q}\max_{j\in d} E_{q_i}\cdot E_{d_j}$. ColBERT L2-normalizes the token embeddings, so each dot product is a cosine, and it pads queries with `[mask]` tokens to a fixed length (**O**, CLM-016). Here $i$ indexes query tokens, $j$ indexes document tokens, and $E_{q_i}$, $E_{d_j}$ are their output token embeddings. Read it in two steps: (1) for each query token, find the single document token most similar to it (the inner $\max_j$); (2) add those best matches over all query tokens (the outer $\sum_i$). With unit vectors each term is at most 1, so the score is at most the number of query tokens.
- **Single-vector dual encoder:** one pooled vector per side; for example, cosine of mean-pooled normalized vectors.

**Mechanism Explanation:**

Late interaction occupies a different storage/compute point from single-vector retrieval and cross-encoder reranking: one vector per document token instead of one per document. Paper speedups do not transfer without index, hardware, and corpus context.

For each depth, record candidate recall before reranking, reranker nDCG/MRR after reranking, truncation, model/batch time, queue time, and total work. If candidate opportunity saturates early but reranking cost rises, deeper is not automatically better.

**Worked Example 1 (synthetic 2-D token vectors, all unit-normalized):**
*Input.* Query tokens $q_1=(1,0)$ and $q_2=(0,1)$. Document d1 tokens: $(0.995,0.0995)$, $(0.196,0.981)$, $(0.707,0.707)$. Document d2 tokens: $(0.707,0.707)$, $(0.743,0.669)$.

*Steps.* For d1, $\max_j q_1\cdot d_j=0.995$ and $\max_j q_2\cdot d_j=0.981$, so MaxSim $=1.976$. For d2, the maxima are $0.743$ and $0.707$, so MaxSim $=1.450$. For contrast, mean-pooling each side and normalizing gives single-vector cosines of $0.9995$ for d1 and $0.9997$ for d2. (Since $q_k=(1,0)$ or $(0,1)$, each dot product $q_1\cdot d_j$ is just the first coordinate of $d_j$ and $q_2\cdot d_j$ the second. The pooled query is $(0.5,0.5)$; the pooled d1 is $(0.633,0.596)$ and the pooled d2 is $(0.725,0.688)$, and the cosine of each with $(0.5,0.5)$ gives the two values above.)

*Result.* The pooled single-vector score slightly prefers d2. MaxSim clearly prefers d1, which has a token matching each query token, while d2 has only "in-between" tokens.

*Interpretation and limits.* MaxSim keeps per-token evidence that pooling averages away, at the cost of storing 3 + 2 vectors instead of 2. A cross-encoder would instead run a forward pass over the concatenated query and d1, then again for d2. Its score cannot be decomposed this way, and it cannot be precomputed.

**Worked Example 2 (depth frontier):**
If candidate recall saturates by depth 50 but P99 rerank latency rises through depth 200 with no judged gain, the deeper setting is dominated for that workload. A different corpus or batch shape requires remeasurement.

**Knowledge Check:** Why can a perfect reranker not recover a relevant document missing from its candidate pool? Which step of ColBERT scoring can be done before the query arrives, and why can a cross-encoder not do the same?

**Guided Practice:** (a) Add a d3 with tokens $(0,1)$ and $(0,1)$ to Worked Example 1 and compute its MaxSim. (b) Sweep candidate depth, truncation, model, and batch size; record pre-rerank opportunity, post-rerank quality, queue/compute time, and memory.

**Feedback Contract:**
- *Expected Evidence*: (a) $\max q_1\cdot d_j=0$ and $\max q_2\cdot d_j=1$, so MaxSim $=1.0$. A perfect match on one query token cannot compensate for no match on the other. (b) A depth frontier with identical queries and candidates and full cost.
- *Common Failure*: Reporting reranker-only latency, or describing ColBERT as a "cross-encoder" because it is "token-level".
- *Diagnostic Hint*: Ask where query–document interaction happens: inside the encoder (cross-encoder) or after separate encoding (late interaction).
- *Concept to Revisit*: Interaction Point and Precomputability.

**Learning Outcome:** Select candidate and rerank depth jointly under capacity and latency constraints.

*(Effort: 45m instruction, 25m practice)*

### Lesson 9.6 — Metrics, Judgments, and the Quality–Latency Frontier

**Engineering Question:**
Which relevance and service measurements support a production retrieval decision?

**Concepts & Definitions:**

- $Recall@k=|Rel\cap Top_k|/|Rel|$ measures known relevant-set coverage.
- $RR=1/r$ uses the first relevant rank $r$ ($RR=0$ if no relevant item is retrieved); MRR averages queries.
- $DCG@k=\sum_{i=1}^k g(rel_i)/\log_2(i+1)$; nDCG divides by ideal DCG computed from **all** judged items for the query, not only the retrieved ones (**D**, CLM-009).

where:

- $Rel$ is the set of judged-relevant documents for the query and $Top_k$ is the set of the system's first $k$ results.
- $r$ is the 1-based rank of the first relevant result; over a query set $Q$, $MRR=\frac{1}{|Q|}\sum_{q\in Q}RR_q$.
- $i$ is a rank position, $rel_i$ is the judged grade of the document at rank $i$, and $g(\cdot)$ maps a grade to a gain.
- $IDCG@k$ is the DCG@$k$ of the ideal list: all judged documents for the query sorted by grade, highest first, then cut at $k$. $nDCG@k=DCG@k/IDCG@k$, so $0\le nDCG@k\le1$.

*Step-by-step for DCG.* Rank 1 has discount $\log_2 2=1$, rank 2 has $\log_2 3=1.585$, rank 3 has $\log_2 4=2$, rank 4 has $\log_2 5=2.322$, rank 5 has $\log_2 6=2.585$. Each gain is divided by its rank's discount and the results are summed; dividing by IDCG makes queries with different numbers of relevant documents comparable.

*Conventions used in this module:* the gain is linear, $g(rel)=rel$, and the discount is $\log_2(i+1)$. **Unjudged** documents get gain 0, and the unjudged count in the top $k$ is reported alongside the metric. **Zero IDCG** (no relevant judgment) makes nDCG undefined: such queries are excluded from the nDCG and MRR means, and their count is reported. **Ties** in system score are broken by ascending document ID before any metric is computed. Ties inside the ideal ordering do not change IDCG.

**Mechanism Explanation:**

Declare binary/graded relevance, cutoff, gain, discount, query unit, ties, duplicates, zero-relevant queries, and unjudged policy. Incomplete pools can favor systems similar to the pooling methods. BEIR's heterogeneous results are evidence against a universal retriever ranking (**O**, CLM-008).

End-to-end latency includes query processing/embedding, branch searches, filters, fusion, fetch, reranking, network, and queueing (**D**, CLM-010). Parallel branch latency follows the join critical path; both branches consume capacity. Report per-stage and end-to-end distributions, quality under load, timeout/cancellation, and SLO-goodput.

**Worked Example 1 (synthetic judgments, three queries, $k=5$):**
*Input.*
- q1: judgments are d2=3, d5=2, d9=1, d4=0, d1=0; d7 is unjudged. The system returns (d4, d2, d7, d9, d1).
- q2: judgments are d3=2, d8=0, d6=0. The system returns (d8, d6, d3).
- q3: no document is judged relevant.

*Steps for q1.* Gains are $(0,3,0,1,0)$, with d7 treated as 0. $DCG=3/\log_2 3+1/\log_2 5=1.8928+0.4307=2.3235$. The ideal gains are $(3,2,1,0,0)$, so $IDCG=3+2/\log_2 3+1/\log_2 4=3+1.2619+0.5=4.7619$ and $nDCG=0.4879$. The first relevant document is at rank 2, so $RR=0.5$. Recall@5 counting $rel\ge1$ is $2/3$. The unjudged count is 1.

*Steps for q2.* $DCG=2/\log_2 4=1.0$ and $IDCG=2$, so $nDCG=0.5$ and $RR=1/3$. (Here $k=5$ but only 3 results are returned, so ranks 4–5 contribute 0; the ideal list is $(2,0,0)$, so $IDCG=2/\log_2 2=2$. The only relevant document, d3, is at rank 3.)

*Steps for q3.* $IDCG=0$, so nDCG is undefined; q3 is excluded and reported as "1 of 3 queries has no relevant judgment".

*Result.* Mean nDCG@5 is $0.494$ and MRR is $0.417$ over 2 scored queries, with 1 excluded: $(0.4879+0.5)/2=0.494$ and $(0.5+1/3)/2=0.417$. Had q3 been scored as 0, the means would be $0.329$ and $0.278$ (the same sums divided by 3). The policy moves the headline by about a third.

*Sensitivity to the unjudged document.* If d7 were actually relevant with grade 2, q1's gains become $(0,3,2,1,0)$ and IDCG uses $(3,2,2,1,0)$, giving $nDCG=0.584$ instead of $0.488$ ($DCG=3/1.585+2/2+1/2.322=3.3235$ and $IDCG=3+2/1.585+2/2+1/2.322=5.6926$). A single unjudged document in the top 5 is worth reporting.

*Interpretation and limits.* The metric definition includes the zero-IDCG, unjudged, and tie rules; change any of them and the headline changes. Comparisons between systems must use identical rules and query sets, and paired uncertainty across queries.

**Worked Example 2 (quality under load):**
Two systems can have the same nDCG while one times out on selective-filter queries. Count timed-out, cancelled, and incomplete requests in the declared offered/admitted population rather than computing quality only on successes.

**Knowledge Check:** Why can incomplete judgment pools favor systems similar to the pool constructors? Why must IDCG use all judged items rather than only retrieved ones?

**Guided Practice:** (a) For q2, the system instead returns (d3, d8, d6). Compute nDCG@3 and RR, then the two-query means with q1 unchanged. (b) Evaluate a stable query set at increasing open-loop load. Join judgments, stage traces, timeouts, and resource use; report quality, latency, and goodput by slice.

**Feedback Contract:**
- *Expected Evidence*: (a) $DCG=IDCG=2$, so $nDCG=1.0$ and $RR=1$. The means become nDCG $=(0.4879+1)/2=0.744$ and MRR $=(0.5+1)/2=0.75$, with q3 still excluded and reported. (b) A pinned judgment policy, unjudged handling, uncertainty, stage and end-to-end tails, and complete denominators.
- *Common Failure*: Computing IDCG from the retrieved list only (which makes q1's nDCG look higher), silently dropping zero-relevant queries, or selecting on one average metric.
- *Diagnostic Hint*: Check whether your nDCG can exceed 1 or equals 1 for a list that misses relevant items. Either means IDCG was built from the wrong set.
- *Concept to Revisit*: Metric Conventions; Denominator Discipline.

**Learning Outcome:** Choose a Pareto point rather than a metric-only winner.

*(Effort: 55m instruction, 35m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — Robertson and Zaragoza (2009). Reading pointers: Eq. 3.2–3.3 (RSJ weight and the IDF form without relevance information) and §3.4, Eq. 3.12–3.13 (length normalization and saturation).
- [Efficient and robust approximate nearest neighbor search using HNSW graphs](https://arxiv.org/abs/1603.09320) — Malkov and Yashunin (2018). Reading pointers: Algorithm 2 (SEARCH-LAYER) and Algorithm 5 (K-NN-SEARCH).
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) — Karpukhin et al. (EMNLP 2020).
- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) — Cormack, Clarke, and Büttcher (SIGIR 2009). Reading pointer: §1, the RRFscore formula with $k=60$ ([author-hosted copy](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf), read 2026-09-30).
- [ColBERT](https://arxiv.org/abs/2004.12832) — Khattab and Zaharia (SIGIR 2020). Reading pointers: §3.1 (late interaction as a sum of MaxSim) and §3.2 (query `[mask]` augmentation, L2-normalized embeddings).
- [BEIR](https://arxiv.org/abs/2104.08663) — Thakur et al. (2021).

**CURRENT DEFAULT** (the module's *recommended baseline*, derived from CLM-001, CLM-004, CLM-006, and CLM-010; not a measured survey of industry adoption or of any one engine's defaults): a measured BM25 baseline, embedding-version pinning, ANN-versus-exact audit, hybrid candidate lineage where justified, bounded reranking, slice metrics, and end-to-end timing.

**WORKLOAD-DEPENDENT:** analyzer, dense model, distance, HNSW parameters, filter strategy, fusion method, reranker, depths, and metric weights.

**FRONTIER / NEWER WORK** (none is a universal replacement for workload-grounded baselines):

- Multi-vector compression: [ColBERTv2](https://arxiv.org/abs/2112.01488) — Santhanam et al. (NAACL 2022; arXiv v3 July 2022). The authors report that residual compression substantially reduces the late-interaction index footprint while improving quality on their benchmarks. Author-reported, abstract-level reading; the ratio is omitted here because its hardware, corpus, and baseline context was not read (CLM-017).
- Reasoning-intensive retrieval benchmarks: [BRIGHT](https://arxiv.org/abs/2407.12883) — Su et al. (arXiv v1 July 2024, v4 March 2025). 1,384 real-world queries where keyword or semantic matching is usually insufficient. Abstract-level reading (CLM-018).
- Retrieval agents: [Search-R1](https://arxiv.org/abs/2503.09516) — Jin et al. (arXiv v1 March 2025, v5 August 2025). Trains an LLM with reinforcement learning to issue search queries during step-by-step reasoning. Abstract-level reading; agent orchestration belongs to Modules 10 and 12 (CLM-019).
- `TODO_VERIFY` (CLM-020): learned sparse retrieval and LLM-based reranking or query representations were **not** opened from primary sources in this revision. No specific result is claimed.

**LEGACY / INSUFFICIENT:** dense-only by default; ANN recall called retrieval recall; a single average metric; unpinned embeddings; raw lexical/vector score addition without calibration; microbenchmark latency used as service latency.

**PRODUCTION SOURCE TRACE**

- Repository: `apache/lucene`
- Revision: `e357029271b3de560a6ff13c9cab4d4c8103b53b`
- Verified: first read 2026-09-26; the same revision was re-read on 2026-09-30. Static inspection only, nothing was executed.
- Files/symbols (**O**, CLM-011): `BM25Similarity.idf/scorer`, `KnnFloatVectorQuery.approximateSearch`, and `HnswGraphSearcher.search/searchLevel` in the registry-recorded paths.
- Observed on 2026-09-30:
  - `BM25Similarity.idf` returns `log(1 + (docCount - docFreq + 0.5)/(docFreq + 0.5))`, and `BM25Scorer.doScore` computes `weight - weight/(1 + freq * normInverse)` with no $(k_1+1)$ factor (CLM-013).
  - `HnswGraphSearcher.searchLevel` adds neighbors that fail `acceptOrds` to the candidate queue but does not collect them as results.
  - `AbstractKnnVectorQuery.getLeafResults` chooses exact search when the filter cost is at most the per-leaf $k$, passes `cost + 1` as the visit limit otherwise, and falls back to exact search when the approximate search stops early (CLM-014).
- Scope: pinned Lucene behavior, not a universal BM25/HNSW or filtered-search definition. Engines built on Lucene may configure or override these paths.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Lexical vs Exact Dense Retrieval

- **Objective:** build versioned BM25 and exact-dense baselines and isolate analysis/representation failures.
- **Pre-Registered Hypothesis:** lexical and dense branches will show complementary strengths on prespecified identifier and semantic slices, not universal dominance.
- **Independent Variables:** analyzer/BM25 parameters, encoder/normalization/metric, corpus snapshot, and query slice.
- **Dependent Variables:** raw ranks, Recall/MRR/nDCG by slice, overlap/opportunity, latency, and errors.
- **Break & Falsify:** use IDs, misspellings, multilingual terms, paraphrases, corpus-stat changes, and analyzer mismatch.
- **Alignment:** Lessons 9.1 and 9.2.
- **Effort Estimate:** 4h.

### LAB B — HNSW and Filter Phase Diagram

- **Objective:** map exact-to-ANN loss under build/search, filter, load, and lifecycle conditions.
- **Pre-Registered Hypothesis:** higher search effort will improve ANN recall in the tested range but latency/memory/update cost and selective filters will prevent a universal optimum.
- **Independent Variables:** build/search parameters, size, concurrency, filter selectivity/correlation, and update/delete churn.
- **Dependent Variables:** ANN and relevant recall@k, visited work, fallback, build/update cost, memory, stage/tail latency, revision convergence, and ACL violations.
- **Break & Falsify:** inject deletion churn, rare ACL groups, correlated metadata, replica lag, and adversarial filter selectivity.
- **Alignment:** Lesson 9.3.
- **Effort Estimate:** 4.5h.

### LAB C — Hybrid Fusion and Rerank Depth

- **Objective:** compare branch opportunity, fusion, and bounded reranking on a quality–latency–capacity frontier.
- **Pre-Registered Hypothesis:** fusion/reranking will help only where branch complementarity and candidate opportunity justify their extra work.
- **Independent Variables:** branch depth, RRF weights/constant, candidate depth, truncation, model, and batch size.
- **Dependent Variables:** union opportunity, duplicate fragmentation, pre/post-rerank metrics, total work, tail latency, memory, and SLO-goodput.
- **Break & Falsify:** use redundant branches, missing-from-all items, truncated relevant spans, and loaded queues.
- **Alignment:** Lessons 9.4 and 9.5.
- **Effort Estimate:** 4.5h.

### LAB D — Offline-to-Loaded Retrieval Release

- **Objective:** evaluate a candidate index across representative slices and production-like open-loop load before release.
- **Pre-Registered Hypothesis:** at least one offline winner will lose its advantage or violate a guardrail after filter, freshness, or loaded-tail effects are included.
- **Independent Variables:** index release, slice, offered load, filter, cache, update state, timeout, and concurrency.
- **Dependent Variables:** IR metrics with uncertainty, exact/ANN gap, freshness, stage/end-to-end tails, queueing, failures, capacity, and goodput.
- **Break & Falsify:** include no-relevant queries, bursts, timeouts, cancellations, stale replicas, and changed traffic mix.
- **Alignment:** Lesson 9.6 and Incident 09.1.
- **Effort Estimate:** 4h.

## 07 Break / Incident Scenarios

### Incident 09.1 — Better nDCG, Missing Authorized Results

**Incident Symptoms:**
A release changes the encoder, HNSW parameters, hybrid weights, and reranker. Offline average nDCG rises, but selective tenant queries miss recent authorized documents, P99 rises, exact identifiers regress, and one replica reports an older index revision.

**Diagnostic Protocol (Task):**
1. **Form Competing Hypotheses:** corpus/index lag, analyzer mismatch, encoder shift, ANN/filter interaction, lifecycle/replica divergence, identity fusion bug, reranker truncation, deeper candidate cost, ACL error, judgment bias, or traffic change.
2. **Identify Missing Evidence:** query/corpus/judgment and component manifests, replica revisions, analyzer tokens, exact and ANN candidates, filter/fallback work, fusion identity, reranker input/output, queue/load, timeout, and authorization decisions.
3. **Design Discriminating Measurements:** replay matched queries through lexical, exact dense, filtered ANN, fusion, and reranker checkpoints; cross old/new one factor at a time and probe every replica.
4. **Rank Explanations:** use earliest candidate divergence, matched deltas, stage timing, and authorization invariants; do not infer one cause from aggregate nDCG.
5. **Intervene:** roll back or gate the discriminated component, converge the index, repair analysis/filter/identity/truncation, or reduce loaded depth as evidence supports.
6. **Remeasure:** repeat slice quality, exact–ANN gap, freshness, authorization, stage/tail latency, timeout, capacity, and canary rollback checks.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem — Multilingual Tenant-Aware Technical Search

Design retrieval for multilingual technical search with exact identifiers, semantic questions, tenant ACLs, updates, and a latency SLO.

**Required Deliverables:**
1. relevance/filter contract and versioned corpus, embedding, and index manifests;
2. BM25 and exact-dense baselines with analyzer, IDF convention, and similarity semantics, including one hand-reproduced BM25 score;
3. exact-versus-ANN/filter phase map including lifecycle churn and memory/capacity;
4. branch complementarity (union oracle), identity-normalized fusion with declared $c$, weights, and missing-item policy, and a rerank-depth frontier;
5. metric/judgment protocol stating gain, discount, cutoff, zero-IDCG, unjudged, and tie rules, with per-query results, uncertainty, and slices;
6. stage and end-to-end loaded tests with timeouts, failures, and goodput;
7. pinned production source trace and scoped generalization;
8. authorization/update/delete canaries and rollback;
9. an incident diagnosis separating representation, approximation, selection, lifecycle, and system loss.

## 09 Required Evidence & Rubric

### Required Artifact: Retrieval Stage Trace and Release Report

Submit manifests, raw branch/stage rankings, exact–ANN/filter comparisons, judgment and uncertainty records, loaded latency/capacity results, lifecycle probes, source trace, and canary/rollback decision.

### Rubric Dimensions

- **Mechanism:** *Insufficient* says “vector search.” *Competent* traces analyzer, encoder, metric, ANN, filter, fusion, and reranker. *Strong* links candidate losses and lifecycle state to observable stages.
- **Mathematics:** *Insufficient* quotes formulas. *Competent* states assumptions for BM25, ANN recall, RRF, and IR metrics. *Strong* validates predictions against raw ranks and counterexamples.
- **Evaluation:** *Insufficient* reports one mean. *Competent* pins population/judgments/slices. *Strong* audits incomplete judgments, uncertainty, temporal and authorization cases.
- **Performance:** *Insufficient* gives microbenchmark averages. *Competent* measures stage/tails/load. *Strong* separates critical path, total work, memory, queueing, failures, and goodput.
- **Diagnosis:** *Insufficient* guesses. *Competent* uses exact search and checkpoints. *Strong* performs matched one-factor replays and ranks interacting causes.
- **Operations:** *Insufficient* omits updates and ACLs. *Competent* tests them. *Strong* proves replica/delete convergence, canary, and rollback.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| BM25 scoring and analyzer contract | 9.1 | 9.1 Guided Practice (a) hand score, (b) pinned-index score; LAB A | Mastery deliverable 2; Incident 09.1 steps 2–3 (analyzer tokens, exact identifiers) | LAB A analyzer/index manifest, score decomposition, identifier-slice ranks |
| Dense similarity semantics | 9.2 | 9.2 Guided Practice (metric × normalization grid); LAB A | Mastery deliverable 2 | LAB A encoder/metric manifest and exact dense ranks |
| HNSW search and filtered ANN | 9.3 | 9.3 Guided Practice (a) hand trace, (b) effort/filter sweep; LAB B | Mastery deliverables 3, 8; Incident 09.1 steps 3–4 (earliest candidate divergence, ACL invariant) | LAB B exact-vs-ANN phase map with visited work, fallback counts, zero ACL violations |
| Pinned source trace | 9.1 and 9.3 (Lucene observations); §05 Production Source Trace | LAB A/B source reading | Mastery deliverable 7 | Trace of `BM25Similarity.idf`, `HnswGraphSearcher.searchLevel`, and `AbstractKnnVectorQuery.getLeafResults` at the pinned commit, with scope statement |
| Hybrid fusion | 9.4 | 9.4 Guided Practice (a) weighted RRF, (b) union oracle; LAB C | Mastery deliverable 4; Incident 09.1 step 2 (fusion identity) | LAB C branch lineage, union oracle, fused ranks with $c$, weights, and missing-item policy |
| Reranking and late interaction | 9.5 | 9.5 Guided Practice (a) MaxSim, (b) depth sweep; LAB C | Mastery deliverable 4; Incident 09.1 step 2 (reranker truncation) | LAB C depth frontier with pre/post-rerank quality and full cost |
| Metrics and system trade-offs | 9.6 | 9.6 Guided Practice (a) nDCG/MRR, (b) loaded evaluation; LAB D | Mastery deliverables 5–6, 9; Incident 09.1 steps 4, 6 | LAB D judgment policy (zero-IDCG, unjudged, tie rules), per-query metrics with uncertainty, loaded stage/tail results, goodput |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner passes when they can reproduce BM25 behavior, validate embedding/metric semantics, isolate exact-versus-ANN loss, break filtered HNSW, verify update/delete visibility, justify fusion by complementarity, prove reranker candidate bounds, choose correct metrics, trace current source, and defend a quality–latency–capacity release with rollback.

### Module Wrap-Up (Final Mental Model Reconstruction)

Retrieval is a chain of lossy transformations. Instrument every candidate and lifecycle boundary, compare approximation with exact search, compare rankings with relevance judgments, and accept complexity only where it improves the declared production frontier.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
