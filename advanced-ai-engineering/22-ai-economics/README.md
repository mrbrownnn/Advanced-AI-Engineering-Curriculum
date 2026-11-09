# Module 22 — AI Economics

## 00 Why This Module Exists

A cheap call is not necessarily a cheap task. Rejections, retries, fallbacks, cache writes, failed streams, idle replicas, quality failures, and late answers can move cost outside the denominator that a dashboard reports. AI economics therefore starts with task identity and success, not a vendor price table.

**Module Orientation**
- **Engineering Problem**: Choose a model-system policy that minimizes fully loaded cost per successful offered task while satisfying hard quality, latency, policy, and capacity constraints.
- **What You Will Do**: Build a task/attempt/spend ledger; model retries and failures; compare API and self-hosted capacity; trace a pinned cost calculator and router; test prefix and semantic caches; compare routing with cascades; and defend a sensitivity-tested decision frontier.
- **Environment**: Python 3.10+ and a spreadsheet or dataframe library. Labs use synthetic fixtures; optional provider or self-hosted measurements must record revision, configuration, workload, and invoice boundary.
- **Evidence Rule**: Every registry-backed statement carries **O** (source observation), **D** (derivation), or **H** (hypothesis) plus `CLM-###`. Pricing pages are dated commercial snapshots, never source-code traces. Unless explicitly called a dated snapshot, prices are synthetic.

## 01 Baseline Assumptions

- Module 04: offered versus admitted load, goodput, latency SLOs, saturation knees, and workload-faithful capacity measurement.
- Module 12: physical attempts, retry amplification, and pre-dispatch budget reservation.
- Module 15: paired quality evaluation, confidence intervals, slices, and noninferiority margins.
- Module 17: shadowing, canaries, rollback, and attribution limits.
- First taught here: offered/accepted/successful economic denominators, marginal/fixed/fully loaded cost, dated price snapshots, invoice reconciliation, cache economics, escalation economics, and flip-point analysis.

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
  statistical: SELECTIVE
  production_reasoning: REQUIRED
  failure_analysis: REQUIRED
  falsification: REQUIRED
  security: SELECTIVE
  economics: REQUIRED
  architecture_tradeoff: REQUIRED
  research_connection: REQUIRED

estimated_effort:
  instruction: 4.5h       # 45m x 6 lessons
  guided_practice: 2h     # 20m x 6 lessons
  labs: 13h               # 3h + 3.5h + 3h + 3.5h
  assessment: 3h          # mastery 2.5h + incident 0.5h
  source_trace: 2h        # Lessons 22.3/22.5, Labs A/D, and Section 09 are one activity
  total: 24.5h
```

Source-trace time is counted once. Lesson practice excludes the trace. Lab effort excludes the trace.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

Offered root task → admission → physical attempts (retry/fallback/cache/model route) → billed usage → terminal outcome → quality and latency gates → successful task. Join this immutable ledger to a versioned price snapshot and invoice. Classify marginal, fixed, and fully loaded cost. Then compare API, capacity, cache, router, and cascade policies only after hard constraints; sweep uncertain inputs and report flip points.

## 04 Lessons

### Lesson 22.1 — Denominator-Safe Task Economics

**Engineering Question:**
What did one successful user task cost after rejections, retries, failures, late outputs, and invalid answers?

**Concepts & Definitions:**
- **Offered task**: one root user intent at ingress, before admission.
- **Accepted task**: an offered task admitted to execution.
- **Physical/billed attempt**: each provider or local execution, including retry and fallback.
- **Successful task**: an offered task whose terminal output passes the declared correctness, policy, and latency predicate.
- **Attempt ledger**: immutable linkage from root task to admission, attempts, usage, route, cache state, spend, and terminal result. Reporting per call or accepted task can hide billed failures and retries (**D**, `CLM-017`).
- Generic retry budgets can prevent multiplicative overload, but published constants are system-specific (**O**, `CLM-012`). User re-prompts and abandonment can make provider and user objectives differ (**O**, abstract-only `CLM-005`).

**Mechanism Explanation:**
Assign a root ID before admission. Append one row per attempt; never replace failed rows. Apply the success predicate once at the root task. Reconcile billed spend to attempts, then aggregate the same cohort by offered, accepted, billed, and successful denominators. A gateway may log zero for some failed calls, unpriced models, or gateway-cache hits, so logs alone are not invoices (**O**, `CLM-014`).

**Quantitative Model / Derivation (D):**
For nonzero counts:
$$\frac{C}{N_{success}}=\frac{C}{N_{billed}}\frac{N_{billed}}{N_{accepted}}\frac{N_{accepted}}{N_{success}}$$
Report $C/N_{offered}$ separately. With independent attempts, per-attempt success $p$, and cap $k$:
$$E[A_k]=\frac{1-(1-p)^k}{p},\qquad P(success\ by\ k)=1-(1-p)^k$$
Homogeneous independent attempts keep expected attempts per success at $1/p$; heterogeneous or correlated failures do not (**D**, `CLM-018`).

where: $C$ = total billed spend for the cohort (\$); $N_{offered}$, $N_{accepted}$, $N_{billed}$, $N_{success}$ = counts of offered tasks, accepted tasks, billed physical attempts, and successful tasks in the same cohort; $p$ = probability one attempt succeeds; $k$ = retry cap (maximum attempts per task); $A_k$ = attempts actually used by one task (stop at first success or at $k$).

Step-by-step (why the factorization holds): multiply $C/N_{success}$ by $1=N_{billed}/N_{billed}$ and $1=N_{accepted}/N_{accepted}$, then regroup: $\frac{C}{N_{success}}=\frac{C}{N_{billed}}\cdot\frac{N_{billed}}{N_{accepted}}\cdot\frac{N_{accepted}}{N_{success}}$, i.e. (price per attempt) × (attempts per accepted task) × (accepted tasks per success). Each factor is a separate lever: price, retry amplification, and quality loss.

Step-by-step (expected attempts): attempt $i$ happens only if the first $i-1$ failed, so $P(\text{attempt } i)=(1-p)^{i-1}$. Summing the geometric series, $E[A_k]=\sum_{i=1}^{k}(1-p)^{i-1}=\frac{1-(1-p)^k}{1-(1-p)}=\frac{1-(1-p)^k}{p}$. The task fails only if all $k$ attempts fail, so $P(success\ by\ k)=1-(1-p)^k$. Dividing the two: attempts per success $=\frac{1-(1-p)^k}{p}\big/\left(1-(1-p)^k\right)=1/p$ for every $k$.

**Worked Example (synthetic, CLM-027):**
Input: 10,000 offered; 9,200 accepted; 10,051 billed attempts; 8,529 successful tasks; \$0.006 per billed attempt.
1. Spend $C=10{,}051\times0.006=\$60.306$.
2. Attempts/accepted $=10{,}051/9{,}200=1.0925$.
3. Accepted/success $=9{,}200/8{,}529=1.0787$.
4. Cost/success $=60.306/8{,}529=\$0.00707$; factor check $0.006\times1.0925\times1.0787=0.00707$.
5. Cost/offered $=60.306/10{,}000=\$0.00603$.
Result: successful-task cost is 17.8% above nominal call price ($0.00707/0.006=1.178$). Limits: equal attempt prices and complete billing join; zero-success windows report counts, not infinity hidden by filtering.

**Knowledge Check:**
1. Why does cost per accepted task improve when rejections rise, even if offered-user utility worsens?
2. When is retry independence implausible?

**Guided Practice:**
For easy tasks (90%, $p=.95$) and hard tasks (10%, $p=.20$), compute expected attempts and successes at $k=1$ and $k=3$. Find marginal attempts per added success.

**Feedback Contract:**
- *Expected Evidence*: $k=1$: attempts 1.0, success .875, 1.143 attempts/success. $k=3$: attempts 1.19125, success .94869, 1.256 attempts/success. Added successes cost $(1.19125-1)/(.94869-.875)=2.595$ attempts each.
- *How the numbers arise*: mix classes by traffic share. $k=1$: success $=.9(.95)+.1(.20)=.875$; attempts/success $=1/.875=1.143$. $k=3$: easy $E[A_3]=(1-.05^3)/.95=1.0525$, hard $E[A_3]=(1-.8^3)/.2=2.44$, so attempts $=.9(1.0525)+.1(2.44)=1.19125$; success $=.9(1-.05^3)+.1(1-.8^3)=.94869$; $1.19125/.94869=1.256$. The mixture ratio exceeds $1/p$ of the easy class because extra retries land mostly on hard tasks, so the marginal success (2.595 attempts) is much dearer than the average (1.256).
- *Common Failure*: Counting retries as new users or excluding billed failures.
- *Diagnostic Hint*: Can every physical attempt be joined to exactly one root task?
- *Concept to Revisit*: Offered versus admitted load (Module 04).

**Learning Outcome:**
Build a denominator-safe ledger and compute offered, accepted, attempt, and successful-task economics.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 22.2 — Marginal, Fixed, Fully Loaded, and Build-versus-API Cost

**Engineering Question:**
When does self-hosting beat an API after idle capacity, redundancy, labor, and SLO headroom are included?

**Concepts & Definitions:**
- **Marginal cost** changes when one more task is served within the current capacity step.
- **Fixed cost** does not change within the chosen horizon: platform work, amortized setup, baseline staffing.
- **Fully loaded cost** includes fixed and marginal execution, idle provisioned capacity, redundancy, operations, evaluation, and incident burden.
- **Provisioned utilization** uses sustainable SLO capacity, not peak benchmark throughput or sampled GPU activity.
- Published on-premise break-even examples can omit staffing, maintenance, redundancy, and partial utilization; their numerical break-evens do not transfer (**O**, `CLM-011`).

**Mechanism Explanation:**
Measure per-replica sustainable rate $\mu$ under the target workload and SLO. Provision for peak-to-average ratio (PAR), headroom $h$, and redundancy floor. Monthly cost moves in replica-sized steps, while API cost can be closer to linear. Recompute both with equal task success and latency contracts.

**Quantitative Model / Derivation (D, CLM-019):**
$$R=\max\left(R_{min},\left\lceil\frac{\lambda_{avg}PAR}{\mu h}\right\rceil\right),\quad C_{self}=F+Rc_{rep}$$
$$u_{avg}=\frac{\lambda_{avg}}{R\mu},\quad CPS_{self}=\frac{C_{self}}{Nq}$$
where $q$ is success rate and $N$ monthly tasks. API comparison uses reconciled cost per task divided by its success rate. Replica steps can create multiple local winner flips.

Other symbols: $R$ = provisioned replicas; $R_{min}$ = redundancy floor; $\lambda_{avg}=N/T_{month}$ = mean arrival rate (tasks/s, $T_{month}$ = seconds per month); PAR = peak-to-average ratio, so $\lambda_{avg}PAR$ is the peak rate; $\mu$ = sustainable tasks/s per replica at the SLO; $h\in(0,1]$ = headroom fraction (plan each replica to run at most $h\mu$ at peak); $F$ = fixed monthly cost; $c_{rep}$ = monthly cost per replica; $u_{avg}$ = mean utilization of provisioned SLO capacity; $CPS$ = cost per successful task.

Step-by-step: (1) peak load $=\lambda_{avg}PAR$; (2) usable capacity per replica $=\mu h$; (3) replicas needed $=$ peak/usable, rounded up because replicas are integers, then raised to $R_{min}$; (4) monthly cost $=F+Rc_{rep}$; (5) divide by successful tasks $Nq$. Break-even at fixed $R$: self-host wins when $\frac{F+Rc_{rep}}{Nq}<CPS_{API}$, i.e. $N>\frac{F+Rc_{rep}}{q\,CPS_{API}}$. Because $R$ itself jumps with $N$, this threshold must be checked inside each replica interval—the source of multiple flips.

**Worked Example (synthetic, CLM-027):**
Input: $F=\$9{,}000/month$, $c_{rep}=\$1{,}825$, $\mu=2.0$ tasks/s at SLO, $h=.70$, PAR=3, $R_{min}=2$, $N=8{,}000{,}000$, $q=.999$. Month=2,592,000 s. API equivalent=\$0.002362/task at equal quality.
1. $\lambda_{avg}=8{,}000{,}000/2{,}592{,}000=3.0864$/s.
2. $R=\lceil3.0864\times3/(2\times.7)\rceil=7$.
3. $C=9{,}000+7\times1{,}825=\$21{,}775$.
4. $u=3.0864/(7\times2)=22.05\%$; low average utilization is deliberate peak headroom.
5. $CPS=21{,}775/(8{,}000{,}000\times.999)=\$0.002725$; API wins at this volume.
Unit check: month $=30\times24\times3600=2{,}592{,}000$ s $=720$ h, so $c_{rep}=\$1{,}825$/month $\approx\$2.53$ per replica-hour; one replica's SLO capacity is $2.0\times2{,}592{,}000=5{,}184{,}000$ tasks/month, so a fully busy replica would cost $1{,}825/5{,}184{,}000\approx\$0.00035$/task. The realized replica cost here is $7\times1{,}825/8{,}000{,}000\approx\$0.0016$/task—about 4.5× higher—because $u_{avg}\approx22\%$ (and $F$ adds $\$0.0011$/task more).
Flip mechanics (illustrative, treating \$0.002362 as the API cost per successful task so $q$ cancels at equal quality): $R$ rises by one each time $N$ crosses $R\mu h\,T_{month}/PAR=R\times1{,}209{,}600$. With $R=9$ (valid up to 10.886M), self-host needs $N>(9{,}000+9\times1{,}825)/0.002362\approx10.76$M; the 10th replica then pushes the threshold to $\approx11.54$M, the 11th (from 12.096M) to $\approx12.31$M. Beyond that, each replica raises the threshold by only $1{,}825/0.002362\approx0.77$M while the replica interval is 1.21M wide, so the win persists.
Result: self-hosting first wins near 10.763M tasks but replica boundaries can reverse it; the registry fixture records later losing intervals and final win from 12.308M (**D**, `CLM-019`). Limits: no lead time, contract minimum, or capacity failure modeled.

**Knowledge Check:**
1. Why is accelerator rental alone not fully loaded cost?
2. Why can increasing volume make self-hosting lose immediately after it was winning?

**Guided Practice:**
Sweep 2M–15M tasks/month for PAR 1.5, 3, and 4. Plot replicas, utilization, and cost/success; identify every flip.

**Feedback Contract:**
- *Expected Evidence*: Integer replica steps, SLO-measured $\mu$, and higher first-win volume as PAR rises—the Lab B hypothesis (**H**, `CLM-026`).
- *Common Failure*: Provisioning from average arrivals or dividing by all completions instead of successful tasks.
- *Diagnostic Hint*: Which replica serves the peak but sits idle at the mean?
- *Concept to Revisit*: Saturation and headroom (Module 04).

**Learning Outcome:**
Classify costs and defend build-versus-API break-even with utilization, capacity steps, redundancy, and quality parity.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 22.3 — Dated Prices, Runtime Cost Accounting, and Invoice Reconciliation

**Engineering Question:**
How does billed usage become a trustworthy task cost when price modifiers and gateway behavior change?

**Concepts & Definitions:**
- **Price snapshot**: URL, access date, model/version, token class, cache/batch/tier/region modifiers, and contract discount.
- **Invoice reconciliation**: compare attempt-level gateway estimates with provider-billed usage and money; neither a public page nor a library price map is the invoice (**D**, `CLM-028`).
- Dated official snapshot, accessed 2026-10-01: one vendor documented 5-minute cache writes at 1.25× base input, 1-hour writes at 2×, reads at 0.1×, plus model exceptions (**O**, `CLM-008`). The same vendor documented 50% batch pricing, 24-hour availability, a 1.1× US-only modifier for specified model generations, and negotiated volume pricing (**O**, `CLM-010`). These are snapshots, not current-price guarantees.

**Mechanism Explanation — Pinned LiteLLM Trace (O):**
At commit `ed4caebb652728143276e71f27a0d92e6635e261`, static inspection only:
1. `cost_per_token` reaches `generic_cost_per_token`.
2. `parse_prompt_tokens_details` separates uncached, cache-read, and 5m/1h creation tokens; `_get_token_base_cost` selects rates and thresholds; `_calculate_input_cost` plus output modality rates produce token cost; regional uplift follows (**O**, `CLM-013`).
3. `completion_cost` adds built-in-tool/additional costs, then discount, then margin. `batch_cost_calculator` uses configured batch rates or a half-cost heuristic fallback (**O**, `CLM-015`).
4. Gateway-cache hits, missing price-map entries, and failures without stashed partial usage can record zero (**O**, `CLM-014`).
Pricing documentation is commercial evidence. These symbols form the source-code trace.

**Quantitative Model / Derivation:**
Partition usage into disjoint classes before multiplying:
$$C=\sum_j tokens_j\,rate_j + C_{tools}+C_{fixed}$$
Apply modifiers in the documented/runtime order; never count cached tokens both as cached and uncached. Reconciliation residual $\Delta=C_{invoice}-\sum C_{gateway}$ must be explained by usage, rate, modifier, rounding, timing, or missing-attempt differences.

where: $j$ indexes disjoint token classes (e.g. uncached input, cache write, cache read, output); $tokens_j$ = tokens billed in class $j$; $rate_j$ = effective price per token for that class after modifiers (list rates are usually quoted per MTok, so $rate_j=\text{(\$/MTok)}/10^6$); $C_{tools}$ = tool/additional charges; $C_{fixed}$ = non-usage charges in the same invoice window; $C_{invoice}$ = provider-billed money; $C_{gateway}$ = per-attempt gateway estimate.

**Worked Example (dated snapshot, not a decision fixture):**
Using only `CLM-008`'s 2026-10-01 listed example: 1.2M base-input tokens, 0.3M 5m-write tokens, 2.5M read tokens, and 0.4M output tokens for the named \$1/\$1.25/\$0.10/\$5 per MTok classes.
1. Input=\$1.20; writes=\$0.375; reads=\$0.25; output=\$2.00.
   - Arithmetic (tokens in MTok × \$/MTok): $1.2\times1=1.20$; $0.3\times1.25=0.375$; $2.5\times0.10=0.25$; $0.4\times5=2.00$. The write and read rates are the 1.25× and 0.1× modifiers applied to the \$1 base input rate.
2. Total list estimate=\$3.825.
3. If invoice line is \$3.95, residual=\$0.125 (3.27%); do not silently force equality.
   - Arithmetic: $\Delta=3.95-3.825=0.125$; relative to the estimate, $0.125/3.825=3.27\%$. Check region, contract, rounding, and omitted tool charges.
Limits: page accessed after research cutoff by one day; re-open before any real decision.

**Knowledge Check:**
1. Why can a missing price-map entry create a financially dangerous zero?
2. Why must failed streams preserve partial usage?

**Guided Practice:**
Trace the pinned symbols from entry point to discount/margin. Build five fixtures: cache read/write split, tier threshold, missing model, gateway hit, interrupted stream. Predict logged cost before execution.

**Feedback Contract:**
- *Expected Evidence*: Full commit, file, symbol, execution path, disjoint classes, expected zero paths, and an invoice residual table.
- *Common Failure*: Calling a pricing-page review a source-code trace.
- *Diagnostic Hint*: Which function produced usage, which map produced rates, and which system produced billed money?
- *Concept to Revisit*: Attempt ledger (22.1).

**Learning Outcome:**
Create a dated price snapshot, trace runtime cost accounting, and reconcile its estimate with billing.

*(Effort: 45m instruction, 20m practice; source trace counted separately)*

---

### Lesson 22.4 — Prefix and Semantic Cache Economics

**Engineering Question:**
When does a cache lower successful-task cost without violating quality or latency constraints?

**Concepts & Definitions:**
- **Prefix cache** reuses an exact rendered prefix; stable material must precede dynamic fields. One official cross-vendor snapshot also required full-prefix matching, model-specific minimum length, write/read modifiers, and expiry (**O**, `CLM-009`).
- Long-horizon agent results show prompt-cache benefit depends on stable boundaries and minimums; naive full-context caching can regress latency in an evaluated configuration (**O**, `CLM-007`).
- **Semantic cache** reuses a response for a similar prompt. Static similarity thresholds provide no universal correctness guarantee; vCache evaluates learned per-prompt thresholds under its own error definition (**O**, `CLM-006`).

**Mechanism Explanation:**
Prefix miss writes at premium $w$; a later exact hit reads at $r$. TTL, eviction, routing, minimum length, and dynamic fields determine hit probability. Semantic lookup always pays embedding/lookup cost; a hit avoids generation but can be wrong or stale. Cost optimization occurs only inside the quality-feasible set.

**Quantitative Model / Derivation:**
Prefix written once then read $n$ times beats uncached iff
$$n>\frac{w-1}{1-r}.$$
With per-request hit share $h$, factor $(1-h)w+hr<1$ iff $h>(w-1)/(w-r)$ (**D**, `CLM-020`). Under exponential gaps of mean $g$ and refreshed TTL $\tau$, $h=1-e^{-\tau/g}$.
Semantic fixture:
$$C=c_e+(1-h)c_L,\quad S=s(1-h\epsilon),\quad CPS=C/S$$
under declared independence (**D**, `CLM-021`).

where (prefix): costs are in units of the base input price per prefix token, so uncached use costs 1; $w$ = write multiplier (>1); $r$ = read multiplier (<1); $n$ = later reads of one written prefix; $h$ = per-request hit share (not the headroom $h$ of Lesson 22.2); $g$ = mean gap between requests sharing the prefix; $\tau$ = TTL, refreshed on each hit.
where (semantic): $c_e$ = embedding/lookup cost paid on every request; $c_L$ = LLM generation cost paid on misses; $s$ = success rate of generated answers; $h$ = semantic hit share; $\epsilon$ = probability a hit returns a wrong/stale answer; $C$, $S$ = expected cost and success per request.

Step-by-step (prefix): write once and read $n$ times costs $w+nr$; uncached costs $n+1$. Cached wins iff $w+nr<n+1 \iff n(1-r)>w-1 \iff n>\frac{w-1}{1-r}$. Per request: a miss costs $w$, a hit $r$, so expected factor $(1-h)w+hr=w-h(w-r)<1 \iff h>\frac{w-1}{w-r}$. With exponential gaps, a request hits iff its gap is shorter than $\tau$: $h=P(gap<\tau)=1-e^{-\tau/g}$.
Step-by-step (semantic): every request pays $c_e$; only misses pay $c_L$, giving $C=c_e+(1-h)c_L$. Misses succeed with prob. $s$, hits with $s(1-\epsilon)$ (assumed independence), so $S=(1-h)s+hs(1-\epsilon)=s(1-h\epsilon)$.

**Worked Example (synthetic, CLM-027):**
Prefix $w=1.25,r=.1,g=10$ min. At $\tau=5$, $h=1-e^{-.5}=.3935$; factor $=.6065(1.25)+.3935(.1)=.7975$, a 20.25% input saving. Break-even $h>.25/1.15=.2174$.
Semantic: $c_L=.006,s=.93,c_e=.00002$. Policy A $h=.30,\epsilon=.02$: $C=.00422$, $S=.92442$, CPS=$.004565$. Policy B $h=.55,\epsilon=.15$: $C=.00272$, $S=.853275$, CPS=$.003188$ but fails a hard .90 success floor. Result: cheaper point estimate B is infeasible.
Arithmetic: prefix saving $=1-.7975=20.25\%$; break-even $(1.25-1)/(1.25-.1)=.25/1.15$. Policy A: $C=.00002+.70(.006)=.00422$, $S=.93(1-.30\times.02)=.93(.994)=.92442$, CPS $=.00422/.92442=.004565$. Policy B: $C=.00002+.45(.006)=.00272$, $S=.93(1-.55\times.15)=.93(.9175)=.853275<.90$, CPS $=.00272/.853275=.003188$.
Reuse count (illustrative, multipliers from Lesson 22.3's snapshot): 5m write $w=1.25$, $r=.1$ gives $n>.25/.9=.28$, so one read pays back; 1h write $w=2$ gives $n>1/.9=1.11$, so at least two reads are needed.

**Knowledge Check:**
1. Why can one reuse pay for a 5m write but not a 1h write?
2. Why is overall semantic-cache success insufficient?

**Guided Practice:**
Move a timestamp from before to after a stable prefix, hold gaps below TTL, and compare read/write tokens, TTFT, and cost. Pre-register the hit-share falsifier (**H**, `CLM-025`).

**Feedback Contract:**
- *Expected Evidence*: With dynamic-first, predicted hit share below .05; stable-first above .8 after warmup; quality reported separately for semantic hits and misses.
- *Common Failure*: Treating cache-hit rate as benefit without write premium or wrong-hit loss.
- *Diagnostic Hint*: What exact bytes/tokens define identity, and what failure can a hit introduce?
- *Concept to Revisit*: Hard quality gates (Module 15).

**Learning Outcome:**
Model prefix and semantic cache cost, latency, and quality and falsify assumed hit economics.

*(Effort: 45m instruction, 20m practice)*

---

### Lesson 22.5 — Routing, Cascades, and Escalation Cost

**Engineering Question:**
When should selection happen before generation, and when is paying for a cheap answer before escalation worthwhile?

**Concepts & Definitions:**
- **Router** predicts before generation and pays router plus one chosen model. RouteLLM uses a strong-win estimate and threshold; its benchmark savings and overhead are scoped to evaluated pairs/hardware (**O**, `CLM-002`).
- **Cascade** runs a weak model, verifies afterward, and escalates selected tasks. FrugalGPT is a reference learned cascade with historical, dataset-specific savings (**O**, `CLM-001`).
- Quality estimators bound both designs: ex-ante for routing, post-hoc for cascading (**O**, `CLM-003`). A 2026 preprint attributes a cascade disadvantage in four of five evaluated datasets mainly to unavoidable weak-stage generation cost (**O**, `CLM-004`).
- One deployment abstract reports cost reduction and response acceptance, not offered-task success; denominator matters (**O**, abstract-only `CLM-029`). Industry-wide adoption and offered-success savings remain `TODO_VERIFY` (**TODO_VERIFY**, `CLM-030`).

**Mechanism Explanation:**
A router scores the request then chooses one branch. A cascade always pays weak generation and verification, accepts some cheap correct answers, falsely accepts some wrong answers, and escalates the rest. Distribution shift changes weak accuracy and verifier rates jointly.

**Quantitative Model / Derivation (D, CLM-022):**
With weak accuracy $a$, verifier true-accept $t$, false-accept $f$, strong accuracy on escalations $b$:
$$e=1-[at+(1-a)f]$$
$$C=c_W+c_v+ec_S,\quad S=at+eb,\quad WrongAccepted=(1-a)f$$
Cost-only break-even versus always-strong requires $e<1-(c_W+c_v)/c_S$.

where: $a$ = probability the weak answer is correct; $t$ = P(verifier accepts | weak correct); $f$ = P(verifier accepts | weak wrong); $b$ = strong-model success on escalated tasks; $e$ = escalation share; $c_W$, $c_v$, $c_S$ = per-task cost of weak generation, verification, and strong generation; $C$, $S$ = expected cost and success per task; $WrongAccepted$ = share of tasks ending with an accepted wrong weak answer.

Step-by-step: accepted weak share $=at$ (correct, accepted) $+(1-a)f$ (wrong, accepted); everything else escalates, so $e=1-[at+(1-a)f]$. Every task pays $c_W+c_v$; only escalations pay $c_S$, giving $C=c_W+c_v+ec_S$. Successes come from correct accepted answers ($at$) plus successful escalations ($eb$). Break-even: $c_W+c_v+ec_S<c_S \iff e<1-(c_W+c_v)/c_S$.

**Worked Example (synthetic, CLM-027):**
$a=.70,t=.90,f=.20,b=.88,c_W=.0006,c_v=.0002,c_S=.006$.
1. Accepted weak share $=.63+.06=.69$; escalation $e=.31$.
2. Cost=$.0006+.0002+.31(.006)=.00266$.
3. Success=$.63+.31(.88)=.9028$; wrong accepted=.06.
4. CPS=$.002946$; escalated latency contains both stages.
   - Arithmetic: $at=.70\times.90=.63$; $(1-a)f=.30\times.20=.06$; CPS $=.00266/.9028=.002946$.
5. Cost-only threshold $e<1-.0008/.006=.8667$, but quality can still reject the policy.

**Pinned RouteLLM Trace (O, CLM-016):**
At commit `0b64fdafe049e596a3f5657c219329f24af24198`, `Controller.completion` parses and validates the threshold, routes on last-message content, then `Router.route` selects strong when win rate ≥ threshold. Invalid causal-classifier output fails to strong. `calibrate_threshold.py` uses an Arena-score quantile for requested strong-call share. Static inspection only; research code, not current default.

**Knowledge Check:**
1. Why can identical escalation rate hide different quality?
2. Why does calibration to a strong-call share not calibrate task success?

**Guided Practice:**
Compare always-strong, fixed-share router, and cascade on the same held-out tasks. Shift traffic toward weak-model failures. Measure success intervals, latency, route/escalation share, false accepts, and CPS.

**Feedback Contract:**
- *Expected Evidence*: Cascade cost includes every weak and verifier call; shift test evaluates the hypothesis that cascade CPS rises more than fixed-share routing (**H**, `CLM-024`).
- *Common Failure*: Pricing only escalated strong calls or treating verifier confidence as ground truth.
- *Diagnostic Hint*: Which stages execute on every task, and where can a wrong answer terminate?
- *Concept to Revisit*: Conditional denominators (22.1).

**Learning Outcome:**
Compare routing and cascading with complete execution paths, estimator error, shift, and pinned source behavior.

*(Effort: 45m instruction, 20m practice; source trace counted separately)*

---

### Lesson 22.6 — Constrained Decision Frontiers, Sensitivity, and Flip Points

**Engineering Question:**
How do we choose among API, self-hosted, cached, routed, and cascaded systems without hiding constraints in a weighted score?

**Concepts & Definitions:**
- **Hard constraint**: mandatory quality lower bound, latency upper bound, policy rule, or capacity requirement.
- **Feasible set**: options whose intervals and loaded measurements satisfy every hard constraint.
- **Decision frontier**: nondominated feasible options in cost, quality, latency, and capacity.
- **Flip point**: uncertain input value at which recommendation changes while other inputs stay fixed.
- A scalar $\hat q-\lambda\hat c$ (estimated quality minus $\lambda$ × estimated cost; here $\lambda$ is a cost-penalty weight, not the arrival rate of Lesson 22.2) is useful only after feasibility and only to the extent estimators are valid (**O**, `CLM-003`; **D**, `CLM-023`).

**Mechanism Explanation:**
Predeclare constraints. Measure every option on matched traffic. Remove infeasible options using quality intervals and loaded latency—not point estimates. Rank survivors by fully loaded cost per successful task. Sweep price, volume, PAR, cache hit rate, escalation, quality, and latency. Report joint scenarios, reservation, and rollback triggers.

**Quantitative Model / Trade-off Comparison (D):**
For option $j$, feasible iff $LCB(S_j)\ge S_{min}$, $P95(L_j)\le L_{max}$, capacity ≥ peak, and policy predicates pass. Among feasible options minimize $C_j/N_{success,j}$. Solve equality between leading options for each flip point; one-at-a-time sensitivity is conditional, not joint uncertainty.

where: $LCB(S_j)$ = lower confidence bound on option $j$'s success rate; $S_{min}$ = required success floor; $P95(L_j)$ = 95th-percentile loaded latency; $L_{max}$ = latency ceiling. Flip point for one input $x$: hold all else fixed and solve $CPS_A(x)=CPS_B(x)$ for the two leading feasible options (re-checking feasibility, since $x$ may also move quality or latency).

**Worked Example (synthetic, CLM-027):**
At 8M tasks/month, constraints are success LCB ≥ .88 and P95 ≤2.5 s.
- Always-cheap: LCB .82, P95 1.1 s, CPS .00060 → infeasible quality.
- Cascade: LCB .8800 but P95 3.1 s, CPS .002946 → infeasible latency (boundary quality alone is not enough).
- Self-hosted: LCB .91, P95 2.2 s, CPS .002975 → feasible.
- Cached router: LCB .90, P95 1.8 s, CPS .002582 → feasible and selected.
Sensitivity from the declared fixture: selection flips at prefix-hit rate .705, PAR 2.30, or fixed self-host cost \$6,124/month (**D**, `CLM-023`). Limits: each holds other values fixed; use scenarios or Monte Carlo for joint uncertainty.

**Knowledge Check:**
1. Why can a Pareto-frontier point still be unacceptable?
2. Why is a one-variable flip point not a confidence interval?

**Guided Practice:**
Create low/base/high scenarios for volume, success, PAR, hit rate, and effective price. Recompute feasibility first, then winner. State reservation and rollback thresholds.

**Feedback Contract:**
- *Expected Evidence*: Hard-gate table, CPS among survivors, at least three flip points, joint adverse scenario, and measurement resolving each uncertainty.
- *Common Failure*: Ranking all options by weighted utility before removing SLO violations.
- *Diagnostic Hint*: Would the selected option remain legal if its cost were zero?
- *Concept to Revisit*: Noninferiority and loaded latency (Modules 15 and 04).

**Learning Outcome:**
Construct and defend a constrained decision frontier with sensitivity, uncertainty, flip points, and rollback.

*(Effort: 45m instruction, 20m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**
- [FrugalGPT](https://arxiv.org/abs/2305.05176), Chen et al. 2023: cost model and learned cascades; historical prices and dataset-specific author results (**O**, `CLM-001`).
- [RouteLLM](https://arxiv.org/abs/2406.18665), Ong et al. 2024/2025: threshold routers and scoped evaluation (**O**, `CLM-002`).
- [Unified Routing and Cascading](https://arxiv.org/abs/2410.10347), Dekoninck et al. 2024/2025: estimator-relative optimization (**O**, `CLM-003`).
- [Google SRE Handling Overload](https://sre.google/sre-book/handling-overload/): retry amplification and budgets; generic system practice (**O**, `CLM-012`).

**WORKLOAD-DEPENDENT**
- [On-Premise LLM Cost-Benefit Analysis](https://arxiv.org/abs/2509.18101), Pan et al. 2025: break-even model with stated omissions (**O**, `CLM-011`).
- [vCache](https://arxiv.org/abs/2502.03771), Schroeder et al. 2025/2026: semantic-cache error constraints (**O**, `CLM-006`).
- [Prompt Caching for Long-Horizon Tasks](https://arxiv.org/abs/2601.06007), Lumer et al. 2026: workload-specific cache boundary results (**O**, `CLM-007`).

**FRONTIER**
- [Is Escalation Worth It?](https://arxiv.org/abs/2605.06350), Bouchard 2026, preprint (**O**, `CLM-004`).
- [Routing, Cascades, and User Choice](https://arxiv.org/abs/2602.09902), Mahmood 2026, abstract-only (**O**, `CLM-005`).
- [RouteNLP](https://arxiv.org/abs/2604.23577), Guo et al. 2026, abstract-only single deployment report (**O**, `CLM-029`).

**DATED PRICE SNAPSHOTS — NOT SOURCE CODE**
- Official prompt-caching and pricing pages accessed 2026-10-01 (**O**, `CLM-008`, `CLM-009`, `CLM-010`). Re-open before use; negotiated prices may differ.

**PRODUCTION SOURCE TRACE**
- LiteLLM commit `ed4caebb652728143276e71f27a0d92e6635e261`: `litellm/litellm_core_utils/llm_cost_calc/utils.py::{generic_cost_per_token,_get_token_base_cost,parse_prompt_tokens_details,_calculate_input_cost,calculate_cache_writing_cost}`; `litellm/cost_calculator.py::{cost_per_token,completion_cost,_apply_cost_discount,_apply_cost_margin,batch_cost_calculator}`; logging failure/partial-usage paths. Static only (**O**, `CLM-013`, `CLM-014`, `CLM-015`).
- RouteLLM commit `0b64fdafe049e596a3f5657c219329f24af24198`: `Router.route`, `CausalLLMRouter.calculate_strong_win_rate`, controller parsing/validation/routing, and calibration quantile. Static only (**O**, `CLM-016`).

**UNVERIFIED LANDSCAPE**
- Adoption and offered-success savings across industry remain unknown (**TODO_VERIFY**, `CLM-030`).

---

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY → DEFEND`. Synthetic inputs remain labeled; real measurements record population, configuration, date, and uncertainty.

### LAB A — Task, Attempt, Usage, and Invoice Ledger
- **Hypothesis**: Reconciled cost per successful offered task exceeds per-call gateway cost when retries, failures, and quality/latency failures are retained.
- **Independent Variables**: Retry cap, failure correlation, rejection rate, partial stream, price-map completeness.
- **Dependent Variables**: Offered/accepted/success counts, attempts, gateway estimate, invoice residual, CPS.
- **Build/Measure**: Implement ledger and five LiteLLM trace fixtures; use dated snapshot or synthetic rates.
- **Break & Falsify**: Construct a complete, no-retry workload where both costs match. Hypothesis fails if full ledger still cannot explain the residual.
- **Alignment**: Lessons 22.1 and 22.3; `CLM-012`–`CLM-018`, `CLM-028`.
- **Effort Estimate**: 3h total; source trace separate.

### LAB B — Build-versus-API Capacity Frontier
- **Hypothesis**: First self-hosting win moves to higher volume as PAR rises (**H**, `CLM-026`).
- **Independent Variables**: Volume, PAR, headroom, redundancy, $\mu$, fixed cost, replica price, success.
- **Dependent Variables**: Replicas, utilization, capacity margin, fully loaded CPS, flip points.
- **Break & Falsify**: Sweep replica boundaries; find any non-monotone or lowest-volume win and explain whether assumptions caused it.
- **Alignment**: Lessons 22.2 and 22.6.
- **Effort Estimate**: 3.5h total.

### LAB C — Cache Boundary and Quality Debt
- **Hypothesis**: A dynamic field before the stable prefix collapses exact-cache hits and moving it after restores them (**H**, `CLM-025`).
- **Independent Variables**: Prefix layout, TTL/gaps, minimum length, semantic threshold, shift/staleness.
- **Dependent Variables**: Read/write tokens, hit share, TTFT, hit/miss success intervals, CPS.
- **Break & Falsify**: Short prompts below minimum; gaps beyond TTL; stale semantic entries; a dynamic-first hit share above .5 falsifies the proposed boundary mechanism until explained.
- **Alignment**: Lesson 22.4; `CLM-006`–`CLM-010`, `CLM-020`–`CLM-021`.
- **Effort Estimate**: 3h total.

### LAB D — Router, Cascade, Shift, and Decision Frontier
- **Hypothesis**: On a weak-model-hard shift, cascade escalation and wrong acceptance rise, increasing cascade CPS more than a fixed-share router (**H**, `CLM-024`).
- **Independent Variables**: Traffic mix, threshold, strong share, verifier $t/f$, model costs, latency load.
- **Dependent Variables**: Route/escalation share, false accepts, paired success interval, P95 latency, CPS.
- **Break & Falsify**: Search a shift where cascade CPS rises no more than router CPS or remains feasible with unchanged escalation. Preserve null results.
- **Alignment**: Lessons 22.5–22.6; pinned RouteLLM trace.
- **Effort Estimate**: 3.5h total; source trace separate.

## 07 Break / Incident Scenarios

### Incident 22.1 — The 40% “Savings” That Increased Spend

**Symptoms**: Finance reports invoice spend up 18% after a dashboard claimed 40% savings. Offered traffic is flat. Gateway cost excludes cache hits and most failures; retries rose from 1.04 to 1.31 attempts/accepted task; prefix hit share fell after a timestamp moved to the prompt front; cascade escalation rose; accepted-task latency remains healthy while rejection rose. (Scale check: with price per attempt unchanged, the retry factor alone multiplies spend per accepted task by $1.31/1.04\approx1.26$, i.e. +26%.)

**Diagnostic Protocol**:
1. **Competing hypotheses**: denominator laundering; missing failed-stream usage; stale rate map/modifiers; cache-boundary regression; cascade shift; retry amplification; invoice-window mismatch.
2. **Initial ranking**: tie each symptom to predictions; do not infer invoice error from dashboard disagreement.
3. **Missing evidence**: root/attempt join, offered/accepted/success counts, partial usage, price-map revision, invoice line items, read/write tokens, escalation/false accepts, quality and latency gates.
4. **Discriminating tests**: replay fixed tasks without retries; restore prefix boundary; reconcile one invoice sample; compare cascade rates on old/new slices.
5. **Diagnosis**: rank supported causes and quantify each residual; retain unexplained remainder.
6. **Mitigation**: cap retries, restore stable-prefix order, fail missing prices closed for accounting, constrain cascade, and freeze rollout if success/latency gates fail.
7. **Remeasurement**: same offered cohort; invoice residual <1%, complete attempt join >99.9%, success LCB and P95 gates pass, and savings reproduced on billed spend.

---

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem — Support Assistant Economics

**Synthetic fixture (CLM-027):** 8M offered tasks/month; 8% rejected; hard gates: success LCB ≥.88, P95 ≤2.5 s, no cross-tenant cache reuse. Strong model S costs \$2.4/MTok input and \$12/MTok output; weak W costs \$.24/\$1.2. Mean task: 2,000 input and 300 output tokens. Retry cap 2. Candidate systems: always-S API; W/S router; W→verifier→S cascade; prefix+semantic cache before router; self-hosted option using Lesson 22.2 fixture. Base measured/synthetic policy values are those in Lessons 22.2, 22.4–22.6. Treat all as exercise inputs, not vendor rates. Unit conversion for one attempt: S $=2{,}000\times\$2.4/10^6+300\times\$12/10^6=\$0.0048+\$0.0036=\$0.0084$; W $=2{,}000\times\$0.24/10^6+300\times\$1.2/10^6=\$0.00084$.

**Required Deliverables**:
1. Task/attempt/spend schema and denominator table for offered, accepted, billed, successful, rejected, failed, late, and invalid tasks.
2. Retry analysis with easy/hard classes, marginal attempts per added success, and cap recommendation.
3. Dated price-snapshot schema plus one synthetic pricing ledger and invoice reconciliation; no claim that fixture rates are current.
4. Marginal/fixed/fully loaded classification and API/self-host capacity model with $\mu$, PAR, headroom, redundancy, utilization, and replica steps.
5. Prefix and semantic-cache model with TTL/minimum/boundary behavior, wrong-hit quality, and latency.
6. Router-versus-cascade execution and cost model, including verifier error, escalation, shift, and false accepts.
7. Feasibility table applying quality, latency, policy, and capacity hard gates before cost ranking.
8. Decision frontier with at least five one-way flip points and two joint uncertainty scenarios.
9. Production source trace for both pinned repositories, separating static observations from executed fixtures.
10. Incident 22.1 diagnosis, rollout reservation, rollback triggers, and open `TODO_VERIFY` items.

## 09 Required Evidence & Rubric

### Required Artifacts
- Immutable ledger extract and reconciliation report.
- Capacity/build-versus-API workbook or executable notebook.
- Cache and route/cascade experiment reports with preregistered hypotheses and falsifiers.
- Constrained decision record with intervals and flip points.
- Pinned production source trace covering LiteLLM and RouteLLM symbols in Section 05.

### Rubric
- **Denominators and Accounting**: *Insufficient* reports per-call/accepted cost and drops failures. *Competent* reconciles root tasks, attempts, success, and invoice. *Strong* also quantifies residuals and marginal retry economics.
- **Capacity and Fully Loaded Cost**: *Insufficient* compares API with accelerator rental or mean capacity. *Competent* includes fixed/marginal cost, PAR, SLO $\mu$, headroom, redundancy, and utilization. *Strong* models replica-boundary reversals and measured drift.
- **Cache/Route Mechanisms**: *Insufficient* counts hits or strong-call share alone. *Competent* models full path and quality loss. *Strong* falsifies under TTL, boundary, verifier, and distribution shift.
- **Decision Quality**: *Insufficient* selects lowest point cost. *Competent* hard-gates then compares CPS. *Strong* gives intervals, joint scenarios, flip points, reservation, and rollback.
- **Source Reasoning**: *Insufficient* cites pricing pages as code. *Competent* traces pinned entry points and symbols. *Strong* tests fixtures and distinguishes map estimate, gateway log, and invoice.
- **Incident Diagnosis**: *Insufficient* guesses one cause. *Competent* follows steps 1–7. *Strong* apportions spend delta, preserves unexplained residual, and proves recovery on same cohort.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Denominator-safe CPS and retries | 22.1 | LAB A | Deliverables 1–2; Incident 1–5 | Ledger, factorization, marginal retry table |
| Fully loaded build-versus-API | 22.2 | LAB B | Deliverable 4 | Capacity curve, replica steps, utilization |
| Dated pricing and reconciliation | 22.3 | LAB A | Deliverables 3, 9; Incident 3–7 | Snapshot, pinned trace, invoice residual |
| Cache economics and quality | 22.4 | LAB C | Deliverable 5; Incident 4–7 | Token-class ledger, hit/miss quality |
| Routing/cascade economics | 22.5 | LAB D | Deliverable 6; Incident 3–6 | Execution-path cost, shift report |
| Constrained frontier and sensitivity | 22.6 | LABs B/D | Deliverables 7–8, 10 | Feasibility table, flip points, rollback |
| Production source trace | 22.3, 22.5 | LABs A/D | Deliverable 9 | Full commits, files, symbols, static/executed labels |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria
Learner can:
1. Preserve offered, accepted, attempted, billed, and successful populations in one ledger.
2. Quantify retries/failures and marginal cost per added success.
3. Separate marginal, fixed, and fully loaded cost and provision from SLO capacity.
4. Produce and reconcile a dated price snapshot without calling it current indefinitely.
5. Model cache, router, and cascade economics with quality and latency gates.
6. Trace pinned cost-accounting and routing implementations.
7. Select only among feasible options and defend flip points, uncertainty, and rollback.

### Module Wrap-Up
Core invariant: money follows physical work, but value follows successful offered tasks. Join both before optimizing. Prices are versioned inputs; utilization is a consequence of capacity policy; cache hits and cheap routes are useful only inside hard constraints. Final sequence: ledger → reconcile → classify cost → measure capacity/mechanisms → gate feasibility → compare CPS → sweep uncertainty → reserve and roll back.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
