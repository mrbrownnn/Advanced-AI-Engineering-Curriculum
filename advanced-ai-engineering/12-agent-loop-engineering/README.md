# Module 12 — Agent Loop Engineering

## 00 Why This Module Exists

An agent is not a prompt that keeps talking. It is a controller that repeatedly converts an observed state into a proposed action, subjects that action to deterministic validation and authority policy, executes it through an external interface, incorporates the resulting observation, and stops for an explicit reason.

```text
goal + current state + remaining budget
                 |
                 v
             model proposal
                 |
        validate + authorize
          /              \
   reject/repair       execute tool
          |                |
          +---- observation envelope
                           |
                 verify progress/effect
                     /           \
             continue/replan   terminal state
```

This module owns the observe–decide–act loop, tool and observation contracts, budgets and stopping, retry/replanning, progress/stall detection, reflection as a fallible mechanism, authority boundaries, trajectory telemetry, and agent evaluation. Module 11 owns context and memory; Module 13 owns the broader model harness and structured-output reliability; Module 14 owns durable checkpoint/resume and idempotent side-effect execution; Module 18 owns full threat modeling.

**Research cutoff:** 2026-09-30. Sources re-opened on that date are marked in the registry; other entries keep their original access dates.

**Module Orientation**
- **Engineering Problem**: Maximize verified task utility while bounding invalid effects, policy violations, latency, tokens, calls, and cost.
- **What You Will Do**: Implement and instrument a bounded controller, define tool and observation contracts, inject ambiguous effects and recovery failures, compare progress and reflection mechanisms, trace a pinned LangGraph path, and defend a governed operational-agent design.
- **Environment**: Python 3.10+ for controller and fault-injection harnesses. Sandboxed mock tools and an append-only trajectory store are required; external side effects are optional and must remain disposable or simulated.
- **Evidence Rule**: Label source observations (**O**), explicit derivations (**D**), and telemetry-dependent hypotheses (**H**). Model claims about success, failure, or tool completion are not execution evidence.

## 01 Baseline Assumptions

- Module 00: measurands, experimental units, uncertainty, and falsification.
- Module 04: queueing, end-to-end latency, throughput, goodput, overload, and backpressure.
- Module 07: behavioral uncertainty, calibration, and abstention.
- Modules 08–10: lineage, retrieval, evidence assembly, and citation correctness.
- Module 11: context accounting, typed memory, versioning, and memory provenance.

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
  economics: SELECTIVE
  architecture_tradeoff: REQUIRED
  research_connection: SELECTIVE

estimated_effort:
  instruction: 4h        # lesson instruction: 35+35+60+35+35+40 min = 240 min
  guided_practice: 3h    # lesson practice: 25+25+35+30+30+35 min = 180 min
  labs: 12h              # LAB A 3h + LAB B 3h + LAB C 3h + LAB D 3h
  assessment: 3h         # Mastery transfer problem 2.5h + Incident 12.1 0.5h
  source_trace: 2h       # Section 05 LangGraph trace practice and the Section 09 Required Artifact, counted once
  total: 24h
```
Each category is counted once. Lab analysis is not also counted as guided practice, and the source trace is not also counted as assessment time.

The learner must be able to define a loop as an explicit state machine; design typed tool and observation contracts; separate proposal from authorization; enforce independent resource and semantic stops; classify failures before retrying; detect no-progress and oscillation without blocking legitimate iteration; evaluate reflection against grounded feedback; inject tool anomalies; and diagnose complete trajectories rather than final answers alone.

---

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
                         controller-owned state
                      /          |             \
                 goal/plan   budgets/policy   effect ledger
                      \          |             /
                           model proposal
                                 |
                  schema validation + authorization
                         /               \
                    invalid             valid
                 repair or stop      tool execution
                                           |
                              typed observation envelope
                                           |
                     progress? error class? postcondition?
                       /             |               \
                   continue        replan          terminate
```

Keep these distinctions explicit:

1. **Proposal:** what action did the model request?
2. **Authorization:** is this caller permitted to perform that action now?
3. **Execution:** what did the tool attempt, and with which contract/version?
4. **Effect:** did the external state change, not change, or become unknown?
5. **Observation:** what evidence was returned, parsed, truncated, or rejected?
6. **Progress:** did a verified subgoal or state predicate improve?
7. **Termination:** did the episode succeed, fail, abstain, escalate, or exhaust a budget?

## 04 Lessons

### Lesson 12.1 — Model the Loop as a Bounded Controller

**Engineering Question:**
Which state and decisions must remain controller-owned so that an agent episode is bounded, inspectable, and replayable?

**Concepts & Definitions:**
- **Episode**: one bounded attempt to achieve a declared goal.
- **Controller state**: typed state owned by deterministic orchestration rather than model prose.
- **Terminal reason**: verified success, explicit failure, abstention, escalation, cancellation, or budget exhaustion.

**Quantitative Model / Derivation:**
Represent one episode as

$$
\tau=(s_0,a_0,o_1,s_1,\ldots,a_{T-1},o_T,s_T),\qquad
s_{t+1}=F(s_t,a_t,o_{t+1}).
$$

where:
- $\tau$: the trajectory, the full record of one episode;
- $s_t$: controller state after step $t$ ($s_0$ is the initial state);
- $a_t$: the action the controller dispatched from state $s_t$;
- $o_{t+1}$: the observation returned by that action;
- $T$: the number of steps the episode actually took;
- $F$: the controller's deterministic transition function.

Read the formula left to right: from $s_0$ the controller dispatches $a_0$, receives $o_1$, and computes $s_1=F(s_0,a_0,o_1)$; then repeats. A 2-step read-only lookup is $\tau=(s_0,a_0,o_1,s_1,a_1,o_2,s_2)$ with $T=2$. Because $F$ is deterministic, replaying the recorded $(a_t,o_{t+1})$ pairs from $s_0$ reproduces every $s_t$; that is what "replayable" means here.

This is exact bookkeeping for a declared state schema. It is not automatically a Markov model: if `s_t` omits relevant history, permissions, hidden environment state, or pending effects, it is not sufficient to predict the next transition.

**Mechanism Explanation:**
The controller—not the model—owns legal states, action validation, authorization, tool dispatch, budget accounting, terminal outcomes, and the trajectory ledger. The model may emit `call_tool`, `respond`, `abstain`, `ask`, or `escalate`; each is a proposal until the controller validates it.

ReAct is a reference pattern for interleaving reasoning and environment actions on evaluated tasks. It does not prove that an unbounded reasoning/action transcript is safe or generally reliable (**O**, CLM-002).

**Worked Example:**
Trace `READY → PROPOSED → AUTHORIZED → DISPATCHED → OBSERVED → VERIFIED → SUCCEEDED` for a read-only lookup. Then replace the result with a timeout after dispatch: the legal next state is `EFFECT_UNKNOWN`, not automatic success or an assumption that nothing happened.

**Knowledge Check:**
1. Why is a model-produced “done” message only a proposal?
2. Which omitted variables would make the displayed state insufficient for replay or diagnosis?

**Guided Practice:**
Write a transition table with allowed predecessor states, guards, emitted evidence, and terminal reasons. Include malformed proposal, denied authorization, cancellation, and unknown effect.

**Feedback Contract:**
- *Expected Evidence*: Every transition has one deterministic owner and a recorded guard; no terminal success is reachable from an unverified effect.
- *Common Failure*: Treating the transcript as the complete state or allowing the model to choose its own authorization result.
- *Diagnostic Hint*: Can two different external realities produce the same recorded state?
- *Concept to Revisit*: Controller-Owned State.

**Learning Outcome:**
Implement a deterministic loop whose state, transitions, and terminal reason can be replayed and inspected.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 12.2 — Tool and Observation Contracts

**Engineering Question:**
How can a controller distinguish a syntactically valid call, an authorized execution, and a verified external effect?

**Concepts & Definitions:**
A production tool contract includes:

- stable name and version, typed arguments, validation, and error schema;
- caller identity, capability scope, preconditions, and approval policy;
- read-only, reversible, idempotent, compensatable, or irreversible effect class;
- deadline, cancellation, retry and idempotency semantics;
- structured result, effect status, postcondition evidence, and provenance.

**Mechanism Explanation:**
Natural-language descriptions help model selection but are not an execution contract. Toolformer is evidence that a model can learn decisions about whether, when, and how to invoke scoped APIs; it does not supply runtime permission, transaction, timeout, or retry policy (**O**, CLM-003).

Every result becomes an observation envelope containing call ID, tool/version, canonical arguments, timestamps, raw and parsed output, validation result, error class, effect state, truncation, and lineage. Empty, truncated, malformed, stale, or adversarial content must not be observationally equivalent to success (**D**, CLM-009).

**Quantitative Model / Derivation:**
For validation predicates $V_{schema}$, $V_{policy}$, and $V_{pre}$, dispatch is permitted only when
$$D=V_{schema}\land V_{policy}\land V_{pre}.$$
This is an exact controller rule for the declared contract; it says nothing about whether the eventual external effect is correct.

where $V_{schema}$ is true when the arguments match the typed schema, $V_{policy}$ when this caller is authorized for this action now, $V_{pre}$ when the tool's preconditions hold, and $\land$ is logical AND. One false predicate is enough to block dispatch. In the worked example below, the first attempt has $V_{schema}=\text{true}$, $V_{policy}=\text{false}$, so $D=\text{false}$. (Here $D$ is a yes/no dispatch decision; in Lesson 12.3, $D$ denotes the root deadline, a time.)

**Worked Example:**
A proposal requests `transfer(amount=100, account=B)`. The schema passes, but the caller lacks account scope, so policy validation rejects before dispatch. A second authorized attempt times out after dispatch; the envelope records `effect=unknown` and requires a postcondition query before retry.

**Knowledge Check:**
1. Why does valid JSON not imply authorized execution?
2. Which envelope field distinguishes “no response” from “no effect”?

**Independent Practice:**
Define one read-only, one idempotent-write, and one irreversible tool contract. Specify validation, authority, timeout, cancellation, effect evidence, and retry semantics.

**Feedback Contract:**
- *Expected Evidence*: Contract validation precedes dispatch; the observation preserves raw data, parsed data, effect certainty, and postcondition evidence.
- *Common Failure*: Trusting a success string or assuming a timeout means the write did not commit.
- *Diagnostic Hint*: What independent evidence names the external effect?
- *Concept to Revisit*: Proposal–Authorization–Effect Separation.

**Learning Outcome:**
Reject malformed and unauthorized calls before execution and preserve enough evidence to distinguish output text from real effects.

*(Effort: 35m instruction, 25m practice)*

---

### Lesson 12.3 — Budgets, Stops, Retry, and Replanning

**Engineering Question:**
How should an agent stop or recover when different resources, deadlines, and effect-certainty states conflict?

**Concepts & Definitions:**
- **Hard budget**: a maximum on a named resource that the controller can enforce *before* an action is dispatched, because the action's worst-case charge is bounded by something the controller sets or counts.
- **Estimated budget**: a target on a resource whose charge cannot be bounded exactly before dispatch. It is enforced by a conservative reservation plus a backstop (timeout, cancellation, kill), and overruns are reported.
- **Reservation**: an amount of budget set aside for one action at admission time, equal to that action's worst-case charge. It is held until the action's actual charge is known.
- **Semantic stop**: a terminal predicate based on verified task state rather than step count alone.
- **Unknown effect**: dispatch occurred but available evidence cannot establish whether the external commit happened.

**Quantitative Model / Derivation:**
Let the budget vector be

$$
B=(B_{steps},B_{model},B_{tool},B_{tokens},B_{time},B_{cost},B_{repeat},B_{errors}).
$$

Each entry is the limit for one resource: steps, model calls, tool calls, tokens, wall-clock time, cost, repeated actions, and errors. Each is enforced on its own; staying under one limit says nothing about the others.

Checking "consumed < limit" before each step is not enough. Suppose 0.24 of a 0.25 cost budget is spent. The check passes, the next model call costs 0.07, and the episode ends at 0.31. A check on what was *consumed* cannot stop an action whose own cost crosses the limit. The controller therefore keeps a ledger with three numbers per resource $r$ (**D**, CLM-015):

- $B_r$: the limit;
- $C_r$: committed — actual charges already reconciled;
- $R_r$: reserved — holds for actions that are admitted but not yet reconciled.

The invariant is

$$
C_r+R_r\le B_r\quad\text{for every hard resource } r,\text{ at all times.}
$$

**Admission (before dispatch).** For a proposed action $a$, compute its reservation vector $q(a)$, the worst-case charge under caps the controller enforces. Admit only if $C_r+R_r+q_r(a)\le B_r$ for every $r$, and in that case add $q(a)$ to $R$. The comparison and the addition must be one atomic step. Otherwise reject before dispatch with the reason and the failing resource; the controller may then choose a cheaper action, compact the input, or stop.

Apply this to the opening case, assuming the next call's worst case equals its actual 0.07: $C+R+q=0.24+0+0.07=0.31>0.25$, so the call is rejected *before* dispatch and spend stays at 0.24. The consumed-only check ($0.24<0.25$) would have admitted it.

**Reconciliation (after the action).** When the actual charge $u(a)$ is known, set $C\leftarrow C+u(a)$ and $R\leftarrow R-q(a)$. The difference $q(a)-u(a)$ returns to the available budget; that is the refund. If the action is cancelled or times out and the charge is not yet known, the reservation stays held. It is released only against a usage record, or charged in full if none arrives.

Step by step for row 1 of Example B below (cost only), starting from $C=0$, $R=0$, $B=0.25$:
1. Reserve: $q=12{,}000\times3\times10^{-6}+2{,}000\times15\times10^{-6}=0.036+0.030=0.066$.
2. Admit: $0+0+0.066\le0.25$, so $R\leftarrow0.066$.
3. Actual: 700 output tokens, $u=0.036+700\times15\times10^{-6}=0.036+0.0105=0.0465$.
4. Reconcile: $C\leftarrow0+0.0465=0.0465$, $R\leftarrow0.066-0.066=0$; refund $q-u=0.0195$.

Since $u\le q$ always holds for a correct worst-case reservation, $C$ grows by less than $R$ shrinks, so reconciliation can never break the invariant.

**Remaining deadline.** Time cannot be reserved the way cost can; it passes whether or not the action finishes. The root episode has one absolute deadline $D$. Every child call receives a timeout of at most $D-t_{now}-t_{reserve}$, where $t_{reserve}$ is the time kept back for postcondition verification and the final response. A child never receives a fresh timeout of its own.

where $t_{now}$ is the current time on the same clock as $D$. Example: with $D=10$ s, $t_{now}=7$ s, and $t_{reserve}=0.5$ s, the child may wait at most $10-7-0.5=2.5$ s, even if its SDK default is 5 s; the effective timeout is $\min(\text{SDK timeout},\ D-t_{now}-t_{reserve})$. Example A below walks through exactly this case.

**Nested retries** (**D**, CLM-016). If the controller allows $k_c$ attempts of a logical call and the SDK or client underneath retries each of them up to $k_s$ more times, the worst case is $k_c\,(1+k_s)$ physical attempts. The reservation for one logical call must cover $(1+k_s)$ physical attempts, or inner retries must be disabled. Counting only controller-level attempts understates the worst case by the factor $(1+k_s)$.

Why the product: each controller attempt is one original physical attempt plus up to $k_s$ inner retries, so $1+k_s$ physical attempts; the controller repeats that up to $k_c$ times. With $k_c=3$ and $k_s=2$: $3\times(1+2)=9$ physical attempts, while a controller that counts only its own attempts records 3, an undercount by the factor $1+k_s=3$.

**Which limits are hard and which are estimates:**

| Resource | Why the worst case is or is not known before dispatch | Class |
|---|---|---|
| number of model calls, logical tool calls, physical attempts | the controller counts them, provided inner retries are counted or disabled | hard |
| input tokens of a model call | counted on the serialized request with the target tokenizer (Module 11) | hard if counted exactly; estimate if a heuristic counter is used |
| output tokens of a model call | bounded by the maximum-output parameter, if the server enforces it | hard under that condition |
| cost of a model call | (input tokens × input price) + (maximum output tokens × output price), with a pinned price table | hard if every billed token is covered by those two bounds; estimate if the provider bills tokens that no request parameter bounds |
| cost of a tool call | per-attempt price × reserved attempts | hard only if the tool has a per-call price cap; otherwise estimate |
| wall-clock time | a timeout bounds how long the controller *waits*, not how long the remote work continues | estimate with a hard backstop for the wait |
| side effects | a timeout or cancellation does not undo a commit | not a budget; handled by effect verification |

A step cap bounds one dimension; it neither proves success nor prevents a single expensive or harmful step. Semantic outcomes include verified success, explicit failure, safe abstention, escalation, cancellation, and unknown effect. Module 18 uses this ledger for its cost ceiling under attack; durable storage of the ledger across crashes is a Module 14 concern.

**Mechanism Explanation:**
Classify a failed call before choosing a response, because retry and recovery differ by class (**D**, CLM-006):

| Failure class | Typical next decision |
|---|---|
| Schema/validation | repair arguments or stop |
| Authorization/policy | do not retry unchanged; request approved path |
| Transient transport | bounded backoff/retry when effect is known absent |
| Rate limit | respect server signal and remaining deadline/budget |
| Deterministic application error | change plan or input; blind retry is futile |
| Semantic corruption | validate against invariants or independent evidence |
| Timeout/partial or unknown effect | verify postcondition before any retry |
| Permanently unavailable tool | choose a valid alternative, replan, or escalate |

ToolMaze's 2026 benchmark crosses explicit/implicit with transient/permanent perturbations and reports that anomaly recovery remains distinct from happy-path execution in its setup. Treat it as a frontier failure-injection design, not a universal production estimate.

**Worked Example A — one root deadline** (exercise assumptions).
Assume a root deadline of 10 s, a maximum of three tool attempts, and observed attempt durations of 2 s, 3 s, and 4 s with 1 s total backoff. The sequential path consumes the full 10 s ($2+3+4+1=10$); a fourth attempt is illegal even if its local SDK timeout would permit it. If attempt two has unknown effect, verification precedes attempt three.

Now apply the remaining-deadline rule to the unknown-effect case, with 0.5 s backoff before each retry, a 1 s postcondition query, a 5 s SDK default timeout, and $t_{reserve}=0.5$ s:

| Time (s) | Event | Remaining deadline |
|---:|---|---:|
| 0.0–2.0 | attempt 1 fails before dispatch (effect known absent) | 8.0 |
| 2.0–2.5 | backoff | 7.5 |
| 2.5–5.5 | attempt 2 times out after dispatch: `EFFECT_UNKNOWN` | 4.5 |
| 5.5–6.5 | postcondition query: effect absent | 3.5 |
| 6.5–7.0 | backoff | 3.0 |
| 7.0 | attempt 3 dispatched with timeout $\min(5,\ 3.0-0.5)=2.5$ s | — |

Each "Remaining deadline" value is $10$ minus the time at the end of that row (for example $10-5.5=4.5$). Attempt 3 starts at 7.0 s with 3.0 s left; keeping $t_{reserve}=0.5$ s back gives $2.5$ s, which is smaller than the SDK's 5 s, so 2.5 s is used. It is cut at $7.0+2.5=9.5$ s, leaving $10-9.5=0.5$ s.

Result: attempt 3 gets 2.5 s, not the SDK's 5 s. An attempt that needs 4 s is cut at 9.5 s. If it is a write, the state is again `EFFECT_UNKNOWN`, and only 0.5 s remains, which is less than the 1 s query. The episode ends as *escalated with unknown effect*. It does not end as success, and it does not retry. Interpretation: verification consumed time that the original 2 + 3 + 4 + 1 schedule did not budget for, so the same three attempts no longer fit. The deadline, not the attempt count, was the binding limit.

**Worked Example B — reservation ledger with fan-out, rejection, and retries** (synthetic prices and counts, registry CLM-017; not any provider's price list).

Inputs:

- Limits: 6 model calls, 8 physical tool attempts, 0.25 USD.
- Prices: input 3 USD and output 15 USD per million tokens. The search tool costs 0.01 USD per physical attempt and its SDK retries up to 2 times, so one logical search can make 3 attempts. Write tools have inner retries disabled.
- Model-call reservation: exact input tokens × input price + maximum output tokens × output price.

Per-token prices are $3\times10^{-6}$ and $15\times10^{-6}$ USD. So 12,000 input tokens cost $12{,}000\times3\times10^{-6}=0.036$, and a 2,000-token output cap reserves $2{,}000\times15\times10^{-6}=0.030$. The "Actual $u$" column uses the same formula with the real output count. In the Check column, each sum is $C+R+q$ for the resource named (cost unless stated). For example, row 3a: $C=0.0765$, $R=0$, $q=0.210$, sum $0.2865>0.25$.

| # | Action | Reservation $q$ | Check $C+R+q\le B$ | Decision | Actual $u$ | After: $C$ (calls / attempts / USD), $R$ |
|---|---|---|---|---|---|---|
| 1 | model call, 12,000 input, max 2,000 output | 1 call, $0.036+0.030=0.066$ | $0.066\le0.25$ | ADMIT | 700 output: $0.0465$; refund $0.0195$ | 1 / 0 / 0.0465, $R=0$ |
| 2a | search branch 1 | 3 attempts, 0.03 | attempts $0+0+3\le8$ | ADMIT | — | $R$: 3 attempts, 0.03 |
| 2b | search branch 2 (parallel) | 3 attempts, 0.03 | $0+3+3\le8$ | ADMIT | — | $R$: 6 attempts, 0.06 |
| 2c | search branch 3 (parallel) | 3 attempts, 0.03 | $0+6+3=9>8$ | **REJECT before dispatch** (tool attempts) | — | unchanged |
| 2d | branches 1 and 2 finish | — | — | reconcile | 1 attempt (0.01) and 2 attempts (0.02); refunds: 3 attempts, 0.03 | 1 / 3 / 0.0765, $R=0$ |
| 3a | model call, 60,000 input, max 2,000 output | 1 call, $0.180+0.030=0.210$ | $0.0765+0.210=0.2865>0.25$ | **REJECT before dispatch** (cost) | — | unchanged |
| 3b | same call after compaction to 20,000 input | 1 call, $0.060+0.030=0.090$ | $0.1665\le0.25$ | ADMIT | 1,200 output: $0.078$; refund $0.012$ | 2 / 3 / 0.1545, $R=0$ |
| 4 | write tool, 1 attempt, times out after dispatch | 1 attempt | $3+1\le8$ | ADMIT | attempt consumed; effect unknown | 2 / 4 / 0.1545 |
| 5 | postcondition read | 1 attempt | $4+1\le8$ | ADMIT | 1 attempt; write is confirmed | 2 / 5 / 0.1545 |
| 6 | final model call, 21,000 input, max 1,000 output | 1 call, $0.063+0.015=0.078$ | $0.2325\le0.25$ | ADMIT | 400 output: $0.069$; refund $0.009$ | 3 / 5 / 0.2235, $R=0$ |

Reading the trace:

- **Rows 2a–2c need an atomic ledger.** If the three parallel branches each read "0 attempts used" and then each add 3, all three are admitted and the worst case is 9 attempts against a limit of 8. With an atomic compare-and-add (or a single ledger owner), the third sees 6 already reserved and is rejected. After row 2d the refund makes room: a later request for branch 3 would pass ($3+0+3\le8$).
- **Row 3a is the case a "consumed < limit" check misses.** Consumed cost is 0.0765, well under 0.25, yet the action's own worst case would cross the limit. It is refused before any token is sent.
- **Row 4 is a retry decision, not only a budget entry.** The write is not retried. Row 5 verifies first, as in Example A.
- **Cancellation.** Suppose the user cancels while the row 3b call is in flight. The controller admits nothing new. The 0.090 reservation stays held, because the provider may bill tokens already generated. It is reconciled when a usage record arrives and charged in full if none does. A cancelled write keeps the effect state `EFFECT_UNKNOWN` until verified.
- **Nested retries.** With controller-level retries of 3 and the search SDK's 2 inner retries, one logical search could make $3\times(1+2)=9$ physical attempts, more than the whole attempt budget. Reserving 3 per controller attempt keeps that visible.

Limits: the final 0.2235 USD is below 0.25 because every cost in this fixture is bounded by a controller-set parameter. If the provider billed tokens that the maximum-output parameter does not cover, the cost column would be an estimate and the invariant would hold only for the reserved amounts. The ledger in this example lives in memory; surviving a controller crash needs the durable ledger of Module 14.

**Knowledge Check:**
1. Why is retrying a deterministic authorization failure unchanged futile?
2. Why must nested SDK and controller retries share one root budget?
3. After row 2d, the model proposes two parallel searches and one model call with 30,000 input tokens and a 2,000-token output cap. Which are admitted?

**Guided Practice:**
Map each failure-table row to `repair`, `retry`, `verify`, `replan`, `alternative`, `escalate`, or `stop`, including the evidence required to leave `effect=unknown`. Then recompute Example B with the cost limit lowered to 0.20 USD and say where the trace first changes.

**Feedback Contract:**
- *Expected Output*: The decision uses error class, effect certainty, remaining deadline, and all budget dimensions. Knowledge Check 3, admitting in the order listed: after row 2d, $C$ is 1 call, 3 attempts, 0.0765 USD. The two searches reserve 6 attempts and 0.06: $3+6=9>8$, so only the first is admitted (3 attempts, 0.03). The model call reserves $0.090+0.030=0.120$: $0.0765+0.03+0.120=0.2265\le0.25$, admitted. Guided Practice: with a 0.20 limit, rows 1–3b are unchanged ($0.1665\le0.20$ at 3b). Row 6 is the first difference: $0.1545+0.078=0.2325>0.20$, so the final call is rejected before dispatch and the controller must shrink the input, lower the output cap, or stop with a budget terminal reason.
- *Common Failure*: Resetting deadline or attempt count at each layer; checking only consumed budget; refunding a cancelled call before its usage is known.
- *Diagnostic Hint*: Is the next action reducing uncertainty or merely repeating work? For the ledger: at every row, does $C+R$ stay at or below $B$ for each resource, and can two concurrent admissions both read the same $R$?
- *Concept to Revisit*: Root Budget and Effect Certainty; reservation, reconciliation, and refund.

**Learning Outcome:**
Map error class and effect certainty to retry, repair, verify, replan, alternative, escalation, or stop, and admit an action only when its reservation fits every remaining hard budget and the remaining deadline.

*(Effort: 60m instruction, 35m practice)*

---

### Lesson 12.4 — Progress, Cycles, and Reflection

**Engineering Question:**
How can a controller stop genuine no-progress without terminating legitimate polling or iterative refinement?

**Concepts & Definitions:**
Step count is not progress. Instrument verified subgoals, state hashes/deltas, action signatures, repeated error classes, tool-result novelty, plan similarity, and remaining budget. Repeated actions, alternating states, or nearly identical plans are useful stall signals, but legitimate polling and iterative refinement can look similar (**H**, CLM-010).

**Mechanism Explanation:**
Treat online stall detection as a hypothesis. Compare a preregistered detector with a step-cap baseline and report early-stop savings, false stops, recovered success, effect errors, and cost. A detector that merely stops hard tasks sooner may reduce spend while destroying utility.

Reflexion is a reference mechanism that stores verbal feedback for later trials. Reflection is not independent evidence: a fluent explanation can preserve a wrong diagnosis (**O**, CLM-008). Admit a reflection into state or memory only with its source, outcome, confidence, validity window, and evidence; compare self-reflection with no-reflection, external-feedback, and oracle-feedback baselines.

**Quantitative Model / Trade-off Comparison:**
For a detector, report false-stop rate among episodes that the baseline later solves, saved calls among genuinely stalled episodes, and net verified utility after cost. No single count is sufficient because aggressive stopping can improve cost while reducing success.

**Worked Example:**
Compare two traces with action signature `poll(job-7)` repeated four times. In one, the observation version and job progress advance; in the other, the same stale observation repeats. A repetition count alone flags both, while state-delta evidence separates them.

**Knowledge Check:**
1. What evidence distinguishes an oscillation from a valid two-phase protocol?
2. Why is self-reflection not an independent oracle?

**Guided Practice:**
Pre-register a stall detector and compare step-cap, detector-only, self-reflection, external-feedback, and oracle-feedback conditions on paired episodes.

**Feedback Contract:**
- *Expected Evidence*: Report success, false stops, saved work, effect errors, and cost by task/failure slice.
- *Common Failure*: Calling every repeated action a loop or treating reflective prose as verified state.
- *Diagnostic Hint*: Did any externally grounded predicate improve?
- *Concept to Revisit*: Verified Progress.

**Learning Outcome:**
Detect genuine no-progress and use feedback without turning self-critique into truth.

*(Effort: 35m instruction, 30m practice)*

---

### Lesson 12.5 — Authority and Effect Verification

**Engineering Question:**
Where must authorization and postcondition checks sit so that model text cannot widen authority or fabricate completion?

**Concepts & Definitions:**
The model is not an authorization oracle. Enforce least-privilege credentials, tenant and object scope, rate and value limits, preconditions, previews, required approvals, and prohibited transitions outside the prompt. Untrusted observations cannot grant new authority.

**Mechanism Explanation:**
For high-impact actions, separate:

```text
proposed intent -> policy decision -> approved command -> execution
        -> effect receipt -> independent postcondition -> user-visible claim
```

If a timeout occurs after dispatch, the effect may be unknown. Do not equate an absent response with no effect or retry automatically. Module 12 owns the decision to verify, stop, or escalate; Module 14 develops durable idempotency, checkpoint, and compensation mechanics; Module 18 develops adversarial security controls.

**Quantitative Model / Derivation:**
Let $A(c,o,v)$ be the controller policy for caller $c$, object $o$, and proposed value $v$, and let $P(e)$ be an independent postcondition for effect $e$. A success claim is permitted only when dispatch was authorized and $P(e)$ is observed; model confidence is not an input to either predicate.

In symbols: $\text{claim success}\iff A(c,o,v)\land P(e)$, where both are true/false values computed outside the model. Lost response on an authorized deployment: $A=\text{true}$, $P$ not yet observed, so no success claim; after a versioned read-back matches the approved artifact, $P=\text{true}$ and the claim is allowed.

**Worked Example:**
A deployment proposal passes schema validation but exceeds the caller's environment scope. The controller denies it without sending credentials. For an authorized deployment whose response is lost, a versioned read-back must match the approved artifact before the episode can claim success.

**Knowledge Check:**
1. Can an observation containing “admin approved” grant authority?
2. Why is an effect receipt stronger than generated narration yet sometimes still insufficient?

**Independent Practice:**
Design a preview/approve/execute/read-back protocol with an immutable approval binding and test wrong tenant, changed value, expired approval, and stale postcondition.

**Feedback Contract:**
- *Expected Evidence*: Policy decisions and postconditions are external to the model and bound to exact intent/version.
- *Common Failure*: Reusing an approval after arguments change.
- *Diagnostic Hint*: Which controller record binds caller, object, value, and time?
- *Concept to Revisit*: External Authority Boundary.

**Learning Outcome:**
Demonstrate that a model cannot widen authority and cannot claim a side effect without verifiable evidence.

*(Effort: 35m instruction, 30m practice)*

---

### Lesson 12.6 — Trajectory Evaluation and Diagnosis

**Engineering Question:**
Which trajectory-level measurements distinguish a useful recovery mechanism from one that hides failures or amplifies cost?

**Concepts & Definitions:**
AgentBench is reference evidence that interactive environments expose heterogeneous failure modes. A single aggregate score cannot establish general agent capability (**O**, CLM-012).

**Quantitative Model / Derivation:**
For a strictly sequential episode, wall time decomposes into controller, model, validation, tool, observation-processing, queue, and network components. For parallel branches, use critical-path timing; summing overlapping spans overstates elapsed time.

Report by workload and failure slice:

- verified task success, partial completion, abstention, escalation, and rejection;
- side-effect correctness, duplicate/unknown effects, policy violations, and human intervention;
- steps, model/tool calls, retries, tokens, tool/model latency, elapsed time, and cost;
- recovery conditioned on perturbation and on opportunity to recover;
- terminal reason and unfinished episodes.

Success-only scores can reward wasteful or unsafe paths, so these outcomes are reported together (**D**, CLM-013).

Define episode goodput under a declared SLO as verified, policy-compliant successes per wall-clock time. Define cost of success with all included episode spend and an explicit zero-success policy. Never drop aborted, rejected, timed-out, or failed episodes merely because they lack a final answer.

**Mechanism Explanation:**
Join controller, model, validation, tool, effect, queue, and client spans with stable episode/call/effect identities. Slice by workload, failure injection, authorization class, terminal reason, and recovery opportunity before interpreting an aggregate.

**Worked Example:**
In a 60 s observation window, assume 20 offered episodes, 12 verified policy-compliant successes within the SLO, three failures, two abstentions, one rejection, and two unfinished episodes. Declared episode goodput is $12/60=0.2$ successful episodes/s; it is not $12/15$ and unfinished work remains visible.

Check the accounting: $12+3+2+1+2=20$, so every offered episode lands in exactly one class. Goodput divides verified successes by *time* (60 s), giving a rate. Dividing by a subset of episodes instead, such as the 15 that ended in success or failure ($12+3$), gives a ratio ($0.8$) with different units that silently drops abstentions, rejections, and unfinished work. Over all 20 offered episodes the verified success fraction is $12/20=0.6$.

**Knowledge Check:**
1. Why can final-answer accuracy hide duplicate effects?
2. When does summing branch latencies overstate end-to-end time?

**Guided Practice:**
Build a trajectory report that preserves all terminal classes and compare a retry policy on paired injected-failure episodes.

**Feedback Contract:**
- *Expected Evidence*: Offered denominator, terminal reason, effect correctness, opportunity-to-recover, critical-path latency, all-attempt tokens/cost, and confidence procedure are explicit.
- *Common Failure*: Reporting only completed episodes or only the final successful attempt.
- *Diagnostic Hint*: Which missing class would make the policy look better?
- *Concept to Revisit*: Trajectory-Level Goodput.

**Learning Outcome:**
Diagnose `SYMPTOM → COMPETING HYPOTHESES → MISSING EVIDENCE → DISCRIMINATING MEASUREMENT → RANKED EXPLANATION → INTERVENTION → REMEASUREMENT` from full trajectories.

*(Effort: 40m instruction, 35m practice)*

## 05 Literature & Production Source Map

**REFERENCE / BASELINE**

- [ReAct](https://arxiv.org/abs/2210.03629) — Yao et al., ICLR 2023: interleaved reasoning and actions on scoped tasks.
- [Toolformer](https://arxiv.org/abs/2302.04761) — Schick et al., 2023: learned decisions about API use.
- [Reflexion](https://arxiv.org/abs/2303.11366) — Shinn et al., 2023: verbal feedback retained across trials.
- [AgentBench](https://arxiv.org/abs/2308.03688) — Liu et al., ICLR 2024; arXiv revised 2025: multi-environment interactive evaluation.

*Scope:* each paper supports only its evaluated tasks, models, and environments. These four entries were not re-read in this revision and keep their earlier access dates.

**RECOMMENDED BASELINE (course position, not a surveyed industry default):** controller-owned transitions; schema validation before dispatch; least-privilege authorization outside the model; independent call/time/token/cost limits enforced by pre-dispatch reservation; structured errors; trajectory telemetry; explicit terminal reasons. These follow from the derivations in Lessons 12.1–12.3 and 12.5 (**D**, CLM-001, CLM-004, CLM-005, CLM-011, CLM-015). This module has no survey of how many frameworks implement them; registry entries labelled "CURRENT DEFAULT" mean this recommended baseline. The one implementation inspected below enforces a step limit and, in the files read, no cost, token, or time budget.

**WORKLOAD-DEPENDENT:** planner shape, reflection, retry count/backoff, progress metric, verifier, approval placement, degree of autonomy, parallelism, and utility weights.

**FRONTIER:** [ToolMaze](https://arxiv.org/abs/2606.05806) (Zhu et al., arXiv v1, 2026-06-04): a benchmark that crosses DAG-shaped task topology with explicit/implicit and transient/permanent tool perturbations. Its abstract reports that perturbations degrade performance across nearly all evaluated models, most sharply under implicit semantic failures, and that complex topologies lead to futile trial-and-error (**O**, CLM-007). Only the abstract was read on 2026-09-30, so the paper's rates are not quoted and its methods were not audited. Benchmark transfer to real side effects remains open.

**LEGACY / INSUFFICIENT:** parse free-form action text and execute it directly; retry every exception; stop only when the model says “done”; trust reflection as verification; hide failed trajectories; report only final success; use one framework's loop as the definition of agents.

**PRODUCTION SOURCE TRACE**

- Repository: `langchain-ai/langgraph`
- Revision: `7daa3ab49d678a5da75edb08baa87db4a2be52c3`
- Verified: 2026-09-30 (all three files re-read at this revision), static inspection only; nothing was executed.
- Files/symbols: `libs/prebuilt/langgraph/prebuilt/chat_agent_executor.py::create_react_agent`, `libs/langgraph/langgraph/pregel/main.py::Pregel.stream`, and `libs/langgraph/langgraph/errors.py::GraphRecursionError`.
- Entry path: compiled graph invocation/stream → model node → `AIMessage.tool_calls` → tool node → `ToolMessage` observations → model until no tool calls or another terminal path; Pregel stream checks a configurable recursion limit and raises `GraphRecursionError` when exhausted without a stop.
- Observed (**O**, CLM-014):
  - `create_react_agent` carries a `@deprecated` decorator whose message says it has moved to `langchain.agents` and to import `create_agent` from there.
  - The inner `should_continue` returns the end of the graph (or a post-model/structured-response node) when the last message is not an `AIMessage` with tool calls, and routes to the tools node otherwise.
  - `_are_more_steps_needed` reads `remaining_steps` from state and returns true when it is below 2 and the response has tool calls; the model node then returns a fixed "need more steps" message instead of the tool-calling response.
  - In `Pregel.stream`, a loop status of `"out_of_steps"` raises `GraphRecursionError` with a message naming the `recursion_limit` config key. `GraphRecursionError` subclasses `RecursionError`.
- Relation to Lesson 12.3: both guards count *steps*. A text search of `chat_agent_executor.py` for token, cost, or budget handling found none, so the reservation ledger of Lesson 12.3 is something the learner adds around this loop. That is a statement about the file read, not about the whole library or about hooks a user can attach.
- Scope: an implementation example. The recursion limit is a guardrail, not a semantic-success verifier and not a cost bound.
- **Trace practice (the 2h `source_trace` effort, together with the Section 09 artifact):** at the pinned revision, follow one tool-calling turn from `should_continue` to the tools node and back, and find where `"out_of_steps"` is checked. Answer in writing: (1) which condition ends the loop normally, and who produces the value it tests? (2) a model emits one tool call per turn forever; which guard stops it, and what has been spent by then? Expected answers: (1) the last message has no tool calls; that value comes from the model, so normal termination is a model proposal, not a verified outcome; (2) the step/recursion limit; every model and tool call up to that limit has already been dispatched and paid for, because a step count does not bound cost per step. A common error is to call the recursion limit a budget.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — Deterministic Loop and Budget Guard

- **Objective**: Implement explicit states, legal transitions, semantic terminal predicates, and independent budgets; record a replayable trajectory and terminal reason.
- **Pre-Registered Hypothesis**: A state-delta/cycle detector will reduce calls on injected no-progress episodes relative to a step-cap baseline without exceeding a declared false-stop tolerance on solvable episodes.
- **Independent Variables**: Stop policy, task topology, progress signal, polling behavior, and budget vector.
- **Dependent Variables**: Verified success, false-stop rate, calls/tokens/time/cost, terminal reason, and replay agreement.
- **Break & Falsify**: Inject endless calls, alternating plans, identical actions, one expensive step, false `done`, and legitimate polling. A detector that saves work only by stopping recoverable episodes falsifies the claimed benefit. For the budget guard, replay Lesson 12.3 Example B as an executed test: an action whose reservation exceeds the remaining budget must be rejected before dispatch; three concurrent fan-out admissions must not exceed the attempt limit; a cancelled in-flight call must keep its reservation until usage is known; inner SDK retries must be counted. Any run where $C+R>B$ for a hard resource, or where a rejected action was dispatched, falsifies the guard. Prices and counts are synthetic.
- **Alignment**: Lessons 12.1, 12.3, and 12.4.
- **Effort Estimate**: 2.5h implementation, 0.5h analysis (3h total).

### LAB B — Tool Contract and Observation Chaos

- **Objective**: Define versioned read-only, idempotent-write, reversible-write, and irreversible tool contracts with typed observation envelopes.
- **Pre-Registered Hypothesis**: Contract-first dispatch will block malformed/unauthorized calls and preserve unknown-effect states without converting them into false success.
- **Independent Variables**: Effect class, authorization, response integrity, dispatch timing, timeout, and postcondition availability.
- **Dependent Variables**: Pre-dispatch rejection, effect-state classification, unsafe retries, duplicate effects, and evidence completeness.
- **Break & Falsify**: Inject malformed arguments, permission denial, rate limits, timeout before/after dispatch, partial/corrupt/truncated/stale results, and unknown effects. Any unauthorized dispatch or unverified success falsifies the contract.
- **Alignment**: Lessons 12.2 and 12.5.
- **Effort Estimate**: 2.5h implementation, 0.5h analysis (3h total).

### LAB C — Recovery, Replanning, and Reflection

- **Objective**: Implement classified retry, repair, verification, replanning, alternatives, escalation, and reflection.
- **Pre-Registered Hypothesis**: Classified recovery will improve verified success per unit cost over blind retry on recoverable perturbations; self-reflection will not be assumed equivalent to external feedback.
- **Independent Variables**: Failure explicitness/permanence, tool topology, recovery policy, and feedback source.
- **Dependent Variables**: Verified recovery, unsafe effects, attempts, latency, cost, false success, and escalation.
- **Break & Falsify**: Cross explicit/implicit with transient/permanent failures; compare no recovery, blind retry, classified recovery, self-reflection, external feedback, and oracle feedback. Include a workload where extra attempts reduce utility.
- **Alignment**: Lessons 12.3–12.5.
- **Effort Estimate**: 2.5h implementation, 0.5h analysis (3h total).

### LAB D — Trajectory Evaluation Under Load

- **Objective**: Evaluate mixed task lengths, tool latencies, effect classes, concurrency, and injected failures from complete trajectories.
- **Pre-Registered Hypothesis**: Full offered-episode accounting will expose a different recovery ranking than success-only final-answer scoring on at least one preregistered injected-failure slice.
- **Independent Variables**: Concurrency, task length, tool latency, effect class, failure injection, and recovery policy.
- **Dependent Variables**: Task/effect correctness, violations, intervention, recovery, calls/tokens, critical-path latency, cost, goodput, aborts, and unfinished episodes.
- **Break & Falsify**: Remove failed attempts or sum overlapping spans and measure the induced ranking error; a workload with unchanged ranking limits the claim rather than invalidating the experiment.
- **Alignment**: Lesson 12.6 and Incident 12.1.
- **Effort Estimate**: 2.5h implementation, 0.5h analysis (3h total).

## 07 Break / Incident Scenarios

### Incident 12.1 — The Agent Repeats a High-Impact Action

- **Incident Symptoms**: After a release, an agent calls a payment/message/deployment tool, times out, retries, alternates between `verify` and `execute`, exhausts its budget, and reports success. Some effects are duplicated while some recoverable episodes stop early.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: Include timeout before dispatch, commit before lost acknowledgement, missing idempotency/effect receipt, malformed observation, authorization drift, deterministic error mislabeled transient, stale read-back, progress-detector error, model/planner regression, queue delay, and unavailable alternative path.
  2. *Rank Initial Plausibility*: Use the duplicate-effect audit and timeout location without treating either as conclusive.
  3. *Identify Missing Evidence*: Recover call/effect IDs, canonical arguments, authorization decisions, dispatch/receipt times, raw and parsed outputs, postconditions, state deltas, budgets, retries, versions, spans, terminal reasons, and external audit records.
  4. *Design Discriminating Tests*: Reconcile each effect ID before replay, disable automatic retry, and replay captured observations through old/new controllers. State which result falsifies each leading hypothesis.
  5. *Execute Causal Diagnosis*: Reconstruct complete trajectories and rank the earliest supported failing boundary, including interacting mechanisms.
  6. *Prescribe Mitigation and Prevention*: Freeze retries for unknown effects, reconcile affected operations, then add atomic effect identity, postcondition verification, classified recovery, and safe rollout gates as applicable.
  7. *Remeasure*: Predeclare acceptable duplicate/unknown effects, verified success, false stops, latency, and cost; repeat the same fault injections after the change.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Governed Operations Assistant

Design an agent controller that reads operational state, proposes changes, performs approved writes, tolerates changing tool behavior, and stops safely under a shared latency/cost budget. The workload includes read-only investigation, reversible configuration edits, an irreversible notification, concurrent episodes, and injected unknown-effect timeouts.

**Transfer fixture (SYNTHETIC — exercise inputs, not measurements or real prices).** Every submission is run or reasoned against the same fixture, so two reviewers grade the same thing. A submission may add assumptions if it lists them.

*Operational limits per episode:*

| Limit | Value | Class |
|---|---|---|
| Root deadline | 120 s, with 5 s kept back for verification and the final response | estimate with hard wait backstop |
| Model calls | 10 | hard |
| Physical tool attempts (inner retries included) | 16 | hard |
| Cost | 0.40 USD | hard under the price table below |
| Model call size | input ≤ 24,000 tokens (exact count), output cap 1,500 tokens | hard |
| Prices | input 3 USD and output 15 USD per million tokens | pinned for the exercise |
| Concurrency | at most 4 parallel read branches; at most 1 write in flight per target object | hard |

*Tools:*

| Tool | Effect class | Cost per attempt | Inner retries | Timeout | Idempotency at the receiver | Postcondition evidence |
|---|---|---:|---:|---:|---|---|
| `get_config(service, primary)` | read-only | 0 | 2 | 3 s | — | — (a replica read can be up to 10 s stale) |
| `search_logs(query)` | read-only | 0.005 USD | 2 | 8 s | — | result carries a `truncated` flag |
| `apply_config_patch(service, patch, base_version, key)` | reversible write | 0 | 0 | 10 s | key + intent hash + outcome stored atomically; base-version mismatch rejected | success receipt with the new version, or `get_config(primary=true)` returning version and patch hash |
| `rollback_config(service, to_version, key)` | compensating write | 0 | 0 | 10 s | same scheme | same read |
| `send_notification(channel, text, client_msg_id)` | irreversible | 0.002 USD | 0 | 5 s | none | success receipt with a message ID; without a receipt, `list_notifications(client_msg_id)`, which is complete only 60 s after the send |

*Effect-state fixture:* service `checkout` is at config version v12 with `timeout_ms=30000`. The approved change sets `timeout_ms=45000`. The approval record binds service, patch hash `p-7c1`, base version v12, approver, and an expiry at $t=90$ s. After a verified apply, exactly one notification goes to `#ops`.

*Fault script (identical for every submission):*

| ID | Fault | Seed A | Seed B |
|---|---|---|---|
| F1 | first physical attempt of the first `search_logs` returns 503 before dispatch | yes | yes |
| F2 | one `search_logs` result arrives with `truncated=true` | yes | yes |
| F3 | first `apply_config_patch` times out after dispatch | patch **committed** | patch **not committed** |
| F4 | replica `get_config` returns v12 for 10 s after any commit | yes | yes |
| F5 | first `send_notification` times out after dispatch | message **delivered** | message **not delivered** |
| F6 | (seed C only) the user cancels 2 s after the first `apply_config_patch` is dispatched; the patch committed | — | — |

*Success predicate.* An episode passes only if all of the following hold:

1. Final primary config is v13 with patch hash `p-7c1` (seeds A and B), or the episode ends in a declared non-success terminal state with the true config version recorded (seed C: cancelled, effect verified as committed, rollback either executed with its own approval or handed to a human).
2. The notification log holds at most one message for the `client_msg_id`, and exactly one if the episode reports success.
3. No success is reported for an effect that was not verified through a receipt or the listed postcondition read. Ending as *escalated with unknown notification effect* is acceptable; reporting success without verification is not.
4. $C+R\le B$ for every hard resource at every ledger row, no action was dispatched after a pre-dispatch rejection, and no child call received a timeout beyond the remaining deadline minus 5 s.
5. No write was dispatched without a matching unexpired approval, and none after $t=90$ s under the config approval.
6. The trajectory record contains every attempt, including failed and cancelled ones, with a terminal reason.

**Required Deliverables**:
1. State/transition schemas and explicit owners for proposal, authorization, execution, effect, observation, progress, and termination, including the legal successors of `EFFECT_UNKNOWN`.
2. Versioned tool/observation contracts for the five fixture tools with effect classes, deadlines, cancellation, and postconditions.
3. Authority/approval policy plus immutable binding from preview to approved command, applied to the fixture approval record and its expiry.
4. Budget ledger design (limits, reservation rule per tool and per model call, atomic admission, reconciliation, cancellation handling, hard-versus-estimate classification), semantic stops, and failure-to-recovery decision table for F1–F6.
5. Ledger and decision trace for seeds A, B, and C: each row with action, reservation, admission decision, actual charge, $C$ and $R$, remaining deadline, effect state, and terminal reason.
6. Stall/cycle detector and paired reflection experiment with false-stop analysis.
7. Anomaly matrix, complete trajectory metrics, and load/fault results.
8. Release, kill, rollback, and reconciliation plan, plus evidence-backed diagnosis and remediation of Incident 12.1.
9. Production source trace (the Section 09 Required Artifact).

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit a pinned trace of a production agent-loop implementation. For the reference LangGraph revision, map compiled invocation through model and tool nodes, observation return, stop behavior, and recursion-limit failure. Separate static source observation from controller requirements and note the inspected API's deprecation status.

The trace must record, for each finding: repository, exact commit, file, symbol, entry path, what was observed, whether the code was executed or only read, and whether the finding generalizes beyond this implementation.

### Reference Checks for Mastery Deliverables 4–5 (fixture inputs only)

Reviewers use these to check arithmetic and decisions. Timelines differ between submissions; the decisions below do not.

- **Reservation arithmetic:** the worst-case model call reserves $24{,}000\times3\times10^{-6}+1{,}500\times15\times10^{-6}=0.072+0.0225=0.0945$ USD. Four such calls reserve 0.378 USD; a fifth would bring the total to 0.4725 and is rejected ($0.378+0.0945=0.4725>0.40$). The 10-call limit is therefore reachable only with smaller exact inputs, and the ledger must use the exact input count per call. One logical `search_logs` reserves 3 physical attempts and 0.015 USD. Four parallel searches reserve 12 of the 16 attempts and 0.06 USD. (One search: $1+2=3$ attempts, $3\times0.005=0.015$ USD; four searches: $4\times3=12$ attempts, $4\times0.015=0.06$ USD.)
- **Deadline:** a child call dispatched at time $t$ gets a timeout of at most $\min(\text{tool timeout},\ 120-t-5)$ s. (This is Lesson 12.3's $D-t_{now}-t_{reserve}$ with $D=120$, $t_{reserve}=5$. Example: `search_logs` dispatched at $t=110$ s gets $\min(8,\ 5)=5$ s.)
- **F1:** a 503 before dispatch means the effect is known absent; the inner retry is allowed. Two attempts are consumed and one is refunded.
- **F2:** a truncated result is not complete evidence. Accepted responses are a narrower query or recording the evidence as partial. Treating it as complete is an observation-contract failure.
- **F3 and F4:** after the timeout the state is `EFFECT_UNKNOWN` and the next action is a *primary* read. Seed A: v13 with `p-7c1`, so the apply is verified and not repeated. Seed B: v12, so the effect is known absent, and one retry with the same key and base v12 is legal while $t<90$ s. A submission that verifies through the replica sees v12 in seed A and retries; the receiver's stored outcome prevents a duplicate effect, but the trace must show the wasted attempt, and the choice of read is marked as a contract error.
- **F5:** no receipt means `EFFECT_UNKNOWN`. Resending before the 60 s listing window has passed duplicates the message in seed A and fails predicate 2. Accepted paths: wait, list, then report success (seed A) or resend once with the same `client_msg_id` and take the receipt (seed B); or, if the wait does not fit in the remaining deadline, end as escalated with unknown notification effect.
- **F6:** on cancellation the controller admits no new work except recovery reads, keeps the in-flight reservation, verifies through the primary (v13, committed), and ends as cancelled with a committed effect. A rollback needs its own approval.
- **Ledger rows:** in every seed, each row satisfies $C+R\le B$ for calls, attempts, and cost.

### Rubric Dimensions

- **Control Model**: *Insufficient* relies on transcript/model intent. *Competent* defines legal states, transitions, owners, and terminal reasons. *Strong* proves replay/guard invariants under injected faults.
- **Contracts and Authority**: *Insufficient* validates syntax only, or verifies a write through a read that can be stale. *Competent* enforces typed contracts and external authorization, and uses the listed postcondition for each fixture tool. *Strong* demonstrates preview binding, unknown-effect handling, and independent postconditions under all three seeds.
- **Bounds and Recovery**: *Insufficient* uses one step cap, checks only consumed budget, or retries every error. *Competent* enforces independent budgets by pre-dispatch reservation with atomic admission, propagates the remaining deadline, and gives the reference decisions for F1–F6. *Strong* measures attempt amplification, false recovery, and deadline propagation, and states which fixture limits are hard and which are estimates.
- **Progress and Reflection**: *Insufficient* treats repetition or self-critique as truth. *Competent* evaluates grounded progress and reflection baselines. *Strong* quantifies false stops, saved work, and slice-dependent utility.
- **Evaluation and Diagnosis**: *Insufficient* scores final answers only. *Competent* preserves complete trajectories and competing hypotheses. *Strong* uses discriminating tests, external effect evidence, and remeasurement.

## 10 Capability Traceability Matrix

| Capability | Taught | Practiced | Assessed | Evidence |
|---|---|---|---|---|
| Bounded state-transition controller | Lesson 12.1 | Lesson 12.1 Guided Practice (transition table); LAB A | Mastery Deliverable 1; success predicate 6; Control Model rubric row | Transition table with guards; replayable trajectory with terminal reason |
| Tool and observation contracts | Lesson 12.2 | Lesson 12.2 Independent Practice; LAB B | Mastery Deliverable 2; reference checks F2 and F4; Incident 12.1 steps 1 and 3; Contracts and Authority rubric row | Five fixture tool contracts; observation envelopes; chaos results |
| Budget reservation and deadline control | Lesson 12.3 ledger model and Examples A–B | Lesson 12.3 Knowledge Check 3 and Guided Practice (0.20 USD limit); LAB A budget-guard tests | Mastery Deliverables 4–5; success predicate 4; reference checks for reservation arithmetic and deadline; Bounds and Recovery rubric row | Ledger trace per seed with reservation, decision, actual, $C$, $R$, remaining deadline |
| Classified recovery and no-progress detection | Lessons 12.3–12.4 | Lesson 12.3 Guided Practice (failure table); Lesson 12.4 Guided Practice; LAB A, LAB C | Mastery Deliverables 4 and 6; reference checks F1, F3, F5; Incident 12.1 steps 1–4; Progress and Reflection rubric row | Failure-to-recovery table; false-stop report |
| Authority and effect verification | Lesson 12.5 | Lesson 12.5 Independent Practice; LAB B | Mastery Deliverable 3; success predicates 1–3 and 5; reference checks F3–F6; Incident 12.1 steps 3–6 | Approval binding record; postcondition evidence per write |
| Trajectory evaluation and diagnosis | Lesson 12.6 | Lesson 12.6 Guided Practice; LAB D | Mastery Deliverables 7–8; Incident 12.1 steps 1–7; Evaluation and Diagnosis rubric row | Offered-episode report by terminal class; diagnosis with discriminating tests |
| Production source trace | Section 05 LangGraph trace | Section 05 Trace practice, questions 1–2 | Mastery Deliverable 9 = Section 09 Required Artifact | Pinned trace: repository, commit, file, symbol, entry path, read-versus-executed, generalizability |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner successfully completing Module 12 must be able to:
1. Keep proposal, authorization, execution, effect, observation, progress, and termination distinct.
2. Bound every episode across calls, tokens, time, cost, repetition, errors, and semantic stops, rejecting before dispatch any action whose reservation does not fit the remaining hard budget or deadline, and saying which limits are hard and which are estimates.
3. Refuse blind retry after unknown effects and prove completion with independent evidence.
4. Falsify a progress detector and reflection mechanism without hiding false stops.
5. Trace a pinned runtime without generalizing one framework into the agent definition.
6. Evaluate complete offered trajectories including harms, failures, abstentions, unfinished work, latency, and cost.

### Module Wrap-Up (Final Mental Model Reconstruction)

- **The Core Invariant**: An agent loop is a fallible, resource-bounded control system around a stochastic proposer.
- **The Control Path**: `state → proposal → validation/authorization → execution → observation/effect verification → progress decision → terminal state or next bounded step`.
- Reliability comes from typed transitions, external authority, observable effects, classified recovery, explicit stops, and trajectory-level falsification—not from letting the model continue until its prose sounds complete.

### Cost & Telemetry Hook

- [Lesson 22.1](../22-ai-economics/README.md): cost per verified task success — failed and step-limit-terminated trajectories stay in the cost numerator.
- [Lesson 22.3](../22-ai-economics/README.md): the pre-dispatch reservation ledger is the runtime cost record; reconcile it against billed usage.
- [Lesson 23.1](../23-observability-reliability/README.md) and [Lesson 23.3](../23-observability-reliability/README.md): emit one span per step (proposal, validation, tool call, observation) plus the terminal reason; SLI is the offered task outcome, not loop completion.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
