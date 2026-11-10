# Module 14 — Durable Agent Runtime

## 00 Why This Module Exists

An agent loop that works inside one process is not durable. Workers crash, deployments replace code, queues redeliver work, acknowledgements disappear, humans answer days later, and an external side effect may commit while the runtime still records “unknown.” A durable runtime must recover orchestration state without replaying effects blindly.

```text
workflow input
     |
durable history/checkpoint <---- timers / signals / approvals
     |
deterministic orchestration
     |
activity command + logical effect ID
     |
external system <---- idempotency / fencing / reconciliation
     |
effect receipt or unknown state
     |
history transition / retry / compensate / escalate
```

This module owns workflow history/checkpoints, deterministic replay, activities/effect boundaries, delivery semantics, idempotency, transactional outbox/inbox, sagas and compensation, leases/fencing, fan-out/join, durable waits and HITL, history growth, versioning, and crash recovery. Module 12 owns agent decisions and tool policy; Module 13 owns model-I/O adapters; Module 18 owns threat modeling; Module 23 owns platform-wide reliability operations.

**Research cutoff:** 2026-09-26 for the original claim set. Entries marked *opened 2026-10-01* in Section 05 were read on that date for registry revision 1.1.0; where a publication or revision date is shown, it is on or before 2026-09-30. Claims not marked that way were not re-verified.

**Module Orientation**
- **Engineering Problem**: Recover long-running work to a correct, explainable state without losing, duplicating, or falsely “rolling back” externally visible effects.
- **What You Will Do**: Build durable history and effect boundaries, crash every commit window, implement idempotency/compensation/fencing, exercise durable joins and approvals, replay old histories, trace Temporal, and defend a recovery architecture.
- **Environment**: Python 3.10+, a durable workflow runtime or deterministic local simulator, transactional storage, disposable external-effect mocks, and fault injection around persistence and acknowledgement boundaries.
- **Evidence Rule**: Distinguish source observations (**O**), derivations (**D**), and telemetry-dependent hypotheses (**H**). Runtime marketing terms never replace an explicit persistence and effect contract.
- **If You Only Have One Week**: Lessons 14.1, 14.2, 14.3 — they cover durable state and replay, idempotent effect boundaries, and honest compensation with fencing; joins, timers, recovery telemetry, and versioning build on those three.

## 01 Baseline Assumptions

- Module 00: invariants, experiments, uncertainty, and falsification.
- Module 04: queues, retries as offered load, saturation, and latency distributions.
- Module 08: identity, lineage, versioning, and immutable event evidence.
- Module 12: state transitions, tool contracts, authorization, retries, and terminal states.
- Module 13: canonical requests, attempt lineage, errors, deadlines, and replay manifests.

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
  instruction: 4h       # sum of lesson instruction lines: 40+50+40+45+30+35 min
  guided_practice: 3h   # sum of lesson practice lines: 30+35+30+35+25+25 min
  labs: 12h             # LAB A-D, 3h each
  assessment: 3h        # Mastery transfer problem 2.5h + Incident 14.1 0.5h
  source_trace: 2h      # Section 09 Production Source Trace artifact
  total: 24h
```
Each category is counted once. The source trace is not also counted inside Lesson 14.6 or LAB D, and lab analysis is not counted again as lesson practice.

The learner must be able to state what is durably committed; rebuild state from history; keep nondeterminism and I/O behind recorded boundaries; reason about at-most/at-least/effectively-once behavior; implement atomic idempotency and reconciliation; design honest compensation; fence stale workers; persist joins/timers/approvals; replay-test upgrades; control history growth; and crash the system at every effect boundary.

### LAYER 1: KNOWLEDGE / INSTRUCTIONAL LAYER

## 03 Knowledge Map

```text
                workflow/run identity + code version
                              |
                    persisted ordered history
                              |
                  deterministic state rebuild
                              |
                      command/activity intent
                              |
        +---------------- effect boundary ----------------+
        | operation key | attempt | auth | deadline       |
        +-------------------------------------------------+
                              |
                 external effect + receipt/postcondition
                              |
         committed | absent | failed | partial | unknown
                              |
             continue | retry | compensate | escalate
```

Keep four identities separate: workflow/run, logical transition, activity attempt, and external effect. A retry creates a new attempt but should not invent a new logical effect.

## 04 Lessons

### Lesson 14.1 — Durable State and Deterministic Replay

**Engineering Question:**
What must be durably acknowledged so replay reconstructs the same orchestration decisions without pretending to reconstruct external reality?

**Concepts & Definitions:**
- **Durable prefix**: ordered history/checkpoint acknowledged by the persistence contract.
- **Deterministic replay**: rebuilding state and commands from recorded inputs.
- **Replay divergence**: candidate code emits commands inconsistent with retained history.

**Mechanism Explanation:**

For ordered persisted events `e_1...e_n` and a deterministic reducer `F`:

$$
s_n=\operatorname{fold}(F,s_0,[e_1,\ldots,e_n]).
$$

where:
- $s_0$ is the initial (empty) workflow state, and $s_n$ the state after the first $n$ events;
- $e_i$ is the $i$-th persisted event, in commit order;
- $F(s,e)$ is the reducer: it takes a state and one event and returns the next state, using nothing except its two arguments;
- $\operatorname{fold}$ applies $F$ left to right over the list.

Step-by-step, the fold is just repeated application:

$$
s_1=F(s_0,e_1),\quad s_2=F(s_1,e_2),\quad\ldots,\quad s_n=F(s_{n-1},e_n)
\;\;\Longrightarrow\;\; s_n=F(\cdots F(F(s_0,e_1),e_2)\cdots,e_n).
$$

Two consequences follow directly. (1) Because $F$ reads only its arguments, the same events always give the same $s_n$; a reducer that also reads a clock or a random number has a hidden third argument, and replay can then disagree with the original run. (2) Because $s_n$ depends on $s_{k}$ and the suffix only, a stored state $s_k$ (a checkpoint/snapshot) plus the events $e_{k+1},\ldots,e_n$ gives the same $s_n$ as a full replay from $s_0$. Guided Practice (a) below works this fold on a four-event history.

This reconstructs only state represented by the retained history and schemas (**D**, CLM-001). It does not reconstruct an external payment, message, deployment, or model response unless that observation/effect evidence was recorded.

Replay runtimes compare commands produced by current workflow code with recorded history. Temporal and Azure Durable Functions both document deterministic orchestrator constraints (**O**, CLM-002); that is two runtime families, not every durable engine. Wall clocks, unrecorded random values, mutable environment/configuration, direct network/database/model calls, unordered iteration, and command-changing code edits can diverge.

Declare the persistence acknowledgement: when may a client believe a workflow start, transition, signal, approval, or cancellation is durable? Crash immediately before and after that point.

**Quantitative Model / Derivation:**
For retained events, $s_n=fold(F,s_0,[e_1,\ldots,e_n])$. This is exact only for state represented by the history and schemas; external effects require separate evidence.

**Worked Example (synthetic fixture):**

*Input.* A history store commits each append atomically and sends the acknowledgement only after the commit. It de-duplicates workflow starts by workflow ID and signals by signal ID (a fixture assumption; real runtimes have their own ID-reuse rules). A client starts `wf-42` and later sends signal `approve-1`. A crash is injected on each side of each commit.

| Case | Crash point | Durable prefix after restart | What the client saw | Allowed client action | State after recovery |
|---|---|---|---|---|---|
| S1 | before the `Started` append commits | empty | error, no acknowledgement | retry the start with workflow ID `wf-42` | one run, `[e1 Started]` |
| S2 | after the `Started` append commits, before the acknowledgement is sent | `[e1 Started]` | the same error as S1 | the same retry; the store answers "already started" and returns the existing run | one run, `[e1 Started]` |
| T1 | before `SignalReceived(approve-1)` commits | `[e1, e2, e3]` | error, no acknowledgement | resend with signal ID `approve-1` | `[e1, e2, e3, e4]` |
| T2 | after it commits, before the acknowledgement is sent | `[e1, e2, e3, e4]` | the same error as T1 | the same resend; dropped as a duplicate of `approve-1` | `[e1, e2, e3, e4]`, one `e4` |

*Steps.*
1. Compare the "what the client saw" column: S1 and S2 are identical to the client, and so are T1 and T2.
2. Compare the durable prefix: it differs within each pair. The difference is visible only to the store.
3. Therefore the client's action must be correct in both members of a pair, and the store must make the retry converge.

In plain terms: an acknowledgement travels *after* the commit, so a crash can fall between them. The client then sees a failure for something that is already durable, and the only safe client behavior is "retry with the same ID and let the store recognize it".

*Result.* After recovery every case has exactly one `Started` event and at most one `SignalReceived(approve-1)`. A client may treat the operation as durable only after an acknowledgement. Without one it knows nothing, and "no acknowledgement" must not be read as "did not happen".

*Interpretation and limits.* S2 and T2 are commit-without-acknowledgement at the history layer. Lesson 14.2 meets the same shape at an external receiver, where the history store no longer holds the truth. The table depends on the fixture's de-duplication assumption (**D**, CLM-001); without it, S2 would create a second run.

**Knowledge Check:**
1. Why can unrecorded wall time or randomness break replay?
2. Does replay prove an external payment exists?

**Guided Practice:**
(a) History: `e1 Started`, `e2 StepScheduled(1)`, `e3 StepCompleted(1)`, `e4 StepScheduled(2)`. The reducer sets `status=running` on `Started`, `pending=n` on `StepScheduled(n)`, and `step=n` (clearing `pending`) on `StepCompleted(n)`. Give the state after each prefix. Then replay under a revision that starts a timer before scheduling step 2 and name the first divergence. (b) Build your own four-event history, replay it under unchanged code, inject a command-reordering revision, and record the first divergence.

**Feedback Contract:**
- *Expected Evidence*: (a) After 0–4 events: `{}`; `{running}`; `{running, pending 1}`; `{running, step 1}`; `{running, step 1, pending 2}`. The revision diverges at `e4`: history recorded `StepScheduled(2)`, the candidate emits `StartTimer`. Events `e1`–`e3` replay cleanly, so the divergence is at the first command the change affects, not at the start. (b) Acknowledgement boundary, retained inputs, reducer/command trace, and external-state exclusions.
- *Common Failure*: Calling a chat transcript a complete checkpoint, or concluding that step 1's external effect "happened" because `e3` replays.
- *Diagnostic Hint*: Which nondeterministic value was not recorded? Which recorded command does the candidate emit differently?
- *Concept to Revisit*: Durable Prefix.

**Learning Outcome:**
Rebuild the same workflow state and command sequence from a durable prefix while identifying all state that lives outside it.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 14.2 — Activity Boundaries, Delivery, and Idempotency

**Engineering Question:**
How should recovery act when an external commit may have succeeded but durable acknowledgement was lost?

**Concepts & Definitions:**
- **Activity/effect boundary**: recorded orchestration command around nondeterministic I/O.
- **Logical effect ID**: stable identity shared by delivery attempts.
- **Atomic idempotency**: receiver stores identity, canonical intent, and outcome with the effect.

**Mechanism Explanation:**

Put nondeterministic external I/O behind activities/effects (**O**, CLM-003). Record canonical intent, logical-operation key, attempt, authorization, dispatch, timeout/cancellation, response, effect state, receipt/postcondition, and reconciliation.

The critical ambiguity is:

```text
external commit succeeds
        |
worker/runtime crashes before durable acknowledgement
        |
recovery sees pending/timeout, while external state may already be changed
```

Replay cannot solve this (**D**, CLM-004). If the external system cannot participate in the same atomic transaction, safe recovery needs a stable operation identity and atomic deduplication or a trustworthy postcondition query. For protected state `x` and operation key `k`, the required idempotency property is:

$$
apply(apply(x,k),k)=apply(x,k).
$$

where $x$ is the receiver's protected state before the operation, $k$ is the operation key (always sent with the same canonical intent), and $apply(x,k)$ is the state after the receiver processes one delivery of that keyed operation. Read left to right: delivering the operation a second time must leave the state exactly where the first delivery left it. Numeric check with the debit used below ($x$ = balance 100, $k$ = debit 30): $apply(100,k)=70$; a second delivery must return the stored outcome, so $apply(70,k)=70$, not $40$. A plain `balance -= 30` without the key record gives $40$ and fails the property.

Test it under concurrent duplicates and key reuse with different intent. A header/key is not enough unless the receiver stores key + canonical intent + outcome atomically and defines scope and retention (**D**, CLM-005).

**Receiver-side idempotency contract.** This is the contract other modules reference (Module 10, Lesson 10.2 applies it to index writes). The receiver, not the sender, enforces it:

1. **Key.** `(scope, key)` names one logical effect. Every attempt of that effect reuses it. The key is derived from workflow identity and the logical step, never from an attempt number, a clock, or a random value. The scope keeps tenants and operation types apart.
2. **Atomic check-and-record.** The key record, the protected effect, and the stored outcome commit in one transaction or one conditional write. "Read whether the key exists, then write" in two steps is not this.
3. **Conflict.** The same key with a different canonical intent is rejected and changes nothing. It is neither applied nor answered with the old outcome.
4. **Replay.** The same key with the same intent returns the stored outcome and does not apply the effect again.
5. **Retention.** Key records outlive the longest retry and redelivery window. After a record is purged, the key is new again, so a receiver that purges keys needs a second guard that does not expire with them: a version or base-state check on the protected record (the acceptance rule of Module 10, Lesson 10.2), or a ledger/postcondition lookup. An absolute write such as "set version 7 to this content" can be guarded that way; a relative effect such as "debit 30" cannot, and must be reconciled.

```text
apply(scope, key, intent, epoch):
  h = hash(canonical(intent))
  BEGIN                                    -- one atomic transaction at the receiver
    if epoch < resource.last_epoch:        -- fencing check, Lesson 14.3
        ROLLBACK; return STALE_EPOCH
    INSERT idem(scope, key, intent_hash=h) -- UNIQUE(scope, key): check and record in one step
    if a row for (scope, key) already exists:
        ROLLBACK
        if stored.intent_hash != h: return KEY_CONFLICT   -- same key, different payload
        return REPLAYED(stored.outcome)                   -- same key, same payload
    account.balance -= intent.amount       -- the protected effect
    idem.outcome = receipt                 -- outcome stored with the effect
    resource.last_epoch = epoch
  COMMIT
  return APPLIED(receipt)
```

The epoch comparison is `<`, so a write carrying the current epoch is accepted; Lesson 14.3 explains why equality must pass. The unique constraint is what serializes two concurrent attempts: the second insert waits for, or conflicts with, the first, and can never observe "absent" after the first has committed. The transaction is a derivation (**D**, CLM-019). Published key contracts have the same parts: an expired IETF draft defines a request fingerprint and distinct error responses for a reused key with a different payload and for a retry that arrives while the original is still processing, and one payment API documents storing the first result per key, comparing parameters on reuse, and pruning keys after a retention period (**O**, CLM-017). Those are two documents, not a standard every endpoint follows.

A transactional outbox stores domain change and publish intent in one local transaction. The relay can still publish twice after a crash; consumers need deduplication/inbox semantics, which is the receiver contract above applied to messages (**D**, CLM-006). Track outbox lag, poison rows, ordering, and retention.

**Quantitative Model / Derivation:**
The required idempotency property is $apply(apply(x,k),k)=apply(x,k)$ for the same key and canonical intent. A receiver must reject key reuse with different intent.

*Delivery vocabulary used in this lesson.* Count, for one logical effect, how many times it is *delivered* (sent to the receiver) and how many times it is *applied* (changes the receiver's state):
- **At-most-once delivery**: send once, never retry. Applied 0 or 1 times; a lost message is lost.
- **At-least-once delivery**: retry until an outcome is recorded. Delivered 1 or more times; without receiver deduplication it may be applied more than once.
- **Effect convergence** ("effectively once"): at-least-once delivery combined with a receiver that applies each key at most once. Delivered $\ge 1$ times, applied exactly once, but only for receivers that implement the contract and only while the key record is retained.

A timeout only means the caller stopped waiting; it says nothing about whether the receiver committed. That is why the recovery below asks the receiver (by key) instead of inferring the outcome from the timeout.

**Worked Example (synthetic fixture):**

*Input.* Workflow `wf-42` schedules one debit. Key `k = wf-42/charge/1`, scope `tenant-1/charge`, canonical intent `{"account":"A","amount":30}` with intent hash `h1`. Account A starts at 100. History events are `e1 Started`, `e2 ActivityScheduled(k, h1)`, and `e3 ActivityCompleted(k, receipt)`. The receiver runs the transaction above.

| Case | What happened | Durable prefix | Receiver state before recovery | Allowed recovery | Result |
|---|---|---|---|---|---|
| W1 | worker crashes before dispatch | `[e1, e2]` | no row for `k`; balance 100 | dispatch with `(k, h1)` | `APPLIED r-1`; balance 70 |
| W2 | dispatched; receiver crashes before its commit | `[e1, e2]` | no row (rolled back); balance 100 | dispatch with `(k, h1)` | `APPLIED r-1`; balance 70 |
| W3 commit without acknowledgement | receiver commits; the reply is lost or the worker crashes before `e3` | `[e1, e2]` | row `(k, h1, r-1)`; balance 70 | dispatch with `(k, h1)`, or query by `k` | `REPLAYED r-1`; balance stays 70; append `e3` |
| W4 | crash after `e3` is appended | `[e1, e2, e3]` | row `(k, h1, r-1)`; balance 70 | replay reads `e3`; no dispatch | balance 70 |
| C1 concurrent duplicate | a timeout fires a second attempt with `(k, h1)` while the first is still in flight | `[e1, e2]` | at most one row | none needed; either response carries `r-1` | one `APPLIED`, one `REPLAYED`; one row; balance 70 |
| C2 key conflict | an attempt arrives with `k` but amount 50 (hash `h2`) after `r-1` | `[e1, e2]` or later | row `(k, h1, r-1)`; balance 70 | do not retry; escalate, because key derivation or intent is not deterministic | `KEY_CONFLICT`; balance 70, unchanged |
| C3 stale worker | a worker holding epoch 7 pauses; the owner with epoch 8 has already written; the stale worker resumes and sends a write | owned by epoch 8 | `last_epoch` 8 | the stale worker stops and discards its local state; the epoch-8 owner continues from history | `STALE_EPOCH`; nothing changes |
| R1 retention expiry | as W3, but the key row was purged before the retry | `[e1, e2]` | no row; balance 70 | the key no longer protects; fall back to a guard that did not expire (ledger or postcondition lookup here; a version or base check where the write is absolute), or escalate | a blind retry returns `APPLIED r-2`; balance 40 |

*Steps.*
1. Group by durable prefix. W1, W2, W3, C1, and R1 all show `[e1, e2]`. History alone cannot tell them apart; only W4 is resolved by history.
2. Because the prefix is the same, the recovery action must be safe in every one of those worlds. Dispatching with the same `(k, h1)` is safe in W1–W3 and C1, since the receiver decides. It is unsafe in R1, which is why retention is part of the contract.
3. Check the forbidden actions against W3: a retry under a *new* key applies a second debit (balance 40), and "timeout means not applied" skips an effect that exists.
4. C1 under a check-then-act receiver: both attempts read "absent", both debit, balance 40. Under the atomic transaction: balance 70.

*Result.* With the atomic receiver, account A ends at 70 with one receipt in W1–W4 and C1. C2 and C3 are rejected without a state change. R1 is the one case where the same recovery action duplicates the effect.

*Interpretation and limits.* The cases need different recovery because different parties hold the deciding evidence: the history in W4, the receiver's key table in W1–W3 and C1, the key's intent in C2, the ownership epoch in C3, and nothing at all in R1. A checkpoint or replay gives *at-least-once dispatch of a recorded decision*. Only the receiver's transaction gives *at-most-once application per key, within scope and retention*. The combination converges to one effect for this receiver; it is not exactly-once execution, and it says nothing about a receiver that does not implement the contract (**D**, CLM-004, CLM-019).

**Knowledge Check:**
1. Why is an idempotency header alone insufficient?
2. Why can an outbox relay still publish twice?

**Guided Practice:**
(a) A receiver holds row `(k9, hX, r-7)` for a debit of 5 and the balance is 55. Give the response and the balance for each independent request: (i) `(k9, hX)`; (ii) `(k9, hY)`; (iii) `(k10, hX)`. (b) Race two concurrent attempts with the same key, then reuse the key with different intent; verify one stored outcome and one conflict.

**Feedback Contract:**
- *Expected Evidence*: (a) (i) `REPLAYED r-7`, balance 55. (ii) `KEY_CONFLICT`, balance 55. (iii) `APPLIED` with a new receipt, balance 50: the same payload under a new key is a new effect, which is why a retry must never mint a new key. (b) Stable logical ID, attempt IDs, the atomic receiver record, receipt/postcondition, and a reconciliation path for the retention-expiry case.
- *Common Failure*: Check-then-act deduplication; retry under a fresh key after an unknown commit; answering a key conflict with the stored outcome.
- *Diagnostic Hint*: Which system owns the atomic decision, and which cases share the same durable prefix?
- *Concept to Revisit*: Effect Convergence.

**Learning Outcome:**
Distinguish at-most-once, at-least-once, and effect convergence without claiming universal exactly-once execution.

*(Effort: 50m instruction, 35m practice)*

---

### Lesson 14.3 — Compensation and Concurrent Ownership

**Engineering Question:**
When can compensation restore a business invariant, and how does fencing prevent stale owners from creating later effects?

**Concepts & Definitions:**
- **Compensation**: authorized forward action responding to an earlier effect.
- **Lease**: time-bounded ownership hint.
- **Fencing token**: monotonically increasing epoch enforced by the protected resource.

**Mechanism Explanation:**

The 1987 Sagas paper is the reference model: decompose long-lived work into subtransactions with compensating transactions (**O**, CLM-007). In production, compensation is another forward action (**D**, CLM-008). It has authorization, retries, failures, cost, and evidence; it may restore a business invariant but cannot unread a message, retract a disclosure, erase an observed deployment, or reverse every real-world consequence.

Register compensation before or atomically with the effect intent, preserve dependency order, and expose states such as `compensation_pending`, `compensation_failed`, `partially_compensated`, and `manual_recovery`.

Leases alone do not revoke a paused stale worker. A lease is a promise made by the lease service; a paused worker cannot see that its lease expired, so on resume it still acts as owner. Only a check at the resource that receives the write can stop it. If ownership epoch `q` increases on takeover, the protected resource must reject a write older than its last accepted epoch. Fencing prevents later stale writes only where enforced; it cannot undo an already accepted effect (**D**, CLM-009).

The Chubby lock service paper describes the same idea as a *sequencer*: the lock holder passes the lock's name, mode, and generation number with its requests, and the recipient server rejects a request whose sequencer is no longer valid, checking it either against the lock service or against the most recent sequencer it has observed (**O**, CLM-018). The second option has a window, shown in the trace below.

**Quantitative Model / Trade-off Comparison:**
A resource with last accepted epoch $q_{last}$ accepts a write only when $q\ge q_{last}$, then sets $q_{last}\leftarrow q$, where $q$ is the epoch carried by the incoming write and $\leftarrow$ means "store this value". Per write, step by step: (1) compare $q$ with the stored $q_{last}$; (2) if $q<q_{last}$, reject and change nothing; (3) otherwise apply the write and store $q_{last}\leftarrow q$ (unchanged when $q=q_{last}$, raised when $q>q_{last}$). Example: with $q_{last}=7$, a write with $q=8$ is accepted and $q_{last}$ becomes 8; a later write with $q=7$ is then rejected since $7<8$. This prevents later stale writes where enforced; it cannot undo prior effects, and it rejects an old epoch only after the resource has seen a newer one.

The comparison is $\ge$, not $>$, and that differs on purpose from the version rule of Module 10, Lesson 10.2, which accepts only a strictly newer version:

| | Fencing epoch (this lesson) | Source version (Module 10) |
|---|---|---|
| What the number identifies | the current *owner*; one owner sends many different writes under one epoch | one specific *change* to one record |
| Equal value means | another write from the legitimate owner | the same change again |
| Rule | accept if $q\ge q_{last}$ | accept only if strictly newer; an equal version is a duplicate if the content hash matches and a conflict if it does not |
| What handles a repeated operation | not the epoch: the idempotency key of Lesson 14.2 | the version and content hash themselves |

Accepting an equal epoch is therefore harmless only because the epoch is not the duplicate guard. A resource that needs both protections applies both, in the order of the Lesson 14.2 transaction: reject a stale epoch, then check the key. With $>$ instead of $\ge$, an owner could write once per epoch and no more.

**Worked Example (synthetic fixtures):**

*Input A: saga.* Four steps run in order; the fourth fails permanently, and its failure is a recorded outcome, not a timeout.

| Step | Effect | Registered with the step | After T4 fails |
|---|---|---|---|
| T1 | reserve one unit of inventory | C1: release the reservation | C1 runs second |
| T2 | charge 30 | C2: refund 30 | C2 runs first |
| T3 | send "order confirmed" to the customer | none possible; forward action F3: send a cancellation notice | F3 is sent; T3 stays `irreversible` |
| T4 | create shipment | none needed | failed; no effect to compensate |

*Steps.* Compensate in reverse dependency order: F3, then C2, then C1. Each is a new effect with its own key and receiver contract from Lesson 14.2. While they run the saga is `compensation_pending`.

*Result.* If C2 and C1 succeed, the charge is refunded and the stock is free, and the customer has still read two messages. That is a restored business invariant, not a rollback. If C2 fails permanently, the saga is `compensation_failed`; C1 may still run, giving `partially_compensated`, and the refund goes to `manual_recovery`.

*Limit.* If T4 had *timed out* instead of failing, its state would be unknown. Compensating T1 and T2 first could refund an order whose shipment exists. Reconcile T4 by key before compensating.

*Input B: fencing trace.* A protected resource stores `last_epoch`, initially 6. A lease service grants ownership epochs.

| Step | Event | Lease service | Resource `last_epoch` | Decision at the resource |
|---|---|---|---|---|
| 1 | W1 is granted the lease | epoch 7, W1 | 6 | none |
| 2 | W1 writes `a` with epoch 7 | epoch 7, W1 | 6 → 7 | accept, 7 ≥ 6 |
| 3 | W1 pauses past lease expiry | expired | 7 | none |
| 4 | W2 is granted the lease: the epoch advances | epoch 8, W2 | 7 | none; the resource has not seen 8 |
| 5 | W2 writes `b` with epoch 8 | epoch 8, W2 | 7 → 8 | accept, 8 ≥ 7 |
| 6 | W1 resumes, still believes it owns the lease, writes `c` with epoch 7 | epoch 8, W2 | 8 | **reject** `STALE_EPOCH`, 7 < 8 |

*Result.* The fenced resource applies `a` and `b` and rejects `c`. An unfenced resource accepts `c` at step 6, and a write computed from pre-takeover state lands on top of `b`.

*Limits.* (1) Fencing does not undo `a`. (2) If W1's write arrives between steps 4 and 5, the resource still holds `last_epoch` 7 and accepts it, because a resource that compares against the highest epoch it has seen learns about the takeover only from the new owner's first write. Close that window by making the new owner's first action a barrier write that registers epoch 8 at every protected resource, or by having the resource validate epochs with the lease service. (3) A resource that ignores the epoch is unprotected whatever the lease service does.

**Knowledge Check:**
1. Why is compensation not database rollback?
2. What failure remains if a downstream resource ignores fencing tokens?

**Guided Practice:**
(a) A resource has `last_epoch` 11. Writes arrive in this order with epochs 12, 11, 12, 13, 12. Give each decision and the final `last_epoch`. (b) Implement `compensation_pending`, `failed`, `partially_compensated`, and `manual_recovery` paths plus a fenced/unfenced stale-worker test, including a stale write that arrives before the new owner's first write.

**Feedback Contract:**
- *Expected Evidence*: (a) accept (11 → 12), reject, accept (stays 12), accept (12 → 13), reject; final `last_epoch` 13. (b) Compensation dependency order, authorization/evidence, irreversible cases, resource-enforced epochs, and the barrier write or lease check that closes the takeover window.
- *Common Failure*: Registering compensation after an untracked effect; rejecting the third write in (a) because its epoch "is not new"; checking the epoch in the worker instead of at the resource.
- *Diagnostic Hint*: Can a paused worker still reach the resource? What has the resource seen at that moment?
- *Concept to Revisit*: Forward Recovery and Fencing.

**Learning Outcome:**
Implement saga recovery and worker ownership without using “rollback” or “lock” more strongly than the mechanism supports.

*(Effort: 40m instruction, 30m practice)*

---

### Lesson 14.4 — Fan-Out, Joins, Timers, and Human Decisions

**Engineering Question:**
Which membership, deadline, cancellation, and identity state must persist for parallel and human-gated work to recover correctly?

**Concepts & Definitions:**
- **Durable membership**: expected child set recorded before or with dispatch.
- **Join predicate**: all-of, quorum, any-of, or partial-success rule.
- **Durable wait**: timer/event/approval persisted without occupying worker memory.

**Mechanism Explanation:**

For fan-out, persist the expected child set before/with dispatch, stable child IDs, attempts, per-child outcome, join predicate, deadline, cancellation, and duplicate/late handling. An all-of join's latency is at least the slowest critical child plus scheduling/join overhead; total work sums all attempts. Quorum and any-of change success/cancel policy, not the need for durable membership (**D**, CLM-010).

Long waits must not occupy worker threads or depend on process memory. Persist timer/event/approval identity, correlation, authorization, payload/schema version, deadline, cancellation, and supersession. Authenticate the human and bind the decision to the exact action preview/version; reject duplicate, expired, or late approval according to policy (**D**, CLM-011).

Measure workflow-history and payload growth. High fan-out, frequent signals, verbose model/tool payloads, and repeated retries can make replay and storage expensive. Use object references, child partitions, compaction/snapshots, or continue-as-new semantics only after documenting changed lineage, atomicity, cancellation, and query behavior (**O**, CLM-013).

**Quantitative Model / Derivation:**
For parallel child durations $T_i$, an all-of join cannot succeed before $\max_i T_i$ plus scheduling/join overhead, while total attempted work sums child attempts.

Here $T_i$ is the time from dispatch until child $i$ finally completes (including its retries). The all-of join needs every child, so it waits for the slowest: latency $\ge\max_i T_i$. Work instead adds up every attempt's duration across all children, $\sum_i\sum_j d_{ij}$, where $d_{ij}$ is the running time of attempt $j$ of child $i$. Example from the five-child fixture below, if c4 were allowed to finish at 12 s: completions at 2, 3, 4, 12, 8 s give all-of latency $\ge\max=12$ s, while work is $2+3+4+12+(5+3)=29$ child-seconds. Parallelism shortens the wait, not the work.

History size and replay cost follow from four declared quantities: events per activity attempt, payload size, run shape, and retention.

$$
N_a=N_{steps}\,n_{act}\,\bar a,\qquad
H=e_0+N_a e_a+N_{sig},\qquad
B=H\,m+N_a\,p+N_{sig}\,p_{sig},
$$

$$
R=\lambda\,(T_{open}+T_{retain}),\qquad
t_{replay}=\frac{H}{r_e}+\frac{B}{r_b}.
$$

$N_a$ is activity attempts per run, $H$ history events, $B$ history bytes, $R$ retained histories at steady state (start rate times time in the store), and $t_{replay}$ the time to fetch and replay one history. The formulas are bookkeeping identities for a runtime that records every attempt; every input must be measured for a real runtime.

Reading each formula (symbols are defined in the input table below):
1. $N_a$: each of $N_{steps}$ steps runs $n_{act}$ activities, and each activity takes $\bar a$ attempts on average, so attempts multiply.
2. $H$: a fixed $e_0$ events per run, plus $e_a$ events for every attempt, plus one event per signal.
3. $B$: every event carries metadata $m$; every attempt adds its payload $p$; every signal adds $p_{sig}$. With $m$, $p$, $p_{sig}$ in KiB, $B$ is in KiB.
4. $R$: a run stays in the store while open ($T_{open}$) and then for $T_{retain}$; at a steady start rate $\lambda$, the number present at any moment is rate times time in store (the "arrival rate × time in system" count of Little's law). Units: runs/day × days = runs.
5. $t_{replay}$: process $H$ events at $r_e$ events/s, plus fetch $B$ bytes at $r_b$; convert $B$ to MiB before dividing by $r_b$ in MiB/s.

Symbol note: $H$ (event count) is unrelated to the intent hashes `h1`, `h2` of Lesson 14.2, $e_a$/$e_0$ are counts, not the events `e1`, `e2` of the history tables, and $R$ here is a count of histories, not a receipt.

*Synthetic estimate.* All inputs below are exercise assumptions (CLM-021), not measurements of any runtime.

| Input | Symbol | Value |
|---|---|---|
| agent steps per run; activities per step; mean attempts per activity | $N_{steps}$; $n_{act}$; $\bar a$ | 40; 2; 1.25 |
| events per activity attempt; fixed events per run; signals per run | $e_a$; $e_0$; $N_{sig}$ | 6; 5; 10 |
| metadata per event; payload per attempt, embedded or as reference; payload per signal | $m$; $p$; $p_{sig}$ | 0.3 KiB; 6 KiB or 0.2 KiB; 1 KiB |
| start rate; open duration; retention after close | $\lambda$; $T_{open}$; $T_{retain}$ | 2,000 runs/day; 2 days; 30 days |
| replay rate; history fetch rate; parallel replay slots | $r_e$; $r_b$; none | 5,000 events/s; 20 MiB/s; 16 |

| Quantity | Embedded payloads | Payload references |
|---|---:|---:|
| attempts $N_a = 40\times2\times1.25$ | 100 | 100 |
| events $H = 5+100\times6+10$ | 615 | 615 |
| bytes per history $B$ | $184.5+600+10=794.5$ KiB (0.776 MiB) | $184.5+20+10=214.5$ KiB (0.209 MiB) |
| retained histories $R = 2{,}000\times32$ | 64,000 | 64,000 |
| history storage $R\,B$ | 48.49 GiB | 13.09 GiB, plus 36.62 GiB of payloads in the object store |
| replay time per history | $0.123+0.039=0.162$ s | $0.123+0.010=0.133$ s |
| open runs $2{,}000\times2$ | 4,000 | 4,000 |
| replay work after a fleet restart; wall time on 16 slots | 647 s; 40.4 s | 534 s; 33.4 s |
| growth per agent step | 15 events, 19.5 KiB | 15 events, 5.0 KiB |

*Step-by-step for the table* (1 MiB = 1,024 KiB; 1 GiB = 1,024 MiB):
1. $N_a=40\times2\times1.25=100$; $H=5+100\times6+10=615$.
2. $B$ embedded $=615\times0.3+100\times6+10\times1=184.5+600+10=794.5$ KiB; $794.5/1024=0.776$ MiB. With references, the middle term becomes $100\times0.2=20$, so $B=214.5$ KiB $=0.209$ MiB.
3. $R=2{,}000\times(2+30)=64{,}000$; storage $64{,}000\times794.5$ KiB $=50{,}848{,}000$ KiB $/1024^2=48.49$ GiB (embedded) and $64{,}000\times214.5$ KiB $=13.09$ GiB (references). The payloads moved out are $64{,}000\times100\times6$ KiB $=36.62$ GiB.
4. $t_{replay}=615/5{,}000+0.776/20=0.123+0.039=0.162$ s embedded; $0.123+0.209/20=0.123+0.010=0.133$ s with references.
5. Open runs $=\lambda T_{open}=2{,}000\times2=4{,}000$. Fleet-restart replay work $=4{,}000\times0.162\approx647$ s, and on 16 slots $647/16\approx40.4$ s; with references $4{,}000\times0.133\approx534$ s and $534/16\approx33.4$ s (computed from unrounded times).
6. Per agent step: events $n_{act}\bar a\,e_a=2\times1.25\times6=15$; bytes $15\times0.3+2.5\times6=4.5+15=19.5$ KiB embedded, $4.5+2.5\times0.2=5.0$ KiB with references.

*Interpretation and limits.*
- References move bytes; they do not remove them. Total storage is 49.71 GiB with references against 48.49 GiB embedded, and the object store must now retain payloads at least as long as the histories that point to them.
- The restart figures assume every open run is at full length, perfect parallelism, and no queueing. They size a bound; they do not predict a recovery time. Replay timing counts as evidence only when measured, and a synthetic value must stay labeled synthetic.
- The replay time with references assumes orchestration code does not read the payloads. If it does, the fetch moves to the object store and returns to the total.
- One mean history cannot bound a heavy-tailed population. Size the largest runs separately.
- Limits are runtime-specific. One runtime documents a warning at 10,240 events or 10 MB and termination at 51,200 events or 50 MB of history per execution (**O**, CLM-020). Reading MB as MiB, this model with embedded payloads crosses the warning by size after 524 steps, before the event-count warning at 681 steps. With references, the event count binds first at 681 steps and size not until 2,045. Derivation for $S$ agent steps: the fixed part is $e_0+N_{sig}=15$ events and $15\times0.3+10=14.5$ KiB, and each step adds 15 events and 19.5 KiB (or 5.0 KiB). Events: $15+15S\le10{,}240\Rightarrow S\le681.7$. Size embedded: $14.5+19.5S\le10{,}240\Rightarrow S\le524.4$. Size with references: $14.5+5.0S\le10{,}240\Rightarrow S\le2{,}045.1$. The smallest bound binds first. The same documentation states that an activity's started event is written together with its terminal event, so $e_a$ per *attempt* is exactly the kind of input that must be measured rather than assumed.

**Worked Example (synthetic fixture):**

*Input.* A parent persists membership `{c1, c2, c3, c4, c5}` with dispatch at $t=0$ and a 10 s join deadline.

| Child | Attempts | Recorded events (time in s) | State at the deadline |
|---|---:|---|---|
| c1 | 1 | completed at 2 | completed |
| c2 | 1 | completed at 3; the same completion delivered again at 3.5 | completed; the repeat is dropped by child ID and attempt |
| c3 | 1 | completed at 4 | completed |
| c4 | 1 | nothing by 10; a completion arrives at 12 | timed out; the result at 12 is recorded as `late` |
| c5 | 2 | attempt 1 fails at 5; attempt 2 completes at 8 | completed |

*Steps.* Evaluate each join predicate over distinct child IDs in the persisted membership, at each event time. Work is counted from $t=0$ to each attempt's end or cancellation: for quorum at 4 s, c1–c3 contribute 2, 3, 4, and c4 and c5 (attempt 1, still running) are cancelled at 4, contributing 4 each. For all-of, c4 runs to the 10 s deadline and c5 contributes its two attempts, $5+(8-5)=5+3$.

| Join policy | Decided at | Outcome | Distinct children counted | Attempted work if cancellation is immediate |
|---|---:|---|---:|---|
| any-of | 2 s | satisfied by c1 | 1 | $5\times2=10$ child-seconds; the other four are cancelled at 2 s |
| quorum 3-of-5 | 4 s | satisfied by c1, c2, c3 | 3 | $2+3+4+4+4=17$ child-seconds; c5's second attempt never starts |
| all-of | 10 s, the deadline | not satisfied, 4 of 5 | 4 | $2+3+4+10+5+3=27$ child-seconds; 29 if c4 cannot be cancelled and runs to 12 |

*Result.* Five children made six attempts. All-of fails at the deadline as an explicit partial outcome. Quorum succeeds at 4 s. An implementation that counts completion *messages* instead of distinct children reaches three at 3.5 s (c1, c2, and c2's repeat) and declares quorum with only two children done.

*Late and human events.* c4's completion at 12 s arrives after the join has closed under every policy. It is recorded and not counted. If c4 performed an external effect, that effect exists and must be reconciled or compensated: quorum does not remove late children. The same rule applies to approvals. Take approval request `ap-9`, bound to action preview `p3`, expiring at $t=100$; each row is an independent scenario:

| Decision received | Rule | Result |
|---|---|---|
| approve `ap-9` for preview `p3` at $t=40$, then the same decision ID again at $t=41$ | duplicate decision ID | accepted once; the repeat is ignored |
| approve `ap-9` for preview `p2` at $t=45$ | preview/version mismatch | rejected `wrong_version` |
| approve `ap-9` for preview `p3` at $t=70$, after the workflow was cancelled at $t=60$ | approval after cancellation | rejected `cancelled`; recorded; no action runs |
| approve `ap-9` for preview `p3` at $t=120$ | past expiry | rejected `expired` |

*Interpretation and limits.* The tables are derivations on a synthetic fixture (**D**, CLM-010, CLM-011). Whether a child can actually be cancelled, and how fast, is a property of the child and decides which work figure applies.

**Knowledge Check:**
1. Why must membership persist before join evaluation?
2. Does quorum remove the need to handle late children?

**Guided Practice:**
(a) With the per-event constants of the synthetic estimate ($e_a=6$, $e_0=5$, $m=0.3$ KiB, $p=6$ KiB or 0.2 KiB) and no signals, estimate events and bytes per history for 120 steps, 3 activities per step, and 1.5 mean attempts. (b) In the five-child fixture, give the decision time of a 4-of-5 quorum. (c) Estimate history entries for a fan-out/retry workload of your own, compare embedded payloads with object references, and state the changed failure modes.

**Feedback Contract:**
- *Expected Evidence*: (a) 540 attempts; $5+540\times6=3{,}245$ events; $973.5+3{,}240=4{,}213.5$ KiB (4.11 MiB) embedded; $973.5+108=1{,}081.5$ KiB (1.06 MiB) with references. (b) 8 s, when c5's second attempt completes; the duplicate of c2 does not count. (c) Stable child/approval IDs, membership, join/deadline/cancel policy, late-event handling, and history growth with every input labeled measured or synthetic and with events per attempt, payload size, and retention stated.
- *Common Failure*: Reconstructing membership from whatever children happen to return; multiplying by activities instead of attempts (3,245 becomes 2,165 events); presenting a synthetic replay time as measured.
- *Diagnostic Hint*: Can recovery distinguish missing from never-scheduled children? Which inputs of the estimate did you measure?
- *Concept to Revisit*: Durable Coordination.

**Learning Outcome:**
Recover parallel and human-gated work with explicit partial, timeout, and late-event semantics.

*(Effort: 45m instruction, 35m practice)*

---

### Lesson 14.5 — Recovery Telemetry and Crash Falsification

**Engineering Question:**
Which correlated telemetry and fault boundaries localize a recovery defect without guessing from a final workflow state?

**Concepts & Definitions:**
- **Crash matrix**: controlled before/after failures around each persistence/effect boundary.
- **Recovery oracle**: declared expected final state and external effects.
- **Effect ledger**: correlation of logical operation, attempts, receipts, and reconciliation.

**Mechanism Explanation:**

Observe separately:

- workflow/run/type/code version, history sequence/size, replay count/failure, and terminal state;
- workflow-task and activity-task queue delay, attempts, heartbeat/progress, timeout, cancellation, and retry;
- logical operation, external effect state, idempotency hit/conflict, receipt/postcondition, compensation, and reconciliation;
- timers/signals/approvals, duplicate/late/rejected events, fan-out membership, and join status;
- final business outcome, latency, cost, duplicate/lost effects, and manual intervention.

One "workflow success" counter cannot separate these (**D**, CLM-014).

Build a crash matrix around every boundary: before/after history append, task delivery, external dispatch, external commit, receipt write, activity completion, outbox publish/mark, compensation, timer firing, signal/approval handling, lease expiry/takeover, and join completion.

The claim that boundary-targeted injection finds more failures than generic restart tests is a hypothesis (**H**, CLM-016). Compare unique actionable failures, escaped incidents, test cost, and oracle divergence. Use a no-failure oracle for final state/effects, while acknowledging that irreversible real-world effects may require a sandbox.

**Quantitative Model / Trade-off Comparison:**
Report recovery-time distribution, duplicate/lost/unknown effects, stuck workflows, replay failures, compensation outcomes, and manual interventions by injected boundary. Counts without opportunity denominators cannot compare fault coverage. An *opportunity* is one injected crash at one point: opportunities $=$ boundaries $\times$ sides (before/after) $\times$ trials per point, and a rate is observed failures divided by the opportunities at the same scope. Guided Practice (a) works this out.

**Worked Example (synthetic fixture):**
*Input.* A crash is injected after the external commit and before activity completion, which is case W3 of Lesson 14.2. Two views exist: the workflow dashboard and the receiver's ledger.
*Steps.* The dashboard shows the activity pending with history `[e1, e2]`. The ledger shows row `(k, h1, r-1)` and balance 70. Joining the two on the logical effect key `k` shows one effect and no completion event.
*Result.* The diagnosis is "committed, acknowledgement lost". The action is reconcile by key, then append the completion. A blind retry under a new key is ruled out.
*Interpretation and limits.* The workflow view alone cannot separate this from W1 or W2. The join works only if both systems record the same key.

**Knowledge Check:**
1. Which signal separates queue delay from replay failure?
2. Why can a generic process restart miss a narrow acknowledgement window?

**Guided Practice:**
(a) A matrix has 6 boundaries, each crashed before and after, with 20 trials per injection point. Three duplicate effects are observed, all at "after external commit, before completion". Report the opportunity count and the duplicate rate overall and at that point. (b) Build and execute a boundary matrix from history append through compensation and join completion, using a no-failure oracle where safe.

**Feedback Contract:**
- *Expected Evidence*: (a) $6\times2\times20=240$ opportunities; $3/240=1.25\%$ overall; $3/20=15\%$ at the one boundary. The overall rate hides a defect that is concentrated at one point, and 20 trials is too few to call the other eleven points safe. (b) Fault timing, opportunity count, correlated workflow/activity/effect IDs, oracle delta, and cleanup.
- *Common Failure*: Declaring recovery correct because the workflow eventually closed; reporting three duplicates without a denominator.
- *Diagnostic Hint*: What did the external system commit, and how many times was each boundary exercised?
- *Concept to Revisit*: Boundary-Targeted Falsification.

**Learning Outcome:**
Localize a recovery defect to persistence, replay, scheduling, activity, external effect, compensation, or coordination.

*(Effort: 30m instruction, 25m practice)*

---

### Lesson 14.6 — Versioning and Replay-Safe Deployment

**Engineering Question:**
How can code evolve while retained histories continue to replay and in-flight work preserves its original semantics?

**Concepts & Definitions:**
- **Command-affecting change**: code/config revision that alters history consumption or emitted commands.
- **Replay corpus**: representative retained histories for every active version/state.
- **Version routing**: explicit mapping from histories/new starts to compatible code.

**Mechanism Explanation:**

Long-running executions cross software releases. Command-affecting changes—reordering activities/timers, changing identities/types, altering branches, or consuming history differently—can break replay. Use runtime-specific patch/version routing and replay representative retained histories under candidate code before canary (**O**, CLM-012).

At Temporal Python SDK revision `eb642b14947bd8bcdf8816cffb6a63869803f5c6`, public workflow `start_activity`/`execute_activity` helpers delegate to the current runtime's `workflow_start_activity`; `Replayer.replay_workflow` and `replay_workflows` consume histories and surface replay failures, including nondeterminism (**O**, CLM-015). In `_workflow_replay_iterator`, an eviction hook converts a cache eviction whose reason is nondeterminism into a `NondeterminismError` and stores it as the result's `replay_failure`; `replay_workflow` raises it by default. Static inspection only, re-read at this commit on 2026-10-01. This is a source-reading example, not a universal workflow architecture.

Test histories from every active version and branch, including timers waiting, activities pending/retrying, compensation in progress, approvals pending, and history near limits. Define how old workers/code remain available, how new starts route, and when histories/data can be retired.

**Quantitative Model / Derivation:**
Capacity planning must measure retained-history count/size, replay latency, task queue delay, and recovery concurrency. These are empirical inputs; one average history cannot bound a heavy-tailed active population.

**Worked Example (synthetic fixture):**
*Input.* A retained history recorded `ActivityScheduled(A)` and then `TimerStarted(T)`. A candidate revision starts the timer before the activity.
*Steps.* Replay feeds the history to the candidate. At the first command the candidate emits `StartTimer` where the history recorded `ScheduleActivity(A)`.
*Result.* Replay fails with a nondeterminism error at that event, and the release is blocked.
*Interpretation and limits.* Either the change is wrapped in a runtime-specific patch/version branch so old histories keep the old order, or old histories are routed to workers running the old code. New executions alone would not have shown the problem.

**Knowledge Check:**
1. Why must pending timers and compensations appear in the replay corpus?
2. When may old workers be retired?

**Independent Practice:**
(a) A replay corpus holds 120 open histories started on v1 and 300 started on v2. Under candidate v3, all 120 v1 histories fail replay at the same command and all 300 v2 histories pass. State the release decision and when v1 workers may be retired. (b) Replay every active branch/version including near-limit histories, then specify canary, routing, rollback, and data-retirement conditions.

**Feedback Contract:**
- *Expected Evidence*: (a) v3 is not safe for v1 histories: $120/420\approx28.6\%$ of open runs would be stranded. Either add a patch/version branch and replay again, or route v1 histories to v1 workers. v1 workers may be retired only when no open v1 history remains and none can be reset or queried against that code. (b) History inventory, replay results, compatibility matrix, measured replay cost, and old-code retention plan.
- *Common Failure*: Testing only newly started workflows; shipping because "most" histories pass.
- *Diagnostic Hint*: Which recorded command does candidate code emit differently, and which population of histories contains it?
- *Concept to Revisit*: Replay-Safe Evolution.

**Learning Outcome:**
Deploy workflow changes without stranding or silently changing in-flight executions.

*(Effort: 35m instruction, 25m practice)*

---

## 05 Literature & Production Source Map

Reading status is stated per entry. *Opened 2026-10-01* means the page or file was read on that date for this revision. Entries without such a note keep their 2026-09 access record in the registry and were not re-read.

**REFERENCE / BASELINE**

- [SAGAS](https://www.cs.princeton.edu/techreports/1987/070.pdf) — Garcia-Molina and Salem, Princeton technical report, January 1987. Opened 2026-10-01: abstract and Section 1 only (scanned document).
  - *Scope*: the saga and compensating-transaction model in Lesson 14.3 (**O**, CLM-007). The abstract frames sagas for a database management system. Whether the report treats irreversible real-world effects was not read in this revision; the statements about them in Lesson 14.3 are this module's derivation (**D**, CLM-008).
- [The Chubby Lock Service for Loosely-Coupled Distributed Systems](https://research.google.com/archive/chubby-osdi06.pdf) — Burrows, OSDI 2006. Opened 2026-10-01: Section 2.4, Locks and sequencers.
  - *Scope*: the sequencer check that Lesson 14.3 calls a fencing token (**O**, CLM-018). One system's design from 2006.

**RECOMMENDED ENGINEERING BASELINE** (this module's derivation; earlier revisions labeled this list "CURRENT DEFAULT")

Durable transition history or checkpoints; explicit activity/effect boundaries; stable operation identities; an atomic receiver idempotency transaction or reconciliation; bounded retries; fencing enforced at the resource; durable timers, events, and approvals; version and replay tests; state/activity/effect telemetry.

These items follow from stated assumptions (**D**, CLM-001, CLM-004, CLM-005, CLM-006, CLM-008, CLM-009, CLM-010, CLM-011, CLM-014, CLM-019). They are a recommendation. They are not the default of any runtime, and this module has not measured how widely they are adopted.

**DOCUMENTED BEHAVIOR OF NAMED RUNTIMES AND APIS** (observed, scoped to the named source)

- [Temporal Workflow Definition](https://docs.temporal.io/workflow-definition) — opened 2026-10-01. Workflow code must be deterministic to support replay; emitted commands are compared with the existing event history; nondeterministic work belongs in activities; versioning or patching is required for command-changing edits (**O**, CLM-002, CLM-003, CLM-012).
- [Azure Durable orchestrator code constraints](https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-code-constraints) — opened 2026-10-01; URL redirect and page date (2026-10-12) re-checked 2026-11-09. Orchestrators replay and must be deterministic; clocks, random values, environment variables, and direct I/O must go through context APIs or activities (**O**, CLM-002).
  - *Scope of both*: two runtime families that use replay. Not every durable engine does, and neither page describes external-effect safety.
- [Temporal Cloud limits](https://docs.temporal.io/evaluate/cloud/limits), [Events and Event History](https://docs.temporal.io/workflow-execution/event), and [Child Workflows](https://docs.temporal.io/child-workflows) — opened 2026-10-01. Event-history warning and termination thresholds, and the child-workflow partitioning guidance (**O**, CLM-013, CLM-020). One runtime's documented limits; they can change.
- [The Idempotency-Key HTTP Header Field](https://datatracker.ietf.org/doc/html/draft-ietf-httpapi-idempotency-key-header-07) (draft-07, 2025-10-15, an expired Internet-Draft) and [one payment API's idempotent-requests page](https://docs.stripe.com/api/idempotent_requests) — opened 2026-10-01 (**O**, CLM-017). A draft and one vendor's API; neither is a standard that endpoints can be assumed to follow.

**INDUSTRY PREVALENCE:** not established by this module. No adoption survey was opened (`TODO_VERIFY`, CLM-023).

**WORKLOAD-DEPENDENT:** event sourcing versus checkpointing, local versus remote activities, history partitioning, saga compensation, heartbeat interval, lease duration, fan-out topology, approval placement, and storage/retention.

**FRONTIER** (each scoped to what was read)

- Durable-execution integrations for agent frameworks. One agent framework's documentation lists integrations with several durable-execution systems, and one runtime vendor publishes recipes for durable agent loops, tool calling, and human-in-the-loop approval ([framework page](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), [vendor cookbook](https://docs.temporal.io/ai/cookbook); both opened 2026-10-01) (**O**, CLM-022). This shows availability. It does not show adoption, and it does not show that the integrations implement the receiver contract of Lesson 14.2.
- Replay-safe AI telemetry and richer durable event streams were named in the earlier revision without a source. No primary source was opened (`TODO_VERIFY`, CLM-023).

**LEGACY / INSUFFICIENT:** save the chat transcript and call it a checkpoint; resume from a step number without effect evidence; assume a timeout means no commit; add an idempotency key without atomic receiver support; call compensation rollback; rely on an unfenced lease; block a worker during HITL; upgrade workflow code without replay tests.

**PRODUCTION SOURCE TRACE**

- Repository: `temporalio/sdk-python`
- Revision: `eb642b14947bd8bcdf8816cffb6a63869803f5c6`
- Verified: 2026-09-26; both files and the symbols below were re-read at this revision on 2026-10-01. Static inspection only on both dates.
- Files/symbols: `temporalio/workflow/_activities.py::{start_activity,execute_activity}` and `temporalio/worker/_replayer.py::{Replayer.replay_workflow,Replayer.replay_workflows,Replayer._workflow_replay_iterator}`.
- Execution paths: workflow activity helper → current runtime `workflow_start_activity` → handle/result; history iterator → replay worker → eviction/result hook → replay result → raise/aggregate failure.
- Observed in the re-read: `start_activity` returns `_Runtime.current().workflow_start_activity(...)` and `execute_activity` awaits the same call; `replay_workflow` raises `result.replay_failure` when `raise_on_replay_failure` is true (the default); `replay_workflows` raises on the first failure or collects failures by run ID; the eviction hook maps a nondeterminism eviction to `NondeterminismError`.
- Scope: one Python SDK snapshot (**O**, CLM-015). The runtime implementation behind `workflow_start_activity`, the server, and the bridge were not read, and recovery was not executed. Other runtimes implement durability differently.

### LAYER 2: ENGINEERING PRACTICE LAYER

## 06 Engineering Labs

All labs follow `PREDICT → BUILD → MEASURE → EXPLAIN → BREAK → IMPROVE → FALSIFY`.

### LAB A — History, Replay, and Crash Recovery

- **Objective**: Build deterministic durable state reconstruction and crash every persistence/acknowledgement boundary.
- **Pre-Registered Hypothesis**: Recorded nondeterministic inputs and replay tests will detect command divergence missed by restart-only tests on the seeded revisions.
- **Independent Variables**: Persistence boundary, nondeterministic input, code revision, and crash timing.
- **Dependent Variables**: Replay agreement/failure, recovered state, command sequence, recovery time, and stuck workflows.

- Implement a stateful workflow with recorded commands and separate activities; build a deterministic reducer or use a durable runtime.
- Inject clocks, randomness, mutable config, unordered iteration, direct I/O, and command-changing code revisions.
- Crash before/after transition acknowledgement, task delivery, activity schedule/result, and checkpoint/history writes.
- Artifact: state reconstruction proof, replay corpus, nondeterminism report, and recovery-time distribution.
- **Break & Falsify**: Inject unrecorded clock/random/config/direct I/O and command reorder; any silent divergent command falsifies replay safety.
- **Alignment**: Lesson 14.1.
- **Effort Estimate**: 3h total.

### LAB B — Effect Safety and Compensation

- **Objective**: Implement atomic receiver idempotency, reconciliation, outbox/inbox, and dependency-aware compensation.
- **Pre-Registered Hypothesis**: Atomic key+intent+outcome storage will prevent duplicate receiver effects under concurrent duplicate delivery where check-then-act does not. Both receivers are built by the learner, so this checks an implementation against the contract; it is not a finding about any product.
- **Independent Variables**: Deduplication design, crash window, concurrency, retention, key conflict, and compensation failure.
- **Dependent Variables**: Duplicate/lost/unknown effects, conflicts, reconciliation time, compensation state, and manual work.

- Build an endpoint with atomic idempotency key + intent hash + stored outcome; contrast a check-then-act implementation.
- Inject concurrent duplicates, crash after commit/before acknowledgement, retention expiry, key conflict, timeout, partial effect, stale read, outbox relay duplicate, and poison message.
- Add dependency-aware compensation and irreversible/manual-recovery cases.
- Artifact: effect ledger; duplicate/loss matrix with one row per case W1–W4, C1–C3, and R1 of Lesson 14.2, each giving durable prefix, receiver state, recovery action, and result; reconciliation procedure; and compensation evidence.
- **Break & Falsify**: Crash after commit/before ack, expire keys, poison relay, reuse keys, and fail compensation; any duplicate effect under supported idempotency scope falsifies the design.
- **Alignment**: Lessons 14.2–14.3.
- **Effort Estimate**: 3h total.

### LAB C — Parallelism, Fencing, and Durable HITL

- **Objective**: Recover fan-out/join, stale-worker takeover, and authenticated long-lived approvals.
- **Pre-Registered Hypothesis**: Resource-enforced epochs will reject resumed stale writers; durable membership and approval binding will preserve declared outcomes after restart.
- **Independent Variables**: Join policy, child outcome/order, lease epoch, fencing enforcement, approval timing/version, and restart.
- **Dependent Variables**: Join correctness, stale-write acceptance, approval rejection, history growth, latency, and intervention.

- Persist fan-out membership and implement all-of, quorum, any-of, partial-success, timeout, and cancellation joins.
- Pause a worker beyond lease expiry, issue a new epoch, then resume the stale worker against fenced and unfenced resources. Include a stale write that arrives before the new owner's first write, with and without a barrier write.
- Wait durably for an authenticated approval; inject duplicate, late, expired, wrong-version, and unauthorized decisions.
- Artifact: join state machine, fencing trace with the epoch advance, approval audit, and a history-growth curve with events per attempt, payload size, and retention declared and each value labeled measured or synthetic.
- **Break & Falsify**: Inject duplicate/late children, timeout/cancel, stale workers, and wrong/late/expired approvals; acceptance of a stale or misbound action falsifies safety.
- **Alignment**: Lessons 14.3–14.4.
- **Effort Estimate**: 3h total.

### LAB D — Upgrade and Recovery Release Gate

- **Objective**: Replay representative retained histories and compare boundary-targeted crashes with generic restarts.
- **Pre-Registered Hypothesis**: Boundary-targeted injection will find at least one actionable seeded recovery defect not exposed by generic restart under the same budget (**H**, CLM-016).
- **Independent Variables**: History version/state/size, candidate code, injection strategy, queue load, and routing policy.
- **Dependent Variables**: Replay failures, unique defects, recovery time, queue delay, effect convergence, compensation, cost, and intervention.

- Replay representative histories under candidate code, including every active version/state and near-limit histories.
- Measure history/payload growth, replay latency, task/activity attempts, queueing, recovery time, external effect convergence, compensation, and manual intervention.
- Compare boundary-targeted crash injection with generic process restarts using a preregistered oracle.
- Artifact: compatibility matrix, crash coverage, canary/rollback/routing plan, and source trace.
- **Break & Falsify**: Include all active states and near-limit histories; a candidate that strands any supported history fails the gate.
- **Alignment**: Lessons 14.5–14.6 and Incident 14.1.
- **Effort Estimate**: 3h total.

## 07 Break / Incident Scenarios

### Incident 14.1 — Resumed Agent Duplicates an Irreversible Action

- **Incident Symptoms**: After a timeout and replay, an external action duplicates, compensation fails, a late approval is applied, a stale worker writes after takeover, and older histories fail after rollout.
- **Diagnostic Protocol (Task)**:
  1. *Formulate Competing Hypotheses*: Lost acknowledgement, non-atomic/expired idempotency, stale read-back, nested retry, outbox duplicate, compensation order, approval binding, unfenced lease, command drift, history pressure, or queue saturation.
  2. *Rank Initial Plausibility*: Use external duplicate and replay evidence without assuming one root cause.
  3. *Identify Missing Evidence*: Recover workflow/history, task attempts, logical operation/intent, receiver records, dispatch/receipt/postconditions, external audit, outbox/inbox, compensation, approval, epochs, replay failure, history size, queue spans, and business outcome.
  4. *Design Discriminating Tests*: Freeze unknown-effect retry, reconcile by logical ID, replay old histories, and inject each crash window with predicted/falsifying outcomes.
  5. *Execute Causal Diagnosis*: Rank the earliest supported boundary and interacting recovery failures.
  6. *Prescribe Mitigation and Prevention*: Reconcile first; then repair atomic identity, fencing, approval/version policy, replay routing, and history controls as evidence requires.
  7. *Remeasure*: Duplicate/lost/unknown effects, stuck rate, replay failures, recovery latency, compensation, and manual work.

### LAYER 3: MASTERY / ASSESSMENT LAYER

## 08 Mastery Assessment

### Enterprise Transfer Problem: Durable Approval and Fulfillment Workflow

Design a workflow that calls models/tools, performs governed writes, fans out fulfillment, waits days for approval, compensates partial success, and crosses worker and code deployments.

**Required Deliverables**:
1. History/state schemas, acknowledgement contract, and replay constraints.
2. Activity/effect ledger plus delivery, idempotency, and reconciliation proof.
3. Outbox/inbox path, saga graph, irreversible/manual-recovery policy, and fencing protocol.
4. Join, cancellation, timer, and authenticated HITL state machines.
5. History/replay capacity bounds and recovery objectives, with events per attempt, payload size, and retention declared and each number labeled measured or synthetic.
6. Representative replay corpus, crash matrix, and pinned source trace.
7. Canary/version-routing/rollback plan and diagnosis of Incident 14.1.

## 09 Required Evidence & Rubric

### Required Artifact: Production Source Trace

Submit a pinned trace of a durable runtime's activity and replay paths. The reference trace uses Temporal Python SDK commit `eb642b14947bd8bcdf8816cffb6a63869803f5c6`. The trace must cover:

1. The workflow-side entry point for scheduling an activity (`start_activity` / `execute_activity`) and the parameters that set timeouts, retry policy, and activity identity.
2. Where the helper hands off to the runtime (`_Runtime.current().workflow_start_activity`), and how far beyond that call the trace followed.
3. The replay entry point (`Replayer.replay_workflow` / `replay_workflows`) and the effect of `raise_on_replay_failure`.
4. How a replay failure is detected and surfaced (`_workflow_replay_iterator`, the eviction hook, `WorkflowReplayResult.replay_failure`).
5. A list of what was not executed or verified: server and bridge behavior, real crash recovery, and any external-effect guarantee.

### Rubric Dimensions

- **Durability and Replay**: *Insufficient* saves a transcript/step. *Competent* defines acknowledgements and deterministic replay. *Strong* crash-tests durable prefixes and every active code path.
- **Effects and Recovery**: *Insufficient* assumes timeout means no effect, retries under a new key, or claims exactly-once from a checkpoint. *Competent* uses a stable effect identity, an atomic receiver transaction, and reconciliation, and gives the correct recovery for commit-without-acknowledgement. *Strong* also proves concurrent-duplicate, key-conflict, stale-worker, retention-expiry, and compensation behavior with a table of durable prefix against receiver state.
- **Coordination and Capacity**: *Insufficient* keeps joins/approvals in memory or counts completion messages instead of distinct children. *Competent* persists membership, timers, approvals, and fencing, and gives a history estimate with declared inputs. *Strong* measures history/replay growth and failure behavior at limits and keeps measured and synthetic numbers apart.
- **Diagnosis and Evolution**: *Insufficient* reports final status only. *Competent* correlates workflow/activity/effect evidence. *Strong* uses discriminating crash tests, replay corpora, routing, and remeasurement.

## 10 Capability Traceability Matrix

Deliverable numbers refer to Section 08, incident steps to Incident 14.1, and trace items to the Section 09 Required Artifact.

| Capability | Taught | Practiced | Assessed | Evidence (artifact reviewers open) |
|---|---|---|---|---|
| Durable history and deterministic replay | 14.1 acknowledgement table | 14.1 Guided Practice (a)–(b); LAB A | Mastery deliverable 1; Incident 14.1 steps 3–4 (recover history; replay old histories); rubric *Durability and Replay* | LAB A state reconstruction proof and replay corpus |
| Effect safety and receiver idempotency | 14.2 receiver transaction and crash table | 14.2 Guided Practice (a)–(b); LAB B | Mastery deliverable 2; Incident 14.1 steps 1, 3, 4 (lost acknowledgement versus non-atomic idempotency; receiver records; reconcile by logical ID); rubric *Effects and Recovery* | LAB B effect ledger and duplicate/loss matrix with rows W1–W4, C1–C3, R1 |
| Outbox and compensation | 14.2 outbox; 14.3 saga table | 14.3 Guided Practice (b); LAB B | Mastery deliverable 3; Incident 14.1 steps 1, 6 (outbox duplicate, compensation order; reconcile first); rubric *Effects and Recovery* | LAB B compensation evidence with irreversible and manual-recovery cases |
| Fencing of stale owners | 14.3 fencing trace | 14.3 Guided Practice (a)–(b); LAB C | Mastery deliverable 3; Incident 14.1 steps 1, 3 (unfenced lease; epochs); rubric *Coordination and Capacity* | LAB C fencing trace with the epoch advance, fenced and unfenced, including the pre-first-write window |
| Fan-out/join and durable HITL | 14.4 five-child join and approval tables | 14.4 Guided Practice (b)–(c); LAB C | Mastery deliverable 4; Incident 14.1 steps 1, 3 (approval binding; approval records); rubric *Coordination and Capacity* | LAB C join state machine and approval audit |
| History and replay capacity | 14.4 history and replay estimate; 14.6 | 14.4 Guided Practice (a), (c); LAB C; LAB D | Mastery deliverable 5; Incident 14.1 steps 1, 7 (history pressure; replay failures); rubric *Coordination and Capacity* | LAB C history-growth curve and LAB D replay-latency measurements, each input labeled measured or synthetic |
| Recovery diagnosis | 14.5 | 14.5 Guided Practice (a)–(b); LAB D | Mastery deliverables 6–7; Incident 14.1 steps 1–7; rubric *Diagnosis and Evolution* | LAB D crash coverage with opportunity denominators |
| Versioning and replay-safe deployment | 14.6 | 14.6 Independent Practice (a)–(b); LAB D | Mastery deliverables 6–7; Incident 14.1 steps 4, 6 (replay old histories; replay routing); rubric *Diagnosis and Evolution* | LAB D compatibility matrix and canary/rollback/routing plan |
| Pinned runtime source trace | 14.6 source paragraph; Section 05 | LAB D (source trace) | Section 09 Required Artifact items 1–5; Mastery deliverable 6 | Trace of `start_activity`/`execute_activity` and `Replayer` at commit `eb642b14…`, with the not-executed list |

## 11 Exit Criteria & Module Wrap-Up

### Exit Criteria

A learner successfully completing Module 14 must be able to:
1. Reconstruct state from a declared durable prefix and control nondeterminism.
2. Refuse exactly-once claims without atomic receiver evidence.
3. Survive duplicate/concurrent delivery and reconcile unknown effects.
4. Model compensation honestly and fence stale owners.
5. Persist joins, timers, cancellations, and approval identity/version.
6. Bound history/replay growth and replay-test every active version.
7. Diagnose crash windows from correlated workflow/activity/effect evidence.

### Module Wrap-Up (Final Mental Model Reconstruction)

- **The Core Invariant**: A durable runtime remembers decisions, while external reality has independent commit points.
- **The Recovery Path**: history → deterministic state/command → activity attempt → external effect evidence → commit, reconcile, compensate, or escalate.
- Correctness comes from history plus effect identity, atomic idempotency or reconciliation, fencing, explicit compensation, durable coordination, and adversarial crash testing—not from replay alone.

### Cost & Telemetry Hook

- [Lesson 22.2](../22-ai-economics/README.md): durable history, timers, and retained state are fixed cost; activity retries are marginal cost — count both per completed workflow.
- [Lesson 22.3](../22-ai-economics/README.md): bill model/tool calls per activity attempt; replay must not re-record cost for attempts already in history.
- [Lesson 23.2](../23-observability-reliability/README.md): propagate trace context across activity retries and worker takeover; emit state/activity/effect telemetry keyed by operation ID so replay does not duplicate spans.
- [Lesson 23.4](../23-observability-reliability/README.md): burn alerts on unreconciled effects and history-limit warnings, not only on workflow failures.

## 12 Competency Targets

```yaml
competency:
  sfia: 5
  bloom: Evaluate -> Create
  solo: Relational -> Extended Abstract
  dreyfus: Competent
```
