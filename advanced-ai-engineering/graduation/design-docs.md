# AI System Design Docs

A graduation design doc is a reviewable decision, not an architecture inventory. State the problem, show evidence, compare viable options, and explain how the proposed system will be released and reversed. Use the [fill-in template](../templates/design-doc.md); keep supporting experiments and raw results linked rather than duplicating them.

## 1. Problem and Decision Boundary

Name the user outcome, current failure, decision owner, reviewers, and deadline. Separate goals, non-goals, hard constraints, and assumptions. State what would make “no change” preferable. Draw or describe the current boundary, proposed boundary, data paths, authority, and external dependencies. Every proposed component should answer a requirement or a measured failure.

**Review check:** Can a reviewer identify the decision being requested and reject scope creep without reading implementation details?

## 2. Workload and SLOs

Specify request/task populations, arrival patterns and bursts, prompt/output/context distributions, query slices, tenant isolation, data freshness, retention, and growth scenarios. Distinguish measured workload from synthetic planning inputs. Define latency clocks, success predicates, quality gates, eligible denominator, and treatment of rejection, retry, cancellation, failure, and unfinished work.

Use [Lesson 4.7](../04-serving-scheduling-capacity/README.md#lesson-47--capacity-planning--bottleneck-discrimination) for capacity assumptions and [Lesson 23.3](../23-observability-reliability/README.md#lesson-233--offered-outcome-slis) for offered-outcome reliability. State hard limits separately from optimization preferences. Unknown service curves should remain unknown or conditional, not become an exact GPU count.

**Review check:** Are acceptance criteria measurable on the actual offered population, including important low-volume slices?

## 3. Evidence and Uncertainty

Keep an evidence ledger with one substantive claim per row:

| ID | O / D / H | Claim | Source or artifact | Conditions / assumptions | Limits / falsifier |
|---|---|---|---|---|---|
| _ | _ | _ | _ | _ | _ |

- **O — observation:** inspected source or actual measurement. Include revision, method, population, hardware/model/workload where relevant, and raw evidence. Author-reported results are not your reproduction.
- **D — derivation:** explicit inputs, units, assumptions, equation, and recomputable arithmetic. Label synthetic inputs.
- **H — hypothesis:** predicted effect, discriminating measurement, and observation that would falsify it.

Mark unresolved facts `TODO_VERIFY`; assign an owner and next check. Separate missing evidence that blocks release from uncertainty that can be accepted temporarily. Record contradictory evidence rather than selecting only favorable results.

**Review check:** Can another engineer reproduce the reasoning and distinguish source facts from predictions?

## 4. Options and Alternatives Considered

Describe the proposed design and at least one genuinely viable alternative, including the current system/no-change option where applicable. Compare options against the same workload, quality, SLO, security, cost, operational, and migration constraints. Include failure modes and required evidence, not only feature lists.

The mandatory **Alternatives Considered** section records rejection reasons, evidence IDs, and conditions that would reverse the decision. “Too complex” needs a concrete maintenance, latency, cost, or risk consequence. Do not invent certainty when two options remain indistinguishable: propose the smallest discriminating experiment.

**Review check:** Would the recommendation still hold if a major assumption changed? State the flip point.

## 5. Cost and Observability Plans

Link a cost worksheet using [Module 22](../22-ai-economics/README.md), especially Lessons 22.1–22.3 and 22.6. Declare horizon and currency/date for prices, fixed versus marginal and fully loaded costs, idle/redundant capacity, training/indexing, telemetry, human operations, failed/retried work, and cost per successful task. Include sensitivity and a budget guardrail; do not compare options with different success denominators.

Link a signal dictionary and runbook using [Module 23](../23-observability-reliability/README.md). For every critical failure mode, state detection signal, boundary, window, owner, alert threshold, and discriminating evidence. Include schema versions, trace propagation/sampling, dropped telemetry, privacy/cardinality budgets, delayed labels, and offered/terminal reconciliation. For elastic serving, include desired versus useful capacity and startup-phase/cache-state telemetry from Lessons 23.7–23.8.

**Review check:** Which missing metric could make the design look healthy while users fail, and how will the system expose that gap?

## 6. Evaluation, Release Gates, and Security

Use [Module 15](../15-evaluation-engineering/README.md), particularly Lessons 15.1–15.4, for experimental units, dataset lineage, grader validity, uncertainty, slices, and predeclared release gates. Use [Lesson 17.5](../17-harness-evolution/README.md#lesson-175--shadow-canary-and-staged-rollout) for shadow/canary stages. Specify baseline/candidate manifests, holdout protection, minimum evidence, guardrails, pass/fail/inconclusive outcomes, and who may approve progression.

Link the threat model and security tests to [Module 18](../18-ai-security/README.md), particularly Lessons 18.1, 18.4–18.6. Name trust boundaries, untrusted inputs, data permissions, tool authority, secrets, output handling, egress, resource limits, and telemetry access/retention. Define a concrete blocking security gate.

**Review check:** Could a favorable aggregate metric hide a quality, authorization, or safety regression that must block release?

## 7. Rollout and Mandatory Rollback Plan

List ordered rollout stages, traffic scope, stage duration or required evidence, responsible owner, and pause/abort conditions. A rollout stage must have a stop rule and a known-safe fallback.

The mandatory **Rollback Plan** must include:

1. **Explicit triggers:** metric/security condition, population/slice, threshold, observation window, minimum evidence, and handling of telemetry loss. Name automatic versus human decisions and the responding owner.
2. **Action and deadline:** traffic shift, feature disable, prior manifest restore, or another safe mitigation; target response time; verification signals; escalation if the action fails.
3. **State compatibility:** following [Lesson 17.6](../17-harness-evolution/README.md#lesson-176--compatibility-rollback-state-and-the-migration-ledger), check model/tokenizer/index/schema/prompt/cache versions, persisted workflow and tool state, in-flight work, duplicate effects, and data migrations. Name backward-readable state, required transforms, irreversible effects, and whether the old release can safely resume.
4. **Rehearsal evidence:** perform rollback on representative state before progression. Preserve ledgers and manifests. If rollback is unsafe, state why and specify the tested forward fix, isolation, or degradation path instead; do not label an untested deploy command a rollback plan.
5. **Recovery check:** repeat the original offered-outcome and slice gates, reconcile state and external effects, and retain evidence before closing the incident.

**Review check:** Can the on-call engineer act without improvising the trigger, target version, or compatibility decision?

## 8. Open Questions and Decision Record

Finish with unresolved questions, owner, due date, next measurement, impact if wrong, and whether each blocks approval. Record the chosen option, dissent, accepted risks, review date, and re-evaluation triggers. A design doc can request an experiment instead of production approval when evidence is insufficient.

## Submission Check

Submit the completed [design-doc template](../templates/design-doc.md) with linked evidence and experiments. Required review surface: workload/SLO contract, O/D/H ledger, viable options and rejection reasons, cost and observability plans, eval/security gates, staged rollout, explicit rollback triggers and state compatibility, and owned open questions. “TBD” in a release-blocking field means the release decision is not ready.
