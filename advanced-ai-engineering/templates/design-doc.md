# Design Doc / RFC — _System and Decision_

Use the [design-doc guide](../graduation/design-docs.md). Replace placeholders with evidence or an owned open question; state “not applicable” with a reason rather than deleting mandatory sections.

- **Status:** _Draft / Experiment requested / In review / Accepted / Rejected / Superseded_
- **Owner / reviewers / approver:** _
- **Decision date / revision:** _
- **Decision requested:** _
- **Related manifests, experiments, and prior decisions:** _

## Problem, Goals, and Non-Goals

- User outcome and current failure: _
- Goals and measurable acceptance criteria: _
- Non-goals and excluded system boundaries: _
- Hard constraints and dependencies: _
- Why now; when no change is preferable: _

## Workload and SLOs

| Property | Value / distribution / units | Measured or synthetic | Evidence / uncertainty |
|---|---|---|---|
| Eligible requests/tasks and slices | _ | _ | _ |
| Arrival rate, bursts, growth | _ | _ | _ |
| Input/output/context lengths | _ | _ | _ |
| Freshness, tenant and data boundaries | _ | _ | _ |
| Latency clock, threshold and window | _ | _ | _ |
| Success and quality predicate | _ | _ | _ |
| Failure/rejection/retry/cancellation/unfinished policy | _ | _ | _ |

Capacity assumptions and unknowns: _ ([Lesson 4.7](../04-serving-scheduling-capacity/README.md#lesson-47--capacity-planning--bottleneck-discrimination)). Offered denominator and reconciliation: _ ([Lesson 23.3](../23-observability-reliability/README.md#lesson-233--offered-outcome-slis)).

## Evidence Ledger — O / D / H

O = inspected observation; D = derivation from explicit assumptions; H = falsifiable hypothesis. Distinguish author-reported results from reproduced measurements. Recompute arithmetic and label synthetic inputs.

| ID | O / D / H | Claim | Source/revision or raw artifact | Conditions, units, assumptions | Limits / falsifying measurement |
|---|---|---|---|---|---|
| _ | _ | _ | _ | _ | _ |

Contradictory evidence and `TODO_VERIFY` items: _

## Proposed Design and Options

Current boundary and proposed components/data/authority flows: _

| Option | Workload/SLO fit | Quality/security constraints | Cost and operational burden | Failure modes / missing evidence |
|---|---|---|---|---|
| Proposed | _ | _ | _ | _ |
| Viable alternative | _ | _ | _ | _ |
| Current system / no change | _ | _ | _ | _ |

Chosen option and decision rationale: _

## Alternatives Considered

Mandatory: include concrete rejection reasons, not a list of names.

| Alternative | Why viable | Rejection reason and evidence IDs | Assumption / flip point that would reopen it |
|---|---|---|---|
| _ | _ | _ | _ |

Smallest experiment needed to distinguish unresolved options: _

## Cost Model

Reference: [Module 22](../22-ai-economics/README.md), Lessons 22.1–22.3 and 22.6.

- Worksheet, horizon, dated prices/currency, and units: _
- Success denominator; fixed/marginal/fully loaded terms: _
- Idle and redundant capacity, training/indexing, failures/retries, telemetry and operations: _
- Sensitivity/flip points and budget guardrail: _
- Assumptions excluded from the model and decision impact: _

## Observability Plan

Reference: [Module 23](../23-observability-reliability/README.md).

| Failure / hypothesis | Signal boundary and population | Window / threshold | Owner / response | Discriminating evidence |
|---|---|---|---|---|
| _ | _ | _ | _ | _ |

- Signal dictionary/schema revisions and offered/terminal ledger: _
- Trace propagation/sampling, loss/completeness and delayed labels: _
- Privacy, cardinality, access and retention limits: _
- If elastic: useful versus desired capacity; startup phases/cache state; warm/sleep/zero policy: _
- Known telemetry blind spots and fallback detection: _

## Evaluation and Release Gates

References: [Module 15](../15-evaluation-engineering/README.md), Lessons 15.1–15.4; [Module 17](../17-harness-evolution/README.md), Lessons 17.1, 17.5–17.6.

- Baseline/candidate manifests and protected datasets/holdouts: _
- Experimental unit, paired comparison, slices, grader validation and uncertainty: _

| Gate | Metric / slice / threshold | Required evidence and window | Pass / fail / inconclusive action | Approver |
|---|---|---|---|---|
| Quality | _ | _ | _ | _ |
| Latency/reliability | _ | _ | _ | _ |
| Cost | _ | _ | _ | _ |
| Security | _ | _ | _ | _ |

## Security Plan

Reference: [Module 18](../18-ai-security/README.md), Lessons 18.1, 18.4–18.6.

- Threat model, assets, trust boundaries, untrusted inputs: _
- Identity, authorization, tool authority, secrets and tenant/data isolation: _
- Output handling, sandbox/egress/resource limits: _
- Security tests and blocking release condition: _
- Telemetry data minimization/access/retention: _

## Rollout Plan

| Ordered stage | Traffic / population | Duration or evidence required | Progress / pause / abort rule | Owner |
|---|---|---|---|---|
| Shadow | _ | _ | _ | _ |
| Canary | _ | _ | _ | _ |
| Expansion | _ | _ | _ | _ |

## Rollback Plan

Mandatory. Reference: [Lesson 17.6 — Compatibility, Rollback State, and the Migration Ledger](../17-harness-evolution/README.md#lesson-176--compatibility-rollback-state-and-the-migration-ledger).

### Explicit Rollback Triggers

| Trigger | Population/slice and signal | Threshold, window, minimum evidence | Automatic or human; owner | Action / response deadline |
|---|---|---|---|---|
| Quality/reliability regression | _ | _ | _ | _ |
| Security/authorization breach | _ | _ | _ | _ |
| Telemetry loss / unverifiable health | _ | _ | _ | _ |

### State Compatibility and Safe Target

- Exact target manifest/version and whether old code can read new state: _
- Model/tokenizer/index/schema/prompt/cache compatibility: _
- Persisted workflows, in-flight requests, retries and duplicate external effects: _
- Data migrations, required transforms, irreversible effects: _
- Admission/drain/traffic-shift steps; owner and deadline: _
- If rollback is unsafe: tested forward-fix/isolation/degradation alternative and escalation: _

### Rehearsal and Recovery Verification

- Rehearsal artifact on representative migrated state: _
- Post-action offered-outcome and slice checks; state/effect reconciliation: _
- Evidence/manifests preserved and incident closure condition: _
- Fallback when rollback itself fails: _

## Open Questions and Accepted Risks

| Question / risk | Owner | Due date | Next measurement / action | Blocks approval? | Impact if wrong |
|---|---|---|---|---|---|
| _ | _ | _ | _ | _ | _ |

## Decision Record

- Approved decision or requested experiment: _
- Rationale and evidence IDs: _
- Dissent and accepted risks with owners: _
- Re-evaluation triggers and next review date: _
