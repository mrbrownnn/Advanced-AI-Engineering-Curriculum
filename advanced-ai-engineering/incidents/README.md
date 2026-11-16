# Incident Exercises

Incidents are diagnostic scenarios that test engineering reasoning under realistic constraints.

- They are not puzzles with a single correct answer.
- They require judgment, measurement, and trade-off analysis.

## Structure

```
incidents/
├── README.md         ← this file
├── template.md       ← incident format
└── examples/
    └── README.md     ← index of example incidents
```

## How Incidents Work

1. The learner receives a **situation** with known facts, unknowns, constraints, and SLOs
2. The solution is **not** provided
3. The learner must produce:
   - Hypotheses
   - Hypotheses ordered by information gain / cost of checking
   - Missing instrumentation they would add
   - Quantitative estimates
   - Discriminating experiments
   - Findings
   - Architecture decision
   - Trade-offs
   - Rollback criteria

## When to Use

- As module checkpoints (applied to that module's domain)
- As capstone integration exercises
- As graduation assessment scenarios
- As cross-module exercises connecting multiple systems
