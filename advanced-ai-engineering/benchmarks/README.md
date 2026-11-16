# Benchmarks

This directory contains the methodology, specification formats, and results for all benchmarks in the curriculum.

## Structure

```
benchmarks/
├── README.md           ← this file
├── methodology.md      ← how to design and run benchmarks
├── workload-spec.md    ← how to define workloads
├── result-schema.md    ← required fields for results
└── results/
    └── README.md       ← index of all benchmark results
```

## Core Principle

**Results are append-only evidence. Never overwrite inconvenient results.**

Every benchmark result is a data point.

- Unexpected results are often more informative than expected ones.
- If a result contradicts your hypothesis, that is signal:
  - Record it.
  - Explain it.
  - Learn from it.

## Quick Reference

- Before running: read [methodology.md](methodology.md)
- Define your workload: follow [workload-spec.md](workload-spec.md)
- Record results: follow [result-schema.md](result-schema.md)
- Store results: append to `results/`
