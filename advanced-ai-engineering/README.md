# Advanced AI Engineering

A competency-driven, 26-week advanced AI engineering program.

## What This Is

A structured curriculum for engineers at Junior+ / SFIA Level 4 targeting strong Middle / SFIA Level 5 technical capability, with selected SFIA Level 6 stretch exercises.

The program covers the complete chain from scientific experimentation through production AI systems:

```
Scientific Experimentation → Foundation Model Internals → Inference GPU Fundamentals
→ KV Cache Engineering → Serving / Scheduling / Capacity → Inference Optimization
→ Reasoning / Test-Time Compute → Model Behavior / Uncertainty → AI Data Engineering
→ Retrieval → Advanced RAG → Context / Memory → Agent Loops → Harnesses
→ Durable Agent Runtime → Evaluation → Falsification → Harness Evolution
→ AI Security → Model Adaptation → Distributed Inference → Multimodal Systems
→ AI Economics → Observability / Reliability → Model-System Co-design → Capstone
```

### Central Principle

```
BUILD → MEASURE → BREAK → EXPLAIN → IMPROVE → DEFEND
```

A system merely "working" is **not** considered completion.

## What This Is NOT

- ❌ A beginner AI course
- ❌ A framework tutorial collection
- ❌ A paper-reading list
- ❌ An LLM application tutorial
- ❌ An interview-cramming repository

## Learning Loop

Every module follows this default loop:

```
Concept → Paper / Official Documentation → Source Reading → Small Implementation
→ Instrument → Benchmark → Break → Diagnose → Optimize → Benchmark Again
→ Engineering Decision → Engineering Report
```

The learner should repeatedly answer:

- What did I assume?
- What did I predict?
- What did I measure?
- How reliable is the measurement?
- What broke?
- Why did it break?
- What evidence supports the explanation?
- What alternatives exist?
- What trade-off did I choose?
- Under what workload would the decision change?
- What is the rollback condition?

## Repository Structure

```
advanced-ai-engineering/
├── 00-scientific-ai-engineering/     Evidence discipline and experimental design
├── 01-foundation-model-internals/    Decoder architecture and accounting
├── 02-inference-gpu-fundamentals/    GPU execution, timing, and profiling
├── 03-kv-cache-engineering/          KV memory lifecycle
├── 04-serving-scheduling-capacity/   Batching, queueing, capacity (golden reference module)
├── 05-inference-optimization/        Kernels, quantization, speculative decoding
├── 06-reasoning-test-time-compute/   Test-time search, verification, budgets
├── 07-model-behavior-uncertainty/    Calibration, abstention, behavior regression
├── 08-ai-data-engineering/           Data lifecycle, lineage, contamination
├── 09-retrieval-engineering/         Search, ANN, fusion, reranking
├── 10-advanced-rag/                  Versioned evidence systems
├── 11-context-memory-engineering/    Context accounting and memory policy
├── 12-agent-loop-engineering/        Bounded agent controllers
├── 13-harness-engineering/           Model I/O contracts and validation
├── 14-durable-agent-runtime/         Durable execution and effects
├── 15-evaluation-engineering/        Evaluation as decision system
├── 16-falsification-engineering/     Systematic counterexample search
├── 17-harness-evolution/             Migration, optimization, rollout
├── 18-ai-security/                   Threat models and authority controls
├── 19-model-adaptation/              Fine-tuning decisions, data, and gates
├── 20-distributed-inference/         Parallelism and communication
├── 21-multimodal-ai-systems/         Multimodal pipelines
├── 22-ai-economics/                  Cost and routing
├── 23-observability-reliability/     Tracing, SLOs, incidents
├── 24-model-system-codesign/         Architecture-system coupling
├── 25-pretraining-dynamics-scaling/ Expert: scaling laws, μP, instabilities
├── 26-distributed-low-precision-training/ Expert: ZeRO/FSDP, FP8
├── 27-rl-post-training-systems/     Expert: RLVR, GRPO estimators, async RL
├── 28-moe-training-dynamics/        Expert: routing, load balancing
├── capstone/                         AI Runtime Platform project
├── graduation/                       Final architecture review
├── research-registry/                Evidence registries per module
├── tools/                            Static curriculum validator and fixtures
├── source-reading/                   Source code study methodology
├── benchmarks/                       Benchmark methodology and results
├── incidents/                        Diagnostic scenario templates
├── datasets/ evals/                  Supporting data and evaluation material
└── templates/                        Checkpoint, report, experiment templates
```

Repository-level material outside this folder: [paper library](../paper-library/), [references](../references/), and satellite tracks (speech, vision-language, generative media, world models).

## Getting Started

1. Read [ROADMAP.md](ROADMAP.md) for the 26-week schedule
2. Read [CURRICULUM.md](CURRICULUM.md) for the module index
3. Read [COMPETENCY.md](COMPETENCY.md) for the competency framework
4. Start with [00-scientific-ai-engineering/](00-scientific-ai-engineering/) before any other module
5. Use [Module 04](04-serving-scheduling-capacity/README.md) as the structural reference and [INSTRUCTIONAL_SCHEMA.md](INSTRUCTIONAL_SCHEMA.md) for the module contract

## Conventions

- All competency metadata uses the format defined in [COMPETENCY.md](COMPETENCY.md)
- Engineering reports follow [templates/engineering-report.md](templates/engineering-report.md)
- Checkpoints follow [templates/checkpoint.md](templates/checkpoint.md)
- Benchmark results are append-only; never overwrite inconvenient results
- Source reading exercises pin a specific commit/tag

## Status

Modules 00–24 (core) and 25–28 (expert) are authored against the [instructional schema](INSTRUCTIONAL_SCHEMA.md), each with a research registry and knowledge model. Modules 29–36 are proposed in [EXPERT-TRACK.md](EXPERT-TRACK.md). Run `python tools/validate_curriculum.py` for the static checks; passing them does not certify technical correctness.
