# Intensive AI Engineer

> A competency-driven journey through modern AI engineering, from models and data to agents, inference systems, and multimodal AI.

## Program Architecture

This program has three layers:

### Layer 1 — Core / Deep Track

**[advanced-ai-engineering/](advanced-ai-engineering/)** — The 26-week professional curriculum (≈845–860h; [PATHS.md](PATHS.md) lists shorter paths) developing deep competency across the full AI engineering chain: from scientific experimentation and model internals through inference systems, agents, evaluation, security, and model-system co-design.

### Layer 2 — Satellite / 80-20 Tracks

Architectural literacy across the broader AI landscape:

| Track | Purpose |
|-------|---------|
| [vision-language-models/](vision-language-models/) | Image understanding, VLM architectures, multimodal serving |
| [speech-ai/](speech-ai/) | Audio representation, ASR, TTS, voice agent systems |
| [generative-media/](generative-media/) | Diffusion, flow matching, image and video generation |
| [world-models-embodied-ai/](world-models-embodied-ai/) | World models, planning, embodied agents, sim-to-real |

**80/20 principle**: Learn the smallest set of concepts that enables reading new papers, analyzing production systems, and identifying what additional knowledge would be needed to specialize.

Satellite tracks are **NOT** superficial keyword surveys. They reach architectural literacy.

### Layer 3 — Paper Radar

**[paper-library/](paper-library/)** — Continuous literature tracking across all AI domains. Curated, classified, and connected to curriculum modules.

---

## The Engineering Evidence Philosophy

This program rejects "topic coverage." Understanding an API or reading a paper is insufficient. The curriculum is governed by a strict [Depth Contract](advanced-ai-engineering/DEPTH-CONTRACT.md), demanding empirical engineering evidence for every completed module.

### Central Engineering Loop

Defined fully in the [Engineering Learning Protocol](advanced-ai-engineering/ENGINEERING-LEARNING-PROTOCOL.md):

```
BUILD → MEASURE → BREAK → EXPLAIN → IMPROVE → DEFEND
```

A system merely "working" is not considered completion. You must break it, diagnose the failure, and defend the trade-offs of your fix.

### Central Literature Loop

```
READ → REPRODUCE → CHALLENGE → CONNECT → APPLY
```

A paper merely "read" is not considered understood. You must reproduce its core mechanism and identify where it fails.

---

## What This Is NOT

- ❌ An attempt to become an expert in every AI subfield
- ❌ A beginner AI course
- ❌ A framework tutorial collection
- ❌ A paper-reading list
- ❌ An LLM application tutorial
- ❌ An interview-cramming repository

## What This IS

**Deep expertise** in the AI Engineering core — inference, serving, agents, evaluation, security, economics, reliability

**+ Architectural literacy** across the broader AI landscape — vision, speech, generative media, world models

---

## The Learner Should Repeatedly Answer

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
Intensive AI Engineer/
│
├── README.md                       ← this file
├── ROADMAP.md                      ← 26-week schedule with satellite integration
├── COMPETENCY.md                   ← SFIA, Bloom, SOLO, Dreyfus framework
│
├── PATHS.md                        ← Application, Inference/Platform, and Full learning paths
├── advanced-ai-engineering/        ← CORE: 32 modules (00–24 and 37–39 core, 25–28 expert)
│   ├── DEPTH-CONTRACT.md           ← Defines the standard for mastery vs topic coverage
│   └── ENGINEERING-LEARNING-PROTOCOL.md ← The rigorous loops used to execute modules
│
├── vision-language-models/         ← SATELLITE: VLM 80/20
├── speech-ai/                      ← SATELLITE: Speech 80/20
├── generative-media/               ← SATELLITE: Generative media 80/20
├── world-models-embodied-ai/       ← SATELLITE: World models 80/20
│
├── paper-library/                  ← Shared paper index across all tracks
├── templates/                      ← Shared templates
└── references/                     ← Quick reference material
```

## Getting Started

1. Read [ROADMAP.md](ROADMAP.md) for the 26-week schedule and [PATHS.md](PATHS.md) to choose a path
2. Read [COMPETENCY.md](COMPETENCY.md) for the competency framework
3. Read [DEPTH-CONTRACT.md](advanced-ai-engineering/DEPTH-CONTRACT.md) and [ENGINEERING-LEARNING-PROTOCOL.md](advanced-ai-engineering/ENGINEERING-LEARNING-PROTOCOL.md) to understand the required rigor.
4. Start with [advanced-ai-engineering/](advanced-ai-engineering/) Module 00
5. Satellite tracks integrate starting Month 2

## Status

32 modules exist: 28 core (00–24 and 37–39) and 4 expert (25–28). Modules 37–39 are numbered after the Expert Track reservation but scheduled inside the core weeks (see [ROADMAP.md](ROADMAP.md)). Modules 29–36 are scope contracts in [EXPERT-TRACK.md](advanced-ai-engineering/EXPERT-TRACK.md), not content. AUTHORED means the README, research registry, and knowledge model pass `advanced-ai-engineering/tools/validate_curriculum.py`; it does not certify technical correctness. Expert-module lab packages are deferred.

| Module | Track | Status |
|---|---|---|
| 00 — Scientific AI Engineering | Core | AUTHORED |
| 01 — Foundation Model Internals | Core | AUTHORED |
| 02 — Inference & GPU Fundamentals | Core | AUTHORED |
| 03 — KV Cache Engineering | Core | AUTHORED |
| 04 — Serving, Scheduling, Capacity | Core | AUTHORED |
| 05 — Inference Optimization | Core | AUTHORED |
| 06 — Reasoning & Test-Time Compute | Core | AUTHORED |
| 07 — Model Behavior & Uncertainty | Core | AUTHORED |
| 08 — AI Data Engineering | Core | AUTHORED |
| 09 — Retrieval Engineering | Core | AUTHORED |
| 10 — Advanced RAG | Core | AUTHORED |
| 11 — Context & Memory Engineering | Core | AUTHORED |
| 12 — Agent Loop Engineering | Core | AUTHORED |
| 13 — Harness Engineering | Core | AUTHORED |
| 14 — Durable Agent Runtime | Core | AUTHORED |
| 15 — Evaluation Engineering | Core | AUTHORED |
| 16 — Falsification Engineering | Core | AUTHORED |
| 17 — Harness Evolution | Core | AUTHORED |
| 18 — AI Security | Core | AUTHORED |
| 19 — Model Adaptation | Core | AUTHORED |
| 20 — Distributed Inference | Core | AUTHORED |
| 21 — Multimodal AI Systems | Core | AUTHORED |
| 22 — AI Economics | Core | AUTHORED |
| 23 — Observability & Reliability | Core | AUTHORED |
| 24 — Model-System Co-design | Core | AUTHORED |
| 37 — Multi-Agent Systems | Core | AUTHORED |
| 38 — Tool Ecosystem & Protocols | Core | AUTHORED |
| 39 — Feedback Loop & AI Product | Core | AUTHORED |
| 25 — Pretraining Dynamics & Scaling | Expert | AUTHORED |
| 26 — Distributed & Low-Precision Training | Expert | AUTHORED |
| 27 — RL Post-Training Systems | Expert | AUTHORED |
| 28 — MoE Training Dynamics | Expert | AUTHORED |
| 29 — Kernel Engineering Below Triton | Expert | PROPOSED |
| 30 — Numerics, Determinism & Reproducibility | Expert | PROPOSED |
| 31 — Beyond Dense Attention | Expert | PROPOSED |
| 32 — Compression: Distillation, Pruning, Merging | Expert | PROPOSED |
| 33 — Mechanistic Interpretability for Engineers | Expert | PROPOSED |
| 34 — Memorization, Privacy & Unlearning | Expert | PROPOSED |
| 35 — Multi-Tenant Adapter & Long-Context Serving | Expert | PROPOSED |
| 36 — Edge Inference & Computer-Use Agents | Expert | PROPOSED |
