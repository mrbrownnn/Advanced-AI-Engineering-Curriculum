# Roadmap — 26 Weeks

## Three Tracks

| Track | Depth | Duration |
|-------|-------|----------|
| **Core** | Deep professional competency | 26 weeks continuous |
| **Satellite** | 80/20 architectural literacy | Integrated months 2–5 |
| **Paper Radar** | Continuous literature awareness | Ongoing |

> **Satellite tracks are NOT prerequisites for most core modules.** They provide breadth alongside the core's depth.

---

## Core Track Schedule

| Week | Module | Focus |
|------|--------|-------|
| 1 | [00 — Scientific AI Engineering](advanced-ai-engineering/00-scientific-ai-engineering/) | Measurement, reproducibility, experimental design, placement |
| 1–2 | [01 — Foundation Model Internals](advanced-ai-engineering/01-foundation-model-internals/) | Decoder architecture, attention variants, parameter and FLOP accounting |
| 2–3 | [02 — Inference GPU Fundamentals](advanced-ai-engineering/02-inference-gpu-fundamentals/) | GPU execution, memory hierarchy, Roofline, profiling, latency metrics |
| 3–4 | [03 — KV Cache Engineering](advanced-ai-engineering/03-kv-cache-engineering/) | KV lifecycle, PagedAttention, fragmentation, prefix/radix caching |
| 4–5 | [04 — Serving, Scheduling, & Capacity](advanced-ai-engineering/04-serving-scheduling-capacity/) | Batching, continuous batching, queueing, SLO-aware scheduling |
| 5–6 | [05 — Inference Optimization](advanced-ai-engineering/05-inference-optimization/) | FlashAttention, quantization, speculative decoding, Triton |
| 6–7 | [06 — Reasoning & Test-Time Compute](advanced-ai-engineering/06-reasoning-test-time-compute/) | Chain-of-thought, test-time scaling, reasoning cost, verification |
| 7–8 | [07 — Model Behavior & Uncertainty](advanced-ai-engineering/07-model-behavior-uncertainty/) | Failure taxonomy, calibration, selective prediction, regression |
| 8–9 | [08 — AI Data Engineering](advanced-ai-engineering/08-ai-data-engineering/) | Provenance, validation, drift, synthetic data, active learning |
| 9–10 | [09 — Retrieval Engineering](advanced-ai-engineering/09-retrieval-engineering/) | BM25, HNSW, hybrid retrieval, reranking, metrics |
| 10–11 | [10 — Advanced RAG](advanced-ai-engineering/10-advanced-rag/) | Incremental indexing, freshness, contradiction, RAG vs long context |
| 11–12 | [11 — Context & Memory Engineering](advanced-ai-engineering/11-context-memory-engineering/) | Context budgeting, memory types, long-context degradation |
| 12–13 | [12 — Agent Loop Engineering](advanced-ai-engineering/12-agent-loop-engineering/) | Agent patterns, tool integration, control, loop pathology |
| 13 | [13 — Harness Engineering](advanced-ai-engineering/13-harness-engineering/) | Prompt engineering as software, structured output, reliability patterns |
| 13–14 | [14 — Durable Agent Runtime](advanced-ai-engineering/14-durable-agent-runtime/) | Checkpoint/resume, idempotency, fan-out/join, HITL |
| 14–15 | [37 — Multi-Agent Systems](advanced-ai-engineering/37-multi-agent-systems/) | When multiple agents beat one, topologies, context passing, cascading failure, token multiplication |
| 15–16 | [38 — Tool Ecosystem & Protocols](advanced-ai-engineering/38-tool-ecosystem-protocols/) | MCP and tool protocols, tool contracts, registry and versioning, per-tool auth |
| 16 | [15 — Evaluation Engineering](advanced-ai-engineering/15-evaluation-engineering/) | Evaluation taxonomy, judge calibration, offline-online mismatch |
| 16–17 | [16 — Falsification Engineering](advanced-ai-engineering/16-falsification-engineering/) | Property testing, metamorphic testing, adversarial evaluation |
| 17–18 | [17 — Harness Evolution](advanced-ai-engineering/17-harness-evolution/) | Migration, prompt optimization, routing, harness lifecycle |
| 18–19 | [18 — AI Security](advanced-ai-engineering/18-ai-security/) | Threat modeling, prompt injection, capability scoping |
| 19 | [19 — Model Adaptation](advanced-ai-engineering/19-model-adaptation/) | SFT, LoRA, DPO, data flywheel, adapt vs prompt/retrieve/route |
| 19–20 | [39 — Feedback Loop & AI Product](advanced-ai-engineering/39-feedback-loop-ai-product/) | Explicit/implicit feedback, feedback bias, feedback to eval and training data, A/B testing, latency/trust UX |
| 20–21 | [20 — Distributed Inference](advanced-ai-engineering/20-distributed-inference/) | TP, PP, DP, EP, interconnects, MoE placement |
| 21–22 | [21 — Multimodal AI Systems](advanced-ai-engineering/21-multimodal-ai-systems/) | Multimodal integration, cross-track synthesis |
| 22–23 | [22 — AI Economics](advanced-ai-engineering/22-ai-economics/) | Cost modeling, routing, Pareto frontier |
| 23–24 | [23 — Observability & Reliability](advanced-ai-engineering/23-observability-reliability/) | Tracing, SLO, alerting, incident response |
| 24 | [24 — Model-System Co-design](advanced-ai-engineering/24-model-system-codesign/) | Architecture-model coupling end-to-end |
| 25 | [Capstone](advanced-ai-engineering/capstone/) | War week — integration, load testing, failure injection |
| 26 | [Graduation](advanced-ai-engineering/graduation/) | Architecture review under ambiguous constraints |

> Weeks are hour-packed from [advanced-ai-engineering/ROADMAP.md](advanced-ai-engineering/ROADMAP.md); a range means the module spans a week boundary. Modules 37–39 are numbered after the Expert Track reservation (25–36) but scheduled where they belong: 37–38 after Module 14, 39 after Module 19.
>
> **Workload warning:** the full track is ≈845–860h (≈32–33 h/week). Shorter role-based paths are in [PATHS.md](PATHS.md).
>
> **Cross-cutting:** Modules 22 (Economics) and 23 (Observability) have dedicated weeks, but each module 00–21 ends with a *Cost & Telemetry Hook* naming the 22/23 lessons that apply to its artifacts. Module 24 integrates both.

**Expert Track (post-graduation, open-ended):** Modules 25–36 — training systems, deep kernels/numerics, non-attention architectures, compression, interpretability, privacy, edge and computer-use agents. See [EXPERT-TRACK.md](advanced-ai-engineering/EXPERT-TRACK.md).

---

## Satellite Track Integration

| Month | Satellite Track | Core Connection |
|-------|----------------|-----------------|
| 1 | — (Core only) | Build engineering foundations |
| 2 | [Vision-Language Models](vision-language-models/) | Pairs with inference and context engineering |
| 3 | [Speech AI](speech-ai/) | Pairs with streaming, real-time serving, agents |
| 4 | [Generative Media](generative-media/) | Pairs with GPU compute, serving, economics |
| 5 | [World Models & Embodied AI](world-models-embodied-ai/) | Pairs with reasoning, planning, agents |
| 6 | Integration through [Module 21](advanced-ai-engineering/21-multimodal-ai-systems/) and Capstone | Synthesis |

---

## Capstone

Capstone development runs continuously from Week 6. Each module produces artifacts that feed into the capstone AI Runtime Platform.

## Paper Radar

Continuous literature awareness maintained through the [paper-library/](paper-library/). New papers are triaged using the CANONICAL / PRODUCTION / FRONTIER / HISTORICAL / UNVALIDATED classification.
