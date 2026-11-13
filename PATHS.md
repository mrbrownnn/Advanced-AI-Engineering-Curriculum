# Learning Paths

Three ways through the core curriculum. Hours are each module's declared `estimated_effort.total`; bridge lessons are the prerequisite lessons a path's modules cite from modules the path skips (check each module's `## 01 Baseline Assumptions`). Every module README has an **If You Only Have One Week** line naming its must-read lessons.

Paths are orderings, not new content. Labs, incidents, and mastery assessments stay as written; a path that skips a module also skips its evidence requirements.

---

## 1. Application AI Engineer — ≈16 weeks, ≈465h (≈29 h/week)

Builds retrieval, agent, tool, evaluation, and product systems on hosted or self-hosted models. Skips GPU internals, distributed inference, and model-system co-design.

| Order | Module | Hours |
|---:|---|---:|
| 1 | [00 — Scientific AI Engineering](advanced-ai-engineering/00-scientific-ai-engineering/) | 23 |
| 2 | [04 — Serving, Scheduling, & Capacity](advanced-ai-engineering/04-serving-scheduling-capacity/) | 25.5 |
| 3 | [07 — Model Behavior & Uncertainty](advanced-ai-engineering/07-model-behavior-uncertainty/) | 33 |
| 4 | [08 — AI Data Engineering](advanced-ai-engineering/08-ai-data-engineering/) | 31 |
| 5 | [09 — Retrieval Engineering](advanced-ai-engineering/09-retrieval-engineering/) | 37 |
| 6 | [10 — Advanced RAG](advanced-ai-engineering/10-advanced-rag/) | 31 |
| 7 | [11 — Context & Memory Engineering](advanced-ai-engineering/11-context-memory-engineering/) | 31 |
| 8 | [12 — Agent Loop Engineering](advanced-ai-engineering/12-agent-loop-engineering/) | 24 |
| 9 | [13 — Harness Engineering](advanced-ai-engineering/13-harness-engineering/) | 24 |
| 10 | [14 — Durable Agent Runtime](advanced-ai-engineering/14-durable-agent-runtime/) | 24 |
| 11 | [37 — Multi-Agent Systems](advanced-ai-engineering/37-multi-agent-systems/) | 24 |
| 12 | [38 — Tool Ecosystem & Protocols](advanced-ai-engineering/38-tool-ecosystem-protocols/) | 24 |
| 13 | [15 — Evaluation Engineering](advanced-ai-engineering/15-evaluation-engineering/) | 24 |
| 14 | [17 — Harness Evolution](advanced-ai-engineering/17-harness-evolution/) | 24 |
| 15 | [18 — AI Security](advanced-ai-engineering/18-ai-security/) | 26 |
| 16 | [39 — Feedback Loop & AI Product](advanced-ai-engineering/39-feedback-loop-ai-product/) | 24 |
| 17 | [22 — AI Economics](advanced-ai-engineering/22-ai-economics/) | 24.5 |
| | **Modules** | **454** |

**Bridge lessons (≈11h):** 1.1–1.4 (needed by 07, 11), 3.6 (38), 16.1 (18), 19.4–19.5 (39), 23.2 and 23.5 (39).

---

## 2. Inference / Platform AI Engineer — ≈14 weeks, ≈357h (≈25.5 h/week)

Owns serving cost, latency, capacity, and hardware placement. Skips retrieval, agents, and product feedback.

| Order | Module | Hours |
|---:|---|---:|
| 1 | [00 — Scientific AI Engineering](advanced-ai-engineering/00-scientific-ai-engineering/) | 23 |
| 2 | [01 — Foundation Model Internals](advanced-ai-engineering/01-foundation-model-internals/) | 26 |
| 3 | [02 — Inference GPU Fundamentals](advanced-ai-engineering/02-inference-gpu-fundamentals/) | 30 |
| 4 | [03 — KV Cache Engineering](advanced-ai-engineering/03-kv-cache-engineering/) | 26 |
| 5 | [04 — Serving, Scheduling, & Capacity](advanced-ai-engineering/04-serving-scheduling-capacity/) | 25.5 |
| 6 | [05 — Inference Optimization](advanced-ai-engineering/05-inference-optimization/) | 30.75 |
| 7 | [06 — Reasoning & Test-Time Compute](advanced-ai-engineering/06-reasoning-test-time-compute/) | 30.5 |
| 8 | [15 — Evaluation Engineering](advanced-ai-engineering/15-evaluation-engineering/) | 24 |
| 9 | [20 — Distributed Inference](advanced-ai-engineering/20-distributed-inference/) | 24.5 |
| 10 | [21 — Multimodal AI Systems](advanced-ai-engineering/21-multimodal-ai-systems/) | 24.5 |
| 11 | [22 — AI Economics](advanced-ai-engineering/22-ai-economics/) | 24.5 |
| 12 | [23 — Observability & Reliability](advanced-ai-engineering/23-observability-reliability/) | 34.75 |
| 13 | [24 — Model-System Co-design](advanced-ai-engineering/24-model-system-codesign/) | 25.5 |
| | **Modules** | **349.5** |

**Bridge lessons (≈7.5h):** 8.6 (23), 12.1 (22), 16.1 (23), 17.2 and 17.5 (22, 23), 18.1 (23), 19.2 (21).

Natural continuation: Expert Track Modules 25–28 ([EXPERT-TRACK.md](advanced-ai-engineering/EXPERT-TRACK.md)).

---

## 3. Full Track — 26 weeks, ≈845–860h (≈32–33 h/week)

All core Modules 00–24 and 37–39 in the order of [ROADMAP.md](ROADMAP.md), plus continuous capstone work, the capstone war week, and graduation (with the [design-doc guide](advanced-ai-engineering/graduation/design-docs.md)).

| Component | Hours |
|---|---:|
| Modules 00–24 | 683.25 |
| Modules 37–39 | 72 |
| Capstone war week + graduation | ≈50 |
| Continuous capstone (weeks 6–24) | ≈40–55 |
| **Total** | **≈845–860** |

> **Workload warning.** This is ≈70–80h over the original 730–780h budget and needs ≈32–33 h/week for 26 weeks, with Modules 07, 09, and 23 running longer than a week. If that is not sustainable: stretch to ≈30 weeks at ≈28 h/week, or finish one of the paths above first and add the remaining modules afterward. Do not compress by skipping labs or incidents; that removes the evidence the depth contract requires.

---

## Choosing

- Building products on models you do not train or host at scale → **Application**.
- Running, sizing, or optimizing model serving → **Inference / Platform**.
- Owning the whole chain, or targeting the Expert Track → **Full**.
