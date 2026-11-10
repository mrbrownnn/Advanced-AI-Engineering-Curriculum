# Roadmap — 26 Weeks

The schedule is planned in hours, not one module per week. The global budget is 730–780 hours over 26 weeks (≈28–30 h/week). Module hours come from each module's `estimated_effort.total`.

| Week | Modules | Module hours | Cumulative |
|---|---|---:|---:|
| 1 | [00 — Scientific AI Engineering](00-scientific-ai-engineering/) | 23h | 23h |
| 1–2 | [01 — Foundation Model Internals](01-foundation-model-internals/) | 26h | 49h |
| 2–3 | [02 — Inference GPU Fundamentals](02-inference-gpu-fundamentals/) | 30h | 79h |
| 3–4 | [03 — KV Cache Engineering](03-kv-cache-engineering/) | 26h | 105h |
| 4–5 | [04 — Serving, Scheduling, & Capacity](04-serving-scheduling-capacity/) | 25.5h | 130.5h |
| 5–6 | [05 — Inference Optimization](05-inference-optimization/) | 30.75h | 161.25h |
| 6–7 | [06 — Reasoning & Test-Time Compute](06-reasoning-test-time-compute/) | 30.5h | 191.75h |
| 7–8 | [07 — Model Behavior & Uncertainty](07-model-behavior-uncertainty/) | 33h | 224.75h |
| 8–9 | [08 — AI Data Engineering](08-ai-data-engineering/) | 31h | 255.75h |
| 9–10 | [09 — Retrieval Engineering](09-retrieval-engineering/) | 37h | 292.75h |
| 10–11 | [10 — Advanced RAG](10-advanced-rag/) | 31h | 323.75h |
| 11–12 | [11 — Context & Memory Engineering](11-context-memory-engineering/) | 31h | 354.75h |
| 12–13 | [12 — Agent Loop Engineering](12-agent-loop-engineering/) | 24h | 378.75h |
| 13 | [13 — Harness Engineering](13-harness-engineering/) | 24h | 402.75h |
| 13–14 | [14 — Durable Agent Runtime](14-durable-agent-runtime/) | 24h | 426.75h |
| 14–15 | [37 — Multi-Agent Systems](37-multi-agent-systems/) | 24h | 450.75h |
| 15–16 | [38 — Tool Ecosystem & Protocols](38-tool-ecosystem-protocols/) | 24h | 474.75h |
| 16 | [15 — Evaluation Engineering](15-evaluation-engineering/) | 24h | 498.75h |
| 16–17 | [16 — Falsification Engineering](16-falsification-engineering/) | 24h | 522.75h |
| 17–18 | [17 — Harness Evolution](17-harness-evolution/) | 24h | 546.75h |
| 18–19 | [18 — AI Security](18-ai-security/) | 26h | 572.75h |
| 19 | [19 — Model Adaptation](19-model-adaptation/) | 24.75h | 597.5h |
| 19–20 | [39 — Feedback Loop & AI Product](39-feedback-loop-ai-product/) | 24h | 621.5h |
| 20–21 | [20 — Distributed Inference](20-distributed-inference/) | 24.5h | 646h |
| 21–22 | [21 — Multimodal AI Systems](21-multimodal-ai-systems/) | 24.5h | 670.5h |
| 22–23 | [22 — AI Economics](22-ai-economics/) | 24.5h | 695h |
| 23–24 | [23 — Observability & Reliability](23-observability-reliability/) | 34.75h | 729.75h |
| 24 | [24 — Model-System Co-design](24-model-system-codesign/) | 25.5h | 755.25h |
| 25 | [Capstone](capstone/) war week | ≈30h | |
| 26 | [Graduation](graduation/) | ≈20h | |

Week ranges are hour-packed at ≈31.5 module-hours per week; a range means the module spans a week boundary. Modules 37 and 38 follow Module 14 (they build on agent loops, harnesses, and durable runtimes); Module 39 follows Module 19 (it consumes evaluation and adaptation data).

## Budget Check

- Module effort, Modules 00–24 and 37–39: **755.25h** (00–24: 683.25h after the Module 09 and 23 additions; 37–39: 72h).
- Capstone war week and graduation: ≈50h.
- Continuous capstone work in weeks 6–24 at ≈2–3 h/week: ≈40–55h.
- Program total: **≈845–860h**, above the original 730–780h budget by ≈70–80h.

> **Workload warning.** The full track now needs ≈32–33 h/week for 26 weeks. If that is not sustainable, use a path in [../PATHS.md](../PATHS.md), stretch the full track to ≈30 weeks at ≈28 h/week, or defer Modules 21 and 24 to after graduation.

## Pacing Notes

- Weeks 7–12 carry the heaviest single modules (30–32h); expect those weeks to run slightly over 30h, or borrow time from the continuous capstone allowance.
- Modules 09 (37h) and 23 (34.75h) are the heaviest after the retrieval-adaptation and GPU-platform lessons; their weeks run over 31.5h.
- Capstone development runs continuously from Week 6; Week 25 is integration, load testing, failure injection, and defense.

## Post-Graduation Expert Track

Expert Track remains outside the 26-week core budget. Modules 25–28 are authored at 100.5h total; their executable lab packages are deferred. Modules 29–36 remain scoped and are added only when the ownership, durability, and breakability tests in [EXPERT-TRACK.md](EXPERT-TRACK.md) pass.

| Module | Focus | Effort |
|---|---|---:|
| [25](25-pretraining-dynamics-scaling/) | Pretraining dynamics and scaling | 25h |
| [26](26-distributed-low-precision-training/) | Distributed and low-precision training | 25h |
| [27](27-rl-post-training-systems/) | RL post-training systems | 25h |
| [28](28-moe-training-dynamics/) | MoE training dynamics | 25.5h |

## Progressive Integration

- **Evidence discipline** — methodology from Module 00 applied in every lab.
- **Observability** — introduced in Modules 02 and 04, deepened in every module, dedicated Module 23.
- **Cost & telemetry** — each module 00–21 ends with a *Cost & Telemetry Hook* naming the Module 22/23 lessons that apply to its artifacts.
- **Security** — surfaces in agents, harnesses, and retrieval before the dedicated Module 18.
- **Capstone** — absorbs artifacts from Module 04 onward.
