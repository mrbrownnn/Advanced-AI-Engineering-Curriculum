# Roadmap — 26 Weeks

The schedule is planned in hours, not one module per week. The global budget is 730–780 hours over 26 weeks (≈28–30 h/week). Module hours come from each module's `estimated_effort.total`.

| Week | Modules | Module hours | Cumulative |
|---|---|---:|---:|
| 1 | [00 — Scientific AI Engineering](00-scientific-ai-engineering/) | 23h | 23h |
| 2 | [01 — Foundation Model Internals](01-foundation-model-internals/) | 26h | 49h |
| 3 | [02 — Inference GPU Fundamentals](02-inference-gpu-fundamentals/) | 30h | 79h |
| 4 | [03 — KV Cache Engineering](03-kv-cache-engineering/) | 26h | 105h |
| 5 | [04 — Serving, Scheduling, & Capacity](04-serving-scheduling-capacity/) | 25.5h | 130.5h |
| 6 | [05 — Inference Optimization](05-inference-optimization/) | 30.75h | 161.25h |
| 7 | [06 — Reasoning & Test-Time Compute](06-reasoning-test-time-compute/) | 30.5h | 191.75h |
| 8 | [07 — Model Behavior & Uncertainty](07-model-behavior-uncertainty/) | 33h | 224.75h |
| 9 | [08 — AI Data Engineering](08-ai-data-engineering/) | 31h | 255.75h |
| 10 | [09 — Retrieval Engineering](09-retrieval-engineering/) | 30h | 285.75h |
| 11 | [10 — Advanced RAG](10-advanced-rag/) | 31h | 316.75h |
| 12 | [11 — Context & Memory Engineering](11-context-memory-engineering/) | 31h | 347.75h |
| 13–17 | [12 — Agent Loop Engineering](12-agent-loop-engineering/); [13 — Harness Engineering](13-harness-engineering/); [14 — Durable Agent Runtime](14-durable-agent-runtime/); [15 — Evaluation Engineering](15-evaluation-engineering/); [16 — Falsification Engineering](16-falsification-engineering/); [17 — Harness Evolution](17-harness-evolution/) | 144h | 491.75h |
| 18 | [18 — AI Security](18-ai-security/) | 26h | 517.75h |
| 19 | [19 — Model Adaptation](19-model-adaptation/) | 24.75h | 542.5h |
| 20 | [20 — Distributed Inference](20-distributed-inference/) | 24.5h | 567h |
| 21 | [21 — Multimodal AI Systems](21-multimodal-ai-systems/) | 24.5h | 591.5h |
| 22 | [22 — AI Economics](22-ai-economics/) | 24.5h | 616h |
| 23 | [23 — Observability & Reliability](23-observability-reliability/) | 23.5h | 639.5h |
| 24 | [24 — Model-System Co-design](24-model-system-codesign/) | 25.5h | 665h |
| 25 | [Capstone](capstone/) war week | ≈30h | |
| 26 | [Graduation](graduation/) | ≈20h | |

## Budget Check

- Authored Modules 00–24: **665h**.
- Capstone war week and graduation: ≈50h.
- Continuous capstone work in weeks 6–24 at ≈2–3 h/week: ≈40–55h.
- Program total: **≈755–770h**, inside the 730–780h budget.

## Pacing Notes

- **Weeks 13–17 combine six modules.** Modules 12–17 are each 24h; placing them in five weeks (≈28.8 h/week) keeps the load within the weekly budget and frees one week so all 25 modules fit before the capstone week. Module boundaries are kept; only the calendar is shared.
- Weeks 7–12 carry the heaviest single modules (30–32h); expect those weeks to run slightly over 30h, or borrow time from the continuous capstone allowance.
- Weeks 20–24 use the audited effort of the now-authored modules (23.5–25.5h each).
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
- **Security** — surfaces in agents, harnesses, and retrieval before the dedicated Module 18.
- **Capstone** — absorbs artifacts from Module 04 onward.
