# Research Registry

This directory contains the FACTUAL PROVENANCE SYSTEM for the Advanced AI Engineering Curriculum.

**IMPORTANT:**
- This is NOT the learner-facing literature system.
- This registry exists to ensure curriculum writers and compilers do not hallucinate technical facts.
- Research is strictly **module-scoped**.

## Provenance Classes
Every substantive technical statement must belong to one of these classes:
1. `VERIFIED_FACT`: Backed by Tier 1/2 evidence.
2. `DERIVED`: Result derived from explicit assumptions.
3. `EXERCISE_ASSUMPTION`: Hypothetical condition for an exercise (NOT universal hardware behavior).
4. `HYPOTHESIS`: A prediction to be tested.
5. `TODO_VERIFY`: Unverified claim. MUST NOT silently become a `VERIFIED_FACT`.

## Evidence Tier Policy
- **TIER 1 (Primary)**: Papers, official docs, repositories, original benchmarks.
- **TIER 2 (Authoritative)**: Vendor architecture, maintainer blogs.
- **TIER 3 (Discovery Only)**: Independent blogs, community discussions (use to find Tier 1/2, not as canonical evidence).
- **TIER 4 (Do Not Use)**: SEO content, AI summaries.

## Source Code Pinning
Production source claims must include:

- Repository.
- Commit/tag.
- Verification date.
- File.
- Symbol.
- Execution path.

Without a pinned revision, they are `TODO_VERIFY`.

## Versioning & Freshness
- Evidence must include `research_cutoff` and `registry_version`.
- When a module is updated, do not silently overwrite old evidence.
- Make clear:
  - What changed.
  - Which curriculum claims depend on it.
