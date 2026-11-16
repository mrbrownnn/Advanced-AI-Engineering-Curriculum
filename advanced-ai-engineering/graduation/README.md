# Graduation — Architecture Review

## Purpose

Graduation is a comprehensive architecture review under realistic constraints:

- Ambiguous requirements.
- Incomplete information.
- Security constraints.
- SLOs.
- Budget limits.
- Growth projections.

[Design-doc/RFC guide](design-docs.md) · [Fill-in design-doc template](../templates/design-doc.md)

## What Is Tested

- Ambiguous requirements → structured assumptions
- Incomplete telemetry → identifying what's missing
- Quality constraints → evaluation strategy
- SLO targets → capacity and scheduling design
- Security constraints → threat modeling and defense
- GPU/resource budget → economic optimization
- Growth projection → scaling strategy

## Expected Reasoning Pattern

```mermaid
flowchart TD
    subgraph Frame["Frame the problem"]
        A["Unknowns"] --> B["Assumptions"]
        B --> C["Hypotheses"]
        C --> D["Quantitative Model"]
    end
    subgraph Test["Test the model"]
        E["Measurements"] --> F["Experiments"]
        F --> G["Falsification"]
    end
    subgraph Design["Choose the architecture"]
        H["Architecture"] --> I["Economics"]
        I --> J["Risks"]
    end
    subgraph Defend["Plan change and defense"]
        K["Migration"] --> L["Rollback"]
        L --> M["Defense"]
    end
    D --> E
    G --> H
    J --> K
```

## Structure

```
graduation/
├── README.md          ← this file
├── rubric.md          ← assessment criteria
└── incidents/
    └── README.md      ← graduation-level diagnostic scenarios
```

## Status

🏗️ **Skeleton only.** Graduation scenarios will be developed alongside the capstone.
