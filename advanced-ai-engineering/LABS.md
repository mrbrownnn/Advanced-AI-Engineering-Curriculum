# Lab Framework Conventions

Runnable scaffolding for the labs specified in each module's README §06.
The README §06 spec is the source of truth; lab folders add starter code,
fixtures, tests, and a report. They never restate or change the spec.

## Layout

```
NN-module/labs/
  README.md                      ← index: lab, hardware tag, effort, run command
  lab-x-<slug>/
    README.md                    ← link to §06, PREDICT gate, run/submit steps
    report.md                    ← copy of templates/lab-report.md, pre-titled
    starter/                     ← importable skeleton; TODOs raise NotImplementedError
    fixtures/                    ← small synthetic, seeded data (or a generator script)
    tests/                       ← pytest; conftest.py selects the implementation
    experiments/                 ← scripts that write raw JSONL + result.yaml to results/

labs-solutions/NN-module/lab-x-<slug>/
    <same module names as starter/>   ← reference implementation
    BREAK_KEY.md                      ← planted-bug answers, expected observations
```

Trace-only labs (`TRACE`) have `README.md` + `report.md` (questions, not
answers); the answer key lives in `labs-solutions/`.

`labs-solutions/` is the instructor tree. Learners open it only after
submitting `report.md`. Move it to a private branch or repository before
publishing.

## Hardware tags

Each lab README carries one line `**Hardware**: CPU` (or `COLAB`, `TRACE`);
the validator reads it. A lab may say `CPU (optional COLAB extension)` —
only the first tag counts.

`CPU` (default; runs locally or in Colab CPU) · `COLAB` (needs a GPU; ships
`colab.ipynb`, designed for free-tier T4: no FP8/bf16/FlashAttention-2,
checkpoint results to Drive) · `TRACE` (static source reading). No lab
requires the learner's local GPU.

## Validation

`python tools/validate_curriculum.py` also checks, for every module with a
`labs/` directory: one `lab-x-*` folder per `### LAB X`, `README.md` with a
hardware tag, `report.md` with a `## 1 PREDICTION` section, `tests/conftest.py`
when `starter/` exists, `colab.ipynb` for `COLAB` labs, and a matching
`labs-solutions/` folder. It does not run the tests.

## Running

```bash
cd NN-module/labs/lab-x-<slug>
python -m pytest tests -q                       # against your starter/
LAB_IMPL=solution python -m pytest tests -q     # against the reference (instructor)
```

`tests/conftest.py` puts `starter/` on `sys.path`, or
`labs-solutions/NN-module/lab-x-<slug>/` when `LAB_IMPL=solution`.

## Test rules

1. Test invariants and behavior, not magic numbers a learner could paste.
2. Every assertion message carries `HINT:` text drawn from the module's
   Feedback Contract / Diagnostic Hint. Never print the solution.
3. Deterministic: seeded RNG and an injectable clock. No test depends on real
   wall-clock noise or network. Whole lab suite < 10 s on CPU.
4. Against `starter/`, tests collect cleanly and fail with hints (no import
   errors). Against the reference, all pass.

## Learning mechanics

- **Prediction gate**: `report.md` §1–2 are committed before any experiment.
- **Faded scaffolding**: Modules 00–04 ship full skeletons (fill the TODOs);
  05–15 ship interfaces and tests only; 16–24 ship spec and rubric only.
- **Planted bug (BREAK)**: at least one lab per module ships
  `starter/bugged_*.py`, a working-looking implementation with one real bug.
  The learner finds it with instrumentation, not by diffing. Key in `BREAK_KEY.md`.
- **Artifact reuse**: later labs may import earlier lab code; say so in the lab
  README.

## Dependencies

Python 3.10+, `numpy`, `scipy`, `pytest`, `hypothesis`, `pyyaml`. Module 01
adds CPU `torch`. Add nothing else without recording why in the lab README.

## Results

Experiments write raw event data (JSONL) and a `result.yaml` following
[benchmarks/result-schema.md](benchmarks/result-schema.md) into the lab's
`results/` directory. Append-only: never overwrite an earlier run.
