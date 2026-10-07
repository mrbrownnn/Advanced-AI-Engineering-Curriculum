"""Static validator for authored curriculum modules.

Checks JSON-schema validity of each module's evidence registry and knowledge
model, then cross-file integrity between registry, knowledge model, and README.
It also proves that known schema holes stay closed by requiring the negative
fixtures in tools/fixtures/ to FAIL validation.

This is a static linter (see STATIC_LINTER.md). Passing it does not certify
technical correctness.

Usage:
    python tools/validate_curriculum.py            # all modules with a registry and lessons
    python tools/validate_curriculum.py 04 19      # selected module number prefixes
"""
import io
import os
import re
import sys

import jsonschema
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tools", "fixtures")
BASELINE = "04-serving-scheduling-capacity"
PROVENANCE_LABEL = {"VERIFIED_FACT": "O", "DERIVED": "D", "HYPOTHESIS": "H"}


def load(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read() if path.endswith(".md") else yaml.safe_load(f)


EVIDENCE_SCHEMA = load(os.path.join(ROOT, "EVIDENCE_SCHEMA.yaml"))
KM_SCHEMA = load(os.path.join(ROOT, "KNOWLEDGE_MODEL_SCHEMA.yaml"))


def schema_errors(doc, schema):
    validator = jsonschema.Draft7Validator(schema)
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(doc)]


def h2_spine(text):
    return [re.sub(r"^## \d+ ", "", line) for line in text.splitlines() if line.startswith("## ")]


def effort_errors(readme):
    """Compare estimated_effort components with the declared total."""
    block = re.search(r"estimated_effort:\n((?:  \w+: [^\n]+\n)+)", readme)
    if not block:
        return ["missing estimated_effort block"]
    values = {}
    for key, raw in re.findall(r"  (\w+): ([0-9.]+)h", block.group(1)):
        values[key] = float(raw)
    if "total" not in values:
        return ["estimated_effort has no total"]
    parts = sum(v for k, v in values.items() if k != "total")
    if abs(parts - values["total"]) > 1e-6:
        return [f"estimated_effort components sum to {parts}h but total is {values['total']}h"]
    return []


def check_module(mod):
    errs = []
    reg = load(os.path.join(ROOT, "research-registry", f"{mod}.yaml"))
    km = load(os.path.join(ROOT, mod, "KNOWLEDGE_MODEL.yaml"))
    readme = load(os.path.join(ROOT, mod, "README.md"))
    baseline = load(os.path.join(ROOT, BASELINE, "README.md"))

    errs += [f"registry schema {e}" for e in schema_errors(reg, EVIDENCE_SCHEMA)]
    errs += [f"knowledge-model schema {e}" for e in schema_errors(km, KM_SCHEMA)]

    claims = reg.get("claims", [])
    ids = [c["id"] for c in claims]
    if len(ids) != len(set(ids)):
        errs.append("duplicate claim IDs")
    provenance = {c["id"]: c["provenance"] for c in claims}

    sections = [k for k, v in km.items() if isinstance(v, list) and v and isinstance(v[0], dict) and "id" in v[0]]
    km_ids = {e["id"] for k in sections for e in km[k]}
    used = set()
    for k in sections:
        for entry in km[k]:
            for cid in entry.get("evidence_ids", []):
                used.add(cid)
                if cid not in provenance:
                    errs.append(f"KM {entry['id']} cites unknown {cid}")
            for dep in entry.get("derived_from", []):
                if dep not in km_ids:
                    errs.append(f"KM {entry['id']} derived_from unknown {dep}")
    for cap in km.get("candidate_capabilities", []):
        for m in cap["mapped_mechanisms"]:
            if m not in km_ids:
                errs.append(f"capability maps unknown {m}")

    lessons = set(re.findall(r"### Lesson (\d+\.\d+) ", readme))
    labs = set(re.findall(r"### LAB ([A-Z]) ", readme))
    for c in claims:
        usage = c.get("curriculum_usage") or {}
        for lesson in usage.get("lessons", []):
            if lesson not in lessons:
                errs.append(f"{c['id']} targets missing lesson {lesson}")
        for lab in usage.get("labs", []):
            if lab.split("-")[-1] not in labs:
                errs.append(f"{c['id']} targets missing lab {lab}")
        for rel in c.get("related_claims") or []:
            if rel not in provenance:
                errs.append(f"{c['id']} related to unknown {rel}")

    cited = set(re.findall(r"CLM-\d{3}[a-z]?", readme))
    for cid in sorted(cited - set(ids)):
        errs.append(f"README cites unknown {cid}")
    for label, group in re.findall(r"\*\*([ODH])\*\*,\s*((?:CLM-\d{3}[a-z]?(?:,\s*)?)+)", readme):
        for cid in re.findall(r"CLM-\d{3}[a-z]?", group):
            if cid in provenance and PROVENANCE_LABEL.get(provenance[cid]) != label:
                errs.append(f"README labels {cid} as {label} but provenance is {provenance[cid]}")

    declared = set(km.get("required_concepts", [])) | set(km.get("prerequisite_concepts", []))
    for dep in km.get("concept_dependencies", []):
        for concept in [dep["concept"]] + dep["depends_on"]:
            if concept not in declared:
                errs.append(f"concept {concept} used in dependencies but not declared")

    if h2_spine(readme) != h2_spine(baseline):
        errs.append("H2 spine differs from Module 04")
    errs += effort_errors(readme)
    errs += lab_errors(mod, labs)
    if re.search(r"\*\*Concepts & Definitions:\*\*\s*\n\s*\n\*\*", readme):
        errs.append("empty Concepts & Definitions subsection")

    warnings = []
    unused = sorted(set(ids) - used)
    if unused:
        warnings.append(f"claims not referenced by the knowledge model: {unused}")
    uncited = sorted(set(ids) - cited)
    if uncited:
        warnings.append(f"claims not cited in README: {uncited}")
    stats = f"claims={len(ids)} km_entries={len(km_ids)} lessons={len(lessons)} labs={len(labs)}"
    return errs, warnings, stats


def lab_errors(mod, labs):
    """Lab scaffolding contract from LABS.md; only for modules that have a labs/ directory."""
    lab_root = os.path.join(ROOT, mod, "labs")
    if not os.path.isdir(lab_root):
        return []
    errs = []
    dirs = {d[4].upper(): d for d in os.listdir(lab_root) if re.match(r"lab-[a-z]-", d)}
    for lab in sorted(labs):
        if lab not in dirs:
            errs.append(f"LAB {lab} has no labs/lab-{lab.lower()}-* directory")
            continue
        path = os.path.join(lab_root, dirs[lab])
        for name in ("README.md", "report.md"):
            if not os.path.exists(os.path.join(path, name)):
                errs.append(f"{dirs[lab]} missing {name}")
        report = os.path.join(path, "report.md")
        if os.path.exists(report) and "## 1 PREDICTION" not in load(report):
            errs.append(f"{dirs[lab]}/report.md has no '## 1 PREDICTION' section")
        starter = os.path.join(path, "starter")
        if os.path.isdir(starter) and not os.path.exists(os.path.join(path, "tests", "conftest.py")):
            errs.append(f"{dirs[lab]} has starter/ but no tests/conftest.py")
        if os.path.isdir(starter) and not os.path.isdir(os.path.join(path, "experiments")):
            errs.append(f"{dirs[lab]} has starter/ but no experiments/ directory")
        readme = os.path.join(path, "README.md")
        tag = re.search(r"\*\*Hardware\*\*:\s*`?(CPU|COLAB|TRACE)", load(readme)) if os.path.exists(readme) else None
        if os.path.exists(readme) and not tag:
            errs.append(f"{dirs[lab]}/README.md has no '**Hardware**: CPU|COLAB|TRACE' line")
        elif tag and tag.group(1) == "COLAB" and not os.path.exists(os.path.join(path, "colab.ipynb")):
            errs.append(f"{dirs[lab]} is tagged COLAB but has no colab.ipynb")
        solution = os.path.join(ROOT, "labs-solutions", mod, dirs[lab])
        if not os.path.isdir(solution):
            errs.append(f"{dirs[lab]} has no labs-solutions/{mod}/{dirs[lab]}")
        elif not os.path.exists(os.path.join(solution, "BREAK_KEY.md")):
            errs.append(f"labs-solutions/{mod}/{dirs[lab]} missing BREAK_KEY.md")
    for lab in sorted(set(dirs) - set(labs)):
        errs.append(f"{dirs[lab]} has no matching '### LAB {lab}' in README §06")
    return errs


def check_fixtures():
    """Every negative fixture must be rejected; every positive fixture accepted."""
    errs = []
    for name in sorted(os.listdir(FIXTURES)):
        doc = load(os.path.join(FIXTURES, name))
        schema = KM_SCHEMA if name.startswith(("km_", "bad_km_")) else EVIDENCE_SCHEMA
        failed = bool(schema_errors(doc, schema))
        expect_fail = name.startswith("bad_")
        if failed != expect_fail:
            errs.append(f"fixture {name}: expected {'rejection' if expect_fail else 'acceptance'}")
    return errs


def modules(prefixes):
    found = []
    for d in sorted(os.listdir(ROOT)):
        if not re.match(r"\d\d-", d) or (prefixes and d[:2] not in prefixes):
            continue
        has_registry = os.path.exists(os.path.join(ROOT, "research-registry", f"{d}.yaml"))
        readme = os.path.join(ROOT, d, "README.md")
        if has_registry and os.path.exists(readme) and "### Lesson" in load(readme):
            found.append(d)
    return found


def main():
    failures = 0
    for mod in modules(set(sys.argv[1:])):
        errs, warnings, stats = check_module(mod)
        print(f"{'FAIL' if errs else 'ok  '} {mod}: {stats}")
        for e in errs:
            print(f"     error: {e}")
        for w in warnings:
            print(f"     warn:  {w}")
        failures += bool(errs)
    fixture_errs = check_fixtures()
    print(f"{'FAIL' if fixture_errs else 'ok  '} fixtures")
    for e in fixture_errs:
        print(f"     error: {e}")
    failures += bool(fixture_errs)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
