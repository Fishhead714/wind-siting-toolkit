#!/usr/bin/env python3
"""Consistency check for the style assets (stdlib only).

Every `fixed:<key>`, `risk:<level>` and `auto:<family>` colour reference in
report_style.json must exist in theme.json, and every `ramp:<name>` used by the
example configs must exist in theme.json report.ramps.

    python3 tests/test_assets.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
theme = json.loads((ROOT / "assets" / "theme.json").read_text(encoding="utf-8"))["report"]
styles = json.loads((ROOT / "assets" / "report_style.json").read_text(encoding="utf-8"))


def colour_values(obj):
    """Yield every string value except documentation fields (keys starting with '_')."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not k.startswith("_"):
                yield from colour_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from colour_values(v)
    elif isinstance(obj, str):
        yield obj


errors = []
for name, spec in styles.items():
    if name.startswith("_"):
        continue
    if "kind" not in spec:
        errors.append(f"{name}: missing 'kind'")
    for v in colour_values(spec):
        for prefix, table in (("fixed:", theme["fixed"]), ("risk:", theme["risk"]),
                              ("auto:", theme["families"])):
            if v.startswith(prefix) and v[len(prefix):] not in table:
                errors.append(f"{name}: {v} not defined in theme.json")

for cfg in list((ROOT / "examples").glob("*.yaml")) + list((ROOT / "demo").glob("*.yaml")):
    for m in re.finditer(r"ramp:([A-Za-z0-9_]+)", cfg.read_text(encoding="utf-8")):
        if m.group(1) not in theme["ramps"]:
            errors.append(f"{cfg.name}: ramp:{m.group(1)} not defined in theme.json")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print(f"OK: {sum(1 for k in styles if not k.startswith('_'))} categories, all colour references resolve")
