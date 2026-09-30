"""Plain-text / Markdown rendering of screening results."""
from __future__ import annotations

from typing import Sequence

from .engine import DISCLAIMER

_MARK = {"pass": "ok", "marginal": "~", "fail": "X", "unknown": "?"}


def render_markdown(results: Sequence[dict], title: str = "Turbine class suitability screening") -> str:
    L = [f"# {title}", "", f"> {DISCLAIMER}", ""]
    if results and not all(r.get("table_verified") for r in results):
        L += ["> **Class table not verified against the standard** (`table_verified = False`).", ""]
    L += ["| Turbine | Hub (m) | Result | Blocking / to review | Missing |", "|---|---|---|---|---|"]
    for r in results:
        block = ", ".join(r["blocking_hard_limits"] + r["exceeds_class_assumptions"] + r["data_quality_flags"]
                          + r["to_review"] + r["yield_flags"]) or "-"
        hub = "-" if r["hub_height_m"] is None else f"{r['hub_height_m']:g}"
        L.append(f"| {r['turbine']} | {hub} | {r['overall_label']} | {block} | {', '.join(r['missing']) or '-'} |")
    L.append("")
    for r in results:
        hub = "" if r["hub_height_m"] is None else f" @ {r['hub_height_m']:g} m"
        L += [f"## {r['turbine']}{hub}", ""]
        for c in r["constraints"]:
            tail = c["margin"] if c["verdict"] != "unknown" else c["reason"]
            derived = "*" if c["limit_derived"] else ""
            L.append(f"- [{_MARK[c['verdict']]}] **{c['label']}**: site={c['site_value']}, "
                     f"limit={c['limit']}{derived} - {tail}")
        if r["derating_flag"]:
            L.append(f"- Note: {r['derating_flag']}")
        L.append("")
    L.append("`*` limit derived by the tool, not stated by the turbine datasheet.")
    return "\n".join(L) + "\n"
