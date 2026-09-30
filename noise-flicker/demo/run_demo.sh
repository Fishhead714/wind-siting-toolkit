#!/usr/bin/env bash
# Demo: synthetic layout -> noise / shadow-flicker contour GPKG.
# Usage: PY=<python with numpy/geopandas/scipy/contourpy/pyproj> ./run_demo.sh [outdir]
# Turbine parameters come from examples/reference_turbine.json (IEA-3.4-130-RWT; L_WA estimated),
# criteria from examples/criteria.example.json (international guidance, example only),
# spectrum from examples/spectrum.example.json. Nothing is built into the engine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
PY="${PY:-python3}"
OUT="${1:-$HERE/out}"
mkdir -p "$OUT"
REF="$ROOT/examples/reference_turbine.json"
ref() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$REF" "$1"; }

"$PY" "$HERE/make_demo_data.py" "$OUT/demo_site.gpkg"
"$PY" "$ROOT/engine/contours.py" \
  --criteria "$ROOT/examples/criteria.example.json" --country EXAMPLE --period night \
  --spectrum-file "$ROOT/examples/$(ref spectrum_file)" --spectrum "$(ref spectrum)" \
  --turbines "$OUT/demo_site.gpkg" --turbine-layer wtg --t-id-field wtg_id \
  --receptors "$OUT/demo_site.gpkg" --receptor-layer receptors --r-id-field bid \
  --hub-height-m "$(ref hub_height_m)" --rotor-diameter-m "$(ref rotor_diameter_m)" \
  --lwa-dba "$(ref lwa_dBA)" --wind-speed-ms 8 \
  --t-c 20 --rh 70 \
  --utc-offset -2 --blade-chord-m "$(ref blade_chord_m)" \
  --blade-chord-source "$(ref blade_chord_basis)" \
  --wind-rose "0:6,30:5,60:9,90:14,120:12,150:8,180:7,210:8,240:10,270:9,300:7,330:5" \
  --wind-rose-source "synthetic demo wind rose" --sunshine-prob 0.5 \
  --levels-file "$ROOT/examples/contour_levels.example.json" \
  -o "$OUT/demo_contours.gpkg"
