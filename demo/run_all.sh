#!/usr/bin/env bash
# End-to-end demo on SYNTHETIC data: noise-flicker computes contours, maps renders them.
#   PY_GEO   Python with numpy/scipy/shapely/pyproj/geopandas/pyogrio/contourpy (default: python3)
#   PY_QGIS  Python that can `import qgis`                                        (default: python3)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE/.."
PY_GEO="${PY_GEO:-python3}"
PY_QGIS="${PY_QGIS:-python3}"
OUT="$HERE/out"
mkdir -p "$OUT"

echo "[1/2] noise-flicker: synthetic layout -> contours (a few minutes)"
PY="$PY_GEO" bash "$ROOT/noise-flicker/demo/run_demo.sh" "$OUT"

echo "[2/2] maps: contours -> figures"
"$PY_QGIS" "$ROOT/maps/scripts/render_map.py" --config "$HERE/pipeline_maps.yaml" --outdir "$OUT"
echo "Figures are in $OUT"
