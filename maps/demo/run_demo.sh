#!/usr/bin/env bash
# End-to-end demo on SYNTHETIC data. Works from any working directory; it deliberately does
# not cd into the demo directory, so relative paths in demo_maps.yaml are resolved against
# the config file's directory by render_map.py.
#   1. make_demo_data.py          synthetic layers (geopandas, no QGIS)
#   2. derive_developable_area.py site minus exclusions (geopandas, no QGIS)
#   3. render_map.py              three PNG maps (needs a Python that can `import qgis`)
# Environment:
#   PY_GEO   Python with geopandas (default: python3)
#   PY_QGIS  Python with PyQGIS    (default: python3; on Windows use the OSGeo4W python-qgis launcher)
#   OUTDIR   output directory for the PNGs (default: <demo dir>/out)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$HERE/../scripts"
PY_GEO="${PY_GEO:-python3}"
PY_QGIS="${PY_QGIS:-python3}"
OUTDIR="${OUTDIR:-$HERE/out}"

"$PY_GEO" "$HERE/make_demo_data.py"

"$PY_GEO" "$SCRIPTS/derive_developable_area.py" \
  --boundary "$HERE/demo_site.gpkg:boundary" \
  --exclude "$HERE/demo_exclusions.gpkg:dwelling_setback:Dwelling setback (example)" \
  --exclude "$HERE/demo_exclusions.gpkg:road_setback:Road setback (example)" \
  --check-points "$HERE/demo_site.gpkg:wtg" \
  --caption-note "Synthetic data; setback distances are examples only" \
  --out "$HERE/demo_developable.gpkg"

mkdir -p "$OUTDIR"
"$PY_QGIS" "$SCRIPTS/render_map.py" --config "$HERE/demo_maps.yaml" --outdir "$OUTDIR"
echo "done: $OUTDIR"
