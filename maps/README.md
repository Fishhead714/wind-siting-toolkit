# render_map

A command-line map renderer built on PyQGIS. You describe the layers and the maps you want in one
YAML file; `render_map.py` draws each view with the QGIS rendering engine and adds a legend, scale
bar, north arrow, colour bar, attribution and credit line, all sized for the figure's final size on
a page or slide. Styles come from a catalogue of semantic categories (`assets/report_style.json`),
and colours from a shared palette (`assets/theme.json`), so the same kind of feature looks the same
on every map.

The repository also contains `derive_developable_area.py`, a small geopandas helper that subtracts
exclusion layers from a site boundary and reports excluded and developable areas.

## Requirements

- **render_map.py**: a Python interpreter that can `import qgis` (tested with QGIS 3.44), plus PyYAML.
  - Debian/Ubuntu: `sudo apt install python3-qgis python3-yaml`, then use the system `python3`.
  - Windows: OSGeo4W with the `qgis` package; run scripts with the OSGeo4W `python-qgis` launcher.
  - Any platform: `conda install -c conda-forge qgis pyyaml`.
  - Fonts: Noto Sans or DejaVu Sans. For CJK or Thai labels install the matching Noto fonts and add
    them to `font.families` in `assets/theme.json`.
- **derive_developable_area.py**: Python 3.10 or newer with geopandas (>= 0.14) and shapely (>= 2).
  QGIS is not needed.
- Network access only for online basemaps; maps without a basemap render offline.

## Quick start

```bash
# synthetic demo: data, developable area, three maps in demo/out/
PY_GEO=python3 PY_QGIS=python3 demo/run_demo.sh

# your own maps
python3 scripts/render_map.py --config my_maps.yaml --outdir output/
python3 scripts/render_map.py --config my_maps.yaml --outdir output/ --only overview
```

Start from `examples/manual_config_example.yaml` (annotated schema) or `demo/demo_maps.yaml`.
Relative paths in a config are resolved against the config file's directory.
`docs/DESIGN.md` explains the configuration, styling model, layouts and known pitfalls.
`python3 tests/test_assets.py` checks that every colour reference in the style catalogue resolves.

## Basemaps

Built-in aliases: `osm`, `opentopomap`, `esri_gray`, `esri_satellite`. Their attribution is added to
every map automatically. Each tile service has its own terms of use (for example the OpenStreetMap
Foundation Tile Usage Policy forbids heavy bulk use, and Esri services have their own terms); you are
responsible for checking that your use complies. For large batches, use your own tile server or a
local raster.

## Credits

Colour schemes: Paul Tol (qualitative "muted" and "bright"), ColorBrewer by Cynthia Brewer, Mark
Harrower and The Pennsylvania State University, Scientific colour maps by Fabio Crameri ("bilbao"),
and the matplotlib viridis/plasma colormaps by Nathaniel Smith and Stefan van der Walt. See
`THIRD_PARTY_NOTICES.md`. The north arrow is read at run time from the SVG library of your QGIS
installation and is not distributed here.

## License

Copyright 2026 Fishhead714.

GNU General Public License v3.0 or later (see `LICENSE`). The renderer imports PyQGIS (QGIS is
GPL-2.0-or-later; PyQt5 is GPL-3.0). Colour sources are credited in `THIRD_PARTY_NOTICES.md`.
