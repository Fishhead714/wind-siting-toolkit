# render_map: design notes

`render_map.py` turns a YAML configuration plus vector/raster data into finished map figures (PNG)
for reports and slides. The map body is rendered by QGIS itself (`QgsMapRendererCustomPainterJob`,
single-threaded and synchronous), so symbols, labels and reprojection behave exactly as in QGIS.
Decorations (title, legend, scale bar, north arrow, colour bar, locator inset, highlight rings,
credit line) are drawn on top with `QPainter`, sized for the physical size the figure will have on
the page.

This document describes what the code does today. The companion helper
`derive_developable_area.py` is described at the end.

## 1. Configuration

A configuration has four top-level keys:

| Key | Purpose |
|---|---|
| `crs` | Map CRS (e.g. `EPSG:32625`). Use a projected, metric CRS so scale bars and distances are right. If omitted, the CRS of the first vector layer is used (EPSG:3857 if there is none). |
| `defaults` | View settings shared by all views; any key can be overridden in a view. |
| `layers` | A pool of named layers, each loaded and styled once. |
| `views` | One entry per output PNG (`out` = file name without extension); `layers` lists pool keys. |

`examples/manual_config_example.yaml` is an annotated reference and `demo/demo_maps.yaml` is a
runnable example.

### 1.1 Layers

A pool entry is one of three kinds:

- **Vector**: `source` (any OGR data source) plus optional `layername` for multi-table files such as
  GeoPackages. Optional `filter` is a QGIS expression applied as a subset string, so several pool
  entries can show different subsets of one table (e.g. the criterion contour vs. the other contours).
- **Raster**: `raster` (a single-band GeoTIFF or other GDAL raster) with `style: {min, max, colors,
  opacity}`. `colors` is a list of two or more colours (interpolated evenly) or a named ramp
  `"ramp:<name>"` from `theme.json`; without `colors` the full viridis ramp is used. Unknown ramp
  names are an error rather than a silent fallback. `clip_to` (plus `clip_to_where`, an OGR SQL WHERE
  clause) masks the raster to a polygon with `gdal.Warp(cutline...)`; the clipped raster is cached in
  the system temp directory keyed by paths, filter and modification times. When `clip_to` points to a
  file with many features (e.g. all countries), always give `clip_to_where`: otherwise the cutline is
  the union of every feature and neighbouring areas inside the raster's bounding box stay visible.
  A raster with a `label` gets a colour bar.
- **Basemap**: `basemap: <alias>` (see section 5).

Common layer options: `label` (legend text; layers without a label are not listed), `style` (section 2),
`style_overrides` (override fields of a named category for this layer only; the result is registered
as a derived category `<category>@<key>`, so legend grouping and stacking rules still apply),
`label_field` / `labels_enabled` / `label_size_pt`, and label placement extras (`label_placement`:
around, over, horizontal, free, line, curved; `label_offsets`, `label_offset_field`,
`label_distance_mm`, `label_callout`, `label_box`, `label_exclude_ids`, `label_obstacle`,
`label_obstacle_factor`, `force_labels`, `label_top_k_by_area`).

Contour-style labels (`label_placement: line` or `curved`) are placed on the line with a small white
box behind the text, which interrupts the line under the number, and take the line colour; this is the
usual convention for isolines and makes it clear which line a value belongs to.

### 1.2 Relative paths

All file paths that come from the configuration (`source`, `raster`, `clip_to`) are resolved against
the directory that contains the YAML file, not the current working directory. Absolute paths and
URIs (anything containing `://` or starting with `type=`) are passed through unchanged. The renderer
can therefore be started from any directory. `figures_manifest.json`, written to the output
directory, lists the rendered figures with paths relative to that directory; results of earlier runs
into the same directory are merged by view name.

### 1.3 Views

Frequently used view keys (all of them can also be set in `defaults`):

- Canvas: `size` [px], `dpi`, `background`, `margin`, `physical_height_cm`, `preset`.
- Extent: `extent_layers` (union of these layers' extents, default: all vector layers of the view),
  `extent_pad_km`, and `pull_in` (if a listed layer has no feature inside the extent, the extent grows
  to include its nearest feature within `cap_km`, which is then marked).
- Decorations: `title`, `title_on_image`, `legend`, `legend_title`, `legend_position`,
  `legend_max_width_frac`, `scalebar`, `scale_ratio`, `scale_ratio_label`, `north`, `colorbar`,
  `neatline`, `decoration_layout`, `panel_thickness_cm`, `protect_layers`, `locator`, `info_card`.
- Content: `labels` (on/off/auto), `symbol_scale`, `label_font_pt`, `highlight` (rings around features
  selected by `id_field`/`ids`, with an optional legend note), `basemap_gray`.
- Credit line: `source`, `source_label`, `date_label`, `figure_id`, `figure_label`, `revision`,
  `revision_label`.

The order of `views[*].layers` is the drawing order: the first key is drawn on top, as in the QGIS
layers panel. An opaque basemap therefore belongs at the end; the renderer prints a warning when a
basemap is not last, because the legend would otherwise still list layers that the basemap hides.

### 1.4 Presets

`assets/presets.json` defines named bundles of `size`, `physical_height_cm` and `decoration_layout`:
`slide` (12.7 cm, 2/3 of a 19.05 cm 16:9 slide), `slide_tall` (15.2 cm, 4/5), `report_full` (18 cm) and `report_half` (10 cm).
A preset only fills keys that the view or `defaults` does not set; explicit keys always win.

### 1.5 Physical size

Font sizes, symbol-independent decoration sizes and line widths of decorations are computed from the
image height in pixels and `physical_height_cm`, so a figure looks the same when it is placed at that
size in a document. When the resulting resolution drops below 150 pixels per inch, the renderer warns
that the figure will look soft at print size. With `physical_height_cm` set, the scale bar also shows
the representative fraction ("1:N") for that physical size.

## 2. Styling model

### 2.1 Categories (`assets/report_style.json`)

A layer's `style` is normally the name of a category, e.g. `turbine_position`, `restriction_mining`,
`noise_contour`. A category fixes the *identity* of a kind of feature, its geometry kind, shape,
hatch pattern, line style and widths, so the same kind of feature looks the same on every map.
Symbols are built programmatically with the QGIS API (`QgsMarkerSymbol`, `QgsLineSymbol`,
`QgsFillSymbol`, simple symbol layers) instead of hand-written QML, which is sensitive to changes of
the QML schema between QGIS versions. A `style` can also be an inline dict with the same fields or a
path to a `.qml` file. Layers without a style use `point_generic`, `line_generic` or `polygon_generic`.

Supported kinds: `point` (simple marker), `point_ringset` (outer ring, inner ring and centre dot, a
turbine symbol whose size is in millimetres and therefore independent of the layer CRS), `line`
(optional `casing_color`/`casing_width_mm`: a wider pale line under the main line, so the line stays
visible on any basemap) and `fill` (solid, `hatch`, `hatch_cross`, optional `hatch_over_fill`, which
draws a translucent base fill under the hatch so that large areas read as one block).

A line category applied to polygon data (e.g. an administrative boundary stored as polygons) is
converted to a transparent fill with an outline, and an informational message is printed.

Several colours from one table are produced with `categorize_by` + `categories` (QGIS categorized
renderer) or `rules` (rule-based renderer with QGIS expressions), not by loading the same table many
times with different filters (see section 7).

`legend_casing_color` and `legend_fill_color` / `legend_outline_color` / `legend_outline_width_mm` /
`legend_opacity` change only the legend swatch, for cases where the map background and the legend
background need different contrast.

### 2.2 Colours (`assets/theme.json`)

Colours in categories are references resolved at load time:

| Reference | Meaning |
|---|---|
| `auto` | next colour of the qualitative palette (Paul Tol "muted"), one global counter |
| `auto:<family>` | next colour of a hue family (`own_infra` warm, `grid` blue, `reference` violet-grey), one counter per family |
| `fixed:<key>` | a named fixed colour |
| `risk:<level>` | severity colour (`high`, `medium`, `low`, `none`) |
| `ramp:<name>` | raster ramp (`wind_speed`, `wind_speed_dark`, `elevation`) |
| `transparent` or a hex value | literal |

Counters advance in the order in which layers are defined in the configuration, so a given
configuration always produces the same colours. The counters do not know semantics, which has two
consequences worth designing for:

1. Two pool entries that show the same real-world thing (for example one table split by a filter)
   receive different `auto` colours. Give such layers a `fixed:` or literal colour.
2. When a layer is moved from `auto` to a fixed colour, the next `auto` layer takes over the freed
   counter position, and a hand-picked fixed colour can coincide with a qualitative colour that is
   still handed out. If a whole family of layers must stay distinct (for example several land-tenure
   classes), give every member a fixed colour at once and check the values against the qualitative
   palette.

Palette choices follow published guidance: qualitative colours from Paul Tol's colour-blind-safe
schemes; sequential steps from ColorBrewer; continuous ramps that are perceptually uniform, monotonic
in lightness and exclude the background colour (Crameri, Shephard & Heron 2020), with viridis for wind
speed and the Scientific colour map "bilbao" for elevation (no green at the low end, since green is
read as vegetation; Patterson & Jenny 2013). Compliance states use Tol "bright" green / yellow / red
(`status_ok`, `status_warn`, `status_exceed`), kept separate from the `risk` colours used for land
restrictions. Shadow-flicker bands use several hues (from plasma) rather than lightness steps of one
hue, by analogy with NatureScot (2017) para. 60 (advice on ZTV visibility bands). Red and orange restriction classes additionally differ in hatch
pattern and opacity, so the distinction does not rely on colour alone.

Fonts come from `theme.json` `font.families`; Qt falls back per character to the first family that has
the glyph, so mixed-script labels work when suitable fonts (e.g. Noto Sans CJK, Noto Sans Thai) are
installed and listed. Missing fonts are skipped silently and missing glyphs render as boxes, so check
installed fonts (`fc-list`) before rendering non-Latin labels.

## 3. Restriction layers: `exclusion_type`

Restriction categories carry `exclusion_type: hard | review | soft`:

- **hard**: the area cannot be used (opaque fill, or cross hatch over a base fill);
- **review**: restricted, needs a specific permit or study, but not an absolute exclusion (red, single
  hatch, lower opacity);
- **soft**: a recommended buffer or negotiable constraint (hatched, orange).

Legend entries of restriction layers are grouped under one heading per type, in the order hard,
review, soft, after the other entries. The heading texts can be changed with
`exclusion_group_labels: {hard: ..., review: ..., soft: ...}` in `defaults` or a view.
Within the view's layer list, the positions occupied by restriction layers are re-sorted so that hard
exclusions are drawn above review areas and review areas above soft buffers; the relative order of
other layers (turbines, boundaries, basemap) is not changed, so a restriction is never lifted above
the main subject of the map.

Which type a real constraint has is a legal question for each jurisdiction. The category defaults are
generic; override them per layer with `style_overrides: {exclusion_type: ...}`.

## 4. Decorations and layout

`decoration_layout` selects one of:

- `overlay` (default): decorations float on the map. The "protected" subject is the union of
  `protect_layers` (default `extent_layers`; point and line subjects use their convex hull). The legend,
  north arrow, scale bar, colour bar, locator and credit line are placed in corners that do not overlap
  the subject; if none fit, the extent is enlarged by 10 % up to ten times, after which a warning is
  printed. `protect_layers: false` disables this. The legend corner can also be forced with
  `legend_position`.
- `panel-bottom` / `panel-right`: decorations are laid out in a band below or to the right of the map,
  so they never cover data. `panel_thickness_cm` fixes the band thickness, which keeps the map area
  identical across a series of figures. If the decorations do not fit along the band, they are scaled
  down (with a warning) instead of being cut off.
- `panel-auto`: `panel-right` when the content is clearly narrower than the canvas (its aspect ratio is
  below 0.6 of the canvas aspect ratio), else `panel-bottom`. For a series of figures in one document,
  choose a fixed side so that decorations do not jump between figures.

Decoration elements:

- **Legend** lists only layers that have at least one feature inside the final extent; labelled layers
  without features are reported in a short "not present in this view" note instead. Long labels wrap
  and are capped at `legend_max_width_frac` of the canvas width.
- **Scale bar** uses round lengths (1/2/5 x 10^n), labelled at both ends, measured on the WGS84
  ellipsoid.
- **North arrow** uses `arrows/NorthArrow_02.svg` from the QGIS SVG library found through
  `QgsApplication.svgPaths()`, recoloured through the SVG `param(fill)`/`param(outline)` mechanism. If
  the file is not found or cannot be rendered, a simple arrow is drawn instead.
- **Panel ink colour** (north arrow, scale bar, colour bar ticks in panel layouts) is dark on light
  backgrounds and light on dark backgrounds, chosen from the perceived luminance of `background`.
- **Credit line**: basemap attribution, `source`, the CRS with its human-readable name (from the QGIS
  CRS database), the render date and optional figure id / revision. It is always drawn.
- **Neatline**: a thin frame around the whole sheet, including panels (`neatline: false` to disable).
- **Locator inset** (`locator: {layer: <polygon pool key>, width_frac, position}`): the given polygon
  layer with the main extent outlined; very small extents are shown as a dot.

These elements follow common cartographic practice for a map's visual hierarchy: the basemap stays
low in contrast and the thematic layers carry the emphasis.

## 5. Basemaps and attribution

| Alias | Service | Notes |
|---|---|---|
| `osm` | OpenStreetMap standard tiles | follow the OSMF Tile Usage Policy; do not bulk-render |
| `opentopomap` | OpenTopoMap | maximum zoom 17 |
| `esri_gray` | Esri World Light Gray Canvas | key-free light neutral basemap |
| `esri_satellite` | Esri World Imagery | photographic |

Attribution text for the basemaps used in a view is added to the credit line automatically, so it stays
visible on the figure as the OpenStreetMap Foundation's Attribution Guidelines require. Each service has
its own terms of use; check them before commercial use, and add an attribution string when adding a new
tile source.

Photographic basemaps are desaturated and darkened by default (`treatment: false` to disable), , so imagery does not compete with the
foreground layers. Colour vector basemaps (`osm`, `opentopomap`) are turned grey when the view contains
filled polygons (`basemap_gray: auto`), because coloured fills over a coloured basemap mix into
misleading colours; NatureScot (2017) recommends greyscale base mapping for this reason. Views with lines
only keep the colour basemap.

## 6. Other behaviour

- Labels have a white halo by default. Label clutter is not thinned automatically; switch labels off
  on large extents and use `label_top_k_by_area` for area names.
- `highlight` matches `id_field` exactly; ids must be unique in the layer.
- `symbol_scale` and `labels` apply to all layers of a view.
- Exit codes: 0 if every requested view rendered, 1 otherwise. Views with missing pool keys still
  render and are reported.

## 7. Known pitfalls (technical notes)

- **Second table of a GeoPackage.** With some QGIS/GDAL combinations, the first opening of the second
  or later table of a GeoPackage within a process fails with "unable to open database file" (empty
  extent, no features), while a second opening works. The renderer opens every vector layer once
  before creating the real layer and reloads a layer whose extent comes back empty. For the same
  reason, split one table into classes with `categorize_by` or `rules` rather than opening it many
  times with different filters.
- **Tile services that fail silently.** Some tile providers return HTTP 200 with watermark tiles when no
  API key is supplied, so nothing reports an error; look at the output. The built-in aliases do not need
  keys.
- **ArcGIS tile order.** ArcGIS REST tile URLs use `{z}/{y}/{x}`, OSM-style services `{z}/{x}/{y}`.
  Swapping them produces a scrambled image without any error.
- **Sub-pixel line widths.** A line narrower than about one output pixel (e.g. 0.15 mm at 96 dpi is
  about 0.57 px) may not be drawn at all, without any warning. Check widths against the output
  resolution, and check colours by sampling pixels of a rendered image rather than trusting the
  configuration.
- **Opaque layers in the wrong order.** A basemap or opaque raster that is not last in the view's
  layer list hides everything after it while the legend still lists those layers.
- **Ellipsoid.** `QgsProject.ellipsoid()` returns the string `"NONE"` when unset, so an `or` fallback
  never triggers; distances are always measured explicitly on WGS84.
- **Headless shutdown.** Calling `QgsApplication.exitQgis()` after rendering can crash the process
  after all files have been written. The renderer skips it and exits with `os._exit()`.
- **Raster ramps.** Colour ramps with more than two colours are passed to QGIS as gradient stops;
  using only the end colours would silently drop the middle of the ramp.

## 8. derive_developable_area.py

A separate, QGIS-free helper (geopandas and shapely) that computes *site minus the union of exclusion
layers*:

- inputs: `--boundary file:layer` (polygons, or turbine points, in which case the site is their convex
  hull buffered by `--site-buffer-m`), repeated `--exclude file:layer[:label]`, optional `--crs`
  (must be projected), `--min-area-ha` (drop small fragments), `--caption-note` (free text copied into
  the summary for the figure caption);
- outputs (GeoPackage): `developable_area`, `exclusion_union` (pieces of the site with the number of
  overlapping constraints, `n_constraints`) and a `developable_summary` table with excluded and
  *exclusive* area per constraint (area excluded by that constraint alone, a simple measure of how much
  the result depends on it, after Ryberg, Robinius & Stolten 2018); a JSON summary is printed;
- `--check-points file:layer` counts existing turbine points inside each exclusion layer. Points inside
  a layer suggest that the layer is not a true exclusion at that site; such layers are better overlaid
  as review layers than subtracted.

The outputs are drawn with the categories `developable_area` (flat single-colour fill, the "potential
developable area" style of the usual two-step constraint mapping) and `exclusion_overlap` (count classes
in ColorBrewer Purples).

## References

- Crameri, F., Shephard, G. E. & Heron, P. J. (2020). The misuse of colour in science communication.
  *Nature Communications* 11, 5444.
- Crameri, F. Scientific colour maps (bilbao, v1.8). Zenodo, doi:10.5281/zenodo.1243862.
- Tol, P. Colour Schemes. SRON technical note SRON/EPS/TN/09-002.
- Brewer, C., Harrower, M. & The Pennsylvania State University. ColorBrewer 2.0.
- Smith, N. & van der Walt, S. viridis and plasma colormaps (matplotlib).
- Patterson, T. & Jenny, B. (2013). Evaluating cross-blended hypsometric tints: a user study in the
  United States, Switzerland, and Germany. *Cartographic Perspectives* 75.
- NatureScot (Scottish Natural Heritage) (2017). *Visual Representation of Wind Farms*, version 2.2.
- OpenStreetMap Foundation. Licence/Attribution Guidelines; Tile Usage Policy.
- Ryberg, D. S., Robinius, M. & Stolten, D. (2018). Evaluating Land Eligibility Constraints of Renewable
  Energy Sources in Europe. *Energies* 11(5), 1246.
- LAI (2020). Hinweise zur Ermittlung und Beurteilung der optischen Immissionen von Windkraftanlagen
  (WKA-Schattenwurfhinweise), Aktualisierung 2019, Stand 23.01.2020.
