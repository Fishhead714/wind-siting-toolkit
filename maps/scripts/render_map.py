#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_map.py - headless map-figure renderer for vector and raster data, built on QGIS.

Renders one or more map figures described by a YAML config. Requires a Python
interpreter with the PyQGIS bindings available (typically the system Python that
ships with a QGIS installation):

    python3 render_map.py --config map.yaml --outdir output/
    python3 render_map.py --config map.yaml --outdir output/ --only overview

The map itself is drawn by the native QGIS rendering engine
(QgsMapRendererCustomPainterJob, single-threaded synchronous rendering; symbols and
colours go through the real QGIS symbol system, not a matplotlib re-drawing - the
single-threaded choice is deliberate, see the notes further down). Legend, scale bar,
north arrow, "nothing found" notes, highlight circles and off-map markers are painted
on top of the rendered image with QPainter. Decoration sizes are derived from the
final physical size of the figure so they stay legible after being placed in a slide
or report.

Styles come from assets/report_style.json: every key is a semantic layer category
(turbine position, grid line, restriction zone, ...). Symbols are built
programmatically with QgsXXXSymbol.createSimple() rather than hand-written QML, because
the QML schema changes between QGIS versions and hand-written QML can silently break
after an upgrade. Shape and pattern are the identity of a category and do not change
between maps.

Colours come from the "report" namespace of assets/theme.json (a palette grouped by
semantic family: own facilities, grid, reference layers, risk levels, fixed colours for
protected areas and water, ...). Colours are cycled within each family, so the same
config always produces the same colours on every run.

Output sizes and decoration presets come from assets/presets.json.

The config schema is described in docs/DESIGN.md; see examples/ for working configs.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
from osgeo import gdal
from qgis.core import (
    QgsApplication, QgsProject, QgsVectorLayer, QgsRasterLayer,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsMapSettings,
    QgsMapRendererCustomPainterJob, QgsRectangle, QgsSymbolLayerUtils, QgsGeometry,
    QgsFeatureRequest, QgsPointXY, QgsSpatialIndex, QgsDistanceArea,
    QgsMarkerSymbol, QgsLineSymbol, QgsFillSymbol, QgsSimpleFillSymbolLayer, QgsSimpleLineSymbolLayer,
    QgsSimpleMarkerSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsPalLayerSettings, QgsVectorLayerSimpleLabeling, QgsTextFormat,
    QgsRasterShader, QgsColorRampShader, QgsGradientColorRamp, QgsGradientStop,
    QgsSingleBandPseudoColorRenderer,
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsRuleBasedRenderer,
    QgsProperty, QgsSymbol, QgsHueSaturationFilter, QgsWkbTypes, QgsRenderContext,
    Qgis, QgsTextBackgroundSettings, QgsSimpleLineCallout, QgsManhattanLineCallout, QgsCurvedLineCallout,
)
from qgis.PyQt.QtCore import QRectF, QSize, QSizeF, Qt, QPointF, QPoint
from qgis.PyQt.QtGui import (
    QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPen, QPolygonF, QLinearGradient, QImage,
)
from qgis.PyQt.QtSvg import QSvgRenderer

gdal.UseExceptions()

HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE.parent / "assets"
THEME = json.loads((ASSETS_DIR / "theme.json").read_text(encoding="utf-8"))
PALETTE = THEME["report"]
CATEGORIES = json.loads((ASSETS_DIR / "report_style.json").read_text(encoding="utf-8"))
PRESETS = json.loads((ASSETS_DIR / "presets.json").read_text(encoding="utf-8"))
CLIP_CACHE_DIR = Path(tempfile.gettempdir()) / "render_map_clip_cache"

# Directory of the YAML config being rendered; set by main(). Relative file paths in the
# config are resolved against it so a config works regardless of the current directory.
CONFIG_DIR = None


def resolve_path(p):
    """Resolve a file path taken from the YAML config.

    Returns p unchanged when it is empty, already absolute, or looks like a URI / data
    source string (contains "://" or starts with "type="). Otherwise the path is taken
    relative to the config file's directory (CONFIG_DIR) when that is known."""
    if not p:
        return p
    s = str(p)
    if "://" in s or s.startswith("type=") or os.path.isabs(s):
        return p
    if CONFIG_DIR is not None:
        return str(Path(CONFIG_DIR) / s)
    return p


_north_arrow_svg_cache = []


def north_arrow_svg():
    """Path of the north-arrow SVG shipped with QGIS (arrows/NorthArrow_02.svg), or None.

    Searches QgsApplication.svgPaths() and caches the first existing file. The SVG is
    rendered through QgsApplication.svgCache().svgContent(), which applies QGIS's own
    param(fill)/param(outline) substitution, so the arrow can be recoloured freely.
    Callers must fall back to a hand-drawn arrow when this returns None."""
    if not _north_arrow_svg_cache:
        found = None
        for base in QgsApplication.svgPaths():
            cand = Path(base) / "arrows" / "NorthArrow_02.svg"
            if cand.is_file():
                found = str(cand)
                break
        _north_arrow_svg_cache.append(found)
    return _north_arrow_svg_cache[0]


BASEMAPS = {
    "osm": "type=xyz&zmin=0&zmax=19&url=https://tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png",
    # max zoom 17 is the real upper limit of OpenTopoMap (same value as contextily's
    # ctx.providers.OpenTopoMap), not an arbitrary choice.
    "opentopomap": ("type=xyz&zmin=0&zmax=17&url="
                     "https://a.tile.opentopomap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png"),
    # Esri services: the ArcGIS tile path is {z}/{y}/{x}, the reverse of the OSM-style
    # {z}/{x}/{y}; getting it wrong yields a scrambled mosaic without any error.
    # Esri's terms of use require crediting the data providers - see ATTRIBUTION below.
    # Light grey neutral basemap (no API key). Note: CartoDB Positron without a key now
    # returns watermarked tiles with HTTP 200, so it is not usable as a light basemap.
    "esri_gray": ("type=xyz&zmin=0&zmax=16&url="
                   "https://server.arcgisonline.com/ArcGIS/rest/services/"
                   "Canvas/World_Light_Gray_Base/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D"),
    # zmax=19 is the practical upper limit of World_Imagery in most regions.
    "esri_satellite": ("type=xyz&zmin=0&zmax=19&url="
                        "https://server.arcgisonline.com/ArcGIS/rest/services/"
                        "World_Imagery/MapServer/tile/%7Bz%7D/%7By%7D/%7Bx%7D"),
}

# Basemap attribution. The OSM ODbL requires attribution to be clearly legible and not
# hidden, cropped or placed behind a toggle (OSMF Attribution Guidelines,
# osmfoundation.org/wiki/Licence/Attribution_Guidelines); OpenTopoMap (opentopomap.org/about)
# likewise requires it to be "clearly visible". A source credit is also one of the basic
# elements of a map (alongside title, legend, scale bar and north arrow).
ATTRIBUTION = {
    "osm": "© OpenStreetMap contributors",
    "esri_satellite": "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    "esri_gray": "Basemap © Esri, HERE, Garmin, © OpenStreetMap contributors",
    "opentopomap": "Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)",
}


def crs_human_name(authid):
    """Human-readable CRS name for the credit line (e.g. "WGS 84 / UTM zone 33N") so that
    non-GIS readers can tell which coordinate system is used. Falls back to the authid
    itself when QGIS has no description for it."""
    try:
        desc = QgsCoordinateReferenceSystem(authid).description()
    except Exception:
        desc = ""
    return desc or authid


# Photographic (satellite imagery) basemaps laid under the whole canvas unchanged carry
# the same visual weight as the thematic foreground layers and compete with them for
# attention. Following the general figure-ground principle of thematic cartography,
# the basemap is kept low in contrast so the thematic layers carry the emphasis. PHOTO_BASEMAPS marks the basemaps that get this treatment; osm/opentopomap
# are already designed low-saturation line-work basemaps and are not treated.
PHOTO_BASEMAPS = {"esri_satellite"}


# ──────────────────────────── Fonts (multi-script fallback) ────────────────────────────
def pick_font(size, bold=False):
    """Labels often mix scripts (e.g. Latin and non-Latin place names), and a single
    family only covers one of them. QFont.setFamilies() (plural, Qt 5.13+) makes Qt pick,
    per character, the first family in the list that has a glyph for it, so there is no
    need to guess the language from the text. font.families in theme.json is that
    candidate list; changing it changes every figure."""
    fams = set(QFontDatabase().families())
    candidates = [f for f in THEME.get("font", {}).get("families", []) if f in fams]
    if not candidates:
        candidates = [n for n in ("Noto Sans", "DejaVu Sans") if n in fams]
    f = QFont()
    f.setPointSize(size)
    f.setBold(bold)
    if candidates:
        f.setFamilies(candidates)
    return f


# ──────────────────────────── Styling: semantic category -> QGIS symbol ────────────────────────────
def resolve_color(value, counters):
    """'auto' cycles through the global qualitative palette (fallback for layers without a
    semantic category); 'auto:family' cycles independently within that family's palette
    (so own facilities always get warm colours and the grid always gets blues, and other
    layers cannot take their slots); 'risk:high' / 'fixed:water' return fixed semantic
    colours; anything else is returned unchanged (hex / rgba string).
    counters is a dict with one counter per family name, so for a given config the N-th
    use of a family always gets the same colour - this is what keeps colours stable
    between runs.

    Pitfalls:
    - Counters count layer *instances* in the layer pool, not semantic identity. If the
      same data is split into two layer keys (e.g. one copy with a filter to avoid
      overlapping symbols) and both use 'auto:family', the second one gets the next colour
      of the family. For such splits give the layers a fixed colour instead of relying on
      the counter.
    - Bare 'auto' uses the shared '_default' counter across the whole layer pool, in the
      order the layers are written in the config. Two semantically identical layers in
      different figures can therefore get different colours, and a 'fixed:' colour chosen
      for one layer can coincide with a qualitative-palette slot still used by another
      bare-auto layer in the same figure.
    - When moving a layer off the auto counter to a 'fixed:' colour, the next consumer of
      that counter shifts into the freed slot, which may in turn collide with a fixed
      colour. Identify every layer sharing the counter; if they form one semantic family,
      give the whole family fixed colours at once rather than patching one layer."""
    if value == "auto":
        pal = PALETTE["qualitative"]
        idx = counters.get("_default", 0)
        counters["_default"] = idx + 1
        return pal[idx % len(pal)]
    if isinstance(value, str) and value.startswith("auto:"):
        family = value.split(":", 1)[1]
        pal = PALETTE["families"][family]
        idx = counters.get(family, 0)
        counters[family] = idx + 1
        return pal[idx % len(pal)]
    if value == "transparent":
        return "0,0,0,0"
    if isinstance(value, str) and value.startswith("risk:"):
        return PALETTE["risk"][value.split(":", 1)[1]]
    if isinstance(value, str) and value.startswith("fixed:"):
        return PALETTE["fixed"][value.split(":", 1)[1]]
    return value


def reorder_restriction_stack(render_keys, style_keys):
    """Among restriction_* layers, hard exclusions must be drawn above soft buffers: where
    an area is already a hard exclusion, the hatching of a soft buffer showing through only
    adds noise and makes it harder to see that the area is strictly excluded. Draw order
    comes entirely from the order of view.layers (earlier = drawn on top), so writing a soft
    buffer before a hard exclusion is an easy mistake; overlapping hatch patterns then turn
    exactly the most important areas into an unreadable mess.

    The reordering is stable and happens only within the positions originally occupied by
    restriction_* layers (hard before review before soft, relative order within a group
    unchanged). Positions relative to other layers (turbines, basemap, ...) are untouched,
    so a hard exclusion can never end up above the main map subject. Layers that are not
    restriction_* or have no exclusion_type are unaffected.

    "review" (restricted, needs special approval but not absolutely excluded; see
    EXCLUSION_GROUP_LABEL) sits between hard and soft, following the same principle:
    what deserves more attention is drawn higher."""
    ORDER = {"hard": 0, "review": 1, "soft": 2}
    idxs = [i for i, k in enumerate(render_keys)
            if CATEGORIES.get(style_keys.get(k), {}).get("exclusion_type") in ORDER]
    if len(idxs) < 2:
        return render_keys
    subset = [render_keys[i] for i in idxs]
    subset.sort(key=lambda k: ORDER[CATEGORIES[style_keys[k]]["exclusion_type"]])
    out = list(render_keys)
    for i, k in zip(idxs, subset):
        out[i] = k
    return out


def panel_ink_color(bg_hex):
    """Ink colour for panel decorations (north arrow, scale bar, colorbar ticks).

    The panel area shares the same img.fill(QColor(bg)) as the map area, so nothing
    guarantees it is dark; light-grey decorations are nearly invisible on the default
    white background. The background brightness is estimated with the ITU-R BT.601 luma
    approximation: dark backgrounds keep light-grey ink (for dark presets such as
    a dark slide theme), light backgrounds get dark-grey ink. Overlay-mode decorations
    (draw_north / draw_scalebar) are not affected: they use dark ink on a white
    semi-transparent card, a different design for floating over arbitrary basemap content."""
    c = QColor(bg_hex)
    luma = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
    return (60, 60, 60) if luma > 140 else (225, 225, 225)


def build_symbol(spec, counters):
    kind = spec["kind"]
    color = resolve_color(spec.get("color", "auto"), counters)
    outline_raw = spec.get("outline_color")
    if outline_raw == "auto":
        outline_color = color
    elif outline_raw:
        outline_color = resolve_color(outline_raw, counters)
    else:
        outline_color = "#333333"

    if kind == "point":
        sym = QgsMarkerSymbol.createSimple({
            "name": spec.get("shape", "circle"),
            "color": color,
            "size": str(spec.get("size_mm", 3.0)),
            "outline_color": outline_color,
            "outline_width": str(spec.get("outline_width_mm", 0.4)),
        })
    elif kind == "point_ringset":
        # Composite turbine symbol: concentric rings plus a centre dot (the outer ring
        # colour can vary with categorize_by; inner ring and centre use fixed colours).
        # Built from three QgsSimpleMarkerSymbolLayer layers rather than a
        # QgsGeometryGeneratorSymbolLayer with buffer(@geometry, distance): on a
        # geographic-CRS layer (EPSG:4326) the latter passes metres as degrees and produces
        # enormous, invisible rings. Marker layer sizes are always in mm/pixels,
        # independent of the layer CRS, so that problem cannot occur here.
        # Pitfall: color/outline_color at the top of build_symbol are always resolved
        # (default "auto") even though point_ringset does not use them, which would
        # consume a slot of the global auto counter. Categories using this kind must set
        # "color": "transparent" explicitly so the "transparent" branch is taken and the
        # counter is not touched (same issue as described in resolve_color()).
        ring_outer_color = resolve_color(spec.get("ring_outer_color", "auto"), counters)
        ring_inner_color = resolve_color(spec.get("ring_inner_color", "fixed:turbine_ring_inner"), counters)
        center_color = resolve_color(spec.get("center_color", "fixed:turbine_center"), counters)
        outer_layer = QgsSimpleMarkerSymbolLayer()
        outer_layer.setShape(QgsSimpleMarkerSymbolLayer.Circle)
        outer_layer.setSize(float(spec.get("ring_outer_size_mm", 6.5)))
        outer_layer.setColor(QColor(0, 0, 0, 0))
        outer_layer.setStrokeColor(QColor(ring_outer_color))
        outer_layer.setStrokeWidth(float(spec.get("ring_outer_width_mm", 0.6)))
        inner_layer = QgsSimpleMarkerSymbolLayer()
        inner_layer.setShape(QgsSimpleMarkerSymbolLayer.Circle)
        inner_layer.setSize(float(spec.get("ring_inner_size_mm", 4.2)))
        inner_layer.setColor(QColor(0, 0, 0, 0))
        inner_layer.setStrokeColor(QColor(ring_inner_color))
        inner_layer.setStrokeWidth(float(spec.get("ring_inner_width_mm", 0.5)))
        center_layer = QgsSimpleMarkerSymbolLayer()
        center_layer.setShape(QgsSimpleMarkerSymbolLayer.Circle)
        center_layer.setSize(float(spec.get("center_size_mm", 1.8)))
        center_layer.setColor(QColor(center_color))
        center_layer.setStrokeColor(QColor(center_color))
        sym = QgsMarkerSymbol()
        sym.deleteSymbolLayer(0)
        for lyr in (outer_layer, inner_layer, center_layer):
            sym.appendSymbolLayer(lyr)
    elif kind == "line":
        width_mm = float(spec.get("width_mm", 0.6))
        casing_raw = spec.get("casing_color")
        if casing_raw:
            # Any single line colour can match the tone of some basemap (e.g. a brown road
            # over bare soil in satellite imagery) and be visually swallowed by it; changing
            # the colour only moves the problem to another map. The standard cartographic
            # answer is casing: a wider light/white line underneath and the coloured line on
            # top, which keeps contrast on both dark and light backgrounds (same idea as the
            # white halo around labels). Only categories with casing_color use it; other
            # line categories render as before.
            casing_width = width_mm + 2 * float(spec.get("casing_width_mm", 0.5))
            casing_layer = QgsSimpleLineSymbolLayer(QColor(resolve_color(casing_raw, counters)))
            casing_layer.setWidth(casing_width)
            main_layer = QgsSimpleLineSymbolLayer(QColor(color))
            main_layer.setWidth(width_mm)
            if spec.get("line_style"):
                main_layer.setPenStyle(QgsSymbolLayerUtils.decodePenStyle(spec["line_style"]))
            sym = QgsLineSymbol([casing_layer, main_layer])
        else:
            props = {"color": color, "width": str(width_mm)}
            if spec.get("line_style"):
                props["line_style"] = spec["line_style"]
            sym = QgsLineSymbol.createSimple(props)
    elif kind == "fill":
        # hatch_cross (cross-hatching) is reserved for risk:high and looks clearly
        # different from the single-direction hatch of other risk levels, so readers with
        # colour-vision deficiency can distinguish levels of the same hue by texture
        # (redundant encoding).
        HATCH_STYLES = {"hatch": "b_diagonal", "hatch_cross": "diagonal_x"}
        pattern = spec.get("pattern")
        if color not in ("transparent", "0,0,0,0") and pattern in HATCH_STYLES and spec.get("hatch_over_fill"):
            # Hard exclusions should keep the hatch (redundant encoding) and still read as
            # one solid excluded block when they cover large areas; a pure hatch_cross over
            # a large area interleaves with the hatching of soft-buffer layers below and
            # loses visual priority. Two layers are stacked instead: a light, semi-transparent
            # solid base that establishes the shape, and a saturated hatch on top.
            # QgsFillSymbol renders multiple symbol layers natively.
            base_c = QColor(color)
            base_c.setAlpha(140)
            hatch_c = QColor(color)
            outline_c = QColor(outline_color) if outline_color not in ("transparent", "0,0,0,0") else QColor(color)
            base_layer = QgsSimpleFillSymbolLayer(base_c, Qt.SolidPattern, QColor(0, 0, 0, 0))
            hatch_layer = QgsSimpleFillSymbolLayer(hatch_c, QgsSymbolLayerUtils.decodeBrushStyle(HATCH_STYLES[pattern]), outline_c)
            hatch_layer.setStrokeWidth(float(spec.get("outline_width_mm", 0.3)))
            sym = QgsFillSymbol([base_layer, hatch_layer])
        else:
            if color in ("transparent", "0,0,0,0"):
                props = {"color": "0,0,0,0", "style": "no",
                         "outline_color": outline_color,
                         "outline_width": str(spec.get("outline_width_mm", 0.4))}
            else:
                props = {"color": color, "outline_color": outline_color,
                         "outline_width": str(spec.get("outline_width_mm", 0.3))}
                if pattern in HATCH_STYLES:
                    props["style"] = HATCH_STYLES[pattern]
            if spec.get("line_style"):
                props["outline_style"] = spec["line_style"]
            sym = QgsFillSymbol.createSimple(props)
    else:
        raise ValueError(f"Unknown symbol kind: {kind}")
    if "opacity" in spec:
        sym.setOpacity(float(spec["opacity"]))  # QgsSymbol-level method: works for points, lines and fills
    return sym


def resolve_style_spec(style_field, geom_kind):
    """style_field is looked up as a semantic category name first (e.g. 'turbine'); when
    no style is given, fall back to point_generic / line_generic / polygon_generic, the
    generic styles for layers without a semantic category - the semantics are never
    guessed (an arbitrary point layer is not treated as turbines)."""
    if style_field is None:
        default_map = {"point": "point_generic", "line": "line_generic", "polygon": "polygon_generic"}
        style_field = default_map.get(geom_kind, "point_generic")
    if isinstance(style_field, dict):
        return style_field, None
    if isinstance(style_field, str) and style_field.lower().endswith(".qml"):
        return None, style_field
    if style_field in CATEGORIES:
        return CATEGORIES[style_field], None
    raise ValueError(f"Unknown style: {style_field} (neither a built-in category nor a .qml path; "
                      f"built-in categories are listed in assets/report_style.json)")


def adapt_kind_for_geometry(spec, geom_kind):
    """Pitfall: categories that are conceptually lines (e.g. boundary_admin) are almost
    always stored as polygons in real data. A line symbol on a polygon layer does not
    render because the renderer type does not match, so it is converted to a
    "no fill + outline" polygon symbol, which looks like a boundary line."""
    if geom_kind == "polygon" and spec.get("kind") == "line":
        return {"kind": "fill", "color": "transparent",
                "outline_color": spec.get("color", "#333333"),
                "outline_width_mm": spec.get("width_mm", 0.5),
                "line_style": spec.get("line_style")}
    return spec


def build_categorized_renderer(spec, counters):
    """Colour by field value - native multi-colour rendering within one layer, instead of
    several copies of the same table with different filters stacked on each other.
    Pitfall: opening the same GPKG table as four or more separate QgsVectorLayer
    connections (even with different setSubsetString filters) has been observed to trigger
    "unable to open database file" at the GDAL/SQLite level. It is not a threading issue
    (it reproduces with single-threaded QgsMapRendererCustomPainterJob) and the exact
    mechanism is unknown, but it reproduces reliably. Classifying one table by attribute
    must therefore use this native renderer, never "N layers with the same
    source+layername, each with a filter"."""
    field = spec["categorize_by"]
    shared = {k: v for k, v in spec.items() if k not in ("categorize_by", "categories")}
    cats = []
    for value, override in spec["categories"].items():
        sub = {**shared, **override}
        sym = build_symbol(sub, counters)
        cats.append(QgsRendererCategory(value, sym, str(override.get("label", value))))
    return QgsCategorizedSymbolRenderer(field, cats)


def build_rule_renderer(spec, counters):
    """Colour by arbitrary QGIS expression rules (compound conditions such as LIKE/OR that
    a categorized field cannot express). Rendered in one pass over one table, for the same
    reason as in build_categorized_renderer."""
    shared = {k: v for k, v in spec.items() if k != "rules"}
    root_rule = QgsRuleBasedRenderer.Rule(None)
    for r in spec["rules"]:
        sub = {**shared, **{k: v for k, v in r.items() if k not in ("expr", "label")}}
        sym = build_symbol(sub, counters)
        root_rule.appendChild(QgsRuleBasedRenderer.Rule(sym, filterExp=r["expr"], label=r.get("label", "")))
    return QgsRuleBasedRenderer(root_rule)


def apply_style(layer, style_field, counters):
    if not hasattr(layer, "geometryType"):
        return  # raster layers do not use the symbol system
    geom_kind = {0: "point", 1: "line", 2: "polygon"}.get(layer.geometryType())
    spec, qml_path = resolve_style_spec(style_field, geom_kind)
    if qml_path:
        qml_path = resolve_path(qml_path)
        ok, msg = layer.loadNamedStyle(qml_path)
        if not ok:
            print(f"⚠ Failed to load style {qml_path}: {msg}", file=sys.stderr)
        return
    adapted = adapt_kind_for_geometry(spec, geom_kind)
    if adapted is not spec:
        # A line category on polygon data is silently converted to "transparent fill +
        # outline". That is intended for categories such as boundary_admin, but when a
        # category is used in the wrong place (e.g. a centreline road style on road buffer
        # polygons) the whole layer seems to lose its fill. Print an info note as a hint;
        # it is not an error.
        print(f"ℹ {layer.name()}: line category applied to polygon geometry, converted to transparent fill + outline"
              f" (if this layer should have a solid fill, check the category)", file=sys.stderr)
    spec = adapted
    if "categorize_by" in spec:
        layer.setRenderer(build_categorized_renderer(spec, counters))
    elif "rules" in spec:
        layer.setRenderer(build_rule_renderer(spec, counters))
    else:
        layer.setRenderer(QgsSingleSymbolRenderer(build_symbol(spec, counters)))


def apply_labels(layer, field, enabled=True, hide_expr=None, font_pt=8):
    """font_pt: label size in pt (default 8). Useful for reference place-name labels that
    should be visually lighter than turbine IDs and are not rescaled by the label_fs of
    tweak_layers() (see its force_labels parameter).
    hide_expr: QGIS expression; features for which it is true get no native label. Meant
    for points that are both circled by a highlight and carry a dedicated callout: the
    highlight circle (draw_highlights, a QPainter overlay drawn above the native labels)
    inevitably crosses the digits of a label next to the point. The callout already names
    the point, so the plain label is redundant; dropping it is simpler and more reliable
    than trying to route the circle around a label whose final position is decided by the
    QGIS labelling engine at render time."""
    settings = QgsPalLayerSettings()
    if field.strip().startswith(("'", '"')) or "||" in field or "(" in field:
        # QGIS expressions are supported as labels (e.g. "'#' || \"rank\" || ' ' || \"name\""),
        # not only bare field names; plain field names never contain these characters.
        settings.fieldName = field
        settings.isExpression = True
    else:
        settings.fieldName = field
    tf = QgsTextFormat()
    tf.setFont(pick_font(int(round(font_pt))))  # QFont.setPointSize only accepts int (6.5 would crash); setSize keeps the fraction
    tf.setSize(font_pt)
    # White halo: labels sit on basemaps that may be dark or saturated anywhere
    # (OSM/OpenTopoMap/imagery); without a halo dark text disappears on dark areas. A
    # white halo around dark text is baseline practice in cartographic guidance (about
    # 1-2 px, converted to mm here), not an optional embellishment.
    buf = tf.buffer()
    buf.setEnabled(True)
    buf.setSize(0.8)
    buf.setColor(QColor(255, 255, 255, 225))
    tf.setBuffer(buf)
    settings.setFormat(tf)
    if hide_expr:
        settings.dataDefinedProperties().setProperty(
            QgsPalLayerSettings.Show, QgsProperty.fromExpression(f"NOT ({hide_expr})"))
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(bool(enabled))


def clip_raster_to_vector(raster_path, vector_path, where=None):
    """Clip a raster to a vector boundary; cells outside become transparent nodata
    (gdal.Warp(cutlineDSName=..., cropToCutline=True)).

    Without this, a raster keeps the rectangular extent of its source (e.g. national
    downloads that are cut by bounding box, not by the real border), and neighbouring
    countries / sea fill the frame even when a boundary line is drawn on top.

    Pitfall: if vector_path contains many features (e.g. a world borders file) and no
    `where` is given, the cutline is the union of *all* features, not the one nearest the
    raster. A raster's bounding box easily contains parts of neighbouring countries, which
    then remain unclipped along the edge - it runs fine but is silently wrong, and is hard
    to spot when the neighbours' borders are not drawn. Omit `where` only when the vector
    file is known to contain a single feature.

    The cache key is (raster path, vector path, where, both mtimes); results are written to
    the system temp directory and reused, so batch runs over many countries do not re-warp
    large rasters every time. Returns the path of the clipped temporary GeoTIFF."""
    raster_mtime = os.path.getmtime(raster_path)
    vector_mtime = os.path.getmtime(vector_path)
    key = hashlib.sha1(f"{raster_path}|{vector_path}|{where}|{raster_mtime}|{vector_mtime}".encode()).hexdigest()[:16]
    CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLIP_CACHE_DIR / f"{Path(raster_path).stem}__{key}.tif"
    if out_path.exists():
        return str(out_path)
    warp_kwargs = dict(cutlineDSName=str(vector_path), cropToCutline=True, dstNodata="nan", format="GTiff")
    if where:
        warp_kwargs["cutlineWhere"] = where
    gdal.Warp(str(out_path), str(raster_path), **warp_kwargs)
    return str(out_path)


def resolve_ramp(colors):
    """Resolve the raster `colors` field: a list is returned as is; "ramp:<name>" is looked
    up in report.ramps of theme.json; when not set, the full 7-stop viridis
    (= ramps.wind_speed) is used.

    Interpolating only the two viridis end points (#440154 -> #fde725) in RGB passes through
    a greyish purple-brown in the middle and loses perceptual uniformity, hence the full
    ramp. An unknown ramp name raises instead of silently falling back to the default:
    a silent substitution would produce a map that differs from the author's intent without
    any warning."""
    if not colors:
        return PALETTE["ramps"]["wind_speed"]
    if isinstance(colors, str) and colors.startswith("ramp:"):
        name = colors.split(":", 1)[1]
        if name not in PALETTE["ramps"]:
            raise ValueError(f"Unknown colour ramp ramp:{name} (available: {list(PALETTE['ramps'])})")
        return PALETTE["ramps"][name]
    return colors


def apply_raster_style(layer, style_spec):
    """Pseudo-colour rendering of a continuous single-band raster (e.g. wind speed). Not to
    be confused with XYZ basemap tiles, which use their own colours; here local GeoTIFF
    values are interpolated along a colour ramp.

    Pitfall: with three or more colours, passing only colors[0]/colors[-1] to
    QgsGradientColorRamp silently drops the middle stops (a red-yellow-green ramp renders
    as a plain red-to-green interpolation). The intermediate colours are passed as `stops`;
    draw_colorbar() handles the legend the same way."""
    if not style_spec:
        return
    vmin = float(style_spec.get("min", 0))
    vmax = float(style_spec.get("max", 1))
    colors = resolve_ramp(style_spec.get("colors"))
    shader = QgsRasterShader()
    if len(colors) > 2:
        n = len(colors)
        stops = [QgsGradientStop((i / (n - 1)), QColor(c)) for i, c in enumerate(colors) if 0 < i < n - 1]
        ramp = QgsGradientColorRamp(QColor(colors[0]), QColor(colors[-1]), False, stops)
    else:
        ramp = QgsGradientColorRamp(QColor(colors[0]), QColor(colors[-1]))
    color_ramp_shader = QgsColorRampShader(vmin, vmax, ramp, QgsColorRampShader.Interpolated)
    color_ramp_shader.classifyColorRamp()
    shader.setRasterShaderFunction(color_ramp_shader)
    layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))
    layer.setOpacity(float(style_spec.get("opacity", 1.0)))


def apply_basemap_treatment(layer, saturation=-55, brightness=-35, contrast=-12):
    """Desaturate and darken a satellite basemap so that it serves as geographic context and
    does not compete with the thematic foreground (rationale at PHOTO_BASEMAPS). Uses the
    raster layer's own hueSaturationFilter()/brightnessFilter() pipeline, so tiles need no
    preprocessing. The amounts were tuned visually. The three
    values are exposed as defaults; per view only `treatment: false` (turn off) is
    supported for now."""
    layer.hueSaturationFilter().setSaturation(saturation)
    bf = layer.brightnessFilter()
    bf.setBrightness(brightness)
    bf.setContrast(contrast)


# Greyscale conversion of coloured vector basemaps (osm/opentopomap). NatureScot, "Visual
# Representation of Wind Farms" (2017), para. 53, requires greyscale base mapping for ZTV
# figures because a coloured overlay on a coloured basemap mixes (a yellow layer over a
# blue lake reads as green, i.e. woodland). Common practice in wind-farm EIA figure sets is
# to grey the basemap on constraint / layout / shadow-flicker figures while keeping full
# colour on line-only figures (noise contours, site location). Hence the "auto" rule: grey
# the basemap only when the figure contains a solid-filled polygon layer. PHOTO_BASEMAPS
# keep using apply_basemap_treatment above - greying imagery loses feature recognisability.
# Greying is per view: one basemap layer object is shared by several views
# (load_layer_pool loads each layer once), so, like tweak_layers, it is set before
# rendering and restored afterwards instead of being baked in by load_layer.
OUTLINE_ONLY_SYMBOL_LAYERS = {"SimpleLine", "MarkerLine", "HashLine", "ArrowLine", "InterpolatedLine",
                              "RasterLine", "LineburstSymbolLayer", "FilledLine"}


def layer_has_solid_fill(layer):
    """True when a vector polygon layer has a fill symbol with opacity > 0 (outline-only
    boundary polygons do not count). Only the renderer's symbols are inspected, not
    individual features - enough to tell filled zone maps from boundary-only location maps."""
    if not hasattr(layer, "geometryType") or layer.geometryType() != QgsWkbTypes.PolygonGeometry:
        return False
    if layer.opacity() <= 0:
        return False
    renderer = layer.renderer()
    if renderer is None:
        return False
    for sym in renderer.symbols(QgsRenderContext()):
        if sym.type() != QgsSymbol.Fill or sym.opacity() <= 0:
            continue
        for sl in sym.symbolLayers():
            t = sl.layerType()
            if t in OUTLINE_ONLY_SYMBOL_LAYERS:
                continue
            if t == "SimpleFill":
                # No brush or a fully transparent fill = outline-only boundary polygon.
                # Hatching (b_diagonal etc.) counts as fill, like LinePatternFill: hatching
                # over a coloured basemap mixes colours just the same.
                if sl.brushStyle() != Qt.NoBrush and sl.fillColor().alpha() > 0:
                    return True
                continue
            return True  # gradient/SVG/raster image/line pattern/point pattern etc. all count as fill
    return False


def gray_basemaps_for_view(view, defaults, render_keys, layers_by_key, basemap_aliases):
    """Convert coloured vector basemaps to greyscale according to the view's basemap_gray
    (auto/true/false, default auto). Returns a snapshot list for restore_basemap_gray()."""
    raw = view.get("basemap_gray", defaults.get("basemap_gray", "auto"))
    norm = {True: True, False: False, "auto": "auto", "true": True, "on": True, "yes": True,
            "false": False, "off": False, "no": False}
    key = raw.strip().lower() if isinstance(raw, str) else raw
    if key not in norm:
        raise ValueError(f"basemap_gray only accepts auto/true/false (got {raw!r})")
    mode = norm[key]
    targets = [layers_by_key[k] for k in render_keys
               if k in (basemap_aliases or {}) and basemap_aliases[k] not in PHOTO_BASEMAPS]
    if not targets or mode is False:
        return []
    if mode == "auto":
        others = [layers_by_key[k] for k in render_keys if k not in (basemap_aliases or {})]
        if not any(layer_has_solid_fill(l) for l in others):
            return []
    snaps = []
    for lyr in targets:
        hs, bf = lyr.hueSaturationFilter(), lyr.brightnessFilter()
        snaps.append((lyr, hs.grayscaleMode(), bf.brightness(), bf.contrast()))
        hs.setGrayscaleMode(QgsHueSaturationFilter.GrayscaleLuminosity)
        # In pure greyscale the light OSM areas (farmland/green space) turn dark and merge
        # with the roads; brighten slightly and reduce contrast so the basemap recedes to
        # plain geographic context. Values were tuned visually.
        bf.setBrightness(18)
        bf.setContrast(-18)
    return snaps


def restore_basemap_gray(snaps):
    for lyr, gmode, bright, contrast in snaps:
        lyr.hueSaturationFilter().setGrayscaleMode(gmode)
        bf = lyr.brightnessFilter()
        bf.setBrightness(bright)
        bf.setContrast(contrast)


# ──────────────────────────── Layer loading ────────────────────────────
def load_layer(key, spec, counters):
    if "basemap" in spec:
        alias = spec["basemap"]
        uri = BASEMAPS.get(alias)
        if uri is None:
            raise ValueError(f"Unknown basemap alias: {alias} (available: {list(BASEMAPS)})")
        lyr = QgsRasterLayer(uri, spec.get("label") or key, "wms")
        if not lyr.isValid():
            raise RuntimeError(f"Failed to load basemap: {alias}")
        if alias in PHOTO_BASEMAPS and spec.get("treatment", True):
            apply_basemap_treatment(lyr)
        return lyr

    if "raster" in spec:
        path = resolve_path(spec["raster"])
        if spec.get("clip_to"):
            path = clip_raster_to_vector(path, resolve_path(spec["clip_to"]), where=spec.get("clip_to_where"))
        lyr = QgsRasterLayer(path, spec.get("label") or key, "gdal")
        if not lyr.isValid():
            raise RuntimeError(f"Failed to load raster layer [{key}]: {path}")
        apply_raster_style(lyr, spec.get("style"))
        return lyr

    source = resolve_path(spec["source"])
    uri = f"{source}|layername={spec['layername']}" if spec.get("layername") else source
    # Observed with QGIS 3.44 / GDAL 3.8.4: for the second and later tables of a GPKG, the
    # first open within a process fails feature_count / extent / feature access with
    # "unable to open database file", while opening the same table a second time works
    # (unrelated to WAL/journal mode - reproduces in DELETE mode - and to file locking -
    # reproduces after copying the file elsewhere). Missing one table would silently yield
    # no extent. Each vector layer is therefore first opened by a throw-away layer that
    # reads the feature count, and only then is the real layer created.
    _warm = QgsVectorLayer(uri, f"_warm_{key}", "ogr")
    if _warm.isValid():
        _warm.featureCount()
    lyr = QgsVectorLayer(uri, spec.get("label") or key, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"Failed to load layer [{key}]: {source}")

    apply_style(lyr, spec.get("style"), counters)
    if spec.get("filter"):
        lyr.setSubsetString(spec["filter"])
    if spec.get("label_field"):
        excl = spec.get("label_exclude_ids")
        hide_expr = None
        if excl:
            vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in excl["ids"])
            hide_expr = f'"{excl["field"]}" IN ({vals})'
        apply_labels(lyr, spec["label_field"], spec.get("labels_enabled", True), hide_expr=hide_expr,
                     font_pt=spec.get("label_size_pt", 8))
        apply_label_extras(lyr, key, spec)
    elif any(spec.get(f) for f in ("label_offsets", "label_callout", "label_box", "label_distance_mm", "label_placement")):
        print(f"⚠ {key}: label_offsets/label_callout/label_box require label_field; ignored", file=sys.stderr)
    if spec.get("label_obstacle"):
        # Only mark the layer here; the obstacle settings are applied per view in
        # tweak_layers() - see apply_obstacle_for_view() (a view-level labels: off would
        # disable labelsEnabled for the whole layer and the obstacle with it).
        lyr.setCustomProperty("gc_label_obstacle", True)
        lyr.setCustomProperty("gc_label_obstacle_factor", float(spec.get("label_obstacle_factor", 2.0)))
        lyr.setCustomProperty("gc_has_label_text", bool(spec.get("label_field")))
    return lyr


def _top_layer_color(symbol):
    """Colour of the top-most layer of a line symbol. A cased line is "white casing layer +
    coloured line layer" and symbol.color() returns the first (white) layer, which would
    give white labels that vanish on their white halo."""
    layers = symbol.symbolLayers()
    return layers[-1].color() if layers else symbol.color()


def isoline_label_style(layer, settings, spec):
    """Topographic-style labels for isolines (noise dB / shadow-flicker hours). Default
    curved labels sit above/beside the line, so with dense lines it is unclear which line
    a number belongs to, and the view-level label_font_pt makes them as large as turbine
    IDs. Standard isoline labelling is used instead:
    - the number sits on the line (OnLine) and a white box interrupts the line beneath it,
      making the owning line obvious;
    - text colour matches the line colour (rule-based renderers colour per rule);
    - separate size (label_size_pt, default 4 pt, about 6 pt after physical-size scaling),
      not overridden by the view-level size;
    - broken segments are merged and labels repeat at a distance, one or two per line
      rather than numbers everywhere."""
    ls = settings.lineSettings()
    ls.setPlacementFlags(Qgis.LabelLinePlacementFlag.OnLine)
    ls.setMergeLines(True)
    settings.setLineSettings(ls)
    settings.repeatDistance = float(spec.get("label_repeat_mm", 70))
    settings.repeatDistanceUnit = Qgis.RenderUnit.Millimeters
    tf = settings.format()
    # A tight white background box instead of a halo: a halo only surrounds glyphs, so
    # the line shows through the space in "35 dB" and reads as "35-dB". The box covers the
    # whole line segment under the text - the standard broken-line effect of topographic
    # contour labels.
    buf = tf.buffer()
    buf.setEnabled(False)
    tf.setBuffer(buf)
    bg = QgsTextBackgroundSettings()
    bg.setEnabled(True)
    bg.setType(QgsTextBackgroundSettings.ShapeRectangle)
    bg.setSizeType(QgsTextBackgroundSettings.SizeBuffer)
    bg.setSize(QSizeF(0.5, 0.05))
    bg.setSizeUnit(Qgis.RenderUnit.Millimeters)
    bg.setFillColor(QColor(255, 255, 255, 235))
    bg.setStrokeWidth(0)
    bg.setStrokeColor(QColor(0, 0, 0, 0))
    tf.setBackground(bg)
    renderer = layer.renderer()
    if spec.get("label_color"):
        tf.setColor(QColor(resolve_color(spec["label_color"], {})))
    elif isinstance(renderer, QgsSingleSymbolRenderer):
        tf.setColor(_top_layer_color(renderer.symbol()))
    elif isinstance(renderer, QgsRuleBasedRenderer):
        cases = [f"WHEN {r.filterExpression()} THEN '{_top_layer_color(r.symbol()).name()}'"
                 for r in renderer.rootRule().children() if r.symbol() and r.filterExpression()]
        if cases:
            settings.dataDefinedProperties().setProperty(
                QgsPalLayerSettings.Color, QgsProperty.fromExpression("CASE " + " ".join(cases) + " END"))
    settings.setFormat(tf)
    layer.setCustomProperty("gc_label_pt", float(spec.get("label_size_pt", 4)))


def _label_color(value, opacity=None):
    c = QColor(resolve_color(value, {}))  # label accessory colours should be explicit; empty counters keep the global palette untouched
    if opacity is not None:
        c.setAlphaF(float(opacity))
    return c


def apply_label_extras(layer, key, spec):
    """Optional label settings, e.g. for when the automatic placement puts a place-name
    label right on top of a turbine. All are optional; without them the label settings are
    exactly as produced by apply_labels().

    label_offsets: {<value>: [dx_mm, dy_mm]}   pin individual labels manually (dx to the
        right, dy up, in mm on the output canvas - the same unit as symbol size_mm); the
        label is centred on "anchor + offset". Features not listed are placed by PAL as
        usual. <value> matches $id (fid) by default; with label_offset_field: <field or
        expression> it matches that value instead. Implemented as a data-defined
        PositionPoint (layer CRS coordinates) with anchor = point_on_surface.
        **Requires a metric projected map CRS** (e.g. UTM). Pinned labels no longer take
        part in collision avoidance.
    label_distance_mm: distance of the label from its anchor (dist for AroundPoint on point
        layers); with callouts it often needs to be larger, otherwise the leader line is too
        short to see.
    label_callout: {style: line|manhattan|curved, color, width_mm, min_length_mm}
        leader lines (QgsCallout) - drawn only when a label is displaced from its feature
        (pinned or pushed away), not when it sits next to it.
    label_box: {fill, fill_opacity, outline, outline_width_mm, padding_mm, radius_mm}
        label background box (QgsTextBackgroundSettings rectangle); the white halo is kept."""
    labeling = layer.labeling()
    if labeling is None:
        return
    settings = labeling.settings()
    changed = False

    if spec.get("label_placement"):
        modes = {"around": Qgis.LabelPlacement.AroundPoint, "over": Qgis.LabelPlacement.OverPoint,
                 "horizontal": Qgis.LabelPlacement.Horizontal, "free": Qgis.LabelPlacement.Free,
                 # isolines (noise / shadow flicker) are conventionally labelled on the line; line features need line/curved
                 "line": Qgis.LabelPlacement.Line, "curved": Qgis.LabelPlacement.Curved}
        mode = modes.get(spec["label_placement"])
        if mode is None:
            sys.exit(f"✗ Layer {key}: label_placement only supports {list(modes)}, got {spec['label_placement']!r}")
        settings.placement = mode
        changed = True
        if spec["label_placement"] in ("line", "curved"):
            isoline_label_style(layer, settings, spec)

    if spec.get("label_distance_mm") is not None:
        settings.dist = float(spec["label_distance_mm"])
        settings.distUnits = Qgis.RenderUnit.Millimeters
        changed = True

    offsets = spec.get("label_offsets")
    if offsets:
        fld = spec.get("label_offset_field")
        if fld:
            key_expr = fld if (fld.strip().startswith(("'", '"')) or "(" in fld) else f'"{fld}"'
        else:
            key_expr = "$id"
        pairs = [[k_, float(v[0]), float(v[1])] for k_, v in offsets.items()]
        # The PositionPoint expression needs "map units per mm", which depends on the
        # view's extent and dpi, so it is computed in render_view -> tweak_layers; see
        # label_offset_position_expr().
        # Pitfall: @map_scale/@map_crs cannot be used - this script's QgsMapSettings has no
        # map-level expressionContext, so these variables are unavailable when labels are
        # registered; the expression silently returns NULL and labels do not move.
        layer.setCustomProperty("gc_label_offsets", json.dumps({"key_expr": key_expr, "pairs": pairs},
                                                              ensure_ascii=False))
        match_expr = "CASE " + " ".join(f"WHEN {key_expr} = {_expr_literal(k_)} THEN 1"
                                        for k_, _, _ in pairs) + " END"
        ddp = settings.dataDefinedProperties()
        ddp.setProperty(QgsPalLayerSettings.Hali, QgsProperty.fromExpression(
            f"CASE WHEN ({match_expr}) = 1 THEN 'Center' END"))
        ddp.setProperty(QgsPalLayerSettings.Vali, QgsProperty.fromExpression(
            f"CASE WHEN ({match_expr}) = 1 THEN 'Half' END"))
        settings.setDataDefinedProperties(ddp)
        changed = True

    callout_cfg = spec.get("label_callout")
    if callout_cfg:
        style = callout_cfg.get("style", "line")
        cls = {"line": QgsSimpleLineCallout, "manhattan": QgsManhattanLineCallout,
               "curved": QgsCurvedLineCallout}.get(style)
        if cls is None:
            sys.exit(f"✗ Layer {key}: label_callout.style only supports line/manhattan/curved, got {style!r}")
        callout = cls()
        callout.setEnabled(True)
        line_layer = QgsSimpleLineSymbolLayer(_label_color(callout_cfg.get("color", "#333333")))
        line_layer.setWidth(float(callout_cfg.get("width_mm", 0.25)))
        callout.setLineSymbol(QgsLineSymbol([line_layer]))
        callout.setMinimumLength(float(callout_cfg.get("min_length_mm", 0.0)))
        callout.setMinimumLengthUnit(Qgis.RenderUnit.Millimeters)
        settings.setCallout(callout)
        changed = True

    box_cfg = spec.get("label_box")
    if box_cfg:
        tf = settings.format()
        bg = tf.background()
        bg.setEnabled(True)
        bg.setType(QgsTextBackgroundSettings.ShapeRectangle)
        bg.setSizeType(QgsTextBackgroundSettings.SizeBuffer)
        pad = float(box_cfg.get("padding_mm", 0.6))
        bg.setSize(QSizeF(pad, pad * 0.6))
        bg.setSizeUnit(Qgis.RenderUnit.Millimeters)
        bg.setFillColor(_label_color(box_cfg.get("fill", "#FFFFFF"), box_cfg.get("fill_opacity", 0.85)))
        if box_cfg.get("outline"):
            bg.setStrokeColor(_label_color(box_cfg["outline"]))
            bg.setStrokeWidth(float(box_cfg.get("outline_width_mm", 0.2)))
            bg.setStrokeWidthUnit(Qgis.RenderUnit.Millimeters)
        else:
            bg.setStrokeWidth(0)
        if box_cfg.get("radius_mm"):
            r = float(box_cfg["radius_mm"])
            bg.setRadii(QSizeF(r, r))
            bg.setRadiiUnit(Qgis.RenderUnit.Millimeters)
        tf.setBackground(bg)
        settings.setFormat(tf)
        changed = True

    if changed:
        enabled = layer.labelsEnabled()
        layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
        layer.setLabelsEnabled(enabled)


def _expr_literal(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def label_offset_position_expr(layer, map_units_per_mm, map_authid):
    """PositionPoint expression for label_offsets: transform the point_on_surface anchor to
    the map CRS, translate by (dx, dy) x map units per mm, and transform back to the layer
    CRS (QGIS interprets data-defined positions in the layer CRS). Features not listed
    return NULL and are placed automatically by PAL."""
    raw = layer.customProperty("gc_label_offsets", None)
    if not raw:
        return None
    cfg = json.loads(raw)
    lyr_authid = layer.crs().authid()
    whens = []
    for k_, dx, dy in cfg["pairs"]:
        whens.append(
            f"WHEN {cfg['key_expr']} = {_expr_literal(k_)} THEN transform(translate("
            f"transform(point_on_surface($geometry), '{lyr_authid}', '{map_authid}'), "
            f"{dx * map_units_per_mm!r}, {dy * map_units_per_mm!r}), '{map_authid}', '{lyr_authid}')")
    return "CASE " + " ".join(whens) + " END"


def apply_obstacle_for_view(layer):
    """label_obstacle: true (optional label_obstacle_factor, default 2.0, QGIS range 0-10)
    makes this layer (typically turbine points) a labelling obstacle: labels of other
    layers (place names etc.) avoid its symbols. In QGIS the obstacle settings belong to
    the layer's own label settings, and with labelsEnabled=False the layer is excluded from
    the labelling engine entirely, obstacles included - so on views with labels: off a plain
    obstacle flag has no effect. Therefore labelsEnabled is always set to True here and
    "should this layer's text actually be drawn" moves to settings.drawLabels (the
    "Blocking" mode in the QGIS UI: obstacle only, no text).
    Called once per view from tweak_layers() (when labelsEnabled already reflects the view's
    labels switch) and undone by restore_layers()."""
    want_text = bool(layer.labelsEnabled()) and bool(layer.customProperty("gc_has_label_text", False))
    cur = layer.labeling()
    settings = cur.settings() if cur is not None else QgsPalLayerSettings()
    settings.drawLabels = want_text
    obs = settings.obstacleSettings()
    obs.setIsObstacle(True)
    obs.setFactor(float(layer.customProperty("gc_label_obstacle_factor", 2.0)))
    settings.setObstacleSettings(obs)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(settings))
    layer.setLabelsEnabled(True)


def load_layer_pool(cfg):
    counters = {}  # one counter per family name, see resolve_color()
    pool, labels, raster_legend, basemap_aliases, style_keys = {}, {}, {}, {}, {}
    force_label_keys = set()  # layers with force_labels: true - see the force_labels parameter of
                               # tweak_layers() in render_view(); such layers (e.g. reference
                               # place names) ignore the view's labels: on/off switch and are always shown.
    label_topk = {}  # layers with label_top_k_by_area: N, key -> N; see the call of
                      # top_k_ids_by_overlap_area() in render_view(). Many features may fall in
                      # the extent, but only the N with the largest overlap with the extent are
                      # labelled (e.g. sub-district names), instead of a clutter of text.
    dest_crs = None
    crs_setting = cfg.get("crs")
    if crs_setting:
        dest_crs = QgsCoordinateReferenceSystem(crs_setting)
        if not dest_crs.isValid():
            sys.exit(f"✗ Invalid CRS: {crs_setting}")

    project = QgsProject.instance()
    project.clear()

    for key, spec in (cfg.get("layers") or {}).items():
        # style_overrides: per-layer field overrides on top of a named category (e.g.
        # exclusion_type: hard, color), registered in CATEGORIES as a derived category
        # "<category>@<layer key>". Unlike an inline dict style, the derived category still
        # has a string name, so it enters style_keys and the hard/soft legend grouping and
        # reorder_restriction_stack keep working.
        overrides = spec.get("style_overrides")
        if overrides:
            base = spec.get("style")
            if not (isinstance(base, str) and base in CATEGORIES):
                sys.exit(f"✗ Layer {key}: style_overrides requires style to be a built-in category name, got {base!r}")
            derived = f"{base}@{key}"
            CATEGORIES[derived] = {**CATEGORIES[base], **overrides}
            spec = {**spec, "style": derived}
        lyr = load_layer(key, spec, counters)
        pool[key] = lyr
        if isinstance(spec.get("style"), str):
            style_keys[key] = spec["style"]  # only string category names (e.g. "restriction_community");
            # dict / .qml styles have no CATEGORIES entry and are not used by the legend grouping.
        if spec.get("basemap"):
            basemap_aliases[key] = spec["basemap"]
        if spec.get("label"):
            labels[key] = spec["label"]
        if spec.get("force_labels"):
            force_label_keys.add(key)
        if spec.get("label_top_k_by_area"):
            label_topk[key] = int(spec["label_top_k_by_area"])
        if "raster" in spec and spec.get("label") and spec.get("style"):
            st = spec["style"]
            raster_legend[key] = {
                "colors": resolve_ramp(st.get("colors")),
                "min": float(st.get("min", 0)), "max": float(st.get("max", 1)),
                "unit": spec.get("label"),
            }
        if dest_crs is None and hasattr(lyr, "geometryType") and lyr.crs().isValid():
            dest_crs = lyr.crs()  # without an explicit crs, fall back to the CRS of the first vector layer

    if dest_crs is None:
        dest_crs = QgsCoordinateReferenceSystem("EPSG:3857")
    project.setCrs(dest_crs)
    return project, pool, labels, raster_legend, basemap_aliases, style_keys, force_label_keys, label_topk


# ──────────────────────────── Decorations: scale bar / north arrow / legend / notes ────────────────────────────
def nice_number(x):
    """Round a scale-bar length to 1/2/5 x 10^n (nearest; may be above or below x)."""
    if x <= 0:
        return 1
    exp = math.floor(math.log10(x))
    frac = x / (10 ** exp)
    nice = 1 if frac < 1.5 else 2 if frac < 3.5 else 5 if frac < 7.5 else 10
    return nice * (10 ** exp)


def nice_number_floor(x):
    """Round a scale-bar length down only (largest 1/2/5 x 10^n <= x), never above x.

    In panel mode the panel width is measured first (scalebar_panel_layout) and the scale
    bar is drawn later (draw_scalebar_panel). If both used nice_number, which may round up
    (frac=1.6 -> 2, 25% larger), nothing would guarantee that the drawn bar fits in the
    reserved space, and the bar could overflow the legend area. Any "measure first, then
    draw" layout must use this round-down variant - the same rule as legend_layout() /
    draw_legend_at() sharing the same truncated text: never draw wider than measured."""
    if x <= 0:
        return 1
    exp = math.floor(math.log10(x))
    frac = x / (10 ** exp)
    nice = 1 if frac < 2 else 2 if frac < 5 else 5
    return nice * (10 ** exp)


def make_scaler(h_px, h_cm):
    """Derive decoration font sizes from the physical height at which the figure is placed
    in a slide/report - pixel-based sizes become unreadable once the figure is scaled down."""
    h_cm = max(float(h_cm or 12.5), 1.0)
    factor = h_px * 0.0353 / (h_cm * 1.3333)      # 1 pt ~ 0.0353 cm; at Qt's 96 dpi 1 pt ~ 1.333 px

    def fs(target_pt):
        return max(7, int(round(target_pt * factor)))

    fs.k = max(factor, 1.0)
    return fs


def make_distance_area(project):
    """Pitfall: QgsProject.ellipsoid() returns the string 'NONE' by default (not an empty
    string), so `project.ellipsoid() or "WGS84"` never falls back - 'NONE' is truthy.
    willUseEllipsoid() is then False and measureLine() returns degree differences as if
    they were metres (e.g. 3.7 instead of 370000+, giving a "1 m" scale bar). WGS84 must
    be set explicitly and unconditionally."""
    da = QgsDistanceArea()
    da.setSourceCrs(project.crs(), project.transformContext())
    da.setEllipsoid("WGS84")
    return da


def ground_width_m(ext, project):
    """Measure true ground distance with QgsDistanceArea instead of treating CRS units as
    metres - close enough for projected CRSs such as UTM, but clearly wrong for 3857/4326."""
    da = make_distance_area(project)
    y = ext.center().y()
    return da.measureLine(QgsPointXY(ext.xMinimum(), y), QgsPointXY(ext.xMaximum(), y))


def scale_ratio_text(meters_per_px, px_per_cm, h_cm, fmt):
    """Ratio scale text relative to the printed size, e.g. "1:45,000 (map height 12 cm)".

    Map figure frames often state the scale for a paper size ("1:40,000 @ A3"). These
    figures have no fixed paper size but are placed at physical_height_cm, so the text says
    "at a map height of N cm". Apart from the scale bar this is the only information that
    lets readers convert map distances directly, and it is only valid when the physical size
    is known (callers pass ratio only when physical_height_cm is configured explicitly).
    The denominator is rounded to 3 significant figures: pixel -> cm already carries
    placement scaling error, so more digits would be false precision (rounding to thousands
    instead would give e.g. "1:20,871,000" on national-scale maps)."""
    denom = meters_per_px * px_per_cm * 100.0
    step = 10 ** max(0, len(str(int(denom))) - 3)
    d = int(round(denom / step) * step)
    return fmt.format(denom=f"{d:,}", h_cm=h_cm)


def overlay_scalebar_right(w, meters_per_px, fs, ratio):
    """Right edge x of the overlay scale bar (including the ratio text on its right), using
    the same geometry as draw_scalebar."""
    k = fs.k
    bar_m = nice_number(w * 0.18 * meters_per_px)
    bar_px = bar_m / meters_per_px
    right = 22 * k + bar_px
    if ratio:
        fmt, px_per_cm, h_cm = ratio
        right += 8 * k + QFontMetrics(pick_font(fs(8))).horizontalAdvance(
            scale_ratio_text(meters_per_px, px_per_cm, h_cm, fmt))
    return right


def draw_scalebar(painter, w, h, meters_per_px, fs, ratio=None):
    k = fs.k
    target_px = w * 0.18
    bar_m = nice_number(target_px * meters_per_px)
    bar_px = bar_m / meters_per_px
    if bar_px < 40 or bar_px > w * 0.5:
        return
    bh = 9 * k
    x0, y0 = 22 * k, h - 30 * k - bh
    seg = bar_px / 4.0
    painter.setPen(QPen(QColor(0, 0, 0), max(1, k * 0.8)))
    for i in range(4):
        painter.fillRect(QRectF(x0 + i * seg, y0, seg, bh),
                          QColor(255, 255, 255) if i % 2 else QColor(20, 20, 20))
        painter.drawRect(QRectF(x0 + i * seg, y0, seg, bh))
    # Label both ends ("0" on the left, total length on the right). A single total-length
    # label at the left end is easily misread as the left tick value; a scale bar should
    # show its start and end values explicitly.
    end_label = f"{bar_m/1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    painter.setFont(pick_font(fs(9), True))
    fm = painter.fontMetrics()
    off = max(1, int(k))
    ty = y0 - 6 * k

    def halo_text(x, text):
        for dx, dy in ((-off, 0), (off, 0), (0, -off), (0, off)):
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(x + dx), int(ty + dy), text)
        painter.setPen(QColor(15, 15, 15))
        painter.drawText(int(x), int(ty), text)

    halo_text(x0, "0")
    halo_text(x0 + bar_px - fm.horizontalAdvance(end_label), end_label)
    if ratio:
        # The ratio text sits to the right of the bar, vertically centred on it. Below the
        # bar it would collide with the credit line, which in overlay mode sits in the gap
        # directly under the scale bar.
        fmt, px_per_cm, h_cm = ratio
        painter.setFont(pick_font(fs(8)))
        fm2 = painter.fontMetrics()
        ty = y0 + bh / 2 + (fm2.ascent() - fm2.descent()) / 2
        halo_text(x0 + bar_px + 8 * k, scale_ratio_text(meters_per_px, px_per_cm, h_cm, fmt))


def draw_north(painter, w, h, fs):
    """Hand-drawn north arrow in the top-right corner (overlay mode). Radius 19*k with a
    16*k margin: a north arrow is a small auxiliary symbol and should not compete with the
    legend for visual weight. All arrow coordinates are proportional to R, so the arrow
    scales with it."""
    k = fs.k
    R = 19 * k
    cx, cy = w - R - 16 * k, R + 16 * k
    painter.setBrush(QColor(255, 255, 255, 215))
    painter.setPen(QPen(QColor(70, 70, 70), max(1.2, k)))
    painter.drawEllipse(QRectF(cx - R, cy - R, 2 * R, 2 * R))

    tip_y, tail_y, half = cy - R * 0.74, cy + R * 0.10, R * 0.30
    painter.setBrush(QColor(30, 30, 30))
    painter.setPen(Qt.NoPen)
    painter.drawPolygon(QPolygonF([
        QPointF(cx, tip_y), QPointF(cx - half, tail_y),
        QPointF(cx, tail_y - R * 0.14), QPointF(cx + half, tail_y),
    ]))
    painter.setPen(QColor(20, 20, 20))
    painter.setFont(pick_font(fs(10), True))
    painter.drawText(QRectF(cx - R, tail_y, 2 * R, R * 0.9), Qt.AlignCenter, "N")


def colorbar_box_size(unit, fs):
    """Size of draw_colorbar's white box (same geometry, used by overlay planning)."""
    k = fs.k
    bar_w, bar_h, pad, label_h = 14 * k, 150 * k, 8 * k, 14 * k
    w = bar_w + pad * 2 + 32 * k
    if unit:
        w = max(w, QFontMetrics(pick_font(fs(7), True)).horizontalAdvance(unit) + pad * 2)
    return w, bar_h + pad * 2 + (label_h if unit else 0)


def draw_colorbar(painter, tw, th, colors, vmin, vmax, unit, fs):
    """Legend for continuous rasters (e.g. wind speed) - colours without tick values cannot
    be read as numbers. Fixed to the right side below the north arrow, using the same white
    rounded box as the legend and notes."""
    k = fs.k
    bar_w, bar_h = 14 * k, 150 * k
    pad = 8 * k
    label_h = 14 * k
    panel_w = bar_w + pad * 2 + 32 * k
    if unit:
        painter.setFont(pick_font(fs(7), True))
        panel_w = max(panel_w, painter.fontMetrics().horizontalAdvance(unit) + pad * 2)
    panel_h = bar_h + pad * 2 + (label_h if unit else 0)
    x0 = tw - panel_w - 22 * k
    y0 = 22 * k + 62 * k  # clear the north arrow in the top-right corner (radius + margin, with spare gap)

    painter.setBrush(QColor(255, 255, 255, 220))
    painter.setPen(QPen(QColor(90, 90, 90), max(1, k * 0.8)))
    painter.drawRoundedRect(QRectF(x0, y0, panel_w, panel_h), 5 * k, 5 * k)

    bx, by = x0 + pad, y0 + pad
    grad = QLinearGradient(0, by, 0, by + bar_h)
    # Same pitfall as in apply_raster_style(): with 3+ colours, using only the two end
    # colours would make the legend disagree with the multi-stop raster rendering.
    # Position 0 = top = vmax = colors[-1], position 1 = bottom = vmin = colors[0]; the
    # middle colours are placed in between in reverse order (colors run from vmin up to vmax).
    n = len(colors)
    for i, c in enumerate(colors):
        grad.setColorAt(1.0 - i / (n - 1), QColor(c))
    painter.setPen(QPen(QColor(90, 90, 90), max(1, 0.6 * k)))
    painter.setBrush(grad)
    painter.drawRect(QRectF(bx, by, bar_w, bar_h))

    painter.setFont(pick_font(fs(7)))
    painter.setPen(QColor(20, 20, 20))
    painter.drawText(QRectF(bx + bar_w + 4 * k, by - 5 * k, 30 * k, 14 * k),
                      Qt.AlignLeft | Qt.AlignVCenter, f"{vmax:.1f}")
    painter.drawText(QRectF(bx + bar_w + 4 * k, by + bar_h - 9 * k, 30 * k, 14 * k),
                      Qt.AlignLeft | Qt.AlignVCenter, f"{vmin:.1f}")
    if unit:
        painter.setFont(pick_font(fs(7), True))
        painter.drawText(QRectF(x0, y0 + panel_h - label_h - pad * 0.3, panel_w, label_h),
                          Qt.AlignCenter, unit)


def draw_title(painter, w, text, fs):
    k = fs.k
    painter.setFont(pick_font(fs(12), True))
    fm = painter.fontMetrics()
    pad = 10 * k
    hgt = fm.height() + pad * 1.6
    painter.setBrush(QColor(255, 255, 255, 225))
    painter.setPen(Qt.NoPen)
    painter.drawRect(QRectF(0, 0, w, hgt))
    painter.setPen(QColor(20, 20, 20))
    painter.drawText(QRectF(pad, 0, w - 2 * pad, hgt), Qt.AlignVCenter | Qt.AlignLeft, text)


def symbols_of(renderer):
    out = []
    if renderer is None:
        return out
    if hasattr(renderer, "symbol") and renderer.symbol() is not None:
        out.append(renderer.symbol())
    for attr in ("categories", "ranges"):
        if hasattr(renderer, attr):
            for it in getattr(renderer, attr)():
                if it.symbol() is not None:
                    out.append(it.symbol())
    return out


def legend_symbol(layer, size=22, feat=None):
    """Symbol preview taken from the layer's actual renderer - colours are never guessed."""
    try:
        r = layer.renderer()
        if r is None:
            return None
        sym = None
        if feat is not None and hasattr(r, "classAttribute") and hasattr(r, "categories"):
            try:
                attr = r.classAttribute()
                val = feat[attr] if attr in feat.fields().names() else None
                if val is not None:
                    for cat in r.categories():
                        if str(cat.value()) == str(val) and cat.symbol() is not None:
                            sym = cat.symbol()
                            break
            except Exception:
                sym = None
        if sym is None and hasattr(r, "symbol"):
            sym = r.symbol()
        if sym is None and hasattr(r, "categories"):
            cats = [c for c in r.categories() if c.symbol() is not None]
            sym = cats[0].symbol() if cats else None
        if sym is None and hasattr(r, "ranges"):
            rngs = [x for x in r.ranges() if x.symbol() is not None]
            sym = rngs[0].symbol() if rngs else None
        if sym is None:
            return None
        return QgsSymbolLayerUtils.symbolPreviewPixmap(sym, QSize(size, size))
    except Exception:
        return None


LEGEND_FILL_FIELDS = ("legend_fill_color", "legend_outline_color", "legend_outline_width_mm", "legend_opacity")


def legend_fill_override_rows(key, cat_spec, rows):
    """Separate legend swatch colours for fill categories (the polygon counterpart of
    legend_casing_color).

    A semi-transparent fill (opacity < 1) over a dark satellite basemap is seen on the map
    as a composite colour (fill x opacity + dark basemap), while legend_symbol() reuses the
    renderer preview, which on a light legend panel looks pale - the legend no longer
    matches what the map shows. A legend swatch should reproduce the composite appearance.
    As with legend_casing_color, a legend-only symbol is built here; the layer's real
    renderer is not touched.

    Fields (all optional; any one of them activates this path; they may be set in the
    category or via style_overrides):
      legend_fill_color       legend fill colour (usually the composite colour measured on the map, fixed:x or hex)
      legend_outline_color    legend outline colour
      legend_outline_width_mm legend outline width
      legend_opacity          overall opacity of the legend symbol; defaults to 1.0 when
                              legend_fill_color is set (it already is the final composite
                              colour and must not be diluted again by the category opacity),
                              otherwise the category opacity is kept
    pattern/hatch_over_fill are kept; only colours change. Only single-symbol fill
    categories are supported - for categorized/rule renderers a single symbol would paint
    every legend row the same colour, so they are skipped with a warning."""
    if cat_spec.get("kind") != "fill" or "categorize_by" in cat_spec or "rules" in cat_spec or len(rows) != 1:
        print(f"⚠ {key}: legend_fill_* only supports single-symbol fill categories; ignored", file=sys.stderr)
        return rows
    spec = dict(cat_spec)
    if cat_spec.get("legend_fill_color") is not None:
        spec["color"] = cat_spec["legend_fill_color"]
        spec["opacity"] = 1.0
    if cat_spec.get("legend_outline_color") is not None:
        spec["outline_color"] = cat_spec["legend_outline_color"]
    if cat_spec.get("legend_outline_width_mm") is not None:
        spec["outline_width_mm"] = cat_spec["legend_outline_width_mm"]
    if cat_spec.get("legend_opacity") is not None:
        spec["opacity"] = cat_spec["legend_opacity"]
    # Empty counters: legend_* fields should hold explicit colours; a mistaken "auto" gets
    # the first palette colour but does not consume/pollute the global colour counters
    # (map colours are unaffected).
    pm = QgsSymbolLayerUtils.symbolPreviewPixmap(build_symbol(spec, {}), QSize(22, 22))
    return [(pm, lbl) for _, lbl in rows]


def count_in_extent(layer, ext_prj, project, cap=3):
    """The legend must reflect what is actually visible, not list the config blindly -
    decided by the number of features within the map extent."""
    if not hasattr(layer, "getFeatures"):
        return cap
    try:
        rect = QgsCoordinateTransform(project.crs(), layer.crs(), project).transformBoundingBox(ext_prj)
        req = QgsFeatureRequest().setFilterRect(rect).setNoAttributes()
        n = 0
        for _ in layer.getFeatures(req):
            n += 1
            if n >= cap:
                break
        return n
    except Exception:
        return cap  # if counting fails, do not drop the entry; better one legend row too many


def first_feature_in_extent(layer, ext_prj, project):
    if not hasattr(layer, "getFeatures"):
        return None
    try:
        rect = QgsCoordinateTransform(project.crs(), layer.crs(), project).transformBoundingBox(ext_prj)
        req = QgsFeatureRequest().setFilterRect(rect).setLimit(1)
        return next(layer.getFeatures(req), None)
    except Exception:
        return None


def count_matching_in_extent(layer, ext_prj, project, expr, cap=1):
    """Like count_in_extent with an additional expression filter - used by categorized/rule
    renderers to check per class whether it actually occurs within the extent. Queries the
    already opened layer object, so it does not trigger the "same table opened repeatedly"
    problem (see build_categorized_renderer)."""
    if not hasattr(layer, "getFeatures"):
        return cap
    try:
        rect = QgsCoordinateTransform(project.crs(), layer.crs(), project).transformBoundingBox(ext_prj)
        req = QgsFeatureRequest().setFilterRect(rect).setFilterExpression(expr).setNoAttributes()
        n = 0
        for _ in layer.getFeatures(req):
            n += 1
            if n >= cap:
                break
        return n
    except Exception:
        return cap


def top_k_ids_by_overlap_area(layer, ext_prj, project, k):
    """QGIS feature ids ($id, as used by PAL's Show data-defined expression) of the k
    features with the largest intersection area with the extent (ext_prj, project/map CRS),
    in descending order. Intended for reference place-name labels: many features (e.g.
    several sub-districts) may fall in the extent, but only the most prominent ones should
    be labelled. Areas are computed in the layer's own CRS - good enough, no exact
    cross-CRS area (same CRS handling as count_in_extent). Returns an empty list when
    nothing can be determined or on error; callers must handle that ("no limit" or "show
    nothing") and not assume a non-empty result."""
    if not hasattr(layer, "getFeatures"):
        return []
    try:
        rect = QgsCoordinateTransform(project.crs(), layer.crs(), project).transformBoundingBox(ext_prj)
        ext_geom = QgsGeometry.fromRect(rect)
        req = QgsFeatureRequest().setFilterRect(rect)
        scored = []
        for feat in layer.getFeatures(req):
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            inter = geom.intersection(ext_geom)
            area = inter.area() if inter is not None and not inter.isEmpty() else 0.0
            scored.append((area, feat.id()))
        scored.sort(key=lambda x: -x[0])
        return [fid for _, fid in scored[:k]]
    except Exception:
        return []


def legend_rows_for_layer(lyr, ext, project, label):
    """Categorized/rule renderers yield one legend row per category/rule (filtered by
    whether that class actually occurs within the extent); single-symbol renderers yield
    one row. Callers only deal with "how many legend rows does this layer contribute",
    regardless of the renderer type."""
    r = lyr.renderer() if hasattr(lyr, "renderer") else None
    if isinstance(r, QgsCategorizedSymbolRenderer):
        rows = []
        field = r.classAttribute()
        for cat in r.categories():
            val = cat.value()
            expr = f'"{field}" = ' + (f"'{val}'" if isinstance(val, str) else str(val))
            if count_matching_in_extent(lyr, ext, project, expr) > 0:
                rows.append((QgsSymbolLayerUtils.symbolPreviewPixmap(cat.symbol(), QSize(22, 22)),
                             cat.label() or str(val)))
        return rows
    if isinstance(r, QgsRuleBasedRenderer):
        rows = []
        for rule in r.rootRule().children():
            expr = rule.filterExpression()
            if count_matching_in_extent(lyr, ext, project, expr) > 0:
                rows.append((QgsSymbolLayerUtils.symbolPreviewPixmap(rule.symbol(), QSize(22, 22)),
                             rule.label() or expr))
        return rows
    if count_in_extent(lyr, ext, project) > 0:
        return [(legend_symbol(lyr, feat=first_feature_in_extent(lyr, ext, project)), label)]
    return []


def _avoid_latin_word_split(text, cut):
    """Move a cut point found by per-character greedy wrapping back to the start of a run of
    ASCII letters/digits, so that acronyms and model numbers mixed into CJK legend text are
    not split in the middle (a fragment of an acronym is unreadable, unlike a CJK word
    split across lines, where each character still reads on its own). Only ASCII
    alphanumeric runs are handled; CJK word boundaries are not (no lightweight, reliable
    segmentation rule exists). If moving back would leave the line (almost) empty, the
    original cut is kept, so an overlong word cannot empty a whole line."""
    if cut <= 0 or cut >= len(text):
        return cut
    if not (text[cut - 1].isascii() and text[cut - 1].isalnum()
            and text[cut].isascii() and text[cut].isalnum()):
        return cut
    start = cut
    while start > 0 and text[start - 1].isascii() and text[start - 1].isalnum():
        start -= 1
    return start if start > 0 else cut


# Closing punctuation (fullwidth and ASCII) that must not start a line, and opening
# brackets that must not end one. Written as escapes: fullwidth ) corner/lenticular/angle
# brackets, fullwidth comma, ideographic full stop and comma, fullwidth ; : ! ?
_NO_LINE_START = set("\uff09)\u300d\u300f\u3011\u300b\u3009\uff0c\u3002\u3001\uff1b\uff1a\uff01\uff1f%,.;:!?")
_NO_LINE_END = set("\uff08(\u300c\u300e\u3010\u300a\u3008")


def _avoid_orphan_punct(text, cut):
    """CJK line-breaking rule (kinsoku): the next line must not start with closing
    punctuation and this line must not end with an opening bracket - otherwise wrapped
    legend text can leave a lone closing parenthesis on its own line. The cut is moved one
    character back so the preceding character travels with the punctuation to the next
    line; near the start of the line (< 2 characters) the original cut is kept so the line
    is never emptied."""
    if 0 < cut < len(text) and cut > 2:
        if text[cut] in _NO_LINE_START or text[cut - 1] in _NO_LINE_END:
            return cut - 1
    return cut


LEGEND_GAP_K = 10        # gap between swatch and text (x k); 8 left swatch and text almost touching
LEGEND_ROW_MULT = 1.65   # row height = font line height x this factor; legend design guidance
                         # suggests 1.5-2x line spacing (1.45 felt cramped)
LEGEND_TITLE_MULT = 1.7  # same, for the title row


def legend_layout(painter, entries, title, fs, max_w=None):
    """Measure the legend box without drawing it - placement needs the size first.

    max_w: upper bound on the legend box width in pixels. Common legend-design
    guidance (e.g. Esri cartography blog posts) says a legend should not take more
    than roughly 15-20% of the map width, but a content-driven width can grow without
    bound when labels are long. When max_w is given, labels are wrapped to the
    available width using the same rule as draw_note / text_block_panel_layout: at
    most two lines, and only if two lines still do not fit is the text elided with
    an ellipsis. Single-line hard truncation is avoided on purpose, because legend
    text can carry qualifying wording (e.g. "requires special approval, not an
    absolute exclusion") that must not be silently cut off. The title is still
    elided to a single line with QFontMetrics.elidedText (titles are short). Without
    max_w (the default) no width limit is applied. The caller (render_view) derives
    the limit from the canvas width via the legend_max_width_frac setting.

    The returned entries/title are the final, already-wrapped versions; draw_legend_at
    uses exactly these rather than the raw inputs, so sizing and drawing can never
    use different text.

    Each entry may be (pm, label) or (pm, label, is_header); a missing third element
    means is_header=False (backwards compatible). Header entries (group sub-titles,
    e.g. hard exclusions vs. soft buffers) have no symbol swatch (pm is None) and are
    drawn in bold by draw_legend_at. In the returned out_entries the label becomes
    `lines` (list[str] of length 1-2); draw_legend_at makes the row that tall.
    """
    k = fs.k
    painter.setFont(pick_font(fs(8)))
    fm = painter.fontMetrics()
    pad, sym = 9 * k, int(round(17 * k))
    gap = LEGEND_GAP_K * k
    overhead = pad * 2 + sym + gap

    entries = [(e[0], e[1], e[2] if len(e) > 2 else False) for e in entries]
    row_h = fm.height() * LEGEND_ROW_MULT

    def wrap_label(lbl, avail):
        if fm.horizontalAdvance(lbl) <= avail:
            return [lbl]
        lines, remaining = [], lbl
        while remaining and len(lines) < 2:
            cut = len(remaining)
            while cut > 1 and fm.horizontalAdvance(remaining[:cut]) > avail:
                cut -= 1
            cut = _avoid_orphan_punct(remaining, _avoid_latin_word_split(remaining, cut))
            lines.append(remaining[:cut])
            remaining = remaining[cut:]
        if remaining:
            last = lines[-1]
            while last and fm.horizontalAdvance(last + "…") > avail:
                last = last[:-1]
            lines[-1] = (last or lines[-1][:1]) + "…"
        return lines

    if max_w:
        label_budget = max(20, max_w - overhead)
        out_entries = [(pm, wrap_label(lbl, label_budget), is_header) for pm, lbl, is_header in entries]
    else:
        out_entries = [(pm, [lbl], is_header) for pm, lbl, is_header in entries]

    tw = max((fm.horizontalAdvance(ln) for _, lines, _ in out_entries for ln in lines), default=60)
    box_w = overhead + tw
    title_h = 0
    out_title = title
    if title:
        painter.setFont(pick_font(fs(10), True))
        if max_w:
            title_budget = max(20, max_w - pad * 2)
            out_title = painter.fontMetrics().elidedText(title, Qt.ElideRight, int(title_budget))
        box_w = max(box_w, pad * 2 + painter.fontMetrics().horizontalAdvance(out_title) + 6 * k)
        title_h = painter.fontMetrics().height() * LEGEND_TITLE_MULT
    if max_w:
        box_w = min(box_w, max_w)
    entries_h = sum(row_h * len(lines) for _, lines, _ in out_entries)
    box_h = pad * 2 + entries_h + title_h
    return {"w": box_w, "h": box_h, "row_h": row_h, "pad": pad, "sym": sym, "gap": gap,
            "title_h": title_h, "label_w": tw + 4 * k, "entries": out_entries, "title": out_title}


def draw_legend_at(painter, x0, y0, lay, fs, draw_bg=True):
    """Draw a legend measured by legend_layout.

    entries/title are taken from `lay`: if legend_layout wrapped or elided them, the
    drawn text must be that same text. Passing the raw entries again would let the
    measured box and the drawn text disagree (text clipped by a box that is too short).

    draw_bg=False is used in panel mode, where the legend is one section of a single
    unified panel drawn by draw_unified_panel; drawing another white rounded box would
    produce a card-inside-a-card with a double border. Overlay mode (legend floating
    on the map) keeps draw_bg=True (the default)."""
    k = fs.k
    pad, sym, row_h, title_h, gap = lay["pad"], lay["sym"], lay["row_h"], lay["title_h"], lay["gap"]
    entries, title = lay["entries"], lay["title"]

    if draw_bg:
        # Light card: thin light-grey outline, near-opaque white fill and a slightly
        # larger corner radius - the "lightweight card" treatment common in modern data
        # visualisation tools; softer than a dark outline so it does not compete with
        # the map itself.
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.setPen(QPen(QColor(170, 170, 170), max(1, k * 0.7)))
        painter.drawRoundedRect(QRectF(x0, y0, lay["w"], lay["h"]), 6 * k, 6 * k)

    y = y0 + pad
    if title:
        painter.setFont(pick_font(fs(10), True))
        painter.setPen(QColor(20, 20, 20))
        painter.drawText(QRectF(x0 + pad, y, lay["w"] - 2 * pad, title_h),
                          Qt.AlignLeft | Qt.AlignVCenter, title)
        y += title_h
    painter.setFont(pick_font(fs(8)))
    for pm, lines, is_header in entries:
        entry_h = row_h * len(lines)
        if is_header:
            # Group sub-title: no swatch, bold, not indented - a clear step in the
            # visual hierarchy between headings and entries (distinct size/weight).
            painter.setFont(pick_font(fs(8), True))
            painter.setPen(QColor(70, 70, 70))
            for i, ln in enumerate(lines):
                painter.drawText(QRectF(x0 + pad, y + i * row_h, lay["label_w"] + sym + gap, row_h),
                                  Qt.AlignLeft | Qt.AlignVCenter, ln)
            painter.setFont(pick_font(fs(8)))
            y += entry_h
            continue
        if pm is not None and not pm.isNull():
            # Shared light-grey chip behind every swatch. Point symbols (circles,
            # triangles) leave much more empty space in symbolPreviewPixmap than fill
            # symbols, which almost fill the box, so at the same sym x sym size point
            # entries look "lighter". A common chip gives point symbols the same
            # container edge and evens out the visual weight; semi-transparent hatch
            # fills also read better on light grey than on pure white. The swatch is
            # vertically centred on the entry, which may span two lines.
            painter.setPen(QPen(QColor(225, 225, 225), max(1, k * 0.5)))
            painter.setBrush(QColor(246, 246, 246))
            painter.drawRoundedRect(QRectF(x0 + pad, y + (entry_h - sym) / 2, sym, sym), 2 * k, 2 * k)
            spm = pm.scaled(sym, sym, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(int(x0 + pad), int(y + (entry_h - sym) / 2), spm)
        painter.setPen(QColor(25, 25, 25))
        for i, ln in enumerate(lines):
            painter.drawText(QRectF(x0 + pad + sym + gap, y + i * row_h, lay["label_w"], row_h),
                              Qt.AlignLeft | Qt.AlignVCenter, ln)
        y += entry_h


def region_busyness(img, x0, y0, w, h, bg, stride=6):
    """Fraction of pixels in a region that differ clearly from the background colour -
    a proxy for "is there real content here". Computed purely from rendered pixels,
    so the result is deterministic for the same image."""
    x0i, y0i = max(0, int(x0)), max(0, int(y0))
    x1i, y1i = min(img.width(), int(x0 + w)), min(img.height(), int(y0 + h))
    if x1i <= x0i or y1i <= y0i:
        return 1.0
    total = differ = 0
    br, bgc, bb = bg.red(), bg.green(), bg.blue()
    for y in range(y0i, y1i, stride):
        for x in range(x0i, x1i, stride):
            c = img.pixelColor(x, y)
            total += 1
            if abs(c.red() - br) + abs(c.green() - bgc) + abs(c.blue() - bb) > 40:
                differ += 1
    return (differ / total) if total else 1.0


def pick_legend_position(img, box_w, box_h, margin, bg_hex, bottom_reserve=0, exclude=(), top=0):
    """Pick the emptiest of three candidate corners (top-left / bottom-left /
    bottom-right; top-right is reserved for the north arrow) for the legend, so it
    does not cover the features being shown. Based on the rendered pixels; a pure,
    deterministic function - same config and data always give the same position.
    When a basemap fills the whole frame (e.g. imagery) all corners score alike and
    the choice falls back to top-left.
    bottom_reserve: extra upward shift for the two bottom candidates, leaving room
    for the scale bar.
    exclude: candidate names to skip. The credit line is drawn on a translucent white
    box in the emptier bottom corner; being close to the canvas background colour it
    may not register as "busy" in region_busyness, so the caller must exclude that
    corner explicitly once the credit line position is chosen."""
    bg = QColor(bg_hex)
    W, H = img.width(), img.height()
    candidates = [
        ("top-left", margin, top + margin),
        ("bottom-left", margin, H - margin - bottom_reserve - box_h),
        ("bottom-right", W - margin - box_w, H - margin - bottom_reserve - box_h),
    ]
    candidates = [c for c in candidates if c[0] not in exclude] or candidates
    scored = [(region_busyness(img, x, y, box_w, box_h, bg), name, x, y) for name, x, y in candidates]
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][2], scored[0][3]


def attribution_box_size(text, fs):
    """Measure (without drawing) the credit-line box, so the caller can choose a
    corner first. Sizing and drawing share the same font-size logic."""
    k = fs.k
    fm = QFontMetrics(pick_font(fs(6)))
    pad = 5 * k
    return fm.horizontalAdvance(text) + pad * 2, fm.height() + pad * 0.8


def draw_attribution(painter, map_x, map_y, map_w, map_h, text, fs, align="left"):
    """Small credit line: basemap attribution (ODbL / OpenTopoMap require it to be
    clearly visible and unobstructed) plus data source / CRS / date. It is always
    drawn inside the map area itself, not in the decoration panel - the panel can be
    moved outside the map by decoration_layout, where the credit could be mistaken
    for non-map ancillary information. The font is deliberately a step smaller than
    legend and title text: readable, but not competing with the map content.

    align: "left" or "right". The caller uses attribution_box_size() and
    region_busyness() to pick the emptier bottom corner, since restriction polygons
    or hatch fills often sit in one corner and would show through the text. The box
    background is 235/255 opaque to minimise patterns bleeding through."""
    k = fs.k
    painter.setFont(pick_font(fs(6)))
    fm = painter.fontMetrics()
    pad = 5 * k
    tw = fm.horizontalAdvance(text)
    th = fm.height()
    box_w, box_h = tw + pad * 2, th + pad * 0.8
    if align == "right":
        x0 = map_x + map_w - box_w - 6 * k
    else:
        x0 = map_x + 6 * k
    y0 = map_y + map_h - box_h - 6 * k
    painter.setBrush(QColor(255, 255, 255, 235))
    painter.setPen(Qt.NoPen)
    painter.drawRect(QRectF(x0, y0, box_w, box_h))
    painter.setPen(QColor(60, 60, 60))
    painter.drawText(QRectF(x0 + pad, y0, tw + pad, th + pad), Qt.AlignLeft | Qt.AlignVCenter, text)


def draw_note(painter, x_right, y_bottom, text, fs, color=(200, 40, 40), left_limit=None):
    """Note box anchored at its bottom-right corner, growing up and to the left, so it
    does not collide with the legend or scale bar.

    Long notes (typically a list such as "Not found in view: A, B, C") are wrapped to
    at most two lines rather than truncated, because truncation silently drops list
    items. The preferred break is at the ", " list separator closest to half the
    available width; if there is no separator (a single long label) or the two halves
    still do not fit, the break falls back to the character position closest to half
    width. Only if two lines still do not fit is an ellipsis used - at that length
    some information loss is preferable to an ever-taller box."""
    k = fs.k
    painter.setFont(pick_font(fs(10), True))
    fm = painter.fontMetrics()
    pad, dot = 10 * k, 11 * k
    if left_limit is None:
        left_limit = 22 * k
    max_w = x_right - left_limit
    avail = max_w - pad * 2 - dot - 8 * k

    lines = [text]
    if fm.horizontalAdvance(text) > avail:
        line1 = line2 = None
        if ", " in text:
            parts = text.split(", ")
            best_i, best_diff = None, None
            for i in range(1, len(parts)):
                cand = ", ".join(parts[:i])
                diff = abs(fm.horizontalAdvance(cand) - avail / 2)
                if best_diff is None or diff < best_diff:
                    best_i, best_diff = i, diff
            line1, line2 = ", ".join(parts[:best_i]), ", ".join(parts[best_i:])
        if not (line1 and fm.horizontalAdvance(line1) <= avail and fm.horizontalAdvance(line2) <= avail):
            best_i, best_diff = None, None
            for i in range(1, len(text)):
                diff = abs(fm.horizontalAdvance(text[:i]) - avail / 2)
                if best_diff is None or diff < best_diff:
                    best_i, best_diff = i, diff
            line1, line2 = text[:best_i], text[best_i:]
        if fm.horizontalAdvance(line1) <= avail and fm.horizontalAdvance(line2) <= avail:
            lines = [line1, line2]
    for i, ln in enumerate(lines):
        while ln and fm.horizontalAdvance(ln) > avail and len(ln) > 8:
            ln = ln[:-2] + "…"
        lines[i] = ln

    w = max(fm.horizontalAdvance(ln) for ln in lines) + pad * 2 + dot + 8 * k
    row_h = fm.height()
    hgt = row_h * len(lines) + pad * 1.4
    x, y = x_right - w, y_bottom - hgt
    painter.setBrush(QColor(255, 255, 255, 232))
    painter.setPen(QPen(QColor(*color), max(1.4, k)))
    painter.drawRoundedRect(QRectF(x, y, w, hgt), 5 * k, 5 * k)
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor(*color), max(2.2, k * 1.8)))
    painter.drawEllipse(QRectF(x + pad, y + (hgt - dot) / 2, dot, dot))
    painter.setPen(QColor(30, 30, 30))
    for i, ln in enumerate(lines):
        painter.drawText(QRectF(x + pad + dot + 8 * k, y + i * row_h, w, row_h),
                          Qt.AlignVCenter | Qt.AlignLeft, ln)
    return y


def text_block_panel_layout(painter, text, fs, max_w, font_pt=7, has_dot=False, max_lines=4):
    """Generic sizing function for small text blocks in panel mode (credit line,
    "not found in view" note, highlight caption). Same discipline as legend_layout:
    the returned `lines` are exactly what draw_text_block_panel draws; nothing is
    re-wrapped at draw time. Breaks are found character by character (the position
    where one more character would overflow), adjusted to avoid splitting Latin words
    and orphaning punctuation; O(n^2) is fine for short text. More than max_lines
    lines means abnormally long text: the last line is elided instead of growing the
    panel indefinitely."""
    k = fs.k
    painter.setFont(pick_font(fs(font_pt)))
    fm = painter.fontMetrics()
    pad = 9 * k
    dot = 8 * k if has_dot else 0
    gap = 6 * k if has_dot else 0
    avail = max(20, max_w - pad * 2 - dot - gap)
    lines, remaining = [], text
    while remaining and len(lines) < max_lines:
        cut = len(remaining)
        while cut > 1 and fm.horizontalAdvance(remaining[:cut]) > avail:
            cut -= 1
        cut = _avoid_orphan_punct(remaining, _avoid_latin_word_split(remaining, cut))
        lines.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining and lines:
        last = lines[-1]
        while last and fm.horizontalAdvance(last + "…") > avail:
            last = last[:-1]
        lines[-1] = (last or lines[-1][:1]) + "…"
    row_h = fm.height()
    w = max((fm.horizontalAdvance(ln) for ln in lines), default=0) + pad * 2 + dot + gap
    h = row_h * len(lines) + pad * 1.3
    return {"w": w, "h": h, "lines": lines, "row_h": row_h, "pad": pad, "dot": dot, "gap": gap, "font_pt": font_pt}


def info_card_panel_layout(painter, title, lines, fs, max_w, font_title_pt=8, font_body_pt=7.5):
    """Multi-line statistics card for panel mode: a title plus rows the caller has
    already formatted (e.g. "Area: 12.5 km2" per row).

    Unlike text_block_panel_layout this does NOT re-flow text character by character:
    that function is meant for a paragraph, whereas here each row is an independent,
    semantically complete item. Re-flowing could split one value across two lines, so
    each row is handled on its own and only elided if that single row is too wide
    (other rows are unaffected)."""
    k = fs.k
    pad = 9 * k
    row_gap = 3 * k
    avail = max(20, max_w - pad * 2)

    def _fit(txt, fm):
        if fm.horizontalAdvance(txt) <= avail:
            return txt
        cut = len(txt)
        while cut > 1 and fm.horizontalAdvance(txt[:cut] + "…") > avail:
            cut -= 1
        return txt[:cut] + "…"

    painter.setFont(pick_font(fs(font_title_pt), True))
    fm_title = painter.fontMetrics()
    title_line = _fit(title, fm_title) if title else None
    title_h = fm_title.height() if title_line else 0

    painter.setFont(pick_font(fs(font_body_pt)))
    fm_body = painter.fontMetrics()
    body_lines = [_fit(ln, fm_body) for ln in lines]
    body_row_h = fm_body.height()

    widths = [fm_body.horizontalAdvance(ln) for ln in body_lines]
    if title_line:
        widths.append(fm_title.horizontalAdvance(title_line))
    w = max(widths, default=0) + pad * 2
    h = (title_h + row_gap if title_line else 0) + body_row_h * len(body_lines) + pad * 1.3
    return {"w": w, "h": h, "title": title_line, "lines": body_lines,
            "title_h": title_h, "body_row_h": body_row_h, "row_gap": row_gap,
            "pad": pad, "font_title_pt": font_title_pt, "font_body_pt": font_body_pt}


def draw_info_card_panel(painter, x, y, lay, fs, text_color=(60, 60, 60)):
    pad = lay["pad"]
    yy = y + pad * 0.5
    painter.setPen(QColor(*text_color))
    if lay["title"]:
        painter.setFont(pick_font(fs(lay["font_title_pt"]), True))
        painter.drawText(QRectF(x + pad, yy, lay["w"], lay["title_h"]),
                          Qt.AlignLeft | Qt.AlignVCenter, lay["title"])
        yy += lay["title_h"] + lay["row_gap"]
    painter.setFont(pick_font(fs(lay["font_body_pt"])))
    for ln in lay["lines"]:
        painter.drawText(QRectF(x + pad, yy, lay["w"], lay["body_row_h"]),
                          Qt.AlignLeft | Qt.AlignVCenter, ln)
        yy += lay["body_row_h"]


def draw_text_block_panel(painter, x, y, lay, fs, text_color=(90, 100, 110), dot_color=None):
    k = fs.k
    painter.setFont(pick_font(fs(lay["font_pt"])))
    pad, dot, gap, row_h = lay["pad"], lay["dot"], lay["gap"], lay["row_h"]
    text_x = x + pad + dot + gap
    if dot_color is not None:
        r = dot / 2
        cy = y + pad + row_h / 2
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(*dot_color), max(1.6, k * 1.4)))
        painter.drawEllipse(QRectF(x + pad, cy - r, dot, dot))
    painter.setPen(QColor(*text_color))
    for i, ln in enumerate(lay["lines"]):
        painter.drawText(QRectF(text_x, y + pad * 0.5 + i * row_h, lay["w"], row_h),
                          Qt.AlignVCenter | Qt.AlignLeft, ln)


def count_highlight_hits(layer, id_field, id_set):
    """Same "does this id match" test as draw_highlights, but without drawing and
    without QgsMapSettings. Panel mode needs to know whether a highlight has any hits
    before the final extent/map size is known, so its caption can be included in the
    panel size; draw_highlights depends on pixel transforms and cannot run that early.
    If the matching logic changes, change both functions together, otherwise the panel
    may reserve a caption for rings that are never drawn."""
    if not id_set:
        return 0
    hit = 0
    for f in layer.getFeatures():
        val = str(f[id_field]) if id_field in f.fields().names() else None
        if val in id_set:
            hit += 1
    return hit


def draw_offmap_marker(painter, ms, pt_prj, label, dist_m, fs, canvas_w, canvas_h):
    """When a layer has no features in view, mark the nearest feature at its true
    bearing with its distance, instead of only reporting "not found"."""
    sp = ms.mapToPixel().transform(pt_prj.x(), pt_prj.y())
    x, y = sp.x(), sp.y()
    k = fs.k
    r = 9 * k
    painter.setBrush(QColor(255, 140, 0))
    painter.setPen(QPen(QColor(255, 255, 255), max(2, 1.6 * k)))
    painter.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
    painter.setPen(QPen(QColor(255, 140, 0), max(1.4, 1.1 * k)))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QRectF(x - r * 1.7, y - r * 1.7, 2 * r * 1.7, 2 * r * 1.7))

    text = f"{label}·{dist_m / 1000:.1f}km"
    painter.setFont(pick_font(fs(8), True))
    fm = painter.fontMetrics()
    pad = 6 * k
    tw2 = fm.horizontalAdvance(text) + pad * 2
    th2 = fm.height() + pad
    gap = r * 1.7 + 4 * k
    bx = x + gap if (x + gap + tw2 + 10 * k) <= canvas_w else x - gap - tw2
    by = min(max(y - th2 / 2, 10 * k), canvas_h - th2 - 10 * k)
    painter.setBrush(QColor(255, 140, 0, 235))
    painter.setPen(QPen(QColor(255, 255, 255), max(1, k)))
    painter.drawRoundedRect(QRectF(bx, by, tw2, th2), 3 * k, 3 * k)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(QRectF(bx, by, tw2, th2), Qt.AlignCenter, text)


def draw_highlights(painter, ms, project, layer, id_field, id_set, fs, ring_color=(0, 220, 255), r_scale=1.0,
                     tag=None):
    """Circle the features with the given ids on the map - ids listed only in text
    cannot be found by the reader.

    By default the small tag next to each ring shows the value of id_field itself,
    which works when the id is human-readable (e.g. a point name). When matching on
    an opaque key such as a UUID primary key, that tag would be unreadable; the
    optional `tag` replaces the tag text of every matched feature with the given
    human-readable string. tag=None (default) keeps the original behaviour."""
    if not id_set:
        return 0
    k = fs.k
    m2p = ms.mapToPixel()
    to_prj = QgsCoordinateTransform(layer.crs(), project.crs(), project)
    hit = 0
    for f in layer.getFeatures():
        val = str(f[id_field]) if id_field in f.fields().names() else None
        if val not in id_set:
            continue
        tag_text = tag if tag is not None else val
        g = QgsGeometry(f.geometry())
        if g.isEmpty() or g.transform(to_prj) != 0:
            continue
        p = g.asPoint() if g.type() == 0 else g.centroid().asPoint()
        sp = m2p.transform(p.x(), p.y())
        x, y, r = sp.x(), sp.y(), 15 * k * r_scale
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 255, 255), max(4, 3.2 * k)))
        painter.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))
        painter.setPen(QPen(QColor(*ring_color), max(2.4, 2 * k)))
        painter.drawEllipse(QRectF(x - r, y - r, 2 * r, 2 * r))

        painter.setFont(pick_font(fs(7), True))
        fm = painter.fontMetrics()
        tw2, th2 = fm.horizontalAdvance(tag_text) + 6 * k, fm.height() + 2 * k
        bx, by = x + r * 0.75, y - r - th2 * 0.8
        painter.setBrush(QColor(20, 20, 20, 235))
        painter.setPen(QPen(QColor(*ring_color), max(1.2, k)))
        painter.drawRoundedRect(QRectF(bx, by, tw2, th2), 2 * k, 2 * k)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(bx, by, tw2, th2), Qt.AlignCenter, tag_text)
        hit += 1
    return hit


def nearest_feature_info(layer, ref_point_prj, project, cap_m):
    """When a layer has no features inside the default view, find the feature nearest
    to the reference point, so the view can be expanded to pull it in."""
    if not hasattr(layer, "getFeatures") or not cap_m:
        return None
    try:
        to_layer = QgsCoordinateTransform(project.crs(), layer.crs(), project)
        to_prj = QgsCoordinateTransform(layer.crs(), project.crs(), project)
        rp_layer = to_layer.transform(ref_point_prj)
        idx = QgsSpatialIndex(layer.getFeatures())
        nearest_ids = idx.nearestNeighbor(rp_layer, 1)
        if not nearest_ids:
            return None
        feat = next(layer.getFeatures(QgsFeatureRequest().setFilterFid(nearest_ids[0])), None)
        if feat is None:
            return None
        g = QgsGeometry(feat.geometry())
        if g.isEmpty():
            return None
        nearest_pt = g.nearestPoint(QgsGeometry.fromPointXY(rp_layer))
        if nearest_pt.transform(to_prj) != 0:
            return None
        pt_prj = nearest_pt.asPoint()
        da = make_distance_area(project)
        d = da.measureLine(ref_point_prj, pt_prj)
        if d > cap_m:
            return None
        return {"distance_m": d, "point_prj": pt_prj}
    except Exception:
        return None


# ──────────────────── Temporary symbol scaling / label toggles (always restored after rendering) ────────────────────
def tweak_layers(layers, scale=None, labels=None, label_fs=None, label_font_pt=8, force_labels=None,
                  label_show_ids=None, force_label_font_pt=6, label_offset_ctx=None):
    """Temporarily adjust layers for one view; returns the state for restore_layers().

    label_offset_ctx: (map units per mm, map CRS authid), used to compute a
    PositionPoint expression for layers configured with label_offsets (see
    label_offset_position_expr()).
    label_show_ids: {QgsVectorLayer: [$id, ...]} - only these feature ids are labelled
    on that layer, all others hidden (used with top_k_ids_by_overlap_area(), which
    picks the K most prominent features for the current view extent). Implemented via
    the QgsPalLayerSettings.Show data-defined property, the same mechanism as the
    hide_expr parameter of apply_labels(); here the expression is computed at render
    time because it depends on the view extent, which is unknown when the layer pool
    is loaded.
    force_labels: a set of QgsVectorLayer objects (not keys) whose label visibility is
    not controlled by the `labels` argument (the setLabelsEnabled branch is skipped and
    the labelsEnabled state set at load time, always True, is kept). Used for
    reference place-name labels that must show regardless of the view's labels
    on/off/auto setting. Note: font size is NOT exempt - these layers are still
    rescaled, using force_label_font_pt instead of label_font_pt (see below). Empty
    (default) means no change in behaviour.
    force_label_font_pt: target font size for force_labels layers. It goes through the
    same label_fs() physical-size conversion as the main labels (not a raw point
    size), and defaults to one step smaller than label_font_pt, so both sizes live in
    the same physical scale and background reference labels stay visually lighter
    than the labels of the features being analysed.
    label_fs: the view's fs() scaling function. Vector labels on the map (e.g. point
    ids) are temporarily set to fs(label_font_pt), the same call used for legend body
    text, so the sizes match. apply_labels() runs when the layer pool is loaded (before
    any view's fs is known) and writes a raw QgsTextFormat.setSize(8); that raw 8 pt
    can be less than half of the physically scaled fs(8) on a large canvas, making map
    labels much smaller than legend text. Like symbol_scale, this change is only for
    the current render and is restored afterwards, so the layer pool is not modified
    for the next view."""
    saved = []
    for lyr in layers:
        if not hasattr(lyr, "renderer"):
            continue
        r = lyr.renderer()
        if r is None:
            continue
        labeling = lyr.labeling().clone() if hasattr(lyr, "labeling") and lyr.labeling() is not None else None
        saved.append((lyr, r.clone(), lyr.labelsEnabled() if hasattr(lyr, "labelsEnabled") else None, labeling))
        if scale and scale != 1.0:
            work = r.clone()
            for sym in symbols_of(work):
                try:
                    if sym.type() == 0:
                        sym.setSize(sym.size() * scale)
                    elif sym.type() == 1:
                        sym.setWidth(max(sym.width() * scale, 0.3))
                except Exception:
                    pass
            lyr.setRenderer(work)
        is_forced = force_labels and lyr in force_labels
        if labels is not None and not is_forced and hasattr(lyr, "setLabelsEnabled") and lyr.labeling() is not None:
            lyr.setLabelsEnabled(bool(labels))
        if label_fs is not None and labeling is not None:
            new_labeling = labeling.clone()
            settings = new_labeling.settings()
            tf = settings.format()
            # force_labels layers (reference place names) use their own
            # force_label_font_pt rather than sharing label_font_pt with the main
            # labels: at equal size readers cannot tell background names from the
            # labels of the analysed features, and cartographic hierarchy says
            # background reference information should be lighter. Both go through the
            # same label_fs() conversion, only the target pt differs, so their ratio
            # stays stable.
            target_pt = force_label_font_pt if is_forced else label_font_pt
            if lyr.customProperty("gc_label_pt") is not None:
                target_pt = float(lyr.customProperty("gc_label_pt"))  # isolines carry their own label size, see isoline_label_style
            # label_font_pt is adjustable (default 8 matches legend body text). Where
            # points are densely clustered, the PAL labelling engine may find no
            # non-overlapping candidate at that size and labels overprint each other.
            # A smaller size shrinks the candidate boxes and gives PAL more room - a
            # cheap mitigation, not a guarantee (if points are closer than any label
            # box, there is no overlap-free solution).
            tf.setSize(label_fs(target_pt))
            settings.setFormat(tf)
            show_ids = (label_show_ids or {}).get(lyr)
            if show_ids is not None:
                expr = "$id IN (" + ",".join(str(i) for i in show_ids) + ")" if show_ids else "FALSE"
                settings.dataDefinedProperties().setProperty(
                    QgsPalLayerSettings.Show, QgsProperty.fromExpression(expr))
            if label_offset_ctx is not None:
                pos_expr = label_offset_position_expr(lyr, *label_offset_ctx)
                if pos_expr:
                    settings.dataDefinedProperties().setProperty(
                        QgsPalLayerSettings.PositionPoint, QgsProperty.fromExpression(pos_expr))
            new_labeling.setSettings(settings)
            lyr.setLabeling(new_labeling)
        if lyr.customProperty("gc_label_obstacle", False):
            apply_obstacle_for_view(lyr)  # must run after the labels on/off state above is final, see its docstring
    return saved


def restore_layers(saved):
    for lyr, renderer, labels, labeling in saved:
        lyr.setRenderer(renderer)
        if labels is not None and hasattr(lyr, "setLabelsEnabled"):
            lyr.setLabelsEnabled(labels)
        if labeling is not None and hasattr(lyr, "setLabeling"):
            lyr.setLabeling(labeling)


# ──────────────────────────── Map extent ────────────────────────────
def fix_aspect(e, tw, th):
    if e.height() > 0 and (e.width() / e.height()) < (tw / th):
        e.setXMinimum(e.center().x() - e.height() * tw / th / 2)
        e.setXMaximum(e.center().x() + e.height() * tw / th / 2)
    else:
        e.setYMinimum(e.center().y() - e.width() * th / tw / 2)
        e.setYMaximum(e.center().y() + e.width() * th / tw / 2)
    return e


def grow_by_km(ext, km, project):
    if km <= 0:
        return
    if project.crs().isGeographic():
        lat = ext.center().y()
        deg_per_km = 1.0 / (111.32 * max(math.cos(math.radians(lat)), 0.1))
        ext.grow(km * deg_per_km)
    else:
        ext.grow(km * 1000.0)  # assumes projected units are metres (true for UTM/3857 and most engineering CRSs)


def compute_raw_extent(view, defaults, layers_by_key, project):
    """Compute the view extent without the canvas aspect-ratio correction
    (`fix_aspect`). Reused by `compute_extent`, and used on its own by
    `choose_auto_panel_side` to judge whether the real geographic content is tall and
    narrow or wide. `fix_aspect` stretches the extent to the canvas aspect ratio, after
    which the true shape of the content can no longer be measured, so that judgement
    needs this "raw" version."""
    ext = None
    ext_keys = view.get("extent_layers", defaults.get("extent_layers"))
    if not ext_keys:
        ext_keys = [k for k in view["layers"]
                    if k in layers_by_key and hasattr(layers_by_key[k], "geometryType")]
    for k in ext_keys:
        lyr = layers_by_key.get(k)
        if lyr is None or not hasattr(lyr, "extent"):
            continue
        raw = lyr.extent()
        if raw.isEmpty() and hasattr(lyr, "dataProvider"):
            # With some QGIS/GDAL versions the first in-process open of a GeoPackage
            # table can return an empty extent ("unable to open database file"; a
            # second open works). Without this, a layer would silently yield no extent
            # unless it was pre-opened elsewhere, so reload the provider once and
            # measure again.
            lyr.dataProvider().reloadData()
            lyr.updateExtents()
            raw = lyr.extent()
        if raw.isEmpty():
            print(f"⚠ [{view.get('out')}] extent layer {k!r} has an empty extent (no features after filtering, or the data source failed to open)", file=sys.stderr)
            continue
        e = QgsCoordinateTransform(lyr.crs(), project.crs(), project).transformBoundingBox(raw)
        ext = QgsRectangle(e) if ext is None else (ext.combineExtentWith(e) or ext)
    if ext is None or ext.isEmpty():
        return None
    margin = float(view.get("margin", defaults.get("margin", 0.15)))
    ext.scale(1.0 + margin)
    pad_km = float(view.get("extent_pad_km", 0) or 0)
    if pad_km:
        grow_by_km(ext, pad_km, project)
    return ext


def compute_extent(view, defaults, layers_by_key, project, tw, th):
    ext = compute_raw_extent(view, defaults, layers_by_key, project)
    if ext is None:
        return None
    fix_aspect(ext, tw, th)
    return ext


# ──────────────────────────── Decoration panel (panel-bottom / panel-right) ────────────────────────────
PANEL_AUTO_ASPECT_THRESHOLD = 0.6  # (extent aspect / canvas aspect) below this counts as "clearly narrow" -> right panel;
                                    # an initial value, meant to be tuned against real map sets


def expand_preset(target):
    """Expand the `preset` field of target (`defaults` or a `view`) into concrete
    size/physical_height_cm/decoration_layout defaults. Uses `setdefault`, so only
    fields not written explicitly in target are filled; explicit fields always win -
    a preset is a set of convenient defaults, not a lock. Presets for report/slide
    output are defined in assets/presets.json."""
    name = target.get("preset")
    if not name:
        return
    spec = PRESETS.get(name)
    if not spec:
        available = [k for k in PRESETS if not k.startswith("_")]
        raise ValueError(f"Unknown preset: {name} (available: {available})")
    for key, val in spec.items():
        if key.startswith("_"):
            continue
        target.setdefault(key, val)


def choose_auto_panel_side(view, defaults, layers_by_key, project, tw, th):
    """Rule for `decoration_layout: panel-auto`: if the geographic content is clearly
    narrower than the canvas (e.g. a long, narrow country, where `fix_aspect` would
    pad large empty margins on both sides), put the panel on the right - it uses
    horizontal space that would be wasted anyway, without shrinking the map.
    Otherwise (content roughly square or round, already filling the canvas) put it at
    the bottom."""
    ext = compute_raw_extent(view, defaults, layers_by_key, project)
    if ext is None or ext.height() <= 0 or th <= 0:
        return "bottom"
    ext_aspect = ext.width() / ext.height()
    canvas_aspect = tw / th
    ratio = ext_aspect / canvas_aspect if canvas_aspect > 0 else 1.0
    return "right" if ratio < PANEL_AUTO_ASPECT_THRESHOLD else "bottom"


def nice_ticks(vmin, vmax, target_n=6):
    """A series of "round" tick values in [vmin, vmax] (steps of 1/2/2.5/5 x 10^n) for
    a multi-tick colour bar - one long colour ramp with several tick labels rather
    than only the two end values, as in Global Wind Atlas map exports. A simplified
    MaxNLocator-style approach; it does not try to match matplotlib exactly, only to
    be tidy and sufficient."""
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    raw_step = span / max(target_n, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = magnitude
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * magnitude
        if step >= raw_step:
            break
    start = math.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 1e-6:
        ticks.append(round(v, 3))
        v += step
    return ticks or [vmin, vmax]


def scalebar_panel_layout(fs, target_px=110, ratio=None, widen=True):
    """Reserved size for the panel-mode scale bar. A fixed value is used instead of a
    fraction of canvas width (as `draw_scalebar` does) because the panel thickness
    must be fixed before `meters_per_px` is known (it depends on the final extent) -
    a chicken-and-egg dependency. The fixed reserved width breaks the cycle: at draw
    time the actual bar width is set by `nice_number` rounding and will not equal the
    reservation exactly, but it is of the same magnitude and does not overflow the
    panel (a small visual approximation is acceptable).

    Style: a single thin line with end ticks and text, the minimal scale bar common on
    professional maps (as in Global Wind Atlas exports), not an alternating
    black/white bar on a card."""
    k = fs.k
    lay = {"target_px": target_px * k, "tick_h": 6 * k, "label_h": 14 * k,
           "w": target_px * k, "h": 6 * k + 14 * k + 4 * k}
    if ratio:
        # ratio=(fmt, px_per_cm, h_cm): the ratio number is only known once the final
        # extent is fixed, so one line is reserved using the widest placeholder number -
        # the same "fixed reservation breaks the cycle" idea as above.
        fmt, px_per_cm, h_cm = ratio
        fm = QFontMetrics(pick_font(fs(8)))
        placeholder = fmt.format(denom="00,000,000", h_cm=h_cm)  # fits an 8-digit denominator (national scale, e.g. 1:16,500,000)
        # widen=False (panel-right unified card): the ratio text does not contribute
        # to the card width; otherwise it would widen the whole card and squeeze the
        # map (and could truncate the title). Once the card width is fixed,
        # build_unified_panel fills in ratio_max_w; if the text does not fit at draw
        # time it falls back to plain "1:xx,000".
        lay.update(ratio=ratio, h=lay["h"] + lay["label_h"], ratio_max_w=None)
        if widen:
            lay["w"] = max(lay["w"], fm.horizontalAdvance(placeholder))
    return lay


def draw_scalebar_panel(painter, x0, y0, lay, meters_per_px, fs, ink=(230, 230, 230)):
    k = fs.k
    bar_m = nice_number_floor(lay["target_px"] * meters_per_px)
    bar_px = bar_m / meters_per_px if meters_per_px > 0 else lay["target_px"]
    bar_px = min(bar_px, lay["target_px"])  # belt and braces: never exceed the reservation even if rounding changes
    label = f"{bar_m/1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    tick_h = lay["tick_h"]
    bar_y = y0 + tick_h

    pen = QPen(QColor(*ink), max(1, k * 0.9))
    painter.setPen(pen)
    painter.drawLine(QPointF(x0, bar_y), QPointF(x0 + bar_px, bar_y))
    for xt in (x0, x0 + bar_px / 2, x0 + bar_px):
        painter.drawLine(QPointF(xt, bar_y - tick_h), QPointF(xt, bar_y + tick_h * 0.4))

    painter.setFont(pick_font(fs(8)))
    painter.setPen(QColor(*ink))
    painter.drawText(QRectF(x0, bar_y + tick_h * 0.6, bar_px, lay["label_h"]),
                      Qt.AlignLeft | Qt.AlignTop, "0")
    painter.drawText(QRectF(x0, bar_y + tick_h * 0.6, bar_px, lay["label_h"]),
                      Qt.AlignRight | Qt.AlignTop, label)
    if lay.get("ratio") and meters_per_px > 0:
        fmt, px_per_cm, h_cm = lay["ratio"]
        text = scale_ratio_text(meters_per_px, px_per_cm, h_cm, fmt)
        room = lay.get("ratio_max_w") or lay["w"]
        if painter.fontMetrics().horizontalAdvance(text) > room:
            text = scale_ratio_text(meters_per_px, px_per_cm, h_cm, "1:{denom}")
        painter.drawText(QRectF(x0, bar_y + tick_h * 0.6 + lay["label_h"], max(room, lay["w"]), lay["label_h"]),
                          Qt.AlignLeft | Qt.AlignTop | Qt.TextDontClip, text)


def north_panel_layout(fs):
    k = fs.k
    return {"w": 24 * k, "h": 38 * k}


def draw_north_panel(painter, x0, y0, lay, fs, color=(230, 230, 230)):
    """North arrow for the decoration panel.

    Preferably uses a north-arrow SVG from QGIS's bundled SVG library (located by
    north_arrow_svg()). These SVGs are parameterised (`param(fill)` /
    `param(outline)` placeholders), so `QgsApplication.svgCache().svgContent()` can
    recolour them to any colour and `QSvgRenderer` can draw the resulting bytes
    directly - no need to move the rendering pipeline to the Print Layout API.

    If the SVG cannot be found or is invalid, a hand-drawn arrow (filled triangle
    with a notched tail plus an "N" underneath, the same geometry as draw_north) is
    drawn instead, scaled to the panel slot and in the given colour."""
    w, h = lay["w"], lay["h"]
    c = QColor(*color)
    svg_path = north_arrow_svg()
    if svg_path:
        content = QgsApplication.svgCache().svgContent(
            svg_path, 64.0, c, c, 0.3, 1.0, 0, True, {})
        svg = QSvgRenderer(content)
        if svg.isValid():
            native = svg.defaultSize()
            scale = min(w / native.width(), h / native.height())
            dw, dh = native.width() * scale, native.height() * scale
            svg.render(painter, QRectF(x0 + (w - dw) / 2, y0 + (h - dh) / 2, dw, dh))
            return

    # Fallback: hand-drawn arrow in the top ~60% of the slot, "N" below it.
    cx = x0 + w / 2
    arrow_h = h * 0.6
    tip_y, tail_y = y0, y0 + arrow_h
    half = min(w / 2, arrow_h * 0.30 / 0.84)
    notch = arrow_h * 0.14 / 0.84
    painter.setBrush(c)
    painter.setPen(Qt.NoPen)
    painter.drawPolygon(QPolygonF([
        QPointF(cx, tip_y), QPointF(cx - half, tail_y),
        QPointF(cx, tail_y - notch), QPointF(cx + half, tail_y),
    ]))
    painter.setBrush(Qt.NoBrush)
    painter.setPen(c)
    painter.setFont(pick_font(fs(8), True))
    painter.drawText(QRectF(x0, tail_y, w, h - arrow_h), Qt.AlignCenter, "N")


def colorbar_panel_layout(painter, unit, fs, compact=False):
    """Colour bar for the panel: one long colour ramp with a title and several tick
    labels (as in Global Wind Atlas exports), not a short bar with only its two end
    values on a card. `compact` uses a shorter length for a narrow panel-right column,
    so the sidebar does not squeeze the map; panel-bottom has plenty of horizontal
    room and uses the longer length."""
    k = fs.k
    length = (150 if compact else 300) * k
    thickness = 12 * k
    title_h = 16 * k if unit else 0
    tick_h = 6 * k
    label_h = 14 * k
    return {"length": length, "thickness": thickness, "title_h": title_h,
            "tick_h": tick_h, "label_h": label_h,
            "w": length, "h": title_h + thickness + tick_h + label_h}


def draw_colorbar_panel(painter, x0, y0, lay, colors, vmin, vmax, unit, fs, ink=(225, 225, 225)):
    k = fs.k
    length, thickness = lay["length"], lay["thickness"]
    n = len(colors)

    if unit:
        painter.setFont(pick_font(fs(8), True))
        painter.setPen(QColor(*ink))
        painter.drawText(QRectF(x0, y0, length, lay["title_h"]), Qt.AlignHCenter | Qt.AlignVCenter, unit)

    bar_y = y0 + lay["title_h"]
    grad = QLinearGradient(x0, 0, x0 + length, 0)
    for i, c in enumerate(colors):
        grad.setColorAt(i / (n - 1), QColor(c))
    painter.setPen(Qt.NoPen)
    painter.setBrush(grad)
    painter.drawRect(QRectF(x0, bar_y, length, thickness))
    painter.setPen(QPen(QColor(60, 60, 60), max(1, 0.6 * k)))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(QRectF(x0, bar_y, length, thickness))

    ticks = nice_ticks(vmin, vmax, target_n=6 if length > 200 * k else 4)
    tick_y = bar_y + thickness
    painter.setPen(QPen(QColor(*ink), max(1, k * 0.8)))
    painter.setFont(pick_font(fs(7.5)))
    for v in ticks:
        frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
        xt = x0 + frac * length
        painter.drawLine(QPointF(xt, tick_y), QPointF(xt, tick_y + lay["tick_h"]))
        painter.setPen(QColor(*ink))
        painter.drawText(QRectF(xt - 20 * k, tick_y + lay["tick_h"], 40 * k, lay["label_h"]),
                          Qt.AlignHCenter | Qt.AlignTop, f"{v:g}")
        painter.setPen(QPen(QColor(230, 230, 230), max(1, k * 0.8)))


def layout_panel_items(items, orientation, panel_rect, fs, gap_pt=14, view_name=None):
    """panel-bottom (horizontal) only: items are laid out left to right across the
    canvas width, centred on the short (vertical) axis. panel-right (vertical) uses
    build_unified_panel / draw_unified_panel below - a single panel with internal
    sections and dividers instead of separately framed elements - so only the
    horizontal branch is kept here."""
    k = fs.k
    gap = gap_pt * k
    pad = 16 * k
    total = sum(it["layout"]["w"] for it in items) + gap * max(0, len(items) - 1)
    x = panel_rect.x() + max(pad, (panel_rect.width() - total) / 2)
    placed = []
    for it in items:
        h = it["layout"]["h"]
        y = panel_rect.y() + (panel_rect.height() - h) / 2
        placed.append((it, x, y))
        x += it["layout"]["w"] + gap
    return placed


UNIFIED_PANEL_PAD_K = 18  # inner padding of the build_unified_panel card (x fs.k);
# build_panel_items must subtract it up front when computing each item's wrap width, see below.


def _unified_from_items(panel_items, fs):
    """Sizing only: split panel_items into build_unified_panel arguments using the same rule as render_view."""
    top = [(it["kind"], it["layout"], None) for it in panel_items if it["kind"] in ("north", "scalebar")]
    legend = next((it["layout"] for it in panel_items if it["kind"] == "legend"), None)
    cb = next((it["layout"] for it in panel_items if it["kind"] == "colorbar"), None)
    notes = [(it["kind"], it["layout"], it.get("color")) for it in panel_items
             if it["kind"] in ("locator", "hl_note", "absent_note", "credit", "info_card")]
    return build_unified_panel(top, legend, cb, notes, fs)


def build_unified_panel(top_pieces, legend_piece, colorbar_piece, note_pieces, fs):
    """Lay out the decoration panel as one unit: north arrow + scale bar in one row,
    legend in one section, notes in another, separated by thin horizontal dividers,
    instead of every element drawing its own frame.

    Sections: north + scalebar (one row, centred horizontally, as is conventional for
    icon-like decorations), legend + colorbar (legend section), and
    hl_note / absent_note / credit / locator / info_card (notes section). Contents
    within a section are left-aligned.

    Returns {"w", "h", "pad", "positions": [(kind, rel_x, rel_y, layout, extra), ...],
    "dividers": [rel_y, ...]}; positions and dividers are relative to the card's
    top-left corner, and draw_unified_panel adds the absolute x0, y0.

    The card is never compressed to a max_w here. Legend/note wrap widths are derived
    from a max_w, but the card adds its own padding (UNIFIED_PANEL_PAD_K) around the
    content; compressing the card to max_w afterwards would only shrink the border,
    not the text, and the text would overflow the right edge. Instead the caller
    (build_panel_items) subtracts the padding from max_w before wrapping, so
    "content width + 2 * pad" can never exceed the original max_w."""
    k = fs.k
    pad = UNIFIED_PANEL_PAD_K * k
    row_gap = 10 * k       # gap between items within a section
    section_gap = 13 * k   # space between section content and a divider (symmetric on both sides)

    row1_w = sum(p[1]["w"] for p in top_pieces) + row_gap * max(0, len(top_pieces) - 1)
    row1_h = max((p[1]["h"] for p in top_pieces), default=0)
    widths = [row1_w]
    if legend_piece is not None:
        widths.append(legend_piece["w"])
    if colorbar_piece is not None:
        widths.append(colorbar_piece["w"])
    for _, lay, _ in note_pieces:
        widths.append(lay["w"])
    content_w = max(widths, default=0) + pad * 2
    content_w = max(content_w, pad * 2 + 60 * k)  # minimum width, so an empty card does not collapse to a line

    positions, dividers = [], []
    y = pad
    if top_pieces:
        rx = (content_w - row1_w) / 2  # the whole row is centred horizontally, as is conventional for icon-like decorations
        for kind, lay, extra in top_pieces:
            py = y + (row1_h - lay["h"]) / 2
            positions.append((kind, rx, py, lay, extra))
            if kind == "scalebar" and lay.get("ratio"):
                lay["ratio_max_w"] = content_w - pad * 0.5 - rx  # ratio text may extend up to the card's right inner padding
            rx += lay["w"] + row_gap
        y += row1_h

    have_content = legend_piece is not None or colorbar_piece is not None
    if top_pieces and (have_content or note_pieces):
        y += section_gap
        dividers.append(y)
        y += section_gap

    if legend_piece is not None:
        positions.append(("legend", pad, y, legend_piece, None))
        y += legend_piece["h"]
    if colorbar_piece is not None:
        if legend_piece is not None:
            y += row_gap
        positions.append(("colorbar", pad, y, colorbar_piece, None))
        y += colorbar_piece["h"]

    if have_content and note_pieces:
        y += section_gap
        dividers.append(y)
        y += section_gap

    for i, (kind, lay, extra) in enumerate(note_pieces):
        if i > 0:
            y += row_gap
        positions.append((kind, pad, y, lay, extra))
        y += lay["h"]

    y += pad
    return {"w": content_w, "h": y, "pad": pad, "positions": positions, "dividers": dividers}


def draw_unified_panel(painter, x0, y0, layout, fs, ink=(60, 60, 60)):
    """Draw what build_unified_panel measured: the internal dividers, then each item
    via its own draw_*_panel function (the legend with draw_bg=False, so it does not
    draw its own card).

    There is no card background or outline: content is drawn directly on the panel
    background (panel_bg), and structure is conveyed by the neatline and the internal
    dividers (a card outline would double up with the neatline). Because there is no
    white backing to guarantee contrast, `ink` is the adaptive colour from
    panel_ink_color(), so north arrow / scale bar / colour bar / notes / credit remain
    readable on light and dark panel backgrounds. Legend text colour is not affected
    by `ink`: it keeps its dark text on small light-grey per-row swatch chips, which
    have no readability problem."""
    k = fs.k
    w, pad = layout["w"], layout["pad"]
    # Dividers are one step lighter than the text ink - in full ink they would be too
    # heavy and compete with the text; they only need to hint at sections.
    divider_color = (210, 210, 210) if ink == (60, 60, 60) else (95, 95, 95)
    painter.setPen(QPen(QColor(*divider_color), max(1, k * 0.5), Qt.SolidLine))
    for dy in layout["dividers"]:
        painter.drawLine(QPointF(x0 + pad * 0.6, y0 + dy), QPointF(x0 + w - pad * 0.6, y0 + dy))

    for kind, rx, ry, lay, extra in layout["positions"]:
        px, py = x0 + rx, y0 + ry
        if kind == "north":
            draw_north_panel(painter, px, py, lay, fs, color=ink)
        elif kind == "scalebar":
            draw_scalebar_panel(painter, px, py, lay, extra, fs, ink=ink)
        elif kind == "legend":
            draw_legend_at(painter, px, py, lay, fs, draw_bg=False)
        elif kind == "colorbar":
            draw_colorbar_panel(painter, px, py, lay, extra["colors"], extra["min"], extra["max"],
                                 extra["unit"], fs, ink=ink)
        elif kind == "hl_note":
            draw_text_block_panel(painter, px, py, lay, fs, text_color=ink, dot_color=extra)
        elif kind in ("absent_note", "credit"):
            draw_text_block_panel(painter, px, py, lay, fs, text_color=ink)
        elif kind == "info_card":
            draw_info_card_panel(painter, px, py, lay, fs, text_color=ink)
        elif kind == "locator":
            draw_locator(painter, px, py, lay, extra[0], extra[1], fs)


# ──────────────────────────── Overlay decorations: never cover the main subject ────────────────────────────
# In overlay mode, floating decorations must not cover the main subject being shown.
# Pixel "busyness" alone (region_busyness) does not know which content is the subject,
# so when the subject fills the frame a legend could still be placed over it. Here the
# decision is geometric: subject = protect_layers (default = extent_layers), and each
# floating decoration is placed in a candidate corner that does not intersect the
# subject. If not all fit, the caller expands the view by 10% around its centre and
# retries (up to 10 times, ~2.6x), trading a wider margin for an unobstructed subject.
# If it still does not fit, a warning suggests a panel layout - decorations are never
# silently placed over the subject.

def overlay_protect_geometry(view, defaults, layers_by_key, project, ext):
    keys = view.get("protect_layers", defaults.get("protect_layers"))
    if keys is False:
        return None
    if not keys:
        keys = view.get("extent_layers", defaults.get("extent_layers"))
    if not keys:
        return None
    geoms = []
    for k in keys:
        lyr = layers_by_key.get(k)
        if lyr is None or not hasattr(lyr, "getFeatures"):
            continue
        tr = QgsCoordinateTransform(lyr.crs(), project.crs(), project)
        req = QgsFeatureRequest().setFilterRect(tr.transformBoundingBox(ext, Qgis.TransformDirection.Reverse))
        part = []
        for f in lyr.getFeatures(req):
            g = QgsGeometry(f.geometry())
            if g.isEmpty():
                continue
            g.transform(tr)
            part.append(g)
        if not part:
            continue
        u = QgsGeometry.unaryUnion(part)
        if u.type() != QgsWkbTypes.PolygonGeometry:
            # For point/line subjects (e.g. turbine positions) the protected area is the
            # region they enclose, not the individual points - a legend placed between
            # two rows of points still covers the subject.
            u = u.convexHull()
        geoms.append(u)
    return QgsGeometry.unaryUnion(geoms) if geoms else None


def plan_overlay_decorations(ext, map_w, map_h, fs, protect, sizes):
    """Plan floating decoration positions in map-image pixel coordinates. Returns
    {"credit", "legend", "locator": corner name}, or None if the subject overlaps a
    fixed-position decoration or some decoration finds no free corner. Decorations
    missing from `sizes` are not placed."""
    k = fs.k
    mupp = ext.width() / map_w
    pad = 4 * k

    def hits(x, y, w, h):
        r = QgsRectangle(ext.xMinimum() + (x - pad) * mupp, ext.yMaximum() - (y + h + pad) * mupp,
                         ext.xMinimum() + (x + w + pad) * mupp, ext.yMaximum() - (y - pad) * mupp)
        return protect.intersects(QgsGeometry.fromRect(r))

    def overlap(a, b):
        return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])

    fixed = []
    if sizes.get("north"):
        R = 19 * k
        fixed.append((map_w - 2 * R - 16 * k, 16 * k, 2 * R, 2 * R))
    if sizes.get("colorbar"):
        cw, ch = sizes["colorbar"]
        fixed.append((map_w - cw - 22 * k, 84 * k, cw, ch))
    if sizes.get("scalebar"):
        right = sizes["scalebar"]
        top = map_h - 30 * k - 9 * k - 6 * k - 18 * k
        fixed.append((22 * k, top, right - 22 * k, map_h - top - 30 * k))
    if any(hits(*r) for r in fixed):
        return None
    taken = list(fixed)
    plan = {}

    def choose(name, cands):
        for cn, rect in cands:
            if not hits(*rect) and not any(overlap(rect, t) for t in taken):
                plan[name] = cn
                taken.append(rect)
                return True
        return False

    if sizes.get("credit"):
        cw, ch = sizes["credit"]
        m = 6 * k
        if not choose("credit", [("bottom-right", (map_w - cw - m, map_h - ch - m, cw, ch)),
                                 ("bottom-left", (m, map_h - ch - m, cw, ch))]):
            return None
    if sizes.get("legend"):
        lw, lh = sizes["legend"]
        m, res = 22 * k, (55 * k if sizes.get("scalebar") else 0)
        if not choose("legend", [("top-left", (m, m, lw, lh)),
                                 ("bottom-left", (m, map_h - m - res - lh, lw, lh)),
                                 ("bottom-right", (map_w - m - lw, map_h - m - res - lh, lw, lh)),
                                 ("top-right", (map_w - m - lw, m, lw, lh))]):
            return None
    if sizes.get("locator"):
        lw, lh = sizes["locator"]
        m = 22 * k
        if not choose("locator", [("bottom-right", (map_w - m - lw, map_h - m - lh - sizes.get("credit", (0, 0))[1], lw, lh)),
                                  ("top-left", (m, m, lw, lh)),
                                  ("bottom-left", (m, map_h - m - lh - 60 * k, lw, lh)),
                                  ("top-right", (map_w - m - lw, m, lw, lh))]):
            return None
    return plan


# ──────────────────────────── Locator inset ────────────────────────────
# A small overview map showing where the main map sits. Common professional practice
# for detail maps: inset in a corner at roughly 1/5 of the main map, a pale grey
# regional basemap, and a dark box marking the main map extent; very small sites are
# shown as a dot instead. These defaults follow practitioner convention and can be
# changed in the `locator:` config.
# Config (view or defaults level, off by default):
#   locator: {layer: <key in layers, a polygon layer such as country/province boundaries>, width_frac: 0.22,
#             position: auto|top-left|top-right|bottom-left|bottom-right,
#             fill: "#E4E4E4", outline: "#A8A8A8", box_color: "#111111", site_color: "#C00000"}
# The inset basemap is a clone of `layer` restyled to a uniform pale grey; the original
# layer in the layer pool is not modified (another view may use it as a main layer).
LOCATOR_DEFAULTS = {"width_frac": 0.22, "position": "auto", "fill": "#E4E4E4", "outline": "#A8A8A8",
                    "box_color": "#111111", "site_color": "#C00000"}


def prepare_locator(view, defaults, layers_by_key, project):
    cfg = view.get("locator", defaults.get("locator"))
    if not cfg:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"locator must be a mapping containing at least 'layer' (got {cfg!r})")
    cfg = {**LOCATOR_DEFAULTS, **cfg}
    src = layers_by_key.get(cfg.get("layer"))
    if src is None or not hasattr(src, "geometryType"):
        print(f"⚠ [{view.get('out')}] locator.layer={cfg.get('layer')!r} is not in the layer pool or is not a vector layer; skipping the locator inset",
              file=sys.stderr)
        return None
    if src.geometryType() != QgsWkbTypes.PolygonGeometry:
        print(f"⚠ [{view.get('out')}] locator.layer={cfg.get('layer')!r} is not a polygon layer, so the inset basemap will be empty"
              f" - use a country/province boundary polygon layer instead", file=sys.stderr)
    lyr = src.clone()
    lyr.setSubsetString(src.subsetString())
    sym = QgsFillSymbol.createSimple({"color": cfg["fill"], "outline_color": cfg["outline"],
                                      "outline_width": "0.2", "outline_width_unit": "MM"})
    lyr.setRenderer(QgsSingleSymbolRenderer(sym))
    lyr.setLabelsEnabled(False)
    lyr.setOpacity(1.0)
    ext = QgsCoordinateTransform(lyr.crs(), project.crs(), project).transformBoundingBox(lyr.extent())
    if ext.isEmpty():
        return None
    ext.scale(1.06)
    return {"layer": lyr, "ext": ext, "cfg": cfg, "aspect": ext.height() / max(ext.width(), 1e-9)}


def locator_layout(loc, w_px, max_h_px=None):
    """Inset size: width is given, height follows the basemap extent's aspect ratio;
    if that exceeds max_h_px, the size is constrained by height instead."""
    w = float(w_px)
    h = w * loc["aspect"]
    if max_h_px and h > max_h_px:
        h = float(max_h_px)
        w = h / loc["aspect"]
    return {"w": w, "h": h, "loc": loc}


def draw_locator(painter, x0, y0, lay, main_ext, project, fs):
    loc, k = lay["loc"], fs.k
    w, h = max(8, int(lay["w"])), max(8, int(lay["h"]))
    ms = QgsMapSettings()
    ms.setLayers([loc["layer"]])
    ms.setDestinationCrs(project.crs())
    ms.setOutputSize(QSize(w, h))
    # The inset view must include the main map extent: after margin and aspect-ratio
    # correction the main extent often reaches beyond the basemap's bounds (e.g. over
    # the sea), and the extent box would otherwise be clipped at the inset edge.
    view_ext = QgsRectangle(loc["ext"])
    pad_ext = QgsRectangle(main_ext)
    pad_ext.scale(1.15)
    view_ext.combineExtentWith(pad_ext)
    ms.setExtent(fix_aspect(view_ext, w, h))
    ms.setBackgroundColor(QColor(255, 255, 255))
    img = QImage(QSize(w, h), QImage.Format_ARGB32)
    img.fill(QColor(255, 255, 255))
    p = QPainter(img)
    job = QgsMapRendererCustomPainterJob(ms, p)
    job.start()
    job.waitForFinished()
    # Main map extent: draw a box; if the box would be tiny (a small site on a
    # national map is only a few pixels), draw a dot instead - a box of a few pixels
    # is unreadable and looks like noise.
    m2p = ms.mapToPixel()
    a = m2p.transform(QgsPointXY(main_ext.xMinimum(), main_ext.yMaximum()))
    b = m2p.transform(QgsPointXY(main_ext.xMaximum(), main_ext.yMinimum()))
    rect = QRectF(QPointF(a.x(), a.y()), QPointF(b.x(), b.y())).normalized()
    p.setRenderHint(QPainter.Antialiasing, True)
    if rect.width() < 10 * k or rect.height() < 10 * k:
        r = 4.5 * k
        p.setPen(QPen(QColor(255, 255, 255), max(1, 1.2 * k)))
        p.setBrush(QColor(loc["cfg"]["site_color"]))
        p.drawEllipse(rect.center(), r, r)
    else:
        p.setPen(QPen(QColor(loc["cfg"]["box_color"]), max(1.5, 1.6 * k)))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect)
    p.end()
    painter.drawImage(QPointF(x0, y0), img)
    painter.setPen(QPen(QColor(150, 150, 150), max(1, 0.8 * k)))
    painter.setBrush(Qt.NoBrush)
    painter.drawRect(QRectF(x0, y0, w, h))


# ──────────────────────────── Single view rendering ────────────────────────────
def render_view(project, view, defaults, layers_by_key, layer_labels, raster_legend, outdir,
                 basemap_aliases=None, style_keys=None, force_label_keys=None, label_topk=None):
    key_list = view["layers"]
    present_keys, missing = [], []
    for k in key_list:
        if k in layers_by_key:
            present_keys.append(k)
        else:
            missing.append(k)
    if not present_keys:
        return {"out": view["out"], "status": "no_layers", "missing": missing}

    size = view.get("size", defaults.get("size", [1800, 1200]))
    tw, th = int(size[0]), int(size[1])
    dpi = view.get("dpi", defaults.get("dpi", 150))
    bg = view.get("background", defaults.get("background", "#FFFFFF"))

    # `dpi` only drives QGIS's physical scaling of symbols and fonts; it is not the
    # real pixel density of the figure. The real density is size (pixels) divided by
    # physical_height_cm (the height the figure will occupy on the page). Print
    # convention is 300 DPI (vector lines/text tolerate less than photos, so this is
    # not a hard threshold), but without a check a figure that will look blurry once
    # placed in a report is produced silently. Following the same "warn, don't
    # silently clip" approach as the panel-thickness check, warn below 150 PPI.
    physical_h_cm = view.get("physical_height_cm", defaults.get("physical_height_cm"))
    if physical_h_cm:
        effective_ppi = th * 2.54 / float(physical_h_cm)
        if effective_ppi < 150:
            print(f"⚠ [{view.get('out')}] effective output resolution is about {effective_ppi:.0f} PPI"
                  f" (size={tw}x{th}px, physical_height_cm={physical_h_cm}) - the figure will look"
                  f" blurry at its physical size in a report/slide; increase size or reduce physical_height_cm",
                  file=sys.stderr)

    # The scale ratio text ("1:45,000 (figure height 12 cm)") is only drawn when the
    # physical size is known - without physical_height_cm the printed size is unknown,
    # so the denominator cannot be computed. Better to omit it than to print a wrong one.
    locator = prepare_locator(view, defaults, layers_by_key, project)
    scale_ratio = None
    if physical_h_cm and view.get("scale_ratio", defaults.get("scale_ratio", True)):
        scale_ratio = (view.get("scale_ratio_label", defaults.get("scale_ratio_label", "1:{denom} (figure height {h_cm:g} cm)")),
                       th / float(physical_h_cm), float(f"{float(physical_h_cm):.3g}"))

    decoration_layout = view.get("decoration_layout", defaults.get("decoration_layout", "overlay"))
    if decoration_layout == "panel-auto":
        side = choose_auto_panel_side(view, defaults, layers_by_key, project, tw, th)
        decoration_layout = f"panel-{side}"
    is_panel = decoration_layout in ("panel-bottom", "panel-right")
    orientation = "horizontal" if decoration_layout == "panel-bottom" else "vertical"

    # Reserve space for the title bar BEFORE rendering. Painting an opaque title band
    # over the finished map hides whatever falls in the top strip of the extent
    # (turbine points, labels). Instead, the title bar height is subtracted from the
    # available height first and the geographic extent shrinks accordingly, so map
    # content physically cannot reach the title strip. Collision avoidance (as used for
    # the credit line) does not apply here: the title bar is pinned to the top edge and
    # has no alternative position, so the only fix is to make room.
    # title_h_px is estimated once with make_scaler(th, ...) rather than the reduced
    # height on purpose: title_h_px is typically under 5% of th, and iterating to a
    # fixed point changes font size / bar height by less than pixel rounding error.
    title = view.get("title")
    show_title = bool(title) and view.get("title_on_image", defaults.get("title_on_image", True))
    title_h_px = 0
    if show_title:
        _fs_probe = make_scaler(th, physical_h_cm or 12.5)
        _fm_probe = QFontMetrics(pick_font(_fs_probe(12), True))
        title_h_px = int(round(_fm_probe.height() + 10 * _fs_probe.k * 1.6))

    # ── extent and legend entries ─────────────────────────────────────────
    # Overlay mode (default) uses the final extent for legend filtering.
    # In panel mode the panel thickness must be fixed before rendering, so legend
    # filtering uses the "raw extent" (before padding) instead of the final extent.
    # The two differ only in the narrow edge case where aspect-ratio padding pulls in
    # one extra layer; the raw extent is more conservative and never under-sizes the panel.
    if not is_panel:
        ext = compute_extent(view, defaults, layers_by_key, project, tw, max(1, th - title_h_px))
        if ext is None:
            return {"out": view["out"], "status": "no_extent", "missing": missing}
        legend_ext = ext
    else:
        raw_ext = compute_raw_extent(view, defaults, layers_by_key, project)
        if raw_ext is None:
            return {"out": view["out"], "status": "no_extent", "missing": missing}
        legend_ext = raw_ext

    # restriction_* layers are grouped by exclusion_type (a field in report_style.json)
    # instead of being mixed into one flat list: each group gets its own sub-heading and
    # keeps the original z_order of present_keys within the group. Non-restriction
    # layers are unaffected.
    #
    # Three tiers:
    #   hard   - absolute exclusion, no siting possible.
    #   review - restricted area: siting requires a specific permit or study and may be
    #            height-limited or relocated, but is not prohibited outright. Example:
    #            aerodrome obstacle limitation surfaces (ICAO OLS) usually trigger an
    #            aeronautical study / written permission rather than a ban. The true hard
    #            constraint there is the 3-D penetration result for an individual turbine,
    #            not the whole area.
    #   soft   - buffer, setback recommended.
    # Keeping "review" separate avoids a legend that suggests "inside = not sitable" for
    # area-level constraints that only trigger additional approval. The exclusion_type
    # defaults are global and should be reviewed per project/country.
    EXCLUSION_GROUP_LABEL = {
        "hard": "Hard exclusion (no siting)",
        "review": "Restricted area (special approval required)",
        "soft": "Soft buffer (setback recommended)",
    }
    # Group headings can be overridden at defaults/view level (view wins, merged per key),
    # e.g. exclusion_group_labels: {review: "Approval required"} - useful when a long
    # heading wraps or widens the panel. Without configuration the defaults are used.
    _grp_over = {**(defaults.get("exclusion_group_labels") or {}), **(view.get("exclusion_group_labels") or {})}
    _bad = set(_grp_over) - set(EXCLUSION_GROUP_LABEL)
    if _bad:
        sys.exit(f"✗ [{view.get('out')}] exclusion_group_labels only accepts {list(EXCLUSION_GROUP_LABEL)}, unexpected {sorted(_bad)}")
    EXCLUSION_GROUP_LABEL = {**EXCLUSION_GROUP_LABEL, **_grp_over}
    style_keys = style_keys or {}
    legend_entries, absent_status = [], []
    restriction_groups = {}  # exclusion_type -> [rows...]; collected first, appended to legend_entries at the end
    for k in present_keys:
        label = layer_labels.get(k)
        lyr = layers_by_key[k]
        if not label or not hasattr(lyr, "getFeatures"):
            continue
        rows = legend_rows_for_layer(lyr, legend_ext, project, label)
        if not rows:
            absent_status.append((k, lyr, label))
            continue
        cat_spec = CATEGORIES.get(style_keys.get(k)) if style_keys.get(k) in CATEGORIES else None
        if cat_spec and cat_spec.get("legend_casing_color"):
            # legend_symbol() reuses the layer renderer's QgsSymbol for the preview, so
            # the legend swatch and the map share the same casing_color. A darker casing
            # makes the swatch readable on the legend card but costs contrast against
            # dark basemaps on the map. legend_casing_color decouples the two: build a
            # legend-only symbol preview with the darker casing, leaving the layer's
            # actual renderer (which still uses casing_color) untouched. build_symbol()
            # logic is identical; only the casing_color field in the spec differs.
            # This only applies to single-symbol (non-categorized) layers, where rows has
            # exactly one entry (e.g. road_paved); categorized/rule-based renderers have
            # no legend_casing_color categories. Passing empty counters is safe: this
            # path uses fixed:x colours, not auto:family, so the global palette cycling
            # state is neither consumed nor polluted.
            legend_spec = {**cat_spec, "casing_color": cat_spec["legend_casing_color"]}
            legend_pm = QgsSymbolLayerUtils.symbolPreviewPixmap(build_symbol(legend_spec, {}), QSize(22, 22))
            rows = [(legend_pm, lbl) for _, lbl in rows]
        elif cat_spec and any(cat_spec.get(f) is not None for f in LEGEND_FILL_FIELDS):
            rows = legend_fill_override_rows(k, cat_spec, rows)
        exclusion_type =cat_spec.get("exclusion_type") if cat_spec else None
        if exclusion_type in EXCLUSION_GROUP_LABEL:
            restriction_groups.setdefault(exclusion_type, []).extend(rows)
        else:
            legend_entries.extend(rows)
    for exclusion_type in ("hard", "review", "soft"):  # most important first
        rows = restriction_groups.get(exclusion_type)
        if rows:
            legend_entries.append((None, EXCLUSION_GROUP_LABEL[exclusion_type], True))
            legend_entries.extend(rows)

    # The credit line, the "not present in view extent" note and the highlight notes
    # are computed up front (extent/map_w/map_h are not final yet, and the pixel
    # positions of highlight rings cannot be known). In panel mode they become extra
    # panel items, laid out together with legend / north arrow / scale bar, so that no
    # decoration is left floating on the map while the rest lives in the panel.
    # Overlay mode is unaffected: there is no panel, so they stay in the map corners.
    attribution_texts = sorted({ATTRIBUTION[alias] for k, alias in (basemap_aliases or {}).items()
                                 if k in view.get("layers", []) and alias in ATTRIBUTION})
    credit_parts = list(attribution_texts)
    source_text = view.get("source", defaults.get("source"))
    if source_text:
        _lbl = view.get("source_label", defaults.get("source_label", "Data:"))
        credit_parts.append(f"{_lbl} {source_text}")
    _crs_id = project.crs().authid()
    _crs_human = crs_human_name(_crs_id)
    credit_parts.append(f"CRS: {_crs_human} ({_crs_id})" if _crs_human else f"CRS: {_crs_id}")
    _dlbl = view.get("date_label", defaults.get("date_label", "Date:"))
    credit_parts.append(f"{_dlbl} {date.today().isoformat()}")
    # Optional figure number / revision (view-level figure_id, view/defaults-level
    # revision). Figure numbers with revisions are common in EIA figure sets and make a
    # figure traceable once detached from its document. Deliberately lightweight - no
    # CAD-style title block. Nothing is shown when not configured.
    _fig = view.get("figure_id")
    _rev = view.get("revision", defaults.get("revision"))
    if _fig or _rev:
        _ids = []
        if _fig:
            _ids.append(f"{view.get('figure_label', defaults.get('figure_label', 'Figure:'))} {_fig}")
        if _rev:
            _ids.append(f"{view.get('revision_label', defaults.get('revision_label', 'Version:'))} {_rev}")
        credit_parts.insert(0, " ".join(_ids))
    credit_text = " | ".join(credit_parts)

    # [Known simplification] This is the "absent" list BEFORE pull_in is applied. If a
    # view uses both pull_in and panel mode, some layers may later move from "absent"
    # to "off-map marker", but the panel text has already been measured and worded from
    # this earlier, more conservative list, so it may list a layer or two that end up
    # marked on the map. pull_in can only shorten the absent list, never lengthen it,
    # so the reserved width is never exceeded - the wording is just occasionally stale.
    # Not worth a two-pass layout; left as a known limitation.
    early_absent_labels = [label for _, _, label in absent_status]

    hl_notes_pre = []
    for i, hspec in enumerate(view.get("highlight", []) or []):
        lyr = layers_by_key.get(hspec["layer_key"])
        if lyr is None:
            continue
        color = tuple(hspec.get("color") or PALETTE["highlight_ring"][i % len(PALETTE["highlight_ring"])])
        n = count_highlight_hits(lyr, hspec.get("id_field", "id"), set(str(x) for x in hspec.get("ids", [])))
        if n and hspec.get("label"):
            hl_notes_pre.append((color, hspec["label"]))

    # ── panel mode: measure decorations, fix panel thickness, split map_rect/panel_rect ──
    show_legend = view.get("legend", defaults.get("legend", True)) and bool(legend_entries)
    show_scalebar = view.get("scalebar", defaults.get("scalebar", True))
    show_north = view.get("north", defaults.get("north", True))
    cb_key = next((k for k in present_keys if k in raster_legend), None) \
        if view.get("colorbar", defaults.get("colorbar", True)) else None

    if is_panel:
        meas_img = QImage(QSize(1, 1), QImage.Format_ARGB32)  # Must be held in a variable, not passed inline
        meas_painter = QPainter(meas_img)                     # to QPainter: an inline temporary is garbage-
                                                                # collected while still in use, which raises
                                                                # "Cannot destroy paint device that is being painted"
        physical_h_cm_for_fs = view.get("physical_height_cm", defaults.get("physical_height_cm", 12.5))
        legend_title_text = (view.get("title") if view.get("legend_title", defaults.get("legend_title", False))
                              else None) if show_legend else None
        # legend_max_width_frac: the legend box may not exceed this fraction of the
        # canvas width, implementing the cartographic guideline that the legend should
        # not dominate the map (common reference: 15-20% of map width). The default of
        # 0.32 is more generous because labels in many non-Latin scripts take more
        # width than their English equivalents; a tighter limit would truncate most
        # real legends.
        legend_max_w = tw * view.get("legend_max_width_frac", defaults.get("legend_max_width_frac", 0.32))

        locator_base_k = make_scaler(th, physical_h_cm_for_fs).k

        def build_panel_items(fs_):
            # In panel-right (vertical) the legend/notes end up inside the single card
            # built by build_unified_panel, which adds UNIFIED_PANEL_PAD_K padding on each
            # side. If child items were wrapped at the full legend_max_w (a plain fraction
            # of canvas width, without the card padding), text could reach legend_max_w
            # and the padded card would exceed the budget; shrinking the card border
            # afterwards does not narrow the text, so text was drawn outside the card's
            # right edge. Subtract the card padding for panel-right only; horizontal
            # (panel-bottom, no unified card) keeps the original legend_max_w.
            item_max_w = (legend_max_w - 2 * UNIFIED_PANEL_PAD_K * fs_.k) if orientation != "horizontal" \
                else legend_max_w
            items = []
            if show_north:
                items.append({"kind": "north", "layout": north_panel_layout(fs_)})
            if show_scalebar:
                items.append({"kind": "scalebar", "layout": scalebar_panel_layout(fs_, ratio=scale_ratio, widen=orientation == "horizontal")})
            if show_legend:
                items.append({"kind": "legend",
                              "layout": legend_layout(meas_painter, legend_entries, legend_title_text, fs_,
                                                       max_w=item_max_w)})
            if cb_key:
                info = raster_legend[cb_key]
                items.append({"kind": "colorbar", "info": info,
                              "layout": colorbar_panel_layout(meas_painter, info["unit"], fs_,
                                                               compact=(orientation == "vertical"))})
            # Credit line / absent note / highlight notes live in the panel too, sharing
            # the same "information panel" visual language as legend / north arrow /
            # scale bar.
            #
            # Two font tiers only: hl_note/absent_note are content needed to read the map
            # (as important as legend entries - without them the highlight rings or the
            # missing layers cannot be interpreted), so they match the legend body text
            # (fs(8), font_pt in legend_layout). The credit line is provenance metadata
            # (source / CRS / date) and, by cartographic convention, one step smaller
            # than the legend, so it stays at 6.5 pt.
            if locator is not None:
                # Scale with the panel shrink factor: the locator is sized as a fraction of
                # the canvas, not via fs, so if it did not shrink the rest of the content
                # would have to shrink even more to make room for it.
                sk = fs_.k / locator_base_k
                lw = tw * locator["cfg"]["width_frac"] * sk
                if orientation == "horizontal":
                    lw = min(lw, tw * 0.16)
                else:
                    lw = min(lw, item_max_w)
                # Height cap 24%: tall, narrow countries become very tall when sized by
                # width, which would force the whole card to shrink (and the legend text
                # with it).
                items.append({"kind": "locator", "layout": locator_layout(locator, lw, max_h_px=th * 0.24 * sk)})
            for color, hl_label in hl_notes_pre:
                items.append({"kind": "hl_note", "color": color, "text": hl_label,
                              "layout": text_block_panel_layout(meas_painter, hl_label, fs_,
                                                                 item_max_w, font_pt=8, has_dot=True)})
            if early_absent_labels:
                note_text = (f"Not present in view extent: {', '.join(early_absent_labels[:3])}"
                             + (", etc." if len(early_absent_labels) > 3 else ""))
                items.append({"kind": "absent_note", "text": note_text,
                              "layout": text_block_panel_layout(meas_painter, note_text, fs_,
                                                                 item_max_w, font_pt=8)})
            if credit_text:
                items.append({"kind": "credit", "text": credit_text,
                              # The credit is provenance (source / basis / CRS / date);
                              # truncating it loses traceability - a long source string
                              # would otherwise cut the date off with "...". Allow 6 lines.
                              "layout": text_block_panel_layout(meas_painter, credit_text, fs_,
                                                                 item_max_w, font_pt=6.5, max_lines=6)})
            info_card_cfg = view.get("info_card")
            if info_card_cfg and info_card_cfg.get("lines"):
                items.append({"kind": "info_card",
                              "layout": info_card_panel_layout(meas_painter, info_card_cfg.get("title"),
                                                                info_card_cfg["lines"], fs_, item_max_w)})
            return items

        fs = make_scaler(th, physical_h_cm_for_fs)
        map_fs = fs  # map side (title bar / map labels / highlight rings / neatline) always uses the unscaled fs; see shrink branch below
        panel_items = build_panel_items(fs)
        natural_thickness = None

        # [Gotcha] fs is derived from canvas height (th) + physical_height_cm only, but
        # panel-bottom lays decorations out along the canvas WIDTH (tw). On a tall, narrow
        # canvas (e.g. a portrait overview map, 1400x2400 at 12 cm) with panel-bottom, the
        # height-derived font size may not fit the width, and the legend gets clipped off
        # the canvas edge.
        # Fix: measure how much space the decorations need along the layout axis
        # (horizontal = width, vertical = height); if it exceeds the canvas, shrink fs by
        # the overflow ratio and re-measure. Normal aspect ratios are unaffected, so the
        # "follow the physical size" premise holds; only extreme canvases get the fallback
        # shrink instead of content silently poking out of the canvas.
        gap = 14 * fs.k
        margin = 2 * 16 * fs.k
        if orientation == "horizontal":
            along_axis_total = sum(it["layout"]["w"] for it in panel_items) + gap * max(0, len(panel_items) - 1)
            available = tw - margin
        else:
            along_axis_total = sum(it["layout"]["h"] for it in panel_items) + gap * max(0, len(panel_items) - 1)
            available = th - margin
        if panel_items and along_axis_total > available > 0:
            shrink = max(0.35, (available / along_axis_total) * 0.97)
            print(f"⚠ [{view.get('out')}] decorations exceed the available canvas "
                  f"{'width' if orientation == 'horizontal' else 'height'} "
                  f"({along_axis_total:.0f}px > {available:.0f}px); "
                  f"shrinking to {shrink:.0%} to avoid clipping - triggered by extreme aspect ratios, "
                  f"consider adjusting size or using the other decoration_layout side")
            # Shrinking applies to panel content only. The map side (title bar, map
            # labels, highlight rings, neatline) keeps map_fs (unscaled), and the panel
            # keeps its natural (pre-shrink) thickness with the shrunk card centred in it,
            # so a batch of figures keeps consistent title sizes and map/panel split.
            if orientation == "horizontal":
                natural_thickness = max(it["layout"]["h"] for it in panel_items) + 32 * fs.k
            else:
                natural_thickness = _unified_from_items(panel_items, fs)["w"] + 32 * fs.k
            fs = make_scaler(th * shrink, physical_h_cm_for_fs)
            panel_items = build_panel_items(fs)
        # panel-bottom may still be too wide after shrinking: make_scaler's floors (font
        # 7 px, k 1.0) limit how far shrink can go, and text blocks (credit/notes) wrap at
        # a fixed legend_max_w (32% of the canvas) that does not narrow with scaling. The
        # last item (credit) could then run past the right canvas edge. Fallback: split
        # the remaining width evenly between text blocks and re-wrap them at the actual
        # available width (narrower only means more lines).
        if orientation == "horizontal" and panel_items:
            gap_now = 14 * fs.k
            text_kinds = ("hl_note", "absent_note", "credit")
            total_now = sum(it["layout"]["w"] for it in panel_items) + gap_now * (len(panel_items) - 1)
            texts = [it for it in panel_items if it["kind"] in text_kinds]
            if texts and total_now > available:
                fixed = sum(it["layout"]["w"] for it in panel_items if it["kind"] not in text_kinds)
                share = (available - fixed - gap_now * (len(panel_items) - 1)) / len(texts)
                if share > 60 * fs.k:
                    for it in texts:
                        fpt = it["layout"]["font_pt"]
                        it["layout"] = text_block_panel_layout(meas_painter, it["text"], fs, share,
                                                                font_pt=fpt, has_dot=it["kind"] == "hl_note",
                                                                max_lines=6 if it["kind"] == "credit" else 4)
                else:
                    print(f"⚠ [{view.get('out')}] panel is too narrow even for the text blocks (remaining {share:.0f}px); "
                          f"credit/notes may overflow the canvas - use panel-right or increase the size width")
        meas_painter.end()

        pad = 32 * fs.k
        # panel-right (vertical) uses the overall size of the single card computed by
        # build_unified_panel; panel-bottom (horizontal) keeps the flat-list per-item sum
        # (see layout_panel_items). unified_layout computed here is reused later when
        # drawing (the is_panel else branch) rather than rebuilt - it is a local of this
        # render_view call and stays valid for the whole function.
        unified_layout = None
        if orientation != "horizontal":
            top_pieces = [(it["kind"], it["layout"], None) for it in panel_items
                          if it["kind"] in ("north", "scalebar")]
            legend_piece = next((it["layout"] for it in panel_items if it["kind"] == "legend"), None)
            colorbar_item = next((it for it in panel_items if it["kind"] == "colorbar"), None)
            colorbar_piece = colorbar_item["layout"] if colorbar_item else None
            colorbar_extra = colorbar_item["info"] if colorbar_item else None
            note_pieces = [(it["kind"], it["layout"], it.get("color")) for it in panel_items
                           if it["kind"] in ("locator", "hl_note", "absent_note", "credit", "info_card")]
            unified_layout = build_unified_panel(top_pieces, legend_piece, colorbar_piece, note_pieces, fs)
            if colorbar_extra is not None:
                unified_layout["positions"] = [
                    (kind, rx, ry, lay, colorbar_extra if kind == "colorbar" else extra)
                    for kind, rx, ry, lay, extra in unified_layout["positions"]
                ]
            content_h = unified_layout["h"] + pad
            content_w = unified_layout["w"] + pad
        else:
            content_h = max([it["layout"]["h"] for it in panel_items], default=0) + pad if panel_items else 0
            content_w = max([it["layout"]["w"] for it in panel_items], default=0) + pad if panel_items else 0
        if natural_thickness is not None:
            if orientation == "horizontal":
                content_h = max(content_h, natural_thickness)
            else:
                content_w = max(content_w, natural_thickness)

        # panel_thickness_cm (optional, defaults/view level): pin the panel thickness to a
        # fixed physical size so it does not float with content (number of legend rows,
        # colour bar present or not). This keeps the map:decoration ratio constant across
        # a set of figures. When unset, the panel adapts to its content.
        panel_thickness_cm = view.get("panel_thickness_cm", defaults.get("panel_thickness_cm"))
        px_per_cm = th / max(float(physical_h_cm_for_fs or 12.5), 1.0)

        if orientation == "horizontal":
            if panel_thickness_cm:
                thickness = int(round(panel_thickness_cm * px_per_cm))
                if content_h > thickness:
                    print(f"⚠ [{view.get('out')}] panel content height ({content_h:.0f}px) exceeds fixed "
                          f"panel_thickness_cm={panel_thickness_cm}cm ({thickness}px); decorations will overlap/"
                          f"overflow the panel - increase panel_thickness_cm or trim legend entries")
            else:
                thickness = int(content_h)
            map_w, map_h = tw, max(1, th - thickness - title_h_px)
            map_rect = QRectF(0, title_h_px, map_w, map_h)
            panel_rect = QRectF(0, map_rect.bottom(), tw, thickness)
        else:
            if panel_thickness_cm:
                thickness = int(round(panel_thickness_cm * px_per_cm))
                if content_w > thickness:
                    print(f"⚠ [{view.get('out')}] panel content width ({content_w:.0f}px) exceeds fixed "
                          f"panel_thickness_cm={panel_thickness_cm}cm ({thickness}px); decorations will overlap/"
                          f"overflow the panel - increase panel_thickness_cm or trim legend entries")
            else:
                thickness = int(content_w)
            map_w, map_h = max(1, tw - thickness), max(1, th - title_h_px)
            map_rect = QRectF(0, title_h_px, map_w, map_h)
            # panel_rect does not subtract title_h_px: the panel column sits beside the
            # map column and the title only describes the map, so the panel top does not
            # need to give way (same principle as the title bar spanning map_rect only,
            # not the full tw).
            panel_rect = QRectF(map_w, 0, thickness, th)

        ext = fix_aspect(QgsRectangle(raw_ext), map_w, map_h)
    else:
        map_w, map_h = tw, max(1, th - title_h_px)
        map_rect = QRectF(0, title_h_px, tw, map_h)
        panel_rect = None
        fs = make_scaler(th, view.get("physical_height_cm", defaults.get("physical_height_cm", 12.5)))
        map_fs = fs

    # ── pull_in (grow the extent to mark off-map features) ─────────────────
    pull_cfg = {p["layer_key"]: float(p.get("cap_km", 20)) * 1000.0 for p in (view.get("pull_in") or [])}
    pulled_in, still_absent = [], []
    ref_point = ext.center()
    for k, lyr, label in absent_status:
        info = nearest_feature_info(lyr, ref_point, project, pull_cfg.get(k))
        if info is None:
            still_absent.append(label)
        else:
            pulled_in.append((k, label, info["distance_m"], info["point_prj"]))
    absent = still_absent

    pulled_keys = {k for k, *_ in pulled_in}
    if pulled_in:
        for _, _, _, pt in pulled_in:
            pr = QgsRectangle(pt, pt)
            grow_by_km(pr, 4.0, project)
            ext.combineExtentWith(pr)
        fix_aspect(ext, map_w, map_h)

    # overlay: plan floating decoration positions so they do not cover the subject
    # (see plan_overlay_decorations)
    overlay_plan = None
    if not is_panel:
        protect = overlay_protect_geometry(view, defaults, layers_by_key, project, ext)
        if protect is not None and not protect.isEmpty():
            _mimg = QImage(QSize(1, 1), QImage.Format_ARGB32)
            _mp = QPainter(_mimg)
            sizes = {"north": show_north}
            if cb_key:
                sizes["colorbar"] = colorbar_box_size(raster_legend[cb_key]["unit"], fs)
            if credit_text:
                sizes["credit"] = attribution_box_size(credit_text, fs)
            if show_legend:
                _t = view.get("title") if view.get("legend_title", defaults.get("legend_title", False)) else None
                _ll = legend_layout(_mp, legend_entries, _t, fs, max_w=map_w * view.get(
                    "legend_max_width_frac", defaults.get("legend_max_width_frac", 0.32)))
                sizes["legend"] = (_ll["w"], _ll["h"])
            if locator is not None:
                _lz = locator_layout(locator, map_w * locator["cfg"]["width_frac"], max_h_px=map_h * 0.4)
                sizes["locator"] = (_lz["w"], _lz["h"])
            _mp.end()
            trial = QgsRectangle(ext)
            for attempt in range(11):
                if show_scalebar:
                    sizes["scalebar"] = overlay_scalebar_right(map_w, ground_width_m(trial, project) / map_w,
                                                               fs, scale_ratio)
                overlay_plan = plan_overlay_decorations(trial, map_w, map_h, fs, protect, sizes)
                if overlay_plan is not None:
                    break
                trial.scale(1.1)
            if overlay_plan is None:
                print(f"⚠ [{view.get('out')}] even after growing the extent {1.1 ** 10:.1f}x the floating legend/north arrow"
                      f" etc. cannot avoid the subject (protect_layers) - use a panel layout or fewer floating decorations",
                      file=sys.stderr)
            elif attempt:
                print(f"ℹ [{view.get('out')}] extent grown {1.1 ** attempt:.2f}x so floating decorations do not cover the subject")
                ext = trial

    render_keys = [k for k in present_keys if k not in pulled_keys]
    render_keys = reorder_restriction_stack(render_keys, style_keys)
    # A basemap (opaque XYZ tiles) that is not last in `layers` covers every layer listed
    # after it. This fails silently - the covered layers still appear in the legend - so
    # warn about it.
    _bm_idx = [i for i, k in enumerate(render_keys) if k in (basemap_aliases or {})]
    if _bm_idx and _bm_idx[0] < len(render_keys) - 1:
        _hidden = render_keys[_bm_idx[0] + 1:]
        print(f"⚠ [{view.get('out')}] basemap {render_keys[_bm_idx[0]]!r} is not last in layers "
              f"and will cover {_hidden} (first in layers = topmost; the basemap should be last)", file=sys.stderr)
    stack = [layers_by_key[k] for k in render_keys]

    lbl_mode = view.get("labels", defaults.get("labels", "auto"))
    labels_override = None if lbl_mode in (None, "auto") else bool(lbl_mode in (True, "on"))
    # Reference place-name labels (force_labels: true, e.g. sub-district names) are not
    # controlled by the view-level labels: on/off switch. Views often set labels: off to
    # hide whole-layer labels such as turbine IDs; if place names followed the same
    # labels_override they would be switched off too, defeating "always show place names".
    force_label_objs = {layers_by_key[k] for k in render_keys if k in (force_label_keys or set())}
    # For layers with label_top_k_by_area, query which features overlap the current
    # view's extent the most and keep only their $id for the Show expression. This must
    # run after ext is final (both the panel and overlay branches have converged it by
    # now); it cannot move to load_layer_pool, which knows nothing about any view's extent.
    label_show_ids = {}
    for k in render_keys:
        topk = (label_topk or {}).get(k)
        if topk:
            ids = top_k_ids_by_overlap_area(layers_by_key[k], ext, project, topk)
            if ids:
                label_show_ids[layers_by_key[k]] = ids
    _label_font_pt = view.get("label_font_pt", 8)
    # force_label_font_pt is fixed at 2 pt below _label_font_pt (with a floor so it never
    # becomes unreadable). A relative value rather than a hard-coded 6: _label_font_pt is
    # per-view (dense layouts may lower it from 8 to 6), and a fixed 6 would then collide
    # with the turbine ID size. Both go through label_fs(), and a 2 pt difference remains
    # visible after conversion.
    _force_label_font_pt = max(4, _label_font_pt - 2)
    saved = tweak_layers(stack, scale=view.get("symbol_scale"), labels=labels_override, label_fs=map_fs,
                          label_font_pt=_label_font_pt, force_labels=force_label_objs,
                          label_show_ids=label_show_ids, force_label_font_pt=_force_label_font_pt,
                          label_offset_ctx=(ext.width() / map_w * dpi / 25.4, project.crs().authid()))
    gray_snaps = gray_basemaps_for_view(view, defaults, render_keys, layers_by_key, basemap_aliases)
    try:
        ms = QgsMapSettings()
        ms.setLayers(stack)
        ms.setBackgroundColor(QColor(bg))
        ms.setOutputSize(QSize(map_w, map_h))
        ms.setDestinationCrs(project.crs())
        ms.setExtent(ext)
        ms.setOutputDpi(dpi)
        # [Gotcha] QgsMapRendererParallelJob (one render thread per layer, each with its
        # own connection to the data source) intermittently fails with "unable to open
        # database file" when many layers come from the same large GPKG (not every run,
        # not a fixed layer; the trigger threshold was never pinned down). The
        # single-threaded CustomPainterJob has not reproduced it. Trading some parallel
        # speed for stability is worth it for offline batch rendering.
        map_img = QImage(QSize(map_w, map_h), QImage.Format_ARGB32)
        render_painter = QPainter(map_img)
        job = QgsMapRendererCustomPainterJob(ms, render_painter)
        job.start()
        job.waitForFinished()
        render_painter.end()
    finally:
        restore_layers(saved)
        restore_basemap_gray(gray_snaps)

    img = QImage(QSize(tw, th), QImage.Format_ARGB32)
    img.fill(QColor(bg))
    blit = QPainter(img)
    # Use the integer QPoint overload rather than QRectF: even with a target rectangle
    # the same size as the source, Qt may take an interpolating scale path, producing
    # edge/texture jitter that is invisible but shows up in pixel diffs. The QPoint
    # overload is a plain bitmap copy, guaranteeing pixel-identical output in overlay mode.
    # The blit origin is the top-left of map_rect: when title_h_px > 0, map_img is shorter
    # than img (the title bar height was removed from the extent before rendering) and
    # must be placed below the title strip, not at the origin.
    blit.drawImage(QPoint(int(map_rect.x()), int(map_rect.y())), map_img)
    blit.end()

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)

    # Credit line: data source / CRS / date. Basemap copyright alone (a licensing
    # requirement) is not enough for traceability of delivered figures, so the credit
    # line is drawn unconditionally (unlike basemap attribution, which only appears when
    # such a basemap is used). `source` comes from the optional view/defaults field and
    # is skipped when unset; CRS and date are always available. credit_text /
    # attribution_texts were computed earlier (after collecting legend_entries) and are
    # reused here so the two cannot drift apart.
    #
    # The credit line is only drawn in the map corner in overlay mode. In panel mode it
    # is already a "credit" item in the panel: map and panel are two parts of the same
    # sheet (the neatline frames both), so leaving the credit on the map while the legend
    # moved would be inconsistent.
    if not is_panel:
        # Position: like the legend, compare the busyness of the bottom-left and
        # bottom-right candidates on the clean rendered map (img only contains the
        # blitted map at this point, no decorations yet) and pick the emptier side. A
        # fixed bottom-left corner is quite often covered by restriction areas or
        # hatching; the legend's region_busyness mechanism is reused here.
        credit_box_w, credit_box_h = attribution_box_size(credit_text, fs)
        credit_margin = 6 * fs.k
        credit_candidates = {
            "bottom-left": (map_rect.x() + credit_margin, map_rect.y() + map_h - credit_box_h - credit_margin),
            "bottom-right": (map_rect.x() + map_w - credit_box_w - credit_margin,
                              map_rect.y() + map_h - credit_box_h - credit_margin),
        }
        credit_bg = QColor(bg)
        credit_align = overlay_plan["credit"] if overlay_plan and "credit" in overlay_plan else min(
            credit_candidates,
            key=lambda name: region_busyness(img, *credit_candidates[name], credit_box_w, credit_box_h, credit_bg),
        )
        draw_attribution(painter, map_rect.x(), map_rect.y(), map_w, map_h, credit_text, fs,
                          align="left" if credit_align == "bottom-left" else "right")
    else:
        credit_align = None

    # Legend position (overlay mode only) must be chosen by sampling the clean render
    # before any other decoration is drawn, so our own markers/highlight rings do not
    # affect the "is this area empty" test. Panel mode needs no runtime probing; its
    # positions were computed deterministically by layout_panel_items above.
    # The credit line is already drawn, so its corner is excluded - its semi-transparent
    # white box differs little from a typical light canvas, and the legend's busyness
    # sampling might not reliably avoid it.
    legend_pos, legend_lay, legend_corner = None, None, None
    if not is_panel:
        legend_title_text = None
    if show_legend and not is_panel:
        show_t = view.get("legend_title", defaults.get("legend_title", False))
        legend_title_text = view.get("title") if show_t else None
        legend_max_w = map_w * view.get("legend_max_width_frac", defaults.get("legend_max_width_frac", 0.32))
        legend_lay = legend_layout(painter, legend_entries, legend_title_text, fs, max_w=legend_max_w)
        margin = 22 * fs.k
        forced = view.get("legend_position", defaults.get("legend_position"))
        if forced:
            positions = {
                "top-left": (margin, map_rect.y() + margin),
                "top-right": (tw - margin - legend_lay["w"], map_rect.y() + margin),
                "bottom-left": (margin, th - margin - legend_lay["h"]),
                "bottom-right": (tw - margin - legend_lay["w"], th - margin - legend_lay["h"]),
            }
            legend_pos = positions.get(forced, positions["top-left"])
            legend_corner = forced if forced in positions else "top-left"
        elif overlay_plan and "legend" in overlay_plan:
            legend_corner = overlay_plan["legend"]
            res = 55 * fs.k if show_scalebar else 0
            legend_pos = {
                "top-left": (margin, map_rect.y() + margin),
                "top-right": (tw - margin - legend_lay["w"], map_rect.y() + margin),
                "bottom-left": (margin, th - margin - res - legend_lay["h"]),
                "bottom-right": (tw - margin - legend_lay["w"], th - margin - res - legend_lay["h"]),
            }[legend_corner]
        else:
            bottom_reserve = 55 * fs.k if show_scalebar else 0
            legend_corner, px, py = pick_legend_position(img, legend_lay["w"], legend_lay["h"], margin, bg, bottom_reserve,
                                              exclude={credit_align}, top=map_rect.y())
            legend_pos = (px, py)

    # Off-map markers and highlight rings use ms.mapToPixel(), i.e. coordinates of the
    # map image itself. The map is blitted at the top-left of map_rect (shifted down by
    # the title bar when title_h_px > 0), so the painter must be translated by the same
    # offset, otherwise rings land one title-bar height too high (on the wrong feature).
    painter.save()
    painter.translate(map_rect.x(), map_rect.y())
    marker_canvas_h = map_h - (60 * fs.k if absent else 0)
    for _, label, dist_m, pt in pulled_in:
        draw_offmap_marker(painter, ms, pt, label, dist_m, map_fs, map_w, marker_canvas_h)

    # Highlight rings are always drawn on the map (overlay and panel mode): they mark
    # positions of specific features and are not metadata, so they do not belong in the
    # panel. hl_notes_pre already determined whether each highlight hit anything and
    # what its note says; here we only draw rings, without re-checking n > 0 (avoids
    # two diverging copies of that logic).
    for i, hspec in enumerate(view.get("highlight", []) or []):
        lyr = layers_by_key.get(hspec["layer_key"])
        if lyr is None:
            continue
        color = tuple(hspec.get("color") or PALETTE["highlight_ring"][i % len(PALETTE["highlight_ring"])])
        draw_highlights(painter, ms, project, lyr, hspec.get("id_field", "id"),
                         set(str(x) for x in hspec.get("ids", [])), map_fs, ring_color=color,
                         r_scale=1.0 + i * 0.35, tag=hspec.get("tag"))
    painter.restore()

    if show_legend and not is_panel:
        draw_legend_at(painter, legend_pos[0], legend_pos[1], legend_lay, fs)

    # Locator inset (overlay): of the four corners, the north arrow takes top-right and
    # the scale bar bottom-left, legend and credit have each chosen a corner, and
    # bottom-right may hold notes. Pick the emptiest remaining corner; if none is left,
    # skip with a warning rather than drawing over other decorations.
    if locator is not None and not is_panel:
        loc_lay = locator_layout(locator, map_w * locator["cfg"]["width_frac"], max_h_px=map_h * 0.4)
        lm = 22 * fs.k
        # The scale bar and credit only occupy a thin strip at the corner, not the whole
        # corner - bottom candidates are shifted up to clear them instead of treating the
        # corner as taken (otherwise all four corners can be judged full).
        credit_h = attribution_box_size(credit_text, fs)[1] + 6 * fs.k if credit_text else 0
        bl_up = (60 * fs.k if show_scalebar else 0) + (credit_h if credit_align == "bottom-left" else 0)
        br_up = credit_h if credit_align == "bottom-right" else 0
        br_blocked = False
        if cb_key:
            # The overlay colour bar hugs the right edge, extending about 264k from the top
            # (draw_colorbar's 84k start + 150k bar + padding/unit row). A bottom-right
            # locator can only use the space below it; if that is too small, give up the
            # bottom-right corner (otherwise the locator's top edge covers the colour bar).
            room = map_h - lm - br_up - (264 + 14) * fs.k
            if room >= map_h * 0.12:
                loc_lay = locator_layout(locator, loc_lay["w"], max_h_px=min(loc_lay["h"], room))
            else:
                br_blocked = True
        cands = {
            "top-left": (map_rect.x() + lm, map_rect.y() + lm),
            "top-right": (map_rect.x() + map_w - lm - loc_lay["w"], map_rect.y() + lm),
            "bottom-left": (map_rect.x() + lm, map_rect.y() + map_h - lm - loc_lay["h"] - bl_up),
            "bottom-right": (map_rect.x() + map_w - lm - loc_lay["w"], map_rect.y() + map_h - lm - loc_lay["h"] - br_up),
        }
        want = locator["cfg"]["position"]
        if want == "auto" and overlay_plan and "locator" in overlay_plan:
            pick = overlay_plan["locator"]
        elif want != "auto":
            pick = want if want in cands else None
        else:
            occupied = {legend_corner}
            if show_north or cb_key:  # the overlay colour bar is also top-right (below the north arrow)
                occupied.add("top-right")
            if br_blocked:
                occupied.add("bottom-right")
            if hl_notes_pre or absent:
                occupied.add("bottom-right")
            free = [c for c in ("top-left", "bottom-right", "top-right", "bottom-left") if c not in occupied]
            pick = min(free, key=lambda c: region_busyness(img, *cands[c], loc_lay["w"], loc_lay["h"], QColor(bg))) \
                if free else None
        if pick:
            draw_locator(painter, *cands[pick], loc_lay, ext, project, fs)
        else:
            print(f"⚠ [{view.get('out')}] all four corners are taken by legend/north arrow/scale bar/credit; locator not drawn - "
                  f"use a panel layout, or set locator.position to a specific corner", file=sys.stderr)

    # Highlight notes / "not present in view extent" note: in panel mode these are
    # already panel items (hl_note / absent_note, built from the same hl_notes_pre /
    # early_absent_labels); only overlay mode draws them in the map corner.
    if not is_panel:
        note_x_right = map_w - 22 * fs.k
        note_y_bottom = map_h - 22 * fs.k
        note_left_limit = map_w * 0.32
        if show_scalebar and scale_ratio:
            # In overlay mode the scale ratio text sits to the right of the scale bar, in
            # the same height band as the bottom-right notes - keep the notes' left edge
            # beyond the ratio text, otherwise a long note covers it.
            note_left_limit = max(note_left_limit,
                                  overlay_scalebar_right(tw, ground_width_m(ext, project) / tw, fs, scale_ratio)
                                  + 16 * fs.k)
        for color, text in hl_notes_pre:
            note_y_bottom = draw_note(painter, note_x_right, note_y_bottom, text, fs,
                                       color=color, left_limit=note_left_limit) - 8 * fs.k
        if absent:
            draw_note(painter, note_x_right, note_y_bottom,
                      f"Not present in view extent: {', '.join(absent[:3])}" + (", etc." if len(absent) > 3 else ""),
                      fs, color=(90, 100, 110), left_limit=note_left_limit)

    if not is_panel:
        if show_scalebar:
            gw = ground_width_m(ext, project)
            draw_scalebar(painter, tw, th, gw / tw, fs, ratio=scale_ratio)
        if show_north:
            # Positioned relative to the map area, not the whole canvas: with a title bar
            # the north arrow would otherwise sit inside the title strip. Shift down by the
            # title bar height to align with the top of the map area.
            painter.save()
            painter.translate(0, map_rect.y())
            draw_north(painter, tw, map_h, fs)
            painter.restore()
        if cb_key:
            info = raster_legend[cb_key]
            painter.save()
            painter.translate(0, map_rect.y())  # shift with the north arrow to clear the title bar, otherwise the arrow covers the colour bar
            draw_colorbar(painter, tw, map_h, info["colors"], info["min"], info["max"], info["unit"], fs)
            painter.restore()
    else:
        # Panel decorations (north arrow / scale bar / colour bar ticks) pick a dark or
        # light ink from the perceived brightness of bg: the panel shares img.fill(bg)
        # with the map, and fixed light-grey decorations are nearly invisible on a white
        # background. See panel_ink_color().
        ink = panel_ink_color(bg)
        # Give the whole panel_rect a very light neutral tint (about 4% from bg: darker on
        # light backgrounds, lighter on dark ones). Without it, decorations float on blank
        # canvas the same colour as the map, and a short legend centred in the panel
        # leaves large empty areas that look unfinished rather than like a deliberate
        # sidebar. The tint is light enough not to compete with the white legend card -
        # the common "tinted sidebar + floating card" layout of professional reports.
        panel_bg = QColor(bg).darker(104) if ink == (60, 60, 60) else QColor(bg).lighter(104)
        painter.fillRect(panel_rect, panel_bg)
        gw = ground_width_m(ext, project) if show_scalebar else 0

        if orientation == "horizontal":
            placed = layout_panel_items(panel_items, orientation, panel_rect, fs, view_name=view.get("out"))
            for it, x, y in placed:
                if it["kind"] == "north":
                    draw_north_panel(painter, x, y, it["layout"], fs, color=ink)
                elif it["kind"] == "scalebar":
                    draw_scalebar_panel(painter, x, y, it["layout"], gw / map_w, fs, ink=ink)
                elif it["kind"] == "legend":
                    draw_legend_at(painter, x, y, it["layout"], fs)
                elif it["kind"] == "colorbar":
                    info = it["info"]
                    draw_colorbar_panel(painter, x, y, it["layout"], info["colors"], info["min"], info["max"],
                                         info["unit"], fs, ink=ink)
                elif it["kind"] == "hl_note":
                    draw_text_block_panel(painter, x, y, it["layout"], fs, text_color=ink, dot_color=it["color"])
                elif it["kind"] in ("absent_note", "credit"):
                    draw_text_block_panel(painter, x, y, it["layout"], fs, text_color=ink)
                elif it["kind"] == "info_card":
                    draw_info_card_panel(painter, x, y, it["layout"], fs, text_color=ink)
                elif it["kind"] == "locator":
                    draw_locator(painter, x, y, it["layout"], ext, project, fs)
        else:
            # panel-right is a single unified card: north arrow and scale bar in one row,
            # the legend in one section, notes/credit in another, separated by divider
            # lines instead of a separate border around every element.
            # unified_layout was computed during panel sizing; here we only fill in the
            # scale bar's real meters_per_px (which depends on the final extent, unknown
            # at sizing time) before drawing.
            mpp = gw / map_w if map_w else 0
            unified_layout["positions"] = [
                (kind, rx, ry, lay, mpp if kind == "scalebar" else (ext, project) if kind == "locator" else extra)
                for kind, rx, ry, lay, extra in unified_layout["positions"]
            ]
            card_margin = 24 * fs.k
            card_x = panel_rect.x() + (panel_rect.width() - unified_layout["w"]) / 2
            # panel_rect is always the full canvas height th. The card content is only
            # shrunk when it would overflow (see the shrink branch above); it never grows
            # to fill the panel when the content is much shorter. With little content the
            # card sits at the top of the panel with a large unused area below, which can
            # look unfinished. Default (card_valign unset or "top") keeps the card at the
            # top; `card_valign: center` (view/defaults level) centres the card vertically
            # in the panel - intended for figures whose content is naturally much shorter
            # than the canvas.
            card_valign = view.get("card_valign", defaults.get("card_valign", "top"))
            if card_valign == "center":
                card_y = panel_rect.y() + max(card_margin, (panel_rect.height() - unified_layout["h"]) / 2)
            else:
                card_y = panel_rect.y() + card_margin
            draw_unified_panel(painter, card_x, card_y, unified_layout, fs, ink=ink)

    # The title is on by default: title, legend, scale bar and north arrow are all basic
    # map elements. Figures from the same batch often share basemap, site and layout and
    # differ only in overlaid layers; once a figure is detached from its report (forwarded,
    # moved to an appendix, reordered) it cannot be identified without a title. Set
    # title_on_image: false at view/defaults level to opt out.
    # show_title / title_h_px reuse the values computed at the top of the function (so the
    # title_on_image condition is evaluated in one place only). The geographic extent was
    # already reduced by title_h_px; this only draws the title text into the reserved
    # strip, whose bottom edge coincides with the top of map_rect, so it cannot intrude
    # into map or panel content.
    if show_title:
        draw_title(painter, map_rect.width(), title, map_fs)

    if view.get("neatline", defaults.get("neatline", True)):
        # Neatline: a thin border around the whole sheet is standard on publication-grade
        # maps. Without it the figure edge coincides with the canvas crop line and reads
        # more like a screenshot than a map sheet. It is drawn around the entire canvas
        # (including the panel), not just map_rect - in panel mode map and decorations are
        # two parts of the same sheet and should share one frame.
        painter.setPen(QPen(QColor(60, 60, 60), max(1.2, map_fs.k * 0.9)))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(0.5, 0.5, tw - 1, th - 1))

    painter.end()

    out = Path(outdir) / f"{view['out']}.png"
    img.save(str(out), "PNG")
    return {
        "out": view["out"], "title": view.get("title"), "path": str(out), "status": "ok",
        "missing": missing, "size": [tw, th],
        "absent_in_extent": absent,
        "pulled_in_offmap": [{"label": lb, "nearest_m": d} for _, lb, d, _ in pulled_in],
        "layers": [l.name() for l in stack],
    }


# ──────────────────────────── CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--only", help="render only these views (comma-separated `out` names)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # [Gotcha] qgs.exitQgis() is deliberately NOT called: once objects such as
    # QgsMapRendererParallelJob / QPainter / symbols have been created during
    # rendering, exitQgis() segfaults during teardown (the figures and manifest are
    # already written; the crash happens in cleanup after all data is on disk). The
    # process exits right away and the kernel reclaims all resources, so skipping it is
    # safer; together with os._exit() in __main__ this avoids a known PyQGIS headless crash.
    QgsApplication.setPrefixPath(os.environ.get("QGIS_PREFIX_PATH", "/usr"), True)
    qgs = QgsApplication([], False)
    qgs.initQgis()

    # Relative file paths in the YAML config are resolved against the config file's directory.
    global CONFIG_DIR
    CONFIG_DIR = Path(args.config).resolve().parent

    (project, layers_by_key, layer_labels, raster_legend, basemap_aliases, style_keys,
     force_label_keys, label_topk) = load_layer_pool(cfg)
    print(f"✓ Layer pool loaded: {len(layers_by_key)} layers (project CRS: {project.crs().authid()})")

    defaults = cfg.get("defaults", {})
    views = cfg.get("views") or []
    if not views:
        sys.exit("✗ no views in config")
    only = set(args.only.split(",")) if args.only else None

    expand_preset(defaults)
    for view in views:
        expand_preset(view)

    results = []
    for view in views:
        if only and view["out"] not in only:
            continue
        r = render_view(project, view, defaults, layers_by_key, layer_labels, raster_legend, outdir,
                         basemap_aliases, style_keys, force_label_keys, label_topk)
        results.append(r)
        mark = "✓" if r["status"] == "ok" else "✗"
        miss = f"  [missing layers: {', '.join(r['missing'])}]" if r.get("missing") else ""
        dim = f" {r['size'][0]}×{r['size'][1]}" if r.get("size") else ""
        gone = f"  [no features in extent: {', '.join(r['absent_in_extent'])}]" if r.get("absent_in_extent") else ""
        pulled = r.get("pulled_in_offmap")
        pin = ("  [marked off-map via extended extent: " +
               ", ".join(f"{p['label']}@{p['nearest_m']/1000:.1f}km" for p in pulled) + "]") if pulled else ""
        print(f"  {mark} {r['out']:<20}{dim}  {r.get('title','')}{miss}{gone}{pin}")

    idx = outdir / "figures_manifest.json"
    merged = {}
    if idx.exists():
        for old in json.loads(idx.read_text(encoding="utf-8")):
            merged[old["out"]] = old
    for r in results:
        # The manifest stores paths relative to outdir so the output folder stays portable.
        merged[r["out"]] = {**r, "path": os.path.relpath(r["path"], outdir)} if r.get("path") else r
    idx.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8")

    ok_n = sum(1 for r in results if r["status"] == "ok")
    print(f"\n✓ {ok_n}/{len(results)} figures -> {outdir}")
    for r in results:
        if r.get("missing"):
            print(f"⚠ {r['out']} missing layers: {', '.join(r['missing'])} - the figure is rendered but incomplete")
    return 0 if ok_n == len(results) else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # skip normal interpreter finalization to avoid the PyQGIS teardown crash; see top of main()
