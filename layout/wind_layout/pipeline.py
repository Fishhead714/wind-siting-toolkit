"""Turbine-layout pipeline: terrain classification -> routing -> flat-branch AEP
optimisation or mountain-branch terrain-constrained layout -> directional-ellipse
spacing check. Runs per candidate zone and is idempotent (safe to re-run).

Single source of truth for the orchestration: import this module instead of
copying the logic.

The other modules (``terrainclassify``, ``terrainlayout``, ``layoutoptimize``,
``spacingcheck``, ``siteconditions``, ``rasterutils``) can still be imported on
their own, e.g. to check the spacing of a hand-drawn set of points without running
the whole pipeline.

Environment: TOPFARM constrains the Python version (see requirements.txt); the
whole pipeline runs in one process.

Validity boundary: see ``OUTPUT_DISCLAIMER``. Results are screening /
pre-feasibility level, NOT a bankable energy yield assessment.
"""
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np

from .rasterutils import sample_raster, masked_stats

from .terrainclassify import (
    load_dem, classify_zone, dem_pixel_size_m,
    RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M,
)
from .terrainlayout import compute_slope, compute_valley_mask, optimize_layout_mountain
from .layoutoptimize import (
    optimize_layout, evaluate_layout, compact_footprint, capacity_density_mw_km2,
    check_specific_power, wake_model_sensitivity, recommended_wake_model,
    WAKE_CONSTANTS,
)
from .spacingcheck import check_spacing, dominant_energy_azimuth
from .siteconditions import (
    air_density_from_elevation,
    air_density_spread,
    PLAUSIBLE_SHEAR_ALPHA,
    check_iec_turbulence_suitability,
    inflow_angle_deg,
    omni_inflow_angle_deg,
    IEC_MAX_FLOW_INCLINATION_DEG,
)

from py_wake.site import UniformWeibullSite
from py_wake.wind_farm_models import PropagateDownwind


#: Validity boundary of the outputs. Written into the `disclaimer` field of every
#: summary json so that downstream scripts and reports cannot bypass it. The
#: numbers here (air-density correction, true wake-loss definition, effective
#: turbulence check) look credible, which is exactly why they are easy to mistake
#: for a bankable energy assessment; the boundary therefore travels with them.
OUTPUT_DISCLAIMER = {
    "not_a_bankable_energy_assessment": (
        "This pipeline produces screening / pre-feasibility level layouts and energy "
        "estimates. It is **not a bankable energy yield assessment (EYA)**. Missing: "
        "measured wind data with long-term correction (MCP), a full loss cascade "
        "(availability / electrical / power-curve deviation / environmental / "
        "curtailment), uncertainty combination and P50/P90."
    ),
    "flow_model": (
        "Wakes and flow use py_wake engineering models (flat-terrain flow assumption). "
        "Local speed-up, lee-side separation and terrain-induced turbulence are not "
        "modelled: the mountain branch produces **terrain-constrained candidate "
        "positions**, not wake-optimal positions, and needs review with a "
        "microscale (CFD-type) flow model before any decision."
    ),
    "iec_check_scope": (
        "The effective-turbulence / inflow-angle checks are **screening level**: "
        "turbulence comes from an engineering model, not measurements; the "
        "1.28 * sigma_sigma_hat term of IEC 61400-1 eq. (35) is not included, so the "
        "verdict is looser than the standard; constants follow Ed.3 (2005) as quoted "
        "and Ed.4 (2019) was not checked. Not a substitute for a manufacturer load "
        "check or type certification."
    ),
    "not_covered": [
        "Noise propagation and shadow flicker modelling",
        "Ice throw, aviation obstacles, radar interference",
        "50-year extreme wind speed and extreme turbulence (need long-term measurements + extreme-value fitting)",
        "Cost optimisation of collection lines / roads / land acquisition",
    ],
}


def _digest_model(m):
    """Summarise a py_wake model object into a comparable dict: class name + all scalar / short-sequence attributes.

    Recording only ``type(m).__name__`` would give ``TurboGaussianDeficit(A=0.04)``
    and ``A=0.064`` the same fingerprint, so a re-run would skip and return the
    result of the wrong constant. **Model parameters must enter the fingerprint;
    the class name is not enough.**
    """
    if m is None:
        return None
    out = {"__class__": type(m).__name__}
    for k, v in sorted(vars(m).items()):
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 8 and all(
                isinstance(x, (int, float, bool, str)) for x in v):
            out[k] = list(v)
        elif hasattr(v, "__name__"):          # functions such as ct2a_madsen
            out[k] = v.__name__
        else:
            out[k] = type(v).__name__          # nested models: type only, avoid unbounded expansion
    return out


def _digest_raster(arr, label):
    """Raster digest: shape + NaN-safe statistics. Does not hash the full array (too costly for huge grids)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    finite = a[np.isfinite(a)] if a.size else a
    return {
        "label": label, "shape": list(a.shape), "dtype": str(a.dtype),
        "n_finite": int(finite.size),
        "mean": (round(float(finite.mean()), 6) if finite.size else None),
        "std": (round(float(finite.std()), 6) if finite.size else None),
        "min": (round(float(finite.min()), 6) if finite.size else None),
        "max": (round(float(finite.max()), 6) if finite.size else None),
    }


def _digest_file(path):
    """File digest: path + size + mtime. Swapping the DEM must be detectable."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    st = p.stat()
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


def _diff_params(old, new):
    """List the differing keys of two run_params so an error message says what changed."""
    keys = sorted(set(old) | set(new))
    out = []
    for k in keys:
        a, b = old.get(k, "<missing>"), new.get(k, "<missing>")
        if a != b:
            sa, sb = str(a), str(b)
            if len(sa) > 90:
                sa = sa[:87] + "..."
            if len(sb) > 90:
                sb = sb[:87] + "..."
            out.append(f"{k}: {sa} -> {sb}")
    return "; ".join(out) if out else "(same keys and values but different hash - check for unserialisable objects)"


def _digest_windrose(wr):
    """Wind-rose digest: 12-sector A/k/freq + height. Numbers drift when the upstream dataset is updated; it must be detectable."""
    if not wr:
        return None
    keys = ("sector_freq_pct", "sector_weibull_A", "sector_weibull_k")
    d = {k: [round(float(x), 4) for x in wr[k]] for k in keys if k in wr}
    for k in ("height_used_m", "z0_used", "dominant_sector_deg"):
        if k in wr:
            d[k] = wr[k]
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def _sample_raster_at_points(raster, transform, points_gdf):
    """Sample a WGS84 raster on the DEM grid at the turbine positions; returns an (N,) ndarray (NaN out of bounds)."""
    pts = points_gdf.to_crs(4326)
    return sample_raster(raster, transform, pts.geometry.x.values, pts.geometry.y.values)


def build_site_suitability_table(
    points_gdf, wfm, windrose, elev, dem_transform, azimuth_deg,
    air_density, rotor_diameter_m, turbulence_class="B", v_rated=None, v_out=25.0,
    wohler_exponent=10, ct_applied_upstream=None, wind_speed_ms=None, wfm_iec=None,
    slope_deg_raster=None,
):
    """Per-turbine site-suitability input table.

    A layout that only lists coordinates, a spacing status and one AEP number does
    not answer "can this turbine stand here". After positions are fixed, wind
    conditions should be given per turbine; this table collects them.

    :return: pandas.DataFrame, one row per turbine. Columns include:
        wtg_id / lon / lat / elev_m / air_density_kg_m3 / weibull_A_mean / weibull_k_mean /
        mean_ws_ms / effective_ti_max / effective_ti_at_ws / design_ti / ti_margin /
        ti_verdict / inflow_angle_deg / inflow_angle_flag / aep_net_gwh
    :param wind_speed_ms: optional per-turbine annual mean wind speed (sampled from an
        external raster); None leaves the column empty
    """
    import pandas as pd

    x = points_gdf.geometry.x.values
    y = points_gdf.geometry.y.values
    sim_res = wfm(x, y)

    wgs = points_gdf.to_crs(4326)
    lon, lat = wgs.geometry.x.values, wgs.geometry.y.values
    elev_at = _sample_raster_at_points(elev, dem_transform, points_gdf)
    slope_at = (_sample_raster_at_points(slope_deg_raster, dem_transform, points_gdf)
                if slope_deg_raster is not None else None)

    # Per-turbine air density: with large elevation differences it should be
    # computed per turbine. AEP still uses the zone-representative density (per-
    # turbine power curves would need a multi-turbine-type model), but the table
    # lists each turbine and the caller warns when the spread is large - the
    # "one density for the zone" simplification must not hide in the results.
    dens = air_density_spread(elev_at)

    # Inflow angle: given per sector, omni value as the weighted mean; a site
    # criterion usually takes the maximum over all directions. Both are computed.
    pts_lonlat = np.column_stack([lon, lat])
    inflow_main = inflow_angle_deg(
        elev, dem_transform, pts_lonlat, azimuth_deg, rotor_diameter_m=rotor_diameter_m)
    try:
        omni = omni_inflow_angle_deg(
            elev, dem_transform, pts_lonlat, windrose, weighting="energy",
            rotor_diameter_m=rotor_diameter_m)
        inflow_omni, inflow_max = omni["omni"], omni["max_abs_sector"]
        # Sectors that fail sampling are silently absorbed by np.nanmax; add a
        # valid-sector count so a row's "max over sectors" can be judged.
        inflow_n_valid = np.sum(np.isfinite(omni["per_sector"]), axis=1)
        inflow_n_sectors = omni["per_sector"].shape[1]
    except Exception as e:   # noqa: BLE001 - a failing omni computation must not kill the pipeline
        print(f"  [WARN] omni inflow angle failed ({type(e).__name__}: {e}); only the main-direction value is given")
        inflow_omni = inflow_max = np.full(len(pts_lonlat), np.nan)
        inflow_n_valid = np.zeros(len(pts_lonlat), dtype=int)
        inflow_n_sectors = 0
    # the verdict uses the strictest one (max over all directions)
    inflow = inflow_max

    rows = []
    ti_check = None
    # The IEC check uses the caller's wfm_iec (ambient TI already includes C_CT);
    # without it, fall back to the AEP model.
    wfm_for_iec = wfm_iec if wfm_iec is not None else wfm
    if getattr(wfm_for_iec, "turbulenceModel", None) is not None:
        sim_iec = sim_res if wfm_for_iec is wfm else wfm_for_iec(x, y)
        ti_check = check_iec_turbulence_suitability(
            sim_iec, turbulence_class=turbulence_class, v_rated=v_rated, v_out=v_out,
            wohler_exponent=wohler_exponent, ct_applied_upstream=ct_applied_upstream,
        )

    aep_per_wt = np.atleast_1d(sim_res.aep().sum(["wd", "ws"]).values)
    A = np.mean(windrose["sector_weibull_A"])
    k = np.mean(windrose["sector_weibull_k"])

    for i, wtg_id in enumerate(points_gdf["wtg_id"].values):
        row = {
            "wtg_id": wtg_id,
            "lon": float(lon[i]), "lat": float(lat[i]),
            "elev_m": float(elev_at[i]) if not np.isnan(elev_at[i]) else None,
            "slope_deg": (round(float(slope_at[i]), 2)
                          if slope_at is not None and not np.isnan(slope_at[i]) else None),
            "air_density_kg_m3": (float(dens["per_turbine"][i])
                                  if dens["per_turbine"][i] == dens["per_turbine"][i] else None),
            "air_density_zone_used_for_aep_kg_m3": air_density,
            "weibull_A_mean": float(A), "weibull_k_mean": float(k),
            "mean_ws_ms": float(wind_speed_ms[i]) if wind_speed_ms is not None else None,
            "inflow_angle_main_dir_deg": round(float(inflow_main[i]), 2),
            "inflow_angle_omni_energy_deg": (round(float(inflow_omni[i]), 2)
                                             if inflow_omni[i] == inflow_omni[i] else None),
            "inflow_angle_deg": (round(float(inflow[i]), 2)
                                 if inflow[i] == inflow[i] else None),
            # Three states, not two: a sampling failure (NaN) must not be judged
            # "OK". Treating NaN as OK would give the most favourable verdict to the
            # positions that could not be computed, and a zero flag count could not
            # tell "all fine" from "none computed". ti_verdict treats NaN as FAIL
            # (conservative side); the two must point the same way.
            "inflow_n_valid_sectors": int(inflow_n_valid[i]),
            "inflow_n_sectors": int(inflow_n_sectors),
            "inflow_angle_flag": (
                "UNKNOWN" if inflow[i] != inflow[i]
                else ("EXCEEDS_8DEG_SCREEN" if abs(inflow[i]) > IEC_MAX_FLOW_INCLINATION_DEG
                      else "OK")
            ),
            "aep_net_gwh": float(aep_per_wt[i]),
        }
        if ti_check is not None:
            c = ti_check["per_turbine"][i]
            row.update({
                "effective_ti_max": round(c["max_effective_ti"], 4),
                "effective_ti_at_ws": c["at_ws"],
                "design_ti": round(c["design_ti_at_that_ws"], 4),
                "ti_margin": round(c["margin"], 4),
                "ti_verdict": c["verdict"],
                # The full IEC 61400-1 eq. (35) criterion is sigma_1 >= I_eff V +
                # 1.28 sigma_sigma_hat. This pipeline has no sigma_sigma_hat
                # (needs a measured turbulence series), so the verdict is looser
                # than the standard. The fact is written into every row, not just
                # the docs: a CSV gets copied into reports and mails, docs do not.
                "ti_verdict_basis": "IEC61400-1_eq35_without_1.28sigma_hat_term",
                "ti_verdict_is_lenient_vs_iec": True,
            })
        else:
            row.update({"effective_ti_max": None, "effective_ti_at_ws": None,
                        "design_ti": None, "ti_margin": None, "ti_verdict": "NOT_CHECKED",
                        "ti_verdict_basis": None, "ti_verdict_is_lenient_vs_iec": None})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["air_density_spread"] = {
        k: v for k, v in dens.items() if k != "per_turbine"}
    df.attrs["ti_check_meta"] = (
        {kk: vv for kk, vv in ti_check.items() if kk != "per_turbine"} if ti_check else None
    )
    return df


def run_pipeline(
    zones_gdf,
    dem_path,
    usable_land_fn,
    turbine,
    windrose_fn,
    n_wt_fn,
    out_dir,
    id_col="id",
    exclusion_gdfs_fn=None,
    rotor_diameter_m=None,
    flat_threshold_m=100.0,
    downwind_d=5.0,
    crosswind_d=3.0,
    optimizer_min_spacing_d=None,
    mountain_min_spacing_d=3.0,
    buildability_slope_deg=17.0,
    rix_threshold_pct=5.0,
    valley_tpi_radius_m=500.0,
    valley_tpi_std_threshold=-1.0,
    search_window_d=1.0,
    mountain_priority_raster=None,
    wake_deficit_model=None,
    ti=0.10,
    seed=42,
    turbine_factory=None,
    mean_temp_c=None,
    use_smart_start=True,
    turbulence_model="default",
    turbulence_class="B",
    v_rated=None,
    v_out=25.0,
    wohler_exponent=10,
    ct_correction=1.0,
    wind_speed_raster=None,
    min_wind_speed_ms=None,
    target_capacity_density_mw_km2=None,
    turbine_rated_mw=None,
    wake_model_sensitivity_check=True,
    shear_alpha=None,
    shear_alpha_raster=None,
    offshore=False,
    check_callbacks_on_skip=False,
    mountain_threshold_m=300.0,
    rix_gentle_pct=1.0,
    rix_max_pixel_size_m=None,
    min_dem_coverage_pct=95.0,
):
    """Run the full pipeline per candidate zone: terrain classification -> flat branch
    (AEP optimisation) or mountain branch (terrain-constrained layout) -> directional-
    ellipse spacing check. Outputs go to out_dir/{zone_id}_turbines.gpkg +
    {zone_id}_pipeline_summary.json; a zone that already has a summary is skipped
    (idempotent, safe to re-run) after a parameter-fingerprint check.

    All numeric defaults below are **example defaults**, not calibrated
    recommendations; pass values that suit your own project.

    :param zones_gdf: geopandas GeoDataFrame, **WGS84 (EPSG:4326)**, candidate zones;
        must contain the unique-id column named by id_col
    :param dem_path: DEM GeoTIFF path (WGS84 degrees), used for terrain classification
        and the mountain branch
    :param usable_land_fn: (zone_row) -> geopandas.GeoDataFrame in a projected (metre)
        CRS: the usable land of that zone. Supplied by the caller; setback / road
        clipping is project-specific upstream logic and outside this pipeline.
    :param turbine: a constructed py_wake WindTurbine instance (e.g. GenericWindTurbine)
        used by the flat branch; if ``rotor_diameter_m`` is not given it is read from
        ``turbine.diameter()``
    :param windrose_fn: (zone_id) -> dict with sector_freq_pct (12 values, %) /
        sector_weibull_A (12) / sector_weibull_k (12) / dominant_sector_deg. Optional
        height_used_m (height the wind rose refers to). The flat branch builds the site
        from it; both branches use the energy-weighted dominant direction for the
        final directional check.
    :param n_wt_fn: (zone_row) -> int, target turbine count of the flat branch (TOPFARM
        needs a fixed count and does not optimise the number); the mountain branch does
        not call it (local-maxima selection decides the number of candidates)
    :param out_dir: output directory
    :param id_col: unique-id column name in zones_gdf
    :param exclusion_gdfs_fn: optional (zone_row) -> list[GeoDataFrame], exclusion zones
        of the flat branch (e.g. setback areas not already subtracted from usable land)
    :param rotor_diameter_m: rotor diameter, default read from turbine.diameter()
    :param flat_threshold_m: relief threshold of the terrain-classification routing, default 100 m
    :param downwind_d/crosswind_d: directional spacing-check criterion, default 5D/3D
    :param optimizer_min_spacing_d: circular minimum-spacing multiple used during the flat
        optimisation, default = downwind_d. NOTE: this makes the directional ellipse check
        mathematically always true (see the tautology warning below); lower it towards
        crosswind_d for the check to carry information.
    :param mountain_min_spacing_d: minimum-spacing multiple of the mountain branch, default 3.0
    :param buildability_slope_deg: per-pixel slope hard exclusion of the mountain branch
        (constructability), default 17 deg (example default, no source). Its meaning
        differs from the RIX critical slope and from the IEC inflow limit; they are not
        interchangeable (see terrainlayout).
    :param rix_threshold_pct: RIX threshold (%) of the terrain classification, default 5.
        Above it a zone goes to the mountain branch even with a moderate relief.
    :param mountain_threshold_m: rolling / mountainous boundary of the three severity
        tiers; affects report wording only
    :param min_dem_coverage_pct: minimum share of a zone's area inside the DEM extent
        (%), default 95. Below it a warning is issued.
    :param rix_max_pixel_size_m: coarsest DEM pixel (metres) at which RIX may overturn a
        relief verdict. None uses ``terrainclassify.RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M``.
        RIX collapses towards 0 as the DEM gets coarser, so a low RIX on a coarse DEM
        does not prove the terrain is gentle.
    :param rix_gentle_pct: RIX below this may **overturn a relief exceedance** to flat
        (the lower edge of the three-tier routing)
    :param valley_tpi_radius_m/valley_tpi_std_threshold/search_window_d: remaining
        mountain-branch parameters, passed to terrainlayout
    :param mountain_priority_raster: optional raster on the DEM grid (e.g. a wind-speed
        grid), the ranking basis of mountain-branch candidates. **Falls back to
        ``wind_speed_raster`` if not given**, then to elevation; elevation is not wind
        energy density, so picking purely by height selects "high but not windy" spots.
    :param wake_deficit_model: wake model. Default ``recommended_wake_model(offshore)``:
        onshore ``TurboGaussianDeficit(A=WAKE_CONSTANTS['turbopark_A_onshore'])``, an
        **example default from a single reported tuning, calibrate yourself** (see
        ``layoutoptimize.WAKE_CONSTANTS``); not the py_wake library default A=0.04.
    :param offshore: whether the site is offshore, selects the wake constant
    :param check_callbacks_on_skip: on an idempotent skip, also re-check ``windrose_fn``.
        Default False because the callback may do network I/O and the skip path should
        not; the price is that **a changed wind rose is not detected automatically**,
        only a digest is stored in the summary for manual comparison.
    :param ti: turbulence intensity, py_wake generic default 0.10 if not measured
    :param seed: random seed of the flat-branch initial layout
    :param turbine_factory: optional ``(air_density) -> py_wake WindTurbine``. When given,
        the turbine is rebuilt **per zone** with the density of the zone's representative
        elevation, instead of the sea-level 1.225 kg/m^3. Without it the turbine is used
        as given and the result carries ``air_density_applied=False``; **at high
        elevation the AEP is then systematically high**, and the report must say so.
        Typical use::

            from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
            turbine_factory = lambda rho: GenericWindTurbine(
                name="G3MW", diameter=126, hub_height=90, power_norm=3000,
                turbulence_intensity=0.10, air_density=rho)
    :param mean_temp_c: annual mean site temperature (deg C) for the air density. None
        uses the ISA profile; hot sites are less dense in reality, pass a measurement.
    :param use_smart_start: whether the flat branch uses TOPFARM smart_start, default True
    :param turbulence_model: py_wake turbulence model instance. Default string "default"
        -> STF2017TurbulenceModel(), used for the per-turbine effective turbulence and
        suitability check. None does not attach one - **then there is no wake-added
        turbulence and no suitability check**. Attaching it changes the AEP slightly.
    :param turbulence_class: turbine class 'A+'|'A'|'B'|'C', looks up I_ref
    :param v_rated: rated wind speed m/s; needed for the 0.6*V_r ~ V_out assessment band
    :param v_out: cut-out wind speed m/s, default 25
    :param wohler_exponent: material Woehler exponent m, default 10 (blade composites);
        steel structures typically use 4
    :param ct_correction: complex-terrain turbulence structure correction C_CT. Default
        1.0; the mountain branch automatically uses 1.15 (value for the case with no
        measured turbulence components, unverified; see siteconditions)
    :param wind_speed_raster: optional wind-speed raster on the DEM grid. Together with
        ``min_wind_speed_ms`` it drops low-wind positions.
    :param min_wind_speed_ms: minimum annual mean wind speed m/s for a position to be kept.
        After dropping, the **AEP is recomputed** - the count changed so the wake
        interaction changed and the old number must not be reused.
    :param target_capacity_density_mw_km2: target installed-capacity density (MW/km^2).
        When given, the usable land is first tightened to a compact footprint of
        ``n_wt x turbine_rated_mw / density`` and then optimised. **Without it, pure AEP
        maximisation spreads the turbines over the whole candidate zone**, which can be
        far larger than the area the turbines need.
    :param turbine_rated_mw: rated capacity per turbine, MW, used with the parameter above;
        without it the tightening is skipped
    :param wake_model_sensitivity_check: per zone, additionally run other wake models and
        write all wake losses to the summary. The same coordinates with a different model
        can give very different wake losses, **so one number should not appear alone in a
        deliverable**. Default True; pass False to save time on large zones.
    :param shear_alpha: wind-shear exponent alpha (scalar). When given, the wind resource
        is scaled to hub height by the power law and recorded; without it and without
        ``shear_alpha_raster`` the result records ``hub_height_shear_applied=False``.
    :param shear_alpha_raster: alpha raster on the DEM grid, **takes precedence over the
        scalar**. See ``siteconditions.shear_alpha_from_two_heights()`` for deriving it
        from a product's own height layers. The zone mean at its geometry is used.
    :return: list[dict], one per candidate zone; fields are built in the function body
    """
    if zones_gdf.crs is None or zones_gdf.crs.to_epsg() != 4326:
        raise ValueError("zones_gdf must be in WGS84 (EPSG:4326) - call to_crs(4326) first")

    if rotor_diameter_m is None:
        rotor_diameter_m = float(turbine.diameter())
    if optimizer_min_spacing_d is None:
        optimizer_min_spacing_d = downwind_d
    if wake_deficit_model is None:
        # Default to the **onshore** example constant. py_wake's TurboGaussianDeficit
        # default A=0.04 is an offshore value and would over-estimate onshore wake loss.
        wake_deficit_model, _wake_note = recommended_wake_model(offshore=offshore)
        print(f"  wake model (default): {_wake_note}")
    if turbulence_model == "default":
        from py_wake.turbulence_models import STF2017TurbulenceModel
        turbulence_model = STF2017TurbulenceModel()

    # Ranking basis of the mountain branch: explicit > wind-speed raster > elevation (fallback)
    mountain_priority = mountain_priority_raster
    if mountain_priority is None and wind_speed_raster is not None:
        mountain_priority = wind_speed_raster

    # Parameter fingerprint: the idempotent skip only looks at whether the file
    # exists, so re-running after changing turbine / spacing / turbulence class would
    # silently return the OLD result. The parameters that affect the result are
    # hashed into a fingerprint stored in the summary; before a skip it is compared
    # and a mismatch is an error so a person decides - neither silent recompute nor
    # silent skip.
    run_params = {
        "rotor_diameter_m": rotor_diameter_m, "flat_threshold_m": flat_threshold_m,
        "downwind_d": downwind_d, "crosswind_d": crosswind_d,
        "optimizer_min_spacing_d": optimizer_min_spacing_d,
        "mountain_min_spacing_d": mountain_min_spacing_d,
        "buildability_slope_deg": buildability_slope_deg,
        "rix_threshold_pct": rix_threshold_pct,
        "valley_tpi_radius_m": valley_tpi_radius_m,
        "valley_tpi_std_threshold": valley_tpi_std_threshold,
        "search_window_d": search_window_d, "ti": ti, "seed": seed,
        "turbulence_class": turbulence_class, "v_rated": v_rated, "v_out": v_out,
        "wohler_exponent": wohler_exponent, "ct_correction": ct_correction,
        "min_wind_speed_ms": min_wind_speed_ms,
        "target_capacity_density_mw_km2": target_capacity_density_mw_km2,
        "turbine_rated_mw": turbine_rated_mw,
        "turbine_factory_used": turbine_factory is not None,
        "wake_deficit_model": _digest_model(wake_deficit_model),
        "turbulence_model": _digest_model(turbulence_model),
        "dem": _digest_file(dem_path),
        "mean_temp_c": mean_temp_c,
        "offshore": offshore,
        "use_smart_start": use_smart_start,
        "mountain_threshold_m": mountain_threshold_m,
        "rix_gentle_pct": rix_gentle_pct,
        "rix_max_pixel_size_m": rix_max_pixel_size_m,
        "min_dem_coverage_pct": min_dem_coverage_pct,
        "shear_alpha": shear_alpha,
        "wake_model_sensitivity_check": wake_model_sensitivity_check,
        "wind_speed_raster": _digest_raster(wind_speed_raster, "wind_speed"),
        "shear_alpha_raster": _digest_raster(shear_alpha_raster, "shear_alpha"),
        "mountain_priority_raster": _digest_raster(mountain_priority_raster, "mountain_priority"),
        # Turbine: key attributes read from the instance; with turbine_factory it is
        # rebuilt per zone, so the base turbine is recorded here
        "turbine": {
            "class": type(turbine).__name__,
            "diameter_m": float(turbine.diameter()),
            "hub_height_m": float(turbine.hub_height()),
        },
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(run_params, sort_keys=True, default=str).encode()).hexdigest()[:16]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    elev, dem_transform = load_dem(dem_path)
    _lat_c = dem_transform.f + dem_transform.e * elev.shape[0] / 2
    _px = max(dem_pixel_size_m(dem_transform, _lat_c))
    _px_limit = (rix_max_pixel_size_m if rix_max_pixel_size_m is not None
                 else RIX_MAX_PIXEL_SIZE_FOR_OVERTURN_M)
    print(f"DEM pixel about {_px:.0f} m")
    if _px > _px_limit:
        print(f"  [WARN] DEM pixel {_px:.0f} m is coarser than {_px_limit:.0f} m - RIX is "
              "systematically under-estimated, so a near-zero RIX is **not allowed** to "
              "overturn the relief verdict this run. To use the flat-overturn tier, use "
              "a finer DEM.")
    print("Computing global slope + TPI valley raster (shared by the mountain branch, computed once)...")
    slope_deg = compute_slope(elev, dem_transform)
    valley_mask = compute_valley_mask(
        elev, dem_transform, radius_m=valley_tpi_radius_m, tpi_std_threshold=valley_tpi_std_threshold
    )

    results = []
    for _, row in zones_gdf.iterrows():
        zone_id = row[id_col]
        out_gpkg = out_dir / f"{zone_id}_turbines.gpkg"
        summary_path = out_dir / f"{zone_id}_pipeline_summary.json"
        if summary_path.exists():
            prev = json.loads(summary_path.read_text(encoding="utf-8"))
            prev_fp = prev.get("run_fingerprint")
            if prev_fp is not None and prev_fp != run_fingerprint:
                # say **which** parameter changed, not just two hashes
                diff = _diff_params(prev.get("run_params") or {}, run_params)
                raise RuntimeError(
                    f"{zone_id} already has a result, but **its parameters differ from this run** "
                    f"(existing fingerprint {prev_fp}, this run {run_fingerprint}).\n"
                    f"Differences: {diff}\n"
                    "An idempotent skip would hand you a result computed with the old "
                    f"parameters. Decide explicitly: delete {summary_path} to recompute, "
                    "or use a new out_dir."
                )
            # Per-zone inputs resolved from callbacks (n_wt / wind rose) are not in
            # the global parameters but decide the result directly.
            if check_callbacks_on_skip:
                _wr = _digest_windrose(windrose_fn(zone_id))
                if prev.get("windrose_digest") and prev["windrose_digest"] != _wr:
                    raise RuntimeError(
                        f"{zone_id} already has a result, but **the wind rose changed** "
                        f"(existing {prev['windrose_digest']}, this run {_wr}). "
                        f"Delete {summary_path} to recompute, or use a new out_dir.")
            elif prev.get("windrose_digest"):
                print(f"       wind-rose digest {prev['windrose_digest']} (not re-checked; "
                      "pass check_callbacks_on_skip=True to compare the callback too)")
            print(f"[SKIP] {zone_id} already has a result (parameter fingerprint matches {run_fingerprint})")
            results.append(prev)
            continue

        terrain = classify_zone(row.geometry, elev, dem_transform, flat_threshold_m=flat_threshold_m,
                                mountain_threshold_m=mountain_threshold_m,
                                rix_gentle_pct=rix_gentle_pct,
                                slope_deg=slope_deg, rix_threshold_pct=rix_threshold_pct,
                                min_dem_coverage_pct=min_dem_coverage_pct,
                                **({"rix_max_pixel_size_m": rix_max_pixel_size_m}
                                   if rix_max_pixel_size_m is not None else {}))
        rix_txt = f" RIX={terrain['rix_pct']:.1f}%" if terrain.get("rix_pct") is not None else ""
        print(f"{zone_id}: relief={terrain['relief_m']}m{rix_txt} tier={terrain['tier']} "
              f"route={terrain['route']} ({terrain.get('route_reason', '')})")

        usable_land = usable_land_fn(row)
        exclusion_gdfs = exclusion_gdfs_fn(row) if exclusion_gdfs_fn else None
        windrose = windrose_fn(zone_id)
        # The spacing check uses the **energy**-dominant direction, not the wind
        # rose's own dominant_sector_deg (that one is frequency-dominant: a sector
        # that blows often but weakly is not the largest wake risk).
        azimuth = dominant_energy_azimuth(windrose)
        azimuth_freq = windrose.get("dominant_sector_deg")

        # Air density: correct the turbine power curve with the zone's representative elevation
        zone_elev_m = terrain.get("elev_mean_m")
        air_density = (
            air_density_from_elevation(zone_elev_m, temp_c=mean_temp_c)
            if zone_elev_m is not None else None
        )
        if turbine_factory is not None and air_density is not None:
            zone_turbine = turbine_factory(air_density)
            air_density_applied = True
        else:
            zone_turbine = turbine
            air_density_applied = False
            if air_density is not None:
                print(f"  [WARN] no turbine_factory: AEP still uses the turbine's own air density "
                      f"(the zone's representative elevation {zone_elev_m:.0f} m corresponds to "
                      f"{air_density:.3f} kg/m3, {(1 - air_density / 1.225) * 100:.1f}% below 1.225)")

        # specific-power plausibility gate
        if turbine_rated_mw:
            check_specific_power(turbine_rated_mw * 1000, rotor_diameter_m,
                                 label=f"{zone_id} ")

        # Hub-height shear conversion: without it, a wind rose whose height differs
        # from hub height would pass with no notice.
        wr_A = list(windrose["sector_weibull_A"])
        shear_applied = False
        alpha_used = shear_alpha
        alpha_source = "scalar" if shear_alpha is not None else None
        wr_height = windrose.get("height_used_m")
        hub_h = float(zone_turbine.hub_height())

        # alpha: raster first (mean within the zone geometry), scalar second
        if shear_alpha_raster is not None:
            st = masked_stats(shear_alpha_raster, dem_transform, row.geometry)
            if st["n_pixels"]:
                alpha_used = st["mean"]
                alpha_source = "raster_zone_mean"

        if wr_height is None:
            print("  [WARN] the wind rose has no height_used_m, so whether a shear conversion "
                  "is needed cannot be judged - AEP uses the wind rose at its original height; "
                  "if that differs from hub height the result is systematically biased")
        elif alpha_used is None:
            if abs(wr_height - hub_h) > 1:
                print(f"  [WARN] wind-rose height {wr_height:.0f} m != hub height {hub_h:.0f} m "
                      "but neither shear_alpha nor shear_alpha_raster was given: **no shear conversion done**")
        elif abs(wr_height - hub_h) > 1:
            lo, hi = PLAUSIBLE_SHEAR_ALPHA
            if not (lo <= alpha_used <= hi):
                print(f"  [WARN] alpha={alpha_used:.3f} is outside the plausible range {lo}-{hi}; no conversion")
            else:
                factor = (hub_h / float(wr_height)) ** float(alpha_used)
                wr_A = [a * factor for a in wr_A]
                shear_applied = True
                print(f"  shear conversion {wr_height:.0f} m -> {hub_h:.0f} m "
                      f"(alpha={alpha_used:.4f}, source={alpha_source}), Weibull A x{factor:.4f}")

        site = UniformWeibullSite(
            p_wd=[f / 100 for f in windrose["sector_freq_pct"]],
            a=wr_A, k=windrose["sector_weibull_k"], ti=ti,
        )
        zone_ct = ct_correction
        if terrain["route"] == "mountain_branch" and ct_correction == 1.0:
            zone_ct = 1.15   # complex-terrain turbulence structure correction (see siteconditions)
        wfm = PropagateDownwind(site, zone_turbine, wake_deficitModel=wake_deficit_model,
                                turbulenceModel=turbulence_model)
        # The IEC turbulence check uses a **separate** wfm: the ambient TI is
        # multiplied by the complex-terrain correction C_CT and py_wake then combines
        # the wake-added turbulence. C_CT must not be multiplied onto the combined
        # I_eff: the m-th power mean is homogeneous, so that merely scales the whole
        # criterion. A separate model rather than editing ``site`` because C_CT is a
        # **load assessment** correction and must not change AEP (ambient TI affects
        # wake expansion).
        if zone_ct and zone_ct != 1.0:
            # Use the same (shear-converted) wr_A as the AEP site: the correction
            # only changes ti, it must not swap the wind speed distribution.
            site_iec = UniformWeibullSite(
                p_wd=[f / 100 for f in windrose["sector_freq_pct"]],
                a=wr_A, k=windrose["sector_weibull_k"],
                ti=ti * zone_ct,
            )
            wfm_iec = PropagateDownwind(site_iec, zone_turbine,
                                        wake_deficitModel=wake_deficit_model,
                                        turbulenceModel=turbulence_model)
        else:
            wfm_iec = wfm
        site_conditions = {
            "elev_mean_m": zone_elev_m,
            # Zone-representative density: what the AEP was computed with. Per-turbine
            # densities are in the site_suitability table.
            "air_density_kg_m3": air_density,
            "air_density_applied": air_density_applied,
            "mean_temp_c": mean_temp_c,
            "ct_correction": zone_ct,
            "windrose_height_m": wr_height,
            "hub_height_m": float(zone_turbine.hub_height()),
            "shear_alpha": alpha_used,
            "shear_alpha_source": alpha_source,
            # explicit declaration: the IEC check and the AEP use the same wind climate
            "iec_uses_same_wind_climate_as_aep": True,
            "hub_height_shear_applied": shear_applied,
            "specific_power_w_m2": (
                round(check_specific_power(turbine_rated_mw * 1000, rotor_diameter_m)[0], 1)
                if turbine_rated_mw else None),
        }

        if terrain["route"] == "flat_branch":
            n_wt = n_wt_fn(row)
            footprint_info = None
            if target_capacity_density_mw_km2 and turbine_rated_mw:
                target_area = n_wt * turbine_rated_mw / target_capacity_density_mw_km2
                usable_land, footprint_info = compact_footprint(
                    usable_land, target_area_km2=target_area,
                    priority_raster=(wind_speed_raster if wind_speed_raster is not None
                                     else mountain_priority),
                    dem_transform=dem_transform,
                )
                if footprint_info.get("shrunk"):
                    print(f"  footprint tightened {footprint_info['original_area_km2']:.0f} -> "
                          f"{footprint_info['final_area_km2']:.1f} km2"
                          f" (target density {target_capacity_density_mw_km2} MW/km2)")
            opt = optimize_layout(
                wfm, n_wt=n_wt, usable_land_gdf=usable_land, exclusion_gdfs=exclusion_gdfs,
                rotor_diameter_m=rotor_diameter_m, min_spacing_d=optimizer_min_spacing_d, seed=seed,
                id_prefix=str(zone_id).upper(), use_smart_start=use_smart_start,
            )
            points_gdf = opt["points_gdf"]
            branch_info = {
                "aep_before_gwh": opt["aep_before_gwh"], "aep_after_gwh": opt["aep_after_gwh"],
                "aep_vs_initial_pct": opt["aep_vs_initial_pct"], "optimizer_feasible": opt["feasible"],
                "aep_gross_gwh": opt["aep_gross_gwh"], "aep_net_gwh": opt["aep_net_gwh"],
                "wake_loss_pct": opt["wake_loss_pct"],
                "init_method": opt["init_method"],
                # persisted so the spacing evidence in the deliverable is not only
                # the (possibly tautological) ellipse check
                "min_spacing_actual_m": opt["min_spacing_actual_m"],
                "feasible": opt["feasible"],
                "boundary_snap_corrections": opt["boundary_snap_corrections"],
                "footprint": footprint_info,
                "capacity_density_mw_km2": (
                    capacity_density_mw_km2(n_wt, turbine_rated_mw,
                                            usable_land.geometry.area.sum() / 1e6)
                    if turbine_rated_mw else None
                ),
            }
        else:
            opt = optimize_layout_mountain(
                usable_land, elev, slope_deg, valley_mask, dem_transform,
                rotor_diameter_m=rotor_diameter_m, min_spacing_d=mountain_min_spacing_d,
                buildability_slope_deg=buildability_slope_deg, search_window_d=search_window_d,
                priority_raster=mountain_priority, id_prefix=str(zone_id).upper(),
                # Select points with the directional ellipse criterion. A circular
                # approximation with min_spacing_d (default downwind_d=5D) is the
                # ellipse's **largest** required distance, which would make the
                # downstream check always true and over-constrain the crosswind
                # direction.
                spacing_ellipse=(downwind_d * rotor_diameter_m / 2,
                                 crosswind_d * rotor_diameter_m / 2, azimuth),
            )
            points_gdf = opt["points_gdf"]
            _mpts = opt["points_gdf"]
            if len(_mpts) > 1:
                _xy = np.column_stack([_mpts.geometry.x.values, _mpts.geometry.y.values])
                _d = np.hypot(_xy[:, None, 0] - _xy[None, :, 0],
                              _xy[:, None, 1] - _xy[None, :, 1])
                np.fill_diagonal(_d, np.inf)
                _min_sp = float(_d.min())
            else:
                _min_sp = None
            _area_km2 = float(usable_land.geometry.area.sum() / 1e6)
            branch_info = {
                "n_peak_candidates": opt["n_peak_candidates"],
                "min_spacing_actual_m": _min_sp,
                "usable_area_km2": round(_area_km2, 3),
                "capacity_density_mw_km2": (
                    capacity_density_mw_km2(len(_mpts), turbine_rated_mw, _area_km2)
                    if turbine_rated_mw and _area_km2 else None),
                "priority_source": opt["priority_source"],
                "buildability_slope_deg": opt["buildability_slope_deg"],
                # kept so that spacing_check_is_independent is computed correctly
                "spacing_criterion": opt.get("spacing_criterion"),
                "min_local_relief_m": opt.get("min_local_relief_m"),
            }
            # The mountain branch does not **optimise** AEP, but its candidates can
            # still be **evaluated**, so both branches expose aligned output fields.
            # This number uses a flat-terrain-assumption wake model: local speed-up
            # and lee effects are outside it, order of magnitude only.
            if len(points_gdf) > 0:
                energy = evaluate_layout(
                    wfm, points_gdf.geometry.x.values, points_gdf.geometry.y.values)
                branch_info.update(energy)
                branch_info["energy_model_caveat"] = (
                    "Engineering wake model with a flat-terrain flow assumption; in complex "
                    "terrain order-of-magnitude reference only, not a design basis")

        # Low-wind filter + recompute of energy after dropping
        ws_at_points = None
        filter_info = None
        if len(points_gdf) > 0 and wind_speed_raster is not None:
            ws_at_points = _sample_raster_at_points(wind_speed_raster, dem_transform, points_gdf)
            if min_wind_speed_ms is not None:
                keep = ws_at_points >= min_wind_speed_ms
                n_drop = int((~keep).sum())
                filter_info = {
                    "min_wind_speed_ms": min_wind_speed_ms,
                    "n_dropped": n_drop,
                    "dropped_wtg_ids": list(points_gdf["wtg_id"].values[~keep]),
                    "ws_range_ms": [float(np.nanmin(ws_at_points)), float(np.nanmax(ws_at_points))],
                }
                if n_drop:
                    print(f"  dropped {n_drop} position(s) below {min_wind_speed_ms} m/s "
                          f"(zone wind speed {filter_info['ws_range_ms'][0]:.2f}~"
                          f"{filter_info['ws_range_ms'][1]:.2f} m/s)")
                    points_gdf = points_gdf[keep].reset_index(drop=True)
                    ws_at_points = ws_at_points[keep]
                    if len(points_gdf) > 0:
                        # count changed => wake interaction changed => recompute
                        branch_info.update(evaluate_layout(
                            wfm, points_gdf.geometry.x.values, points_gdf.geometry.y.values))
                        branch_info["aep_recomputed_after_filter"] = True
                        # aep_before/after and the vs-initial percentage describe the
                        # "optimisation process" of the pre-filter count; after the
                        # filter there is no matching layout, so they are set to None
                        # with a reason instead of leaving an old number that looks usable.
                        for _stale in ("aep_before_gwh", "aep_after_gwh",
                                       "aep_vs_initial_pct"):
                            if _stale in branch_info:
                                branch_info[_stale] = None
                        branch_info["aep_optimization_metrics_invalidated_by_filter"] = (
                            "The low-wind filter changed the turbine count, so the "
                            "before/after optimisation comparison lost its reference; set to None")
                    else:
                        for kk in ("aep_gross_gwh", "aep_net_gwh", "wake_loss_pct"):
                            branch_info[kk] = 0.0 if kk != "wake_loss_pct" else None
                        branch_info["aep_recomputed_after_filter"] = True
        if filter_info is not None:
            branch_info["wind_speed_filter"] = filter_info

        # Consistency of the two wind-resource sources: AEP uses the wind rose, the
        # CSV mean_ws / mountain selection use wind_speed_raster. Their source and
        # height may differ; do not leave them side by side unmarked.
        if ws_at_points is not None and len(ws_at_points):
            # Per-sector Gamma(1+1/k), frequency-weighted (not a fixed k=2 factor)
            _A = np.asarray(windrose["sector_weibull_A"], dtype=float)
            _k = np.asarray(windrose["sector_weibull_k"], dtype=float)
            _fr = np.asarray(windrose["sector_freq_pct"], dtype=float)
            _sector_mean = _A * np.array([math.gamma(1 + 1 / ki) for ki in _k])
            wr_mean = float(np.average(_sector_mean, weights=_fr))
            ras_mean = float(np.nanmean(ws_at_points))
            diff_pct = (ras_mean / wr_mean - 1) * 100 if wr_mean else float("nan")
            branch_info["wind_source_consistency"] = {
                "windrose_mean_ws_ms": round(wr_mean, 3),
                "raster_mean_ws_ms": round(ras_mean, 3),
                "diff_pct": round(diff_pct, 1),
                "windrose_height_m": wr_height,
                "note": "AEP is computed from the wind rose; the raster is only used for selection / filtering / report columns",
            }
            if abs(diff_pct) > 15:  # example threshold, adjust to your data quality
                print(f"  [WARN] the two wind-resource sources differ by {diff_pct:+.1f}% "
                      f"(wind rose {wr_mean:.2f} vs raster {ras_mean:.2f} m/s). "
                      "AEP and mean_ws_ms in the CSV come from different sources; say so in the report")

        if len(points_gdf) == 0:
            print(f"  [WARN] {zone_id}: this branch produced 0 positions")
            result = {
                "zone_id": zone_id, "route": terrain["route"], "terrain": terrain,
                "n_turbines": 0, "branch_info": branch_info,
                "site_conditions": site_conditions, "spacing_clean": None,
                "windrose": windrose,
                "run_fingerprint": run_fingerprint, "run_params": run_params,
                "windrose_digest": _digest_windrose(windrose),
                "disclaimer": OUTPUT_DISCLAIMER,
            }
            summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        spacing = check_spacing(
            points_gdf, rotor_diameter_m=rotor_diameter_m, azimuth_deg=azimuth,
            downwind_d=downwind_d, crosswind_d=crosswind_d, id_col="wtg_id",
        )
        points_gdf = points_gdf.copy()
        points_gdf["status"] = points_gdf["wtg_id"].map(spacing["status"])

        # The flat branch has no terrain / constructability constraint:
        # buildability_slope_deg and valley_mask go only to the mountain branch.
        # TOPFARM constraints are polygons not rasters, so a slope constraint would
        # need vectorising (costly, fragile); this does a **post-check with an
        # explicit count**: it cannot block, but it must not be silent.
        slope_at_pts = _sample_raster_at_points(slope_deg, dem_transform, points_gdf)
        n_slope_exceed = int(np.nansum(slope_at_pts > buildability_slope_deg))
        in_valley = _sample_raster_at_points(
            valley_mask.astype(np.float32), dem_transform, points_gdf) > 0.5
        n_in_valley = int(np.nansum(in_valley))
        if terrain["route"] == "flat_branch" and (n_slope_exceed or n_in_valley):
            print(f"  [WARN] the flat-branch output has {n_slope_exceed} turbine(s) steeper than "
                  f"{buildability_slope_deg} deg and {n_in_valley} inside a TPI valley. "
                  "The flat branch has no hard terrain constraint; these need manual "
                  "confirmation or the mountain branch")

        # Per-turbine site-suitability input table
        suitability = build_site_suitability_table(
            points_gdf, wfm, windrose, elev, dem_transform, azimuth,
            air_density=air_density, rotor_diameter_m=rotor_diameter_m,
            turbulence_class=turbulence_class, v_rated=v_rated, v_out=v_out,
            wohler_exponent=wohler_exponent, ct_applied_upstream=zone_ct,
            wind_speed_ms=ws_at_points, wfm_iec=wfm_iec, slope_deg_raster=slope_deg,
        )
        suitability_csv = out_dir / f"{zone_id}_site_suitability.csv"
        suitability.to_csv(suitability_csv, index=False, encoding="utf-8-sig")
        if wake_model_sensitivity_check:
            sens = wake_model_sensitivity(site, zone_turbine,
                                          points_gdf.geometry.x.values,
                                          points_gdf.geometry.y.values)
            branch_info["wake_model_constants"] = dict(WAKE_CONSTANTS)
            branch_info["wake_model_sensitivity"] = {
                k: (round(v["wake_loss_pct"], 2) if "wake_loss_pct" in v else v)
                for k, v in sens.items()
            }
            losses = [v["wake_loss_pct"] for v in sens.values() if "wake_loss_pct" in v]
            if losses and max(losses) - min(losses) > 5:
                print(f"  [WARN] wake loss is highly sensitive to the model choice: "
                      f"{min(losses):.1f}%~{max(losses):.1f}% (across {len(losses)} models). "
                      "Do not report a single number")

        dens_meta = suitability.attrs.get("air_density_spread", {})
        if dens_meta.get("spread_pct", 0) > 2.0:  # example threshold
            print(f"  [WARN] {zone_id}: per-turbine air-density spread within the zone "
                  f"{dens_meta['spread_pct']:.1f}% ({dens_meta['min']:.3f}~{dens_meta['max']:.3f} kg/m3) "
                  "- with large elevation differences density should be computed per turbine; "
                  "the zone-representative value used for AEP is coarse")
        n_ti_fail = int((suitability["ti_verdict"] == "FAIL").sum())
        n_inflow_flag = int((suitability["inflow_angle_flag"] == "EXCEEDS_8DEG_SCREEN").sum())
        n_inflow_unknown = int((suitability["inflow_angle_flag"] == "UNKNOWN").sum())
        if n_inflow_unknown:
            print(f"  [WARN] {zone_id}: {n_inflow_unknown} turbine(s) have an inflow angle that "
                  "**could not be computed** (elevation sampling returned no value); they are "
                  "neither compliant nor exceeding and need manual confirmation")
        for col in ("effective_ti_max", "ti_verdict", "inflow_angle_deg",
                    "inflow_angle_omni_energy_deg", "inflow_angle_flag"):
            points_gdf[col] = suitability[col].values

        points_gdf.to_file(out_gpkg, layer="turbines_utm", driver="GPKG")
        points_gdf.to_crs(4326).to_file(out_gpkg, layer="turbines_wgs84", driver="GPKG")
        spacing["ellipses_gdf"].to_file(out_gpkg, layer="spacing_ellipses", driver="GPKG")

        # Report the check's independence honestly. It fails in two ways:
        #   (a) selection used the same ellipse criterion -> this step only verifies
        #       that the constraint was not broken afterwards;
        #   (b) selection used a circle radius >= downwind_d*D -> the ellipse
        #       criterion is **mathematically always true** and the check is void.
        # The flat-branch default optimizer_min_spacing_d = downwind_d = 5D falls in (b).
        _sel = branch_info.get("spacing_criterion") or ""
        _circ_d = (mountain_min_spacing_d if terrain["route"] == "mountain_branch"
                   else optimizer_min_spacing_d)
        _tautological = bool(_circ_d and _circ_d >= downwind_d)
        spacing_independent = (not _sel.startswith("directional_ellipse")
                               and not _tautological)
        # The largest centre distance the ellipse requires is downwind_d*D, so any
        # isotropic floor >= downwind_d hides the ellipse completely: the crosswind
        # density gain is lost and the check degenerates.
        if _tautological:
            print(f"  [WARN] selection circle radius {_circ_d}D >= downwind_d {downwind_d}D - the "
                  "directional-ellipse check is **mathematically always true**, so "
                  "spacing_clean=true carries no information. To make the check meaningful, "
                  f"lower it towards crosswind_d ({crosswind_d}D)")
        if (terrain["route"] == "mountain_branch"
                and mountain_min_spacing_d and mountain_min_spacing_d >= downwind_d):
            print(f"  [WARN] mountain_min_spacing_d={mountain_min_spacing_d}D >= "
                  f"downwind_d={downwind_d}D - the isotropic floor fully overrides the "
                  "directional-ellipse criterion; the closer crosswind spacing that would be "
                  "allowed is lost and the spacing check degenerates. To use the ellipse "
                  f"criterion lower mountain_min_spacing_d to crosswind_d ({crosswind_d}D) or set it to None")
        n_conflict = sum(1 for s in spacing["status"].values() if s == "CONFLICT")
        n_near = sum(1 for s in spacing["status"].values() if s == "NEAR")
        result = {
            "zone_id": zone_id, "route": terrain["route"], "terrain": terrain,
            "n_turbines": len(points_gdf), "branch_info": branch_info,
            "site_conditions": site_conditions,
            "dominant_wind_direction_deg": azimuth,
            "dominant_wind_direction_source": "energy_weighted",
            "dominant_freq_direction_deg": azimuth_freq,
            "n_spacing_conflict": n_conflict, "n_spacing_near": n_near,
            "spacing_clean": n_conflict == 0,
            "spacing_selection_criterion": branch_info.get("spacing_criterion"),
            "spacing_check_is_independent": spacing_independent,
            "spacing_check_tautological": _tautological,
            "spacing_check_note": (
                f"WARNING - **this check is void**: the isotropic circle radius used for "
                f"selection, {_circ_d}D, is >= downwind_d {downwind_d}D, and the largest centre "
                f"distance the ellipse criterion requires is exactly downwind_d*D, so a CONFLICT "
                "cannot occur. spacing_clean=true is an identity by construction, not a "
                "check result."
                if _tautological else
                "Selection already used the same ellipse criterion; this check verifies "
                "'was the constraint broken by later steps' (edge push-off / filtering / "
                "manual moves), not an independent discovery"
                if not spacing_independent else
                "Selection used an isotropic circular approximation with a radius below "
                "downwind_d*D; this check is independent of the selection criterion and valid"),
            "n_slope_exceed": n_slope_exceed,
            "n_in_valley": n_in_valley,
            "terrain_constraint_applied_in_selection": terrain["route"] == "mountain_branch",
            "n_iec_ti_fail": n_ti_fail,
            "n_inflow_angle_flag": n_inflow_flag,
            "n_inflow_angle_unknown": n_inflow_unknown,
            "iec_ti_check": suitability.attrs.get("ti_check_meta"),
            "air_density_spread": suitability.attrs.get("air_density_spread"),
            "output_gpkg": str(out_gpkg),
            "site_suitability_csv": str(suitability_csv),
            # The wind rose is persisted: the most important AEP input must be in the
            # deliverable, otherwise the result is neither auditable nor reproducible,
            # and numbers would drift silently when the upstream dataset is updated.
            "windrose": windrose,
            "run_fingerprint": run_fingerprint,
            "run_params": run_params,
            "windrose_digest": _digest_windrose(windrose),
            "disclaimer": OUTPUT_DISCLAIMER,
        }
        summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        # Always return the JSON round-tripped result: the skip branch returns
        # json.loads output, so a fresh computation returning in-memory objects
        # (tuples stay tuples, numpy scalars stay numpy) would differ in type.
        results.append(json.loads(summary_path.read_text(encoding="utf-8")))
        print(f"  -> {result['n_turbines']} turbines route={terrain['route']} "
              f"spacing CONFLICT={n_conflict}/NEAR={n_near} "
              f"IEC turbulence FAIL={n_ti_fail} inflow>8deg={n_inflow_flag}"
              + (f" inflow unknown={n_inflow_unknown}" if n_inflow_unknown else ""))

    return results
