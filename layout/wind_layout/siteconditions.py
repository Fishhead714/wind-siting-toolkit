"""Site-condition parameters: the inputs a per-turbine site suitability table needs.

Single source of truth: import this module instead of re-copying these formulas.

The module covers the quantities that can be computed from free data:
  - air_density_from_elevation()  -- air density (ISA profile, ideal-gas form)
  - effective_ti() / ntm_sigma1() / check_iec_turbulence_suitability()
                                  -- turbulence + IEC 61400-1 suitability criterion direction
  - shear_alpha_from_two_heights() / extrapolate_wind_speed()
                                  -- power-law shear exponent from multi-height data
  - inflow_angle_deg() / omni_inflow_angle_deg() / air_density_spread()
                                  -- inflow angle (per sector and omni-weighted),
                                     per-turbine air density

Deliberately NOT done (they need measurements or the normative text; nothing is
invented here): 50-year extreme wind speed, extreme environmental turbulence, and
a standard method for spatial extrapolation of shear.

## Provenance of the IEC constants

The constants below follow IEC 61400-1 Ed.3 (2005) as quoted; the package author
could not re-verify them against the normative text for this release and they are
marked "unverified" in THIRD_PARTY_NOTICES.md. Treat them as examples to be
checked by you:

  - Table 1: I_ref  A=0.16  B=0.14  C=0.12
  - Eq. (11): sigma_1 = I_ref (0.75 V_hub + b), b = 5.6 m/s (representative NTM value)
  - Eq. (D.1): I_eff(V_hub) = [ integral p(theta|V_hub) I^m(theta|V_hub) dtheta ]^(1/m),
               m = material Woehler exponent
  - Eq. (35): suitability criterion sigma_1 >= I_eff V_hub + 1.28 sigma_sigma_hat
  - inflow inclination: up to 8 deg, assumed constant with height
  - complex-terrain turbulence structure correction C_CT, 1.15 when no measurements exist

**Edition boundary**: all of it is Ed.3. Ed.4 (2019) adds an A+ class and some
constants may differ. Every constant here can be overridden by argument; for a
formal conclusion under Ed.4 check the normative text yourself.
"""
import math

import numpy as np

# ISA (International Standard Atmosphere) sea-level reference values
_ISA_P0_PA = 101325.0
_ISA_T0_K = 288.15
_ISA_LAPSE_K_PER_M = 0.0065
_ISA_G = 9.80665
_R_DRY_AIR = 287.058          # J/(kg K), specific gas constant of dry air
_M_AIR = 0.0289644            # kg/mol
_R_UNIVERSAL = 8.31447        # J/(mol K)


def air_density_from_elevation(elev_m, temp_c=None, humidity_correction=False):
    """Air density from elevation (optionally measured temperature), kg/m^3.

    Power curves are given at the standard density 1.225 kg/m^3 and a density
    deviation scales output power directly. At positions high above sea level the
    density is markedly lower (on the order of 1% per 100 m in the ISA); ignoring it over-estimates energy
    systematically, and the more mountainous the site the more the highest points
    are picked - exactly the positions that most need to be conservative. (py_wake's
    ``GenericWindTurbine(air_density=1.225)`` default is a sea-level value that is
    never overridden unless you do it.)

    :param elev_m: position elevation, metres (hub height is stricter, ground
        elevation is fine at screening accuracy)
    :param temp_c: annual mean temperature, deg C. None uses the ISA profile
        (15 C - 6.5 C/km), an engineering approximation. Real ground temperatures
        in warm climates are usually higher than ISA, so the true density is lower
        than this function returns; pass a measured annual mean when available.
    :param humidity_correction: not implemented (see below); True raises.
    :return: air density, kg/m^3

    ## Relation to the standards

    IEC 61400-12-1 gives the density formula with a humidity term and an
    alternative ideal-gas form rho = P / (R T). This function follows the
    alternative form with an ISA pressure profile; the dry-air assumption changes
    density by roughly 0.5% under normal onshore conditions. Without measured
    pressure the standards allow an ISA-based estimate (-0.65 C/100 m, sea-level
    1013.25 hPa). (Clause numbers not re-verified here.)

    Where site elevation differences are large the density should be computed
    **per turbine**, not as one representative value. ``air_density_spread`` gives
    the spread so you can decide whether one value is enough.
    """
    if humidity_correction:
        raise NotImplementedError(
            "Humidity correction is not implemented: it needs relative humidity and "
            "vapour pressure inputs, and its size (~0.5%) is far below the elevation "
            "term; left out on purpose to avoid a parameter without data behind it"
        )

    elev_m = float(elev_m)
    # ISA troposphere barometric formula (valid 0-11 km)
    pressure_pa = _ISA_P0_PA * (
        1 - _ISA_LAPSE_K_PER_M * elev_m / _ISA_T0_K
    ) ** (_ISA_G * _M_AIR / (_R_UNIVERSAL * _ISA_LAPSE_K_PER_M))

    if temp_c is None:
        temp_k = _ISA_T0_K - _ISA_LAPSE_K_PER_M * elev_m
    else:
        temp_k = float(temp_c) + 273.15

    return pressure_pa / (_R_DRY_AIR * temp_k)


def air_density_spread(elev_m_array, temp_c=None):
    """A batch of turbine elevations -> per-turbine air density and the spread within the batch.

    Answers "is one representative density enough": ``spread_pct`` is the relative
    difference between the largest and smallest density.

    :return: {'per_turbine': ndarray, 'min', 'max', 'mean', 'spread_pct'}
    """
    rho = np.array([air_density_from_elevation(e, temp_c) if e == e else np.nan
                    for e in np.asarray(elev_m_array, dtype=float)])
    lo, hi = np.nanmin(rho), np.nanmax(rho)
    return {"per_turbine": rho, "min": float(lo), "max": float(hi),
            "mean": float(np.nanmean(rho)),
            "spread_pct": float((hi / lo - 1) * 100) if lo > 0 else float("nan")}


def density_power_scaling(air_density, reference_density=1.225):
    """First-order scaling ratio of density on output power.

    For **explaining magnitude in a report only**. Do not use it to scale AEP: the
    correct way is to pass air_density to the py_wake turbine constructor so power
    and thrust curves are recomputed (the thrust coefficient is affected as well,
    which a first-order scaling cannot capture).
    """
    return air_density / reference_density


# ---------------------------------------------------------------------------
# IEC 61400-1 turbulence: constants, normal turbulence model, effective TI,
# suitability criterion
# ---------------------------------------------------------------------------

#: Reference turbulence intensity I_ref per turbulence class (expected value at
#: 15 m/s), IEC 61400-1 Table 1. A/B/C are Ed.3 values as quoted; **A+ is an
#: Ed.4 addition and the 0.18 here is unverified**; check both before drawing a
#: formal conclusion.
IEC_IREF = {"A+": 0.18, "A": 0.16, "B": 0.14, "C": 0.12}

#: Default complex-terrain turbulence structure correction when no turbulence
#: components are measured (Ed.3 clause 11.9, footnote; unverified here).
IEC_CCT_COMPLEX_TERRAIN = 1.15


def ntm_sigma1(iref, v_hub, b=5.6):
    """IEC 61400-1 eq. (11): representative value of the normal-turbulence-model standard deviation (90th percentile).

    sigma_1 = I_ref (0.75 V_hub + b), b = 5.6 m/s

    :param iref: reference turbulence intensity, from IEC_IREF[class] or the
        turbine's actual value from the manufacturer
    :param v_hub: hub-height wind speed, m/s (scalar or ndarray)
    :param b: constant of eq. (11); 5.6 m/s in Ed.3. **Ed.4 may differ** (see the
        module header)
    :return: sigma_1, m/s
    """
    return iref * (0.75 * np.asarray(v_hub, dtype=float) + b)


def design_ti(iref, v_hub, b=5.6):
    """Design turbulence intensity curve of a turbine class = sigma_1 / V_hub. The site effective TI must stay below it."""
    v = np.asarray(v_hub, dtype=float)
    return ntm_sigma1(iref, v, b) / v


def effective_ti(sim_res, wohler_exponent=10):
    """Effective turbulence intensity per IEC 61400-1 eq. (D.1): direction-probability weighted, m-th power mean.

        I_eff(V_hub) = [ sum_theta p(theta|V_hub) * I(theta, V_hub)^m ]^(1/m)

    The input I(theta, V) is py_wake's ``sim_res.TI_eff`` (ambient turbulence plus
    wake-added turbulence). The wind farm model must be built with
    ``STF2017TurbulenceModel`` or ``STF2005TurbulenceModel`` (py_wake's
    implementations of the Frandsen weighting). **Without a turbulence model
    ``TI_eff`` is meaningless and this function raises; it never falls back to
    ambient turbulence silently.**

    :param sim_res: py_wake SimulationResult with TI_eff and the probability P
    :param wohler_exponent: material Woehler (S-N curve) exponent m. Default 10 (a
        common value for composite blades); welded steel parts typically use 4.
        The result depends on m, which is why the standard asks for it per
        component material.

    The result does not include the complex-terrain correction C_CT. **C_CT must
    be applied to the ambient turbulence**, i.e. when building the site,
    ``UniformWeibullSite(ti=ti*C_CT)``, and py_wake then combines it with the
    wake-added turbulence. Because the m-th power mean is homogeneous,
    multiplying ``TI_eff`` or multiplying the result is mathematically identical
    and merely inflates the whole criterion; moving the factor to the input of the
    weighting is not a fix, applying it to the ambient turbulence before the wake
    combination is. That is why this function deliberately does not accept a C_CT
    argument: it forces the caller to handle it when building the site.

    :return: xarray.DataArray, dims (wt, ws): effective TI per turbine per wind-speed bin

    **Screening-level implementation, not a type-certification basis**:
      - it uses an engineering wake/turbulence model, not measurements;
      - the full criterion of eq. (35) is sigma_1 >= I_eff V + 1.28 sigma_sigma_hat;
        the last term (standard deviation of the turbulence standard deviation) is
        missing here, so the verdict below is **looser** than the standard, see
        ``check_iec_turbulence_suitability``.
    """
    # Do not test `"TI_eff" not in sim_res`: py_wake keeps TI_eff in its output even
    # when NO turbulence model is attached, equal to the ambient value everywhere.
    # The resulting I_eff would look normal yet contain no wake-added turbulence,
    # exactly the silent degradation to prevent. So the gate sits on the model
    # object, not on the output field.
    if getattr(sim_res.windFarmModel, "turbulenceModel", None) is None:
        raise ValueError(
            "The WindFarmModel has no turbulenceModel: build it with turbulenceModel="
            "STF2017TurbulenceModel() (or STF2005TurbulenceModel()). Without one, "
            "TI_eff equals the ambient turbulence, contains no wake-added turbulence, "
            "and cannot be used as an effective turbulence."
        )
    if "TI_eff" not in sim_res:
        raise ValueError("sim_res has no TI_eff field; cannot compute effective turbulence")
    m = float(wohler_exponent)
    ti = sim_res.TI_eff                       # (wt, wd, ws), ambient + wake-added
    p = sim_res.P                             # probability, (wd, ws) or (wt, wd, ws)
    p_wd = p / p.sum("wd")                    # p(theta|V): normalise over wd per ws
    return ((ti ** m * p_wd).sum("wd")) ** (1 / m)


def check_iec_turbulence_suitability(
    sim_res, turbulence_class="B", v_rated=None, v_out=25.0,
    wohler_exponent=10, ct_applied_upstream=None, iref=None, ntm_b=5.6,
):
    """Per turbine, compare the site effective turbulence with the turbine design turbulence; PASS/FAIL.

    The direction of the criterion follows IEC 61400-1 eq. (35): the turbine's NTM
    sigma_1 must be >= the site's wake + ambient turbulence level. The assessment
    band is the "turbine known" range **0.6 V_r ~ V_out**.

    :param turbulence_class: 'A+'|'A'|'B'|'C', looks up I_ref in IEC_IREF
    :param iref: overrides the class lookup (use the manufacturer's actual value)
    :param v_rated: rated wind speed, m/s. None starts the band at 8 m/s instead
        (V_ref is not available here, so it is not guessed) and says so in the
        returned caveats.
    :return: {
        'per_turbine': {wt_index: {'max_effective_ti', 'at_ws', 'design_ti_at_that_ws',
                                   'margin', 'verdict'}},
        'n_fail': int, 'assessment_band_ms': (lo, hi), 'caveats': [...]
      }
      verdict: 'PASS' (design TI has margin) | 'FAIL' (site effective turbulence
      exceeds the turbine design envelope)
    """
    if iref is None:
        if turbulence_class not in IEC_IREF:
            raise ValueError(f"Unknown turbulence class {turbulence_class!r}, choose from {list(IEC_IREF)}")
        iref = IEC_IREF[turbulence_class]

    ti_eff = effective_ti(sim_res, wohler_exponent)   # (wt, ws)
    ws = np.asarray(ti_eff.ws.values, dtype=float)

    caveats = [
        "Screening level: turbulence comes from a py_wake engineering model, not "
        "measurements; cannot replace type certification or a manufacturer load check",
        "The 1.28 * sigma_sigma_hat term of eq. (35) is missing, so the verdict is "
        "looser than the IEC criterion",
        f"I_ref={iref} (class {turbulence_class}), NTM constant b={ntm_b}, all on an Ed.3 basis (unverified)",
    ]
    if v_rated is None:
        lo = 8.0
        caveats.append("v_rated not given: the band starts at 8 m/s instead of 0.6*V_r - "
                       "differs from the standard's band")
    else:
        lo = 0.6 * float(v_rated)
    hi = float(v_out)

    band = (ws >= lo) & (ws <= hi)
    if not band.any():
        raise ValueError(f"Simulated wind-speed bins {ws.min()}~{ws.max()} m/s do not intersect the assessment band {lo}~{hi}")
    if not ct_applied_upstream or ct_applied_upstream == 1.0:
        caveats.append(
            "Complex-terrain turbulence structure correction C_CT not applied (it "
            "belongs on the ambient TI when building the site, not on the I_eff "
            "result - the latter is mathematically an inflation of the whole "
            "criterion and inflates the FAIL count)")
    else:
        caveats.append(f"C_CT={ct_applied_upstream} was applied to the ambient turbulence when building the site")

    ws_band = ws[band]
    design = design_ti(iref, ws_band, ntm_b)

    per_turbine = {}
    n_fail = 0
    for i, wt in enumerate(np.atleast_1d(ti_eff.wt.values)):
        site = np.asarray(ti_eff.isel(wt=i).values, dtype=float)[band]
        margin = design - site                     # >0 = margin left
        j = int(np.argmin(margin))                 # wind-speed bin with the least margin
        verdict = "PASS" if margin[j] >= 0 else "FAIL"
        n_fail += verdict == "FAIL"
        per_turbine[int(wt)] = {
            "max_effective_ti": float(site[j]),
            "at_ws": float(ws_band[j]),
            "design_ti_at_that_ws": float(design[j]),
            "margin": float(margin[j]),
            "verdict": verdict,
        }

    return {
        "per_turbine": per_turbine,
        "n_fail": n_fail,
        "assessment_band_ms": (lo, hi),
        "turbulence_class": turbulence_class,
        "iref": iref,
        "wohler_exponent": wohler_exponent,
        "ct_applied_upstream": ct_applied_upstream,
        "caveats": caveats,
    }


# ---------------------------------------------------------------------------
# Inflow angle
# ---------------------------------------------------------------------------

#: IEC 61400-1: the effect of the mean inflow inclination is to be considered up
#: to a maximum of 8 degrees (value as quoted, unverified here).
IEC_MAX_FLOW_INCLINATION_DEG = 8.0


def omni_inflow_angle_deg(elev, transform, points_lonlat, windrose, weighting="energy",
                          span_m=None, rotor_diameter_m=None):
    """Omni-directional inflow angle: per-sector inclination, then a sector-weighted average.

    The inflow angle is given **per wind-direction sector**
    (phi_i = arctan(v_z / v_xy), from measurement or a validated flow model); with
    neither, it may be estimated from terrain slope; the omni value is the
    frequency- or energy-weighted mean.

    This function is the terrain-slope version of that rule: the single-direction
    ``inflow_angle_deg()`` only answers "is the dominant direction uphill" and misses
    steep slopes in secondary sectors.

    :param windrose: with sector_freq_pct / sector_weibull_A / sector_weibull_k
    :param weighting: 'energy' (proportional to freq * A^3 * Gamma(1+3/k), default) or 'frequency'
    :return: {'omni': (N,), 'per_sector': (N, n_sectors), 'sector_deg': (n_sectors,),
              'weights': (n_sectors,), 'max_abs_sector': (N,)}
        ``max_abs_sector`` is the largest absolute inclination over **all sectors**
        per turbine; a max-over-directions criterion is stricter than the omni
        average, so both are returned.
    """
    from .spacingcheck import sector_energy_weights

    freq = np.asarray(windrose["sector_freq_pct"], dtype=float)
    n_sec = len(freq)
    centers = np.arange(n_sec) * (360.0 / n_sec)
    if weighting == "energy":
        _, w = sector_energy_weights(windrose)
    elif weighting == "frequency":
        w = freq / freq.sum()
    else:
        raise ValueError(f"weighting must be 'energy' or 'frequency', got {weighting!r}")

    per_sector = np.column_stack([
        inflow_angle_deg(elev, transform, points_lonlat, az, span_m, rotor_diameter_m)
        for az in centers
    ])
    return {
        "omni": (per_sector * w).sum(axis=1),
        "per_sector": per_sector,
        "sector_deg": centers,
        "weights": w,
        "max_abs_sector": np.nanmax(np.abs(per_sector), axis=1),
    }


def inflow_angle_deg(elev, transform, points_lonlat, azimuth_deg, span_m=None,
                     rotor_diameter_m=None):
    """Estimate the terrain inclination along the inflow direction at each turbine from a DEM, degrees. Positive = uphill inflow.

    Method: take a baseline on the **upwind side along the inflow direction** and
    apply arctan to the elevation difference between its two ends. It is a
    simplified version of "assume the flow is parallel to a plane fitted over
    the rotor neighbourhood": a two-point slope along one line approximates the inclination
    component of that fitted plane.

    :param elev: 2D DEM ndarray (geographic-degree grid, nodata=nan)
    :param transform: rasterio Affine on the same grid as elev
    :param points_lonlat: (N,2) ndarray, turbine lon/lat
    :param azimuth_deg: inflow azimuth (compass, clockwise from north): the wind
        blows **from** this direction
    :param span_m: baseline length, metres. Default 5 * rotor_diameter_m; 500 m if
        neither is given.
    :return: (N,) ndarray, inclination in degrees. Upwind lower than the turbine
        -> positive (uphill inflow)

    Two-point differencing, **not** the plane fit a standard asks for, and without
    roughness or flow separation. Use it only to **flag positions that may exceed
    8 degrees for human review**, not as a suitability conclusion.
    """
    if span_m is None:
        span_m = 5.0 * rotor_diameter_m if rotor_diameter_m else 500.0

    pts = np.asarray(points_lonlat, dtype=float)
    lon, lat = pts[:, 0], pts[:, 1]
    theta = math.radians(azimuth_deg)
    # the azimuth points to "where the wind comes from": the upwind side lies there
    dnorth, deast = math.cos(theta), math.sin(theta)
    dlat = (span_m * dnorth) / 111320.0
    dlon = (span_m * deast) / (111320.0 * np.cos(np.radians(lat)))

    from .rasterutils import sample_raster

    z_here = sample_raster(elev, transform, lon, lat)
    z_up = sample_raster(elev, transform, lon + dlon, lat + dlat)
    # upwind lower than the turbine => flow climbs => positive inclination
    return np.degrees(np.arctan((z_here - z_up) / span_m))


# ---------------------------------------------------------------------------
# Wind shear: power-law exponent from multi-height data
# ---------------------------------------------------------------------------

#: Plausible range of the power-law alpha. Outside it the value is treated as
#: noise (model-derived shear can be negative or implausibly large). Onshore typical 0.10-0.35; forest / urban can reach
#: 0.4+; negatives point to a low-level jet or a data problem. Example default.
PLAUSIBLE_SHEAR_ALPHA = (0.0, 0.45)


def shear_alpha_from_two_heights(ws_low, ws_high, h_low, h_high,
                                 clip=PLAUSIBLE_SHEAR_ALPHA):
    """Power-law shear exponent from wind speeds at two heights: alpha = ln(v2/v1) / ln(h2/h1).

    Why multi-height model layers: a wind-shear estimate should come from
    representative measurements or a measurement-calibrated, validated flow model.
    Without a met mast neither is available, but using a single-height model layer
    directly as hub-height speed is worse still, since it assumes alpha = 0. The
    next-best option is to back out alpha from the **same product's own height
    layers** (same model, same surface, free, global coverage).

    **Boundary**: this is a modelled long-term-mean shear, not a measurement. It
    cannot produce diurnal or stability variation or spike events, so it is
    screening level only and the report must say so.

    :param ws_low/ws_high: wind speed at the low / high height (scalar or ndarray, same shape)
    :param h_low/h_high: the heights, metres
    :param clip: (lo, hi) plausible range, values outside become nan. None disables clipping.
    :return: alpha, same shape as the input; implausible values are nan
    """
    lo = np.asarray(ws_low, dtype=float)
    hi = np.asarray(ws_high, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(hi / lo) / math.log(h_high / h_low)
    alpha = np.where((lo > 0) & (hi > 0), alpha, np.nan)
    if clip is not None:
        alpha = np.where((alpha >= clip[0]) & (alpha <= clip[1]), alpha, np.nan)
    return alpha


def extrapolate_wind_speed(ws, h_from, h_to, alpha):
    """Power-law extrapolation: v(h_to) = v(h_from) * (h_to/h_from)^alpha. Returns nan where alpha is nan."""
    return np.asarray(ws, dtype=float) * (float(h_to) / float(h_from)) ** np.asarray(alpha, dtype=float)
