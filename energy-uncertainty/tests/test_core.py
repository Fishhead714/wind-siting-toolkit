import math

import numpy as np
import pytest
from scipy.stats import norm

from wind_energy_uncertainty import (
    EXAMPLE_COMPONENTS, Component, apply_loss_chain, combine_correlated, combine_rss,
    exceedance_table, exceedance_value, horizon_sigma, speed_to_energy_sigma, total_sigma,
)


def test_loss_chain_multiplicative():
    r = apply_loss_chain(100.0, {"a": 0.10, "b": 0.05})
    assert r["net"] == pytest.approx(100 * 0.9 * 0.95)
    assert r["total_loss"] == pytest.approx(1 - 0.855)
    assert apply_loss_chain(50.0, [])["net"] == 50.0
    with pytest.raises(ValueError):
        apply_loss_chain(1.0, [1.0])


def test_rss_closed_form():
    assert combine_rss([3, 4]) == pytest.approx(5.0)
    assert combine_rss([0.03, 0.04]) == pytest.approx(0.05)
    assert combine_rss([]) == 0.0


def test_correlated_limits():
    s = [0.03, 0.04]
    assert combine_correlated(s, np.eye(2)) == pytest.approx(0.05)
    full = [[1, 1], [1, 1]]
    assert combine_correlated(s, full) == pytest.approx(0.07)       # linear sum
    anti = [[1, -1], [-1, 1]]
    assert combine_correlated(s, anti) == pytest.approx(0.01)
    half = [[1, 0.5], [0.5, 1]]
    assert combine_correlated(s, half) == pytest.approx(math.sqrt(0.0009 + 0.0016 + 0.0012))


def test_correlated_rejects_bad_matrix():
    with pytest.raises(ValueError):
        combine_correlated([1, 1], [[1, 0.5], [0.4, 1]])
    with pytest.raises(ValueError):
        combine_correlated([1, 1, 1], np.eye(2))
    with pytest.raises(ValueError):  # not PSD
        combine_correlated([1, 1, 1], [[1, -.9, -.9], [-.9, 1, -.9], [-.9, -.9, 1]])


def test_horizon():
    assert horizon_sigma(0.05, 0.06, 1) == pytest.approx(math.sqrt(0.0025 + 0.0036))
    assert horizon_sigma(0.05, 0.06, 10) == pytest.approx(math.sqrt(0.0025 + 0.00036))
    assert horizon_sigma(0.05, 0.06, 10) < horizon_sigma(0.05, 0.06, 1)
    with pytest.raises(ValueError):
        horizon_sigma(0.05, 0.06, 0.5)


def test_total_sigma_example():
    lt = math.sqrt(0.03**2 + 0.03**2 + 0.04**2 + 0.02**2)
    assert total_sigma(EXAMPLE_COMPONENTS, years=1) == pytest.approx(math.sqrt(lt**2 + 0.06**2))
    assert total_sigma(EXAMPLE_COMPONENTS, years=10) == pytest.approx(math.sqrt(lt**2 + 0.06**2 / 10))
    # long-term only, with a correlation matrix
    comps = [Component("a", 0.03), Component("b", 0.04)]
    assert total_sigma(comps, corr=[[1, 1], [1, 1]]) == pytest.approx(0.07)


def test_exceedance_vs_scipy():
    p50, sigma = 1000.0, 0.08
    for lv in (50, 75, 90, 95, 99):
        assert exceedance_value(p50, sigma, lv) == pytest.approx(norm.ppf(1 - lv / 100, loc=p50, scale=p50 * sigma))
    assert exceedance_value(p50, sigma, 50) == pytest.approx(p50)
    assert exceedance_value(p50, sigma, 90) == pytest.approx(p50 * (1 - 1.2815515655 * sigma))
    t = exceedance_table(p50, sigma)
    assert list(t) == ["P50", "P75", "P90", "P99"]
    assert t["P99"] < t["P90"] < t["P75"] < t["P50"]


def test_exceedance_guards():
    with pytest.raises(ValueError):
        exceedance_value(100, 0.5, 99)   # 1 - 2.33*0.5 < 0
    with pytest.raises(ValueError):
        exceedance_value(100, 0.1, 100)


def test_speed_sensitivity():
    assert speed_to_energy_sigma(0.03, 2.0) == pytest.approx(0.06)
