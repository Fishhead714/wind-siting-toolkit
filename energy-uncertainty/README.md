# wind-energy-uncertainty

A small, dependency-light calculator that turns a **loss chain** and a set of **1-sigma
uncertainty components** into **P50 / P75 / P90 / P99 exceedance values**, for a 1-year or an
N-year horizon. Python, numpy and scipy only.

## Method

All uncertainties are 1-sigma values expressed as a fraction of the P50 energy.

1. **Loss chain (multiplicative):** `net = gross * prod(1 - loss_i)`.
2. **Combination of long-term components:** root-sum-square for independent sources,
   or `sigma = sqrt(s^T R s)` with an explicit correlation matrix `R` (fully correlated
   sources add linearly, anti-correlated ones partly cancel).
3. **Horizon:** year-to-year variability is kept separate. For an N-year window
   `sigma_N^2 = sigma_lt^2 + sigma_iav^2 / N`, so a 1-year value is wider than a 10-year one.
4. **Exceedance under a normal distribution:** `P_p = P50 * (1 - z_p * sigma)`, with
   `z_p = Phi^-1(p)` (P90 uses z = 1.2816). P50 is taken as the mean of the assumed distribution.
5. Optional helper: `speed_to_energy_sigma` scales a wind-speed uncertainty to an energy
   uncertainty with a sensitivity you supply.

## Quick start

```python
from wind_energy_uncertainty import (Component, apply_loss_chain, total_sigma, exceedance_table)

net = apply_loss_chain(100.0, {"wake": 0.08, "availability": 0.03})["net"]
comps = [Component("measurement", 0.03),
         Component("interannual", 0.06, kind="interannual")]
print(exceedance_table(net, total_sigma(comps, years=10)))
```

`python examples/run_synthetic.py` runs a fully synthetic demo. Tests: `pip install pytest && pytest`.

## Assumptions

- Normal distribution of annual energy. The functions raise `ValueError` if the result
  would be non-positive (the normal tail is then unphysical).
- Components are combined as independent unless you pass a correlation matrix.
- **Uncertainty and loss values are your inputs.** `EXAMPLE_COMPONENTS` and the numbers in the
  example script are *illustrative placeholders*, not measured, not recommended, not from any
  real assessment.

## What this is not

- **Not a bankable energy assessment.** It does not produce a P50; it takes one. It does not
  model wind resource, wakes, or measurement quality, and it does not replace an
  independent assessment carried out to industry practice.
- No financial calculations (no price, revenue, debt-sizing or return outputs).
- Not validated against any measured production data.

## Provenance

This is a new implementation from standard textbook statistics, written for this toolkit;
it is not extracted from another code base. See `THIRD_PARTY_NOTICES.md`.

Licence: Apache-2.0.
