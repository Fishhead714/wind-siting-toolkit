# Third-party notices

No third-party source code is copied into this package. It imports numpy and scipy at run
time. Licences were read from installed package metadata in a clean virtual environment
(versions filled below); neither is GPL or AGPL.

| Library | Version checked | Licence (from metadata) |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause (plus bundled permissive components) |
| scipy | 1.18.1 | BSD-3-Clause |

## Formulae

The root-sum-square combination, the correlated form sigma^2 = s^T R s, the
sigma_N^2 = sigma_lt^2 + sigma_iav^2/N horizon scaling and P_p = P50 (1 - z_p sigma) are
standard statistics found in any energy-assessment textbook or guideline; they are not
attributable to a single source and no text from any guideline or standard is reproduced.
This is a new implementation from those formulae.
