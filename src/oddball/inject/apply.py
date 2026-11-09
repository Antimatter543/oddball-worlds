"""Apply a model to a real host light curve (PDCSAP and SAP).

PDCSAP flux is crowding-corrected: the stored normalised PDCSAP is the *target's* flux, so an
astrophysical change ``m(t)`` of the target multiplies it directly.

SAP flux is not crowding-corrected: a fraction ``1 - CROWDSAP`` of the aperture flux comes from
other stars. Normalised by its median, ``SAP = c*T(t) + (1-c)*C`` with ``c = CROWDSAP``, so the
same change of the target gives ``SAP' = SAP * (1 - c * (1 - m))`` to first order: the depth
in SAP is diluted by ``c``. (FLFRCSAP, the fraction of the target's PSF inside the aperture,
cancels in the median normalisation.) When CROWDSAP is missing or not in (0, 1], ``c = 1``.

Per-bin errors are left unchanged: the host's own noise carries through, which is the point of
injecting into real light curves.
"""

from __future__ import annotations

import numpy as np

from oddball.inject.profiles import binned_model


def sap_dilution(crowdsap) -> float:
    c = float(crowdsap) if crowdsap is not None else np.nan
    return c if np.isfinite(c) and 0.0 < c <= 1.0 else 1.0


def apply_model(
    flux: np.ndarray,
    sap_flux: np.ndarray,
    t_centres: np.ndarray,
    params: dict,
    bin_days: float,
    crowdsap=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(flux_inj, sap_inj, m)``; NaNs (gaps) stay NaN."""
    m = binned_model(params, t_centres, bin_days)
    c = sap_dilution(crowdsap)
    m_sap = 1.0 - c * (1.0 - m)
    flux_inj = (np.asarray(flux, np.float64) * m).astype(np.float32)
    sap_inj = (np.asarray(sap_flux, np.float64) * m_sap).astype(np.float32)
    return flux_inj, sap_inj, m
