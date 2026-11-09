"""Injection library: multiplicative flux models applied to real host light curves.

See profiles.py (shapes and the model), sampling.py (parameter draws per kind), apply.py
(PDCSAP + diluted SAP) and build.py (host selection and writing an injection set).
"""

from oddball.inject.apply import apply_model, sap_dilution
from oddball.inject.profiles import binned_model, model_from_params
from oddball.inject.sampling import KINDS, NULL_KINDS, draw_null_event, draw_params

__all__ = [
    "KINDS",
    "NULL_KINDS",
    "apply_model",
    "binned_model",
    "draw_null_event",
    "draw_params",
    "model_from_params",
    "sap_dilution",
]
