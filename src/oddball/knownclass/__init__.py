"""Known-class (explain-away) score: docs/ML_DESIGN.md §4.5.

* ``catalogs``  anonymous catalogue queries (VSX, Prsa+2022 TESS EBs, Gaia DR3 variability),
                cached under ``<data>/knownclass/raw`` with provenance.
* ``labels``    catalogue type -> known class, plus the hard exclusion list (anomaly target
                classes never become training labels).
* ``features``  the per-(tic, sector) feature table (classical features + periodogram and dip
                -count features + stellar context). Any table keyed the same way plugs in.
* ``model``     the known-class classifier (gradient-boosted trees, temperature-calibrated).
* ``explain``   the scorer-agnostic explain-away combiner.
"""
