"""Denmark's wind turbine fleet, as a CohortSurvivalMFA subclass.

Moved here from tests/test_reproduces_dk_wind_mfa.py, where it was first
written specifically to prove the generic engine reproduces dk-wind-mfa's
own published numbers. It lived in the test file on purpose while that was
the only thing that mattered. Now that other code (the Streamlit app, the
supply-risk example) also needs it, it belongs in the package itself, not
tucked inside a test, so nothing outside the tests/ folder has to reach
into it.

Requires dk-wind-mfa's own real modules (load, drivetrain, intensity) on
sys.path, since this class is a thin subclass around them, not a copy of
their logic, real data and real coefficients stay in the one place that
owns them.
"""
from __future__ import annotations

from mfa_engine.cohort_survival import CohortSurvivalMFA


class WindTurbineMFA(CohortSurvivalMFA):
    """The one thing this subclass adds: real, JRC-sourced material
    content per MW, by generator topology. Everything else, censored
    survival estimation, Weibull extrapolation, cohort projection, comes
    from CohortSurvivalMFA unchanged."""

    def material_intensity(self, category: str) -> dict[str, tuple[float, float, float]]:
        """JRC's own table is kg per MW; this engine reports tonnes, to
        match dk-wind-mfa's own published figures, so the kg-to-tonnes
        conversion belongs here, in the one place that knows the source
        data's real unit."""
        import drivetrain as wind_drivetrain
        import intensity as wind_intensity

        topo = wind_drivetrain._TOPO[category]

        def as_tonnes(kg_per_mw: tuple[float, float, float]) -> tuple[float, float, float]:
            return tuple(v / 1000 for v in kg_per_mw)

        out = {
            "neodymium": as_tonnes(wind_intensity.rare_earth("neodymium", topo)),
            "dysprosium": as_tonnes(wind_intensity.rare_earth("dysprosium", topo)),
            "praseodymium": as_tonnes(wind_intensity.rare_earth("praseodymium", topo)),
            "terbium": as_tonnes(wind_intensity.rare_earth("terbium", topo)),
            "copper": as_tonnes(wind_intensity.copper(topo)),
        }
        for material in ("steel", "concrete", "cast iron"):
            out[material] = as_tonnes(wind_intensity.bulk(material))
        return out


def load_and_build(snapshot) -> "WindTurbineMFA":
    """Build a ready-to-run WindTurbineMFA from dk-wind-mfa's own real
    register data. Requires dk-wind-mfa/src on sys.path already, this
    function does not add it, since where that path lives is a decision
    for the caller (a script, a notebook, this app), not for this module."""
    import load as wind_load
    import drivetrain as wind_drivetrain

    existing = wind_drivetrain.add_topology(wind_load.load_existing())
    retired = wind_load.load_retired()

    return WindTurbineMFA(
        existing=existing,
        retired=retired,
        snapshot=snapshot,
        id_col="gsrn",
        unit_col="capacity_mw",
        category_col="topology",
        commissioned_col="connected",
        decommissioned_col="decommissioned",
        lifetime_col="lifetime_years",
    )
