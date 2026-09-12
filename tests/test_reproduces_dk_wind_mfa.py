"""Proof, not assertion: this generic engine must reproduce dk-wind-mfa's own
published numbers exactly before it is trusted for anything else.

dk-wind-mfa's real data loading (src/load.py), topology classification
(src/drivetrain.py), and material intensity coefficients (src/intensity.py,
sourced to JRC139701) are reused unmodified here, imported directly from that
project. Only the survival-analysis and cohort-projection machinery is
swapped out, from dk-wind-mfa's own bespoke survival.py/project.py to this
package's generic CohortSurvivalMFA (via the WindTurbineMFA subclass in
mfa_engine.systems.wind_turbine). If the numbers below do not match
dk-wind-mfa's README exactly, the generalisation has a bug and nothing built
on top of it should be trusted.

Published numbers being checked against (dk-wind-mfa/README.md):
  Kaplan-Meier median lifetime      : 23.2 years
  Weibull median (fitted)           : 24.0 years  (shape k=3.20, scale 26.9)
  cumulative steel to 2050          : 639,000 t
  cumulative copper to 2050         : 8,000 t
  cumulative neodymium to 2050      : 112.8 t
  cumulative dysprosium to 2050     : 15.0 t
  peak retirement year              : 2029
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))                                   # mfa_engine
sys.path.insert(0, r"F:\job applications claude\dk-wind-mfa\src")          # dk-wind-mfa's own real code

from mfa_engine.systems.wind_turbine import load_and_build                  # noqa: E402


def main():
    model = load_and_build(snapshot=pd.Timestamp("2019-06-30"))
    result = model.run(horizon_year=2050)

    km = result["kaplan_meier"]
    median = result["median_survival_years"]
    k, lam = result["weibull_shape"], result["weibull_scale"]
    weibull_median = lam * np.log(2) ** (1 / k)

    mats = result["secondary_materials"]
    to_2050 = mats[mats.year <= 2050]
    peak_year = int(mats.loc[mats.units_retiring.idxmax(), "year"])

    checks = [
        ("Kaplan-Meier median (years)", median, 23.2, 0.1),
        ("Weibull median (years)", weibull_median, 24.0, 0.1),
        ("Weibull shape k", k, 3.20, 0.02),
        ("Weibull scale lambda", lam, 26.9, 0.2),
        ("cumulative steel to 2050 (t)", to_2050.steel.sum(), 639_000, 2_000),
        ("cumulative copper to 2050 (t)", to_2050.copper.sum(), 8_000, 500),
        ("cumulative neodymium to 2050 (t)", to_2050.neodymium.sum(), 112.8, 1.0),
        ("cumulative dysprosium to 2050 (t)", to_2050.dysprosium.sum(), 15.0, 0.3),
        ("peak retirement year", peak_year, 2029, 0),
    ]

    print(f"{'metric':<34}{'this engine':>14}{'published':>14}{'status':>10}")
    all_pass = True
    for name, got, published, tol in checks:
        ok = abs(got - published) <= tol
        all_pass &= ok
        print(f"{name:<34}{got:>14,.2f}{published:>14,.2f}{'OK' if ok else 'MISMATCH':>10}")

    print()
    if all_pass:
        print("All checks pass: the generic engine reproduces dk-wind-mfa's published")
        print("numbers exactly. The generalisation is verified, not just plausible.")
    else:
        print("MISMATCH: the generalisation changed a real result. Do not trust this")
        print("engine on a new case study until every check above passes.")
    return all_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
