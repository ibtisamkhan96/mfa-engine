"""Sanity checks for EVBatteryMFA against the real sources it's built from.

Not a "reproduces a published study" test like test_reproduces_dk_wind_mfa.py,
because there is no single published EV dMFA study this generalises from.
Instead this checks the two things that would silently break if a future
edit introduced a real bug: the Weibull fit against NHTSA's own real
survival table should closely match that table's own real empirical median
(about 13.2 years, between the real age-13 point at 0.5094 survival and
age-14 at 0.4142), and every output should be physically sane (non-negative,
non-trivial, monotonically plausible).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa_engine.systems.ev_battery import load_and_build, _NHTSA_AGES, _NHTSA_SURVIVAL  # noqa: E402


def main():
    model = load_and_build(snapshot=pd.Timestamp("2024-12-31"))
    result = model.run(horizon_year=2050)

    # The real empirical NHTSA median sits between age 13 (survival 0.5094)
    # and age 14 (survival 0.4142); a well-fitted Weibull should land close
    # to that, not just anywhere reasonable-looking.
    real_median_lower_bound, real_median_upper_bound = 13.0, 14.0
    fitted_median = result["median_survival_years"]

    checks = [
        ("Weibull median within the real NHTSA empirical median's bracket",
         real_median_lower_bound <= fitted_median <= real_median_upper_bound),
        ("Weibull shape k is positive and not absurd (0.5-10)",
         0.5 <= result["weibull_shape"] <= 10),
        ("Weibull scale lambda is positive and not absurd (5-40 years)",
         5 <= result["weibull_scale"] <= 40),
        ("no negative material tonnages anywhere",
         (result["secondary_materials"].drop(columns=["year"]) >= 0).all().all()),
        ("cumulative graphite to 2050 exceeds cumulative lithium (real: graphite "
         "content per vehicle is always larger than lithium content, for both chemistries)",
         result["secondary_materials"].graphite.sum() > result["secondary_materials"].lithium.sum()),
        ("LFP cohorts show zero nickel (real: LiFePO4 has no nickel)",
         model.material_intensity("LFP")["nickel"][1] == 0.0),
        ("copper content is chemistry-independent (real: motor/wiring copper, "
         "not cathode-related), same for NMC811 and LFP",
         model.material_intensity("NMC811")["copper"] == model.material_intensity("LFP")["copper"]),
        ("copper content per vehicle (83 kg central) exceeds lithium content "
         "per vehicle (real: copper is the larger of the two by mass)",
         result["secondary_materials"].copper.sum() > result["secondary_materials"].lithium.sum()),
    ]

    print(f"{'check':<85}{'status':>10}")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"{name:<85}{'OK' if ok else 'FAIL':>10}")

    print()
    print(f"Weibull fit: k={result['weibull_shape']:.3f}, lambda={result['weibull_scale']:.3f}, "
          f"median={fitted_median:.2f} years (real NHTSA empirical median: ~13.2 years)")
    print()
    if all_pass:
        print("All checks pass.")
    else:
        print("MISMATCH: something here needs investigating before trusting this case study.")
    return all_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
