"""Sanity checks for mfa_engine.scenarios, the deployment-scenario extension.

The real risk this guards against is not "does the arithmetic look reasonable", it's the
actual bug found while building this module: routing a future-dated cohort through the
base class's normal age0 > 0 filter silently drops it, making every named scenario
collapse to the same real historical-only result regardless of how different their future
sales trajectories actually are. These checks specifically verify the scenarios diverge,
and diverge by a physically sensible amount and direction, not just that nothing crashes.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa_engine.scenarios import (  # noqa: E402
    compare_scenarios, steps_ev_sales, flat_2026_ev_sales, SCENARIOS,
)
from mfa_engine.systems.ev_battery import GLOBAL_EV_SALES_MILLIONS  # noqa: E402


def main():
    checks = []

    # Both scenario functions must agree exactly with the real historical/current data,
    # and with each other, for every year that has already actually happened.
    steps_2035 = steps_ev_sales(2035)
    flat_2035 = flat_2026_ev_sales(2035)
    real_years_match = all(
        steps_2035[y] == GLOBAL_EV_SALES_MILLIONS[y] == flat_2035[y]
        for y in GLOBAL_EV_SALES_MILLIONS
    )
    checks.append(("both scenarios reproduce the real historical/current sales exactly "
                    "(2019-2026), only future years differ", real_years_match))

    checks.append(("STEPS 2026 sales matches the real IEA anchor exactly (23.0M)",
                    steps_2035[2026] == 23.0))
    checks.append(("STEPS 2035 sales implies the real ~50% share against the disclosed "
                    "flat 70.97M total market (35.0-35.6M)",
                    35.0 <= steps_2035[2035] <= 35.6))
    checks.append(("Flat scenario holds exactly at the real 2026 figure through 2035",
                    all(flat_2035[y] == 23.0 for y in range(2027, 2036))))
    steps_2050 = steps_ev_sales(2050)
    checks.append(("STEPS holds at its real 2035 anchor afterwards, not extrapolated past it "
                    "(an earlier version silently reached an 87% EV share by 2050)",
                    all(abs(steps_2050[y] - steps_2050[2035]) < 1e-9 for y in range(2036, 2051))))

    # This is the actual regression check for the bug this module hit while being built:
    # if the age0 > 0 filter were silently back in effect, every one of these would be 0%.
    prior_pct = None
    monotonic = True
    for horizon in (2035, 2040, 2045, 2050):
        combined = compare_scenarios(horizon)
        material_cols = [c for c in combined.columns if c not in ("year", "units_retiring", "scenario")]
        totals = combined.groupby("scenario")[material_cols].sum()
        flat_total = totals.loc["Flat 2026 (no further sales growth)", "lithium"]
        steps_total = totals.loc["STEPS (IEA, ~50% EV sales share by 2035)", "lithium"]
        pct = (steps_total - flat_total) / flat_total * 100 if flat_total else 0.0
        if horizon == 2035:
            checks.append((f"the two scenarios actually diverge by {horizon} (not silently "
                            "identical, the real bug this module hit)", pct > 0.1))
        if prior_pct is not None and pct <= prior_pct:
            monotonic = False
        prior_pct = pct
        checks.append((f"no negative material tonnage anywhere at horizon {horizon}",
                        (combined.drop(columns=["year", "scenario"]) >= 0).all().all()))

    checks.append(("STEPS's lead over Flat grows as the horizon extends further past "
                    "battery lifetime (higher future sales only show up in retirement "
                    "once those batteries are actually old enough to retire)", monotonic))

    print(f"{'check':<95}{'status':>10}")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"{name:<95}{'OK' if ok else 'FAIL':>10}")

    print()
    if all_pass:
        print("All checks pass.")
    else:
        print("MISMATCH: something here needs investigating before trusting this case study.")
    return all_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
