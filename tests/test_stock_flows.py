"""Checks for the stock-flow account and the end-of-life recovery layer.

A dynamic material flow model has to close: for every material, every year,
stock(t) - stock(t-1) = inflow(t) - outflow(t). This checks that directly on all three real
configurations (Danish wind register, global EV sales, and an EV deployment scenario with
cohorts entering after the snapshot), plus the recovery rules that stop end-of-life outflow
being mistaken for recovered material.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
sys.path.insert(0, r"F:\job applications claude\dk-wind-mfa\src")

from mfa_engine.systems.wind_turbine import load_and_build as load_wind  # noqa: E402
from mfa_engine.systems.ev_battery import (  # noqa: E402
    load_and_build as load_ev, SNAPSHOT as EV_SNAPSHOT, GLOBAL_EV_SALES_MILLIONS,
)
from mfa_engine.scenarios import build_scenario, steps_ev_sales  # noqa: E402
from mfa_engine.recovery import recovered, rate_for, REGIMES  # noqa: E402

TOL_T = 1e-6   # tonnes; residuals should sit at floating-point noise, far below this


def main():
    checks = []

    wind_model = load_wind(snapshot=pd.Timestamp("2019-06-30"))
    wind = wind_model.run(horizon_year=2050)
    ev_model = load_ev(snapshot=EV_SNAPSHOT)
    ev = ev_model.run(horizon_year=2050)
    sc_model = build_scenario(steps_ev_sales(2050))
    sc = sc_model.run(horizon_year=2050)

    for label, model, res in (("wind register", wind_model, wind),
                              ("EV sales", ev_model, ev),
                              ("EV STEPS scenario", sc_model, sc)):
        worst = float(model.mass_balance_residual(res["material_stock_flows"]).max())
        checks.append((f"{label}: mass balance closes for every material, every year "
                       f"(worst residual {worst:.1e} t)", worst < TOL_T))
        checks.append((f"{label}: no negative inflow, stock or outflow anywhere",
                       (res["stock_flows"][["inflow_units", "stock_units", "outflow_units"]] >= -1e-9).all().all()))
        mat = res["material_stock_flows"]
        secondary = res["secondary_materials"].set_index("year")
        gaps = [abs(mat[(mat.material == m)].set_index("year").outflow_t - secondary[m]).max()
                for m in mat.material.unique()]
        checks.append((f"{label}: stock-flow outflow equals secondary_materials exactly "
                       "(one source of truth)", max(gaps) < TOL_T))

    wind_sf = wind["stock_flows"].groupby("year").sum(numeric_only=True)
    checks.append(("wind register: no inflow in the projection window (a register only "
                   "holds turbines already standing)", float(wind_sf.inflow_units.sum()) == 0.0))
    checks.append(("wind register: opening stock equals the register's own total capacity",
                   abs(wind_sf.stock_units.iloc[0] - wind_model.existing.capacity_mw.sum()) < 1e-6))

    ev_sf = ev["stock_flows"].groupby("year").sum(numeric_only=True)
    real_sales = sum(GLOBAL_EV_SALES_MILLIONS.values()) * 1e6
    checks.append(("EV sales: total inflow equals total real sales, every cohort entered once",
                   abs(ev_sf.inflow_units.sum() - real_sales) < 1.0))
    checks.append(("EV sales: the 2025 and 2026 cohorts are no longer dropped (inflow in both years)",
                   ev_sf.loc[2025, "inflow_units"] > 0 and ev_sf.loc[2026, "inflow_units"] > 0))
    checks.append(("EV sales: stock at the snapshot is below cumulative sales (inflow-driven: some "
                   "vehicles were already scrapped), not equal to it",
                   ev_sf.loc[2025, "stock_units"] < sum(v for y, v in GLOBAL_EV_SALES_MILLIONS.items()
                                                         if y <= 2025) * 1e6))
    checks.append(("EV sales: stock + cumulative outflow = cumulative inflow at the horizon",
                   abs(ev_sf.stock_units.iloc[-1] + ev_sf.outflow_units.sum() - ev_sf.inflow_units.sum()) < 1.0))

    violations = []
    for label, res in (("wind", wind), ("EV", ev)):
        gross = res["secondary_materials"]
        for regime in REGIMES:
            rec = recovered(gross, regime)
            for m in rec.columns.drop("year"):
                if not rec[m].isna().all() and not (rec[m] <= gross[m] + 1e-9).all():
                    violations.append(f"{label}/{regime}/{m}")
    checks.append((f"recovered never exceeds end-of-life outflow, any material, any regime "
                   f"({len(violations)} violations)", not violations))

    wind_rec = recovered(wind["secondary_materials"])
    nd_share = wind_rec.neodymium.sum() / wind["secondary_materials"].neodymium.sum()
    checks.append((f"neodymium recovery is under 1% of its outflow under current practice "
                   f"(got {nd_share:.2%}), not the 100% the old dashboard assumed", nd_share < 0.01))
    checks.append(("concrete and graphite report NaN (not assessed), not 0 (nothing recovered)",
                   wind_rec.concrete.isna().all() and recovered(ev["secondary_materials"]).graphite.isna().all()))
    checks.append(("EU Batteries Regulation raises lithium above current practice (80% vs <1%)",
                   rate_for("lithium", "EU Batteries Regulation targets (upper bound)").central
                   > rate_for("lithium").central))
    checks.append(("EU regime leaves EV copper on the UNEP rate (whole-vehicle copper, not only "
                   "battery copper)", rate_for("copper", "EU Batteries Regulation targets (upper bound)")
                   == rate_for("copper")))

    print(f"{'check':<104}{'status':>8}")
    all_pass = True
    for name, ok in checks:
        all_pass &= bool(ok)
        print(f"{name:<104}{'OK' if ok else 'FAIL':>8}")
    print()
    print("All checks pass." if all_pass else "FAIL: do not trust the stock-flow or recovery numbers yet.")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
