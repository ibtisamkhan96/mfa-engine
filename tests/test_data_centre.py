"""Checks for the stock-driven engine and the data-centre case study built on it.

The stock-driven model derives inflow from a required stock, so the tests check the things that
would silently break if that derivation were wrong: the stock actually tracks the requirement,
the account closes every year, the spin-up really starts from a steady state, the published
anchors are reproduced exactly, and a falling requirement never produces negative building.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mfa_engine.stock_driven import StockDrivenMFA  # noqa: E402
from mfa_engine.systems.data_centre import (  # noqa: E402
    load_and_build, required_capacity_mw, SCENARIOS, LAYERS,
)
from mfa_engine.recovery import recovered  # noqa: E402

TOL = 1e-6


class _Toy(StockDrivenMFA):
    def material_intensity(self, category):
        return {"copper": (1.0, 1.0, 1.0)}


def main():
    checks = []
    low_name, high_name = SCENARIOS

    results = {}
    for name in SCENARIOS:
        model = load_and_build(scenario=name)
        res = model.run(2060)
        results[name] = res
        worst = float(model.mass_balance_residual(res["material_stock_flows"]).max())
        checks.append((f"{name}: mass balance closes every year (worst {worst:.1e} t)", worst < TOL))

        sf = res["stock_flows"]
        req = model.required_stock_to(2060)
        gaps = [abs(sf[sf.category == cat].set_index("year").stock_units - req).max() for cat in LAYERS]
        checks.append((f"{name}: every layer's stock equals the required capacity every year "
                       "(requirement never falls, so the max(0, .) never binds)", max(gaps) < TOL))
        checks.append((f"{name}: no negative inflow, stock or outflow",
                       (sf[["inflow_units", "stock_units", "outflow_units"]] >= -1e-9).all().all()))

    low = results[low_name]
    req_low = required_capacity_mw(low_name)
    checks.append(("McKinsey anchors reproduced exactly: 60 GW in 2023, 171 GW (low) and 219 GW (high) in 2030",
                   abs(req_low[2023] - 60_000) < TOL and abs(req_low[2030] - 171_000) < TOL
                   and abs(required_capacity_mw(high_name)[2030] - 219_000) < TOL))
    checks.append(("back-cast: 2017 capacity is 60 GW deflated at IEA's 12%/yr, then flat back to 2010 (Masanet 2020)",
                   abs(req_low[2017] - 60_000 / 1.12 ** 6) < TOL and abs(req_low[2010] - req_low[2017]) < TOL))

    flows = low["stock_flows"].groupby("year").sum(numeric_only=True)
    steady = flows.loc[2010:2017]
    checks.append(("spin-up starts in a real steady state: 2010-2017 inflow equals outflow (replacement only)",
                   float((steady.inflow_units - steady.outflow_units).abs().max()) < 1e-6))
    checks.append(("steady-state replacement is non-zero: an old fleet already retires equipment in 2010",
                   float(steady.outflow_units.iloc[0]) > 0))

    copper = low["material_stock_flows"].set_index("year")
    checks.append(("copper in use in 2023 = 60 GW x (12 + 14) t/MW = 1.56 Mt",
                   abs(float(copper.loc[2023, "stock_t"]) - 60_000 * 26) < 1e-3))
    high_copper = results[high_name]["material_stock_flows"].set_index("year")
    checks.append(("high scenario never has less copper in use than the low one",
                   bool((high_copper.stock_t >= copper.stock_t - 1e-6).all())))

    long_run = flows.loc[2050:2060]
    checks.append(("after the requirement stops growing, inflow converges to pure replacement (inflow = outflow by 2060)",
                   abs(float(long_run.inflow_units.iloc[-1] - long_run.outflow_units.iloc[-1]))
                   < 0.01 * float(long_run.inflow_units.iloc[-1])))

    rec = recovered(low["secondary_materials"])
    checks.append(("recovered copper never exceeds copper leaving service",
                   bool((rec.copper <= low["secondary_materials"].copper + 1e-9).all())))

    falling = pd.Series({2020: 100.0, 2021: 100.0, 2022: 40.0, 2023: 40.0, 2024: 90.0})
    toy = _Toy(required_stock=falling, lifetimes={"part": (3.0, 10.0)}, snapshot=pd.Timestamp("2021-12-31"))
    toy_res = toy.run(2030)
    toy_sf = toy_res["stock_flows"].set_index("year")
    checks.append(("a sharply falling requirement never produces negative building (max(0, .) binds in 2022)",
                   float(toy_sf.inflow_units.min()) >= 0 and float(toy_sf.loc[2022, "inflow_units"]) == 0.0))
    checks.append(("and the account still closes when it binds: stock above requirement, never below",
                   float(toy.mass_balance_residual(toy_res["material_stock_flows"]).max()) < TOL
                   and float(toy_sf.loc[2022, "stock_units"]) > 40.0))

    print(f"{'check':<112}{'status':>8}")
    ok_all = True
    for name, ok in checks:
        ok_all &= bool(ok)
        print(f"{name:<112}{'OK' if ok else 'FAIL':>8}")
    print()
    print("All checks pass." if ok_all else "FAIL: do not trust the data-centre numbers yet.")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
