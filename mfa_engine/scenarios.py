"""Technology deployment scenarios: the same CohortSurvivalMFA engine, run under different
real, named future EV sales trajectories, closing one of the genuine gaps the SDU PhD advert
names directly, "the integration of technology deployment scenarios", by actually building
one rather than only asserting the design could support it.

Two real, named scenarios, applied only to the years after this project's own real historical/
current data (GLOBAL_EV_SALES_MILLIONS already covers 2019-2026 with real, cited IEA figures):

STEPS (IEA's own Stated Policies Scenario name): a real, cited endpoint, IEA Global EV Outlook
2026's own statement that EVs reach "around 50% of global car sales by 2035", linearly
interpolated from the real 2026 share (28%, same report) to that real 2035 endpoint, and
converted to a sales volume using one disclosed, held-flat assumption: the total global car
market stays at its real 2025 size (70.97 million/year, Statista/GlobalFleet 2025 aggregate),
since no real, cited total-market projection to 2035 was sourced. After 2035 the share is held
at the real 50% endpoint rather than extrapolating the 2026-2035 slope past the last real
anchor. An earlier version of this module did extrapolate it, silently reaching an 87% EV share
by 2050, a figure no source supports; holding at the last real anchor is the honest default.

Flat-2026 (a floor, not a forecast): EV sales held at the real 2026 level (23 million/year)
through the horizon year. Nobody expects growth to actually stop; this exists so a comparison
against STEPS shows how much of the result is load-bearing on continued growth actually
happening, rather than being an artefact of the engine's own arithmetic.

Both scenarios share the same chemistry-mix treatment (LFP share held at the real 2025 anchor,
55%, for every year beyond it, the same disclosed-flat-extrapolation ev_battery.py itself
applies to 2026), the same snapshot as the dashboard's own EV view (ev_battery.SNAPSHOT, so
both agree on which cohorts are already on the road), and the exact same CohortSurvivalMFA
engine, unmodified: only the sales trajectory fed into it differs between them.
"""
from __future__ import annotations

import pandas as pd

from mfa_engine.systems.ev_battery import (
    EVBatteryMFA, GLOBAL_EV_SALES_MILLIONS, LFP_SHARE_BY_YEAR, SNAPSHOT,
)

_STEPS_SHARE_2026 = 0.28              # real: IEA GEO2026, 28% of total car sales worldwide in 2026
_STEPS_SHARE_2035 = 0.50              # real: IEA GEO2026, "around 50% of global car sales by 2035"
_STEPS_ANCHOR_YEAR = 2035
_TOTAL_CAR_MARKET_MILLIONS = 70.97    # real: 2025 global passenger car sales (Statista/GlobalFleet)
_LFP_SHARE_LATEST = 0.55              # real: IEA GEO2026, "over 55%" LFP share, 2025 anchor

_LAST_REAL_YEAR = max(GLOBAL_EV_SALES_MILLIONS)   # 2026


def steps_ev_sales(horizon_year: int) -> dict[int, float]:
    """Real historical/current years unchanged, then a straight line between the real 2026
    and 2035 STEPS share anchors, held at the 2035 anchor afterwards, each year's share
    applied against the disclosed flat total-market assumption."""
    sales = dict(GLOBAL_EV_SALES_MILLIONS)
    span = _STEPS_ANCHOR_YEAR - _LAST_REAL_YEAR
    for year in range(_LAST_REAL_YEAR + 1, horizon_year + 1):
        t = min(1.0, (year - _LAST_REAL_YEAR) / span)
        share = _STEPS_SHARE_2026 + t * (_STEPS_SHARE_2035 - _STEPS_SHARE_2026)
        sales[year] = share * _TOTAL_CAR_MARKET_MILLIONS
    return sales


def flat_2026_ev_sales(horizon_year: int) -> dict[int, float]:
    """A floor, not a forecast: every year after _LAST_REAL_YEAR repeats its real sales
    volume exactly, so nothing about the chemistry mix or unit conversion differs from the
    real data, only whether growth continues."""
    sales = dict(GLOBAL_EV_SALES_MILLIONS)
    last_real_value = sales[_LAST_REAL_YEAR]
    for year in range(_LAST_REAL_YEAR + 1, horizon_year + 1):
        sales[year] = last_real_value
    return sales


def _lfp_share_for(year: int) -> float:
    return LFP_SHARE_BY_YEAR.get(year, _LFP_SHARE_LATEST)


def build_scenario(sales_by_year: dict[int, float], snapshot: pd.Timestamp = SNAPSHOT) -> EVBatteryMFA:
    """The identical cohort-construction logic ev_battery.load_and_build uses, generalised to
    take any {year: sales_millions} trajectory instead of only the real historical one.
    Cohorts sold after `snapshot` become inflows in the stock account (EVBatteryMFA keeps
    them, see its _projectable), so the same engine, unmodified, runs any named scenario."""
    rows = []
    for year, sales_millions in sales_by_year.items():
        lfp_share = _lfp_share_for(year)
        commissioned = pd.Timestamp(year=year, month=7, day=1)
        total_vehicles = sales_millions * 1e6
        rows.append({"cohort_id": f"{year}_LFP", "vehicles": total_vehicles * lfp_share,
                      "chemistry": "LFP", "commissioned": commissioned})
        rows.append({"cohort_id": f"{year}_NMC811", "vehicles": total_vehicles * (1 - lfp_share),
                      "chemistry": "NMC811", "commissioned": commissioned})
    existing = pd.DataFrame(rows)
    retired = pd.DataFrame(columns=["cohort_id", "vehicles", "chemistry", "commissioned"])
    return EVBatteryMFA(existing=existing, retired=retired, snapshot=snapshot,
                        id_col="cohort_id", unit_col="vehicles", category_col="chemistry",
                        commissioned_col="commissioned")


SCENARIOS = {
    "STEPS (IEA, ~50% EV sales share by 2035)": steps_ev_sales,
    "Flat 2026 (no further sales growth)": flat_2026_ev_sales,
}


def run_scenarios(horizon_year: int, snapshot: pd.Timestamp = SNAPSHOT) -> dict[str, dict]:
    """Every named scenario through the identical engine, full result dict per scenario
    (stock-flow account included), keyed by scenario name."""
    return {name: build_scenario(fn(horizon_year), snapshot).run(horizon_year)
            for name, fn in SCENARIOS.items()}


def compare_scenarios(horizon_year: int, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """One combined year-by-year end-of-life outflow table, tagged by scenario, so a
    difference between rows reflects the deployment trajectory alone, not any change to
    the engine."""
    frames = []
    for name, result in run_scenarios(horizon_year, snapshot).items():
        materials = result["secondary_materials"].copy()
        materials["scenario"] = name
        frames.append(materials)
    return pd.concat(frames, ignore_index=True)


def compare_scenario_stocks(horizon_year: int, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """The same comparison in stock-flow terms: inflow, in-use stock and end-of-life
    outflow per material, tagged by scenario. Stock is where a deployment choice shows up
    first; end-of-life outflow only reflects it once those vehicles are old enough to retire."""
    frames = []
    for name, result in run_scenarios(horizon_year, snapshot).items():
        flows = result["material_stock_flows"].copy()
        flows["scenario"] = name
        frames.append(flows)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    for name, sales_fn in SCENARIOS.items():
        sales = sales_fn(2050)
        print(f"{name}: 2030 -> {sales[2030]:.1f}M, 2035 -> {sales[2035]:.1f}M, "
              f"2050 -> {sales[2050]:.1f}M vehicles/year")
    stocks = compare_scenario_stocks(2050)
    lith = stocks[stocks.material == "lithium"].pivot(index="year", columns="scenario", values="stock_t")
    print("\nlithium in-use stock (t), selected years:")
    print(lith.loc[[2026, 2030, 2035, 2040, 2050]].round(0).to_string())
