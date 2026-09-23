"""Global data-centre infrastructure as a StockDrivenMFA: the SDU PhD advert's second named case
study, "the rapidly expanding infrastructure supporting data centres and AI computing".

Stock-driven, not register-based (like Denmark's turbines) or inflow-driven (like EV sales),
because the real data that exists for data centres is installed capacity, not a register of
facilities or a clean table of what was built each year. Capacity plus each layer's lifetime
determines what has to be built: new capacity, and replacement of whatever wore out.

Every figure below was checked against its source, not recalled.

REQUIRED CAPACITY (MW of data-centre capacity; one MW here means one MW of facility capacity,
the same basis as the copper figure below)
- 2023: 60 GW, and 2030: 171 GW (low) or 219 GW (high). McKinsey & Company, "AI power: Expanding
  data center capacity to meet growing demand", 29 October 2024: "global demand for data center
  capacity could rise ... to reach an annual demand of 171 to 219 gigawatts (GW). ... This
  contrasts with the current demand of 60 GW". Geometric interpolation between these endpoints.
  The growth they imply (about 16.1% and 20.3% a year) is somewhat below the 19-22% the same
  article quotes; the endpoints and quoted rates are not exactly consistent with each other, and
  the endpoints are used as the anchor.
- 2017-2023, back-cast: IEA, Energy and AI (2025): data-centre electricity use "has grown by
  around 12% per year since 2017". Capacity is assumed to track electricity use, a disclosed
  assumption, giving 60 / 1.12^6 = 30.4 GW in 2017.
- 2010-2017, held flat at 30.4 GW. Masanet et al. (2020), Science 367(6481):984-986: global
  data-centre energy use in 2018 "represents a 6% increase compared with 2010", essentially flat,
  under the same capacity-tracks-electricity assumption.
- 2030-2035: IEA Energy and AI base case, electricity supplied to data centres rising from about
  1,000 TWh (2030) to 1,300 TWh (2035), about 5.4% a year, applied to capacity (same assumption).
- After 2035: held flat. No source used here projects beyond 2035.

LAYERS, each with its own lifetime (the reason this needs a stock-driven model at all)
- "facility power and cooling": UPS, switchgear, busways, chillers, pumps, air handlers. Median
  life 20 years. ASHRAE's service-life data put this equipment at 17-25 years (motor starters 17,
  electric motors 18, metal cooling towers >22, centrifugal chillers >25); 20 is one central value
  for the whole layer, a disclosed simplification.
- "grid connection": on-site and near-site substations, transformers and feeders. Median life
  30 years: ASHRAE's median service life for electric transformers.
- Weibull shape k = 3.0 for both, a disclosed assumption: sources publish median lives, not
  distribution shapes. 3.0 because both lifetime curves this engine fits from real data land at
  k of about 3.1-3.2 (Danish wind turbines 3.20, NHTSA vehicle survival 3.14).

MATERIAL CONTENT (copper, tonnes per MW)
World Economic Forum with Kearney (Allgood and Hulak, December 2025), on Microsoft's 80 MW
Chicago site: "about 2,100 tonnes of copper - 26 tonnes/MW including on-site and near-site power
connections; 'facility-only' is roughly 12 tonnes/MW." So 12 t/MW for the facility layer and
26 - 12 = 14 t/MW for the grid connection. Single published points with no reported range, so the
(low, central, high) tuple repeats the value, the convention ev_battery.py already uses. Consistent
in direction with Amoah et al. (2026), Resources Policy 119, 105970, who find grid transmission
and distribution drives 64% of AI data-centre copper demand.

NOT MODELLED, deliberately
The IT layer (servers, storage, network) is left out. It matters for the dynamics, hyperscaler
filings put server lives at 5-6 years, so it is replaced several times per facility lifetime,
but no consistent public per-MW material figure was found, and WEF/Kearney report the roughly
60-75 t of minerals per MW sit "mainly in power and cooling systems, rather than servers". Steel,
aluminium, gallium, germanium and rare earths are likewise left out for lack of a consistent
per-MW source. Each is a stated gap, not a guessed figure.
"""
from __future__ import annotations

import pandas as pd

from mfa_engine.stock_driven import StockDrivenMFA

SNAPSHOT = pd.Timestamp("2023-12-31")   # McKinsey's "current demand of 60 GW": the last real anchor

_GW_2023 = 60.0
_GW_2030 = {"McKinsey low (171 GW by 2030)": 171.0, "McKinsey high (219 GW by 2030)": 219.0}
_IEA_GROWTH_SINCE_2017 = 0.12
_IEA_GROWTH_2030_2035 = (1300 / 1000) ** (1 / 5) - 1

SCENARIOS = list(_GW_2030)
DEFAULT_SCENARIO = SCENARIOS[0]

LAYERS: dict[str, tuple[float, float]] = {
    "facility power and cooling": (3.0, 20.0),
    "grid connection": (3.0, 30.0),
}

_COPPER_T_PER_MW = {
    "facility power and cooling": (12.0, 12.0, 12.0),
    "grid connection": (14.0, 14.0, 14.0),
}


def required_capacity_mw(scenario: str = DEFAULT_SCENARIO) -> pd.Series:
    """Required data-centre capacity in MW, 2010-2035 (held flat afterwards by the engine)."""
    gw_2017 = _GW_2023 / (1 + _IEA_GROWTH_SINCE_2017) ** 6
    gw_2030 = _GW_2030[scenario]
    gw = {}
    for year in range(2010, 2036):
        if year <= 2017:
            gw[year] = gw_2017
        elif year <= 2023:
            gw[year] = gw_2017 * (1 + _IEA_GROWTH_SINCE_2017) ** (year - 2017)
        elif year <= 2030:
            gw[year] = _GW_2023 * (gw_2030 / _GW_2023) ** ((year - 2023) / 7)
        else:
            gw[year] = gw_2030 * (1 + _IEA_GROWTH_2030_2035) ** (year - 2030)
    return pd.Series(gw) * 1000.0


class DataCentreMFA(StockDrivenMFA):
    """Two infrastructure layers with their own lifetimes and real copper content per MW."""

    def material_intensity(self, category: str) -> dict[str, tuple[float, float, float]]:
        return {"copper": _COPPER_T_PER_MW[category]}


def load_and_build(snapshot: pd.Timestamp = SNAPSHOT, scenario: str = DEFAULT_SCENARIO) -> DataCentreMFA:
    return DataCentreMFA(required_stock=required_capacity_mw(scenario), lifetimes=LAYERS, snapshot=snapshot)


def compare_scenario_stocks(horizon_year: int, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """Material stock-flow account for every capacity scenario, tagged by scenario, the same
    shape mfa_engine.scenarios.compare_scenario_stocks returns for EV batteries."""
    frames = []
    for name in SCENARIOS:
        flows = load_and_build(snapshot, name).run(horizon_year)["material_stock_flows"].copy()
        flows["scenario"] = name
        frames.append(flows)
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    for name in SCENARIOS:
        res = load_and_build(scenario=name).run(2050)
        sf = res["stock_flows"].groupby("year")[["inflow_units", "stock_units", "outflow_units"]].sum() / 1000
        mb = load_and_build(scenario=name).mass_balance_residual(res["material_stock_flows"])
        print(f"\n{name}: capacity GW (both layers summed, so 2x capacity), mass balance {mb.max():.1e} t")
        print(sf.loc[[2010, 2017, 2023, 2025, 2030, 2035, 2045]].round(1).to_string())
        cu = res["material_stock_flows"].set_index("year")
        print("copper (kt):", {y: round(float(cu.loc[y, 'stock_t']) / 1000, 1) for y in (2023, 2030, 2035)},
              "stock; outflow 2030:", round(float(cu.loc[2030, 'outflow_t']) / 1000, 1),
              "kt; inflow 2030:", round(float(cu.loc[2030, 'inflow_t']) / 1000, 1), "kt")
