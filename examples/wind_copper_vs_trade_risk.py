"""A real, end-to-end demonstration of the new connector: does the copper
Denmark's wind fleet returns as it retires actually matter, at the scale of
a real measured supply-disruption risk for copper.

Every number in this script comes from one of two already-verified, real
pipelines:
  - dk-wind-mfa's own real turbine register, run through mfa_engine's
    generic CohortSurvivalMFA (verified in tests/test_reproduces_dk_wind_mfa.py
    to reproduce dk-wind-mfa's own published numbers exactly)
  - crm-trade-network's own real 2023 UN Comtrade concentration and
    single-supplier cascade analysis for HS 7403, refined copper

The one number that is not itself an output of either pipeline is the
copper price used to convert tonnes to USD so the two can be compared in
the same unit, and that price is explicitly cited rather than assumed:
2026 average LME copper price, approximately 12,842 USD/t (Trading
Economics / MacroMicro market data, checked live while building this).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                  # mfa_engine
sys.path.insert(0, r"F:\job applications claude\dk-wind-mfa\src")             # dk-wind-mfa's own real code
sys.path.insert(0, r"F:\job applications claude\crm-trade-network\src")       # crm-trade-network's own real code

from mfa_engine import TradeConcentrationRisk, recovery_vs_disruption, print_report  # noqa: E402
from mfa_engine.systems.wind_turbine import load_and_build                          # noqa: E402

# --- 1. Real physical side: run the verified wind engine ------------------
wind_model = load_and_build(snapshot=pd.Timestamp("2019-06-30"))
wind_result = wind_model.run(horizon_year=2050)
copper_by_year = wind_result["secondary_materials"].set_index("year")["copper"]

# --- 2. Real economic side: crm-trade-network's own real copper analysis --
from build import load_edges, bilateral, unified_edges   # noqa: E402
from roles import throughput, role_lookup                 # noqa: E402
from concentration import concentration                   # noqa: E402
from network import cascade                                # noqa: E402

edges_raw = load_edges()
bilat = bilateral(edges_raw)
roles = role_lookup(throughput(edges_raw))
conc = concentration(bilat, roles, "value_usd")
copper_conc = conc[(conc.commodity == "7403") & (conc.stage == "Pooled")].iloc[0]

unified = unified_edges(edges_raw)
copper_edges = unified[unified.commodity == "7403"]
no_slack = cascade(copper_edges, copper_conc.top1, slack=0.0)
with_slack = cascade(copper_edges, copper_conc.top1, slack=0.2)

risk = TradeConcentrationRisk(
    commodity_hs_code="7403",
    commodity_name="refined copper",
    total_trade_usd=float(copper_edges.value_usd.sum()),
    hhi=float(copper_conc.hhi),
    top1_country=copper_conc.top1,
    top1_share=float(copper_conc.top1_share),
    cascade_country_removed=copper_conc.top1,
    cascade_shortfall_usd_no_slack=no_slack["shortfall_usd"],
    cascade_shortfall_usd_20pct_slack=with_slack["shortfall_usd"],
)

# --- 3. The new connection: physical recovery vs. real disruption risk ----
COPPER_PRICE_USD_PER_TONNE = 12_842.0
PRICE_SOURCE = "2026 average LME copper price, Trading Economics / MacroMicro, checked live"

result = recovery_vs_disruption(
    material_tonnes_by_year=copper_by_year,
    price_usd_per_tonne=COPPER_PRICE_USD_PER_TONNE,
    price_source=PRICE_SOURCE,
    risk=risk,
)
print_report(result)
