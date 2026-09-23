"""Connecting a physical dMFA projection to trade-concentration risk for the
same material.

The two halves usually live apart. ODYM and flodym project physical material
flows for a system but carry no supply-risk layer; crm-trade-network
quantifies supply concentration and cascade-shortfall risk from UN Comtrade
data but has no forward-looking physical flow projection. Integrated models
exist, GCMat's economic model for rare earths among them, but tend to be built
around one material rather than to take any technology system's own output.

This module is the connector: take a CohortSurvivalMFA subclass's own real
projected material flow for one material, and a crm-trade-network-style
real concentration/cascade result for that same material, and answer a
question neither project alone can answer: how much does what this
technology system recovers or releases actually matter, relative to a real
measured supply disruption risk for that material.

Every number this module reports either comes directly from one of those
two real, already-verified pipelines, or is a single, explicitly cited
conversion factor (a real market price), never an invented estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeConcentrationRisk:
    """Real, already-computed output from crm-trade-network for one
    commodity: concentration and cascade-shortfall figures, not
    recalculated here, just carried alongside the physical flow so the two
    can be compared honestly, with their real sources kept visible."""

    commodity_hs_code: str
    commodity_name: str
    total_trade_usd: float
    hhi: float
    top1_country: str
    top1_share: float
    cascade_country_removed: str
    cascade_shortfall_usd_no_slack: float
    cascade_shortfall_usd_20pct_slack: float
    source: str = "crm-trade-network, real 2023 UN Comtrade data"


def recovery_vs_disruption(
    material_tonnes_by_year: pd.Series,
    price_usd_per_tonne: float,
    price_source: str,
    risk: TradeConcentrationRisk,
) -> dict:
    """Convert a real physical recovery/release projection (tonnes per
    year, from a CohortSurvivalMFA subclass) into the same unit as the real
    trade-risk figures (USD), using one explicitly cited market price, then
    report what fraction of a real single-supplier disruption's shortfall
    that recovery would actually cover.

    This is deliberately a small, honest ratio, not a claim that recycling
    solves supply concentration. It answers a real question a materials
    scientist or a policy reader would actually ask: does this specific,
    real recovery pathway matter at the scale of a real measured risk, or
    not.
    """
    cumulative_tonnes = float(material_tonnes_by_year.sum())
    cumulative_usd = cumulative_tonnes * price_usd_per_tonne

    share_of_no_slack_shortfall = cumulative_usd / risk.cascade_shortfall_usd_no_slack
    share_of_20pct_slack_shortfall = cumulative_usd / risk.cascade_shortfall_usd_20pct_slack

    return {
        "material": risk.commodity_name,
        "cumulative_recovered_tonnes": cumulative_tonnes,
        "price_usd_per_tonne": price_usd_per_tonne,
        "price_source": price_source,
        "cumulative_recovered_usd": cumulative_usd,
        "disruption_scenario": f"{risk.cascade_country_removed} removed as a supplier "
                                f"({risk.top1_share:.1%} of real {risk.commodity_name} trade)",
        "real_shortfall_usd_no_slack": risk.cascade_shortfall_usd_no_slack,
        "real_shortfall_usd_20pct_slack": risk.cascade_shortfall_usd_20pct_slack,
        "recovered_share_of_no_slack_shortfall": share_of_no_slack_shortfall,
        "recovered_share_of_20pct_slack_shortfall": share_of_20pct_slack_shortfall,
        "risk_source": risk.source,
    }


def print_report(result: dict) -> None:
    print(f"=== Physical recovery vs. real trade-disruption risk: {result['material']} ===\n")
    print(f"Cumulative recovered: {result['cumulative_recovered_tonnes']:,.1f} t")
    print(f"  at {result['price_usd_per_tonne']:,.0f} USD/t ({result['price_source']})")
    print(f"  = ${result['cumulative_recovered_usd']/1e6:,.2f} million\n")
    print(f"Real disruption scenario: {result['disruption_scenario']}")
    print(f"  ({result['risk_source']})")
    print(f"  shortfall, no substitution : ${result['real_shortfall_usd_no_slack']/1e9:,.2f} bn")
    print(f"  shortfall, 20% slack       : ${result['real_shortfall_usd_20pct_slack']/1e9:,.2f} bn\n")
    print(f"This recovery covers:")
    print(f"  {result['recovered_share_of_no_slack_shortfall']:.2%} of the no-substitution shortfall")
    print(f"  {result['recovered_share_of_20pct_slack_shortfall']:.2%} of the 20%-slack shortfall")
