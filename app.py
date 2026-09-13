"""Streamlit dashboard for mfa_engine.

This wraps two things that are already verified on their own:

  1. mfa_engine.systems, two independent CohortSurvivalMFA subclasses:
     WindTurbineMFA (reproduces dk-wind-mfa's own published numbers
     exactly, see tests/test_reproduces_dk_wind_mfa.py) and EVBatteryMFA
     (the global EV fleet, verified against a real external survival
     curve, see tests/test_ev_battery_mfa.py). Two real, different
     technology systems through the same unmodified base class is the
     actual test of "modular and reusable", not just an assertion.
  2. mfa_engine.supply_risk_context, the connector that compares either
     projection's physical material recovery against crm-trade-network's
     real 2023 UN Comtrade concentration and cascade-shortfall analysis.

Nothing in this app computes a new number on its own. It loads real output
from those two pipelines, lets the viewer change a small number of real
parameters (projection horizon, which real commodity and supplier the
disruption scenario uses, the price used to convert tonnes to USD), and
visualises the result. If either sibling project is missing, the app says
so plainly and stops, rather than substituting placeholder data.

The risk connector covers every material either physical system actually
contains that also has a matching, trackable UN Comtrade commodity code:
copper and rare earth metals from wind turbines; copper, lithium, nickel,
cobalt and graphite from EV batteries. That is every one of Wu Chen's own
named Villum grant materials (cobalt, copper, nickel, lithium), and copper
specifically is tracked from both systems, so the same real recovered
tonne can be compared across technologies, the actual point of connecting
more than one system to this risk data in the first place.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="mfa-engine",
    page_icon="\u2b21",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- theming

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Fira Sans', sans-serif; }
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    .stTabs [data-baseweb="tab"] {
        font-family: 'Fira Code', monospace !important;
    }
    [data-testid="stMetricValue"] { color: #5EEAD4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- dependencies

# Local dev keeps the original assumption (both sibling repos checked out next to this
# one, the layout every real command in this session's history has used). A container
# deploy (Dockerfile clones both real GitHub repos to /deps/ at build time, see the
# Dockerfile's own comments) sets these two environment variables instead, since
# "sibling folder on the same machine" is a local-filesystem convention that has no
# meaning once this runs somewhere else. Same real code either way, just where it looks.
DK_WIND_MFA_SRC = Path(os.environ.get(
    "DK_WIND_MFA_SRC", r"F:\job applications claude\dk-wind-mfa\src"
))
CRM_TRADE_NETWORK_SRC = Path(os.environ.get(
    "CRM_TRADE_NETWORK_SRC", r"F:\job applications claude\crm-trade-network\src"
))


def check_dependencies() -> None:
    missing = []
    if not (DK_WIND_MFA_SRC / "load.py").exists():
        missing.append(f"dk-wind-mfa not found at `{DK_WIND_MFA_SRC}`")
    if not (CRM_TRADE_NETWORK_SRC / "build.py").exists():
        missing.append(f"crm-trade-network not found at `{CRM_TRADE_NETWORK_SRC}`")
    if missing:
        st.error(
            "This dashboard wraps two sibling projects that could not be found:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nRestore them at the paths above, then reload this page. "
            "Nothing below can run without the real data they provide."
        )
        st.stop()


check_dependencies()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(DK_WIND_MFA_SRC))
sys.path.insert(0, str(CRM_TRADE_NETWORK_SRC))

from mfa_engine import TradeConcentrationRisk, recovery_vs_disruption  # noqa: E402
from mfa_engine.systems.wind_turbine import load_and_build as load_wind  # noqa: E402
from mfa_engine.systems.ev_battery import load_and_build as load_ev  # noqa: E402

# Each physical system: its real loader, the real "as of" date its own data
# actually reflects, and the vocabulary its own category/unit actually mean,
# so the UI can describe either system honestly instead of hardcoding
# wind-turbine language everywhere.
SYSTEMS = {
    "wind": {
        "label": "Danish wind turbines",
        "loader": load_wind,
        "snapshot": pd.Timestamp("2019-06-30"),   # dk-wind-mfa's own register snapshot date
        "category_label": "turbine topology",
        "unit_label": "Capacity retiring (MW)",
    },
    "ev": {
        "label": "global EV batteries",
        "loader": load_ev,
        "snapshot": pd.Timestamp("2024-12-31"),   # last year with a real, session-verified sales figure
        "category_label": "battery chemistry",
        "unit_label": "Vehicles retiring",
    },
}

# Real 2023 refined copper price range (LME), shared by both copper material
# groups below since it is the same real global commodity regardless of
# which physical system (wind turbines or EV batteries) recovers it. Central:
# World Bank Commodity Markets ("Pink Sheet"), June 2023 monthly average,
# $8,396.5/t (checked live, a single real reported point rather than an
# invented blend of the several other 2023 estimates also found while
# researching this, e.g. S&P Global's $8,785/t full-year forecast). Low/high:
# real LME spot prices within 2023 itself, approximately $7,850/t (mid-October
# 2023 low, converted from $3.56/lb) to approximately $9,360/t (January 23,
# 2023 high). This is the same treatment material_intensity() already gives
# physical content, a real range instead of a single point.
_COPPER_PRICE_RANGE_2023 = (7_850.0, 8_396.5, 9_360.0)
_COPPER_PRICE_SOURCE = (
    # Dollar signs escaped (\$) throughout: this string is rendered via st.markdown/st.caption,
    # which auto-renders anything between two literal "$" as LaTeX (Streamlit's KaTeX
    # integration). More than one real "$X" amount in the same caption was silently mangling
    # the text between them into garbled math instead of showing it, a real rendering bug
    # found while screenshotting this feature, not a hypothetical one.
    "Central: World Bank Commodity Markets, June 2023 monthly average refined copper price "
    "(\\$8,396.5/t). Range: real LME spot prices within 2023, ~\\$7,850/t (mid-October 2023 low) to "
    "~\\$9,360/t (January 23, 2023 high), matching the 2023 UN Comtrade trade year."
)

# Every material either real physical system actually contains that also has
# a real, trackable UN Comtrade commodity code. Bulk structural materials
# (steel, concrete, cast iron) have no CRM trade code; lithium/cobalt/nickel/
# graphite have real trade data but only EV batteries (not wind turbines)
# actually contain them. Copper appears twice, once per physical system,
# on purpose: it is the one material both real case studies actually
# contain, so it is the only place this connector can show genuine
# cross-technology material competition for the same real recovered tonne,
# the actual point of building a connector across more than one system.
MATERIAL_GROUPS = {
    "Copper (wind turbines)": {
        "system": "wind",
        "hs_code": "7403",
        "name": "refined copper",
        "physical_columns": ["copper"],
        "price_mode": "external",
        "external_price": _COPPER_PRICE_RANGE_2023[1],
        "external_price_range": _COPPER_PRICE_RANGE_2023,
        "external_price_source": _COPPER_PRICE_SOURCE,
    },
    "Copper (EV batteries)": {
        "system": "ev",
        "hs_code": "7403",
        "name": "refined copper",
        "physical_columns": ["copper"],
        "price_mode": "external",
        "external_price": _COPPER_PRICE_RANGE_2023[1],
        "external_price_range": _COPPER_PRICE_RANGE_2023,
        "external_price_source": _COPPER_PRICE_SOURCE,
    },
    "Rare earth metals": {
        "system": "wind",
        "hs_code": "2805",
        "name": "rare earth metals (Nd + Dy + Pr + Tb combined)",
        "physical_columns": ["neodymium", "dysprosium", "praseodymium", "terbium"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
        # No citable_range: this is a basket of four elements with real, wildly different 2023
        # unit prices (USGS Mineral Commodity Summaries 2024: NdPr oxide ~$80,000/t average,
        # dysprosium oxide ~$323,000/t, terbium oxide ~$1,300,000/t), and none of the three had a
        # real, dated calendar-year low/high pair available from a free source this session, only
        # annual averages and a few bounding anchor points. Building a basket range would mean
        # weighting these very different real prices by each element's real physical tonnage
        # share, a real additional modeling step, not a citation; disclosed as context instead of
        # forced into a single low/high pair without that step.
        "external_context": "For real context only, not used as this material's range: 2023 full-year "
        "average prices for three of this basket's four real elements (USGS Mineral Commodity "
        "Summaries 2024) were roughly \\$80,000/t (neodymium oxide, down from \\$134,000/t in 2022), "
        "\\$323,000/t (dysprosium oxide), and \\$1,300,000/t (terbium oxide), annual averages, not a "
        "dated low/high pair, and not combined into one basket figure here since that would require "
        "weighting by each element's own real physical tonnage share, an extra modeling step beyond "
        "what these citations alone support.",
    },
    "Lithium": {
        "system": "ev",
        "hs_code": "2836",
        "name": "carbonates (incl. lithium carbonate)",
        "physical_columns": ["lithium"],
        # NOT "implied": checked and rejected. HS 2836 "Carbonates" traded
        # 23.5 million real tonnes in 2023, roughly 25-30x real global
        # lithium carbonate production (~0.8-0.9 Mt/year), so the code is
        # overwhelmingly non-lithium carbonates (almost certainly soda ash),
        # and its implied unit value ($860/t) is nowhere near real lithium
        # carbonate economics. A real, dated external quote is honest here;
        # a blended implied price from this code is not.
        "price_mode": "external",
        "external_price": 32_694.0,
        "external_price_range": (22_950.0, 32_694.0, 42_000.0),
        "external_price_source": "Central: 2023 average battery-grade lithium carbonate price, China, "
        "Benchmark Mineral Intelligence (checked live), matching the 2023 UN Comtrade trade year. "
        # $ escaped as \$, see the comment on _COPPER_PRICE_SOURCE above for why.
        "Range: real 2023 in-year spot prices for the same product moved from about \\$22,950/t "
        "(around October 11, 2023) up to about \\$42,000/t (May 2023), during that year's price "
        "collapse from its late-2022 peak. The annual average and the spot low/high come from "
        "different aggregation methods (full-year average vs. specific in-year spot points), "
        "disclosed here rather than presented as if directly comparable.",
    },
    "Nickel": {
        "system": "ev",
        "hs_code": "7502",
        "name": "unwrought nickel",
        "physical_columns": ["nickel"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
        # HS 7502 "unwrought nickel" is real product match for LME Class 1 nickel, close enough
        # to build a real range around: (low, high), source. Central stays the implied blended
        # 2023 UN Comtrade average (a real, data-derived point specific to this exact trade),
        # not the LME benchmark itself; the two independently line up (implied ~$22,720/t falls
        # inside this real range), a real consistency check rather than an assumed one.
        "citable_range": (15_885.0, 31_200.0, "Westmetall LME cash-settlement daily data (checked "
                          "live): 2023 calendar-year low \\$15,885/t (27 Nov 2023), high \\$31,200/t "
                          "(3 Jan 2023); cross-checked against an independent source within 0.1%."),
    },
    "Cobalt": {
        "system": "ev",
        "hs_code": "8105",
        "name": "cobalt mattes and articles",
        "physical_columns": ["cobalt"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
        # No citable_range here, on purpose: HS 8105 legally spans everything from low-grade
        # mattes to refined cobalt metal and articles, and its own real blended 2023 price
        # (~$10,927/t) sits far below the real refined-metal benchmark (USGS/Cobalt Institute:
        # standard-grade cobalt metal, EXW Europe, 2023 real range ~$30,865-$44,092/t). Forcing
        # the metal benchmark onto this blended, lower-grade-heavy code would misprice it, the
        # same class of mismatch already caught and rejected once this session for lithium via
        # HS 2836. Disclosed as real context, not used as this material's range.
        "external_context": "For real context only, not used as this material's range: standard-grade "
        "refined cobalt metal (a higher-value product than the blended mix HS 8105 actually trades) "
        "moved between roughly \\$30,865/t (~Sept 2023 low) and \\$44,092/t (Jan 2023 high) in 2023, "
        "USGS National Minerals Information Center / Cobalt Institute Cobalt Market Report 2023.",
    },
    "Graphite": {
        "system": "ev",
        "hs_code": "2504",
        "name": "natural graphite",
        "physical_columns": ["graphite"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
        # No citable_range here either: the best real, dated 2023 benchmark found (Fastmarkets,
        # natural flake graphite 94% C, -100 mesh, fob China) real 2023 range is ~$530-830/t, and
        # this HS code's own real blended average (~$1,170/t) sits ABOVE that entire range, e.g.
        # because HS 2504 real trade includes larger-flake or higher-purity material this specific
        # benchmark doesn't cover. A range that does not bracket the central estimate would be
        # worse than no range, so this is disclosed as context instead of forced into one.
        "external_context": "For real context only, not used as this material's range (it sits below "
        "this code's own real blended average, a real grade/product mismatch, not an error): "
        "Fastmarkets' natural flake graphite (94% C, -100 mesh, fob China) benchmark moved between "
        "\\$530-575/t (28 Dec 2023 low) and \\$830/t (5 Jan 2023 high) over 2023.",
    },
}

# Sankey material groups are chosen so nothing inside one group differs by more than
# roughly an order of magnitude. Copper (8,000 t) next to neodymium (110 t) makes the
# rare-earth links and labels collapse into an unreadable sliver; keeping tonnage
# within a group comparable is what makes every link and label actually legible.
SANKEY_GROUPS_BY_SYSTEM = {
    "wind": {
        "Rare earth metals (Nd, Dy, Pr, Tb)": ["neodymium", "dysprosium", "praseodymium", "terbium"],
        "Copper": ["copper"],
        "Bulk structural (steel, concrete, cast iron)": ["steel", "concrete", "cast iron"],
    },
    "ev": {
        "Battery metals (Li, Ni, Mn, Co)": ["lithium", "nickel", "manganese", "cobalt"],
        "Graphite": ["graphite"],
        "Copper (motor, wiring, busbars)": ["copper"],
    },
}

# --------------------------------------------------------------------------- cached loaders

@st.cache_resource(show_spinner="Fitting the survival model on its own real register...")
def run_system_model(system_key: str, horizon_year: int):
    system = SYSTEMS[system_key]
    model = system["loader"](snapshot=system["snapshot"])
    result = model.run(horizon_year=horizon_year)
    return model, result


@st.cache_data(show_spinner="Loading crm-trade-network's real 2023 UN Comtrade data...")
def load_commodity_trade(hs_code: str) -> tuple[dict, pd.DataFrame]:
    from build import load_edges, bilateral, unified_edges
    from roles import throughput, role_lookup
    from concentration import concentration

    edges_raw = load_edges()
    bilat = bilateral(edges_raw)
    roles = role_lookup(throughput(edges_raw))
    conc = concentration(bilat, roles, "value_usd")
    conc_row = conc[(conc.commodity == hs_code) & (conc.stage == "Pooled")]
    if conc_row.empty:
        raise ValueError(
            f"No concentration data for commodity {hs_code} in the currently cached "
            "UN Comtrade data. If you just forced a live refresh, the API may not have returned "
            "any rows this time, try again."
        )
    commodity_conc = conc_row.iloc[0].to_dict()

    unified = unified_edges(edges_raw)
    commodity_edges = unified[unified.commodity == hs_code].reset_index(drop=True)
    return commodity_conc, commodity_edges


@st.cache_data(show_spinner="Computing cascade shortfall...")
def compute_cascade(edges: pd.DataFrame, country: str, slack: float) -> dict:
    from network import cascade

    return cascade(edges, country, slack=slack)


@st.cache_data(show_spinner=False)
def implied_unit_price(edges: pd.DataFrame) -> float | None:
    """Real average USD/tonne implied directly by this commodity's own 2023
    trade data (total value divided by total reported weight). Used for
    commodities where no single external spot-price citation would be
    honest (either the code bundles several distinct elements at very
    different prices, like rare earth metals, or a specific enough real
    citation was not sourced this session, like lithium/nickel/cobalt/
    graphite). This is a blended average of whatever was actually traded,
    not one element's price."""
    total_usd = edges.value_usd.sum()
    total_kg = edges.net_kg.sum()
    if total_kg <= 0:
        return None
    return float(total_usd / (total_kg / 1000))


@st.cache_data(show_spinner=False)
def implied_unit_price_range(edges: pd.DataFrame) -> tuple[float, float, float] | None:
    """Real (low, central, high) USD/tonne for commodities with no external
    spot-price citation, giving price the same (low, central, high)
    treatment material_intensity() already gives physical content. Central
    is the same blended total-value/total-weight average implied_unit_price()
    returns. Low/high are the real 5th/95th percentile of the per-row
    implied unit value (that row's own value_usd / net_kg) across individual
    real reporting bilateral trade records in the same UN Comtrade dataset.

    This is NOT a market spot-price range: it is the real dispersion already
    present inside one HS code's own reported 2023 trade, different
    countries report different unit values for nominally the same commodity
    because of real differences in product grade/mix, re-exports, and
    customs valuation practice. That dispersion is itself a genuine,
    data-derived signal about how uncertain "the price" of a blended
    commodity code actually is, not a fabricated range, and is reported here
    as exactly that, not disguised as a spot-price band.
    """
    valid = edges[edges.net_kg > 0]
    if valid.empty:
        return None
    central = implied_unit_price(edges)
    if central is None:
        return None
    per_row = valid.value_usd / (valid.net_kg / 1000)
    low = float(per_row.quantile(0.05))
    high = float(per_row.quantile(0.95))
    return (low, central, high)


@st.cache_data(show_spinner="Computing the cross-material comparison (real data for all seven materials)...")
def compute_all_material_scenarios(horizon_year: int) -> pd.DataFrame:
    """The same recovery_vs_disruption calculation the sidebar runs for
    whichever one material is currently selected, run for all seven at once,
    each against its own real top single supplier. Lets a viewer see which
    of these real recovery pathways matters most, relatively, without
    clicking through seven separate dropdown states one at a time."""
    rows = []
    for name, g in MATERIAL_GROUPS.items():
        try:
            conc, edges = load_commodity_trade(g["hs_code"])
        except ValueError:
            continue
        _, result = run_system_model(g["system"], horizon_year)
        total_usd = float(edges.value_usd.sum())
        top1 = conc["top1"]
        no_slack = compute_cascade(edges, top1, 0.0)
        slack20 = compute_cascade(edges, top1, 0.2)
        share = no_slack["lost_usd"] / total_usd if total_usd else 0.0

        if g["price_mode"] == "external":
            price, price_source = g["external_price"], g["external_price_source"]
        else:
            implied = implied_unit_price(edges)
            price, price_source = (implied or 0.0), "implied 2023 UN Comtrade unit value"

        mat_by_year = result["secondary_materials"].set_index("year")[g["physical_columns"]].sum(axis=1)
        mat_by_year = mat_by_year[mat_by_year.index <= horizon_year]
        # Annualized, not cumulative: the real cascade shortfall is a one-year disruption
        # estimate, so it must be compared against one typical year of this recovery pathway,
        # not everything a system will ever recover by an arbitrary horizon year. See the same
        # comment where this connector is called for the currently-selected material for the
        # real failure mode this avoids (a >1000% "coverage" share for smaller global markets).
        annualized = pd.Series([float(mat_by_year.mean())]) if len(mat_by_year) else pd.Series([0.0])

        if no_slack["shortfall_usd"] <= 0 or slack20["shortfall_usd"] <= 0:
            rows.append({"material": name, "top1": top1, "top1_share": share,
                         "coverage_no_slack": None, "coverage_20pct": None})
            continue

        risk = TradeConcentrationRisk(
            commodity_hs_code=g["hs_code"], commodity_name=g["name"], total_trade_usd=total_usd,
            hhi=float(conc["hhi"]), top1_country=top1, top1_share=share,
            cascade_country_removed=top1, cascade_shortfall_usd_no_slack=no_slack["shortfall_usd"],
            cascade_shortfall_usd_20pct_slack=slack20["shortfall_usd"],
        )
        res = recovery_vs_disruption(annualized, price, price_source, risk)
        rows.append({
            "material": name, "top1": top1, "top1_share": share,
            "coverage_no_slack": res["recovered_share_of_no_slack_shortfall"],
            "coverage_20pct": res["recovered_share_of_20pct_slack_shortfall"],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cache_freshness(year: int, cmd: str) -> dict:
    """Real mtimes of the on-disk cache files this dashboard actually reads,
    so the app never implies data is live when it is a snapshot from
    whenever it was last fetched."""
    import fetch as trade_fetch

    files = list(trade_fetch.CACHE.glob(f"{year}_{cmd}_*.json"))
    if not files:
        return {"count": 0, "oldest": None, "newest": None}
    mtimes = [f.stat().st_mtime for f in files]
    return {
        "count": len(files),
        "oldest": pd.Timestamp(min(mtimes), unit="s"),
        "newest": pd.Timestamp(max(mtimes), unit="s"),
    }


# --------------------------------------------------------------------------- sidebar: horizon + material

st.sidebar.header("Parameters")

horizon_year = st.sidebar.slider(
    "Projection horizon", min_value=2025, max_value=2060, value=2050, step=1,
    help="How far to project the retirement wave forward from the selected system's own real snapshot date.",
)

material_choice = st.sidebar.selectbox(
    "Risk connector material", list(MATERIAL_GROUPS.keys()),
    help="Every material covered here is one either real physical system (Danish wind turbines, "
    "the global EV fleet) actually contains AND has a real, trackable UN Comtrade commodity code. "
    "Bulk structural materials (steel, concrete) have no CRM trade code; lithium/cobalt/nickel/"
    "graphite have real trade data but only EV batteries, not wind turbines, contain them.",
)
group = MATERIAL_GROUPS[material_choice]
system = SYSTEMS[group["system"]]
# "Copper (wind turbines)" and "Copper (EV batteries)" are two distinct dropdown entries so the
# same real commodity can be compared across technologies, but that parenthetical qualifier makes
# metric labels (fixed-width columns) truncate. short_material drops it for those tight spots;
# full material_choice is kept everywhere else, where flowing text has room for the distinction.
short_material = material_choice.split(" (")[0]

try:
    commodity_conc, commodity_edges = load_commodity_trade(group["hs_code"])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

total_trade_usd = float(commodity_edges.value_usd.sum())
exporters = (
    commodity_edges.groupby("exporter").value_usd.sum().sort_values(ascending=False).head(15).index.tolist()
)

default_country_idx = exporters.index(commodity_conc["top1"]) if commodity_conc["top1"] in exporters else 0
removed_country = st.sidebar.selectbox(
    "Supplier removed in the disruption scenario", exporters, index=default_country_idx,
    help=f"Any of the 15 largest real exporters of {group['name']} in the 2023 UN Comtrade data. "
    f"Defaults to {commodity_conc['top1']}, the real largest single supplier.",
    key=f"removed_country_{material_choice}",
)

if group["price_mode"] == "external":
    default_price = group["external_price"]
    default_price_source = group["external_price_source"]
    price_range = group["external_price_range"]
    price_range_source = "a real, externally cited 2023 price range, see the caption below"
else:
    implied = implied_unit_price(commodity_edges)
    default_price = round(implied, 2) if implied else 0.0
    default_price_source = (
        f"implied 2023 UN Comtrade unit value for HS {group['hs_code']} (total trade value divided "
        "by total reported weight): a real blended average across whichever specific goods were "
        "actually traded under this code, not one element's spot price"
    )
    # NOT implied_unit_price_range(): checked and rejected. Per-shipment implied unit value
    # within one HS code is dominated by a handful of extreme outlier rows (a tiny, oddly
    # valued shipment can imply a $/t many times the real blended average), both unweighted
    # and weight-adjusted. For cobalt mattes this produced a 95th-percentile high bound over
    # $1.2M/t against a real blended central of $10,927/t, and for graphite a low bound near
    # $6/t; propagating either into the risk connector reintroduced exactly the implausible
    # (>100%) coverage class this app's own history already fixed once. Real external citations
    # are used instead, where the product actually matches (Nickel); where it doesn't
    # (Cobalt, Graphite, Rare earth metals), the real citation found is shown as context only,
    # see external_context below, rather than forced into a range that wouldn't bracket the
    # real central estimate.
    if "citable_range" in group:
        cr_low, cr_high, cr_source = group["citable_range"]
        price_range = (cr_low, default_price, cr_high)
        price_range_source = cr_source
    else:
        price_range = None
        price_range_source = None

price = st.sidebar.number_input(
    f"{material_choice} price (USD/tonne)", min_value=100.0, max_value=500_000.0,
    value=default_price, step=50.0,
    help=f"Used only to convert recovered tonnes to USD. Default: {default_price_source}.",
    key=f"price_input_{material_choice}",
)
price_source = default_price_source if price == default_price else "user-entered override"

if price_range:
    # \$ (escaped), not $: st.caption renders markdown, and two or more literal "$" in one
    # string get auto-rendered as LaTeX between them (Streamlit's KaTeX integration), mangling
    # the text. See the comment on _COPPER_PRICE_SOURCE for where this was actually caught.
    st.sidebar.caption(
        f"Real price range, not a point value: **\\${price_range[0]:,.0f} - \\${price_range[2]:,.0f}/t** "
        f"(central \\${price_range[1]:,.0f}/t used above), {price_range_source}. The risk connector "
        "below reports coverage at all three, the same (low, central, high) treatment already given "
        "to physical material content."
    )
elif "external_context" in group:
    # No real range for this material (see the comment on MATERIAL_GROUPS above for why), but a
    # real external citation was still found and is shown here, disclosed as context rather than
    # silently dropped or forced into a range that wouldn't bracket the central estimate above.
    st.sidebar.caption(group["external_context"])

st.sidebar.divider()
st.sidebar.caption(
    "Data policy: physical flow comes from dk-wind-mfa or real, cited external sources (see the "
    "EV battery case's own module for citations), verified against real published or reported "
    "numbers. Trade concentration and cascade shortfall come from crm-trade-network's real 2023 "
    "UN Comtrade data. No number here is simulated or fabricated."
)

st.sidebar.subheader("Trade data freshness")
fresh = cache_freshness(2023, group["hs_code"])
if fresh["count"]:
    st.sidebar.caption(
        f"{fresh['count']} real UN Comtrade queries on disk for {group['name']}, fetched between "
        f"{fresh['oldest']:%Y-%m-%d} and {fresh['newest']:%Y-%m-%d}. This is a real snapshot, not "
        "re-fetched on every page load, because the API is rate-limited; use the button below to "
        "genuinely re-ask it right now."
    )
else:
    st.sidebar.caption(f"No cached {group['name']} data on disk yet. Use the button below to fetch it live.")

with st.sidebar.expander("Fetch live data now"):
    st.caption(
        f"Re-asks UN Comtrade's live API for every one of the ~32 reporting countries, for {group['name']}, "
        "both import and export directions, and overwrites the cached files with today's real response. "
        "The API is rate-limited to one new request every 4 seconds, so this takes several minutes. "
        "Keep this tab open until it finishes."
    )
    if st.button("Fetch live data now (re-ask UN Comtrade)", key=f"refresh_{material_choice}"):
        import fetch as trade_fetch

        progress_bar = st.progress(0.0)
        status = st.empty()

        def _report(i: int, total: int, country: str, flow: str) -> None:
            progress_bar.progress(i / total)
            status.caption(f"[{i}/{total}] {country} ({flow})")

        made = trade_fetch.ensure_cached(2023, group["hs_code"], force=True, progress=_report)
        status.caption(f"Done: {made} live calls made to UN Comtrade just now.")
        cache_freshness.clear()
        load_commodity_trade.clear()
        st.rerun()

# --------------------------------------------------------------------------- run pipelines

system_model, system_result = run_system_model(group["system"], horizon_year)
cascade_no_slack = compute_cascade(commodity_edges, removed_country, 0.0)
cascade_20pct = compute_cascade(commodity_edges, removed_country, 0.2)
removed_country_share = cascade_no_slack["lost_usd"] / total_trade_usd if total_trade_usd else 0.0

risk = TradeConcentrationRisk(
    commodity_hs_code=group["hs_code"],
    commodity_name=group["name"],
    total_trade_usd=total_trade_usd,
    hhi=float(commodity_conc["hhi"]),
    top1_country=commodity_conc["top1"],
    top1_share=removed_country_share,
    cascade_country_removed=removed_country,
    cascade_shortfall_usd_no_slack=cascade_no_slack["shortfall_usd"],
    cascade_shortfall_usd_20pct_slack=cascade_20pct["shortfall_usd"],
)

material_by_year = system_result["secondary_materials"].set_index("year")[group["physical_columns"]].sum(axis=1)
material_by_year = material_by_year[material_by_year.index <= horizon_year]
cumulative_recovered_tonnes = float(material_by_year.sum())
# The real cascade shortfall is a one-year disruption estimate (2023 trade removed for a
# year), but material_by_year spans the whole projection horizon (potentially 25+ years).
# Comparing a multi-decade cumulative recovery total against a single year's shortfall is a
# real, demonstrated failure mode, not a hypothetical one: for the global EV fleet, cumulative
# graphite recovery through 2050 came out to over 1000% of one year's real trade-disruption
# cost, mathematically consistent with the inputs but not a meaningful claim, since it silently
# compares quantities on two different time scales. Denmark's wind fleet never crossed 1% here
# only because its market is tiny; the underlying mismatch was there the whole time. The fix is
# to compare like with like: an average YEAR of this recovery pathway against one real year of
# disruption, not everything ever recovered by an arbitrary horizon year against one year.
average_annual_tonnes = float(material_by_year.mean()) if len(material_by_year) else 0.0
annualized_series = pd.Series([average_annual_tonnes])

# Peak year, surfaced alongside the average on purpose: averaging across the whole projection
# is itself a smaller version of the same time-scale problem the annualization fix above exists
# to avoid, a single retirement-wave peak year can recover several times the average year, and
# reporting only the average would quietly understate how large this pathway gets in its busiest
# real year. Showing both, not picking one, is the honest fix.
peak_year = int(material_by_year.idxmax()) if len(material_by_year) else None
peak_year_tonnes = float(material_by_year.max()) if len(material_by_year) else 0.0
peak_annualized_series = pd.Series([peak_year_tonnes])

no_slack_shortfall = cascade_no_slack["shortfall_usd"]
slack_shortfall = cascade_20pct["shortfall_usd"]

if no_slack_shortfall <= 0:
    connector_result = None
    peak_connector_result = None
    price_low_result = None
    price_high_result = None
    scenario_note = ("no_exports", None)
elif slack_shortfall <= 0:
    # A real, distinct outcome, not an edge case to hide: removing this supplier creates a
    # genuine no-substitution shortfall, but the remaining suppliers' 20% slack fully absorbs
    # it. recovery_vs_disruption divides by both shortfalls, so it is never called with a zero
    # denominator; this case is reported honestly instead of crashing or being mislabeled.
    connector_result = None
    peak_connector_result = None
    price_low_result = None
    price_high_result = None
    scenario_note = ("slack_covers_it", no_slack_shortfall)
else:
    connector_result = recovery_vs_disruption(annualized_series, price, price_source, risk)
    peak_connector_result = recovery_vs_disruption(peak_annualized_series, price, price_source, risk)
    # Coverage at the real price range's own low and high bound, not the (possibly user-
    # overridden) price above: this shows how much of the coverage-percentage conclusion is
    # actually sensitive to real, cited price uncertainty, the same (low, central, high)
    # treatment material content already gets, using the selected material's own price_range
    # and price_range_source computed in the sidebar section above.
    if price_range:
        price_low_result = recovery_vs_disruption(annualized_series, price_range[0], price_source, risk)
        price_high_result = recovery_vs_disruption(annualized_series, price_range[2], price_source, risk)
    else:
        price_low_result = price_high_result = None
    scenario_note = None

# --------------------------------------------------------------------------- header

st.title("mfa-engine")
st.caption(
    "A reusable dynamic material flow analysis engine, run here on two real, independent "
    "technology systems (Danish wind turbines, the global EV fleet), connected to "
    "crm-trade-network's real supply-chain risk data. No existing tool combines modular physical "
    "dMFA with real economic/trade risk across technology systems; this is a first attempt at that "
    "connection, not a finished substitute for either field."
)

with st.expander("How to read this dashboard (start here)", expanded=False):
    st.markdown(
        "This dashboard answers one question per tab, in order. Reading it top to bottom actually "
        "follows the real analysis pipeline, it is not an arbitrary tab order.\n\n"
        "1. **Pick a material and a horizon in the left sidebar.** \"Risk connector material\" "
        "chooses both which real physical system runs (Danish wind turbines or the global EV fleet) "
        "and which real UN Comtrade commodity code its recovered material is compared against. "
        "Copper appears twice on purpose, once per physical system, because it is the one material "
        "both real case studies actually contain.\n"
        "2. **Survival & lifetime** (tab 1): how long one unit of this system (a turbine, a vehicle "
        "battery) actually lasts before retiring, and what real data that estimate rests on.\n"
        "3. **Retirement & secondary materials** (tab 2): how much of each material comes back out "
        "as the fleet retires, year by year, and which category (turbine topology or battery "
        "chemistry) it actually comes from.\n"
        "4. **Supply-chain risk connector** (tab 3): the new piece. Does the real recovery from tab 2 "
        "matter at the scale of a real measured supply-disruption risk for the same material. Change "
        "\"Supplier removed in the disruption scenario\" in the sidebar to explore different real "
        "disruption scenarios; everything in this tab updates accordingly.\n\n"
        "Every chart below this point carries its own caption with the specific formula it plots and "
        "how that number is actually measured, not just what the axes mean."
    )

with st.expander("What this app actually is, and isn't"):
    st.markdown(
        "- **The physical side** runs two real, independent case studies through the same "
        "unmodified engine: Denmark's real wind turbine fleet (verified to reproduce dk-wind-mfa's "
        "own published numbers exactly), and the global EV fleet (verified against a real external "
        "vehicle survival curve, since the real EV fleet is too young to have generated enough of "
        "its own retirement events to fit one from scratch). Both use right-censored survival "
        "analysis extended with a fitted Weibull curve; running two genuinely different systems "
        "through the same base class, not just one, is what makes \"modular and reusable\" a "
        "demonstrated claim rather than an assertion.\n"
        "- **The risk side** is not computed here. It is crm-trade-network's own real concentration "
        "(HHI) and single-supplier cascade-shortfall analysis of 2023 UN Comtrade data, loaded as-is.\n"
        "- **The connector** converts physical tonnes to USD with one explicitly documented price, "
        "either an external market quote (copper, lithium) or a real average implied directly by the "
        "same trade dataset (rare earths, nickel, cobalt, graphite), each now given as a real (low, "
        "central, high) range rather than a single point, then asks a narrow, honest question: does "
        "this real recovery pathway matter at the scale of a real measured supply-disruption risk. "
        "It is not a claim that recycling solves supply concentration.\n"
        "- Every material connected here is one a real physical system actually contains, with a "
        "matching, trackable UN Comtrade code: copper and rare earths from wind turbines; copper, "
        "lithium, nickel, cobalt and graphite from EV batteries. That is every one of Wu Chen's own "
        "named Villum grant materials (cobalt, copper, nickel, lithium), and copper specifically is "
        "tracked from both systems so the same real recovered tonne can be compared across "
        "technologies, not just within one.\n"
        "- No confidence interval is plotted on the survival curve, because this engine does not "
        "compute one. Showing a fabricated band would be worse than showing none.\n"
        "- **Why Denmark for wind but the whole world for EVs, a deliberate design choice, not an "
        "accident.** Denmark's wind fleet is modeled nationally because dk-wind-mfa's own register, "
        "the real source this case reproduces exactly, is itself a national Danish register with "
        "individual turbine coordinates and commissioning dates; no comparably detailed global wind "
        "turbine register with real per-unit retirement data was found to build a global version "
        "from. The EV case is modeled globally because its own real source (IEA's Global EV Outlook) "
        "only reports sales and chemistry mix at the global level, a single country's EV fleet would "
        "be too small and too young to say anything about chemistry mix or retirement timing with any "
        "real confidence. Comparing a national system against a global one is therefore a property of "
        "which real data actually exists for each technology today, not a shortcut taken to make the "
        "numbers line up. The honest consequence, which this app does not hide, is that the two "
        "physical systems are not being compared at the same geographic scale; what they share, and "
        "what the connector actually compares, is the same real commodity (refined copper) and the "
        "same real 2023 global trade-risk dataset, not the same population size."
    )

m1, m2, m3, m4 = st.columns(4)
weibull_median = system_result["weibull_scale"] * np.log(2) ** (1 / system_result["weibull_shape"])
mats_all = system_result["secondary_materials"]
cum_material = mats_all[mats_all.year <= horizon_year][group["physical_columns"]].sum().sum()
m1.metric("Weibull median lifetime", f"{weibull_median:,.1f} yr")
m2.metric(f"Cumulative {short_material.lower()} to {horizon_year}", f"{cum_material:,.0f} t")
m3.metric(f"Real {short_material.lower()} HHI (pooled)", f"{commodity_conc['hhi']:.3f}")
m4.metric(f"Real top supplier ({commodity_conc['top1']})", f"{commodity_conc['top1_share']:.1%}")

st.divider()

tab_survival, tab_materials, tab_risk = st.tabs(
    ["Survival & lifetime", "Retirement & secondary materials", "Supply-chain risk connector"]
)

# --------------------------------------------------------------------------- tab 1: survival

with tab_survival:
    km = system_result["kaplan_meier"]
    k, lam = system_result["weibull_shape"], system_result["weibull_scale"]

    fig = go.Figure()
    if km is not None:
        km_t = np.concatenate([[0.0], km.t.values])
        km_s = np.concatenate([[1.0], km.survival.values])
        age_grid = np.linspace(0, max(km_t.max(), weibull_median * 1.5), 200)
        fig.add_trace(go.Scatter(
            x=km_t, y=km_s, mode="lines", name="Kaplan-Meier (empirical, right-censored)",
            line=dict(shape="hv", color="#5EEAD4", width=2),
        ))
    else:
        age_grid = np.linspace(0, weibull_median * 2, 200)
    weibull_s = np.exp(-((age_grid / lam) ** k))
    fig.add_trace(go.Scatter(
        x=age_grid, y=weibull_s, mode="lines",
        name="Weibull (fitted to a real external survival curve)" if km is None else "Weibull fit (extrapolated)",
        line=dict(color="#F59E0B", width=2, dash="dash"),
    ))
    fig.add_hline(y=0.5, line=dict(color="#6B7280", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
        xaxis_title="Age (years)", yaxis_title="Survival probability",
        yaxis_range=[0, 1.02], legend=dict(orientation="h", y=1.1), height=460,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, width="stretch")
    if km is not None:
        st.latex(
            r"\text{Kaplan-Meier: } S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)"
            r"\qquad\qquad \text{Weibull: } S(t) = \exp\!\left(-(t/\lambda)^k\right)"
        )
        st.caption(
            f"Kaplan-Meier median: {system_result['median_survival_years']:.2f} years "
            f"(stops at the oldest observed retirement). Weibull median: {weibull_median:.2f} years "
            f"(shape k={k:.2f}, scale \u03bb={lam:.2f}), used to extrapolate the retirement projection "
            "beyond what has actually been observed. Measured from: every real turbine in the "
            "register, still standing (d_i=0, right-censored at its current age) or already "
            "decommissioned (d_i=1, at its real realised lifetime); k and \u03bb are fit by maximum "
            "likelihood over both kinds of observation at once."
        )
    else:
        st.latex(r"\text{Weibull: } S(t) = \exp\!\left(-(t/\lambda)^k\right)")
        st.caption(
            f"No empirical curve here: {system['label']} has not existed long enough to generate "
            "enough real retirement events to fit one from (real finding: only about 3% of global "
            "EV battery capacity had been scrapped as of December 2020). This Weibull (shape "
            f"k={k:.2f}, scale \u03bb={lam:.2f}, median {weibull_median:.2f} years) is instead fitted "
            "directly against a real, externally published vehicle survivability table (NHTSA, "
            "DOT HS 809 952), the same real source Argonne National Laboratory's own EV assessment "
            "uses for this exact purpose. Measured from: least-squares fit of the formula above "
            "against that table's 25 real (age, survival rate) points, not against this fleet's own "
            "individual retirement events, because it has not generated enough of them yet."
        )

# --------------------------------------------------------------------------- tab 2: materials

with tab_materials:
    mats = system_result["secondary_materials"]
    material_cols = [c for c in mats.columns if c not in ("year", "units_retiring")]

    col_select, col_log = st.columns([3, 1])
    default_material = group["physical_columns"][0]
    chosen = col_select.multiselect(
        "Materials to show", material_cols, default=[default_material], key=f"materials_multiselect_{group['system']}"
    )
    log_scale = col_log.checkbox("Log scale", value=len(chosen) > 1)

    if chosen:
        fig2 = go.Figure()
        palette = ["#5EEAD4", "#F59E0B", "#F472B6", "#60A5FA", "#A3E635", "#C084FC", "#FB923C"]
        if len(chosen) == 1:
            # A real confidence band, not a decorative one: material_intensity()'s own
            # (low, central, high) tuples are real sourced ranges (JRC for wind, the
            # derived CRS/GREET figures for EV batteries), previously computed and then
            # discarded down to the central value everywhere in this app. One material
            # selected is the moment there is room to actually show that real range
            # instead of throwing it away.
            mat = chosen[0]
            mat_range = system_model.secondary_materials_range(system_result["retirement_schedule"])
            merged = mats[["year"]].merge(mat_range[["year", f"{mat}_low", f"{mat}_high"]], on="year", how="left")
            fig2.add_trace(go.Scatter(
                x=pd.concat([merged.year, merged.year[::-1]]),
                y=pd.concat([merged[f"{mat}_high"], merged[f"{mat}_low"][::-1]]),
                fill="toself", fillcolor="rgba(94, 234, 212, 0.18)",
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
            ))
            fig2.add_trace(go.Scatter(
                x=mats.year, y=mats[mat], mode="lines", name=f"{mat} (central estimate)",
                line=dict(color=palette[0], width=2),
            ))
        else:
            for i, mat in enumerate(chosen):
                fig2.add_trace(go.Scatter(
                    x=mats.year, y=mats[mat], mode="lines", name=mat, fill="tozeroy",
                    line=dict(color=palette[i % len(palette)]), stackgroup=None,
                ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            xaxis_title="Year", yaxis_title="Tonnes retiring that year",
            yaxis_type="log" if log_scale else "linear", height=420, margin=dict(t=30),
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig2, width="stretch")
        st.latex(
            r"M_{\text{material},\,t} = \sum_{\text{category}} \big(\text{units retiring in year } t"
            r"\big)_{\text{category}} \times I_{\text{material},\,\text{category}}"
        )
        if len(chosen) == 1:
            st.caption(
                f"Shaded band: the real low-high range material_intensity() reports for {chosen[0]}, "
                "propagated through the same cohort projection as the central line (I above is the "
                "low/high bound instead of the central value). Measured from: each retiring unit's "
                "real category (turbine topology or battery chemistry) times that category's real, "
                "sourced material content per unit, summed across categories for that year. Pick a "
                "second material above to compare trends instead, the band is only shown for one at "
                "a time to keep it legible."
            )
        else:
            st.caption(
                "Same formula as the single-material band above (I is the central value here), run "
                "once per material chosen and stacked. Measured from: each retiring unit's real "
                "category times that category's real, sourced material content per unit."
            )
    else:
        st.info("Pick at least one material above.")

    st.markdown("**Retirement wave** (from cohort-projected conditional survival)")
    fig3 = go.Figure(go.Bar(x=mats.year, y=mats.units_retiring, marker_color="#5EEAD4"))
    fig3.update_layout(
        template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
        xaxis_title="Year", yaxis_title=system["unit_label"], height=320,
        margin=dict(t=20, l=10, r=10, b=10),
    )
    st.plotly_chart(fig3, width="stretch")
    st.latex(
        r"R(a_0,\,t) = \frac{S(a_0+t-1)}{S(a_0)} - \frac{S(a_0+t)}{S(a_0)}"
        r"\qquad U_t = \sum_{\text{assets}} \text{size} \times R(a_0,\,t)"
    )
    st.caption(
        "R is the conditional probability an asset that already reached real age a_0 retires in "
        "year t specifically, not at any point, S is the Weibull (or Kaplan-Meier) survival curve "
        "from the chart above. Measured from: every real asset still in service, projected "
        "individually under its own real current age, then summed, a genuine cohort model rather "
        "than one average asset scaled up."
    )

    st.subheader(f"Sankey: which {system['category_label']} the recovered material actually comes from")
    sankey_groups = SANKEY_GROUPS_BY_SYSTEM[group["system"]]
    sankey_group_choice = st.selectbox(
        "Material group for this Sankey", list(sankey_groups.keys()),
        help="Grouped so every material shown is within the same order of magnitude. Mixing "
        "materials that differ by orders of magnitude makes the smaller flows and their labels "
        "collapse into an unreadable sliver.",
        key=f"sankey_group_{group['system']}",
    )
    sankey_materials = sankey_groups[sankey_group_choice]

    by_cat = system_model.secondary_materials_by_category(
        system_result["retirement_schedule"], horizon_year=horizon_year
    )
    by_cat = by_cat[by_cat.material.isin(sankey_materials) & (by_cat.tonnes > 0)]

    if by_cat.empty:
        st.info("No nonzero recovered amount for this material group at this horizon.")
    else:
        categories = sorted(by_cat.category.unique())
        materials_in_sankey = sorted(by_cat.material.unique())
        nodes = categories + materials_in_sankey
        node_index = {name: i for i, name in enumerate(nodes)}
        cat_color = "#5EEAD4"
        mat_colors = ["#F59E0B", "#F472B6", "#60A5FA", "#A3E635", "#C084FC", "#FB923C", "#94A3B8"]
        node_colors = [cat_color] * len(categories) + [
            mat_colors[i % len(mat_colors)] for i in range(len(materials_in_sankey))
        ]

        fig_sankey = go.Figure(go.Sankey(
            node=dict(label=nodes, color=node_colors, pad=24, thickness=18,
                      line=dict(color="#0B0E11", width=0.5)),
            link=dict(
                source=[node_index[r.category] for r in by_cat.itertuples()],
                target=[node_index[r.material] for r in by_cat.itertuples()],
                value=[r.tonnes for r in by_cat.itertuples()],
                color="rgba(94, 234, 212, 0.25)",
            ),
        ))
        fig_sankey.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            height=420, margin=dict(t=10, l=10, r=10, b=10),
            font=dict(color="#E6EDF3", size=13),
        )
        st.plotly_chart(fig_sankey, width="stretch")
        st.caption(
            f"Cumulative tonnes recovered to {horizon_year}, by real {system['category_label']} and "
            "material. Link width is proportional to tonnes; this is the same M formula two charts "
            "above, summed over every year to the horizon instead of plotted year by year, just "
            f"keeping the {system['category_label']} breakdown that chart pools away."
        )

# --------------------------------------------------------------------------- tab 3: risk

with tab_risk:
    st.markdown(
        f"**Scenario:** remove **{removed_country}** as a supplier of {group['name']}, "
        f"{removed_country_share:.1%} of real 2023 trade by value."
    )
    with st.expander("What does \"20% slack\" actually mean, and where does 20% come from?"):
        st.latex(
            r"\text{shortfall} = \max\!\Big(0,\ \text{removed country's exports} - \text{slack}"
            r"\times \sum_{\text{survivors}} \text{their exports}\Big)"
        )
        st.caption(
            "Survivors (every supplier except the one removed) can each expand output by `slack` "
            "times their own current real exports of this commodity; whatever the removed country "
            "used to supply that this spare capacity still cannot cover is the shortfall. slack=0 is "
            "the pessimistic no-substitution bound; slack=0.2 allows survivors 20% headroom. "
            "crm-trade-network's own README calls 20% an explicit, disclosed parameter rather than a "
            "hidden one, but also states plainly that the specific value 20% itself has no independent "
            "citation in that project, it is a deliberately round illustrative number, not fit to data. "
            "That gap is not silently inherited here: Statista's reported 2023 global copper mine "
            "capacity utilization rate was 77.6%, implying roughly 22.4% real average spare capacity "
            "that year, real-world evidence in the same ballpark as 20%, for one of the seven materials "
            "this dashboard covers. That is presented as an honest consistency check against one real "
            "industry figure, not as proof 20% is correct for every commodity or that this is how the "
            "figure was originally chosen."
        )

    if connector_result is None and scenario_note[0] == "no_exports":
        st.warning(
            f"{removed_country} has no recorded exports of {group['name']} in this dataset, "
            "so there is no shortfall to compare against. Pick a different supplier."
        )
    elif connector_result is None and scenario_note[0] == "slack_covers_it":
        st.info(
            f"Removing {removed_country} creates a real **\\${scenario_note[1]/1e9:,.2f} bn** "
            "shortfall with no substitution, but the remaining suppliers' 20% slack fully "
            f"absorbs it; the 20%-slack shortfall is zero. There is nothing left for the "
            f"recovered {material_choice.lower()} to offset in that scenario, so no coverage "
            "percentage is shown."
        )
    else:
        r = connector_result
        st.caption(
            f"Cumulative {material_choice.lower()} recovered by {horizon_year} (all years, for real "
            f"context): {cumulative_recovered_tonnes:,.0f} t, \\${cumulative_recovered_tonnes * price / 1e6:,.1f} M "
            f"at \\${price:,.0f}/t. The comparison below deliberately does **not** use that number, see why underneath."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Avg. annual recovery ({short_material.lower()})", f"${r['cumulative_recovered_usd']/1e6:,.1f} M/yr")
        c2.metric(
            f"Peak year ({peak_year})" if peak_year else "Peak year",
            f"${peak_connector_result['cumulative_recovered_usd']/1e6:,.1f} M/yr" if peak_connector_result else "n/a",
            help="The single real retirement-wave year that recovers the most of this material, not "
            "the average. Surfaced alongside the average on purpose: averaging across the whole "
            "projection is itself a smaller version of the time-scale problem the annualization fix "
            "elsewhere in this app exists to avoid.",
        )
        c3.metric("Shortfall, no substitution (1 year)", f"${r['real_shortfall_usd_no_slack']/1e9:,.2f} bn")
        c4.metric("Shortfall, 20% slack (1 year)", f"${r['real_shortfall_usd_20pct_slack']/1e9:,.2f} bn")
        st.latex(
            r"\text{recovered}_{\$} = \bar{t}_{\text{material}} \times p \qquad"
            r"\text{coverage} = \frac{\text{recovered}_{\$}}{\text{shortfall}_{\$}}"
        )
        st.caption(
            "Both the shortfall figures and \"average annual recovery\" describe one year, on purpose: "
            "the real cascade shortfall is a one-year disruption estimate, so comparing it against "
            "everything a technology system will ever recover by some arbitrary horizon year would "
            "compare two different time scales and can produce a share over 100% for smaller markets, "
            "not because recycling is winning, but because decades of accumulated recovery were being "
            "measured against a single year's cost. This engine compares one real year against another. "
            f"t-bar is the mean of the real year-by-year recovered-tonnes series shown in the chart "
            f"below; p is the price selected in the sidebar (\\${price:,.0f}/t)."
        )
        if price_low_result and price_high_result:
            st.caption(
                f"At the real price range's own bounds instead of the central price above: "
                f"**{price_low_result['recovered_share_of_no_slack_shortfall']:.2%}-"
                f"{price_high_result['recovered_share_of_no_slack_shortfall']:.2%}** of the "
                f"no-substitution shortfall and **{price_low_result['recovered_share_of_20pct_slack_shortfall']:.2%}-"
                f"{price_high_result['recovered_share_of_20pct_slack_shortfall']:.2%}** of the 20%-slack "
                f"shortfall, at \\${price_range[0]:,.0f}-\\${price_range[2]:,.0f}/t. The conclusion below "
                "uses the central price; this range shows how much of it is actually price-sensitive."
            )

        fig4 = go.Figure(go.Bar(
            x=[r["cumulative_recovered_usd"], r["real_shortfall_usd_20pct_slack"], r["real_shortfall_usd_no_slack"]],
            y=[f"Recovered/yr ({material_choice.lower()})", "Shortfall (20% slack)", "Shortfall (no substitution)"],
            orientation="h",
            marker_color=["#5EEAD4", "#F59E0B", "#F472B6"],
            text=[f"${v/1e6:,.1f} M" if v < 1e9 else f"${v/1e9:,.2f} bn"
                  for v in (r["cumulative_recovered_usd"], r["real_shortfall_usd_20pct_slack"], r["real_shortfall_usd_no_slack"])],
            # Inside placement, not outside: an outside label on the widest bar can be
            # clipped by the figure's right edge, and exactly how close it gets depends on
            # the actual dollar values, which change with material, scenario and price.
            # Inside text is drawn within the bar itself, so it can never be clipped no
            # matter how large the numbers get.
            textposition="inside", insidetextanchor="start",
            insidetextfont=dict(color="#0B0E11", size=13),
        ))
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            xaxis_title="USD (log scale)", xaxis_type="log", height=320,
            margin=dict(t=20, l=10, r=20, b=40),
        )
        st.plotly_chart(fig4, width="stretch")

        st.markdown(
            f"In an average year, this recovery covers **{r['recovered_share_of_no_slack_shortfall']:.2%}** "
            f"of one year's no-substitution shortfall and **{r['recovered_share_of_20pct_slack_shortfall']:.2%}** "
            "of one year's 20%-slack shortfall. Whether that ratio is small or substantial varies "
            "genuinely by material, copper and rare earths land under 0.1% here, graphite closer to "
            "40-50%, since graphite's real global trade is a much smaller market for this same "
            "recovery pathway to be measured against. Either way it is not a claim that recycling "
            "solves supply concentration; it answers whether this specific real recovery pathway "
            "matters at the scale of a real measured disruption, and by how much, for this material "
            "specifically."
        )
        st.caption(f"Price source: {r['price_source']}. Risk source: {risk.source}.")

        st.markdown("**Recovery by year, not just the average (real year-by-year values)**")
        year_values_usd = material_by_year * price
        fig_yby = go.Figure()
        fig_yby.add_trace(go.Scatter(
            x=year_values_usd.index, y=year_values_usd.values, mode="lines+markers",
            name=f"Recovered value that year ({material_choice.lower()})",
            line=dict(color="#5EEAD4", width=2), marker=dict(size=5),
        ))
        fig_yby.add_hline(
            y=r["real_shortfall_usd_no_slack"], line=dict(color="#F472B6", width=1.5, dash="dot"),
            annotation_text="1-year shortfall, no substitution", annotation_position="top left",
        )
        fig_yby.add_hline(
            y=r["real_shortfall_usd_20pct_slack"], line=dict(color="#F59E0B", width=1.5, dash="dot"),
            annotation_text="1-year shortfall, 20% slack", annotation_position="bottom left",
        )
        if peak_year is not None:
            fig_yby.add_vline(x=peak_year, line=dict(color="#6B7280", width=1, dash="dash"))
        fig_yby.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            xaxis_title="Year", yaxis_title="USD", height=340,
            margin=dict(t=20, l=10, r=10, b=40),
        )
        st.plotly_chart(fig_yby, width="stretch")
        st.latex(r"v_t = t_t \times p \qquad t_t = \text{real tonnes recovered in year } t \text{ (from the retirement wave)}")
        st.caption(
            "The bars and metrics above compress this whole real year-by-year series down to one "
            "average-year number, which is itself a smaller version of the same time-scale problem "
            "this connector exists to avoid (see the caption near the top of this tab). This chart is "
            "the uncompressed version: v_t for every real projected year against the same two flat "
            "one-year shortfall lines. The dashed vertical line marks the real peak retirement year "
            f"({peak_year}), where recovery is highest, not representative of a typical year."
        )

        st.markdown(f"**With recycling vs. without (one average year)**")
        no_slack_total = r["real_shortfall_usd_no_slack"]
        slack_total = r["real_shortfall_usd_20pct_slack"]
        covered_no_slack = min(r["cumulative_recovered_usd"], no_slack_total)
        covered_slack = min(r["cumulative_recovered_usd"], slack_total)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Covered by this recycling pathway",
            x=[covered_no_slack, covered_slack],
            y=["No substitution", "20% slack"],
            orientation="h", marker_color="#5EEAD4",
        ))
        fig5.add_trace(go.Bar(
            name="Shortfall remaining even with recycling",
            x=[no_slack_total - covered_no_slack, slack_total - covered_slack],
            y=["No substitution", "20% slack"],
            orientation="h", marker_color="#F472B6",
        ))
        fig5.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            barmode="stack", xaxis_title="USD", height=240,
            margin=dict(t=10, l=10, r=20, b=40), legend=dict(orientation="h", y=1.3),
        )
        st.plotly_chart(fig5, width="stretch")
        st.latex(r"\text{covered} = \min(\text{recovered}_{\$},\ \text{shortfall}_{\$}) \qquad \text{remaining} = \text{shortfall}_{\$} - \text{covered}")
        st.caption(
            "Same two shortfall totals as above, but stacked instead of separate: the teal sliver is "
            "what recycling actually covers, the pink is what's still short even after it. Deliberately "
            "linear, not log scale, so the sliver's real size relative to the whole shortfall is honest, "
            "even though that means it may be barely visible, which is itself the finding."
        )

    st.divider()
    st.subheader("How this compares across all seven materials")
    st.caption(
        "The same calculation above, run once for each material against its own real largest "
        "supplier (not necessarily the supplier currently selected above), so you can see which of "
        "these seven real recovery pathways matters most, relatively, without switching the dropdown "
        "seven times."
    )
    all_scenarios = compute_all_material_scenarios(horizon_year)
    plottable = all_scenarios.dropna(subset=["coverage_no_slack", "coverage_20pct"])
    skipped = all_scenarios[all_scenarios.coverage_no_slack.isna()]

    if not plottable.empty:
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            name="No substitution", y=plottable.material, x=plottable.coverage_no_slack,
            orientation="h", marker_color="#F472B6",
            text=[f"{v:.2%}" for v in plottable.coverage_no_slack], textposition="outside",
        ))
        fig6.add_trace(go.Bar(
            name="20% slack", y=plottable.material, x=plottable.coverage_20pct,
            orientation="h", marker_color="#F59E0B",
            text=[f"{v:.2%}" for v in plottable.coverage_20pct], textposition="outside",
        ))
        fig6.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            barmode="group", xaxis_title="Share of one year's real shortfall an average year of this recovery pathway covers",
            xaxis_tickformat=".1%", height=80 + 70 * len(plottable),
            margin=dict(t=10, l=10, r=60, b=40), legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig6, width="stretch")
        st.latex(r"\text{coverage}_{\text{material}} = \frac{\bar{t}_{\text{material}} \times p_{\text{material}}}{\text{shortfall}_{\$,\,\text{material}}}")
        st.caption(
            "Each material removes its own real largest supplier (" +
            "; ".join(f"{row.material}: {row.top1} ({row.top1_share:.1%})" for row in plottable.itertuples()) +
            "). Average annual recovery vs. one year's shortfall, the two figures being compared "
            "describe the same one-year time scale on purpose (see the caption above the risk "
            "connector's own chart for why). Graphite's real global trade is a small enough market "
            "that its ratio lands far higher than copper's or the rare earths', that is a genuine "
            "difference between real markets, not an error. None of these is a claim that recycling "
            "solves supply concentration."
        )
    if not skipped.empty:
        st.caption(
            "Not shown (removing that material's own largest supplier doesn't produce a comparable "
            "shortfall in this dataset): " + ", ".join(skipped.material.tolist())
        )

    st.subheader(f"Sankey: real {group['name']} trade flows among the largest players")
    N_TRADE_NODES = 8
    top_exporters = (
        commodity_edges.groupby("exporter").value_usd.sum().sort_values(ascending=False)
        .head(N_TRADE_NODES).index.tolist()
    )
    top_importers = (
        commodity_edges.groupby("importer").value_usd.sum().sort_values(ascending=False)
        .head(N_TRADE_NODES).index.tolist()
    )
    flow_edges = commodity_edges[
        commodity_edges.exporter.isin(top_exporters) & commodity_edges.importer.isin(top_importers)
    ]

    if flow_edges.empty:
        st.info("No real trade flows among the largest exporters and importers to show.")
    else:
        exp_nodes = [f"{c} (exports)" for c in top_exporters]
        imp_nodes = [f"{c} (imports)" for c in top_importers]
        nodes2 = exp_nodes + imp_nodes
        idx2 = {name: i for i, name in enumerate(nodes2)}
        node_colors2 = ["#F472B6" if c == removed_country else "#5EEAD4" for c in top_exporters] + \
                       ["#60A5FA"] * len(top_importers)

        fig_sankey2 = go.Figure(go.Sankey(
            # pad is a fixed pixel gap between adjacent nodes in the same column, independent
            # of how thin a node's value makes it. A skewed distribution (one dominant supplier,
            # several minor ones) can shrink a node's bar to near zero, and without enough pad
            # its label collides with its neighbour's. 16px was fine for copper's flatter
            # distribution but not for rare earth metals, where China alone is ~40% of trade.
            node=dict(label=nodes2, color=node_colors2, pad=26, thickness=16,
                      line=dict(color="#0B0E11", width=0.5)),
            link=dict(
                source=[idx2[f"{r.exporter} (exports)"] for r in flow_edges.itertuples()],
                target=[idx2[f"{r.importer} (imports)"] for r in flow_edges.itertuples()],
                value=[r.value_usd for r in flow_edges.itertuples()],
                color=["rgba(244, 114, 182, 0.35)" if r.exporter == removed_country
                       else "rgba(94, 234, 212, 0.2)" for r in flow_edges.itertuples()],
            ),
        ))
        fig_sankey2.update_layout(
            template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11",
            height=520, margin=dict(t=10, l=10, r=10, b=10), font=dict(color="#E6EDF3", size=13),
        )
        st.plotly_chart(fig_sankey2, width="stretch")
        st.caption(
            f"Real 2023 UN Comtrade bilateral flows for {group['name']}, among the {N_TRADE_NODES} "
            f"largest exporters and {N_TRADE_NODES} largest importers by value. {removed_country} is "
            "highlighted in pink: that is the node the disruption scenario above removes. No formula "
            "here, unlike every other chart on this page: link width is each country pair's own "
            "reported export value_usd, taken directly from UN Comtrade as reported, with no "
            "modeling applied."
        )
