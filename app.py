"""Streamlit dashboard for mfa_engine.

This wraps two things that are already verified on their own:

  1. mfa_engine.systems, three independent technology systems run through the same engine,
     one per textbook dMFA mode: WindTurbineMFA (register-based; reproduces dk-wind-mfa's own
     published numbers exactly, see tests/test_reproduces_dk_wind_mfa.py), EVBatteryMFA
     (inflow-driven; the global EV fleet, verified against a real external survival curve, see
     tests/test_ev_battery_mfa.py) and DataCentreMFA (stock-driven; required capacity from
     McKinsey's published range, see tests/test_data_centre.py). Genuinely different
     technology systems through the same unmodified base classes is the actual test of
     "modular and reusable", not just an assertion.
  2. mfa_engine.supply_risk_context, the connector that compares any system's physical
     material recovery against crm-trade-network's real 2023 UN Comtrade concentration and
     cascade-shortfall analysis.

Nothing in this app computes a new number on its own. It loads real output
from those pipelines, lets the viewer change a small number of real
parameters (projection horizon, which real commodity and supplier the
disruption scenario uses, the price used to convert tonnes to USD), and
visualises the result. If either sibling project is missing, the app says
so plainly and stops, rather than substituting placeholder data.

The risk connector covers every material a physical system actually
contains that also has a matching, trackable UN Comtrade commodity code:
copper and rare earth metals from wind turbines; copper, lithium, nickel,
cobalt and graphite from EV batteries; copper from data centres. That is every
one of Wu Chen's own named Villum grant materials (cobalt, copper, nickel,
lithium), and copper specifically is tracked from all three systems, so the
same real recovered tonne can be compared across technologies, the actual
point of connecting more than one system to this risk data in the first place.

Layout: a multipage app (st.navigation). The sidebar holds the few real parameters every page
shares (material, horizon, recovery assumption, disrupted supplier, price); the pages follow the
analysis in order, from how long one unit lasts to what a supply disruption would cost. The
visual system (validated colours, chart styling, shared page components) lives in ui/theme.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="mfa-engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="auto",   # open on desktop, tucked away on phones
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

# Streamlit Community Cloud only clones the one repo it's pointed at, with no build
# step of its own to run the Dockerfile's git clone in. So if neither the env var
# (Docker/Railway) nor the local dev path (this machine) resolves to a real checkout,
# clone both real, public sibling repos straight from GitHub into a cache directory at
# import time instead, and point the two path variables there. A shallow, idempotent
# clone of two small repos (data included) takes seconds, and re-runs harmlessly on
# every cold start since it skips straight past an already-populated directory.
def _ensure_cloned(repo_url: str, dest: Path) -> None:
    if (dest / ".git").exists():
        return
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)], check=True)


if not (DK_WIND_MFA_SRC / "load.py").exists() or not (CRM_TRADE_NETWORK_SRC / "build.py").exists():
    _cache = Path(tempfile.gettempdir()) / "mfa-engine-deps"
    try:
        if not (DK_WIND_MFA_SRC / "load.py").exists():
            _dk_dir = _cache / "dk-wind-mfa"
            _ensure_cloned("https://github.com/ibtisamkhan96/dk-wind-mfa.git", _dk_dir)
            DK_WIND_MFA_SRC = _dk_dir / "src"
        if not (CRM_TRADE_NETWORK_SRC / "build.py").exists():
            _crm_dir = _cache / "crm-trade-network"
            _ensure_cloned("https://github.com/ibtisamkhan96/crm-trade-network.git", _crm_dir)
            CRM_TRADE_NETWORK_SRC = _crm_dir / "src"
    except Exception:
        pass   # check_dependencies() below reports this plainly rather than crashing here


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

from mfa_engine import (  # noqa: E402
    TradeConcentrationRisk, recovery_vs_disruption,
    minimum_diversification, EU_CRMA_TARGET_SHARE, EU_CRMA_SOURCE,
    build_context, ask_dashboard,
)
from mfa_engine.systems.wind_turbine import load_and_build as load_wind  # noqa: E402
from mfa_engine.systems.ev_battery import load_and_build as load_ev, SNAPSHOT as EV_SNAPSHOT  # noqa: E402
from mfa_engine.scenarios import (  # noqa: E402
    compare_scenario_stocks, SCENARIOS, build_scenario, steps_ev_sales,
)
from mfa_engine.systems.data_centre import (  # noqa: E402
    load_and_build as load_dc, SNAPSHOT as DC_SNAPSHOT,
    compare_scenario_stocks as compare_dc_scenario_stocks, SCENARIOS as DC_SCENARIOS,
)
from mfa_engine.recovery import (  # noqa: E402
    REGIMES_BY_SYSTEM, recovered as apply_recovery, rate_for,
)
from ui.theme import (  # noqa: E402
    FAMILIES, MATERIAL_COLORS, MATERIAL_ORDER, SYSTEM_COLORS, SLATE, SLATE_LIGHT, NEUTRAL, CRITICAL, AXIS,
    esc, rgba, usd, inject_css, style_fig, show_chart, end_labels, download_csv, page_header, takeaway,
    sidebar_label, material_tile,
)
from ui.supply_map import build_payload as build_map_payload, load_supply, supply_map  # noqa: E402
from ui.guides import PAGE_GUIDES, READING_ANY_PAGE  # noqa: E402

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
        "snapshot": EV_SNAPSHOT,   # end of 2025, the last complete year of real sales; see ev_battery.SNAPSHOT
        "category_label": "battery chemistry",
        "unit_label": "Vehicles retiring",
    },
    "dc": {
        "label": "global data centres",
        "loader": load_dc,
        "snapshot": DC_SNAPSHOT,   # end of 2023, McKinsey's "current demand of 60 GW"; see data_centre.SNAPSHOT
        "category_label": "infrastructure layer",
        "unit_label": "Capacity retiring (MW)",
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

# Real per-material spare-capacity ("slack") figures, replacing a single flat 20% applied to
# every material regardless of whether 20% actually means anything for that specific one.
# crm-trade-network's own README calls 20% a "buried assumption" with no independent citation
# at all; live research this session found that copper is the ONE material here with a real,
# published, industry-association capacity-utilization statistic to replace it with. For the
# other five, the honest finding is not "we didn't look hard enough", each has a real,
# specific, structural reason no comparable figure is published anywhere:
#   - Cobalt: ~74% of real 2023 mine output was a copper byproduct (DR Congo) and ~7% a nickel
#     byproduct (Indonesia), per USGS Mineral Commodity Summaries 2024; byproduct metals are not
#     mined against their own dedicated capacity, so no cobalt-specific utilization exists.
#   - Rare earths: China manages supply through administrative mining and separation quotas
#     (240,000 t / 230,000 t REO respectively in 2023); USGS's own production figure for China
#     is that same quota by construction, so a utilization ratio cannot be derived from it.
#   - Lithium, graphite, nickel: no ICSG-equivalent industry body publishes a capacity-utilization
#     series for any of these the way ICSG does for copper (INSG publishes nickel production and
#     market balance, e.g. a real 2023 surplus of 163kt, but not capacity; USGS publishes only
#     production and reserves for lithium and graphite).
# 20% is kept for those five as the same illustrative, disclosed assumption crm-trade-network's
# own connector already uses, not silently replaced with an invented material-specific number.
_COPPER_REAL_SLACK = 0.224
_COPPER_REAL_SLACK_SOURCE = (
    "ICSG (International Copper Study Group), 2023 real global copper mine capacity "
    "utilization: 77.6% (a historic or near-historic low), implying 22.4% real spare capacity, "
    "republished via Statista; ICSG's own primary publications are subscription-only, so this "
    "is verified against the originating organization's methodology and republished figure, "
    "not the primary document itself."
)
_DEFAULT_SLACK = 0.20
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
        "real_slack": _COPPER_REAL_SLACK,
        "real_slack_source": _COPPER_REAL_SLACK_SOURCE,
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
        "real_slack": _COPPER_REAL_SLACK,
        "real_slack_source": _COPPER_REAL_SLACK_SOURCE,
    },
    "Copper (data centres)": {
        "system": "dc",
        "hs_code": "7403",
        "name": "refined copper",
        "physical_columns": ["copper"],
        "price_mode": "external",
        "external_price": _COPPER_PRICE_RANGE_2023[1],
        "external_price_range": _COPPER_PRICE_RANGE_2023,
        "external_price_source": _COPPER_PRICE_SOURCE,
        "real_slack": _COPPER_REAL_SLACK,
        "real_slack_source": _COPPER_REAL_SLACK_SOURCE,
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
        "real_slack": _DEFAULT_SLACK,
        "real_slack_note": "No real capacity-utilization figure exists for rare earths: China manages "
        "supply through administrative mining and separation quotas (240,000 t / 230,000 t REO in "
        "2023), and USGS's own reported production for China is that same quota by construction, "
        "so a real utilization ratio cannot be derived from it (USGS Mineral Commodity Summaries "
        "2024). 20% remains the same disclosed, illustrative assumption, not a citation.",
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
        "real_slack": _DEFAULT_SLACK,
        "real_slack_note": "No real capacity-utilization figure exists for lithium: no ICSG-equivalent "
        "industry body publishes one, and USGS Mineral Commodity Summaries 2024 reports only "
        "production and reserves. Real 2023 spot prices fell over 75% from their late-2022 peak, "
        "implying real spare capacity existed, but no source publishes a clean percentage for it. "
        "20% remains the same disclosed, illustrative assumption, not a citation.",
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
        "real_slack": _DEFAULT_SLACK,
        "real_slack_note": "No real capacity-utilization figure exists for nickel: INSG (International "
        "Nickel Study Group) publishes real production and market-balance data (a real 2023 global "
        "surplus of 163,000 t, per its own April 2024 release) but not a capacity-utilization series "
        "the way ICSG does for copper. 20% remains the same disclosed, illustrative assumption, not "
        "a citation.",
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
        "real_slack": _DEFAULT_SLACK,
        "real_slack_note": "No real capacity-utilization figure exists for cobalt, for a structural "
        "reason, not a lack of looking: about 74% of real 2023 mine output was a copper byproduct "
        "(DR Congo) and about 7% a nickel byproduct (Indonesia), per USGS Mineral Commodity "
        "Summaries 2024. Byproduct metals are not mined against their own dedicated capacity, so no "
        "cobalt-specific utilization statistic is published anywhere. 20% remains the same "
        "disclosed, illustrative assumption, not a citation.",
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
        "real_slack": _DEFAULT_SLACK,
        "real_slack_note": "No real capacity-utilization figure exists for graphite: USGS Mineral "
        "Commodity Summaries 2024 reports only production and reserves, and no ICSG-equivalent body "
        "tracks graphite capacity. The IEA's Global Critical Minerals Outlook 2024 notes only "
        "qualitatively that anode-grade graphite capacity outside China runs at low utilization, "
        "with no published percentage. 20% remains the same disclosed, illustrative assumption, "
        "not a citation.",
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
    "dc": {
        "Copper (facility power, cooling, grid connection)": ["copper"],
    },
}

# --------------------------------------------------------------------------- cached loaders

@st.cache_resource(show_spinner="Fitting the survival model on its own real register...")
def run_system_model(system_key: str, horizon_year: int):
    system = SYSTEMS[system_key]
    model = system["loader"](snapshot=system["snapshot"])
    result = model.run(horizon_year=horizon_year)
    return model, result


@st.cache_data(show_spinner="Running all three systems for the copper comparison...")
def compute_copper_across_systems(horizon_year: int) -> pd.DataFrame:
    """Copper stock-flow account for all three systems on the same basis: each system's own
    forward projection where one exists. The EV base run holds only real sales through 2026, so
    comparing it against the data-centre projection would show EVs adding no copper at all after
    2026, which is an artefact of the input, not a finding; IEA's STEPS trajectory is used
    instead. Denmark's register has no projection of new turbines yet, so wind shows its
    existing fleet only, and says so in its label."""
    runs = {
        "Danish wind turbines (existing fleet only)": run_system_model("wind", horizon_year)[1],
        "global EV batteries (IEA STEPS)": build_scenario(steps_ev_sales(horizon_year)).run(horizon_year),
        "global data centres (McKinsey low)": run_system_model("dc", horizon_year)[1],
    }
    frames = []
    for label, res in runs.items():
        cu = res["material_stock_flows"]
        cu = cu[cu.material == "copper"].copy()
        cu["system"] = label
        frames.append(cu)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Running both deployment scenarios through the same engine...")
def compute_scenario_stocks(system_key: str, horizon_year: int) -> pd.DataFrame:
    if system_key == "dc":
        return compare_dc_scenario_stocks(horizon_year)
    return compare_scenario_stocks(horizon_year)


# Per-system wording for the scenario section, so one piece of code serves both systems that
# have real scenario pairs. "names" is (upper scenario, baseline), the order the gap is quoted in.
SCENARIO_UI = {
    "ev": {
        "names": list(SCENARIOS.keys()),
        "upper_short": "STEPS",
        "base_short": "the flat-2026 floor",
        "title": "Deployment scenarios: does the future sales trajectory actually matter?",
        "intro": (
            "The base run on every other page uses the real historical/current EV sales data alone (2019-2026, "
            "IEA Global EV Outlook). This asks a different question: if sales keep growing along "
            "IEA's own named Stated Policies Scenario (STEPS, reaching roughly 50% of global car "
            "sales by 2035) instead of flatlining at the 2026 level, how much does that change the "
            "stock in use and the material coming back out, run through the exact same engine, "
            "only the future sales input differs."
        ),
        "slider_help": (
            "Kept separate from the main horizon slider on purpose: EV battery median lifetime is "
            "around 13 years, so a deployment scenario only shows up in end-of-life flows once "
            "enough time has passed for the additional sales to retire."
        ),
        "radio_help": (
            "Stock is where a deployment choice shows up first. End-of-life outflow, and the "
            "recovered share of it, only reflect it once those vehicles are old enough to retire."
        ),
        "closing": (
            "The two lines are identical through the real sales years (2019-2026) by construction "
            "and diverge only afterward. In-use stock diverges almost immediately; end-of-life "
            "outflow lags it by roughly one battery lifetime (median about 13 years), which is why a "
            "deployment choice made now barely shows up in recycling until the late 2030s. Sources: "
            "mfa_engine/scenarios.py; sales anchors are real and cited (IEA Global EV Outlook 2026). "
            "The flat future car market, the post-2026 chemistry mix, and holding STEPS at its real "
            "2035 endpoint afterwards are disclosed assumptions, not citations."
        ),
    },
    "dc": {
        "names": [DC_SCENARIOS[1], DC_SCENARIOS[0]],
        "upper_short": "the high case",
        "base_short": "the low case",
        "title": "Capacity scenarios: how much does the 2030 build-out range matter?",
        "intro": (
            "McKinsey (October 2024) puts global data-centre capacity demand at 171 to 219 GW by "
            "2030, up from 60 GW in 2023. This runs both ends of that published range through the "
            "same stock-driven engine: identical history, identical lifetimes and copper content, "
            "only the required capacity differs."
        ),
        "slider_help": (
            "Facility equipment lasts about 20 years and grid connections about 30, so the "
            "difference in end-of-life copper only appears once the 2020s build-out starts to retire."
        ),
        "radio_help": (
            "In-use stock reflects the capacity choice immediately. End-of-life outflow, and the "
            "recovered share of it, only reflect it one equipment lifetime later."
        ),
        "closing": (
            "The two lines match through 2023 by construction. The gap in copper in use is the "
            "capacity gap itself, since copper per MW is the same in both; the gap in end-of-life "
            "copper opens decades later, because what is built for AI in the 2020s mostly retires "
            "in the 2040s and 2050s. Sources: mfa_engine/systems/data_centre.py (McKinsey 2024, IEA "
            "Energy and AI 2025, Masanet et al. 2020, WEF/Kearney 2025, ASHRAE)."
        ),
    },
}


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


@st.cache_data(show_spinner="Running the cascading-failure simulation...")
def compute_fragility(edges: pd.DataFrame, country: str, beta: float = 0.2) -> dict:
    """A second, dynamic view of the same disruption, using crm-trade-network's
    shock_propagation module (added there to replicate, at single-trade-layer
    resolution, the linear-threshold cascade Wu Chen's group runs for cobalt:
    Ouyang, Liu, Liu, Chen, Wang, Pang, He, Liu, Environ. Sci. Ecotechnol. 29,
    2026, 100654). compute_cascade above asks a static question, how much trade
    value does this country directly supply; this asks a dynamic one, once that
    country is gone, how many OTHER countries does the disruption itself take
    down, round by round, as their own trade partners collapse in turn."""
    from network import build_graph
    from shock_propagation import simulate_cascade

    G = build_graph(edges)
    return simulate_cascade(G, country, beta=beta)


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
def compute_all_material_scenarios(horizon_year: int, recovery_regime: str) -> pd.DataFrame:
    """The same recovery_vs_disruption calculation the sidebar runs for
    whichever one material is currently selected, run for all seven at once,
    each against its own real top single supplier. Lets a viewer see which
    of these real recovery pathways matters most, relatively, without
    clicking through seven separate dropdown states one at a time. Each
    system uses the selected recovery regime only where that regime actually
    applies to it (the EU Batteries Regulation covers batteries, not wind
    turbines), falling back to current practice otherwise."""
    rows = []
    for name, g in MATERIAL_GROUPS.items():
        regime = (recovery_regime if recovery_regime in REGIMES_BY_SYSTEM[g["system"]]
                  else REGIMES_BY_SYSTEM[g["system"]][0])
        start_year = SYSTEMS[g["system"]]["snapshot"].year
        try:
            conc, edges = load_commodity_trade(g["hs_code"])
        except ValueError:
            continue
        _, result = run_system_model(g["system"], horizon_year)
        total_usd = float(edges.value_usd.sum())
        top1 = conc["top1"]
        no_slack = compute_cascade(edges, top1, 0.0)
        slack20 = compute_cascade(edges, top1, g["real_slack"])
        share = no_slack["lost_usd"] / total_usd if total_usd else 0.0

        if g["price_mode"] == "external":
            price, price_source = g["external_price"], g["external_price_source"]
        else:
            implied = implied_unit_price(edges)
            price, price_source = (implied or 0.0), "implied 2023 UN Comtrade unit value"

        mat_by_year = (apply_recovery(result["secondary_materials"], regime, "central")
                       .set_index("year")[g["physical_columns"]].sum(axis=1, min_count=1))
        mat_by_year = mat_by_year[(mat_by_year.index >= start_year) & (mat_by_year.index <= horizon_year)]
        if mat_by_year.isna().all():
            rows.append({"material": name, "top1": top1, "top1_share": share,
                         "coverage_no_slack": None, "coverage_20pct": None,
                         "real_slack": g["real_slack"], "note": "no cited recycling rate"})
            continue
        # Annualized, not cumulative: the real cascade shortfall is a one-year disruption
        # estimate, so it must be compared against one typical year of this recovery pathway,
        # not everything a system will ever recover by an arbitrary horizon year. See the same
        # comment where this connector is called for the currently-selected material for the
        # real failure mode this avoids (a >1000% "coverage" share for smaller global markets).
        annualized = pd.Series([float(mat_by_year.mean())]) if len(mat_by_year) else pd.Series([0.0])

        if no_slack["shortfall_usd"] <= 0 or slack20["shortfall_usd"] <= 0:
            rows.append({"material": name, "top1": top1, "top1_share": share,
                         "coverage_no_slack": None, "coverage_20pct": None,
                         "real_slack": g["real_slack"]})
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
            "real_slack": g["real_slack"],
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


# --------------------------------------------------------------------------- sidebar: material

# The sidebar asks for a material first and then, for copper only, which system it is found in:
# copper is the one material all three systems contain, so it is the only material where that
# second question means anything.
FAMILY_TO_GROUP = {
    "Lithium": "Lithium", "Nickel": "Nickel", "Cobalt": "Cobalt", "Graphite": "Graphite",
    "Rare earths": "Rare earth metals",
}
COPPER_IN = {
    "Wind": "Copper (wind turbines)",
    "EVs": "Copper (EV batteries)",
    "Data centres": "Copper (data centres)",
}
FOUND_IN = {"wind": "in Danish wind turbines", "ev": "in the global EV fleet", "dc": "in global data centres"}
UNIT_WORDS = {"wind": "turbines", "ev": "EV batteries", "dc": "installed equipment"}

ASSETS = Path(__file__).resolve().parent / "assets"
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")

tile_slot = st.sidebar.empty()
family = st.sidebar.pills(
    "Material", list(FAMILIES), default="Copper", required=True, key="material",
    bind="query-params",
    help="Every material covered here is one a real physical system (Danish wind turbines, the "
    "global EV fleet, global data centres) actually contains AND has a real, trackable UN "
    "Comtrade commodity code. Bulk structural materials (steel, concrete) have no CRM trade code; "
    "lithium, cobalt, nickel and graphite have real trade data but only EV batteries contain them.",
)
if family == "Copper":
    copper_in = st.sidebar.segmented_control(
        "Found in", list(COPPER_IN), default="Wind", required=True, key="in",
        bind="query-params", width="stretch",
        help="Copper is the one material all three systems contain, so the same real recovered "
        "tonne can be compared across technologies.",
    )
    material_choice = COPPER_IN[copper_in]
else:
    material_choice = FAMILY_TO_GROUP[family]

group = MATERIAL_GROUPS[material_choice]
system = SYSTEMS[group["system"]]
# The copper entries carry a "(wind turbines)"-style qualifier that fixed-width labels truncate;
# short_material drops it where the system is already shown next to it.
short_material = material_choice.split(" (")[0]
ACCENT = FAMILIES[family]["color"]
inject_css(ACCENT)
tile_slot.html(material_tile(
    family, short_material,
    ("Nd, Dy, Pr and Tb " if family == "Rare earths" else "") + FOUND_IN[group["system"]],
))
CHIP = f"{short_material} \u00b7 {system['label']}"

# Page links are drawn here, under the material picker and above the settings, rather than by
# st.navigation's own sidebar menu: that menu always sits at the very top of the sidebar, and with
# twelve pages it pushed the material picker, the one control every page depends on, below the
# fold of a laptop screen. The slot is filled at the end of the script, once the pages exist.
nav_slot = st.sidebar.container(gap=None)

# --------------------------------------------------------------------------- sidebar: projection

sidebar_label("Projection")
horizon_year = st.sidebar.slider(
    "Horizon", min_value=2025, max_value=2060, value=2050, step=1, key="horizon",
    bind="query-params",
    help="How far to project the retirement wave forward from the selected system's own real snapshot date.",
)

REGIME_SHORT = {
    "Current practice (UNEP IRP 2011)": "Current practice",
    "EU Batteries Regulation targets (upper bound)": "EU 2031 battery targets",
}
REGIME_CAPTION = {
    "Current practice (UNEP IRP 2011)": "Measured rates, UNEP IRP 2011",
    "EU Batteries Regulation targets (upper bound)": "Upper bound: every battery collected",
}
regimes = REGIMES_BY_SYSTEM[group["system"]]
if len(regimes) > 1:
    recovery_regime = st.sidebar.radio(
        "End-of-life recovery", regimes, format_func=REGIME_SHORT.get,
        captions=[REGIME_CAPTION[r] for r in regimes], key=f"recovery_regime_{group['system']}",
        help="How much of the material leaving service is actually recycled back into use. "
        "Current practice uses the UN International Resource Panel's measured end-of-life recycling "
        "rates (2011): over 50% for copper, nickel, cobalt; under 1% for lithium and every rare earth. "
        "The EU Batteries Regulation option applies its 2031 recovery targets (lithium 80%, cobalt and "
        "nickel 95%) and assumes every battery is collected, an upper bound, not a forecast. It is only "
        "offered for EV batteries, since the Regulation does not cover wind turbines.",
    )
else:
    recovery_regime = regimes[0]
    st.sidebar.caption(
        "End-of-life recovery: **current practice**, the UN International Resource Panel's measured "
        "recycling rates (2011). The EU battery targets apply to EV batteries only."
    )

# --------------------------------------------------------------------------- sidebar: disruption

try:
    commodity_conc, commodity_edges = load_commodity_trade(group["hs_code"])
except ValueError as exc:
    st.error(str(exc))
    st.stop()

total_trade_usd = float(commodity_edges.value_usd.sum())
_by_exporter = commodity_edges.groupby("exporter").value_usd.sum()
top1_unified_share = float(_by_exporter.get(commodity_conc["top1"], 0.0) / total_trade_usd) if total_trade_usd else 0.0
exporters = (
    commodity_edges.groupby("exporter").value_usd.sum().sort_values(ascending=False).head(15).index.tolist()
)

sidebar_label("Disruption scenario")
default_country_idx = exporters.index(commodity_conc["top1"]) if commodity_conc["top1"] in exporters else 0
removed_country = st.sidebar.selectbox(
    "Supplier removed", exporters, index=default_country_idx,
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

with st.sidebar.expander("Price assumption", icon=":material/payments:"):
    price = st.number_input(
        "USD per tonne", min_value=100.0, max_value=500_000.0,
        value=default_price, step=50.0,
        help=f"Used only to convert recovered tonnes to USD. Default: {default_price_source}.",
        key=f"price_input_{material_choice}",
    )
    if price_range:
        # \$ (escaped), not $: st.caption renders markdown, and two or more literal "$" in one
        # string get auto-rendered as LaTeX between them (Streamlit's KaTeX integration), mangling
        # the text. See the comment on _COPPER_PRICE_SOURCE for where this was actually caught.
        st.caption(
            f"Real price range, not a point value: **\\${price_range[0]:,.0f} - \\${price_range[2]:,.0f}/t** "
            f"(central \\${price_range[1]:,.0f}/t used above), {price_range_source}. The disruption page "
            "reports coverage at all three, the same (low, central, high) treatment already given "
            "to physical material content."
        )
    elif "external_context" in group:
        # No real range for this material (see the comment on MATERIAL_GROUPS above for why), but a
        # real external citation was still found and is shown here, disclosed as context rather than
        # silently dropped or forced into a range that wouldn't bracket the central estimate above.
        st.caption(group["external_context"])
price_source = default_price_source if price == default_price else "user-entered override"

fresh = cache_freshness(2023, group["hs_code"])
st.sidebar.divider()
st.sidebar.html(
    '<div class="sb-foot">'
    + (f"Trade data: {fresh['count']} UN Comtrade queries for {esc(group['name'])}, last fetched "
       f"{fresh['newest']:%d %b %Y}. " if fresh["count"]
       else f"No cached {esc(group['name'])} trade data on disk yet. ")
    + "Every figure is computed from cited real data; nothing is simulated. Sources and a live "
    "refresh are on <b>Methods &amp; data</b>.</div>"
    # Streamlit's own navigation closes the sidebar after a tap on a phone; these custom page links
    # would leave it open over the page, so the same behaviour is added here (narrow screens only).
    + "<script>(() => { if (window.__mfaNavCloser) return; window.__mfaNavCloser = true;"
    " document.addEventListener('click', (ev) => {"
    " const link = ev.target.closest('[data-testid=\"stSidebar\"] [data-testid=\"stPageLink-NavLink\"]');"
    " if (!link || window.innerWidth > 768) return;"
    " const btn = document.querySelector('[data-testid=\"stSidebarCollapseButton\"] button');"
    " if (btn) setTimeout(() => btn.click(), 60); }, true); })();</script>",
    unsafe_allow_javascript=True,
)

# --------------------------------------------------------------------------- run pipelines

system_model, system_result = run_system_model(group["system"], horizon_year)
cascade_no_slack = compute_cascade(commodity_edges, removed_country, 0.0)
cascade_20pct = compute_cascade(commodity_edges, removed_country, group["real_slack"])
removed_country_share = cascade_no_slack["lost_usd"] / total_trade_usd if total_trade_usd else 0.0
fragility = compute_fragility(commodity_edges, removed_country, beta=0.2)
fragility_universe = pd.concat([commodity_edges.exporter, commodity_edges.importer]).nunique()

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

projection_start_year = system["snapshot"].year


def _projection_window(series: pd.Series) -> pd.Series:
    # The EV system is inflow-driven, so its account starts at the first real sales year
    # (2019), not the snapshot. The connector compares one typical PROJECTED year against one
    # real year of disruption, so it averages from the snapshot year onward only; the
    # modelled pre-snapshot years belong to history, not to the projection.
    return series[(series.index >= projection_start_year) & (series.index <= horizon_year)]


# Gross end-of-life outflow vs what is actually recovered: the dashboard used to treat every
# tonne leaving service as recovered. That overstated copper recovery by roughly half and
# lithium or rare earth recovery by more than a hundredfold, since their real end-of-life
# recycling rates are under 1% (UNEP IRP 2011). mfa_engine.recovery applies the real, cited
# rate for the selected regime; everything described as "recovered" below uses it.
gross_outflow = system_result["secondary_materials"]
recovered_central = apply_recovery(gross_outflow, recovery_regime, "central")
outflow_by_year = _projection_window(
    gross_outflow.set_index("year")[group["physical_columns"]].sum(axis=1))
material_by_year = _projection_window(
    recovered_central.set_index("year")[group["physical_columns"]].sum(axis=1, min_count=1))
recovery_assessed = bool(material_by_year.notna().any())
material_rates = {m: rate_for(m, recovery_regime) for m in group["physical_columns"]}
if not recovery_assessed:
    material_by_year = material_by_year.fillna(0.0)
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

recovery_low_result = recovery_high_result = None
if not recovery_assessed:
    # A real, distinct outcome, not a zero: no cited end-of-life recycling rate exists for
    # this material (graphite is outside UNEP IRP 2011's scope, and the EU Batteries
    # Regulation sets it no target), so no honest recovered tonnage can be computed.
    connector_result = None
    peak_connector_result = None
    price_low_result = None
    price_high_result = None
    scenario_note = ("no_recovery_rate", None)
elif no_slack_shortfall <= 0:
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
    # Coverage at the recycling rate's own low and high bound: most rates are published as
    # bins (UNEP's ">50%", "<1%"), so the central value is a disclosed bin midpoint and this
    # range is the honest spread around it.
    for bound in ("low", "high"):
        rec_bound = _projection_window(apply_recovery(gross_outflow, recovery_regime, bound)
                                       .set_index("year")[group["physical_columns"]].sum(axis=1))
        res_bound = recovery_vs_disruption(pd.Series([float(rec_bound.mean())]), price, price_source, risk)
        if bound == "low":
            recovery_low_result = res_bound
        else:
            recovery_high_result = res_bound
    scenario_note = None

layer_lifetimes = system_result.get("layer_lifetimes")
if layer_lifetimes:
    weibull_median = max(v["median"] for v in layer_lifetimes.values())
else:
    weibull_median = system_result["weibull_scale"] * np.log(2) ** (1 / system_result["weibull_shape"])

# Diversification only needs the trade data, not a recovery rate, so it is computed for every
# material (it used to be skipped whenever the recovery connector had nothing to report). Its
# target slider lives on the Diversification page; reading the slider's own session-state key
# here keeps the number the Ask page is grounded in identical to the one on that page, and falls
# back to the EU CRMA benchmark when the page has not been opened.
_TARGET_DEFAULT_PCT = int(round(EU_CRMA_TARGET_SHARE * 100))
target_share = st.session_state.get(f"target_share_{material_choice}", _TARGET_DEFAULT_PCT) / 100.0
div_no_slack = minimum_diversification(commodity_edges, target_share, 0.0)
div_slack = minimum_diversification(commodity_edges, target_share, group["real_slack"])

# The selected material's own stock-flow account, pooled over its physical columns (four for the
# rare-earth basket, one otherwise), for the headline figures on the overview.
_msf_group = system_result["material_stock_flows"]
_msf_group = _msf_group[_msf_group.material.isin(group["physical_columns"])].groupby("year").sum(numeric_only=True)
stock_at_start = float(_msf_group.stock_t.get(projection_start_year, 0.0))
outflow_peak_year = int(outflow_by_year.idxmax()) if len(outflow_by_year) and outflow_by_year.max() > 0 else None


# --------------------------------------------------------------------------- page scaffolding

FLOW = ["overview", "lifetimes", "stocks", "eol", "scenarios", "supplymap", "disruption", "cascade",
        "diversification", "copper", "materials"]
SECTION = {
    "overview": "Overview", "lifetimes": "Physical flows", "stocks": "Physical flows",
    "eol": "Physical flows", "scenarios": "Physical flows", "supplymap": "Supply risk", "disruption": "Supply risk",
    "cascade": "Supply risk", "diversification": "Supply risk", "copper": "Compare",
    "materials": "Compare", "ask": "Tools", "methods": "Reference",
}


@st.dialog("How to read this page", width="large")
def show_guide(key: str) -> None:
    guide = PAGE_GUIDES[key]
    st.markdown(f"#### {PAGES[key].title}")
    st.markdown(f"**What it shows.** {guide['shows']}")
    st.markdown(f"**How to read it.** {guide['read']}")
    st.markdown(f"**Method, in brief.** {guide['method']}")
    st.markdown(f"**Try this.** {guide['try']}")
    st.divider()
    st.markdown("#### Reading any page")
    st.markdown(READING_ANY_PAGE)
    st.caption("Full sources, assumptions and what this dashboard deliberately does not claim are on the "
               "Methods & data page.")


def header(key: str, title: str, lede: str) -> None:
    """Page title block, with the "How to read this page" button at its top right. On a phone
    the two columns stack and the button follows the introduction."""
    where = (f"{SECTION[key]} \u00b7 step {FLOW.index(key)} of {len(FLOW) - 1}"
             if key in FLOW[1:] else SECTION[key])
    head, guide = st.columns([5, 1.7], vertical_alignment="top")
    with head:
        page_header(where, title, lede, CHIP)
    with guide:
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button("How to read this page", key=f"guide_{key}", icon=":material/menu_book:",
                         help="What this page shows, how to read its charts, and the method in brief."):
                show_guide(key)


def footer(key: str) -> None:
    """Previous / next links in reading order, so the analysis can be read start to finish
    without going back to the sidebar."""
    st.space("medium")
    st.divider()
    left, right = st.columns(2)
    if key in FLOW:
        i = FLOW.index(key)
        prev_key = FLOW[i - 1] if i > 0 else None
        next_key = FLOW[i + 1] if i < len(FLOW) - 1 else None
    else:
        prev_key, next_key = "overview", None
    if prev_key:
        left.page_link(PAGES[prev_key], label=f"Back: {PAGES[prev_key].title}", icon=":material/arrow_back:")
    if next_key:
        with right.container(horizontal=True, horizontal_alignment="right"):
            st.page_link(PAGES[next_key], label=f"Next: {PAGES[next_key].title}",
                         icon=":material/arrow_forward:", icon_position="right")


def how_calculated():
    return st.expander("How this is calculated", icon=":material/functions:")


def _pct(x: float) -> str:
    return f"{x:.2%}"


def _plural(label: str, n: int) -> str:
    """'turbine topology' -> 'turbine topologies', 'infrastructure layer' -> 'infrastructure layers'."""
    if n == 1:
        return label
    return label[:-1] + "ies" if label.endswith("y") and label[-2:-1] not in "aeiou" else label + "s"


def _at_edge(year: int, series: pd.Series) -> bool:
    """True when a 'peak' is only the last year shown, i.e. the series is still rising at the horizon."""
    return len(series) > 1 and year == int(series.index.max()) and series.iloc[-1] > series.iloc[-2]


HOW_TO_READ = (
    "1. **Pick a material in the sidebar.** It decides which real system runs (Danish wind "
    "turbines, the global EV fleet or global data centres) and which UN Comtrade commodity its "
    "recovered material is compared against. Copper is in all three systems, so for copper you "
    "also choose where it is found.\n"
    "2. **Read the pages in order.** *Physical flows*: how long each unit lasts, the stock-flow "
    "account, what comes back out at end of life, and how a deployment scenario changes it. "
    "*Supply risk*: what losing one supplier costs, how that shock spreads through the trade "
    "network, and what diversifying would take. *Compare*: copper across all three "
    "technologies, then every material side by side. Each page links to the next one at the "
    "bottom.\n"
    "3. **Every chart leads with one sentence**, computed from the numbers on screen. The "
    "formula, the data it is measured from and every assumption sit in the *How this is "
    "calculated* panel under the chart, and each chart's data downloads as CSV."
)


def _headline() -> str:
    """The dashboard's one-line answer for the selected material and supplier."""
    mat = esc(short_material.lower())
    who = esc(removed_country)
    if connector_result is not None:
        r = connector_result
        return (
            f"In an average year to {horizon_year}, recycled {mat} from {esc(system['label'])} would "
            f"cover <b>{_pct(r['recovered_share_of_no_slack_shortfall'])}</b> of one year's supply "
            f"shortfall if <b>{who}</b> stopped exporting ({removed_country_share:.1%} of world trade), "
            f"or <b>{_pct(r['recovered_share_of_20pct_slack_shortfall'])}</b> if the other suppliers "
            f"used {group['real_slack']:.1%} spare capacity."
        )
    if scenario_note[0] == "no_recovery_rate":
        return (
            f"No cited end-of-life recycling rate exists for {mat}, so no recovered tonnage is "
            f"reported: <b>{float(outflow_by_year.mean()):,.0f} t/yr</b> leaves service on average, "
            f"{projection_start_year}-{horizon_year}, with no measured rate for how much comes back."
        )
    if scenario_note[0] == "slack_covers_it":
        return (
            f"Losing <b>{who}</b> opens a <b>{esc(usd(scenario_note[1]))}</b> gap with no substitution, "
            f"but the other suppliers' {group['real_slack']:.1%} spare capacity closes it entirely, "
            "so there is no shortfall left for recycling to cover."
        )
    return (f"<b>{who}</b> has no recorded exports of {esc(group['name'])} in this dataset, so "
            "removing it creates no shortfall. Pick a different supplier in the sidebar.")


# --------------------------------------------------------------------------- pages

def page_overview():
    header(
        "overview", f"{short_material} {FOUND_IN[group['system']]}",
        f"How long these assets last, how much {esc(short_material.lower())} comes back out when they "
        "retire, and whether that recovery matters against a real, measured supply disruption. "
        "Change the material in the sidebar and every page follows it.",
    )
    takeaway(_headline())

    m1, m2, m3, m4 = st.columns(4)
    if layer_lifetimes:
        m1.metric("Median lifetimes (by layer)",
                  " / ".join(f"{v['median']:.0f}" for v in layer_lifetimes.values()) + " yr",
                  delta=" / ".join(name.split()[0] for name in layer_lifetimes), delta_color="off", delta_arrow="off",
                  help="; ".join(f"{name}: {v['median']:.0f} years" for name, v in layer_lifetimes.items()),
                  border=True)
    else:
        m1.metric("Weibull median lifetime", f"{weibull_median:,.1f} yr", border=True,
                  delta="half retired by this age", delta_color="off", delta_arrow="off",
                  help="The age by which half of all units have retired, from the fitted survival curve.")
    m2.metric(
        f"Recovered, {projection_start_year}-{horizon_year}",
        f"{cumulative_recovered_tonnes:,.0f} t" if recovery_assessed else "not assessed",
        delta=f"of {float(outflow_by_year.sum()):,.0f} t retired", delta_color="off", delta_arrow="off",
        help=f"Tonnes of {short_material.lower()} actually recycled back into use under \"{recovery_regime}\", "
        f"out of {float(outflow_by_year.sum()):,.0f} t leaving service over the same years. The gap is "
        "material lost at end of life: not collected, dissipated, or downcycled.",
        border=True,
    )
    hhi = float(commodity_conc["hhi"])
    m3.metric("Concentration (HHI)", f"{hhi:.3f}",
              delta="highly concentrated" if hhi > 0.25 else "under the 0.25 threshold",
              delta_color="off", delta_arrow="off", border=True,
              help="Herfindahl-Hirschman index of 2023 supplier shares by trade value, all importers "
              "pooled (crm-trade-network): the sum of squared shares. 1.0 is a single source, 0.1 is "
              "roughly ten equal ones; competition authorities treat 0.25 as highly concentrated.")
    m4.metric("Largest supplier's share", f"{commodity_conc['top1_share']:.1%}",
              delta=commodity_conc["top1"], delta_color="off", delta_arrow="off",
              border=True, help="The largest single supplier's share of 2023 world trade by value, "
              "UN Comtrade via crm-trade-network. Like the HHI, it is measured on importer-reported "
              "flows; the disruption pages use crm-trade-network's unified edge list, built from both "
              "importer and exporter reports (the larger of the two where they disagree), where "
              f"{commodity_conc['top1']} holds {top1_unified_share:.1%}.")

    st.subheader("Follow the analysis")
    if connector_result is not None:
        cov_fig = f"{_pct(connector_result['recovered_share_of_no_slack_shortfall'])} covered"
    else:
        cov_fig = {"no_recovery_rate": "no cited rate", "slack_covers_it": "slack absorbs it",
                   "no_exports": "no shortfall"}[scenario_note[0]]
    cards = [
        ("lifetimes", "01", "How long units last", f"{weibull_median:.0f} yr median",
         "The survival curve every projected retirement rests on."),
        ("stocks", "02", "Stocks and flows", f"{stock_at_start:,.0f} t in use, {projection_start_year}",
         f"{short_material} entering service, in use and leaving it, balanced every year."),
        ("eol", "03", "What comes back out",
         ("no outflow" if not outflow_peak_year
          else f"rising to {outflow_peak_year}" if _at_edge(outflow_peak_year, outflow_by_year)
          else f"peaks in {outflow_peak_year}"),
         "Material leaving service each year, and which units it comes from."),
        ("disruption", "04", "Disruption and recovery", cov_fig,
         f"Recycling measured against losing {removed_country} as a supplier for one year."),
    ]
    for col, (key, num, title, figure, desc) in zip(st.columns(4), cards):
        with col.container(border=True, height="stretch"):
            st.html(f'<div class="step-card"><div class="num">{num}</div><div class="title">{esc(title)}</div>'
                    f'<div class="fig">{esc(figure)}</div><div class="desc">{esc(desc)}</div></div>')
            st.page_link(PAGES[key], label="Open", icon=":material/arrow_forward:", icon_position="right")

    st.subheader("Go further")
    with st.container(horizontal=True, gap="small"):
        for key in ("supplymap", "scenarios", "cascade", "diversification", "copper", "materials", "ask", "methods"):
            st.page_link(PAGES[key], label=PAGES[key].title, icon=PAGES[key].icon)

    footer("overview")


def page_lifetimes():
    header(
        "lifetimes", "How long each unit lasts",
        "The survival curve behind every projected retirement: the share of units still in service "
        "at each age. Everything downstream, the retirement wave, end-of-life material and "
        "recovery, is timed by this curve.",
    )
    fig = go.Figure()
    if layer_lifetimes:
        takeaway("One lifetime per infrastructure layer, not one per site: "
                 + "; ".join(f"{esc(name)} lasts a median <b>{v['median']:.0f} years</b>"
                             for name, v in layer_lifetimes.items())
                 + ". Worn-out equipment is replaced on site, which is why this system keeps "
                 "consuming copper after its capacity stops growing.")
        age_grid = np.linspace(0, max(v["median"] for v in layer_lifetimes.values()) * 2.2, 200)
        rows = []
        for (name, v), color in zip(layer_lifetimes.items(), (ACCENT, SLATE)):
            s = np.exp(-((age_grid / v["scale"]) ** v["shape"]))
            fig.add_trace(go.Scatter(x=age_grid, y=s, mode="lines", name=f"{name} (median {v['median']:.0f} yr)",
                                     line=dict(color=color, width=2.5), hovertemplate="%{y:.1%}"))
            fig.add_trace(go.Scatter(x=[v["median"]], y=[0.5], mode="markers", marker=dict(size=9, color=color),
                                     showlegend=False, hoverinfo="skip"))
            rows.append(pd.DataFrame({"curve": f"Weibull, {name}", "age_years": age_grid, "survival": s}))
        survival_table = pd.concat(rows, ignore_index=True)
    else:
        km = system_result["kaplan_meier"]
        k, lam = system_result["weibull_shape"], system_result["weibull_scale"]
        if km is not None:
            takeaway(f"Half of all {UNIT_WORDS[group['system']]} have retired by age "
                     f"<b>{weibull_median:.1f}</b> on the fitted Weibull curve, the one every "
                     "projection uses. The empirical Kaplan-Meier median, which stops at the oldest "
                     f"retirement actually observed, is <b>{system_result['median_survival_years']:.1f}</b> years.")
        else:
            takeaway(f"Half of all {UNIT_WORDS[group['system']]} are expected to retire by age "
                     f"<b>{weibull_median:.1f}</b>, from a Weibull curve fitted to a published vehicle "
                     "survival table, since the fleet is too young to have produced enough retirements "
                     "of its own.")
        rows = []
        if km is not None:
            km_t = np.concatenate([[0.0], km.t.values])
            km_s = np.concatenate([[1.0], km.survival.values])
            age_grid = np.linspace(0, max(km_t.max(), weibull_median * 1.5), 200)
            fig.add_trace(go.Scatter(x=km_t, y=km_s, mode="lines", name="Kaplan-Meier (empirical, right-censored)",
                                     line=dict(shape="hv", color=SLATE, width=2), hovertemplate="%{y:.1%}"))
            rows.append(pd.DataFrame({"curve": "Kaplan-Meier", "age_years": km_t, "survival": km_s}))
        else:
            age_grid = np.linspace(0, weibull_median * 2, 200)
        weibull_s = np.exp(-((age_grid / lam) ** k))
        fig.add_trace(go.Scatter(
            x=age_grid, y=weibull_s, mode="lines",
            name="Weibull (fitted to a real external survival curve)" if km is None else "Weibull fit (extrapolated)",
            line=dict(color=ACCENT, width=2.5, dash="dash" if km is not None else "solid"),
            hovertemplate="%{y:.1%}",
        ))
        fig.add_trace(go.Scatter(x=[weibull_median], y=[0.5], mode="markers", marker=dict(size=9, color=ACCENT),
                                 showlegend=False, hoverinfo="skip"))
        rows.append(pd.DataFrame({"curve": "Weibull", "age_years": age_grid, "survival": weibull_s}))
        survival_table = pd.concat(rows, ignore_index=True)

    fig.add_hline(y=0.5, line=dict(color=NEUTRAL, width=1, dash="dot"),
                  annotation_text="half retired", annotation_position="top right")
    style_fig(fig, height=450)
    fig.update_xaxes(title_text="Age (years)")
    fig.update_yaxes(title_text="Share still in service", range=[0, 1.04], tickformat=".0%")
    show_chart(fig, "chart_survival")

    if not layer_lifetimes:
        c1, c2, c3 = st.columns(3)
        c1.metric("Weibull median lifetime", f"{weibull_median:,.1f} yr", border=True)
        c2.metric("Shape k", f"{system_result['weibull_shape']:.2f}", border=True,
                  help="Above 1 means the retirement rate rises with age (wear-out), as expected for equipment.")
        c3.metric("Scale \u03bb", f"{system_result['weibull_scale']:.2f} yr", border=True,
                  help="The age by which about 63% of units have retired.")

    with how_calculated():
        if layer_lifetimes:
            st.latex(r"\text{Weibull: } S(t) = \exp\!\left(-(t/\lambda)^k\right)")
            st.caption(
                "One lifetime per infrastructure layer, not one for the whole fleet: a data centre is a "
                "long-lived site whose parts wear out at different rates, which is why this system uses a "
                "stock-driven model (required capacity plus each layer's lifetime decide what has to be "
                "built each year, growth plus replacement). Median lives from ASHRAE's service-life data: "
                "30 years for electric transformers (grid connection), 17-25 years for facility power and "
                "cooling equipment, 20 used as one central value for that layer. Shape k=3.0 is a disclosed "
                "assumption, sources publish median lives, not curve shapes; both curves this engine fits "
                "from real data land at k of about 3.1-3.2. Servers are not modelled: no consistent public "
                "per-MW material figure was found, see mfa_engine/systems/data_centre.py."
            )
        elif system_result["kaplan_meier"] is not None:
            k, lam = system_result["weibull_shape"], system_result["weibull_scale"]
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
                "likelihood over both kinds of observation at once. No confidence band is drawn, because "
                "this engine does not compute one; a fabricated band would be worse than none."
            )
        else:
            k, lam = system_result["weibull_shape"], system_result["weibull_scale"]
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
    download_csv(survival_table, f"survival_{group['system']}.csv", "dl_survival")
    footer("lifetimes")


def _mirror_ticks(lo: float, hi: float, n: int = 6) -> tuple[list[float], list[str]]:
    """Tick positions for an axis with outflow drawn below zero, labelled with absolute values
    so a bar below the line reads as '2k tonnes leaving', not as a negative tonnage."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    vals = np.arange(np.floor(lo / step) * step, hi + step * 0.51, step)

    def fmt(v: float) -> str:
        v = abs(v)
        if v >= 1e6:
            return f"{v / 1e6:,.3g}M"
        if v >= 1e3:
            return f"{v / 1e3:,.3g}k"
        return f"{v:,.3g}"

    return [float(v) for v in vals], [fmt(v) for v in vals]


def page_stocks():
    header(
        "stocks", "Stocks and flows",
        "The full account, one material at a time: what enters service, what is in use, what "
        "leaves at end of life, and the part of that outflow actually recycled back into use. "
        "A real stock-flow model has to balance every year.",
    )
    msf = system_result["material_stock_flows"]
    present = set(msf.material.unique())
    sf_materials = [m for m in MATERIAL_ORDER[group["system"]] if m in present]
    sf_default = group["physical_columns"][0]
    sf_mat = st.pills(
        "Material", sf_materials, default=sf_default if sf_default in sf_materials else sf_materials[0],
        required=True, format_func=str.capitalize,
        key=f"sf_material_{material_choice}",   # per material, so it follows the sidebar choice
    )
    sf_one = msf[(msf.material == sf_mat) & (msf.year <= horizon_year)].sort_values("year")
    sf_rate = rate_for(sf_mat, recovery_regime)
    mcolor = MATERIAL_COLORS.get(sf_mat, ACCENT)
    recovered_t = sf_one.outflow_t * sf_rate.central if sf_rate is not None else None

    proj = sf_one[sf_one.year >= projection_start_year]
    stock_start = float(sf_one.set_index("year").stock_t.get(projection_start_year, float("nan")))
    stock_end = float(sf_one.stock_t.iloc[-1]) if len(sf_one) else float("nan")
    out_total = float(proj.outflow_t.sum())
    text = (f"<b>{stock_start:,.0f} t</b> of {esc(sf_mat)} was in "
            f"use in {projection_start_year}, and <b>{stock_end:,.0f} t</b> by {horizon_year}. "
            f"<b>{out_total:,.0f} t</b> leaves service over {projection_start_year}-{horizon_year}")
    if sf_rate is not None:
        text += f", of which about <b>{out_total * sf_rate.central:,.0f} t</b> is recycled back into use."
    else:
        text += f"; there is no cited recycling rate for {esc(sf_mat)}, so no recovered share is drawn."
    text += {
        "wind": " Denmark's register lists only turbines already standing, so the stock can only fall.",
        "ev": " This base run uses real sales through 2026 only, so the stock drains afterwards; the "
              "Scenarios page adds future sales.",
        "dc": " Required capacity keeps growing to 2035, so the stock keeps rising.",
    }[group["system"]]
    takeaway(text)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.42, 0.58], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=sf_one.year, y=sf_one.stock_t, name="In use (stock)", mode="lines",
                             line=dict(color=mcolor, width=2.5), fill="tozeroy", fillcolor=rgba(mcolor, 0.12),
                             hovertemplate="%{y:,.0f} t"), row=1, col=1)
    has_inflow = sf_one.inflow_t.sum() > 0
    if has_inflow:
        fig.add_trace(go.Bar(x=sf_one.year, y=sf_one.inflow_t, name="Entering service (inflow)",
                             marker_color=SLATE, hovertemplate="%{y:,.0f} t"), row=2, col=1)
    fig.add_trace(go.Bar(x=sf_one.year, y=-sf_one.outflow_t, customdata=sf_one.outflow_t,
                         name="Leaving service (end-of-life outflow)", marker_color=rgba(mcolor, 0.38),
                         hovertemplate="%{customdata:,.0f} t"), row=2, col=1)
    if recovered_t is not None:
        fig.add_trace(go.Bar(x=sf_one.year, y=-recovered_t, customdata=recovered_t,
                             name="Recovered (recycled back into use)", marker_color=mcolor,
                             hovertemplate="%{customdata:,.0f} t"), row=2, col=1)
    if len(sf_one) and projection_start_year > sf_one.year.min():
        fig.add_vline(x=projection_start_year, line=dict(color=NEUTRAL, width=1, dash="dash"))
        for text, anchor, shift in (("history", "right", -5), ("projection", "left", 5)):
            fig.add_annotation(x=projection_start_year, y=0.97, xref="x", yref="y domain", text=text,
                               showarrow=False, xanchor=anchor, xshift=shift, yanchor="top", font=dict(size=11))
    lo = -float(sf_one.outflow_t.max()) if len(sf_one) else 0.0
    hi = float(sf_one.inflow_t.max()) if has_inflow else 0.0
    tickvals, ticktext = _mirror_ticks(lo, hi)
    style_fig(fig, height=560)
    fig.update_layout(barmode="overlay", bargap=0.12)
    fig.update_yaxes(title_text="In use, t", tickformat="~s", row=1, col=1)
    fig.update_yaxes(title_text="In / out per year, t", tickvals=tickvals, ticktext=ticktext,
                     zeroline=True, zerolinecolor=AXIS, zerolinewidth=1, row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    show_chart(fig, "chart_stock_flows")

    sf_residual = float(system_model.mass_balance_residual(msf).get(sf_mat, 0.0))
    if sf_residual < 1e-6:
        st.badge(f"Mass balance closes every year: worst residual {sf_residual:.1e} t",
                 icon=":material/check_circle:", color="green")
    else:
        st.badge(f"Mass balance does not close: worst residual {sf_residual:.1e} t",
                 icon=":material/error:", color="red")

    with how_calculated():
        st.latex(r"S_t - S_{t-1} = I_t - O_t \qquad R_t = O_t \times \mathrm{EoL\text{-}RR}")
        if sf_rate is not None:
            sf_rate_text = (f"End-of-life recycling rate (EoL-RR) {sf_rate.central:.1%}, range "
                            f"{sf_rate.low:.1%}-{sf_rate.high:.1%}, {sf_rate.source}."
                            + (f" {sf_rate.note}" if sf_rate.note else ""))
        else:
            sf_rate_text = (f"No cited end-of-life recycling rate for {sf_mat}, so no recovered bar is "
                            "drawn rather than a guessed one.")
        st.caption(
            f"Mass balance closes every year for {sf_mat}: worst residual {sf_residual:.1e} t, "
            f"floating-point noise. {sf_rate_text} Outflow and recovery are drawn below zero so the "
            "two sides of the account read against each other; the axis labels are absolute tonnes. "
            + {
                "wind": "Denmark's register only lists turbines already standing, so there is no inflow "
                        "in the projection yet; adding real future installations is a planned next step.",
                "ev": "EV sales are an inflow table, so each year's cohort enters as inflow and decays "
                      "from its own sale year; left of the dashed line is modelled history, right of it "
                      "is projection.",
                "dc": "Data centres are stock-driven: the capacity needed each year is the input, and "
                      "inflow is derived from it, new capacity plus replacement of worn-out equipment, "
                      "which is why inflow keeps flowing after capacity stops growing in 2035. Left of "
                      "the dashed line is anchored history, right of it is projection.",
            }[group["system"]]
        )
    table = sf_one[["year", "inflow_t", "stock_t", "outflow_t"]].copy()
    if recovered_t is not None:
        table["recovered_t"] = recovered_t.values
    download_csv(table, f"stock_flows_{group['system']}_{sf_mat.replace(' ', '_')}.csv", "dl_stock_flows")
    footer("stocks")


def page_eol():
    header(
        "eol", "What comes back out",
        f"Material leaving service each year as the fleet retires, the retirement wave behind it, and "
        f"which {esc(system['category_label'])} it comes from. These are gross end-of-life tonnes, "
        "before any recycling rate is applied.",
    )
    mats = system_result["secondary_materials"]
    present = [c for c in mats.columns if c not in ("year", "units_retiring")]
    order = [m for m in MATERIAL_ORDER[group["system"]] if m in present]
    default_material = group["physical_columns"][0]

    st.subheader("Material leaving service")
    c_pick, c_log = st.columns([4, 1], vertical_alignment="bottom")
    with c_pick:
        chosen = st.pills("Materials to show", order, selection_mode="multi", default=[default_material],
                          format_func=str.capitalize, key=f"materials_multiselect_{group['system']}")
    chosen = [m for m in order if m in (chosen or [])]   # fixed legend order, whatever the click order
    with c_log:
        log_scale = st.toggle("Log scale", value=len(chosen) > 1)

    if chosen:
        peaks = {m: (int(mats.loc[mats[m].idxmax(), "year"]), float(mats[m].max())) for m in chosen}
        series_by_year = mats.set_index("year")
        if len(chosen) == 1:
            m = chosen[0]
            y, t = peaks[m]
            if _at_edge(y, series_by_year[m]):
                lead = (f"{esc(m.capitalize())} leaving service is still rising at the horizon: "
                        f"<b>{t:,.0f} t</b> in <b>{y}</b> (central estimate).")
            else:
                lead = (f"{esc(m.capitalize())} leaving service peaks in <b>{y}</b> at "
                        f"<b>{t:,.0f} t</b> that year (central estimate).")
            takeaway(lead + " The band is the published low-high range of how much of it each unit contains.")
        else:
            takeaway("Peak year leaving service: " + "; ".join(
                f"{esc(m)} <b>{y}</b>{' (still rising)' if _at_edge(y, series_by_year[m]) else ''} "
                f"({t:,.0f} t)" for m, (y, t) in peaks.items()) + ".")
        fig2 = go.Figure()
        if len(chosen) == 1:
            # A real confidence band, not a decorative one: material_intensity()'s own
            # (low, central, high) tuples are real sourced ranges (JRC for wind, the
            # derived CRS/GREET figures for EV batteries), previously computed and then
            # discarded down to the central value everywhere in this app. One material
            # selected is the moment there is room to actually show that real range
            # instead of throwing it away.
            mat = chosen[0]
            color = MATERIAL_COLORS.get(mat, ACCENT)
            mat_range = system_model.secondary_materials_range(system_result["retirement_schedule"])
            merged = mats[["year"]].merge(mat_range[["year", f"{mat}_low", f"{mat}_high"]], on="year", how="left")
            fig2.add_trace(go.Scatter(
                x=pd.concat([merged.year, merged.year[::-1]]),
                y=pd.concat([merged[f"{mat}_high"], merged[f"{mat}_low"][::-1]]),
                fill="toself", fillcolor=rgba(color, 0.16),
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="low-high range",
            ))
            fig2.add_trace(go.Scatter(x=mats.year, y=mats[mat], mode="lines", name=f"{mat} (central estimate)",
                                      line=dict(color=color, width=2.5), hovertemplate="%{y:,.1f} t"))
            table = merged.merge(mats[["year", mat]], on="year")
        else:
            # No fill here, on purpose: a filled tozeroy area per material looked reasonable with one
            # line, but with several materials of genuinely different real magnitude (steel in the
            # hundreds of thousands of tonnes next to copper in the thousands) each fill draws over the
            # others underneath it, visually erasing the smaller-magnitude lines instead of comparing
            # them. Plain lines compare real magnitudes honestly regardless of scale difference.
            for mat in chosen:
                fig2.add_trace(go.Scatter(x=mats.year, y=mats[mat], mode="lines", name=mat,
                                          line=dict(color=MATERIAL_COLORS.get(mat, NEUTRAL), width=2.2),
                                          hovertemplate="%{y:,.1f} t"))
            if len(chosen) <= 4:
                end_labels(fig2, {m: mats.set_index("year")[m] for m in chosen}, log=log_scale)
            table = mats[["year"] + chosen]
        style_fig(fig2, height=430)
        fig2.update_xaxes(title_text="Year")
        fig2.update_yaxes(title_text="Tonnes leaving service that year", type="log" if log_scale else "linear",
                          **({"dtick": 1, "tickformat": "~s"} if log_scale else {}))
        if len(chosen) > 1 and len(chosen) <= 4:
            fig2.update_layout(margin=dict(t=44, l=8, r=110, b=8))
        show_chart(fig2, "chart_eol_materials")
        with how_calculated():
            st.latex(
                r"M_{\text{material},\,t} = \sum_{\text{category}} \big(\text{units retiring in year } t"
                r"\big)_{\text{category}} \times I_{\text{material},\,\text{category}}"
            )
            if len(chosen) == 1:
                st.caption(
                    f"Shaded band: the real low-high range material_intensity() reports for {chosen[0]}, "
                    "propagated through the same cohort projection as the central line (I above is the "
                    "low/high bound instead of the central value). Measured from: each retiring unit's "
                    f"real {system['category_label']} times that category's real, sourced material content "
                    "per unit, summed across categories for that year. Pick a second material to compare "
                    "trends instead; the band is only shown for one at a time to keep it legible."
                )
            else:
                st.caption(
                    "Same formula as the single-material band (I is the central value here), run once per "
                    f"material chosen. Measured from: each retiring unit's real {system['category_label']} "
                    "times that category's real, sourced material content per unit. Colours are fixed per "
                    "material and drawn in one fixed order, so a colour always means the same material."
                )
        download_csv(table, f"end_of_life_{group['system']}.csv", "dl_eol")
    else:
        st.info("Pick at least one material above.")

    st.subheader("Retirement wave")
    wave_peak = mats.loc[mats.units_retiring.idxmax()] if len(mats) else None
    if wave_peak is not None:
        unit_phrase = {"wind": "MW of turbine capacity", "ev": "vehicles", "dc": "MW of equipment"}[group["system"]]
        wave = mats.set_index("year").units_retiring
        if _at_edge(int(wave_peak.year), wave):
            takeaway(f"Retirements are still rising at the horizon: <b>{wave_peak.units_retiring:,.0f}</b> "
                     f"{unit_phrase} leave service in <b>{int(wave_peak.year)}</b>.")
        else:
            takeaway(f"Retirements peak in <b>{int(wave_peak.year)}</b>, when "
                     f"<b>{wave_peak.units_retiring:,.0f}</b> {unit_phrase} leave service.")
    fig3 = go.Figure(go.Bar(x=mats.year, y=mats.units_retiring, marker_color=SLATE, name=system["unit_label"],
                            hovertemplate="%{y:,.0f}<extra></extra>"))
    style_fig(fig3, height=320, legend=False, hovermode="closest")
    fig3.update_xaxes(title_text="Year")
    fig3.update_yaxes(title_text=system["unit_label"])
    show_chart(fig3, "chart_retirement_wave")
    with how_calculated():
        st.latex(
            r"R(a_0,\,t) = \frac{S(a_0+t-1)}{S(a_0)} - \frac{S(a_0+t)}{S(a_0)}"
            r"\qquad U_t = \sum_{\text{assets}} \text{size} \times R(a_0,\,t)"
        )
        st.caption(
            "R is the conditional probability an asset that already reached real age a_0 retires in "
            "year t specifically, not at any point, S is the Weibull (or Kaplan-Meier) survival curve "
            "from the Lifetimes page. Measured from: every real asset still in service, projected "
            "individually under its own real current age, then summed, a genuine cohort model rather "
            "than one average asset scaled up. The EV fleet enters as yearly sales cohorts instead, and "
            "data-centre equipment as the capacity needed each year; both decay through the same curve."
        )
    download_csv(mats[["year", "units_retiring"]], f"retirement_wave_{group['system']}.csv", "dl_wave")

    st.subheader(f"Which {system['category_label']} it comes from")
    sankey_groups = SANKEY_GROUPS_BY_SYSTEM[group["system"]]
    names = list(sankey_groups)
    default_group = next((n for n, cols in sankey_groups.items() if default_material in cols), names[0])
    sankey_group_choice = st.segmented_control(
        "Material group", names, default=default_group, required=True,
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
        st.info("No nonzero end-of-life amount for this material group at this horizon.")
    else:
        cat_totals = by_cat.groupby("category").tonnes.sum().sort_values(ascending=False)
        takeaway(f"The <b>{esc(cat_totals.index[0])}</b> {esc(system['category_label'])} supplies "
                 f"<b>{cat_totals.iloc[0] / cat_totals.sum():.0%}</b> of the {esc(sankey_group_choice)} "
                 f"leaving service to {horizon_year} "
                 f"({cat_totals.sum():,.0f} t in total, across {len(cat_totals)} "
                 f"{esc(_plural(system['category_label'], len(cat_totals)))}).")
        categories = sorted(by_cat.category.unique())
        materials_in_sankey = [m for m in MATERIAL_ORDER[group["system"]] if m in set(by_cat.material)]
        nodes = categories + materials_in_sankey
        node_index = {name: i for i, name in enumerate(nodes)}
        node_colors = [SLATE] * len(categories) + [MATERIAL_COLORS.get(m, NEUTRAL) for m in materials_in_sankey]
        fig_sankey = go.Figure(go.Sankey(
            node=dict(label=nodes, color=node_colors, pad=24, thickness=16, line=dict(width=0),
                      hovertemplate="%{label}: %{value:,.0f} t<extra></extra>"),
            link=dict(
                source=[node_index[r.category] for r in by_cat.itertuples()],
                target=[node_index[r.material] for r in by_cat.itertuples()],
                value=[r.tonnes for r in by_cat.itertuples()],
                color=[rgba(MATERIAL_COLORS.get(r.material, NEUTRAL), 0.32) for r in by_cat.itertuples()],
                hovertemplate="%{source.label} to %{target.label}: %{value:,.0f} t<extra></extra>",
            ),
            textfont=dict(size=13, shadow="none"),
        ))
        style_fig(fig_sankey, height=430, legend=False, hovermode="closest", margin=dict(t=10, l=8, r=8, b=10))
        show_chart(fig_sankey, "chart_category_sankey")
        with how_calculated():
            st.caption(
                f"Cumulative end-of-life tonnes to {horizon_year}, by real {system['category_label']} and "
                "material. Link width is proportional to tonnes; this is the same M formula as the "
                "material chart above, summed over every year to the horizon instead of plotted year by "
                f"year, keeping the {system['category_label']} breakdown that chart pools away. Gross "
                "tonnes leaving service, before any recycling rate."
            )
        download_csv(by_cat, f"end_of_life_by_category_{group['system']}.csv", "dl_by_cat")
    footer("eol")


def page_scenarios():
    scenario_ui = SCENARIO_UI.get(group["system"])
    if not scenario_ui:
        header("scenarios", "Scenarios",
               "How much a different future changes the stock in use and the material coming back out.")
        st.info(
            "Denmark's register only lists turbines already standing, so the wind system has no future "
            "deployment scenarios yet; adding real future installations is a planned next step. "
            "Scenarios exist for EV batteries (IEA STEPS vs a flat 2026 market) and data centres "
            "(McKinsey's low vs high 2030 capacity). Pick lithium, nickel, cobalt or graphite, or copper "
            "found in EV batteries or data centres, in the sidebar.",
            icon=":material/info:",
        )
        footer("scenarios")
        return
    title, question = scenario_ui["title"].split(": ", 1)
    header("scenarios", title, f"<b>{esc(question[0].upper() + question[1:])}</b> {esc(scenario_ui['intro'])}")

    c1, c2, c3 = st.columns([2, 2, 3], vertical_alignment="bottom")
    with c1:
        scenario_horizon = st.slider(
            "Scenario horizon year", min_value=2035, max_value=2060, value=2050, step=5,
            key=f"scenario_horizon_{group['system']}", help=scenario_ui["slider_help"],
        )
    scenario_flows = compute_scenario_stocks(group["system"], scenario_horizon)
    present = set(scenario_flows.material.unique())
    sc_materials = [m for m in MATERIAL_ORDER[group["system"]] if m in present]
    with c2:
        scenario_material = st.selectbox(
            "Material", sc_materials, format_func=str.capitalize,
            index=sc_materials.index(group["physical_columns"][0]) if group["physical_columns"][0] in sc_materials else 0,
            key=f"scenario_material_{material_choice}",   # per material, so it follows the sidebar choice
        )
    with c3:
        scenario_view = st.segmented_control(
            "Show", ["In-use stock", "End-of-life outflow", "Recovered"], default="In-use stock",
            required=True, key="scenario_view", help=scenario_ui["radio_help"],
        )
    sc_rate = rate_for(scenario_material, recovery_regime)
    sc_one = scenario_flows[(scenario_flows.material == scenario_material)
                            & (scenario_flows.year <= scenario_horizon)].copy()
    if scenario_view == "In-use stock":
        sc_one["value"], sc_axis = sc_one.stock_t, "Tonnes in use"
    elif scenario_view == "End-of-life outflow":
        sc_one["value"], sc_axis = sc_one.outflow_t, "Tonnes leaving service that year"
    else:
        sc_one["value"] = sc_one.outflow_t * sc_rate.central if sc_rate else float("nan")
        sc_axis = "Tonnes recovered that year"

    if scenario_view == "Recovered" and sc_rate is None:
        st.info(f"No cited end-of-life recycling rate for {scenario_material}, so no recovered "
                "series is shown. Switch to in-use stock or end-of-life outflow.")
        footer("scenarios")
        return

    upper_name, base_name = scenario_ui["names"]
    by_sc = {name: sub.set_index("year")["value"] for name, sub in sc_one.groupby("scenario")}
    if scenario_view == "In-use stock":
        sc_vals = {name: float(s.get(scenario_horizon, 0.0)) for name, s in by_sc.items()}
        sc_label = f"in use in {scenario_horizon}"
    else:
        sc_vals = {name: float(s[s.index >= projection_start_year].sum()) for name, s in by_sc.items()}
        sc_label = f"cumulative {projection_start_year}-{scenario_horizon}"
    _base_v = sc_vals.get(base_name, 0.0)
    _gap = (sc_vals.get(upper_name, 0.0) - _base_v) / _base_v if _base_v else 0.0
    takeaway(f"Computed live, not quoted: {esc(scenario_ui['upper_short'])} is <b>{_gap:.1%}</b> above "
             f"{esc(scenario_ui['base_short'])} ({scenario_view.lower()}, {sc_label}), for "
             f"{esc(scenario_material)}.")

    sc1, sc2 = st.columns(2)
    for col, name in zip((sc1, sc2), (upper_name, base_name)):
        col.metric(name, f"{sc_vals.get(name, 0.0):,.0f} t", help=sc_label, border=True)

    color = MATERIAL_COLORS.get(scenario_material, ACCENT)
    fig = go.Figure()
    if base_name in by_sc:
        s = by_sc[base_name]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=base_name, legendrank=2,
                                 line=dict(color=NEUTRAL, width=2, dash="dash"), hovertemplate="%{y:,.0f} t"))
    if upper_name in by_sc:
        s = by_sc[upper_name]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=upper_name, legendrank=1,
                                 line=dict(color=color, width=2.5), fill="tonexty" if base_name in by_sc else None,
                                 fillcolor=rgba(color, 0.12), hovertemplate="%{y:,.0f} t"))
    style_fig(fig, height=420)
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text=sc_axis)
    show_chart(fig, "chart_scenarios")
    with how_calculated():
        st.caption(
            "The shaded area is the gap between the two scenarios. " + scenario_ui["closing"]
        )
    download_csv(sc_one[["scenario", "year", "material", "stock_t", "outflow_t", "value"]].rename(
        columns={"value": sc_axis.lower().replace(" ", "_")}),
        f"scenarios_{group['system']}_{scenario_material}.csv", "dl_scenarios")
    footer("scenarios")


def page_supply_map():
    header(
        "supplymap", "Where it comes from, and where it goes",
        f"Where {esc(short_material.lower())} is mined, how it moves between countries, and what happens "
        "to that network when one supplier stops. Hover a country for its numbers, click it to isolate "
        f"its trade, and press play to replay losing <b>{esc(removed_country)}</b>, the supplier set in "
        "the sidebar.",
    )
    supply = load_supply()
    fam = supply[supply.family == family]
    measures = [m for m in ("Mine production", "Reserves") if m in set(fam.measure)]
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    with c1:
        if len(measures) > 1:
            measure = st.segmented_control("Circles show", measures, default=measures[0], required=True,
                                           key=f"map_measure_{family}")
        else:
            measure = measures[0]
            st.caption(f"Circles show mine production; the source series has no reserves table for "
                       f"{family.lower()}.")
    with c2:
        top_n = st.slider(
            "Largest trade flows drawn", min_value=10, max_value=60, value=30, step=5, key="map_top_n",
            help="Drawing every bilateral link would bury the map, and the largest links carry most of "
            "the value. The removed supplier's own largest flows are always added, so the replay has them.",
        )

    from build import REEXPORT_HUBS
    payload, notes = build_map_payload(
        family=family, measure=measure, edges=commodity_edges, top_n=top_n,
        removed_country=removed_country, cascade=fragility, accent=ACCENT, commodity=group["name"],
        hs_code=group["hs_code"], partners_json=str(CRM_TRADE_NETWORK_SRC.parent / "data" / "partners.json"),
        hubs=set(REEXPORT_HUBS.values()), universe=fragility_universe,
    )

    rows = fam[(fam.measure == measure) & (fam.iso3 != "")].sort_values("tonnes", ascending=False)
    top = rows.iloc[0]
    has_share = str(top.share_pct).strip() != ""
    if measure == "Mine production":
        lead = (f"<b>{esc(top.country)}</b> mines <b>{float(top.share_pct):.0f}%</b> of the world's "
                if has_share else f"<b>{esc(top.country)}</b> is the largest miner of ")
        lead += f"{esc(family.lower())} ({esc(payload['bubbles'][0]['value'])} in {notes['year']})."
    else:
        lead = (f"<b>{esc(top.country)}</b> holds <b>{float(top.share_pct):.0f}%</b> of known "
                if has_share else f"<b>{esc(top.country)}</b> holds the largest known ")
        lead += f"{esc(family.lower())} reserves ({esc(payload['bubbles'][0]['value'])})."
    e = commodity_edges[commodity_edges.exporter != commodity_edges.importer]
    big = e.sort_values("value_usd", ascending=False).iloc[0]
    lead += (f" The largest single trade flow is <b>{esc(big.exporter)}</b> to <b>{esc(big.importer)}</b>, "
             f"{esc(usd(float(big.value_usd)))} in 2023.")
    takeaway(lead)

    supply_map(payload, key="supply_map")

    with how_calculated():
        st.markdown(
            f"- **Circles and shading**: {measure.lower()} by country, {notes['unit']}, from USGS Mineral "
            f"Commodity Summaries 2025 via the Materials Data Series (data/mine_supply.csv). Circle area is "
            "proportional to tonnes. The source lists the leading countries only"
            + (", plus: " + "; ".join(f"{n} {t}" for n, t in notes["other_supply"]) + " (not placed on the map)."
               if notes["other_supply"] else ".")
            + f"\n- **Arcs**: 2023 UN Comtrade trade in {group['name']} (HS {group['hs_code']}), by value, the "
            f"same edge list every risk page uses: built from both importer and exporter reports, the larger "
            f"of the two where they disagree. The {notes['flows_drawn']} arcs are the largest links plus "
            f"{removed_country}'s own largest ones. Moving dots run from exporter to importer. This is one "
            "stage of the chain, the traded form in that HS code, not mine to refinery to product."
            + "\n- **Years**: production and reserves are 2024 (USGS 2025 edition); trade is 2023, the year "
            "every trade figure in this dashboard uses. The map puts them side by side, it does not "
            "combine them in any calculation."
            + "\n- **Replay**: crm-trade-network's cascade model (the same run as the Cascade page), removing "
            f"{removed_country} and letting any country that loses over 20% of its trade in this commodity "
            "fail in turn, shown round by round."
            + (f" {notes['cascade_offmap']} of the countries that fail have no position on this map "
               "(territories and aggregates UN Comtrade reports separately)." if notes["cascade_offmap"] else "")
            + "\n- **Placing countries**: by ISO3 code, trade partners through UN Comtrade's own partner "
            "table (Taiwan is reported as \"Other Asia, nes\"). Outlines are Natural Earth 1:110m, anchor "
            "points Natural Earth's 1:50m label points, drawn in the equal-area Equal Earth projection."
            + (" Not placed (no single country position): " + ", ".join(notes["unplaced_trade"]) + "."
               if notes["unplaced_trade"] else "")
            + ("\n- **Re-export hubs** (the Netherlands, Belgium, Singapore, Hong Kong, the UAE): part of "
               "their trade is transit rather than origin, and the tooltip says so.")
        )
        if group["hs_code"] == "2836":
            st.caption("For lithium, HS 2836 covers all carbonates, and most of that trade by weight is not "
                       "lithium carbonate (see the Disruption page's price note). Read the arcs as where "
                       "carbonate trade goes, not as lithium tonnes.")
        if not notes["has_history"]:
            st.caption("The installed crm-trade-network does not report cascade rounds yet, so the replay "
                       "button is off; the Cascade page still shows the final result.")
    d1, d2 = st.columns(2)
    with d1:
        download_csv(fam[fam.measure == measure][["country", "iso3", "tonnes", "share_pct", "unit", "year", "source"]],
                     f"{family.lower().replace(' ', '_')}_{measure.lower().replace(' ', '_')}.csv", "dl_map_supply",
                     label="Download circles data (CSV)")
    with d2:
        download_csv(pd.DataFrame([{"exporter": f["fromName"], "importer": f["toName"], "value": f["usd"],
                                    "tonnes": f["tonnes"]} for f in payload["flows"]]),
                     f"trade_arcs_hs{group['hs_code']}.csv", "dl_map_flows", label="Download arcs data (CSV)")
    footer("supplymap")


def page_disruption():
    header(
        "disruption", "Disruption and recovery",
        f"What if <b>{esc(removed_country)}</b> stopped exporting {esc(group['name'])} for a year? It "
        f"supplied <b>{removed_country_share:.1%}</b> of 2023 world trade by value. This page asks "
        f"whether recycling {esc(short_material.lower())} matters at the scale of that gap. Change the "
        "supplier in the sidebar.",
    )
    _slack_pct_label = f"{group['real_slack']:.1%}"
    with st.expander(f"What does \"{_slack_pct_label} slack\" mean for {short_material.lower()}, "
                     "and where does it come from?", icon=":material/tune:"):
        st.latex(
            r"\text{shortfall} = \max\!\Big(0,\ \text{removed country's exports} - \text{slack}"
            r"\times \sum_{\text{survivors}} \text{their exports}\Big)"
        )
        st.caption(
            "Survivors (every supplier except the one removed) can each expand output by `slack` "
            "times their own current real exports of this commodity; whatever the removed country "
            "used to supply that this spare capacity still cannot cover is the shortfall. slack=0 is "
            f"always the pessimistic no-substitution bound; slack={group['real_slack']:.3f} allows "
            f"survivors {_slack_pct_label} headroom for {short_material.lower()} specifically, not "
            "the same flat 20% applied to every material regardless of whether 20% means anything "
            "for that one. crm-trade-network's own README calls 20% an explicit, disclosed parameter "
            "rather than a hidden one, but also states plainly that the specific value has no "
            "independent citation in that project, a deliberately round illustrative number, not "
            "fit to data."
        )
        if "real_slack_source" in group:
            st.caption(f"**Real, cited figure used for {short_material.lower()}:** {group['real_slack_source']}")
        elif "real_slack_note" in group:
            st.caption(group["real_slack_note"])

    if connector_result is None and scenario_note[0] == "no_recovery_rate":
        st.info(
            f"No cited end-of-life recycling rate exists for {short_material.lower()}, so this "
            "dashboard does not report a recovered tonnage for it rather than guessing one. The UN "
            "International Resource Panel's 2011 assessment does not cover carbon/graphite, and "
            "the EU Batteries Regulation sets graphite no recovery target. Its end-of-life "
            f"outflow is still real and shown on the Stocks and flows page: "
            f"**{float(outflow_by_year.mean()):,.0f} t/yr** on average, {projection_start_year}-{horizon_year}.",
            icon=":material/info:",
        )
    elif connector_result is None and scenario_note[0] == "no_exports":
        st.warning(
            f"{removed_country} has no recorded exports of {group['name']} in this dataset, "
            "so there is no shortfall to compare against. Pick a different supplier in the sidebar.",
            icon=":material/warning:",
        )
    elif connector_result is None and scenario_note[0] == "slack_covers_it":
        st.info(
            f"Removing {removed_country} creates a real **\\${scenario_note[1]/1e9:,.2f} bn** "
            f"shortfall with no substitution, but the remaining suppliers' {group['real_slack']:.1%} "
            f"slack fully absorbs it; the {group['real_slack']:.1%}-slack shortfall is zero. There is "
            f"nothing left for the recovered {material_choice.lower()} to offset in that scenario, so "
            "no coverage percentage is shown.",
            icon=":material/info:",
        )
    else:
        r = connector_result
        takeaway(
            f"In an average year, this recovery covers <b>{_pct(r['recovered_share_of_no_slack_shortfall'])}</b> "
            f"of one year's no-substitution shortfall and <b>{_pct(r['recovered_share_of_20pct_slack_shortfall'])}</b> "
            f"of one year's {group['real_slack']:.1%}-slack shortfall."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Avg. annual recovery ({short_material.lower()})", f"${r['cumulative_recovered_usd']/1e6:,.1f} M/yr",
                  border=True)
        c2.metric(
            f"Peak year ({peak_year})" if peak_year else "Peak year",
            f"${peak_connector_result['cumulative_recovered_usd']/1e6:,.1f} M/yr" if peak_connector_result else "n/a",
            help="The single real retirement-wave year that recovers the most of this material, not "
            "the average. Surfaced alongside the average on purpose: averaging across the whole "
            "projection is itself a smaller version of the time-scale problem the annualization fix "
            "elsewhere in this app exists to avoid.",
            border=True,
        )
        c3.metric("Shortfall, no substitution", f"${r['real_shortfall_usd_no_slack']/1e9:,.2f} bn", border=True,
                  help="One year of disruption at 2023 trade values, with no other supplier stepping in.")
        c4.metric(f"Shortfall, {group['real_slack']:.1%} slack",
                  f"${r['real_shortfall_usd_20pct_slack']/1e9:,.2f} bn", border=True,
                  help="One year of disruption at 2023 trade values, with every other supplier able to "
                  f"expand by {group['real_slack']:.1%}.")

        _rate_labels = sorted({f"{rt.central:.1%} ({rt.source})" for rt in material_rates.values() if rt})
        _gross_avg = float(outflow_by_year.mean()) if len(outflow_by_year) else 0.0
        st.markdown(
            f"**Recovered, not just retired.** On average **{_gross_avg:,.0f} t/yr** of "
            f"{short_material.lower()} leaves service; at the end-of-life recycling rate of "
            f"{', '.join(_rate_labels)}, about **{average_annual_tonnes:,.0f} t/yr** of it actually "
            "comes back as usable material. Everything below uses the recovered figure."
        )

        vals = [r["cumulative_recovered_usd"], r["real_shortfall_usd_20pct_slack"], r["real_shortfall_usd_no_slack"]]
        labels = [f"Recovered per year ({short_material.lower()})", f"Shortfall ({group['real_slack']:.1%} slack)",
                  "Shortfall (no substitution)"]
        fig4 = go.Figure(go.Bar(
            x=vals, y=labels, orientation="h", marker_color=[ACCENT, SLATE_LIGHT, SLATE],
            # Labels outside the bar, in ink, not inside in white: white text fails contrast on the
            # light slate bar. The axis range below leaves room so the longest label is never clipped.
            text=[usd(v) for v in vals], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{text}<extra></extra>",
        ))
        positive = [v for v in vals if v > 0]
        lo = np.log10(min(positive)) if positive else 0.0
        hi = np.log10(max(positive)) if positive else 1.0
        style_fig(fig4, height=260, legend=False, hovermode="closest", margin=dict(t=10, l=8, r=24, b=8))
        decades = list(range(int(np.floor(lo - 0.7)), int(np.ceil(hi + 0.55)) + 1))
        fig4.update_xaxes(type="log", range=[lo - 0.7, hi + 0.55], title_text="USD per year, log scale",
                          tickvals=[10.0 ** d for d in decades], ticktext=[usd(10.0 ** d) for d in decades])
        fig4.update_yaxes(showgrid=False, autorange="reversed")
        show_chart(fig4, "chart_recovery_vs_shortfall")

        with how_calculated():
            st.latex(
                r"\text{recovered}_{\$} = \bar{t}_{\text{material}} \times p \qquad"
                r"\text{coverage} = \frac{\text{recovered}_{\$}}{\text{shortfall}_{\$}}"
            )
            if recovery_low_result and recovery_high_result:
                st.caption(
                    "Across the recycling rate's own published range (most are reported as bins, so the "
                    "central value above is a disclosed bin midpoint): "
                    f"**{recovery_low_result['recovered_share_of_no_slack_shortfall']:.2%}-"
                    f"{recovery_high_result['recovered_share_of_no_slack_shortfall']:.2%}** of the "
                    "no-substitution shortfall."
                )
            st.caption(
                f"Cumulative {material_choice.lower()} recovered, {projection_start_year}-{horizon_year}: "
                f"{cumulative_recovered_tonnes:,.0f} t, \\${cumulative_recovered_tonnes * price / 1e6:,.1f} M "
                f"at \\${price:,.0f}/t. The comparison deliberately does **not** use that number, see why next."
            )
            st.caption(
                "Both the shortfall figures and \"average annual recovery\" describe one year, on purpose: "
                "the real cascade shortfall is a one-year disruption estimate, so comparing it against "
                "everything a technology system will ever recover by some arbitrary horizon year would "
                "compare two different time scales and can produce a share over 100% for smaller markets, "
                "not because recycling is winning, but because decades of accumulated recovery were being "
                "measured against a single year's cost. This engine compares one real year against another. "
                f"t-bar is the mean of the real year-by-year recovered-tonnes series shown in the next "
                f"chart; p is the price set in the sidebar (\\${price:,.0f}/t)."
            )
            if price_low_result and price_high_result:
                st.caption(
                    f"At the real price range's own bounds instead of the central price: "
                    f"**{price_low_result['recovered_share_of_no_slack_shortfall']:.2%}-"
                    f"{price_high_result['recovered_share_of_no_slack_shortfall']:.2%}** of the "
                    f"no-substitution shortfall and **{price_low_result['recovered_share_of_20pct_slack_shortfall']:.2%}-"
                    f"{price_high_result['recovered_share_of_20pct_slack_shortfall']:.2%}** of the {group['real_slack']:.1%}-slack "
                    f"shortfall, at \\${price_range[0]:,.0f}-\\${price_range[2]:,.0f}/t. The headline "
                    "uses the central price; this range shows how much of it is actually price-sensitive."
                )
            st.caption(
                "Whether the coverage ratio is small or substantial varies genuinely by material, set by two "
                "real things together: how much of the material is actually recycled at end of life, and "
                "how large its real global trade is (the All materials page compares them on the same "
                "basis). It is not a claim that recycling solves supply concentration; it answers whether "
                "this specific real recovery pathway matters at the scale of a real measured disruption. "
                f"Price source: {r['price_source']}. Risk source: {risk.source}."
            )

        st.subheader("Recovery by year, not just the average")
        year_values_usd = material_by_year * price
        peak_share = (peak_connector_result["recovered_share_of_no_slack_shortfall"]
                      if peak_connector_result else None)
        if peak_year is not None and peak_share is not None:
            if _at_edge(peak_year, year_values_usd):
                takeaway(f"Recovery is still rising at the horizon: in <b>{peak_year}</b> it is worth "
                         f"<b>{esc(usd(float(year_values_usd.max())))}</b>, {_pct(peak_share)} of one year's "
                         "no-substitution shortfall.")
            else:
                takeaway(f"Even in its busiest year, <b>{peak_year}</b>, recovery is worth "
                         f"<b>{esc(usd(float(year_values_usd.max())))}</b>, {_pct(peak_share)} of one year's "
                         "no-substitution shortfall.")
        fig_yby = go.Figure()
        fig_yby.add_trace(go.Scatter(
            x=year_values_usd.index, y=year_values_usd.values, mode="lines+markers",
            name=f"Recovered value that year ({short_material.lower()})",
            line=dict(color=ACCENT, width=2.5), marker=dict(size=6, color=ACCENT), hovertemplate="$%{y:,.0f}",
        ))
        fig_yby.add_hline(y=r["real_shortfall_usd_no_slack"], line=dict(color=SLATE, width=1.5, dash="dot"),
                          annotation_text="1-year shortfall, no substitution", annotation_position="top left")
        fig_yby.add_hline(y=r["real_shortfall_usd_20pct_slack"], line=dict(color=SLATE_LIGHT, width=1.5, dash="dot"),
                          annotation_text=f"1-year shortfall, {group['real_slack']:.1%} slack",
                          annotation_position="bottom left")
        if peak_year is not None:
            fig_yby.add_vline(x=peak_year, line=dict(color=NEUTRAL, width=1, dash="dash"))
        style_fig(fig_yby, height=360)
        fig_yby.update_xaxes(title_text="Year")
        fig_yby.update_yaxes(title_text="USD", tickprefix="$", tickformat="~s")
        show_chart(fig_yby, "chart_recovery_by_year")
        with how_calculated():
            st.latex(r"v_t = t_t \times p \qquad t_t = \text{real tonnes recovered in year } t "
                     r"\text{ (from the retirement wave)}")
            st.caption(
                "The metrics above compress this whole real year-by-year series down to one average-year "
                "number, which is itself a smaller version of the same time-scale problem this connector "
                "exists to avoid. This chart is the uncompressed version: v_t for every real projected "
                "year against the same two flat one-year shortfall lines. The dashed vertical line marks "
                f"the real peak retirement year ({peak_year}), where recovery is highest, not "
                "representative of a typical year."
            )
        download_csv(pd.DataFrame({"year": year_values_usd.index, "recovered_t": material_by_year.values,
                                   "recovered_usd": year_values_usd.values}),
                     f"recovery_by_year_{short_material.lower().replace(' ', '_')}.csv", "dl_recovery_by_year")

        st.subheader("With recycling vs without (one average year)")
        no_slack_total = r["real_shortfall_usd_no_slack"]
        slack_total = r["real_shortfall_usd_20pct_slack"]
        covered_no_slack = min(r["cumulative_recovered_usd"], no_slack_total)
        covered_slack = min(r["cumulative_recovered_usd"], slack_total)
        slack_label = f"{group['real_slack']:.1%} slack"
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Covered by this recycling pathway", x=[covered_no_slack, covered_slack],
                              y=["No substitution", slack_label], orientation="h", marker_color=ACCENT,
                              hovertemplate="$%{x:,.0f}"))
        fig5.add_trace(go.Bar(name="Shortfall remaining even with recycling",
                              x=[no_slack_total - covered_no_slack, slack_total - covered_slack],
                              y=["No substitution", slack_label], orientation="h", marker_color=SLATE,
                              hovertemplate="$%{x:,.0f}"))
        style_fig(fig5, height=250, hovermode="y unified")
        fig5.update_layout(barmode="stack")
        fig5.update_xaxes(title_text="USD", tickprefix="$", tickformat="~s")
        fig5.update_yaxes(showgrid=False, autorange="reversed")
        show_chart(fig5, "chart_covered_stack")
        with how_calculated():
            st.latex(r"\text{covered} = \min(\text{recovered}_{\$},\ \text{shortfall}_{\$}) \qquad "
                     r"\text{remaining} = \text{shortfall}_{\$} - \text{covered}")
            st.caption(
                "Same two shortfall totals as above, but stacked instead of separate: the coloured sliver "
                "is what recycling actually covers, the slate is what is still short even after it. "
                "Deliberately linear, not log scale, so the sliver's real size relative to the whole "
                "shortfall is honest, even though that means it may be barely visible, which is itself "
                "the finding."
            )
    footer("disruption")


def page_cascade():
    header(
        "cascade", "How the shock spreads",
        f"The disruption page asks a static question: how much trade value does {esc(removed_country)} "
        "directly supply. This asks a dynamic one: once it is gone, how many <i>other</i> countries "
        "does the disruption itself take down, as their own trade partners collapse in turn, round by round?",
    )
    takeaway(f"Removing <b>{esc(removed_country)}</b> takes down <b>{fragility['avalanche_size']}</b> of the "
             f"{fragility_universe - 1} other countries in this trade network, "
             f"<b>{fragility['avalanche_fraction']:.1%}</b> of it, in {fragility['rounds']} rounds.")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric(
        "Countries that also collapse", f"{fragility['avalanche_size']} of {fragility_universe - 1}", border=True,
        help="A country collapses once the trade value it has already lost, because a partner "
        "it bought from or sold to collapsed first, exceeds 20% of its own total trade in this "
        "commodity. Counted here after the cascade has fully run its course.",
    )
    fc2.metric("Share of the network affected", f"{fragility['avalanche_fraction']:.1%}", border=True)
    fc3.metric("Rounds for the cascade to finish", f"{fragility['rounds']}", border=True)
    with how_calculated():
        st.markdown(
            "This is the same cascading-failure method Wu Chen's own group runs for cobalt (Ouyang, Liu, "
            "Liu, Chen, Wang, Pang, He, Liu, *Environ. Sci. Ecotechnol.* 29, 2026, 100654), applied here "
            "at the one real trade layer this project has rather than their six real cobalt life-cycle "
            "stages, see crm-trade-network's own README for the full, honest scope of that difference."
        )
        st.caption(
            "Real, run-yourself result, not borrowed from the published paper: at this same 20% "
            "failure threshold, this model showed a sharp, abrupt jump between near-total collapse "
            "and near-total immunity across every commodity tested (rare earth compounds, cobalt, "
            "both lithium codes), the same \"robust-yet-fragile\" pattern reported in Wu Chen's "
            "cobalt paper and, separately, her 2024 lithium network paper. Source: crm-trade-network, "
            "src/shock_propagation.py, real 2023 UN Comtrade data."
        )

    st.subheader(f"Who trades {group['name']} with whom")
    N_TRADE_NODES = 8
    top_exporters = (commodity_edges.groupby("exporter").value_usd.sum().sort_values(ascending=False)
                     .head(N_TRADE_NODES).index.tolist())
    top_importers = (commodity_edges.groupby("importer").value_usd.sum().sort_values(ascending=False)
                     .head(N_TRADE_NODES).index.tolist())
    flow_edges = commodity_edges[
        commodity_edges.exporter.isin(top_exporters) & commodity_edges.importer.isin(top_importers)
    ]
    if flow_edges.empty:
        st.info("No real trade flows among the largest exporters and importers to show.")
    else:
        own = commodity_edges[commodity_edges.exporter == removed_country]
        if not own.empty:
            buyers = own.groupby("importer").value_usd.sum().sort_values(ascending=False)
            takeaway(f"<b>{esc(removed_country)}</b>'s exports are drawn in red. Its largest buyer is "
                     f"<b>{esc(buyers.index[0])}</b>, taking {buyers.iloc[0] / buyers.sum():.0%} of them.")
        exp_nodes = [f"{c} (exports{', removed' if c == removed_country else ''})" for c in top_exporters]
        imp_nodes = [f"{c} (imports)" for c in top_importers]
        nodes2 = exp_nodes + imp_nodes
        exp_index = {c: i for i, c in enumerate(top_exporters)}
        imp_index = {c: len(top_exporters) + i for i, c in enumerate(top_importers)}
        node_colors2 = ([CRITICAL if c == removed_country else SLATE for c in top_exporters]
                        + [SLATE_LIGHT] * len(top_importers))
        fig_sankey2 = go.Figure(go.Sankey(
            # pad is a fixed pixel gap between adjacent nodes in the same column, independent
            # of how thin a node's value makes it. A skewed distribution (one dominant supplier,
            # several minor ones) can shrink a node's bar to near zero, and without enough pad
            # its label collides with its neighbour's. 16px was fine for copper's flatter
            # distribution but not for rare earth metals, where China alone is ~40% of trade.
            node=dict(label=nodes2, color=node_colors2, pad=26, thickness=16, line=dict(width=0),
                      hovertemplate="%{label}: $%{value:,.0f}<extra></extra>"),
            link=dict(
                source=[exp_index[r.exporter] for r in flow_edges.itertuples()],
                target=[imp_index[r.importer] for r in flow_edges.itertuples()],
                value=[r.value_usd for r in flow_edges.itertuples()],
                color=[rgba(CRITICAL, 0.38) if r.exporter == removed_country else "rgba(128,128,128,0.22)"
                       for r in flow_edges.itertuples()],
                hovertemplate="%{source.label} to %{target.label}: $%{value:,.0f}<extra></extra>",
            ),
            textfont=dict(size=13, shadow="none"),
        ))
        style_fig(fig_sankey2, height=540, legend=False, hovermode="closest", margin=dict(t=10, l=8, r=8, b=10))
        show_chart(fig_sankey2, "chart_trade_sankey")
        with how_calculated():
            st.caption(
                f"Real 2023 UN Comtrade bilateral flows for {group['name']}, among the {N_TRADE_NODES} "
                f"largest exporters and {N_TRADE_NODES} largest importers by value. {removed_country} is "
                "drawn in red: that is the node the disruption scenario removes. No formula here: link "
                "width is each country pair's own reported export value_usd, taken directly from UN "
                "Comtrade as reported, with no modeling applied."
            )
        download_csv(flow_edges[["exporter", "importer", "value_usd", "net_kg"]],
                     f"trade_flows_hs{group['hs_code']}.csv", "dl_trade_flows")
    footer("cascade")


def page_diversification():
    header(
        "diversification", "What it would take to diversify",
        "Not how bad the concentration is, but how much real trade would actually have to move to "
        "fix it: the smallest reallocation of 2023 export value across existing suppliers that brings "
        "the largest single supplier's share down to a target.",
    )
    target_pct = st.slider(
        "Target: maximum share of trade for any one supplier", min_value=10, max_value=90,
        value=_TARGET_DEFAULT_PCT, step=5, format="%d%%", help=EU_CRMA_SOURCE,
        key=f"target_share_{material_choice}",
    )
    # target_share and both LP results are computed at module level from this same widget key, so
    # they already match the slider; recomputed here only if the two ever disagree.
    t_share, d_no, d_slack = target_share, div_no_slack, div_slack
    if abs(target_pct / 100.0 - t_share) > 1e-9:
        t_share = target_pct / 100.0
        d_no = minimum_diversification(commodity_edges, t_share, 0.0)
        d_slack = minimum_diversification(commodity_edges, t_share, group["real_slack"])
    current_top = float(d_slack.before.iloc[0] / d_slack.before.sum()) if len(d_slack.before) else 0.0
    if current_top <= t_share + 1e-9:
        takeaway(f"Every supplier is already under the <b>{t_share:.0%}</b> cap: the largest, "
                 f"<b>{esc(d_slack.before.index[0])}</b>, holds <b>{current_top:.1%}</b> of trade, so nothing "
                 "has to move. Lower the target to see what diversifying further would take.")
    elif d_slack.feasible:
        takeaway(f"Bringing every supplier under <b>{t_share:.0%}</b> means moving "
                 f"<b>{esc(usd(d_slack.reallocated_usd))}</b> of trade "
                 f"({d_slack.reallocated_usd / total_trade_usd:.1%} of the total) when suppliers can "
                 f"expand by up to {group['real_slack']:.1%}.")
    else:
        takeaway(f"A <b>{t_share:.0%}</b> cap is <b>not achievable</b> by reallocating existing "
                 f"suppliers' trade within {group['real_slack']:.1%} spare capacity.")
    dc1, dc2 = st.columns(2)
    dc1.metric("Reallocation needed, no substitution",
               f"${d_no.reallocated_usd/1e6:,.1f} M" if d_no.feasible else "Not achievable",
               border=True)
    dc2.metric(f"Reallocation needed, {group['real_slack']:.1%} slack",
               f"${d_slack.reallocated_usd/1e6:,.1f} M" if d_slack.feasible else "Not achievable",
               border=True)

    if d_slack.feasible and d_slack.after is not None:
        top_n = 8
        before_top = d_slack.before.head(top_n)
        after_aligned = d_slack.after.reindex(before_top.index)
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(name="Real current (2023)", y=before_top.index, x=before_top.values,
                                 orientation="h", marker_color=SLATE_LIGHT, hovertemplate="$%{x:,.0f}"))
        fig_div.add_trace(go.Bar(name=f"After (target {t_share:.0%}, {group['real_slack']:.1%} slack)",
                                 y=before_top.index, x=after_aligned.values, orientation="h",
                                 marker_color=ACCENT, hovertemplate="$%{x:,.0f}"))
        fig_div.add_vline(x=t_share * total_trade_usd, line=dict(color=NEUTRAL, width=1, dash="dot"),
                          annotation_text=f"{t_share:.0%} cap", annotation_position="top")
        style_fig(fig_div, height=110 + 42 * len(before_top), hovermode="y unified")
        fig_div.update_layout(barmode="group", bargap=0.25)
        fig_div.update_xaxes(title_text="Export value, USD", tickprefix="$", tickformat="~s")
        fig_div.update_yaxes(showgrid=False, autorange="reversed")
        show_chart(fig_div, "chart_diversification")
        download_csv(pd.DataFrame({"supplier": before_top.index, "current_usd": before_top.values,
                                   "after_usd": after_aligned.values}),
                     f"diversification_hs{group['hs_code']}.csv", "dl_diversification")
    with how_calculated():
        st.latex(
            r"\min \sum_i d_i \ \text{s.t.}\ \sum_i x_i = T,\ |x_i - c_i| \le d_i,\ "
            r"0 \le x_i \le \min(\text{target}\cdot T,\ c_i(1+\text{slack}))"
        )
        st.caption(
            "A linear program (scipy.optimize.linprog). c_i is each real supplier's current 2023 export "
            "value, x_i its reallocated value, d_i how much that changes either way, T the real total "
            "trade value (conserved, not grown or shrunk). The ceiling on how much any supplier could "
            "realistically expand into is the same real spare-capacity assumption (slack) used everywhere "
            "else. At 0% slack every supplier's ceiling collapses to exactly its own current volume, so if "
            "the current top share already exceeds the target, no reallocation is mathematically possible "
            "at all, a real finding about why spare capacity matters, not a bug. The default target is the "
            "EU Critical Raw Materials Act benchmark."
        )
        st.caption(d_slack.note)
    footer("diversification")


def page_copper():
    header(
        "copper", "One material, three technologies: copper",
        "Copper is the one material all three systems contain, so it is where competition between "
        "low-carbon and digital technologies for the same material becomes visible.",
    )
    cu_view = st.segmented_control(
        "Show", ["Entering service (inflow)", "In use (stock)", "Leaving service (outflow)"],
        default="In use (stock)", required=True, key="cu_cross_view",
    )
    cu_col = {"Entering service (inflow)": "inflow_t", "In use (stock)": "stock_t",
              "Leaving service (outflow)": "outflow_t"}[cu_view]
    cu_all = compute_copper_across_systems(horizon_year)
    cu_all = cu_all[cu_all.year <= horizon_year]
    cu_keys = {"Danish wind turbines (existing fleet only)": "wind",
               "global EV batteries (IEA STEPS)": "ev",
               "global data centres (McKinsey low)": "dc"}
    year_ref = min(2030, horizon_year)
    cu_ref = cu_all[cu_all.year == year_ref].set_index("system")[cu_col]
    if cu_ref.sum() > 0:
        short = {"wind": "Danish wind turbines", "ev": "EV batteries", "dc": "data centres"}
        shares = sorted(((v / cu_ref.sum(), short[cu_keys[k]]) for k, v in cu_ref.items()), reverse=True)
        parts = [f"{name} <b>{sh:.0%}</b>" if sh >= 0.005 else f"{name} <b>under 1%</b>" for sh, name in shares]
        takeaway(f"Of the copper {cu_view.split(' (')[0].lower()} across the three systems in {year_ref}: "
                 + ", ".join(parts) + ".")
    fig_cu = go.Figure()
    for sys_label, sub in cu_all.groupby("system"):
        key = cu_keys.get(sys_label)
        selected = group["name"] == "refined copper" and key == group["system"]
        fig_cu.add_trace(go.Scatter(
            x=sub.year, y=sub[cu_col], mode="lines", name=sys_label + (" (selected)" if selected else ""),
            line=dict(color=SYSTEM_COLORS.get(key, NEUTRAL), width=3.5 if selected else 2.2),
            hovertemplate="%{y:,.0f} t",
        ))
    style_fig(fig_cu, height=430)
    fig_cu.update_xaxes(title_text="Year")
    fig_cu.update_yaxes(title_text=f"Copper, tonnes ({cu_view.lower()}), log scale", type="log",
                        dtick=1, tickformat="~s")
    show_chart(fig_cu, "chart_copper_cross")
    cu_cols = st.columns(len(cu_ref))
    short_label = {"wind": "Danish wind turbines", "ev": "Global EV batteries", "dc": "Global data centres"}
    for col, (sys_label, value) in zip(cu_cols, cu_ref.items()):
        col.metric(f"{short_label[cu_keys[sys_label]]}, {year_ref}", f"{value/1000:,.1f} kt", border=True,
                   help=f"{sys_label}: copper {cu_view.split(' (')[0].lower()} in {year_ref}.")
    with how_calculated():
        st.caption(
            "Same axes for all three: copper entering service (new builds and replacements), copper in "
            "use, and copper leaving service. The scales differ because the data does: Denmark's wind "
            "fleet is one country's register, EV batteries and data centres are global. Log scale, so "
            "three systems of very different size stay readable on one chart. Each system runs its own "
            "forward projection so the comparison is like for like: data centres on McKinsey's low "
            "capacity case (171 GW by 2030), EVs on IEA's STEPS sales trajectory. Denmark's wind register "
            "has no projection of new turbines yet, so wind shows its existing fleet only: no inflow, and "
            "a stock that falls as turbines retire. EV copper is whole-vehicle copper (motor, wiring, "
            "busbars); wind copper is generator and cabling copper per JRC; data-centre copper is facility "
            "plus grid connection per WEF/Kearney. Everything here is gross physical flow, before any "
            "end-of-life recycling rate is applied. The thicker line is the system currently selected in "
            "the sidebar, when that material is copper."
        )
    download_csv(cu_all[["system", "year", "inflow_t", "stock_t", "outflow_t"]], "copper_across_systems.csv",
                 "dl_copper_cross")
    footer("copper")


def page_materials():
    header(
        "materials", "Which recovery pathway matters most",
        "The disruption calculation, run once for every material against its own largest supplier "
        "(not necessarily the supplier selected in the sidebar), so all of them sit on the same basis.",
    )
    all_scenarios = compute_all_material_scenarios(horizon_year, recovery_regime)
    plottable = all_scenarios.dropna(subset=["coverage_no_slack", "coverage_20pct"])
    skipped = all_scenarios[all_scenarios.coverage_no_slack.isna()]

    if not plottable.empty:
        plottable = plottable.sort_values("coverage_no_slack", ascending=False)
        best = plottable.iloc[0]
        takeaway(f"<b>{esc(best.material)}</b> has the recovery pathway that matters most: an average year "
                 f"covers <b>{best.coverage_no_slack:.2%}</b> of one year's shortfall if "
                 f"{esc(best.top1)} stopped exporting. The selected material is in bold.")
        ticktext = [f"<b>{esc(m)}</b>" if m == material_choice else esc(m) for m in plottable.material]
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(name="No substitution", y=plottable.material, x=plottable.coverage_no_slack,
                              orientation="h", marker_color=SLATE,
                              text=[f"{v:.2%}" for v in plottable.coverage_no_slack], textposition="outside",
                              cliponaxis=False, hovertemplate="%{x:.2%}"))
        fig6.add_trace(go.Bar(name="With real spare capacity", y=plottable.material, x=plottable.coverage_20pct,
                              orientation="h", marker_color=SLATE_LIGHT,
                              text=[f"{v:.2%}" for v in plottable.coverage_20pct], textposition="outside",
                              cliponaxis=False, hovertemplate="%{x:.2%}"))
        style_fig(fig6, height=110 + 70 * len(plottable), hovermode="y unified", margin=dict(t=44, l=8, r=70, b=8))
        fig6.update_layout(barmode="group", bargap=0.28)
        fig6.update_xaxes(title_text="Share of one year's real shortfall an average year of recovery covers",
                          tickformat=".1%")
        fig6.update_yaxes(showgrid=False, tickmode="array", tickvals=list(plottable.material), ticktext=ticktext,
                          autorange="reversed")
        show_chart(fig6, "chart_all_materials")
        with how_calculated():
            st.latex(r"\text{coverage}_{\text{material}} = \frac{\bar{t}_{\text{material}} \times "
                     r"p_{\text{material}}}{\text{shortfall}_{\$,\,\text{material}}}")
            st.caption(
                "\"With real spare capacity\" uses each material's own real slack, not one flat number: "
                "copper's is a real, cited 22.4% (ICSG 2023 capacity utilization); the others use the "
                "same disclosed 20% illustrative assumption crm-trade-network's own connector already "
                "uses, since no comparable published figure exists for any of them (the slack panel on the "
                "Disruption page gives each material's specific reason). Each material removes its own "
                "real largest supplier (" +
                "; ".join(f"{row.material}: {row.top1} ({row.top1_share:.1%})" for row in plottable.itertuples()) +
                "). Average annual recovery vs one year's shortfall: the two figures describe the same "
                "one-year time scale on purpose. A small real global market lifts a material's ratio, "
                "a genuine difference between real markets, not an error. None of these is a claim that "
                "recycling solves supply concentration. Both bars for a material are the same quantity "
                "under a harsher and a milder assumption, so they share one hue, darker for the harsher."
            )
        download_csv(all_scenarios, f"all_materials_{horizon_year}.csv", "dl_all_materials")
    if not skipped.empty:
        _no_rate = (skipped[skipped["note"] == "no cited recycling rate"] if "note" in skipped
                    else skipped.iloc[0:0])
        _no_shortfall = skipped.drop(_no_rate.index)
        if not _no_rate.empty:
            st.caption("Not shown, no cited end-of-life recycling rate to compute a recovered "
                       "tonnage from: " + ", ".join(_no_rate.material.tolist()))
        if not _no_shortfall.empty:
            st.caption(
                "Not shown (removing that material's own largest supplier doesn't produce a comparable "
                "shortfall in this dataset): " + ", ".join(_no_shortfall.material.tolist())
            )
    footer("materials")


def page_ask():
    header(
        "ask", "Ask the data",
        "Grounded, not a general chatbot: Claude is given only the numbers already computed for the "
        "selected material and told to answer from those alone, and to say plainly when a question "
        "asks about something they don't cover.",
    )
    with st.form("ask_form", border=True):
        api_key = st.text_input(
            "Your Anthropic API key", type="password", key="ask_api_key",
            help="Bring-your-own-key, the same real pattern already live on this project's "
            "battery-electrode-screening-agent demo: a shared key would mean one visitor's questions "
            "bill another visitor's account. Used for one call and never stored. Get one at "
            "console.anthropic.com/settings/keys.",
        )
        question = st.text_input(
            "Your question", key="ask_question",
            placeholder="e.g. why is so little of the neodymium leaving service actually recovered?",
        )
        asked = st.form_submit_button("Ask", type="primary", icon=":material/send:")
    if asked:
        if not api_key:
            st.warning("Enter your own Anthropic API key above first.")
        elif not question:
            st.warning("Type a question first.")
        else:
            all_scenarios = compute_all_material_scenarios(horizon_year, recovery_regime)
            context = build_context(
                material_label=material_choice, system_label=system["label"], horizon_year=horizon_year,
                hhi=f"{commodity_conc['hhi']:.3f}", top1_country=commodity_conc["top1"],
                top1_share=f"{commodity_conc['top1_share']:.1%}",
                removed_country=removed_country, removed_country_share=f"{removed_country_share:.1%}",
                no_slack_shortfall=f"${no_slack_shortfall/1e9:,.2f} bn",
                slack_shortfall=f"${slack_shortfall/1e9:,.2f} bn",
                recovery_regime=recovery_regime,
                recovery_rate=(", ".join(sorted({f"{m}: {rt.central:.1%} ({rt.source})"
                                                 for m, rt in material_rates.items() if rt}))
                               or "no cited end-of-life recycling rate for this material"),
                gross_outflow_t=f"{float(outflow_by_year.mean()):,.0f} t/yr" if len(outflow_by_year) else None,
                recovered_t=f"{average_annual_tonnes:,.0f} t/yr" if recovery_assessed else "not assessed",
                avg_annual_usd=f"${connector_result['cumulative_recovered_usd']/1e6:,.1f} M/yr" if connector_result else None,
                peak_year=peak_year,
                peak_usd=f"${peak_connector_result['cumulative_recovered_usd']/1e6:,.1f} M" if peak_connector_result else None,
                coverage_no_slack=f"{connector_result['recovered_share_of_no_slack_shortfall']:.2%}" if connector_result else None,
                coverage_20pct=f"{connector_result['recovered_share_of_20pct_slack_shortfall']:.2%}" if connector_result else None,
                real_slack_pct=f"{group['real_slack']:.1%}",
                price=f"${price:,.0f}/t", price_source=price_source,
                price_range=(f"${price_low_result['recovered_share_of_no_slack_shortfall']:.2%}-"
                             f"{price_high_result['recovered_share_of_no_slack_shortfall']:.2%} coverage across the "
                             f"real cited price range") if (price_low_result and price_high_result) else None,
                target_share=f"{target_share:.0%}" if target_share else None,
                div_no_slack_note=div_no_slack.note if div_no_slack else None,
                div_slack_note=div_slack.note if div_slack else None,
                cross_material_table=(
                    "; ".join(
                        f"{row.material}: {row.coverage_no_slack:.2%}/{row.coverage_20pct:.2%} "
                        f"(no-substitution/{row.real_slack:.1%}-slack coverage, each material's own real "
                        "slack figure, not a shared flat rate)"
                        for row in all_scenarios.dropna(subset=["coverage_no_slack"]).itertuples()
                    ) if not all_scenarios.empty else None
                ),
            )
            try:
                with st.spinner("Asking Claude, grounded only in the numbers on these pages..."):
                    answer = ask_dashboard(api_key, question, context)
                with st.container(border=True):
                    st.markdown(f"**Answer:** {answer}")
                with st.expander("What was actually sent as context (so you can check it isn't inventing anything)"):
                    st.code(context, language=None)
            except Exception as exc:
                st.error(f"Real error calling the Anthropic API: {exc}")
    footer("ask")


def page_methods():
    header(
        "methods", "Methods and data",
        "What this dashboard is, what it deliberately is not, where every number comes from, and how "
        "fresh the trade data is.",
    )
    st.subheader("How to read this dashboard")
    st.markdown(HOW_TO_READ)

    st.subheader("What this app actually is, and isn't")
    st.markdown(
        "- **The physical side** runs three real, independent case studies through the same engine, "
        "one per textbook dMFA mode: Denmark's real wind turbine fleet (register-based, verified to "
        "reproduce dk-wind-mfa's own published numbers exactly), the global EV fleet (inflow-driven, "
        "verified against a real external vehicle survival curve, since the real EV fleet is too young "
        "to have generated enough of its own retirement events to fit one from scratch), and global data "
        "centres (stock-driven: required capacity from McKinsey's published 2030 range, per-layer "
        "lifetimes from ASHRAE). Wind and EV lifetimes come from survival analysis with a fitted Weibull "
        "curve; running genuinely different systems through the same base classes, not just one, is "
        "what makes \"modular and reusable\" a demonstrated claim rather than an assertion.\n"
        "- **The risk side** is not computed here. It is crm-trade-network's own real concentration "
        "(HHI) and single-supplier cascade-shortfall analysis of 2023 UN Comtrade data, loaded as-is.\n"
        "- **The connector** takes the tonnes actually recycled at end of life (the UN International "
        "Resource Panel's measured recycling rates, or the EU battery targets as an upper bound), "
        "converts them to USD with one explicitly documented price, either an external market quote "
        "(copper, lithium) or a real average implied directly by the same trade dataset (rare earths, "
        "nickel, cobalt, graphite), each given as a real (low, central, high) range where one exists, "
        "then asks a narrow, honest question: does this real recovery pathway matter at the scale of a "
        "real measured supply-disruption risk. It is not a claim that recycling solves supply "
        "concentration.\n"
        "- Every material connected here is one a real physical system actually contains, with a "
        "matching, trackable UN Comtrade code: copper and rare earths from wind turbines; copper, "
        "lithium, nickel, cobalt and graphite from EV batteries; copper from data centres. That is every "
        "one of Wu Chen's own named Villum grant materials (cobalt, copper, nickel, lithium), and copper "
        "specifically is tracked from all three systems so the same real recovered tonne can be compared "
        "across technologies, not just within one.\n"
        "- No confidence interval is plotted on the survival curve, because this engine does not "
        "compute one. Showing a fabricated band would be worse than showing none.\n"
        "- **Why Denmark for wind but the whole world for EVs and data centres, a deliberate design "
        "choice, not an accident.** Denmark's wind fleet is modeled nationally because dk-wind-mfa's own "
        "register, the real source this case reproduces exactly, is itself a national Danish register "
        "with individual turbine coordinates and commissioning dates; no comparably detailed global wind "
        "turbine register with real per-unit retirement data was found to build a global version from. "
        "The EV case is modeled globally because its own real source (IEA's Global EV Outlook) only "
        "reports sales and chemistry mix at the global level, a single country's EV fleet would be too "
        "small and too young to say anything about chemistry mix or retirement timing with any real "
        "confidence; the data-centre capacity figures are likewise published globally. Comparing a "
        "national system against global ones is therefore a property of which real data actually exists "
        "for each technology today, not a shortcut taken to make the numbers line up. The honest "
        "consequence, which this app does not hide, is that the physical systems are not being compared "
        "at the same geographic scale; what they share, and what the connector actually compares, is the "
        "same real commodity (refined copper) and the same real 2023 global trade-risk dataset, not the "
        "same population size."
    )

    st.subheader("Data policy")
    st.markdown(
        "Physical flows come from dk-wind-mfa or real, cited external sources (each system's own module "
        "lists its citations), verified against real published or reported numbers. Trade concentration "
        "and cascade shortfall come from crm-trade-network's real 2023 UN Comtrade data. No number here "
        "is simulated or fabricated. Colours are checked for colour-vision deficiency and contrast in "
        "both light and dark mode (ui/theme.py)."
    )

    st.subheader("Trade data freshness")
    if fresh["count"]:
        st.markdown(
            f"{fresh['count']} real UN Comtrade queries on disk for {group['name']}, fetched between "
            f"{fresh['oldest']:%Y-%m-%d} and {fresh['newest']:%Y-%m-%d}. This is a real snapshot, not "
            "re-fetched on every page load, because the API is rate-limited; the button below genuinely "
            "re-asks it right now."
        )
    else:
        st.markdown(f"No cached {group['name']} data on disk yet. Use the button below to fetch it live.")
    with st.container(border=True):
        st.caption(
            f"Re-asks UN Comtrade's live API for every one of the ~32 reporting countries, for {group['name']}, "
            "both import and export directions, and overwrites the cached files with today's real response. "
            "The API is rate-limited to one new request every 4 seconds, so this takes several minutes. "
            "Keep this tab open until it finishes."
        )
        if st.button("Fetch live data now (re-ask UN Comtrade)", key=f"refresh_{material_choice}",
                     icon=":material/cloud_download:"):
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
    footer("methods")


# --------------------------------------------------------------------------- navigation

PAGES = {
    "overview": st.Page(page_overview, title="Overview", icon=":material/space_dashboard:",
                        url_path="overview", default=True),
    "lifetimes": st.Page(page_lifetimes, title="Lifetimes", icon=":material/hourglass_bottom:", url_path="lifetimes"),
    "stocks": st.Page(page_stocks, title="Stocks & flows", icon=":material/swap_vert:", url_path="stocks-and-flows"),
    "eol": st.Page(page_eol, title="End-of-life materials", icon=":material/recycling:", url_path="end-of-life"),
    "scenarios": st.Page(page_scenarios, title="Scenarios", icon=":material/alt_route:", url_path="scenarios"),
    "supplymap": st.Page(page_supply_map, title="Supply map", icon=":material/public:", url_path="supply-map"),
    "disruption": st.Page(page_disruption, title="Disruption & recovery", icon=":material/crisis_alert:",
                          url_path="disruption"),
    "cascade": st.Page(page_cascade, title="Cascade & trade network", icon=":material/hub:", url_path="cascade"),
    "diversification": st.Page(page_diversification, title="Diversification", icon=":material/call_split:",
                               url_path="diversification"),
    "copper": st.Page(page_copper, title="Copper across technologies", icon=":material/stacked_line_chart:",
                      url_path="copper"),
    "materials": st.Page(page_materials, title="All materials", icon=":material/leaderboard:", url_path="all-materials"),
    "ask": st.Page(page_ask, title="Ask the data", icon=":material/forum:", url_path="ask"),
    "methods": st.Page(page_methods, title="Methods & data", icon=":material/menu_book:", url_path="methods"),
}
NAV_SECTIONS = {
    "Start": ["overview"],
    "Physical flows": ["lifetimes", "stocks", "eol", "scenarios"],
    "Supply risk": ["supplymap", "disruption", "cascade", "diversification"],
    "Compare": ["copper", "materials"],
    "Reference": ["ask", "methods"],
}
navigation = st.navigation(
    {section: [PAGES[k] for k in keys] for section, keys in NAV_SECTIONS.items()}, position="hidden",
)
# st.page_link gives no hook for "this is the current page", so the current link is found by its
# href (the default page links to "") and marked with the material colour.
current_key = next((k for k, pg_ in PAGES.items() if pg_.url_path == navigation.url_path), "overview")
current_href = "" if current_key == "overview" else PAGES[current_key].url_path
st.html(
    "<style>[data-testid='stSidebar'] a[data-testid='stPageLink-NavLink']"
    f"[href='{current_href}'] {{ box-shadow: inset 3px 0 0 var(--mat); "
    "background: rgba(var(--mat-rgb), .15) !important; font-weight: 600; }</style>"
)
with nav_slot:
    for section, keys in NAV_SECTIONS.items():
        st.html(f'<div class="sb-label nav-label">{esc(section)}</div>')
        for k in keys:
            st.page_link(PAGES[k], label=PAGES[k].title, icon=PAGES[k].icon, width="stretch")
navigation.run()
