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
copper and rare earth metals from wind turbines, lithium, nickel, cobalt
and graphite from EV batteries. That is every one of Wu Chen's own named
Villum grant materials (cobalt, copper, nickel, lithium), not just copper.
"""
from __future__ import annotations

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

DK_WIND_MFA_SRC = Path(r"F:\job applications claude\dk-wind-mfa\src")
CRM_TRADE_NETWORK_SRC = Path(r"F:\job applications claude\crm-trade-network\src")


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

# Every material either real physical system actually contains that also has
# a real, trackable UN Comtrade commodity code. Bulk structural materials
# (steel, concrete, cast iron) have no CRM trade code; lithium/cobalt/nickel/
# graphite have real trade data but only EV batteries (not wind turbines)
# actually contain them.
MATERIAL_GROUPS = {
    "Copper": {
        "system": "wind",
        "hs_code": "7403",
        "name": "refined copper",
        "physical_columns": ["copper"],
        "price_mode": "external",
        "external_price": 12_842.0,
        "external_price_source": "2026 average LME copper price, Trading Economics / MacroMicro, checked live",
    },
    "Rare earth metals": {
        "system": "wind",
        "hs_code": "2805",
        "name": "rare earth metals (Nd + Dy + Pr + Tb combined)",
        "physical_columns": ["neodymium", "dysprosium", "praseodymium", "terbium"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
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
        "external_price_source": "2023 average battery-grade lithium carbonate price, China, "
        "Benchmark Mineral Intelligence (checked live), matching the 2023 UN Comtrade trade year",
    },
    "Nickel": {
        "system": "ev",
        "hs_code": "7502",
        "name": "unwrought nickel",
        "physical_columns": ["nickel"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
    },
    "Cobalt": {
        "system": "ev",
        "hs_code": "8105",
        "name": "cobalt mattes and articles",
        "physical_columns": ["cobalt"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
    },
    "Graphite": {
        "system": "ev",
        "hs_code": "2504",
        "name": "natural graphite",
        "physical_columns": ["graphite"],
        "price_mode": "implied",
        "external_price": None,
        "external_price_source": None,
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
else:
    implied = implied_unit_price(commodity_edges)
    default_price = round(implied, 2) if implied else 0.0
    default_price_source = (
        f"implied 2023 UN Comtrade unit value for HS {group['hs_code']} (total trade value divided "
        "by total reported weight): a real blended average across whichever specific goods were "
        "actually traded under this code, not one element's spot price"
    )

price = st.sidebar.number_input(
    f"{material_choice} price (USD/tonne)", min_value=100.0, max_value=500_000.0,
    value=default_price, step=50.0,
    help=f"Used only to convert recovered tonnes to USD. Default: {default_price_source}.",
    key=f"price_input_{material_choice}",
)
price_source = default_price_source if price == default_price else "user-entered override"

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

no_slack_shortfall = cascade_no_slack["shortfall_usd"]
slack_shortfall = cascade_20pct["shortfall_usd"]

if no_slack_shortfall <= 0:
    connector_result = None
    scenario_note = ("no_exports", None)
elif slack_shortfall <= 0:
    # A real, distinct outcome, not an edge case to hide: removing this supplier creates a
    # genuine no-substitution shortfall, but the remaining suppliers' 20% slack fully absorbs
    # it. recovery_vs_disruption divides by both shortfalls, so it is never called with a zero
    # denominator; this case is reported honestly instead of crashing or being mislabeled.
    connector_result = None
    scenario_note = ("slack_covers_it", no_slack_shortfall)
else:
    connector_result = recovery_vs_disruption(material_by_year, price, price_source, risk)
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
        "either an external market quote (copper) or a real average implied directly by the same "
        "trade dataset, then asks a narrow, honest question: does this real recovery pathway matter "
        "at the scale of a real measured supply-disruption risk. It is not a claim that recycling "
        "solves supply concentration.\n"
        "- Every material connected here is one a real physical system actually contains, with a "
        "matching, trackable UN Comtrade code: copper and rare earths from wind turbines; lithium, "
        "nickel, cobalt and graphite from EV batteries. That is every one of Wu Chen's own named "
        "Villum grant materials (cobalt, copper, nickel, lithium).\n"
        "- No confidence interval is plotted on the survival curve, because this engine does not "
        "compute one. Showing a fabricated band would be worse than showing none."
    )

m1, m2, m3, m4 = st.columns(4)
weibull_median = system_result["weibull_scale"] * np.log(2) ** (1 / system_result["weibull_shape"])
mats_all = system_result["secondary_materials"]
cum_material = mats_all[mats_all.year <= horizon_year][group["physical_columns"]].sum().sum()
m1.metric("Weibull median lifetime", f"{weibull_median:,.1f} yr")
m2.metric(f"Cumulative {material_choice.lower()} to {horizon_year}", f"{cum_material:,.0f} t")
m3.metric(f"Real {material_choice.lower()} HHI (pooled)", f"{commodity_conc['hhi']:.3f}")
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
        st.caption(
            f"Kaplan-Meier median: {system_result['median_survival_years']:.2f} years "
            f"(stops at the oldest observed retirement). Weibull median: {weibull_median:.2f} years "
            f"(shape k={k:.2f}, scale \u03bb={lam:.2f}), used to extrapolate the retirement projection "
            "beyond what has actually been observed."
        )
    else:
        st.caption(
            f"No empirical curve here: {system['label']} has not existed long enough to generate "
            "enough real retirement events to fit one from (real finding: only about 3% of global "
            "EV battery capacity had been scrapped as of December 2020). This Weibull (shape "
            f"k={k:.2f}, scale \u03bb={lam:.2f}, median {weibull_median:.2f} years) is instead fitted "
            "directly against a real, externally published vehicle survivability table (NHTSA, "
            "DOT HS 809 952), the same real source Argonne National Laboratory's own EV assessment "
            "uses for this exact purpose."
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
        if len(chosen) == 1:
            st.caption(
                f"Shaded band: the real low-high range material_intensity() reports for {chosen[0]}, "
                "propagated through the same cohort projection as the central line. Pick a second "
                "material above to compare trends instead, the band is only shown for one at a time "
                "to keep it legible."
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
            "material. Link width is proportional to tonnes; this is the same arithmetic as the area "
            "chart above, just keeping the breakdown that chart pools away."
        )

# --------------------------------------------------------------------------- tab 3: risk

with tab_risk:
    st.markdown(
        f"**Scenario:** remove **{removed_country}** as a supplier of {group['name']}, "
        f"{removed_country_share:.1%} of real 2023 trade by value."
    )

    if connector_result is None and scenario_note[0] == "no_exports":
        st.warning(
            f"{removed_country} has no recorded exports of {group['name']} in this dataset, "
            "so there is no shortfall to compare against. Pick a different supplier."
        )
    elif connector_result is None and scenario_note[0] == "slack_covers_it":
        st.info(
            f"Removing {removed_country} creates a real **${scenario_note[1]/1e9:,.2f} bn** "
            "shortfall with no substitution, but the remaining suppliers' 20% slack fully "
            f"absorbs it; the 20%-slack shortfall is zero. There is nothing left for the "
            f"recovered {material_choice.lower()} to offset in that scenario, so no coverage "
            "percentage is shown."
        )
    else:
        r = connector_result
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Recovered value ({material_choice.lower()}, to horizon)", f"${r['cumulative_recovered_usd']/1e6:,.1f} M")
        c2.metric("Shortfall, no substitution", f"${r['real_shortfall_usd_no_slack']/1e9:,.2f} bn")
        c3.metric("Shortfall, 20% slack", f"${r['real_shortfall_usd_20pct_slack']/1e9:,.2f} bn")

        fig4 = go.Figure(go.Bar(
            x=[r["cumulative_recovered_usd"], r["real_shortfall_usd_20pct_slack"], r["real_shortfall_usd_no_slack"]],
            y=[f"Recovered ({material_choice.lower()})", "Shortfall (20% slack)", "Shortfall (no substitution)"],
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
            f"This recovery covers **{r['recovered_share_of_no_slack_shortfall']:.2%}** of the "
            f"no-substitution shortfall and **{r['recovered_share_of_20pct_slack_shortfall']:.2%}** "
            "of the 20%-slack shortfall. This is a small, honest ratio, not a claim that recycling "
            "solves supply concentration; it answers whether this specific real recovery pathway "
            "matters at the scale of a real measured disruption, and by how much."
        )
        st.caption(f"Price source: {r['price_source']}. Risk source: {risk.source}.")

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
            "highlighted in pink: that is the node the disruption scenario above removes."
        )
