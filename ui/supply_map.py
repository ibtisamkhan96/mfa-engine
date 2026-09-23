"""The Supply map page's data: where a material is mined, where it is traded, and how a
disruption spreads, prepared in Python so the browser only has to animate it.

Three layers, each from data this project already verifies elsewhere:
  - mine production or reserves by country: data/mine_supply.csv (USGS Mineral Commodity
    Summaries 2025 via the Materials Data Series, see scripts/build_mine_supply.py);
  - bilateral trade flows: the same 2023 UN Comtrade edge list every risk page uses;
  - the cascade replay: crm-trade-network's simulate_cascade, round by round.

Geometry (data/world_map.json) is projected once by scripts/build_world_map.py. Countries are
matched by ISO3 code: trade partners through UN Comtrade's own partner table, producers through
the explicit table in build_mine_supply.py. Anything that cannot be placed is counted and
reported on the page, never silently dropped.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.theme import CRITICAL, SLATE

ROOT = Path(__file__).resolve().parent.parent
WORLD_FILE = ROOT / "data" / "world_map.json"
SUPPLY_FILE = ROOT / "data" / "mine_supply.csv"

# UN Comtrade reports Taiwan's trade as "Other Asia, nes" (code 490); every other name maps
# through the ISO3 code in Comtrade's own partner table.
COMTRADE_OVERRIDES = {"Other Asia, nes": "TWN"}

MAX_BUBBLE = 26.0      # SVG units, on a map 1000 units wide
MIN_FLOW_W, MAX_FLOW_W = 0.7, 6.0


@st.cache_data(show_spinner=False)
def load_world() -> dict:
    return json.loads(WORLD_FILE.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_supply() -> pd.DataFrame:
    return pd.read_csv(SUPPLY_FILE, keep_default_na=False)


@st.cache_data(show_spinner=False)
def comtrade_iso3(partners_json: str) -> dict[str, str]:
    rows = json.loads(Path(partners_json).read_text(encoding="utf-8"))["results"]
    out = {r["PartnerDesc"]: r.get("PartnerCodeIsoAlpha3") or "" for r in rows}
    out.update(COMTRADE_OVERRIDES)
    return out


def _fmt_t(t: float) -> str:
    if t >= 1e6:
        return f"{t / 1e6:,.2f} Mt"
    if t >= 1e3:
        return f"{t / 1e3:,.1f} kt"
    return f"{t:,.0f} t"


def _fmt_usd(v: float) -> str:
    if v >= 1e9:
        return f"${v / 1e9:,.2f} bn"
    if v >= 1e6:
        return f"${v / 1e6:,.1f} M"
    return f"${v:,.0f}"


def _arc(x1: float, y1: float, x2: float, y2: float, height: float) -> tuple[float, float]:
    """Control point of a quadratic arc that bows away from the equator side, so flows read
    as arcs over the map rather than straight lines through it, and two opposite flows
    between the same pair of countries do not sit on top of each other."""
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy) or 1.0
    nx, ny = dy / dist, -dx / dist             # left-hand normal of the travel direction
    bend = min(0.24 * dist, 120.0)
    cx, cy = mx + nx * bend, my + ny * bend
    return round(min(max(cx, 2.0), 998.0), 1), round(min(max(cy, 2.0), height - 2.0), 1)


def build_payload(*, family: str, measure: str, edges: pd.DataFrame, top_n: int,
                  removed_country: str, cascade: dict, accent: str, commodity: str,
                  hs_code: str, partners_json: str, hubs: set[str], universe: int) -> tuple[dict, dict]:
    """Everything the map component draws, plus notes for the page (what could not be placed)."""
    world = load_world()
    anchors = world["anchors"]
    iso = comtrade_iso3(partners_json)

    def place(name: str) -> tuple[str, list[float]] | None:
        code = iso.get(name, "")
        return (code, anchors[code]) if code in anchors else None

    # ---- production / reserves circles
    rows = load_supply()
    rows = rows[(rows.family == family) & (rows.measure == measure)]
    placed_rows = rows[rows.iso3 != ""]
    other_rows = rows[rows.iso3 == ""]
    vmax = float(placed_rows.tonnes.max()) if len(placed_rows) else 1.0
    bubbles, tint = [], {}
    for r in placed_rows.sort_values("tonnes", ascending=False).itertuples():
        if r.iso3 not in anchors:
            continue
        x, y = anchors[r.iso3]
        share = float(r.share_pct) if str(r.share_pct).strip() else None
        bubbles.append({
            "a3": r.iso3, "name": r.country, "x": x, "y": y,
            "r": round(MAX_BUBBLE * math.sqrt(r.tonnes / vmax), 2),
            "value": _fmt_t(r.tonnes), "share": f"{share:.0f}% of world" if share is not None else "",
        })
        tint[r.iso3] = round(0.10 + 0.32 * math.sqrt(r.tonnes / vmax), 3)
    unit = rows.unit.iloc[0] if len(rows) else ""
    year = int(rows.year.iloc[0]) if len(rows) else None

    # ---- trade flows: the largest links, plus the removed supplier's own largest ones so the
    # disruption replay always has its arcs on the map
    e = edges[edges.exporter != edges.importer].sort_values("value_usd", ascending=False)
    chosen = e.head(top_n)
    own = e[e.exporter == removed_country].head(8)
    chosen = pd.concat([chosen, own]).drop_duplicates(subset=["exporter", "importer"])
    vmax_f = float(chosen.value_usd.max()) if len(chosen) else 1.0
    flows, unplaced = [], set()
    for r in chosen.itertuples():
        a, b = place(r.exporter), place(r.importer)
        if not a or not b:
            unplaced.update(n for n, p in ((r.exporter, a), (r.importer, b)) if not p)
            continue
        (fa3, (x1, y1)), (ta3, (x2, y2)) = a, b
        cx, cy = _arc(x1, y1, x2, y2, world["height"])
        flows.append({
            "from": fa3, "to": ta3, "fromName": r.exporter, "toName": r.importer,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy,
            "w": round(MIN_FLOW_W + (MAX_FLOW_W - MIN_FLOW_W) * math.sqrt(r.value_usd / vmax_f), 2),
            "usd": _fmt_usd(r.value_usd), "tonnes": _fmt_t(r.net_kg / 1000) if r.net_kg > 0 else "n/a",
            "removed": r.exporter == removed_country,
        })

    # ---- per-country tooltip facts, from the full edge list (not only the arcs drawn)
    info = {}
    by_exp = edges.groupby("exporter").value_usd.sum()
    by_imp = edges.groupby("importer").value_usd.sum()
    total_trade = float(edges.value_usd.sum()) or 1.0
    names_on_map = {f["fromName"] for f in flows} | {f["toName"] for f in flows} | {removed_country}
    for name in names_on_map:
        p = place(name)
        if not p:
            continue
        partners = (edges[edges.exporter == name].groupby("importer").value_usd.sum()
                    .sort_values(ascending=False).head(3))
        info[p[0]] = {
            "name": "Taiwan (reported as \"Other Asia, nes\")" if name == "Other Asia, nes" else name,
            "exports": f"{_fmt_usd(float(by_exp.get(name, 0.0)))} · {by_exp.get(name, 0.0) / total_trade:.1%} of world trade",
            "imports": _fmt_usd(float(by_imp.get(name, 0.0))),
            "buyers": [f"{k} {_fmt_usd(float(v))}" for k, v in partners.items()],
            "hub": name in hubs,
        }
    for b in bubbles:
        info.setdefault(b["a3"], {"name": b["name"]})
        info[b["a3"]]["mined"] = f"{b['value']}" + (f" · {b['share']}" if b["share"] else "")

    # ---- cascade replay, round by round
    history = cascade.get("history")
    rounds, offmap = [], 0
    if history is not None:
        for wave in history:
            placed_wave = []
            for name in wave:
                p = place(name)
                if p:
                    placed_wave.append({"a3": p[0], "name": name, "x": p[1][0], "y": p[1][1]})
                else:
                    offmap += 1
            rounds.append({"countries": placed_wave, "count": len(wave)})
    removed = place(removed_country)

    h = accent.lstrip("#")
    payload = {
        "map": {"width": world["width"], "height": world["height"], "graticule": world["graticule"],
                "countries": [{"codes": c["codes"], "d": c["d"]} for c in world["countries"]]},
        "style": {"accent": accent, "accentRgb": ",".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4)),
                  "critical": CRITICAL, "slate": SLATE},
        "tint": tint, "bubbles": bubbles, "flows": flows, "info": info,
        "removed": ({"a3": removed[0], "name": removed_country, "x": removed[1][0], "y": removed[1][1]}
                    if removed else None),
        "cascade": ({"rounds": rounds, "universe": universe, "offmap": offmap}
                    if history is not None else None),
        "text": {
            "measure": f"{measure}{f', {year}' if year else ''}", "unit": unit,
            "flows": f"Trade in {commodity} (HS {hs_code}), 2023, by value",
            "removedName": removed_country,
        },
        "a11y": (f"World map of {family.lower()}: circles sized by {measure.lower()}, arcs showing the "
                 f"{len(flows)} largest trade flows of {commodity} in 2023."),
    }
    notes = {
        "unplaced_trade": sorted(unplaced),
        "other_supply": [(r.country, _fmt_t(r.tonnes)) for r in other_rows.itertuples()],
        "flows_drawn": len(flows), "bubbles": len(bubbles), "unit": unit, "year": year,
        "cascade_offmap": offmap, "has_history": history is not None,
    }
    return payload, notes


@st.cache_resource(show_spinner=False)
def _component_source() -> tuple[str, str]:
    here = Path(__file__).resolve().parent
    return ((here / "supply_map.css").read_text(encoding="utf-8"),
            (here / "supply_map.js").read_text(encoding="utf-8"))


def supply_map(payload: dict, key: str):
    """Mount the animated map. Registered on every run, the pattern Streamlit's own docs use:
    the component registry is rebuilt per script run, so a definition cached across runs goes
    stale ("Component is not registered") from the second run on."""
    css, js = _component_source()
    component = st.components.v2.component("mfa_supply_map", html='<div class="sm-mount"></div>', css=css, js=js)
    return component(data=payload, key=key, width="stretch", height="content")
