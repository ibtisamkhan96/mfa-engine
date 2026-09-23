"""The dashboard's visual system: colours, chart styling, and the page components every page
shares, kept apart from app.py so the analysis code there reads as analysis.

Colour is assigned by job, never by taste, and every chart colour below was checked with the
dataviz palette validator (lightness band, chroma floor, separation under simulated protanopia
and deuteranopia, normal-vision separation, contrast) in BOTH light and dark mode, against the
two app backgrounds set in .streamlit/config.toml (#FAFAF7 light, #111316 dark). One hex per
entity serves both modes, so every colour sits in the lightness band the two modes share
(OKLCH L 0.48-0.67); Plotly trace colours cannot switch with the Streamlit theme, and
st.context.theme is documented as unreliable on first load, so a colour that needs a
mode-specific swap would be wrong half the time.

Validated results, worst pair in each set's fixed legend order (target >= 8 colourblind,
>= 15 normal vision):
  wind materials  colourblind 15.2, normal 20.3+, every colour >= 3:1 on both backgrounds
  EV materials    colourblind 16.2, normal 18.5+, every colour >= 3:1 on both backgrounds
  systems         wind / EV / data centres, three slots
  slate ramp      2-step ordinal ramp, passes the ordinal checks in both modes
"""
from __future__ import annotations

import html
import math

import pandas as pd
import streamlit as st

LIGHT_BG, DARK_BG = "#FAFAF7", "#111316"
FONT = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, 'Cascadia Mono', monospace"

# One colour per physical material, evocative where the validator allows it: copper's own
# colour, lithium's crimson flame test, cobalt blue, nickel-salt green, neodymium-glass violet,
# praseodymium's yellow-green salts, dysprosium amber, terbium's green-teal emission, iron-oxide
# red for cast iron, steel blue. Graphite, steel and concrete are grey in reality, and grey fails
# the chroma floor (a grey series stops doing identity work), so graphite is a slate violet and
# concrete a dusty rose: distinct first, evocative second.
MATERIAL_COLORS = {
    "copper": "#bb5c0c",
    "concrete": "#bf6d9d",
    "dysprosium": "#846305",
    "terbium": "#12a195",
    "neodymium": "#8254c4",
    "praseodymium": "#6d9d2d",
    "steel": "#4790d8",
    "cast iron": "#af3c3a",
    "nickel": "#3aa460",
    "cobalt": "#345fcf",
    "lithium": "#dd435d",
    "graphite": "#8085c8",
    "manganese": "#a23e7c",
}

# The fixed legend order the validator passed, per system. Traces are always drawn in this order,
# whatever subset is selected, so a colour follows its material and never its rank.
MATERIAL_ORDER = {
    "wind": ["copper", "concrete", "dysprosium", "terbium", "neodymium", "praseodymium", "steel", "cast iron"],
    "ev": ["nickel", "cobalt", "lithium", "graphite", "copper", "manganese"],
    "dc": ["copper"],
}

SYSTEM_COLORS = {"wind": "#1997cf", "ev": "#0a7e3a", "dc": "#9a66cf"}

# Ordinal slate ramp for "the same quantity under a harsher vs a milder assumption" (no
# substitution vs spare capacity, current vs reallocated). Neutral on purpose, so the material
# colour stays reserved for what this dashboard measures about that material.
SLATE = "#4E5F8F"
SLATE_LIGHT = "#8E9CC4"
NEUTRAL = "#8A8F98"          # reference lines and baselines
CRITICAL = "#d03b3b"         # status colour: the supplier removed in a disruption, always labelled
GRID = "rgba(128,128,128,0.16)"
AXIS = "rgba(128,128,128,0.45)"

# The material families the sidebar offers, each with the element tile it is drawn as. Rare
# earths are a four-element basket (Nd, Dy, Pr, Tb), shown as REE with the lanthanide range and
# neodymium's colour, the heaviest of the four by tonnage in the wind register.
FAMILIES = {
    "Copper": {"symbol": "Cu", "z": "29", "color": MATERIAL_COLORS["copper"]},
    "Lithium": {"symbol": "Li", "z": "3", "color": MATERIAL_COLORS["lithium"]},
    "Nickel": {"symbol": "Ni", "z": "28", "color": MATERIAL_COLORS["nickel"]},
    "Cobalt": {"symbol": "Co", "z": "27", "color": MATERIAL_COLORS["cobalt"]},
    "Graphite": {"symbol": "C", "z": "6", "color": MATERIAL_COLORS["graphite"]},
    "Rare earths": {"symbol": "REE", "z": "57-71", "color": MATERIAL_COLORS["neodymium"]},
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def usd(value: float) -> str:
    """Compact dollar figure for HTML and metric values (not markdown: see md_usd)."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f} bn"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.1f} M"
    return f"${value:,.0f}"


def md_usd(value: float) -> str:
    """usd() for markdown, where two literal dollar signs in one string render as LaTeX."""
    return usd(value).replace("$", "\\$")


# --------------------------------------------------------------------------- global CSS

_CSS = """
:root {
  --mat: __ACCENT__;
  --mat-rgb: __ACCENT_RGB__;
  --ease-out: cubic-bezier(.22, .8, .26, 1);
  --font: __FONT__;
  --mono: __MONO__;
  --hairline: rgba(128, 128, 128, .24);
}

/* The material colour runs along the top edge and eases to the new colour on every switch. */
header[data-testid="stHeader"] {
  box-shadow: inset 0 3px 0 var(--mat);
  transition: box-shadow .6s var(--ease-out);
}

/* Entrance: each top-level block rises in as it mounts, lightly staggered. A page switch mounts
   a fresh set of blocks, so every page arrives the same way; widget reruns do not remount
   existing blocks, so changing a control never replays it. */
@keyframes mfa-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > * {
  animation: mfa-rise .55s var(--ease-out) both;
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(2) { animation-delay: .04s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(3) { animation-delay: .08s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(4) { animation-delay: .12s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(5) { animation-delay: .16s; }
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > *:nth-child(n+6) { animation-delay: .2s; }

[data-testid="stMainBlockContainer"] { padding-top: 4.2rem; padding-bottom: 4rem; max-width: 1240px; }

/* ---------- page header */
.page-head { margin: 0 0 .4rem; }
.page-head .eyebrow {
  display: flex; flex-wrap: wrap; gap: .5rem; align-items: center;
  font: 600 .72rem/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase;
}
.page-head .eyebrow .where { opacity: .72; }
.chip {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .28rem .6rem; border-radius: 999px;
  background: rgba(var(--mat-rgb), .13); border: 1px solid rgba(var(--mat-rgb), .5);
  font: 500 .78rem/1.2 var(--font); letter-spacing: 0; text-transform: none;
  transition: background-color .5s var(--ease-out), border-color .5s var(--ease-out);
}
.chip .dot { width: .55rem; height: .55rem; border-radius: 50%; background: var(--mat); flex: none; }
.page-head h1 {
  font-family: var(--font); font-weight: 700; letter-spacing: -.02em; line-height: 1.12;
  font-size: clamp(1.65rem, 1.15rem + 1.7vw, 2.45rem); margin: .6rem 0 .45rem; padding: 0;
}
.page-head .lede { font-size: 1.04rem; line-height: 1.6; opacity: .8; max-width: 74ch; margin: 0; }

/* ---------- one-line takeaway: the sentence a reader should leave a chart with */
.takeaway {
  border-left: 3px solid var(--mat); background: rgba(var(--mat-rgb), .08);
  border-radius: 0 10px 10px 0; padding: .7rem 1rem; margin: .1rem 0 .2rem;
  font-size: 1.02rem; line-height: 1.55; max-width: 92ch;
  transition: border-color .5s var(--ease-out), background-color .5s var(--ease-out);
}
.takeaway b { font-weight: 600; }
.muted { opacity: .72; }

/* ---------- metric cards */
[data-testid="stMetric"] {
  position: relative; overflow: hidden;
  transition: transform .22s var(--ease-out), box-shadow .22s var(--ease-out), border-color .22s var(--ease-out);
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: var(--mat); opacity: .85; transition: background-color .5s var(--ease-out);
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 10px 28px -16px rgba(0, 0, 0, .45); }
[data-testid="stMetricValue"] { font-size: clamp(1.25rem, 1.05rem + .9vw, 1.9rem); font-weight: 600; letter-spacing: -.01em; }
[data-testid="stMetricLabel"] { opacity: .85; }

/* ---------- overview step cards lift on hover and pick up the material colour */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stHtml"] > .step-card) {
  transition: transform .22s var(--ease-out), box-shadow .22s var(--ease-out), border-color .22s var(--ease-out);
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stHtml"] > .step-card):hover {
  transform: translateY(-3px); border-color: rgba(var(--mat-rgb), .6);
  box-shadow: 0 14px 30px -18px rgba(0, 0, 0, .5);
}
.step-card .num { font: 600 .74rem/1 var(--mono); letter-spacing: .08em; color: var(--mat); }
.step-card .title { font-weight: 600; font-size: 1.05rem; margin: .45rem 0 .25rem; }
.step-card .fig { font-size: 1.35rem; font-weight: 600; letter-spacing: -.01em; margin: .15rem 0 .1rem; }
.step-card .desc { font-size: .9rem; line-height: 1.5; opacity: .75; }

/* ---------- expanders: the "how this is calculated" layer */
[data-testid="stExpander"] details { border-radius: 10px; transition: border-color .2s var(--ease-out); }
[data-testid="stExpander"] details:hover { border-color: rgba(var(--mat-rgb), .5); }
[data-testid="stExpander"] summary { font-weight: 500; }

/* ---------- the supply map's space is held only until its script has drawn it (the component
   renders nothing before then), so the page below does not jump down half a screen when it
   appears. The script then sets data-sm-ready and the element takes the map's natural height:
   a fixed ratio kept after that let a wrapped legend spill over the next element. */
.st-key-supply_map:not([data-sm-ready]) { aspect-ratio: 1000 / 540; }

/* ---------- the "How to read this page" button stays on one line at any width */
[class*="st-key-guide_"] button, [class*="st-key-guide_"] button p { white-space: nowrap; }

/* ---------- glossary */
.gl-count { font: 600 .72rem/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; opacity: .65; margin: .2rem 0 .5rem; }
.gl-index { display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 .6rem; }
.gl-index a {
  font-size: .82rem; line-height: 1.2; padding: .28rem .6rem; border-radius: 999px; text-decoration: none;
  color: inherit; border: 1px solid var(--hairline);
  transition: background-color .15s ease, border-color .15s ease;
}
.gl-index a:hover, .gl-index a:focus-visible { border-color: var(--mat); background: rgba(var(--mat-rgb), .1); }
.gl-card { scroll-margin-top: 5rem; }
.gl-card:target .gl-term { color: var(--mat); }
.gl-term { font-weight: 600; font-size: 1.05rem; display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem; transition: color .4s ease; }
.gl-aka { font: 500 .78rem/1.2 var(--mono); opacity: .65; }
.gl-plain { margin: .35rem 0 0; line-height: 1.6; }
.gl-sym { margin: 0; font-size: .86rem; line-height: 1.55; opacity: .75; }
.gl-see { font: 600 .68rem/1 var(--mono); letter-spacing: .07em; text-transform: uppercase; opacity: .6; }

/* ---------- charts sit on the page, not in a box */
[data-testid="stPlotlyChart"] { border-radius: 10px; }

/* ---------- sidebar */
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {
  border-radius: 8px; transition: background-color .15s ease, transform .15s var(--ease-out), box-shadow .3s var(--ease-out);
}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover { transform: translateX(2px); }
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {
  box-shadow: inset 3px 0 0 var(--mat); background: rgba(var(--mat-rgb), .14);
}
/* Streamlit gives page links a -6px margin that assumes the default gap between blocks; the nav
   container has no gap, so the margins are reset here or neighbouring rows would overlap. */
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(> [data-testid="stPageLink"]) { margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {
  border-radius: 8px; margin: 0;
  transition: background-color .15s ease, transform .15s var(--ease-out), box-shadow .3s var(--ease-out);
}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover { transform: translateX(2px); }
.nav-label { margin: .7rem 0 .2rem .55rem; }
/* The selected material (and, for copper, the selected system) wears the material colour. Kept to
   the sidebar: pills in the page body select materials that each have their own colour. */
[data-testid="stSidebar"] [data-testid="stBaseButton-pillsActive"],
[data-testid="stSidebar"] [data-testid="stBaseButton-segmented_controlActive"] {
  background: rgba(var(--mat-rgb), .16) !important; border-color: var(--mat) !important;
  color: inherit !important; transition: background-color .4s var(--ease-out), border-color .4s var(--ease-out);
}
[data-testid="stNavSectionHeader"] {
  font: 600 .7rem/1.2 var(--mono) !important; letter-spacing: .09em; text-transform: uppercase; opacity: .62;
}
.sb-label {
  font: 600 .7rem/1.2 var(--mono); letter-spacing: .09em; text-transform: uppercase;
  opacity: .62; margin: .9rem 0 .1rem;
}
.mat-tile { display: flex; align-items: center; gap: .8rem; padding: .15rem 0 .1rem; }
.mat-tile .el {
  width: 3.6rem; height: 3.6rem; flex: none; border-radius: 12px; color: #fff; position: relative;
  background: linear-gradient(315deg, var(--c), color-mix(in oklab, var(--c) 70%, #000));
  box-shadow: 0 6px 18px -8px var(--c), inset 0 1px 0 rgba(255, 255, 255, .22);
  display: flex; align-items: center; justify-content: center;
  transition: background .5s var(--ease-out), box-shadow .5s var(--ease-out);
  animation: tile-in .45s var(--ease-out) both;
}
@keyframes tile-in { from { transform: scale(.92) rotate(-4deg); opacity: .4; } to { transform: none; opacity: 1; } }
.mat-tile .el .sym { font: 700 1.45rem/1 var(--font); letter-spacing: -.02em; }
.mat-tile .el .sym.long { font-size: 1.05rem; }
.mat-tile .el .z { position: absolute; top: .3rem; left: .4rem; font: 500 .58rem/1 var(--mono); opacity: .9; }
.mat-tile .name { font-weight: 600; font-size: 1.02rem; line-height: 1.25; }
.mat-tile .sys { font-size: .84rem; opacity: .75; line-height: 1.35; }
.sb-foot { font-size: .78rem; line-height: 1.5; opacity: .7; }

/* ---------- small screens */
@media (max-width: 640px) {
  .page-head .lede { font-size: .98rem; }
  .takeaway { font-size: .97rem; }
}

/* ---------- motion is decoration here, never information: honour the OS setting */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .001ms !important; animation-delay: 0s !important;
    animation-iteration-count: 1 !important; transition-duration: .001ms !important;
  }
}
"""


def inject_css(accent: str) -> None:
    h = accent.lstrip("#")
    accent_rgb = ", ".join(str(int(h[i:i + 2], 16)) for i in (0, 2, 4))
    css = (_CSS.replace("__ACCENT_RGB__", accent_rgb).replace("__ACCENT__", accent)
           .replace("__FONT__", FONT).replace("__MONO__", MONO))
    st.html(f"<style>{css}</style>")


# --------------------------------------------------------------------------- charts

def style_fig(fig, *, height: int = 400, hovermode: str | bool = "x unified",
              legend: bool = True, margin: dict | None = None):
    """Shared chart chrome. Backgrounds are transparent and font colour is left to Streamlit's
    own Plotly theme, so the same figure reads correctly in light and dark mode; grid and axis
    lines are neutral greys at low opacity, recessive on either background."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13),
        margin=margin or dict(t=44 if legend else 16, l=8, r=12, b=8),
        hovermode=hovermode,
        hoverlabel=dict(font=dict(family=FONT, size=13)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", title_text="", font=dict(size=12.5)),
        # Eased re-draws: when a control changes, bars and lines move to their new values
        # instead of snapping (Plotly animates only between compatible traces, and skips it
        # for Sankey diagrams).
        transition=dict(duration=380, easing="cubic-in-out"),
    )
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor=AXIS,
                     ticks="outside", ticklen=4, tickcolor=AXIS, zeroline=False,
                     title_font=dict(size=12.5))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     title_font=dict(size=12.5))
    return fig


def show_chart(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit", key=key, config=PLOTLY_CONFIG)


def end_labels(fig, series: dict[str, pd.Series], *, log: bool = False) -> None:
    """Direct labels at each line's last point, the secondary encoding the validator asks for
    once a subset of materials puts two unvalidated neighbours side by side. Text stays in ink,
    never in the series colour; only the small marker carries the colour."""
    for name, s in series.items():
        s = s.dropna()
        if s.empty:
            continue
        x, y = s.index[-1], float(s.iloc[-1])
        if log and y <= 0:
            continue
        fig.add_annotation(x=x, y=math.log10(y) if log else y, text=esc(name),
                           showarrow=False, xanchor="left", xshift=6, font=dict(size=12))


def download_csv(df: pd.DataFrame, filename: str, key: str, label: str = "Download data (CSV)") -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, type="tertiary", icon=":material/download:",
                       on_click="ignore")


# --------------------------------------------------------------------------- page components

def page_header(where: str, title: str, lede: str, chip: str) -> None:
    """lede is HTML (may contain <b>); anything interpolated into it must be escaped by the caller."""
    st.html(
        f'<div class="page-head"><div class="eyebrow"><span class="where">{esc(where)}</span>'
        f'<span class="chip"><span class="dot"></span>{esc(chip)}</span></div>'
        f"<h1>{esc(title)}</h1><p class=\"lede\">{lede}</p></div>"
    )


def takeaway(html_text: str) -> None:
    """One sentence, computed from the data on screen. html_text may contain <b> tags; every
    interpolated value must already be escaped by the caller (numbers are safe as they are)."""
    st.html(f'<div class="takeaway">{html_text}</div>')


def sidebar_label(text: str) -> None:
    st.sidebar.html(f'<div class="sb-label">{esc(text)}</div>')


def material_tile(family: str, name: str, found_in: str) -> str:
    meta = FAMILIES[family]
    long = " long" if len(meta["symbol"]) > 2 else ""
    return (
        f'<div class="mat-tile" style="--c:{meta["color"]}">'
        f'<div class="el" aria-hidden="true"><span class="z">{esc(meta["z"])}</span>'
        f'<span class="sym{long}">{esc(meta["symbol"])}</span></div>'
        f'<div><div class="name">{esc(name)}</div><div class="sys">{esc(found_in)}</div></div></div>'
    )
