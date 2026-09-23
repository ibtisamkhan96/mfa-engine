# mfa-engine

A reusable dynamic material flow analysis (dMFA) engine, connected to a real
supply-chain risk dataset. Built to answer one question the existing
literature on modular MFA tools (ODYM, flodym, MISO2, GCMat) does not:
does a real physical recovery pathway matter at the scale of a real,
measured supply-disruption risk, across more than one technology system.

## What's actually in here

**One reusable engine** (`mfa_engine/cohort_survival.py`): a base class,
`CohortSurvivalMFA`, implementing right-censored survival analysis
(Kaplan-Meier), Weibull extrapolation, cohort-by-cohort conditional-survival
retirement projection, and secondary material accounting, once. A subclass
provides exactly one thing: `material_intensity()`, real sourced material
content by category. Everything else is inherited unchanged. This mirrors
ODYM/flodym's own actual design (a shared base class, object-based
subsystems), not just the name "modular."

**All three textbook dMFA modes, sharing one material-accounting layer**
(`mfa_engine/accounting.py`): register-based (Denmark's turbines, assets known to be
standing), inflow-driven (EV sales, each cohort decaying from its own sale year), and
**stock-driven** (`mfa_engine/stock_driven.py`: data centres, where the required stock is the
input and inflow is derived as growth plus replacement of worn-out equipment). Unit-to-tonne
conversion lives in one place, so the three model families can never drift into different
material arithmetic.

**Three real, independent case studies**:

- `mfa_engine/systems/wind_turbine.py`: Denmark's real wind turbine fleet.
  Verified to reproduce [dk-wind-mfa](../dk-wind-mfa)'s own published numbers
  exactly (`tests/test_reproduces_dk_wind_mfa.py`, 9/9 checks).
- `mfa_engine/systems/ev_battery.py`: the global EV battery fleet. The real
  fleet is too young to have generated enough retirement events to fit a
  survival model from its own data (Argonne National Laboratory's own
  finding: ~3% of global PEV battery capacity scrapped as of Dec 2020), so
  this case sources its Weibull parameters from a real, externally published
  vehicle survivability table (NHTSA, DOT HS 809 952), the same real source
  Argonne's own EV assessment uses for this exact purpose. Every other
  number (annual sales, chemistry mix, material content per chemistry) is
  either a real, cited figure or a disclosed derivation from real, cited
  figures, see the module's own docstring for full citations.
- `mfa_engine/systems/data_centre.py`: global data-centre infrastructure, the SDU
  advert's second named case study. Stock-driven, because what exists for data centres is
  installed capacity, not a register or a build table. Capacity: McKinsey (Oct 2024), 60 GW
  in 2023 and 171-219 GW by 2030; back-cast with IEA's ~12%/yr growth since 2017 and
  Masanet et al. (2020, *Science*)'s finding that data-centre energy was flat 2010-2018;
  IEA's 2030-2035 base-case growth after that. Two layers with their own lifetimes (ASHRAE:
  transformers 30 years, facility power and cooling 17-25, 20 used) and real copper content
  (WEF/Kearney on Microsoft's 80 MW Chicago site: 12 t/MW facility, 26 t/MW with grid
  connection). Servers are deliberately not modelled: no consistent public per-MW figure
  exists, and WEF/Kearney report the minerals sit mainly in power and cooling. Real result:
  about 0.66-1.0 Mt of new copper a year by 2030, landing between Macquarie's (330-420 kt) and
  Amoah et al. (2026)'s estimates without either being an input.

**One material across all three technologies.** The dashboard puts copper from wind, EVs
and data centres on the same axes, each on its own forward projection, the "competition for
materials across low-carbon and digital technologies" question the SDU advert asks. Real
result: by 2030, new data centres would need about 0.66 Mt of copper a year, roughly 30% of
what all new EVs worldwide need (2.2 Mt under IEA STEPS).

Running three genuinely different systems through the same accounting,
without modifying it beyond one additive method
(`fit_weibull_to_curve`, for populations with a published curve but no
individual censored data of their own), is the actual test of "modular and
reusable," not just an assertion.

**One optimization layer** (`mfa_engine/diversification.py`): a real linear
program (`scipy.optimize.linprog`) answering a different question than the
rest of the connector, not how concentrated a material's supply is, but the
minimum real disruption (least 2023 trade reallocated) needed to bring it
under a target, using the same real spare-capacity (`slack`) assumption
already used elsewhere as the ceiling on how much any real supplier could
realistically absorb. The default target (65%) is not an invented round
number: it is the EU Critical Raw Materials Act's own real, adopted
benchmark (Regulation (EU) 2024/1252, Article 5(1)(b)). A real, useful
degenerate case falls out of the formulation for free: at zero slack, no
reallocation is ever mathematically possible, the model's own way of
saying diversification requires real spare capacity to diversify into.

**The `slack` value itself is real and material-specific where that exists,
disclosed and honest where it doesn't.** A flat 20% "spare capacity" figure
used to be applied to every material, an assumption crm-trade-network's own
README already flagged as buried and uncited. Real research changed this:
copper alone has a citable, published industry-association figure, ICSG's
2023 real global mine capacity utilization of 77.6%, implying 22.4% real
spare capacity (republished via Statista, since ICSG's own primary
publications are subscription-only). No comparable figure exists for
rare earth metals, lithium, nickel, cobalt or graphite, for real, specific,
structural reasons documented per material in the dashboard's own "What
does slack mean" expander, not just "no data found": ~74% of 2023 cobalt
mine output is a copper byproduct and ~7% a nickel byproduct (USGS Mineral
Commodity Summaries 2024), and byproduct metals are never mined against
their own dedicated capacity; China manages rare earth supply through
administrative mining/separation quotas, so USGS's own reported production
figure is the quota by construction, no utilization ratio can be derived
from it; lithium, nickel and graphite have no ICSG-equivalent body
publishing a capacity-utilization series at all (USGS/INSG publish
production, reserves and market balance, not capacity). Those five keep
the same disclosed 20% illustrative assumption, now labeled as exactly
that rather than presented as though it meant the same thing for every
material.

**One grounded AI feature** (`mfa_engine/ask.py`): a "ask this dashboard a
question" box, not a general chatbot bolted on. Claude is given only the
real numbers already computed for the currently selected material and
instructed to answer from those alone, and to say so plainly if a
question asks about something they don't cover, the same "disclose the
real gap rather than invent an answer" discipline used everywhere else in
this project, applied to a conversational interface. Bring-your-own-key
(the same real pattern already live on the
[battery-electrode-screening-agent](../battery-electrode-screening-agent)
demo), so no key is stored and no visitor's usage bills another's account.

**One connector** (`mfa_engine/supply_risk_context.py`): compares either
system's real projected material recovery (end-of-life outflow times the cited
recycling rate, see below, not the whole outflow) against
[crm-trade-network](../crm-trade-network)'s real 2023 UN Comtrade
concentration (HHI) and single-supplier cascade-shortfall analysis. Covers
every material either physical system actually contains that also has a
real, trackable UN Comtrade code: copper and rare earth metals (wind
turbines), copper, lithium, nickel, cobalt and graphite (EV batteries).
Copper is tracked from both systems on purpose, the one material both real
case studies actually contain, so the connector can show genuine
cross-technology material competition for the same real recovered tonne,
not just a single system's own numbers in isolation.

**That connector now also runs crm-trade-network's cascading-failure model, not just
its static concentration numbers.** The dashboard's Cascade & trade network page runs
[crm-trade-network](../crm-trade-network)'s `shock_propagation` module (built there to
replicate, at single-trade-layer resolution, the linear-threshold cascade Ouyang et
al. run for cobalt: Ouyang, Liu, Liu, Chen, Wang, Pang, He, Liu, *Environ. Sci.
Ecotechnol.* 29, 2026, 100654) against the same 2023 trade network already loaded for
the static HHI/shortfall figures, surfacing how many *other* countries a disruption
itself takes down as their own trade partners collapse in turn, not only how much trade
value the removed country directly supplied. The Cascade page also re-runs the cascade across failure
thresholds from 5% to 95%, because the published result is where the network switches from
collapse to resilience, which one fixed threshold cannot show: losing Chile brings down the
whole refined-copper network up to a 25% threshold, and under 5% of it from 55% upward.

**A full stock-flow account, not only an outflow projection**
(`CohortSurvivalMFA.project_stock_flows`): inflow, in-use stock and end-of-life outflow
per year and per material, the account a dynamic MFA is defined by, closing to
floating-point noise every year (stock(t) - stock(t-1) = inflow(t) - outflow(t), checked in
`tests/test_stock_flows.py`, worst residual about 2e-9 t). The engine distinguishes the two
textbook dMFA modes explicitly. Denmark's wind turbines come from a register of assets
known to be standing, so they are stock-driven, with conditional survival. The EV fleet
comes from sales figures, so it is **inflow-driven**: each cohort decays from its own sale
year, and the stock at the snapshot is sales times survival, not total sales. The wind
case still reproduces dk-wind-mfa's published numbers bit for bit after this refactor.

**End-of-life recovery with real, cited rates** (`mfa_engine/recovery.py`). Material
leaving service is not material recovered. The functional end-of-life recycling rate
(EoL-RR) from the UN International Resource Panel (UNEP 2011, read directly from its
page-26 table) is applied per material: over 50% for copper, nickel, cobalt and manganese,
70-90% for iron and steel, **under 1% for lithium and every rare earth element**. For EV
batteries a second regime applies the EU Batteries Regulation's 2031 recovery targets
(Regulation (EU) 2023/1542, Annex XII Part C: lithium 80%, cobalt and nickel 95%),
assuming full collection, an upper bound. Real result: under current practice, Denmark's
wind fleet returns about 1 t of its roughly 137 t of rare earth outflow to 2050; EV lithium
recovery rises from about 3,100 t to about 493,000 t between current practice and the
Regulation's targets, a 160-fold policy lever. Graphite has no cited rate in either
source, so the dashboard reports it as not assessed rather than inventing one.

**Two real deployment scenarios** (`mfa_engine/scenarios.py`): the same unmodified
`CohortSurvivalMFA` engine, run under two different real, named future EV sales
trajectories, closing the "technology deployment scenarios" gap the SDU PhD advert
names directly rather than only asserting the design could support one. IEA's own
Stated Policies Scenario (STEPS, EVs reaching roughly 50% of global car sales by 2035,
held at that real endpoint afterwards rather than extrapolated past it) against a
flat-2026 floor (a comparison baseline, not a forecast). Real result, in stock-flow
terms: by 2050 STEPS puts about 53% more lithium into in-use stock than the flat floor,
but only about 22% more into end-of-life outflow over 2025-2050, since outflow lags stock
by roughly one battery lifetime (median about 13 years). A deployment choice made now
barely shows up in recycling until the late 2030s.

**Four silent errors found and fixed while building this**, each now covered by a
test: (1) the dashboard pinned the EV snapshot to 2024-12-31, so the real 2025 and 2026
sales, about 43 million vehicles, were dropped by the engine's own "already in service"
filter and never reached the dashboard at all; (2) the STEPS trajectory kept extending its
2026-2035 slope past 2035, silently reaching an 87% EV share by 2050, a figure no source
gives; (3) the EV sales table was treated as if every vehicle ever sold were still on the
road, overstating the stock and pushing retirements that already happened into the future. (4) recovered lithium, counted as lithium metal content, was valued at the lithium
carbonate price, understating every lithium dollar figure 5.32-fold (one tonne of lithium is
73.89 / 13.88 = 5.32 tonnes of lithium carbonate); lithium is now valued per tonne of lithium
at 5.32 times the carbonate price, with the conversion stated where the price is set
(`tests/test_lithium_valuation.py`).

**One dashboard** (`app.py`, Streamlit), fourteen pages. Overview first,
then *Physical flows* (Lifetimes, Stocks & flows, End-of-life materials, Scenarios), *Supply
risk* (Supply map, Disruption & recovery, Cascade & trade network, Diversification), *Compare* (Copper
across technologies, All materials) and *Reference* (Glossary, Ask the data, Methods & data).
The Glossary explains all 45 technical terms in plain words, with the exact formula the code
computes for each and links to where it appears (`ui/glossary.py`). The
sidebar holds what every page shares, in the order you need it: the material (and, for copper,
which of the three systems it sits in), then the page list, then the horizon, recovery
assumption, disrupted supplier and price. Material, system and horizon are kept in the URL, so
a particular view can be sent as a link. Back and next links at the foot of each page follow
the same order, and a "How to read this page" button at the top of every page opens a short
guide: what the page shows, how to read its charts, the method in brief, and one thing to try
(`ui/guides.py`).

Every chart opens with one sentence computed from the numbers on screen, for example "In an
average year to 2050, recycled copper from global EV batteries would cover 10.80% of one year's
supply shortfall if Chile stopped exporting". The formula, the data it is measured from and
the assumptions sit in a "How this is calculated" panel underneath, and the chart's data
downloads as CSV. The charts: Kaplan-Meier and Weibull survival curves; a stock-and-flow chart
with the stock on top and the yearly flows below it (outflow drawn under the zero line,
recovered tonnes overlaid on it); end-of-life material with the real low/high content range
as a band; Sankey diagrams for the physical flow and the trade network; recovery against the
shortfall year by year, with the peak year next to the average; and real (low, central, high)
price ranges where the traded product matches a market benchmark. A "fetch real data now"
control re-asks UN Comtrade live.

**An animated supply map** (`ui/supply_map.py`, `ui/supply_map.js`). A world map with three
layers, all from data the rest of the dashboard already uses. Circles and shading show where the
material is mined, or where its reserves sit (USGS Mineral Commodity Summaries 2025, 2024 data,
via the Materials Data Series: `data/mine_supply.csv`, built by `scripts/build_mine_supply.py`).
Arcs show the largest 2023 UN Comtrade trade links, with dots moving from exporter to importer.
"Play the disruption" replays crm-trade-network's cascade round by round: the supplier set in
the sidebar stops exporting, then every country that loses over 20% of its trade in that
commodity fails in turn, and the map turns them red as they go. Hover a country for its numbers;
click it to isolate its trade. The geometry is Natural Earth, projected once in Python with the
equal-area Equal Earth projection (`data/world_map.json`, built by `scripts/build_world_map.py`),
so the browser only animates: no mapping library or CDN is loaded at runtime. It is a Streamlit
v2 custom component written in plain SVG and the Web Animations API, follows light and dark mode
through Streamlit's theme variables, pauses when off screen, and draws a still map under
`prefers-reduced-motion`. Production (2024) and trade (2023) are a year apart and are shown side
by side, never combined in a calculation; the arcs are one stage of the chain, the traded form
in each HS code.

**Colours, themes and layout** (`ui/theme.py`, `.streamlit/config.toml`). Each
material has its own colour, used for its sidebar tile, the accent line at the top of the
page and its charts: copper's own orange-brown, lithium's crimson flame-test red, cobalt
blue, nickel-salt green, violet for neodymium and the rare earths. Graphite, steel and concrete
are grey in reality, and grey fails the validator's chroma floor, so graphite is a slate
violet and concrete a dusty rose. Every chart palette went through a colour-vision validator
(separation under simulated protanopia and deuteranopia, normal-vision separation, contrast
against the background) in both light and dark mode; the worst neighbouring pair scores 15.2
for the wind materials and 16.2 for the EV materials against a target of 8. Each colour is a
single hex that passes on both backgrounds, because Plotly trace colours do not switch with
the Streamlit theme. Light and dark themes are both defined and follow the viewer's OS
setting. No chart uses two y-axes. Motion is limited to short entrance and colour
transitions, and turns off under `prefers-reduced-motion`. The layout was checked in a real
browser at 1440 px wide in both themes and at phone width (390 px), with no horizontal
scrolling on any page.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `dk-wind-mfa` and `crm-trade-network` as sibling directories (the
app checks for them on startup and says so plainly if they're missing).

## Running it as a container

```bash
docker build -t mfa-engine .
docker run -p 8501:8501 mfa-engine
```

The `Dockerfile` clones both real sibling repos from their own public GitHub
URLs at build time (a container has no "sibling folder", so `app.py` reads
`DK_WIND_MFA_SRC`/`CRM_TRADE_NETWORK_SRC` from the environment when set,
falling back to the local sibling-directory convention otherwise). Verified
by an actual build and run, not just written and assumed: the container
serves the full dashboard, including the live UN Comtrade-backed risk
connector, identically to local dev. One real thing this caught: an
unpinned `pandas>=2.0` resolves to the newest 3.x release in a fresh image,
which breaks a `pd.NA`-into-a-bool-column assignment in dk-wind-mfa's own
`load.py` that works fine under the pandas 2.x this whole project has
actually been tested against, so the Dockerfile pins `pandas<3.0`
explicitly rather than silently taking on an untested major version.

Deploys to Railway straight from this `Dockerfile`: create a new Railway
project from this GitHub repo, Railway auto-detects the `Dockerfile` and
supplies `$PORT` at runtime (the `CMD` already reads it), no extra
configuration needed. Connecting the GitHub repo means a future push here
redeploys automatically.

## Running it on Streamlit Community Cloud

Streamlit Community Cloud clones only this one repo and never runs the
`Dockerfile`, so the two sibling dependencies can't be provided the same
way. `app.py` handles this itself: if `DK_WIND_MFA_SRC`/
`CRM_TRADE_NETWORK_SRC` aren't set and no local sibling checkout exists, it
shallow-clones both real, public repos straight from GitHub into a cache
directory at import time, before falling through to the same
`check_dependencies()` used everywhere else. `requirements.txt` includes
`xlrd` for exactly this path (dk-wind-mfa's own real dependency for reading
its register's legacy `.xls` file, otherwise only installed via its own
`requirements.txt`, which Streamlit Cloud never looks at).

Verified with an actual simulation of Streamlit Cloud's own constraints,
not just written and assumed: a bare `python:3.11-slim` container, only
this repo mounted (no sibling folders visible, no Dockerfile run), a plain
`pip install -r requirements.txt`, then `streamlit run app.py`. It cloned
both sibling repos on the fly and produced identical numbers to local dev
and the Docker path.

To deploy: share.streamlit.io &rarr; New app &rarr; point it at this repo
and `app.py`. No secrets or extra configuration needed.

## Verification

```bash
python tests/test_reproduces_dk_wind_mfa.py   # exact match against dk-wind-mfa's published numbers
python tests/test_ev_battery_mfa.py           # sanity checks against the real NHTSA/CRS/GREET sources
python tests/test_diversification.py          # LP respects its own constraints against real trade data
python tests/test_scenarios.py                # the two deployment scenarios genuinely diverge, and by how much
python tests/test_stock_flows.py              # mass balance closes every year; recovered never exceeds outflow
python tests/test_data_centre.py              # stock-driven engine: tracks the requirement, steady-state spin-up, no negative building
python tests/test_dashboard_pages.py          # every page for every material renders cleanly; every page's reading guide opens
python tests/test_lithium_valuation.py        # recovered lithium is valued as lithium carbonate equivalent (x5.32)
```

## What this honestly is not

Not a finished substitute for either field. No life-cycle assessment, no
geospatial analysis yet (the Supply map places country-level data on a world map, but the wind
turbine register's own coordinates and municipality data are not used), no formal
uncertainty quantification yet (Monte Carlo over lifetimes, material
intensities and recycling rates, all of which already carry real low/high
ranges; today those ranges are shown as bounds, not propagated jointly), no
product-specific recycling rates (UNEP's EoL-RR figures are element averages
across all products, so battery-embedded manganese in particular is likely
over-credited, flagged in `recovery.py`), and the risk side is a single real 2023 snapshot
rather than a multi-year trend (crm-trade-network's own fetcher supports
other years, but a full second year is hundreds of rate-limited live UN
Comtrade calls, real hours of wall-clock time, not something to trigger
casually). Scenario analysis exists for EV sales (STEPS vs a flat floor) and data-centre
capacity (McKinsey's low and high 2030 cases); the wind turbine system has no equivalent
"future installations" scenario yet, since it currently only projects the retirement of
turbines already in the real Danish register. The data-centre system is copper-only and
leaves the server layer out (no consistent public per-MW material figure was found), and it
assumes capacity tracks electricity use wherever it borrows IEA growth rates; both are stated
in its module docstring rather than filled with guessed figures.

Denmark's wind fleet is modeled nationally, the global EV fleet globally,
a deliberate consequence of which real data actually exists for each
technology today (a national register with real per-unit retirement data
for wind; only global-level sales and chemistry-mix reporting for EVs), not
an accident, see the dashboard's Methods & data page for the full reasoning.

Price is given a real (low, central, high) range for copper, lithium and
nickel, where the traded product genuinely matches a real, dated market
benchmark. For cobalt, graphite and rare earth metals, the best real
benchmarks found either price a different-grade product than what the UN
Comtrade code actually blends, or (rare earths) would need weighting four
elements' very different real prices by their own physical tonnage share to
combine into one basket figure. Rather than force a range that wouldn't
honestly bracket the real data-derived central estimate, those three show
the real benchmark as disclosed context instead, a real, known gap, not a
silently dropped one.
