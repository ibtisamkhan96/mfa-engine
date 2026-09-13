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

**Two real, independent case studies**, run through that same unmodified
engine:

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

Running two genuinely different systems through the same base class,
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
system's real projected material recovery against
[crm-trade-network](../crm-trade-network)'s real 2023 UN Comtrade
concentration (HHI) and single-supplier cascade-shortfall analysis. Covers
every material either physical system actually contains that also has a
real, trackable UN Comtrade code: copper and rare earth metals (wind
turbines), copper, lithium, nickel, cobalt and graphite (EV batteries).
Copper is tracked from both systems on purpose, the one material both real
case studies actually contain, so the connector can show genuine
cross-technology material competition for the same real recovered tonne,
not just a single system's own numbers in isolation.

**One dashboard** (`app.py`, Streamlit): both systems, all seven
material/system combinations, a Kaplan-Meier/Weibull survival chart, a real
confidence band on secondary material output (propagating the real
low/high ranges `material_intensity()` reports rather than collapsing to a
point estimate), Sankey diagrams for both the physical flow and the trade
network, a year-by-year (not just averaged) recovery-vs-shortfall chart
with the real peak retirement year surfaced alongside the average, real
(low, central, high) price ranges (external citations where the traded
product genuinely matches a real market benchmark, disclosed context
instead of a forced range where it doesn't), and the risk connector
itself, with a live "fetch real data now" control against UN Comtrade.
Every chart carries its own caption with the actual formula it plots and
how that number is measured, and a "how to read this dashboard" section
walks through the intended reading order.

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
```

## What this honestly is not

Not a finished substitute for either field. No scenario analysis, no
life-cycle assessment, no geospatial layer yet (though the wind turbine
register carries real coordinates and municipality data that would support
one), and the risk side is a single real 2023 snapshot rather than a
multi-year trend (crm-trade-network's own fetcher supports other years, but
a full second year is hundreds of rate-limited live UN Comtrade calls, real
hours of wall-clock time, not something to trigger casually).

Denmark's wind fleet is modeled nationally, the global EV fleet globally,
a deliberate consequence of which real data actually exists for each
technology today (a national register with real per-unit retirement data
for wind; only global-level sales and chemistry-mix reporting for EVs), not
an accident, see the dashboard's own "What this app actually is" expander
for the full reasoning.

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
