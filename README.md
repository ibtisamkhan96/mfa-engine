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

**One connector** (`mfa_engine/supply_risk_context.py`): compares either
system's real projected material recovery against
[crm-trade-network](../crm-trade-network)'s real 2023 UN Comtrade
concentration (HHI) and single-supplier cascade-shortfall analysis. Covers
every material either physical system actually contains that also has a
real, trackable UN Comtrade code: copper and rare earth metals (wind
turbines), lithium, nickel, cobalt and graphite (EV batteries).

**One dashboard** (`app.py`, Streamlit): both systems, all six materials,
a Kaplan-Meier/Weibull survival chart, a real confidence band on secondary
material output (propagating the real low/high ranges `material_intensity()`
reports rather than collapsing to a point estimate), Sankey diagrams for
both the physical flow and the trade network, and the risk connector itself,
with a live "fetch real data now" control against UN Comtrade.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires `dk-wind-mfa` and `crm-trade-network` as sibling directories (the
app checks for them on startup and says so plainly if they're missing).

## Verification

```bash
python tests/test_reproduces_dk_wind_mfa.py   # exact match against dk-wind-mfa's published numbers
python tests/test_ev_battery_mfa.py           # sanity checks against the real NHTSA/CRS/GREET sources
```

## What this honestly is not

Not a finished substitute for either field. No scenario analysis, no
life-cycle assessment, no geospatial layer yet (though the wind turbine
register carries real coordinates and municipality data that would support
one), and the risk side is a single real 2023 snapshot rather than a
multi-year trend (crm-trade-network's own fetcher supports other years, but
a full second year is hundreds of rate-limited live UN Comtrade calls, real
hours of wall-clock time, not something to trigger casually).
