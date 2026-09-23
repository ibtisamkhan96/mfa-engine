"""A reusable dynamic material flow analysis engine for any population of
long-lived, individually trackable assets: wind turbines, servers, EV
batteries, anything where

  - some assets are still in service (their eventual lifetime is unknown,
    a right-censored observation)
  - some assets have already been retired (a known, realised lifetime)
  - each asset belongs to a category that determines how much material it
    contains per unit of its own size
  - the real question is dynamic, not just "how much material is in the
    standing stock today", but "when does it come back out, and how much"

This is the same design idea as ODYM/flodym: one reusable base class
(CohortSurvivalMFA) implements the actual survival-analysis and
cohort-projection machinery once. A subclass provides exactly one piece of
real, sourced domain knowledge, material_intensity(), for its own specific
asset system. Everything else, censored survival estimation, Weibull
extrapolation, cohort-by-cohort retirement projection, is inherited
unchanged.

The survival-analysis and cohort-projection arithmetic below is a direct
generalisation of the wind-turbine-specific code originally written in
dk-wind-mfa's own src/survival.py and src/project.py: the same arithmetic,
with the wind-specific column names and JRC coefficients removed. This
generalisation is only trusted because it was verified to reproduce
dk-wind-mfa's own published numbers exactly, see
tests/test_reproduces_dk_wind_mfa.py in this package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mfa_engine.accounting import MaterialAccounting


@dataclass
class CohortSurvivalMFA(MaterialAccounting):
    """Base class. Construct with two asset tables and the column names that
    carry the information this engine actually needs, then call
    .run(horizon_year) to get a year-by-year retirement schedule and the
    secondary material flow that comes with it.

    existing: one row per asset still in service.
    retired: one row per asset already taken out of service.

    Column name arguments let this work on any real dataset's own column
    names, rather than forcing every user to rename their data to match
    this engine, the same reason flodym's own DimensionSet takes column
    names rather than hardcoding them.
    """

    existing: pd.DataFrame
    retired: pd.DataFrame
    snapshot: pd.Timestamp
    id_col: str
    unit_col: str            # the asset's own size measure, e.g. capacity_mw, or 1 for a simple count
    category_col: str        # determines material intensity, e.g. drivetrain topology, device class
    commissioned_col: str
    decommissioned_col: Optional[str] = None   # used only if lifetime_col is not already provided
    lifetime_col: Optional[str] = None         # a realised-lifetime column on `retired`, if already computed

    # How `existing` should be read, the textbook dMFA distinction (Muller 2006; Pauliuk &
    # Heeren's ODYM). False, the default: `existing` is an asset REGISTER, every row is known
    # to still be in service at `snapshot`, so each asset's future is conditional on having
    # survived to its current age, S(a)/S(a0). True: `existing` is an INFLOW table (sales
    # or installations by year), nothing is known about which units survived, so each
    # cohort decays from its own entry year under unconditional survival S(a), and the
    # account starts at the first cohort's entry year, not at the snapshot. Treating sales
    # as if they were survivors overstates the stock and pushes retirements that already
    # happened into the future.
    inflow_driven = False

    # ----------------------------------------------------------------- survival
    def build_observations(self) -> pd.DataFrame:
        """One row per asset: how long it lasted (or has lasted so far), and
        whether that ended in retirement (event=1) or is still ongoing,
        right-censored, as of `snapshot` (event=0).

        Deliberately pools every category into one survival curve here,
        matching the original wind-turbine pipeline this engine generalises:
        the lifetime model itself is fitted across the whole population, not
        per category, category only re-enters later, in
        project_retirement_schedule, to split the projected retirement wave
        by what each retiring asset actually contains. Fitting per category
        would usually starve smaller categories of retirement events to fit
        a reliable curve from.
        """
        retired = self.retired.copy()
        if self.lifetime_col and self.lifetime_col in retired.columns:
            duration = retired[self.lifetime_col].astype(float)
        else:
            duration = (retired[self.decommissioned_col] - retired[self.commissioned_col]).dt.days / 365.25
        ret = pd.DataFrame({
            "id": retired[self.id_col].values,
            "unit": retired[self.unit_col].values,
            "duration": duration.values,
            "event": 1,
        })
        ret = ret[ret.duration.notna() & (ret.duration > 0)].reset_index(drop=True)

        existing = self.existing.copy()
        age = (self.snapshot - existing[self.commissioned_col]).dt.days / 365.25
        ex = pd.DataFrame({
            "id": existing[self.id_col].values,
            "unit": existing[self.unit_col].values,
            "duration": age.values,
            "event": 0,
        })
        ex = ex[ex.duration > 0].reset_index(drop=True)

        return pd.concat([ret, ex], ignore_index=True)

    @staticmethod
    def kaplan_meier(duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
        """Survival curve S(t): probability an asset is still in service at age t.
        Uses every observation, including the still-in-service ones, treating
        them as censored, rather than averaging only the completed lifetimes,
        which systematically under-counts long-lived assets (right-censoring
        bias)."""
        duration = np.asarray(duration, dtype=float)
        event = np.asarray(event, dtype=int)

        times = np.unique(duration[event == 1])
        n_at_risk, n_events, surv = [], [], []
        s = 1.0
        for t in times:
            at_risk = int((duration >= t).sum())
            died = int(((duration == t) & (event == 1)).sum())
            s *= (1 - died / at_risk) if at_risk else 1.0
            n_at_risk.append(at_risk); n_events.append(died); surv.append(s)

        return pd.DataFrame({"t": times, "at_risk": n_at_risk, "events": n_events, "survival": surv})

    @staticmethod
    def median_survival(km: pd.DataFrame) -> float:
        below = km[km.survival <= 0.5]
        return float(below.t.iloc[0]) if len(below) else float("nan")

    @staticmethod
    def survival_at(km: pd.DataFrame, t: float) -> float:
        prior = km[km.t <= t]
        return float(prior.survival.iloc[-1]) if len(prior) else 1.0

    # ------------------------------------------------------------- extrapolation
    @staticmethod
    def fit_weibull(duration: np.ndarray, event: np.ndarray) -> tuple[float, float]:
        """Weibull (shape k, scale lam) fitted by maximum likelihood on the
        same right-censored data. Retired assets contribute their density,
        still-in-service ones contribute their survival function, so the fit
        does not repeat the bias a naive average would. Needed because the
        empirical Kaplan-Meier curve stops at the oldest observed retirement,
        and a parametric extension is needed to project further into the
        future."""
        t = np.clip(np.asarray(duration, float), 1e-6, None)
        d = np.asarray(event, int)

        def neg_ll(p):
            k, lam = np.exp(p)
            z = (t / lam) ** k
            log_f = np.log(k / lam) + (k - 1) * np.log(t / lam) - z
            return -(d * log_f - (1 - d) * z).sum()

        res = minimize(neg_ll, x0=np.log([2.0, 25.0]), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        k, lam = np.exp(res.x)
        return float(k), float(lam)

    @staticmethod
    def weibull_survival(age: np.ndarray, k: float, lam: float) -> np.ndarray:
        return np.exp(-((np.clip(age, 0, None) / lam) ** k))

    @staticmethod
    def fit_weibull_to_curve(ages: np.ndarray, survival_rates: np.ndarray) -> tuple[float, float]:
        """A second, real way to get (k, lam): least-squares curve-fitting
        against an already-published aggregate survival-rate-by-age table,
        rather than maximum likelihood on this population's own individual
        censored lifetimes (fit_weibull, above).

        Needed for populations too young to have generated real retirement
        events yet: fit_weibull has nothing to fit from if every observation
        is still censored (zero deaths), which is the honest state of a
        population like the global EV fleet today. The right move in that
        case is what real published methodology already does for exactly
        this problem, borrow a real, externally-published survival curve for
        a comparable population, rather than inventing a lifetime for a
        population that has not lived long enough to reveal its own."""
        ages = np.asarray(ages, dtype=float)
        survival_rates = np.asarray(survival_rates, dtype=float)

        def sse(p):
            k, lam = np.exp(p)
            return np.sum((CohortSurvivalMFA.weibull_survival(ages, k, lam) - survival_rates) ** 2)

        res = minimize(sse, x0=np.log([3.0, 15.0]), method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
        k, lam = np.exp(res.x)
        return float(k), float(lam)

    # ------------------------------------------------------------------ cohorts
    def _projectable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Which assets enter the projection. The default keeps only assets
        already in service at `snapshot` (age0 > 0): in a real asset register,
        a commissioning date after the snapshot is a data error, not a future
        asset. A subclass whose cohorts come from a deployment table (sales
        or installations by year) overrides this to keep post-snapshot
        cohorts, which are then genuine future inflows, see
        project_stock_flows."""
        return df[df.age0 > 0]

    def project_stock_flows(self, horizon_year: int, k: float, lam: float) -> pd.DataFrame:
        """The full stock-flow account, one row per (year, category):
        inflow_units (assets entering service that year), stock_units (assets
        in service at that point), outflow_units (assets retiring that year).

        Every asset is projected individually from its own age, then summed,
        so this is a genuine cohort model, not one average asset scaled up.
        For a register (inflow_driven False) that uses conditional survival:
        an asset known to have reached age a only faces what remains of the
        curve, S(b)/S(a). For an inflow table (inflow_driven True) each cohort
        decays from its own entry year under unconditional survival S(b), see
        the inflow_driven class attribute for why the two differ.

        An asset commissioned after `snapshot` (only possible when a
        subclass's _projectable keeps it) counts as nothing, not stock and
        not outflow, until its own commissioning year, then enters once as
        inflow. That makes the account close exactly every year:
        stock(t) - stock(t-1) = inflow(t) - outflow(t). tests/test_stock_flows.py
        checks this rather than trusting it."""
        df = self.existing.copy()
        df["age0"] = (self.snapshot - df[self.commissioned_col]).dt.days / 365.25
        df = self._projectable(df)

        base_year = self.snapshot.year
        if self.inflow_driven:
            start_year = int(df[self.commissioned_col].dt.year.min())
            s0 = np.ones((len(df), 1))
        else:
            start_year = base_year
            s0 = self.weibull_survival(df["age0"].values, k, lam)[:, None]
        years = np.arange(start_year, horizon_year + 1)
        ages = df["age0"].values[:, None] + (years - base_year)[None, :]

        surv = np.clip(self.weibull_survival(ages, k, lam) / s0, 0, 1)
        in_service = ages >= 0
        # A register's first year is opening stock (it was already standing); an inflow
        # table's first year is the first cohort ENTERING service, so it is inflow.
        before_first_year = (np.zeros_like(in_service[:, [0]]) if self.inflow_driven
                             else in_service[:, [0]])
        was_in_service = np.hstack([before_first_year, in_service[:, :-1]])

        units = df[self.unit_col].values[:, None].astype(float)
        retiring_fraction = np.hstack([1 - surv[:, [0]], surv[:, :-1] - surv[:, 1:]])
        flows = {
            "inflow_units": units * (in_service & ~was_in_service),
            "stock_units": units * surv * in_service,
            "outflow_units": units * retiring_fraction,
        }

        categories = df[self.category_col].values
        merged = None
        for name, values in flows.items():
            wide = pd.DataFrame(values, columns=years)
            wide["category"] = categories
            long = (wide.melt(id_vars=["category"], var_name="year", value_name=name)
                        .groupby(["year", "category"], as_index=False)[name].sum())
            merged = long if merged is None else merged.merge(long, on=["year", "category"])
        return merged

    def project_retirement_schedule(self, horizon_year: int, k: float, lam: float) -> pd.DataFrame:
        """Units retiring each year, by category: the outflow column of
        project_stock_flows, kept as its own method because the secondary
        material accounting below and every test built before stock-flow
        accounting existed read it in exactly this shape. One source of
        truth, so the retirement wave and the stock account can never
        disagree with each other."""
        sf = self.project_stock_flows(horizon_year, k, lam)
        return sf[["year", "category", "outflow_units"]].rename(columns={"outflow_units": "units_retiring"})

    # --------------------------------------------------------------------- run
    def _results_from_fit(self, horizon_year: int, k: float, lam: float) -> dict:
        """Everything downstream of the lifetime fit, shared by every
        subclass however it obtains (k, lam): the stock-flow account, the
        retirement wave read from it, and both in material terms."""
        stock_flows = self.project_stock_flows(horizon_year, k, lam)
        schedule = stock_flows[["year", "category", "outflow_units"]].rename(
            columns={"outflow_units": "units_retiring"})
        return {
            "weibull_shape": k,
            "weibull_scale": lam,
            "stock_flows": stock_flows,
            "material_stock_flows": self.material_stock_flows(stock_flows),
            "retirement_schedule": schedule,
            "secondary_materials": self.secondary_materials(schedule),
        }

    def run(self, horizon_year: int) -> dict:
        """The whole pipeline, start to finish: fit the lifetime model from
        real censored data, then build the full stock-flow account and the
        end-of-life material outflow that comes with it. Returns everything
        a caller would want to inspect, not only the final table, so the
        lifetime fit itself can be checked, not just trusted."""
        obs = self.build_observations()
        km = self.kaplan_meier(obs.duration.values, obs.event.values)
        k, lam = self.fit_weibull(obs.duration.values, obs.event.values)
        return {
            "observations": obs,
            "kaplan_meier": km,
            "median_survival_years": self.median_survival(km),
            **self._results_from_fit(horizon_year, k, lam),
        }
