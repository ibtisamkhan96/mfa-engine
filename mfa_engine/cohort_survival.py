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


@dataclass
class CohortSurvivalMFA:
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
    def project_retirement_schedule(self, horizon_year: int, k: float, lam: float) -> pd.DataFrame:
        """Capacity (or count) retiring each year, by category, from
        conditional survival: an asset that has already reached age a does
        not face the whole survival curve, only what remains of it,
        S(b)/S(a). Each asset is projected individually under its own age
        and then summed, so the result is a genuine cohort model, not a
        single average asset scaled up."""
        df = self.existing.copy()
        df["age0"] = (self.snapshot - df[self.commissioned_col]).dt.days / 365.25
        df = df[df.age0 > 0]

        base_year = self.snapshot.year
        years = np.arange(base_year, horizon_year + 1)
        ages = df["age0"].values[:, None] + (years - base_year)[None, :]

        s0 = self.weibull_survival(df["age0"].values, k, lam)[:, None]
        surv = np.clip(self.weibull_survival(ages, k, lam) / s0, 0, 1)

        retiring_fraction = np.hstack([1 - surv[:, [0]], surv[:, :-1] - surv[:, 1:]])
        units_retiring = df[self.unit_col].values[:, None] * retiring_fraction

        out = pd.DataFrame(units_retiring, columns=years)
        out["category"] = df[self.category_col].values
        long = out.melt(id_vars=["category"], var_name="year", value_name="units_retiring")
        return long.groupby(["year", "category"], as_index=False).units_retiring.sum()

    # ------------------------------------------------------------- materials
    def material_intensity(self, category: str) -> dict[str, tuple[float, float, float]]:
        """Real, sourced material content per unit for one category, as
        {material_name: (low, central, high)}. This is the one piece of
        real domain knowledge every subclass must provide, everything above
        this method is generic and inherited unchanged. Raising here rather
        than returning made-up numbers is deliberate: a subclass that
        forgets to override this must fail loudly, not silently produce a
        fabricated result."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement material_intensity(category) "
            "with real, sourced coefficients. This base class does not guess."
        )

    def secondary_materials(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Material released each year by the retiring stock, using each
        category's own real material intensity. Uses the central estimate of
        each sourced range."""
        rows = []
        for year, grp in schedule.groupby("year"):
            rec = {"year": int(year), "units_retiring": grp.units_retiring.sum()}
            materials: dict[str, float] = {}
            for category, sub in grp.groupby("category"):
                intensity = self.material_intensity(category)
                for material, (_, central, __) in intensity.items():
                    materials[material] = materials.get(material, 0.0) + sub.units_retiring.sum() * central
            rec.update(materials)
            rows.append(rec)
        return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    def secondary_materials_range(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """The real (low, high) bounds secondary_materials() itself receives
        from material_intensity() but discards, keeping only the central
        value. Every source this engine's two real subclasses use (JRC for
        wind turbines, the CRS/GREET-derived figures for EV batteries)
        reports these as a genuine range, not a single point estimate, and
        collapsing straight to the centre throws that real information
        away. A separate method rather than changing secondary_materials()
        itself: the pooled central-value table is what a year-by-year total
        needs and is already relied on elsewhere (tests, the Sankey), this
        exists for the one real thing it cannot show, how much uncertainty
        the source data itself actually carries.

        Returns one row per year with `{material}_low` and `{material}_high`
        columns alongside secondary_materials()'s own `{material}` central
        columns, so a caller can plot a real band around a real line rather
        than inventing one."""
        rows = []
        for year, grp in schedule.groupby("year"):
            rec = {"year": int(year)}
            low_totals: dict[str, float] = {}
            high_totals: dict[str, float] = {}
            for category, sub in grp.groupby("category"):
                intensity = self.material_intensity(category)
                units = sub.units_retiring.sum()
                for material, (low, _, high) in intensity.items():
                    low_totals[material] = low_totals.get(material, 0.0) + units * low
                    high_totals[material] = high_totals.get(material, 0.0) + units * high
            for material in low_totals:
                rec[f"{material}_low"] = low_totals[material]
                rec[f"{material}_high"] = high_totals[material]
            rows.append(rec)
        return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    def secondary_materials_by_category(self, schedule: pd.DataFrame,
                                         horizon_year: Optional[int] = None) -> pd.DataFrame:
        """Same arithmetic as secondary_materials, but keeps the category
        breakdown instead of pooling it away: one row per (category,
        material), summed over every year up to horizon_year (or the whole
        schedule, if not given). secondary_materials() deliberately pools
        across category because that is what a year-by-year total needs;
        this exists for the one thing that pooled view cannot show, which
        asset category is the source of which material. Summing this
        output across category reproduces secondary_materials()'s own
        per-material totals exactly."""
        df = schedule if horizon_year is None else schedule[schedule.year <= horizon_year]
        rows = []
        for category, sub in df.groupby("category"):
            intensity = self.material_intensity(category)
            total_units = sub.units_retiring.sum()
            for material, (_, central, __) in intensity.items():
                rows.append({"category": category, "material": material,
                             "tonnes": total_units * central})
        return pd.DataFrame(rows)

    # --------------------------------------------------------------------- run
    def run(self, horizon_year: int) -> dict:
        """The whole pipeline, start to finish: fit the lifetime model from
        real censored data, project the cohort-by-cohort retirement wave out
        to horizon_year, and compute the secondary material flow that comes
        with it. Returns everything a caller would want to inspect, rather
        than only the final table, so the lifetime fit itself can be
        checked, not just trusted."""
        obs = self.build_observations()
        km = self.kaplan_meier(obs.duration.values, obs.event.values)
        k, lam = self.fit_weibull(obs.duration.values, obs.event.values)
        schedule = self.project_retirement_schedule(horizon_year, k, lam)
        materials = self.secondary_materials(schedule)
        return {
            "observations": obs,
            "kaplan_meier": km,
            "weibull_shape": k,
            "weibull_scale": lam,
            "median_survival_years": self.median_survival(km),
            "retirement_schedule": schedule,
            "secondary_materials": materials,
        }
