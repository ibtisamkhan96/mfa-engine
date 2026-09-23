"""Stock-driven dynamic MFA, the third textbook dMFA mode (Muller 2006; the same idea as ODYM's
stock-driven dynamic stock model).

CohortSurvivalMFA covers two ways of describing a system: a register of what is standing (Denmark's
wind turbines) or an inflow table of what was built (global EV sales). Some systems are only
described by the stock they NEED each year: installed data-centre capacity, a building floor
area, a length of grid cable. Given that required stock and each component's lifetime, this model
derives what has to be built every year: growth of the stock plus replacement of whatever reached
end of life. For long-lived systems with shorter-lived parts, replacement is most of the story.

Discrete annual convention. A cohort built in year c survives to the end of year t with
probability S(t - c), with S(0) = 1 (nothing built in a year also retires that same year) and a
Weibull curve S(a) = exp(-(a/lambda)^k), lambda derived from the component's median life. Per
component category:

    stock(t)   = sum over c <= t of inflow(c) * S(t - c)
    outflow(t) = sum over c <  t of inflow(c) * (S(t - 1 - c) - S(t - c))
    inflow(t)  = max(0, required(t) - stock from earlier cohorts still standing in year t)

The max(0, .) only binds if the requirement falls faster than equipment retires on its own. The
account then reports the stock actually standing, above the requirement, rather than inventing
early demolition, and it still closes every year: stock(t) - stock(t-1) = inflow(t) - outflow(t).

Initial age structure: the first required value is treated as a steady state, built by a constant
past inflow I* = required(first) / sum over a of S(a) across a long spin-up. The first reported
year therefore already has a realistic mix of old and new equipment and a realistic replacement
flow, instead of a brand-new fleet with no retirements. Spin-up years are computed, then dropped;
nothing is reported before the first year of the required-stock series.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mfa_engine.accounting import MaterialAccounting


@dataclass
class StockDrivenMFA(MaterialAccounting):
    required_stock: pd.Series                      # units required in service, indexed by calendar year
    lifetimes: dict[str, tuple[float, float]]      # category -> (Weibull shape k, median life in years)
    snapshot: pd.Timestamp                         # last year backed by real data; later years are projection
    spin_up_years: int = 300

    @staticmethod
    def weibull_scale_from_median(k: float, median: float) -> float:
        return median / np.log(2) ** (1 / k)

    def survival_curve(self, category: str, n_years: int) -> np.ndarray:
        k, median = self.lifetimes[category]
        lam = self.weibull_scale_from_median(k, median)
        ages = np.arange(n_years)
        return np.exp(-((ages / lam) ** k))

    def required_stock_to(self, horizon_year: int) -> pd.Series:
        """The required stock for every year from the series' first year to horizon_year,
        held flat past its own last year: a requirement nobody has projected is not invented."""
        series = self.required_stock.sort_index()
        years = np.arange(int(series.index.min()), horizon_year + 1)
        return series.reindex(years).ffill()

    def project_stock_flows(self, horizon_year: int) -> pd.DataFrame:
        """One row per (year, category): inflow_units, stock_units, outflow_units, the same shape
        CohortSurvivalMFA.project_stock_flows returns, so everything downstream is shared."""
        required = self.required_stock_to(horizon_year)
        years = required.index.values
        m, n = self.spin_up_years, self.spin_up_years + len(years)
        target = np.concatenate([np.full(m, required.iloc[0]), required.values])

        rows = []
        for category in self.lifetimes:
            surv = self.survival_curve(category, n + 1)
            inflow = np.zeros(n)
            inflow[:m] = required.iloc[0] / surv[:m].sum()
            for t in range(m, n):
                cohorts = np.arange(t)
                standing = float(np.dot(inflow[:t], surv[t - cohorts]))
                inflow[t] = max(0.0, target[t] - standing)

            stock = np.empty(n)
            outflow = np.zeros(n)
            for t in range(n):
                cohorts = np.arange(t + 1)
                stock[t] = np.dot(inflow[:t + 1], surv[t - cohorts])
                if t:
                    older = cohorts[:-1]
                    outflow[t] = np.dot(inflow[:t], surv[t - 1 - older] - surv[t - older])

            for i, year in enumerate(years):
                rows.append({"year": int(year), "category": category,
                             "inflow_units": inflow[m + i], "stock_units": stock[m + i],
                             "outflow_units": outflow[m + i]})
        return pd.DataFrame(rows)

    def layer_lifetimes(self) -> dict[str, dict[str, float]]:
        return {cat: {"shape": k, "median": median, "scale": self.weibull_scale_from_median(k, median)}
                for cat, (k, median) in self.lifetimes.items()}

    def run(self, horizon_year: int) -> dict:
        stock_flows = self.project_stock_flows(horizon_year)
        schedule = stock_flows[["year", "category", "outflow_units"]].rename(
            columns={"outflow_units": "units_retiring"})
        return {
            "observations": None,           # a required-stock model has no individual assets to observe
            "kaplan_meier": None,
            "weibull_shape": None,          # one lifetime per layer instead, see layer_lifetimes
            "weibull_scale": None,
            "median_survival_years": None,
            "layer_lifetimes": self.layer_lifetimes(),
            "required_stock": self.required_stock_to(horizon_year),
            "stock_flows": stock_flows,
            "material_stock_flows": self.material_stock_flows(stock_flows),
            "retirement_schedule": schedule,
            "secondary_materials": self.secondary_materials(schedule),
        }
