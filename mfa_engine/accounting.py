"""Material accounting shared by every dynamic MFA model in this package.

Two model families live here: CohortSurvivalMFA (asset registers and inflow tables, where each
asset or sales cohort is followed from its own age) and StockDrivenMFA (systems described by the
stock they need each year, where inflow is derived as growth plus replacement). They answer
"how many units enter, stand and leave each year" in different ways, but once that unit-level
account exists, turning units into tonnes of each material is identical arithmetic. It lives
here once, so the two families can never drift into slightly different material maths.

A model only has to provide material_intensity(category), real sourced material content per
unit for each category, as {material: (low, central, high)}.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


class MaterialAccounting:

    def material_intensity(self, category: str) -> dict[str, tuple[float, float, float]]:
        """Real, sourced material content per unit for one category, as
        {material_name: (low, central, high)}. This is the one piece of
        real domain knowledge every subclass must provide, everything else
        is generic and inherited unchanged. Raising here rather than
        returning made-up numbers is deliberate: a subclass that forgets to
        override this must fail loudly, not silently produce a fabricated
        result."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement material_intensity(category) "
            "with real, sourced coefficients. This base class does not guess."
        )

    def secondary_materials(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """Material leaving service each year inside the retiring stock
        (gross end-of-life outflow), using each category's own real material
        intensity, central estimate of each sourced range.

        This is NOT the same as material recovered. Only a fraction of an
        end-of-life flow is actually collected and recycled, and for some
        materials that fraction is under 1% (UNEP IRP 2011: lithium and every
        rare earth element). mfa_engine.recovery applies real, cited
        end-of-life recycling rates to this outflow; use that for anything
        described as recovered or secondary supply. The method keeps its
        original name because dk-wind-mfa's own published figures, which
        tests/test_reproduces_dk_wind_mfa.py checks against, are exactly
        this gross outflow."""
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
        value. Every source this package's real subclasses use reports these
        as a genuine range where one exists, and collapsing straight to the
        centre throws that real information away. A separate method rather
        than changing secondary_materials() itself: the pooled central-value
        table is what a year-by-year total needs and is already relied on
        elsewhere (tests, the Sankey), this exists for the one real thing it
        cannot show, how much uncertainty the source data itself carries.

        Returns one row per year with `{material}_low` and `{material}_high`
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
        schedule, if not given). Summing this output across category
        reproduces secondary_materials()'s own per-material totals exactly."""
        df = schedule if horizon_year is None else schedule[schedule.year <= horizon_year]
        rows = []
        for category, sub in df.groupby("category"):
            intensity = self.material_intensity(category)
            total_units = sub.units_retiring.sum()
            for material, (_, central, __) in intensity.items():
                rows.append({"category": category, "material": material,
                             "tonnes": total_units * central})
        return pd.DataFrame(rows)

    def material_stock_flows(self, stock_flows: pd.DataFrame) -> pd.DataFrame:
        """A unit-level stock-flow account converted to tonnes of each
        material, one row per (year, material): inflow_t, stock_t, outflow_t.
        Central intensity estimate, the same one secondary_materials uses,
        so outflow_t here equals secondary_materials' own column exactly.
        Unit to tonne conversion is linear, so the unit-level mass balance
        carries through to every material unchanged."""
        rows = []
        for (year, category), grp in stock_flows.groupby(["year", "category"]):
            for material, (_, central, __) in self.material_intensity(category).items():
                rows.append({
                    "year": int(year), "material": material,
                    "inflow_t": grp.inflow_units.sum() * central,
                    "stock_t": grp.stock_units.sum() * central,
                    "outflow_t": grp.outflow_units.sum() * central,
                })
        return (pd.DataFrame(rows).groupby(["year", "material"], as_index=False)
                  [["inflow_t", "stock_t", "outflow_t"]].sum())

    @staticmethod
    def mass_balance_residual(material_flows: pd.DataFrame) -> pd.Series:
        """Largest absolute violation, per material, of
        stock(t) - stock(t-1) = inflow(t) - outflow(t), in tonnes. A real
        stock-flow model has to close; this is how it is checked, not
        assumed. Should be at floating-point noise level."""
        residuals = {}
        for material, grp in material_flows.sort_values("year").groupby("material"):
            d_stock = grp.stock_t.diff().iloc[1:]
            net = (grp.inflow_t - grp.outflow_t).iloc[1:]
            residuals[material] = float((d_stock - net).abs().max()) if len(d_stock) else 0.0
        return pd.Series(residuals, name="max_abs_residual_t")
