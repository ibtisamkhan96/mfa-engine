"""End-of-life recovery: the step between material leaving service and material actually
coming back as secondary supply.

Before this module existed, the dashboard compared a material's whole end-of-life outflow
against a real trade-disruption shortfall, as if every tonne leaving service were recycled.
For copper that overstates recovery by roughly half; for lithium and every rare earth element
in this project it overstates it by more than a hundredfold, since their real end-of-life
recycling rates are under 1%.

The metric used is the functional end-of-life recycling rate (EoL-RR), the International
Resource Panel's standard: the share of a metal in end-of-life products that is recycled back
into functional use, as opposed to lost to landfill, dissipation, or "downcycling" into a
dominant flow such as common steel scrap. Collection losses and process losses are both
already inside this one number, so it is applied once to the gross outflow, never multiplied
by a separate collection rate on top.

Sources, both checked against the primary documents, not recalled:

UNEP (2011). Assessing Mineral Resources in Society: Metal Stocks & Recycling Rates.
International Resource Panel summary of Graedel, T.E. et al. (2011), Recycling Rates of
Metals: A Status Report. Page 26 periodic table, read directly from the rendered page:
manganese, iron, cobalt, nickel and copper in the ">50%" bin; lithium, praseodymium,
neodymium, terbium and dysprosium in the "<1%" bin. Page 23: "an end-of-life recycling rate of
70-90% can be estimated for iron and steel." Carbon (graphite) is not assessed at all.

Regulation (EU) 2023/1542 (EU Batteries Regulation), Annex XII Part C, material recovery
targets for battery recycling: lithium 50% by 31 December 2027 and 80% by 31 December 2031;
cobalt, copper, lead and nickel 90% and 95%. These are targets for the recycling process
itself, not a collection rate, so the regime built on them assumes every end-of-life EV
battery is collected: an upper bound, not a forecast.

Where a source reports a bin rather than a point, (low, central, high) is (bin lower edge,
bin midpoint, bin upper edge). The midpoint is a disclosed convention, not a published point
estimate. A material with no cited rate is None, and applying it yields NaN rather than 0:
"not assessed" and "nothing recovered" are different claims and are kept apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RecoveryRate:
    low: float
    central: float
    high: float
    source: str
    note: str = ""


_UNEP_ABOVE_50 = "UNEP IRP 2011, EoL-RR >50% bin"
_UNEP_BELOW_1 = "UNEP IRP 2011, EoL-RR <1% bin"
_UNEP_IRON = "UNEP IRP 2011, iron and steel EoL-RR 70-90%"
_EU_BATTERY = "Regulation (EU) 2023/1542 Annex XII Part C (2027 target, 2031 target)"


def _above_50(note: str = "") -> RecoveryRate:
    return RecoveryRate(0.50, 0.75, 1.00, _UNEP_ABOVE_50, note)


def _below_1() -> RecoveryRate:
    return RecoveryRate(0.0, 0.005, 0.01, _UNEP_BELOW_1)


_IRON_STEEL = RecoveryRate(0.70, 0.80, 0.90, _UNEP_IRON)

CURRENT_PRACTICE: dict[str, Optional[RecoveryRate]] = {
    "copper": _above_50(),
    "nickel": _above_50(),
    "cobalt": _above_50(),
    "manganese": _above_50(
        "UNEP's manganese figure is dominated by manganese recycled inside steel scrap; "
        "manganese in battery cathodes is often not recovered, so this is likely optimistic "
        "for the EV system."),
    "steel": _IRON_STEEL,
    "cast iron": _IRON_STEEL,
    "lithium": _below_1(),
    "neodymium": _below_1(),
    "praseodymium": _below_1(),
    "dysprosium": _below_1(),
    "terbium": _below_1(),
    "graphite": None,     # carbon is not assessed in UNEP IRP 2011
    "concrete": None,     # not a metal, outside UNEP IRP 2011's scope
}

EU_BATTERY_TARGETS: dict[str, Optional[RecoveryRate]] = {
    **CURRENT_PRACTICE,
    "lithium": RecoveryRate(0.50, 0.80, 0.80, _EU_BATTERY),
    "cobalt": RecoveryRate(0.90, 0.95, 0.95, _EU_BATTERY),
    "nickel": RecoveryRate(0.90, 0.95, 0.95, _EU_BATTERY),
    # Copper keeps the UNEP rate on purpose: this project's EV copper figure is whole-vehicle
    # copper (motor, wiring harness, busbars), not only battery copper, so the Regulation's
    # battery-copper target does not cleanly apply to it. Manganese and graphite have no
    # target in Annex XII Part C at all.
}

REGIMES: dict[str, dict[str, Optional[RecoveryRate]]] = {
    "Current practice (UNEP IRP 2011)": CURRENT_PRACTICE,
    "EU Batteries Regulation targets (upper bound)": EU_BATTERY_TARGETS,
}

# The Batteries Regulation governs batteries only; offering it for wind turbines would be
# meaningless, so each system lists only the regimes that genuinely apply to it.
REGIMES_BY_SYSTEM: dict[str, list[str]] = {
    "wind": ["Current practice (UNEP IRP 2011)"],
    "ev": ["Current practice (UNEP IRP 2011)", "EU Batteries Regulation targets (upper bound)"],
    "dc": ["Current practice (UNEP IRP 2011)"],
}

DEFAULT_REGIME = "Current practice (UNEP IRP 2011)"


def rate_for(material: str, regime: str = DEFAULT_REGIME) -> Optional[RecoveryRate]:
    return REGIMES[regime].get(material)


def recovered(outflow: pd.DataFrame, regime: str = DEFAULT_REGIME,
              bound: str = "central") -> pd.DataFrame:
    """Tonnes actually recovered: secondary_materials()' gross end-of-life outflow (wide, one
    column per material) times each material's end-of-life recycling rate. `bound` is
    "low", "central" or "high". Materials with no cited rate come back as NaN."""
    rates = REGIMES[regime]
    out = outflow[["year"]].copy()
    for col in outflow.columns:
        if col in ("year", "units_retiring"):
            continue
        rate = rates.get(col)
        out[col] = outflow[col] * getattr(rate, bound) if rate is not None else np.nan
    return out
