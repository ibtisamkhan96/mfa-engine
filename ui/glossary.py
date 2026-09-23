"""The Glossary page's content: every technical term and formula the dashboard uses.

Each entry has a plain one-sentence definition first, then (where there is one) the exact formula
the code computes, what its symbols mean, and the pages where the term appears. The formulas are
written from the code itself (mfa_engine/, crm-trade-network/src/), not from textbooks, so they
describe what this dashboard actually does, including its simplifications.
"""

GROUPS = [
    "Lifetimes and survival",
    "Stocks and flows",
    "Recycling",
    "Trade and supply risk",
    "Cascades and networks",
    "Scenarios and policy",
    "Reading the charts",
]

TERMS = [
    # ------------------------------------------------------------------ lifetimes
    dict(term="Survival curve", aka="S(t)", group="Lifetimes and survival",
         plain="The share of units (turbines, batteries, data-centre equipment) still in service at each "
               "age. It starts at 100% for new units and falls as they retire.",
         pages=["lifetimes"]),
    dict(term="Kaplan-Meier estimate", group="Lifetimes and survival",
         plain="A survival curve measured directly from what happened: at every age where some units "
               "retired, the curve drops by the share of units still in service that retired then.",
         formula=r"S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)",
         symbols="d_i: units that retired at age t_i. n_i: units still in service just before t_i (the "
                 "units at risk).",
         pages=["lifetimes"]),
    dict(term="Right-censoring", group="Lifetimes and survival",
         plain="A unit still standing is only known to have lasted at least its current age. It is "
               "counted as surviving so far, not dropped and not treated as retired; either mistake would "
               "make lifetimes look shorter than they are.",
         pages=["lifetimes"]),
    dict(term="Weibull curve", aka="shape k, scale λ", group="Lifetimes and survival",
         plain="A smooth two-parameter survival curve, fitted to the data and used to project "
               "retirements beyond the oldest ages anyone has observed.",
         formula=r"S(t) = \exp\!\left(-(t/\lambda)^k\right)",
         symbols="k (shape): above 1 means the retirement rate rises with age, as with wear-out. "
                 "λ (scale): the age by which about 63% of units have retired.",
         pages=["lifetimes"]),
    dict(term="Median lifetime", group="Lifetimes and survival",
         plain="The age by which half of all units have retired.",
         formula=r"t_{50} = \lambda\,(\ln 2)^{1/k}",
         symbols="Read off the fitted Weibull curve, where it crosses 50%.",
         pages=["lifetimes", "overview"]),
    dict(term="Conditional survival", group="Lifetimes and survival",
         plain="For a unit already standing at age a₀, only the rest of the curve applies. The "
               "chance it retires in a given future year comes from the curve divided by its survival "
               "so far.",
         formula=r"R(a_0,\,t) = \frac{S(a_0+t-1)}{S(a_0)} - \frac{S(a_0+t)}{S(a_0)}",
         symbols="R: probability a unit that has reached age a₀ retires in year t of the projection. "
                 "Used for the Danish turbine register.",
         pages=["eol"]),
    dict(term="Cohort", group="Lifetimes and survival",
         plain="All units that entered service in the same year (for EVs, also with the same battery "
               "chemistry). Each cohort ages and retires on its own, and the cohorts are then added up, "
               "rather than treating the fleet as one average unit.",
         pages=["stocks", "eol"]),

    # ------------------------------------------------------------------ stocks and flows
    dict(term="Dynamic material flow analysis", aka="dMFA", group="Stocks and flows",
         plain="Tracking how much of a material enters use, sits in use and leaves use, year by year, "
               "so that future waste and recycling can be projected.",
         pages=["overview", "stocks"]),
    dict(term="In-use stock", group="Stocks and flows",
         plain="Material inside products that are still in service in a given year.",
         pages=["stocks", "copper"]),
    dict(term="Inflow", group="Stocks and flows",
         plain="Material entering service in a year: new turbines, new cars, new or replacement "
               "data-centre equipment.",
         pages=["stocks", "copper"]),
    dict(term="End-of-life outflow", group="Stocks and flows",
         plain="Material leaving service in a year as units retire, before any recycling.",
         pages=["stocks", "eol"]),
    dict(term="Mass balance", group="Stocks and flows",
         plain="The account has to close: each year the stock changes by exactly what came in minus "
               "what went out. The green badge on the Stocks & flows page is this check.",
         formula=r"S_t - S_{t-1} = I_t - O_t",
         symbols="S_t: in-use stock in year t. I_t: inflow. O_t: end-of-life outflow.",
         pages=["stocks"]),
    dict(term="Register-based model", group="Stocks and flows",
         plain="Starts from a list of units known to be standing (Denmark's national turbine register) "
               "and projects each one forward from its own age with conditional survival.",
         pages=["lifetimes", "stocks"]),
    dict(term="Inflow-driven model", group="Stocks and flows",
         plain="Starts from what was sold each year (EV sales). Each year's sales decay from their own "
               "sale year, and the stock is whatever is still in service.",
         formula=r"S_t = \sum_{c \le t} I_c \, S(t - c)",
         symbols="I_c: units sold in year c. S(t - c): share of them still in service at age t - c.",
         pages=["stocks", "scenarios"]),
    dict(term="Stock-driven model", group="Stocks and flows",
         plain="Starts from how much stock the system needs (data-centre capacity) and works out what "
               "has to be built: whatever is needed that earlier builds no longer cover.",
         formula=r"I_t = \max\!\Big(0,\; S^{*}_t - \sum_{c < t} I_c \, S(t - c)\Big)",
         symbols="S*_t: stock required in year t. The sum is what earlier builds still have standing. "
                 "Before the first year the system starts in a steady state, building "
                 "S*_0 / Σ S(a) a year.",
         pages=["stocks", "lifetimes"]),
    dict(term="Material intensity", group="Stocks and flows",
         plain="Tonnes of a material per unit: per turbine type, per battery chemistry, or per MW of "
               "data-centre capacity. Material leaving service is units retiring times this.",
         formula=r"M_{m,t} = \sum_{c} U_{c,t} \times I_{m,c}",
         symbols="M_{m,t}: tonnes of material m leaving service in year t. U_{c,t}: units of category c "
                 "retiring that year. I_{m,c}: tonnes of m per unit of c.",
         pages=["eol"]),
    dict(term="Retirement wave", group="Stocks and flows",
         plain="The number of units leaving service each year. Its timing decides when material "
               "comes back.",
         pages=["eol"]),
    dict(term="Low-high content range", group="Stocks and flows",
         plain="The published range of how much of a material each unit contains, carried through the "
               "whole calculation and drawn as the shaded band on the End-of-life page.",
         pages=["eol"]),

    # ------------------------------------------------------------------ recycling
    dict(term="End-of-life recycling rate", aka="EoL-RR", group="Recycling",
         plain="The share of a material in retired products that is actually recycled back into use, "
               "after collection and processing losses. Only this share counts as recovered.",
         formula=r"R_t = O_t \times \mathrm{EoL\text{-}RR}",
         symbols="R_t: tonnes recovered in year t. O_t: tonnes leaving service.",
         pages=["stocks", "disruption"]),
    dict(term="Current practice (UNEP IRP 2011)", group="Recycling",
         plain="Measured recycling rates from the UN International Resource Panel: over 50% for copper, "
               "nickel, cobalt and manganese, 70-90% for iron and steel, under 1% for lithium and the rare "
               "earths. Rates given as bands use the band's midpoint, with its edges as the range.",
         pages=["stocks", "disruption"]),
    dict(term="EU Batteries Regulation targets", group="Recycling",
         plain="Recovery targets in EU law for battery recycling (for 2031: lithium 80%, cobalt and "
               "nickel 95%), applied as if every battery were collected. An upper bound, for EV "
               "batteries only.",
         pages=["stocks", "materials"]),
    dict(term="Lithium carbonate equivalent", aka="LCE", group="Recycling",
         plain="Lithium is traded and priced as lithium carbonate, while the model counts lithium "
               "metal. Recovered lithium is converted to carbonate before it is given a dollar value.",
         formula=r"\frac{M_{\mathrm{Li_2CO_3}}}{2\,M_{\mathrm{Li}}} = \frac{73.89}{2 \times 6.94} = 5.32",
         symbols="One tonne of lithium is 5.32 tonnes of lithium carbonate.",
         pages=["disruption"]),

    # ------------------------------------------------------------------ trade and supply risk
    dict(term="HS code", group="Trade and supply risk",
         plain="The Harmonized System code customs agencies use to classify goods. Four digits is a "
               "family (2836: all carbonates); six digits is a specific product (2836.91: lithium "
               "carbonate).",
         pages=["supplymap", "disruption"]),
    dict(term="UN Comtrade", group="Trade and supply risk",
         plain="The UN's database of what each country reports it imported and exported, by product "
               "and partner. Every trade figure here is from its 2023 data.",
         pages=["supplymap", "cascade", "methods"]),
    dict(term="Supplier concentration", aka="HHI", group="Trade and supply risk",
         plain="The Herfindahl-Hirschman index: square every supplier's share of the trade and add them "
               "up. 1 means a single supplier, 0.1 is roughly ten equal ones, and above 0.25 counts as "
               "highly concentrated.",
         formula=r"\mathrm{HHI} = \sum_i s_i^{2}",
         symbols="s_i: supplier i's share of trade value, between 0 and 1.",
         pages=["overview"]),
    dict(term="Largest supplier's share", group="Trade and supply risk",
         plain="The share of world trade value coming from the single biggest exporter.",
         formula=r"s_{\mathrm{top}} = \frac{x_{\mathrm{top}}}{\sum_i x_i}",
         symbols="x_i: exports of supplier i, in US dollars.",
         pages=["overview", "supplymap", "disruption"]),
    dict(term="Importer- and exporter-reported trade", group="Trade and supply risk",
         plain="Both sides of a trade report it, and their figures often disagree. Concentration uses "
               "importers' reports; the disruption, cascade and map pages use one list built from both, "
               "keeping the larger figure where they differ. That is why a supplier's share can differ "
               "slightly between pages.",
         pages=["overview", "disruption"]),
    dict(term="Re-export hub", group="Trade and supply risk",
         plain="A country whose trade is partly goods passing through rather than made or used there: "
               "the Netherlands, Belgium, Singapore, Hong Kong and the UAE are flagged.",
         pages=["supplymap"]),
    dict(term="Supply shortfall", group="Trade and supply risk",
         plain="The trade value that goes missing for a year if one supplier stops exporting and the "
               "others cannot make it up.",
         formula=r"\text{shortfall} = \max\!\Big(0,\; X_{\text{removed}} - \text{slack} \times "
                 r"\sum_{\text{others}} X_i\Big)",
         symbols="X: a supplier's exports in US dollars. With slack = 0 no one steps in (the "
                 "no-substitution case).",
         pages=["disruption", "materials"]),
    dict(term="Spare capacity", aka="slack", group="Trade and supply risk",
         plain="How much the remaining suppliers could raise their exports. 22.4% for copper, from the "
               "International Copper Study Group's 2023 mine utilisation; 20%, a disclosed assumption, "
               "for the other materials, which have no published equivalent.",
         pages=["disruption", "diversification"]),
    dict(term="Coverage", group="Trade and supply risk",
         plain="How much of one year's shortfall an average year of recycled material would fill, with "
               "both sides in dollars.",
         formula=r"\text{coverage} = \frac{\bar{t} \times p}{\text{shortfall}}",
         symbols="t̄: average tonnes recovered a year. p: price per tonne (for lithium, times 5.32; see "
                 "Lithium carbonate equivalent).",
         pages=["disruption", "materials", "overview"]),
    dict(term="Average year and peak year", group="Trade and supply risk",
         plain="Recovery is averaged over the projection years so it compares like for like with a "
               "one-year shortfall. The single busiest year is shown next to it, so the average does not "
               "hide how large recovery gets at its peak.",
         formula=r"\bar{t} = \frac{1}{N} \sum_{t=1}^{N} R_t",
         symbols="R_t: tonnes recovered in year t. N: years in the projection.",
         pages=["disruption"]),

    # ------------------------------------------------------------------ cascades
    dict(term="Cascade", aka="linear-threshold model", group="Cascades and networks",
         plain="Remove one country from the trade network, then check every other country: any that has "
               "lost too much of its trade, because partners it bought from or sold to have failed, fails "
               "too. This repeats round after round until no more countries fail.",
         formula=r"\text{country } n \text{ fails when } \frac{L^{\text{in}}_n + L^{\text{out}}_n}"
                 r"{T^{\text{in}}_n + T^{\text{out}}_n} > \beta",
         symbols="L: trade value country n has lost (in: imports it can no longer buy, out: exports it can "
                 "no longer sell). T: its total imports and exports of this commodity. β: the failure "
                 "threshold.",
         pages=["cascade", "supplymap"]),
    dict(term="Failure threshold", aka="β", group="Cascades and networks",
         plain="The share of its own trade a country can lose and still carry on. A low threshold means "
               "fragile countries. The pages use 20%; the threshold chart tries every value from 5% to "
               "95%.",
         pages=["cascade"]),
    dict(term="Transition band", group="Cascades and networks",
         plain="The range of thresholds over which the network switches from almost total collapse "
               "(95% or more failing) to almost none (under 5%).",
         pages=["cascade"]),
    dict(term="Robust-yet-fragile", group="Cascades and networks",
         plain="A network that shrugs off random losses but collapses when the right single country is "
               "removed, with the switch happening abruptly rather than gradually.",
         pages=["cascade"]),
    dict(term="Rounds", group="Cascades and networks",
         plain="The number of waves of failure before the cascade stops.",
         pages=["cascade", "supplymap"]),

    # ------------------------------------------------------------------ scenarios and policy
    dict(term="STEPS", aka="IEA Stated Policies Scenario", group="Scenarios and policy",
         plain="The International Energy Agency's projection of EV sales under policies already in place, "
               "reaching about half of global car sales by 2035; held at that level after 2035 rather "
               "than extended.",
         pages=["scenarios", "copper"]),
    dict(term="Flat-2026 floor", group="Scenarios and policy",
         plain="A comparison case in which EV sales stay at their 2026 level. A baseline to measure "
               "STEPS against, not a forecast.",
         pages=["scenarios"]),
    dict(term="McKinsey capacity cases", group="Scenarios and policy",
         plain="Data-centre capacity demand of 60 GW in 2023, rising to 171 GW (low case) or 219 GW "
               "(high case) by 2030.",
         pages=["scenarios", "copper"]),
    dict(term="Minimum-cost diversification", group="Scenarios and policy",
         plain="A linear program that moves as little trade as possible between existing suppliers so "
               "that none exceeds a cap, keeping total trade the same.",
         formula=r"\min \sum_i d_i \ \text{ s.t. } \sum_i x_i = T,\ |x_i - c_i| \le d_i,\ "
                 r"0 \le x_i \le \min(\text{cap} \cdot T,\ c_i(1+\text{slack}))",
         symbols="c_i: supplier i's current exports. x_i: its exports after reallocation. d_i: how much "
                 "that changes. T: total trade.",
         pages=["diversification"]),
    dict(term="EU CRMA 65% benchmark", group="Scenarios and policy",
         plain="The EU Critical Raw Materials Act benchmark that no more than 65% of any strategic raw "
               "material should come from a single third country. It is the default cap on the "
               "Diversification page.",
         pages=["diversification"]),

    # ------------------------------------------------------------------ charts
    dict(term="Log scale", group="Reading the charts",
         plain="An axis on which each major gridline is ten times the one before, so equal gaps mean "
               "equal ratios and very small and very large values fit on one chart.",
         pages=["copper", "disruption", "eol"]),
    dict(term="Sankey diagram", group="Reading the charts",
         plain="A flow chart in which the width of each link is proportional to the amount flowing "
               "along it.",
         pages=["eol", "cascade"]),
    dict(term="Equal Earth projection", group="Reading the charts",
         plain="The world-map projection on the Supply map. It is equal-area: a country's drawn area is "
               "proportional to its real area, so large and small countries are not distorted.",
         pages=["supplymap"]),
]
