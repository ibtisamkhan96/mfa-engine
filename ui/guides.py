"""The "How to read this page" guide behind the button at the top of every page.

Each entry says what the page shows, how to read its charts, the method in a sentence or two,
and one thing worth trying. The full methods, with sources, stay in each chart's "How this is
calculated" panel and on the Methods & data page; this is the short version a first-time reader
needs before looking at the charts.
"""

READING_ANY_PAGE = """
- **The coloured box under each title is the takeaway**: one sentence worked out from the numbers
  on screen, so it changes when you change the settings.
- **"How this is calculated"** under each chart holds the formula, the data it comes from and every
  assumption.
- **Hover a chart** for exact values. The camera icon in a chart's corner saves it as a PNG, and
  every chart has a CSV download underneath.
- **The sidebar applies to every page**: material first, then the page list, then the horizon, the
  recycling assumption, the supplier removed in the disruption, and the price.
- **Share a view**: the address bar keeps the material, system and horizon, so a copied link opens
  the same view.
- **Reading order**: the back and next links at the bottom of each page walk through the analysis
  from start to finish.
- **New to a term?** The Glossary page, under Reference, explains every term and formula in plain
  words and shows where each one appears.
"""

PAGE_GUIDES = {
    "overview": {
        "shows": "The headline answer for the material picked in the sidebar, and the four steps of the "
                 "analysis that lead to it.",
        "read": "The coloured box is the one-line answer. The four cards underneath summarise each step "
                "with its key figure; open any of them for the detail. The small grey line under each "
                "card's value gives its context.",
        "method": "The answer compares an average year of recycled material, valued at the price in the "
                  "sidebar, with the one-year cost of losing the largest supplier in 2023 UN Comtrade "
                  "trade data.",
        "try": "Pick another material in the sidebar. Every number and colour on every page follows it.",
    },
    "lifetimes": {
        "shows": "The share of units still in service at each age: turbines, EV batteries, or data-centre "
                 "equipment, depending on the material's system.",
        "read": "Every curve starts at 100% and falls as units retire. Where it crosses the dotted 50% line "
                "is the median lifetime, marked with a dot. For wind, the stepped line is what actually "
                "happened in Denmark's register (Kaplan-Meier) and the dashed line is the fitted Weibull "
                "curve the projections use beyond it.",
        "method": "Wind: Weibull fitted by maximum likelihood to every turbine in the register, including "
                  "those still standing. EVs: Weibull fitted to NHTSA's published vehicle survival table, "
                  "since the fleet is too young for its own. Data centres: ASHRAE median service lives with "
                  "a disclosed curve shape.",
        "try": "Compare copper in wind turbines with copper in EVs: a median around 24 years against 13.",
    },
    "stocks": {
        "shows": "The full material account for one material: how much is in use, how much enters service "
                 "each year, and how much leaves it.",
        "read": "Top panel: tonnes in use. Bottom panel: bars above the zero line enter service, bars below "
                "it leave. In each lower bar, the solid part is recycled back into use and the pale part "
                "is lost at end of life. The axis shows absolute tonnes on both sides of zero.",
        "method": "Each year the change in stock equals inflow minus outflow; the green badge confirms the "
                  "account closes. Recovered tonnes are outflow times the material's cited end-of-life "
                  "recycling rate (UNEP IRP 2011, or the EU 2031 battery targets).",
        "try": "Switch the recovery assumption in the sidebar for an EV material and watch the solid part "
               "of the bars change.",
    },
    "eol": {
        "shows": "Material leaving service each year, the retirement wave of units behind it, and which "
                 "turbine type, battery chemistry or infrastructure layer it comes from.",
        "read": "Pick materials with the chips above the first chart. With one material, the shaded band is "
                "the published low-high range of how much of it each unit contains. Turn on log scale to "
                "compare materials of very different size. In the Sankey diagram, link width is tonnes, "
                "from the category on the left to the material on the right.",
        "method": "Units retiring each year, times the material content per unit, summed across categories. "
                  "These are gross tonnes, before any recycling rate.",
        "try": "Select two materials to see whether they peak in the same year.",
    },
    "scenarios": {
        "shows": "Two futures run through the same engine: IEA's Stated Policies Scenario against a flat "
                 "2026 car market for EVs, or McKinsey's low and high 2030 capacity for data centres.",
        "read": "The solid coloured line is the higher scenario, the dashed grey line the baseline, and the "
                "shaded area the gap between them. The in-use stock responds at once; end-of-life outflow "
                "lags by roughly one lifetime.",
        "method": "Only the future input changes (EV sales or required data-centre capacity). History, "
                  "lifetimes and material content are identical in both runs. Wind has no scenarios yet.",
        "try": "Switch between in-use stock and end-of-life outflow to see the lag.",
    },
    "supplymap": {
        "shows": "Where the material is mined, where it is traded, and how a disruption spreads through "
                 "that trade network.",
        "read": "Circle area is tonnes mined (or known reserves, if you switch), and countries are shaded by "
                "the same measure. Arcs are the largest 2023 trade links: thicker means more value, and the "
                "moving dots travel from exporter to importer. Hover a country for its numbers, click it to "
                "isolate its trade, click it again to clear.",
        "method": "Production and reserves: USGS Mineral Commodity Summaries 2025 (2024 data). Trade: 2023 UN "
                  "Comtrade, the same data every risk page uses. The replay is crm-trade-network's cascade: "
                  "the removed supplier stops, then any country losing over 20% of its trade in this "
                  "commodity fails in turn.",
        "try": "Press \"Play the disruption\", then change the supplier removed in the sidebar and play it "
               "again.",
    },
    "disruption": {
        "shows": "Whether recycling this material matters at the scale of losing one supplier for a year.",
        "read": "The cards compare one average year of recovery with two one-year shortfalls: with no other "
                "supplier stepping in, and with the others using their spare capacity. The bar chart is on a "
                "log scale; the year-by-year chart shows recovery against the two shortfall lines; the "
                "stacked bar shows how much of the gap recycling would cover.",
        "method": "Shortfall from crm-trade-network's cascade on 2023 UN Comtrade data. Recovery is recycled "
                  "tonnes times the price in the sidebar. Both are one-year figures on purpose, so the "
                  "comparison is like for like.",
        "try": "Change \"Supplier removed\" in the sidebar, or open \"Price assumption\" to test another price.",
    },
    "cascade": {
        "shows": "How many other countries fail once the removed supplier stops trading, and who trades "
                 "with whom.",
        "read": "The cards give the number of countries that fail, their share of the network and how many "
                "rounds it takes, at a 20% failure threshold. The threshold chart then re-runs the cascade "
                "from 5% to 95%: the shaded band is where the network switches from collapse to "
                "near-immunity. In the Sankey diagram, exporters are on the left and importers on the right; "
                "the removed supplier and its trade are red.",
        "method": "A linear-threshold cascade: a country fails once it has lost over 20% of its trade in this "
                  "commodity, because partners failed first. This follows Ouyang et al.'s cobalt method (2026), "
                  "applied to one trade layer instead of six life-cycle stages.",
        "try": "Change the supplier removed in the sidebar and see whether the transition band moves; then "
               "open the Supply map page and press play to watch the rounds on a map.",
    },
    "diversification": {
        "shows": "How much trade would have to move to bring every supplier under a maximum share.",
        "read": "The slider sets the cap. The cards show the reallocation needed with and without spare "
                "capacity; the bars compare each supplier's current exports with the reallocated ones, "
                "and the dotted line marks the cap.",
        "method": "A linear program that moves as little trade as possible, keeps total trade the same, and "
                  "lets no supplier grow beyond its current exports plus its spare capacity. The default "
                  "65% cap is the EU Critical Raw Materials Act benchmark.",
        "try": "Lower the cap until the reallocation becomes \"Not achievable\".",
    },
    "copper": {
        "shows": "Copper entering service, in use and leaving service in all three systems at once.",
        "read": "The axis is logarithmic, so equal gaps mean equal ratios and systems of very different "
                "size fit on one chart. When the sidebar material is copper, the thicker line is its system. Denmark is one "
                "country; EVs and data centres are global.",
        "method": "Each system runs its own forward projection: IEA STEPS sales for EVs, McKinsey's low "
                  "capacity case for data centres, and the existing fleet for Danish wind. These are gross "
                  "flows, before recycling.",
        "try": "Switch to \"Entering service\" to compare how much new copper each system needs.",
    },
    "materials": {
        "shows": "Every material's recycling measured against a one-year loss of its own largest supplier.",
        "read": "A longer bar means recycling matters more for that material. The dark bar assumes no other "
                "supplier steps in, the light bar allows spare capacity. The material picked in the sidebar "
                "is in bold.",
        "method": "The same calculation as the Disruption page, run once per material against its own "
                  "largest supplier. Materials without a cited recycling rate are listed under the chart, "
                  "not plotted.",
        "try": "With an EV material picked in the sidebar, switch the recovery assumption to the EU battery "
               "targets and watch the EV materials move.",
    },
    "glossary": {
        "shows": "Every technical term and formula the dashboard uses, grouped by topic.",
        "read": "Each card gives a plain explanation first, then the exact formula the code computes (where "
                "there is one), what its symbols mean, and links to the pages where the term appears. The "
                "index at the top jumps straight to any term.",
        "method": "The formulas are written from the dashboard's own code, not from textbooks, so they "
                  "describe what this dashboard actually calculates, including its simplifications.",
        "try": "Search for \"slack\" or \"threshold\", then follow a link to the page where it is used.",
    },
    "ask": {
        "shows": "A question box answered by Claude using only the numbers already computed on these pages.",
        "read": "Each answer comes with the exact context that was sent, so you can check it did not invent "
                "anything.",
        "method": "Claude is told to answer from those numbers alone and to say so when a question goes "
                  "beyond them. It needs your own Anthropic API key, which is used for one call and never "
                  "stored.",
        "try": "Ask why so little of a material leaving service is recycled.",
    },
    "methods": {
        "shows": "What this dashboard is and is not, where every number comes from, and how fresh the trade "
                 "data is.",
        "read": "Start with \"What this app actually is, and isn't\"; the data policy and freshness sections "
                "follow.",
        "method": "Every figure comes from cited real data: dk-wind-mfa's register, published EV and "
                  "data-centre sources, USGS, and 2023 UN Comtrade via crm-trade-network.",
        "try": "The live refresh re-asks UN Comtrade and takes several minutes; use it only when you need "
               "today's data.",
    },
}
