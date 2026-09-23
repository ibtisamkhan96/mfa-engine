"""Build data/mine_supply.csv: where each material is mined, and where its reserves sit.

Every figure is copied, not re-estimated, from the Materials Data Series repositories
(github.com/ibtisamkhan96/materials-<name>), whose own READMEs cite USGS Mineral Commodity
Summaries 2025 for these tables: 2024 mine production, and reserves as reported in that edition.
The tables list the leading producers; where a source row covers "other countries" it is kept
(with no country code) so the page can say how much of the total sits off the map.

Units follow the source: copper, nickel, cobalt and lithium are metal content; rare earths are
rare-earth-oxide equivalent; graphite is natural graphite. Everything is converted to tonnes.

Run from the repo root:  python scripts/build_mine_supply.py [path/to/Materials Data Series]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERIES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Materials Data Series"
OUT = ROOT / "data" / "mine_supply.csv"
SOURCE = "USGS Mineral Commodity Summaries 2025, via the Materials Data Series ({repo})"

# (family, measure, repo, file, value column, multiplier to tonnes, unit, year)
TABLES = [
    ("Copper", "Mine production", "materials-copper", "mine_production_by_country_2024.csv", "mine_mt_2024", 1e6, "t copper content", 2024),
    ("Lithium", "Mine production", "materials-lithium", "production_by_country_2024.csv", "production_kt_2024", 1e3, "t lithium content", 2024),
    ("Lithium", "Reserves", "materials-lithium", "reserves_by_country.csv", "reserves_mt", 1e6, "t lithium content", 2024),
    ("Nickel", "Mine production", "materials-nickel", "mine_production_by_country_2024.csv", "mine_kt_2024", 1e3, "t nickel content", 2024),
    ("Cobalt", "Mine production", "materials-cobalt", "production_by_country.csv", "production_t", 1.0, "t cobalt content", 2024),
    ("Cobalt", "Reserves", "materials-cobalt", "reserves_by_country.csv", "reserves_t", 1.0, "t cobalt content", 2024),
    ("Graphite", "Mine production", "materials-graphite", "mine_production.csv", "mine_t", 1.0, "t natural graphite", 2024),
    ("Graphite", "Reserves", "materials-graphite", "reserves.csv", "reserves_mt", 1e6, "t natural graphite", 2024),
    ("Rare earths", "Mine production", "materials-rare-earths", "mine_production_by_country_2024.csv", "mine_kt_2024", 1e3, "t rare-earth oxide", 2024),
    ("Rare earths", "Reserves", "materials-rare-earths", "reserves_by_country.csv", "reserves_mt", 1e6, "t rare-earth oxide", 2024),
]

ISO3 = {
    "Argentina": "ARG", "Australia": "AUS", "Bolivia": "BOL", "Brazil": "BRA", "Canada": "CAN",
    "Chile": "CHL", "China": "CHN", "Cuba": "CUB", "DR Congo": "COD", "India": "IND",
    "Indonesia": "IDN", "Kazakhstan": "KAZ", "Madagascar": "MDG", "Mozambique": "MOZ",
    "Myanmar": "MMR", "New Caledonia": "NCL", "Nigeria": "NGA", "Peru": "PER",
    "Philippines": "PHL", "Russia": "RUS", "Tanzania": "TZA", "Thailand": "THA",
    "United States": "USA", "Vietnam": "VNM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}
OTHER = {"Other", "Other countries"}


def main() -> None:
    rows = []
    for family, measure, repo, fname, col, mult, unit, year in TABLES:
        with open(SERIES / repo / "data" / fname, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = r["country"].strip()
                if name not in ISO3 and name not in OTHER:
                    raise SystemExit(f"{repo}/{fname}: no ISO3 code for {name!r}; add it to ISO3")
                rows.append({
                    "family": family, "measure": measure, "country": name,
                    "iso3": ISO3.get(name, ""), "tonnes": round(float(r[col]) * mult, 3),
                    # the source's own world share; not recomputed, since the tables list only
                    # the leading producers and their sum is not the world total
                    "share_pct": r.get("share_pct", ""),
                    "unit": unit, "year": year, "source": SOURCE.format(repo=repo),
                })
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
