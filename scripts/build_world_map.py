"""Build data/world_map.json: the world map the Supply map page draws, projected once here.

The browser only animates; it never projects geometry or loads a mapping library. This script
downloads Natural Earth's admin-0 country outlines (public domain, via the nvkelso/natural-earth-
vector repository), projects them with the Equal Earth projection (Savric, Patterson and Jenny
2018, an equal-area projection, so a country's drawn size is comparable to any other's), and
writes SVG path strings plus one anchor point per country.

Outlines come from the 1:110m set (light, enough detail at dashboard size). Anchor points, where
production circles sit and trade arcs start and end, come from Natural Earth's own label points
in the 1:50m set, which also covers small trade hubs the 1:110m set leaves out entirely
(Singapore, Hong Kong, Malta, Bahrain and others). Antarctica is dropped and the map is cropped
to the inhabited latitudes.

Run from the repo root:  python scripts/build_world_map.py
"""
from __future__ import annotations

import json
import math
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "world_map.json"
BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
SETS = {"110m": "ne_110m_admin_0_countries.geojson", "50m": "ne_50m_admin_0_countries.geojson"}

WIDTH = 1000.0                   # the full 360 degrees of longitude spans this many SVG units
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
M = math.sqrt(3) / 2
X_MAX = math.pi / (M * A1)       # Equal Earth x at longitude 180, latitude 0
SCALE = (WIDTH / 2) / X_MAX
LAT_MIN, LAT_MAX = -57.0, 84.0   # crop: southern tip of South America to northern Greenland


def equal_earth(lon: float, lat: float) -> tuple[float, float]:
    lam, phi = math.radians(lon), math.radians(lat)
    theta = math.asin(M * math.sin(phi))
    t2, t6 = theta * theta, theta ** 6
    x = lam * math.cos(theta) / (M * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)))
    y = theta * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return x * SCALE, -y * SCALE


_, Y_TOP = equal_earth(0, LAT_MAX)
_, Y_BOTTOM = equal_earth(0, LAT_MIN)


def to_svg(lon: float, lat: float) -> tuple[float, float]:
    x, y = equal_earth(lon, lat)
    return round(x + WIDTH / 2, 1), round(y - Y_TOP, 1)


def ring_path(ring: list[list[float]]) -> str:
    pts, last = [], None
    for lon, lat in ring:
        p = to_svg(lon, lat)
        if last is None or abs(p[0] - last[0]) + abs(p[1] - last[1]) >= 0.35:
            pts.append(p)
            last = p
    if len(pts) < 3:
        return ""
    return "M" + "L".join(f"{x:g},{y:g}" for x, y in pts) + "Z"


def geometry_path(geom: dict) -> str:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    return "".join(ring_path(ring) for poly in polys for ring in poly)


def fetch(name: str, cache: Path) -> dict:
    path = cache / name
    if not path.exists():
        urllib.request.urlretrieve(BASE + name, path)
    return json.loads(path.read_text(encoding="utf-8"))


def codes(props: dict) -> list[str]:
    """Every three-letter code a country is known by. Natural Earth's ISO_A3 is -99 for a few
    countries (France and Norway among them); ADM0_A3 and ISO_A3_EH cover those."""
    out = []
    for key in ("ADM0_A3", "ISO_A3_EH", "ISO_A3", "SOV_A3"):
        v = props.get(key)
        if v and v != "-99" and v not in out:
            out.append(v)
    return out[:3]


def main() -> None:
    cache = Path(tempfile.gettempdir()) / "mfa-engine-natural-earth"
    cache.mkdir(exist_ok=True)
    small, detailed = fetch(SETS["110m"], cache), fetch(SETS["50m"], cache)

    countries, anchors, names = [], {}, {}
    for feat in detailed["features"]:
        p = feat["properties"]
        if p.get("LABEL_X") is None:
            continue
        for code in codes(p):
            anchors.setdefault(code, to_svg(p["LABEL_X"], p["LABEL_Y"]))
            names.setdefault(code, p.get("NAME_EN") or p.get("NAME"))
    for feat in small["features"]:
        p = feat["properties"]
        if p.get("ADM0_A3") == "ATA":
            continue
        d = geometry_path(feat["geometry"])
        if d:
            countries.append({"codes": codes(p), "name": p.get("NAME_EN") or p.get("NAME"), "d": d})

    graticule = []
    for lon in range(-180, 181, 30):
        graticule.append("M" + "L".join(f"{x:g},{y:g}" for x, y in
                                         (to_svg(lon, lat) for lat in range(int(LAT_MIN), int(LAT_MAX) + 1, 2))))
    for lat in (-30, 0, 30, 60):
        graticule.append("M" + "L".join(f"{x:g},{y:g}" for x, y in
                                         (to_svg(lon, lat) for lon in range(-180, 181, 3))))

    out = {
        "source": "Natural Earth admin-0 countries (public domain): 1:110m outlines, 1:50m label points. "
                  "Equal Earth projection (Savric, Patterson and Jenny 2018).",
        "width": WIDTH, "height": round(Y_BOTTOM - Y_TOP, 1),
        "graticule": graticule, "countries": countries,
        "anchors": {k: list(v) for k, v in anchors.items()}, "names": names,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{OUT}: {len(countries)} outlines, {len(anchors)} anchor codes, "
          f"{OUT.stat().st_size / 1024:.0f} KB, viewBox 0 0 {WIDTH:g} {out['height']:g}")


if __name__ == "__main__":
    main()
