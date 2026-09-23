"""Renders every dashboard page for every material, headless, and checks nothing breaks.

The engine tests check the numbers; this checks the app that shows them. Each of the thirteen pages
is run once per material choice (eight, counting copper once per system) through Streamlit's own
AppTest harness, and the check fails on any uncaught exception, any st.error box, or a page that
opens with the wrong title. It also confirms the sidebar's material picker drives the page: the
overview's title must name the material and system that were picked.

Takes a minute or two: the first run fits every system and loads every commodity's trade data.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.util import calc_hash  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = {
    "overview": None,   # title depends on the material, checked separately below
    "lifetimes": "How long each unit lasts",
    "stocks-and-flows": "Stocks and flows",
    "end-of-life": "What comes back out",
    "scenarios": None,  # "Scenarios", "Deployment scenarios" or "Capacity scenarios" by system
    "supply-map": "Where it comes from, and where it goes",
    "disruption": "Disruption and recovery",
    "cascade": "How the shock spreads",
    "diversification": "What it would take to diversify",
    "copper": "One material, three technologies: copper",
    "all-materials": "Which recovery pathway matters most",
    "glossary": "Terms and formulas",
    "ask": "Ask the data",
    "methods": "Methods and data",
}
MATERIALS = [
    ("Copper", "Wind", "Copper in Danish wind turbines"),
    ("Copper", "EVs", "Copper in the global EV fleet"),
    ("Copper", "Data centres", "Copper in global data centres"),
    ("Rare earths", None, "Rare earth metals in Danish wind turbines"),
    ("Lithium", None, "Lithium in the global EV fleet"),
    ("Nickel", None, "Nickel in the global EV fleet"),
    ("Cobalt", None, "Cobalt in the global EV fleet"),
    ("Graphite", None, "Graphite in the global EV fleet"),
]


# Page url -> the key its "How to read this page" button uses.
GUIDE_KEYS = {
    "overview": "overview", "lifetimes": "lifetimes", "stocks-and-flows": "stocks", "end-of-life": "eol",
    "scenarios": "scenarios", "supply-map": "supplymap", "disruption": "disruption", "cascade": "cascade",
    "diversification": "diversification", "copper": "copper", "all-materials": "materials", "ask": "ask",
    "methods": "methods", "glossary": "glossary",
}


def page_title(at) -> str | None:
    for element in at.get("html"):
        body = getattr(element.proto, "body", "")
        if "<h1>" in body:
            return body.split("<h1>", 1)[1].split("</h1>", 1)[0]
    return None


def main():
    checks = []
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    for family, found_in, overview_title in MATERIALS:
        at.button_group(key="material").set_value(family).run()
        if found_in:
            at.button_group(key="in").set_value(found_in).run()
        label = f"{family}{' / ' + found_in if found_in else ''}"
        for url, want in PAGES.items():
            at._page_hash = calc_hash(url)
            at.run()
            problems = [e.value for e in at.exception] + [e.value for e in at.error]
            title = page_title(at)
            expected = overview_title if url == "overview" else want
            title_ok = title is not None and (expected is None or title == expected)
            checks.append((f"{label:<22} {url:<18} renders cleanly" + (f" ({problems[0][:60]})" if problems else ""),
                           not problems and title_ok))
            if family == "Copper" and found_in == "Wind":
                # the guide dialog, once per page: it must open, and hold the four guide sections
                at.button(key=f"guide_{GUIDE_KEYS[url]}").click().run()
                text = " ".join(m.value for m in at.markdown)
                dialog_ok = (not at.exception and all(h in text for h in (
                    "What it shows.", "How to read it.", "Method, in brief.", "Try this.")))
                checks.append((f"{label:<22} {url:<18} guide dialog opens", dialog_ok))

    print(f"{'check':<100}{'status':>8}")
    ok_all = True
    for name, ok in checks:
        ok_all &= bool(ok)
        print(f"{name:<100}{'OK' if ok else 'FAIL':>8}")
    print()
    print(f"All {len(checks)} page renders pass." if ok_all else "FAIL: a dashboard page is broken.")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
