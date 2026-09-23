"""Recovered lithium must be valued as lithium carbonate equivalent, not at the carbonate price
per tonne of lithium metal.

The EV model counts lithium as metal content (cathode stoichiometry); lithium is priced as
lithium carbonate. One tonne of lithium is 73.89 / (2 x 6.94) = 5.32 tonnes of Li2CO3, so the
dollar value of recovered lithium is tonnes x carbonate price x 5.32. Before this was caught,
the factor was missing and every lithium dollar figure, and lithium's coverage of the supply
shortfall, was 5.32 times too low with no error anywhere. This renders the Disruption page for
lithium and checks the figure the page shows against that arithmetic, at the default price.
"""
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest  # noqa: E402
from streamlit.util import calc_hash  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")
LCE_PER_T_LI = 73.89 / (2 * 6.94)
PRICE = 32_694.0   # the app's default 2023 battery-grade lithium carbonate price, USD per tonne


def main():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.button_group(key="material").set_value("Lithium").run()
    at._page_hash = calc_hash("disruption")
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    text = " ".join(m.value for m in at.markdown)
    tonnes = float(re.search(r"about \*\*([\d,]+) t/yr\*\* of it actually", text).group(1).replace(",", ""))
    shown = next(m.value for m in at.metric if m.label.startswith("Avg. annual recovery"))
    shown_musd = float(re.search(r"\$([\d,.]+) M/yr", shown).group(1).replace(",", ""))
    expected_musd = tonnes * PRICE * LCE_PER_T_LI / 1e6
    without_factor = tonnes * PRICE / 1e6

    checks = [
        (f"LCE factor is 73.89 / 13.88 = {LCE_PER_T_LI:.3f}", abs(LCE_PER_T_LI - 5.323) < 0.001),
        (f"page shows ${shown_musd:,.1f} M/yr for {tonnes:,.0f} t/yr recovered; tonnes x price x 5.32 = "
         f"${expected_musd:,.1f} M/yr (rounding on the page allows 3%)",
         abs(shown_musd - expected_musd) <= 0.03 * expected_musd + 0.05),
        (f"and it is not the old, unconverted figure (${without_factor:,.1f} M/yr)",
         abs(shown_musd - without_factor) > 0.5 * without_factor),
    ]
    print(f"{'check':<118}{'status':>8}")
    ok_all = True
    for name, ok in checks:
        ok_all &= bool(ok)
        print(f"{name:<118}{'OK' if ok else 'FAIL':>8}")
    print()
    print("All checks pass." if ok_all else "FAIL: lithium is not valued as lithium carbonate equivalent.")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
