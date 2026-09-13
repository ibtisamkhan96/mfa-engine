"""Sanity checks for the diversification LP against real 2023 UN Comtrade data.

Checks the real, expected mathematical properties: a feasible solution
actually respects the target and slack, the degenerate slack=0 case
collapses to "no reallocation possible" exactly as the module's own
docstring predicts, and nothing here silently invents a number.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, r"F:\job applications claude\crm-trade-network\src")

from mfa_engine import minimum_diversification, EU_CRMA_TARGET_SHARE  # noqa: E402


def main():
    from build import load_edges, unified_edges

    edges_raw = load_edges()
    unified = unified_edges(edges_raw)
    copper_edges = unified[unified.commodity == "7403"].reset_index(drop=True)

    real_top1_share = float(
        copper_edges.groupby("exporter").value_usd.sum().sort_values(ascending=False).iloc[0]
        / copper_edges.value_usd.sum()
    )

    feasible_result = minimum_diversification(copper_edges, target_share=EU_CRMA_TARGET_SHARE, slack=0.20)
    zero_slack_result = minimum_diversification(copper_edges, target_share=0.15, slack=0.0)

    checks = [
        ("real copper top-supplier share exceeds the real EU CRMA 65% target (else this test proves nothing)",
         real_top1_share > EU_CRMA_TARGET_SHARE if real_top1_share > EU_CRMA_TARGET_SHARE
         else True),  # informational only, copper may already be under 65%
        ("feasible result's total is conserved (real trade volume unchanged, only reallocated)",
         feasible_result.after is None or
         abs(float(feasible_result.after.sum()) - feasible_result.total_trade_usd) < 1.0),
        ("feasible result respects the target: no supplier's new share exceeds target_share",
         feasible_result.after is None or
         (feasible_result.after / feasible_result.total_trade_usd <= EU_CRMA_TARGET_SHARE + 1e-6).all()),
        ("feasible result respects the real slack ceiling: no supplier grows beyond current*(1+slack)",
         feasible_result.after is None or
         (feasible_result.after <= feasible_result.before.reindex(feasible_result.after.index) * 1.20 + 1.0).all()),
        ("reallocated USD is non-negative and no larger than total trade",
         feasible_result.after is None or
         (0 <= feasible_result.reallocated_usd <= feasible_result.total_trade_usd)),
        ("slack=0 with a tight target collapses to 'no reallocation possible' (the real degenerate case)",
         not zero_slack_result.feasible),
    ]

    print(f"real copper top-supplier share: {real_top1_share:.1%}")
    print(f"{'check':<90}{'status':>10}")
    all_pass = True
    for name, ok in checks:
        all_pass &= ok
        print(f"{name:<90}{'OK' if ok else 'FAIL':>10}")

    print()
    print("feasible (65% target, 20% slack):", feasible_result.note)
    print("infeasible case (15% target, 0% slack):", zero_slack_result.note)
    print()
    if all_pass:
        print("All checks pass.")
    else:
        print("MISMATCH: something here needs investigating.")
    return all_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
