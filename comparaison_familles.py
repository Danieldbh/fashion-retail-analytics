"""Reconcile staged retail transactions against category-level control totals.

Inputs: ``fastmag_familles_total.xls`` (tab-delimited text despite its name)
and ``fastmag_staged_v2.csv``. Compare signed TTC amounts (including VAT)
and quantities by family, not HT revenue. Shipping, coupons and gift cards
are excluded from both sides; ~AVOIR and ~BOUTON may exist only in staging.

For a synthetic dataset, the control file can be generated from the same
fictional source: reconciliation then tests transformation consistency,
not accuracy against an independent external accounting system.
A mismatch produces a nonzero exit status.

Example: python comparaison_familles.py
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STAGED_FILE = Path("fastmag_staged_v2.csv")
REFERENCE_FILE = Path("fastmag_familles_total.xls")
EXCLUDED_SYSTEM_FAMILIES = {"~SHIPPING", "~COUPON", "~GIFTCARD"}
EXPECTED_STAGED_ONLY_FAMILIES = {"~AVOIR", "~BOUTON"}
AMOUNT_TOLERANCE_ILS = 0.01  # A one-agorot category-level comparison tolerance.


def load_reference(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference report: {path}")

    # This .xls file is tab-delimited text; Excel binary parsing would be wrong.
    ref = pd.read_csv(path, sep="\t")
    names = {"Category", "Total", "Quantite"}
    if not names.issubset(ref.columns):
        raise ValueError(f"Reference missing columns: {sorted(names - set(ref.columns))}")
    ref = ref.rename(columns={
        "Category": "Famille", "Total": "Total_Reference_TTC", "Quantite": "Qty_Reference",
    })[["Famille", "Total_Reference_TTC", "Qty_Reference"]].copy()
    if ref.isna().any().any():
        raise ValueError("Reference report contains missing category, amount or quantity")
    if ref["Famille"].duplicated().any():
        raise ValueError("Reference report has duplicate Famille keys")
    ref["Total_Reference_TTC"] = pd.to_numeric(ref["Total_Reference_TTC"], errors="raise")
    ref["Qty_Reference"] = pd.to_numeric(ref["Qty_Reference"], errors="raise")
    if not np.isfinite(ref[["Total_Reference_TTC", "Qty_Reference"]].to_numpy()).all():
        raise ValueError("Reference contains non-finite amounts or quantities")
    return ref


def load_staged(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing staged input: {path}")

    staged = pd.read_csv(path, low_memory=False)
    required = {"Famille", "Total_1", "Line_Quantite"}
    if not required.issubset(staged.columns):
        raise ValueError(f"Staged file missing columns: {sorted(required - set(staged.columns))}")
    if {"Vente", "Ligne"}.issubset(staged.columns):
        duplicates = staged.duplicated(["Vente", "Ligne"]).sum()
        if duplicates:
            raise ValueError(f"Found {duplicates} duplicate (Vente, Ligne) transaction keys")

    staged = staged.loc[~staged["Famille"].isin(EXCLUDED_SYSTEM_FAMILIES)].copy()
    if staged[["Famille", "Total_1", "Line_Quantite"]].isna().any().any():
        raise ValueError("Staged file has missing categories, amount or quantity")
    for col in ("Total_1", "Line_Quantite"):
        staged[col] = pd.to_numeric(staged[col], errors="raise")
        if not np.isfinite(staged[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Staged file contains non-finite {col}")

    # Preserve signed quantities and amounts, including returns and adjustments.
    return staged.groupby("Famille", as_index=False).agg(
        Total_Staged_TTC=("Total_1", "sum"),
        Qty_Staged=("Line_Quantite", "sum"),
    )


def reconcile(ref: pd.DataFrame, staged: pd.DataFrame) -> bool:
    merged = ref.merge(staged, on="Famille", how="outer", indicator=True, validate="one_to_one")
    missing_in_staged = sorted(merged.loc[merged["_merge"].eq("left_only"), "Famille"].tolist())
    staged_only = merged.loc[merged["_merge"].eq("right_only")].copy()
    unexpected_staged = sorted(set(staged_only["Famille"]) - EXPECTED_STAGED_ONLY_FAMILIES)

    matched = merged.loc[merged["_merge"].eq("both")].copy()
    matched["Sales_Diff_TTC"] = (matched["Total_Reference_TTC"] - matched["Total_Staged_TTC"]).round(2)
    matched["Qty_Diff"] = matched["Qty_Reference"] - matched["Qty_Staged"]
    differences = matched.loc[
        matched["Sales_Diff_TTC"].abs().gt(AMOUNT_TOLERANCE_ILS)
        | matched["Qty_Diff"].ne(0)
    ].copy()

    print("=== FASTMAG FAMILY RECONCILIATION — TTC AND SIGNED QUANTITIES ===")
    print(f"Reference families: {len(ref)} | Staged families: {len(staged)} | Matched: {len(matched)}")
    print(f"Reference families missing in staged: {len(missing_in_staged)}")
    print(f"Unexpected staged-only families: {len(unexpected_staged)}")
    print(f"Matched families with differences: {len(differences)}")
    print(f"Max absolute TTC difference (matched): {matched['Sales_Diff_TTC'].abs().max():,.2f} ILS")
    print(f"Max absolute quantity difference (matched): {matched['Qty_Diff'].abs().max():,.0f}")

    if not staged_only.empty:
        print("\nStaged-only categories (not in reference report):")
        print(staged_only[["Famille", "Total_Staged_TTC", "Qty_Staged"]].to_string(index=False))
        print("Expected exceptions: ~AVOIR and ~BOUTON (retained, not silently removed).")
    if missing_in_staged:
        print("\nFAIL — reference families missing in staged:", missing_in_staged)
    if unexpected_staged:
        print("\nFAIL — unexpected staged-only families:", unexpected_staged)
    if not differences.empty:
        print("\nFAIL — mismatching matched families:")
        print(differences[["Famille", "Total_Reference_TTC", "Total_Staged_TTC", "Sales_Diff_TTC", "Qty_Reference", "Qty_Staged", "Qty_Diff"]].to_string(index=False))

    passed = not missing_in_staged and not unexpected_staged and differences.empty
    print("\nPASS: all shared family totals and quantities reconcile." if passed
          else "\nFAIL: family reconciliation needs investigation.")
    return bool(passed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, default=STAGED_FILE)
    parser.add_argument("--reference", type=Path, default=REFERENCE_FILE)
    args = parser.parse_args()
    passed = reconcile(load_reference(args.reference), load_staged(args.staged))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
