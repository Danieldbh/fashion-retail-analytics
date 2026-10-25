"""Audit estimated net merchandise margin by date, category and sales channel.

Input: ``fastmag_staged_v2.csv``. Sales and returns retain their source signs;
nonmerchandise lines are excluded. Cost estimates use the approximate 4.0
EUR-to-ILS factor computed during staging. Aggregations are checked against
the transaction-line total. ``--export PATH`` optionally writes daily audit
groups; those aggregates must not also be imported as a second Power BI fact.

Examples:
    python sales_by_famille.py
    python sales_by_famille.py --export family_margin_daily.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STAGED_FILE = Path("fastmag_staged_v2.csv")
MERCHANDISE_LINE_TYPES = {"ARTICLE", "RETURN"}
REQUIRED_COLUMNS = {
    "Famille", "Line_Type", "Clean_Date", "Vendeur", "Line_Quantite",
    "Final_Total_Line_HT", "PrixAchat_ILS",
}


def build_margin_audit(staged_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not staged_file.is_file():
        raise FileNotFoundError(f"Missing staged input: {staged_file}")

    df = pd.read_csv(staged_file, low_memory=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required staged columns: {sorted(missing)}")

    if {"Vente", "Ligne"}.issubset(df.columns):
        duplicates = df.duplicated(["Vente", "Ligne"]).sum()
        if duplicates:
            raise ValueError(f"Found {duplicates} duplicate (Vente, Ligne) transaction keys")

    merchandise = df.loc[df["Line_Type"].isin(MERCHANDISE_LINE_TYPES)].copy()
    if merchandise.empty:
        raise ValueError("No ARTICLE / RETURN merchandise lines found")

    if merchandise[["Famille", "Vendeur", "Line_Quantite", "Final_Total_Line_HT", "PrixAchat_ILS"]].isna().any().any():
        raise ValueError("Missing values in merchandise grouping or margin columns")

    merchandise["Clean_Date"] = pd.to_datetime(merchandise["Clean_Date"], errors="raise")
    if merchandise["Clean_Date"].isna().any():
        raise ValueError("Missing merchandise transaction dates")

    for name in ("Line_Quantite", "Final_Total_Line_HT", "PrixAchat_ILS"):
        merchandise[name] = pd.to_numeric(merchandise[name], errors="raise")
        if not np.isfinite(merchandise[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {name}")

    # Channel definition: only seller WEB is the Web channel.
    # All other sellers belong to the physical channel. Do not duplicate rows
    # to create an 'Omnichannel' category (it would double count in BI).
    normalized_seller = merchandise["Vendeur"].astype(str).str.strip().str.upper()
    if normalized_seller.eq("").any():
        raise ValueError("Blank merchandise seller/channel values")
    merchandise["Channel"] = np.where(normalized_seller.eq("WEB"), "Web", "Physical")

    # Calculate NET estimated merchandise margin AFTER signed sales/returns.
    # Respect accounting signs already present in Final_Total_Line_HT and
    # Line_Quantite, including positive entries classified as RETURN.
    # We do not force all RETURN lines negative: some are positive exchanges.
    merchandise["Net_Revenue_HT"] = merchandise["Final_Total_Line_HT"]
    merchandise["Signed_Estimated_Cost_ILS"] = (
        merchandise["PrixAchat_ILS"] * merchandise["Line_Quantite"]
    )
    merchandise["Estimated_Net_Margin_ILS"] = (
        merchandise["Net_Revenue_HT"] - merchandise["Signed_Estimated_Cost_ILS"]
    )

    # DATE + CATEGORY + CHANNEL grain, WITHOUT a duplicate Omnichannel total.
    # Preserve full precision; Power BI should calculate dynamic Top N from
    # the base fact table rather than loading this optional audit export.
    daily = merchandise.groupby(
        ["Clean_Date", "Famille", "Channel"], as_index=False
    ).agg(
        Net_Quantity=("Line_Quantite", "sum"),
        Net_Revenue_HT=("Net_Revenue_HT", "sum"),
        Signed_Estimated_Cost_ILS=("Signed_Estimated_Cost_ILS", "sum"),
        Estimated_Net_Margin_ILS=("Estimated_Net_Margin_ILS", "sum"),
        Line_Count=("Line_Type", "size"),
    )

    by_family = daily.groupby("Famille", as_index=False).agg(
        Net_Quantity=("Net_Quantity", "sum"),
        Net_Revenue_HT=("Net_Revenue_HT", "sum"),
        Signed_Estimated_Cost_ILS=("Signed_Estimated_Cost_ILS", "sum"),
        Estimated_Net_Margin_ILS=("Estimated_Net_Margin_ILS", "sum"),
    ).sort_values("Estimated_Net_Margin_ILS", ascending=False)

    # Guard against aggregation mistakes.
    if not np.isclose(
        daily["Estimated_Net_Margin_ILS"].sum(),
        merchandise["Estimated_Net_Margin_ILS"].sum(),
        atol=0.01,
        rtol=0,
    ):
        raise AssertionError("Aggregated margin fails line-level reconciliation")
    return daily, by_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, default=STAGED_FILE)
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Optional daily family/channel audit CSV (not an additional Power BI fact table)",
    )
    args = parser.parse_args()

    daily, by_family = build_margin_audit(args.staged)
    print("=== ESTIMATED NET MERCHANDISE MARGIN (AFTER RETURNS) ===")
    print(f"Source: {args.staged}")
    print(f"Included product families: {len(by_family):,}")
    print(f"Daily family/channel audit groups: {len(daily):,}")
    print(f"Net revenue HT: {daily['Net_Revenue_HT'].sum():,.2f} ILS")
    print(f"Signed estimated cost: {daily['Signed_Estimated_Cost_ILS'].sum():,.2f} ILS")
    print(f"Estimated net margin: {daily['Estimated_Net_Margin_ILS'].sum():,.2f} ILS")
    print("\nTop 10 product families for reference only (not pre-filtered for Power BI):")
    print(by_family[["Famille", "Estimated_Net_Margin_ILS"]].head(10).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nNote: costs rely on an approximate EUR-to-ILS conversion factor (4.0).")
    print("~BOUTON is included. Signed returns/exchanges are included as recorded.")
    print("Power BI should calculate margin and Top N from fact_sales_items dynamically.")

    if args.export is not None:
        daily.to_csv(args.export, index=False)
        print(f"\nDaily margin audit exported: {args.export}")
    else:
        print("No audit CSV requested (use --export PATH to save the daily groups).")


if __name__ == "__main__":
    main()
