
"""Generate a synthetic POS transaction export and matching category control.

The generator reads no transaction files and uses only invented customer IDs,
products, prices, purchasing preferences and behavioral probabilities.
Requires ``generate_synthetic_retail.py`` alongside this file for its simulation
functions. The two outputs are TAB-DELIMITED text with ``.xls`` filenames,
matching the legacy ingestion format expected by main.py:

* ``fastmag (2).xls``: raw transaction-line source for main.py.
* ``fastmag_familles_total.xls``: totals grouped from the *same* synthetic
  source for comparaison_familles.py; not independent accounting evidence.

Default reporting period: 2024-01-01 through 2026-06-16.
The random seed makes the simulation reproducible. Existing input files are
not overwritten; select a fresh output directory to regenerate them.

Example:
    python generate_raw_fastmag.py --output-dir synthetic_raw --customers 1400 --seed 42
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from generate_synthetic_retail import (
    catalog,
    customer_documents,
    make_staged,
)

# Columns produced by main.py, not present in the raw export.
DERIVED_COLUMNS = {
    "Line_Quantite",
    "Line_Type",
    "PrixAchat_ILS",
    "Clean_Date",
    "Day_Of_Week",
    "Final_Unit_Price_HT",
    "Final_Total_Line_HT",
}

# The synthetic reference mimics a category report that omits system lines and adjustments.
REFERENCE_EXCLUSIONS = {
    "~SHIPPING",
    "~COUPON",
    "~GIFTCARD",
    "~AVOIR",
    "~BOUTON",
}


def generate_raw(output_dir, customers, seed):
    output_dir = Path(output_dir).resolve()

    raw_file = output_dir / "fastmag (2).xls"
    reference_file = output_dir / "fastmag_familles_total.xls"

    # Never overwrite existing inputs accidentally.
    if raw_file.exists() or reference_file.exists():
        raise FileExistsError(
            "Synthetic raw files already exist. "
            "Use a new, empty output directory."
        )

    rng = np.random.default_rng(seed)


    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2026-06-16")


    # 1. Generate an entirely fictional product catalog.
    _, products_by_family = catalog(rng)

    # 2. Simulate customer purchases, returns and exchanges.
    documents = customer_documents(
        rng,
        customers,
        start,
        end,
        products_by_family,
    )

    # 3. Construct transaction lines internally.
    # This is used only to obtain consistent simulated records.
    simulated = make_staged(documents, start)

    # 4. Remove all columns that main.py must calculate.
    raw = simulated.drop(
        columns=list(DERIVED_COLUMNS)
    ).rename(
        columns={"Basket_Total_Items": "Quantite"}
    )

    # 5. Verify transaction keys.
    if raw.duplicated(["Vente", "Ligne"]).any():
        raise ValueError("Duplicate transaction-line identifiers")

    if raw[["Vente", "Ligne", "Client", "StockID"]].isna().any().any():
        raise ValueError("Missing required transaction identifiers")

    # 6. Verify accounting amounts.
    calculated_total = (
        raw["Prix"]
        * (1 - raw["Remise"] / 100)
        * raw["Quantite_1"]
    )

    if not np.allclose(
        calculated_total,
        raw["Total_1"],
        atol=0.02,
        rtol=1e-6,
    ):
        raise ValueError("Raw transaction amounts do not reconcile")

    # 7. Verify that purchases and returns exist.
    if not raw["Quantite_1"].gt(0).any():
        raise ValueError("No positive purchases generated")

    if not raw["Quantite_1"].lt(0).any():
        raise ValueError("No returns generated")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Export actual input format used by main.py:
    # tab-delimited text with a legacy .xls filename.
    raw.to_csv(
        raw_file,
        sep="\t",
        index=False,
        encoding="latin1",
    )

    # 9. Derive reference totals from the SAME fictional source, with a
    #    separate family aggregation for a reproducible reconciliation test.
    reference_rows = raw.loc[
        ~raw["Famille"].isin(REFERENCE_EXCLUSIONS)
    ]

    reference = (
        reference_rows
        .groupby("Famille", as_index=False)
        .agg(
            Total=("Total_1", "sum"),
            Quantite=("Quantite_1", "sum"),
        )
        .rename(columns={"Famille": "Category"})
    )

    reference.to_csv(
        reference_file,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(f"Synthetic reporting period: {start.date()} through {end.date()}")
    print("Synthetic raw files generated:")
    print(raw_file)
    print(reference_file)
    print(f"Customers: {raw['Client'].nunique():,}")
    print(f"Transactions: {raw['Vente'].nunique():,}")
    print(f"Transaction lines: {len(raw):,}")
    print(f"Product families: {raw['Famille'].nunique():,}")
    print("Raw financial checks: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="synthetic_raw",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=1400,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.customers < 100:
        parser.error("Use at least 100 customers")

    generate_raw(
        args.output_dir,
        args.customers,
        args.seed,
    )
