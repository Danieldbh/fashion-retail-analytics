"""Create returning-customer features and a historical 365-day churn label.

Input: ``fastmag_staged_v2.csv``; output:
``customer_churn_features_returning_v2.csv``.

Only customers with at least two qualifying purchase invoices before the
exclusive historical cutoff enter training. A label of 1 means the customer
made no qualifying purchase in the following 365 days; 0 means they did.
The same feature builders are imported by Churn_Model.py for current scoring,
so training and inference use identical definitions and encoding rules.
Importing this module does not execute the export.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SHARED FEATURE ENGINEERING
# ---------------------------------------------------------------------------

CATEGORY_COLUMNS = ["First_Category", "Favorite_Category"]
MIN_RETURNING_PURCHASES = 2


def prepare_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Prepare eligible merchandise lines using the existing Line_Type rules."""
    required = {
        "Line_Type", "Clean_Date", "Client", "Famille", "Final_Total_Line_HT",
        "Remise", "Line_Quantite", "Prix", "TVA", "Nature", "NumeroFacture",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw.loc[raw["Line_Type"].isin(["ARTICLE", "RETURN"])].copy()
    df["Clean_Date"] = pd.to_datetime(df["Clean_Date"], errors="raise")
    df = df.dropna(subset=["Client"]).copy()
    df["Famille"] = df["Famille"].fillna("UNKNOWN").astype(str)

    revenue = pd.to_numeric(df["Final_Total_Line_HT"], errors="raise")
    qty = pd.to_numeric(df["Line_Quantite"], errors="raise")
    pct = pd.to_numeric(df["Remise"], errors="raise").clip(lower=0) / 100
    price = pd.to_numeric(df["Prix"], errors="raise")
    vat_factor = 1 + pd.to_numeric(df["TVA"], errors="raise") / 100
    if (vat_factor <= 0).any():
        raise ValueError("Invalid VAT factor")

    df["Gross_Sales_HT"] = np.where(revenue > 0, revenue, 0)
    df["Returns_HT"] = np.where(revenue < 0, abs(revenue), 0)
    # Discounts below zero do not count as customer discounts
    # received; return quantities are excluded from the amount received.
    df["Discount_Value"] = price * qty.clip(lower=0) * pct / vat_factor
    df["Is_Qualifying_Purchase"] = (
        df["Nature"].fillna("").astype(str).str.strip().eq("VENTE")
        & qty.gt(0)
        & revenue.gt(0)
    )
    return df


def build_customer_profiles(prepared: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """Build customer features using transactions STRICTLY BEFORE snapshot_date.

    snapshot_date is an exclusive cutoff. Purchase activity includes qualifying
    exchanges, and negative merchandise lines contribute to return features.
    """
    snapshot_date = pd.Timestamp(snapshot_date)
    df_history = prepared.loc[prepared["Clean_Date"] < snapshot_date]
    purchased = df_history.loc[df_history["Is_Qualifying_Purchase"]].copy()
    if purchased.empty:
        raise ValueError("No qualifying historical purchases before snapshot")

    profiles = purchased.groupby("Client").agg(
        First_Purchase_Date=("Clean_Date", "min"),
        Last_Purchase_Date=("Clean_Date", "max"),
        Frequency_Total_Visits=("NumeroFacture", "nunique"),
        Total_Gross_Spend=("Gross_Sales_HT", "sum"),
        Total_Discount_Received=("Discount_Value", "sum"),
    ).reset_index()

    returns = df_history.groupby("Client").agg(
        Total_Returns=("Returns_HT", "sum")
    ).reset_index()
    profiles = profiles.merge(returns, on="Client", how="left", validate="one_to_one")

    profiles["Recency_Days"] = (snapshot_date - profiles["Last_Purchase_Date"]).dt.days
    profiles["Tenure_Days"] = (snapshot_date - profiles["First_Purchase_Date"]).dt.days
    profiles["Avg_Days_Between_Purchases"] = np.where(
        profiles["Frequency_Total_Visits"] > 1,
        (profiles["Tenure_Days"] - profiles["Recency_Days"])
        / (profiles["Frequency_Total_Visits"] - 1),
        -1,
    )
    profiles["Average_Order_Value"] = (
        profiles["Total_Gross_Spend"] / profiles["Frequency_Total_Visits"]
    )
    profiles["Return_Rate_%"] = np.where(
        profiles["Total_Gross_Spend"] > 0,
        profiles["Total_Returns"] / profiles["Total_Gross_Spend"],
        0,
    )
    profiles["Discount_Hunter_Ratio_%"] = np.where(
        profiles["Total_Gross_Spend"] > 0,
        profiles["Total_Discount_Received"]
        / (profiles["Total_Gross_Spend"] + profiles["Total_Discount_Received"]),
        0,
    )

    recent = purchased.loc[
        purchased["Clean_Date"] >= snapshot_date - pd.Timedelta(days=90)
    ]
    recent_stats = recent.groupby("Client").agg(
        Spend_Last_90d=("Gross_Sales_HT", "sum"),
        Visits_Last_90d=("NumeroFacture", "nunique"),
    ).reset_index()
    profiles = profiles.merge(recent_stats, on="Client", how="left", validate="one_to_one")
    profiles[["Spend_Last_90d", "Visits_Last_90d"]] = (
        profiles[["Spend_Last_90d", "Visits_Last_90d"]].fillna(0)
    )

    first_purchases = (
        purchased.sort_values("Clean_Date")
        .groupby("Client").first().reset_index()
    )
    first_purchases["First_Purchase_Month"] = (
        first_purchases["Clean_Date"].dt.month
    )
    first_purchases = first_purchases[
        ["Client", "First_Purchase_Month", "Famille"]
    ].rename(columns={"Famille": "First_Category"})
    profiles = profiles.merge(first_purchases, on="Client", how="left", validate="one_to_one")

    spend_by_cat = (
        purchased.groupby(["Client", "Famille"])["Gross_Sales_HT"]
        .sum().reset_index()
    )
    fav_cat = (
        spend_by_cat.sort_values(
            ["Client", "Gross_Sales_HT"], ascending=[True, False]
        ).groupby("Client").first().reset_index()
    )
    fav_cat = fav_cat[["Client", "Famille"]].rename(
        columns={"Famille": "Favorite_Category"}
    )
    profiles = profiles.merge(fav_cat, on="Client", how="left", validate="one_to_one")
    return profiles


def filter_returning_customers(profiles: pd.DataFrame) -> pd.DataFrame:
    """Keep clients with >=2 distinct qualifying purchase invoices at this snapshot.

    Call only after build_customer_profiles(): the visit count is calculated
    from eligible purchases strictly BEFORE the given snapshot cutoff.
    """
    if "Frequency_Total_Visits" not in profiles.columns:
        raise ValueError("Missing historical qualifying visit count")
    return profiles.loc[
        profiles["Frequency_Total_Visits"] >= MIN_RETURNING_PURCHASES
    ].copy()


def encode_customer_profiles(profiles: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    """One-hot encode categories and optionally align with training columns.

    Client is retained so callers can merge predictions back safely. Never
    include it as a model feature. Unseen categorical values become all-zero
    category indicators when aligned to the training feature columns.
    """
    encoded = pd.get_dummies(profiles, columns=CATEGORY_COLUMNS, drop_first=False)
    encoded = encoded.drop(columns=["First_Purchase_Date", "Last_Purchase_Date"])
    if feature_columns is not None:
        features = encoded.drop(columns=["Client"])
        features = features.reindex(columns=feature_columns, fill_value=0)
        return pd.concat([encoded[["Client"]].reset_index(drop=True),
                          features.reset_index(drop=True)], axis=1)
    return encoded


# ---------------------------------------------------------------------------
# HISTORICAL TRAINING-DATA EXPORT
# ---------------------------------------------------------------------------

INPUT_FILE = "fastmag_staged_v2.csv"
OUTPUT_FILE = "customer_churn_features_returning_v2.csv"

def main():
    prepared = prepare_transactions(pd.read_csv(INPUT_FILE))
    # Use the day AFTER the last observed calendar day as an exclusive end.
    # Thus [snapshot_date, snapshot_date + 365 days) is fully observed,
    # ensuring that the historical label covers 365 observed calendar days.
    last_boundary = prepared["Clean_Date"].max().normalize() + pd.Timedelta(days=1)
    snapshot_date = last_boundary - pd.Timedelta(days=365)

    # The same feature builder will be used for current-customer inference.
    profiles = build_customer_profiles(prepared, snapshot_date)
    profiles = filter_returning_customers(profiles)
    if profiles.empty:
        raise ValueError("No returning customers at historical cutoff")
    features = encode_customer_profiles(profiles)

    future_purchasers = prepared.loc[
        (prepared["Clean_Date"] >= snapshot_date)
        & (prepared["Clean_Date"] < snapshot_date + pd.Timedelta(days=365))
        & prepared["Is_Qualifying_Purchase"],
        "Client",
    ].unique()
    features["Churn_Label"] = np.where(features["Client"].isin(future_purchasers), 0, 1)

    if not features["Client"].is_unique:
        raise ValueError("Expected exactly one feature row per customer")
    if not features["Churn_Label"].isin([0, 1]).all():
        raise ValueError("Invalid churn labels")

    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Created {len(features):,} returning-customer historical profiles (2+ visits).")
    print(f"Feature columns (excluding Client and label): {len(features.columns) - 2}")
    print(f"Saved: {OUTPUT_FILE}")
    print("Eligibility/features: purchases before cutoff; churn label: next 365 days.")


if __name__ == "__main__":
    main()
