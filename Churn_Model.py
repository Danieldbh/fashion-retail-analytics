"""Fit fixed-parameter XGBoost Model B and score current returning customers.

Input: ``customer_churn_features_returning_v2.csv`` and
``fastmag_staged_v2.csv``. Output:
``BI_Customer_Retention_Returning_Tuned_v2.csv``.

Historical labels cover 365 days; a customer's current eligibility requires
at least two qualifying purchase invoices. The stored hyperparameters are
fixed for reproducibility rather than tuned on each new input dataset.
Scores are uncalibrated model estimates; Low (<40%), Medium (40%–<70%), and
High (>=70%) are descriptive dashboard bands, not validated interventions.
The risk bands are distinct from any diagnostic binary decision threshold.
The output contains customer-level records; 
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from Churn_Prep import (
    build_customer_profiles,
    encode_customer_profiles,
    filter_returning_customers,
    prepare_transactions,
)

FEATURE_FILE = "customer_churn_features_returning_v2.csv"
STAGED_FILE = "fastmag_staged_v2.csv"
OUTPUT_FILE = "BI_Customer_Retention_Returning_Tuned_v2.csv"


def assign_risk_tier(probability_percent):
    # Exploratory categories, not validated business decision thresholds.
    if probability_percent >= 70:
        return "High Risk (Critical)"
    if probability_percent >= 40:
        return "Medium Risk (Monitor)"
    return "Safe (Loyal)"


def main():
    history = pd.read_csv(FEATURE_FILE)
    X_train = history.drop(columns=["Client", "Churn_Label"])
    y_train = history["Churn_Label"].astype(int)
    if not (history["Frequency_Total_Visits"] >= 2).all():
        raise ValueError("Historical features include customers with <2 qualifying visits")
    if y_train.nunique() != 2:
        raise ValueError("Training data must include both churn classes")
    if history["Client"].duplicated().any():
        raise ValueError("Repeated customers in historical training matrix")

    n_no_churn = int((y_train == 0).sum())
    n_churn = int((y_train == 1).sum())
    # Fixed Model B configuration; compute the class weight from each training set.
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=2,
        gamma=3.0,
        reg_lambda=3,
        reg_alpha=0.1,
        scale_pos_weight=n_no_churn / n_churn,
        random_state=42,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    print("Tuned Model B fitted. Evaluate generalization separately with held-out data.")

    prepared = prepare_transactions(pd.read_csv(STAGED_FILE))
    # build_customer_profiles() excludes transactions ON its cutoff date.
    # Therefore use the day after the last observed date to include all data.
    current_cutoff = prepared["Clean_Date"].max() + pd.Timedelta(days=1)
    current_profiles = build_customer_profiles(prepared, current_cutoff)
    current_profiles = filter_returning_customers(current_profiles)
    if current_profiles.empty:
        raise ValueError("No returning customers to score")
    current_features = encode_customer_profiles(
        current_profiles, feature_columns=X_train.columns
    ).drop(columns=["Client"])

    if list(current_features.columns) != list(X_train.columns):
        raise ValueError("Historical and current feature columns do not align")
    probabilities = model.predict_proba(current_features)[:, 1] * 100

    output = pd.DataFrame({
        "Client": current_profiles["Client"].to_numpy(),
        "Churn_Probability_%": np.round(probabilities, 1),
        "Total_Gross_Spend": current_profiles["Total_Gross_Spend"].round(2).to_numpy(),
        "Total_Visits": current_profiles["Frequency_Total_Visits"].to_numpy(),
        "Is_One_Time_Buyer": np.where(
            current_profiles["Frequency_Total_Visits"] == 1, "Yes", "No"
        ),
        "Recency_Days": current_profiles["Recency_Days"].to_numpy(),
    })
    output["Retention_Risk_Tier"] = output["Churn_Probability_%"].apply(
        assign_risk_tier
    )
    output = output.sort_values(
        ["Churn_Probability_%", "Total_Gross_Spend"],
        ascending=[False, False],
    ).reset_index(drop=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(output):,} current returning-customer rows to {OUTPUT_FILE}.")
    print("Demo/research scores only; thresholds and calibration are not validated.")
    print("Customer-level output: publish only when all inputs are verified synthetic.")


if __name__ == "__main__":
    main()
