"""Build five DuckDB star-schema tables from the validated staged transactions.

Input: ``fastmag_staged_v2.csv``; output: ``fastmag_retail_v2.db``.
The fact-table grain is one (Vente, Ligne) transaction line. Dimension keys
are StockID, Client, Vendeur and Clean_Date; estimated merchandise margin uses
signed returns and excludes system/nonmerchandise lines.

The fixed calendar covers 2024-01-01 through 2026-06-16, including days with
no transactions, to support the demonstration report's date slicers. Change
both bounds deliberately if the reporting period changes.

Both paths are relative to the current working directory.
"""

import pandas as pd
import duckdb
import numpy as np

# Define staging file path and the final database destination file
staged_file = 'fastmag_staged_v2.csv'
db_file = 'fastmag_retail_v2.db'

print("Initializing DuckDB Relational Modeling Pipeline...")

try:
    # 1. Load the fully clean staged dataset
    df = pd.read_csv(staged_file, low_memory=False)
    print(f"Loaded '{staged_file}' containing {len(df):,} active rows.")
    print("-" * 60)

    # =========================================================================
    # BUILD DIMENSION TABLES (Isolating entities and dropping duplicates)
    # =========================================================================
    print("Extracting unique dimension entities...")

    # A. Product Dimension Lookup
    dim_products = df[
        ['StockID', 'BarCode', 'Designation', 'Famille', 'Couleur', 'Taille', 'Saison', 'SaisonOrigine']].copy()
    dim_products.drop_duplicates(subset=['StockID'], keep='first', inplace=True)

    # B. Customer Dimension Lookup
    dim_customers = df[['Client']].copy()
    dim_customers.drop_duplicates(subset=['Client'], keep='first', inplace=True)

    # C. Vendeur Dimension Lookup
    dim_vendeurs = df[['Vendeur']].copy()
    dim_vendeurs.drop_duplicates(subset=['Vendeur'], keep='first', inplace=True)


    # D. Continuous reporting calendar, including dates with no transactions.

    date_series = pd.to_datetime(df["Clean_Date"], errors="raise")


    # Explicit reporting bounds for the synthetic Power BI dashboard.
    calendar_start = pd.Timestamp("2024-01-01")
    calendar_end = pd.Timestamp("2026-06-16")

    if date_series.min() < calendar_start or date_series.max() > calendar_end:
        raise ValueError("Synthetic transactions fall outside the reporting calendar")

    all_dates = pd.date_range(
        start=calendar_start,
        end=calendar_end,
        freq="D"
    )


    dim_calendar = pd.DataFrame({
        "Clean_Date": all_dates.strftime("%Y-%m-%d"),
        "Date": all_dates.strftime("%d/%m/%Y"),
        "Day_Of_Week": all_dates.day_name(),
        "Month_Number": all_dates.month,
        "Month_Name": all_dates.month_name(),
        "Year": all_dates.year,
        "Is_Weekend": all_dates.dayofweek.isin([4, 5]).astype(int)
    })

    print(f"Continuous calendar created: {len(dim_calendar):,} days")
    print(f" - Engineered 'dim_products'  | Unique Styles: {len(dim_products):,}")
    print(f" - Engineered 'dim_customers' | Unique Clients: {len(dim_customers):,}")
    print(f" - Engineered 'dim_vendeurs'  | Unique Channels: {len(dim_vendeurs):,}")
    print(f" - Engineered 'dim_calendar'  | Calendar Days:    {len(dim_calendar):,}")
    print("-" * 60)

    # =========================================================================
    # BUILD THE CENTRAL FACT TABLE
    # =========================================================================
    print("Structuring centralized 'fact_sales_items' matrix...")

    # Select the fact fields and calculate estimated net merchandise margin.
    fact_sales_items = df[[
        'Vente', 'Ligne', 'NumeroFacture', 'Clean_Date', 'Client', 'Vendeur', 'StockID',
        'Line_Quantite', 'PrixVente', 'Prix', 'Remise', 'Total_1', 'PrixAchat_ILS', 'Line_Type',
        'Final_Unit_Price_HT', 'Final_Total_Line_HT'
    ]].copy()

    # Estimate net merchandise margin AFTER returns/exchanges. Both HT revenue
    # and quantity already carry accounting signs; do not reverse them again.
    # The approximate purchase cost was converted to ILS during staging.
    # System/non-merchandise rows have no meaningful merchandise margin.
    merchandise_lines = fact_sales_items['Line_Type'].isin(['ARTICLE', 'RETURN'])
    fact_sales_items['Estimated_Net_Margin_ILS'] = np.where(
        merchandise_lines,
        fact_sales_items['Final_Total_Line_HT']
        - fact_sales_items['PrixAchat_ILS'] * fact_sales_items['Line_Quantite'],
        np.nan,
    )

    print(f" - Engineered 'fact_sales_items' | Transactional Rows: {len(fact_sales_items):,}")
    print("-" * 60)

    # =========================================================================
    # LOADING INTO DUCKDB INTERNALS
    # =========================================================================
    print(f"Connecting to DuckDB target storage environment: '{db_file}'...")

    conn = duckdb.connect(db_file)

    conn.execute("DROP TABLE IF EXISTS fact_sales_items;")
    conn.execute("DROP TABLE IF EXISTS dim_products;")
    conn.execute("DROP TABLE IF EXISTS dim_customers;")
    conn.execute("DROP TABLE IF EXISTS dim_vendeurs;")
    conn.execute("DROP TABLE IF EXISTS dim_calendar;")

    print("Streaming memory layers into structured database architecture...")
    conn.execute("CREATE TABLE dim_products AS SELECT * FROM dim_products;")
    conn.execute("CREATE TABLE dim_customers AS SELECT * FROM dim_customers;")
    conn.execute("CREATE TABLE dim_vendeurs AS SELECT * FROM dim_vendeurs;")
    conn.execute("CREATE TABLE dim_calendar AS SELECT * FROM dim_calendar;")
    conn.execute("CREATE TABLE fact_sales_items AS SELECT * FROM fact_sales_items;")

    print("\nDatabase Schema Finalization Verification:")
    print(conn.execute("SHOW TABLES;").df())
    print("-" * 60)

    conn.close()
    print("Database connection closed cleanly.")
    print(f"SUCCESS: Star Schema saved natively inside file: '{db_file}'")
    print("-" * 60)

except FileNotFoundError:
    print(f"Error: Missing staged input '{staged_file}'. Run main.py first.")
    raise
except Exception as e:
    print(f"DuckDB schema construction failed: {e}")
    raise