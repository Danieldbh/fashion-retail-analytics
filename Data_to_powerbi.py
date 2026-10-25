"""Export five DuckDB star-schema tables as headered CSV files for Power BI.

Run after DuckDB.py from the same working directory. This exporter creates
``powerbi_tables_v2`` and writes one CSV per fact/dimension table. Churn scores
and association rules are produced separately by Churn_Model.py and
MarketBasket_to_bi.py; they are not sourced from this database exporter.
"""

import duckdb
import os

# Refuse to create an empty database silently if the modeling step was skipped.
if not os.path.isfile('fastmag_retail_v2.db'):
    raise FileNotFoundError('Missing fastmag_retail_v2.db; run DuckDB.py first.')

os.makedirs('powerbi_tables_v2', exist_ok=True)

# Connect to the relational database file
conn = duckdb.connect('fastmag_retail_v2.db')

tables = ['fact_sales_items', 'dim_products', 'dim_customers', 'dim_vendeurs', 'dim_calendar']

print("Exporting database schema layers for Power BI ingestion...")
for table in tables:
    output_path = f"powerbi_tables_v2/{table}.csv"
    conn.execute(f"COPY {table} TO '{output_path}' (HEADER, DELIMITER ',');")
    print(f" - Exported: {table} -> {output_path}")

conn.close()
print("Export complete: five fact/dimension CSVs prepared for Power BI.")