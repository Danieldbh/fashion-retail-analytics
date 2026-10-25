"""Clean a tab-delimited retail POS export into a validated transaction-line CSV.

Input: ``fastmag (2).xls`` (tab-separated text, not a binary Excel workbook).
Output: ``fastmag_staged_v2.csv`` in the current working directory.

Each output row represents one unique (Vente, Ligne) transaction line. This
pipeline retains signed sales/returns, classifies nonmerchandise lines, derives
line quantities, validates TTC totals, and calculates VAT-exclusive (HT) amounts.
Purchase costs are converted using a fixed, approximate 4.0 EUR-to-ILS factor;
all downstream margin figures are therefore estimates.

Run from the directory containing the input file: ``python path/to/main.py``.
"""

import pandas as pd
import numpy as np

# Paths are relative to the current working directory.
original_file = 'fastmag (2).xls'
staged_output_file = 'fastmag_staged_v2.csv'

print('Starting Unified Data Engineering Pipeline...')

try:
    # STEP 1: INGESTION
    df = pd.read_csv(original_file, sep='\t', encoding='latin1', low_memory=False)
    total_rows = len(df)
    print(f'Step 1: Loaded {total_rows:,} rows and {len(df.columns)} columns.')

    # STEP 2: REMOVE UNWANTED COLUMNS AND DEDUPLICATE
    null_cols_to_drop = [
        col for col in df.columns
        if df[col].isna().sum() / total_rows > 0.90
    ]

    constants_drop = [
        'CodeMag', 'CodeMag_1', 'CodeMag_2', 'HeureLivSouhaite', 'TauxDevise',
        'DepotBanque', 'GPExport', 'VenteHT', 'Recu', 'Controler', 'ClientVIP',
        'Reliquat', 'Commande'
    ]
    redundancy_drop = [
        'ID', 'ModReg', 'Vente_2', 'RefPaiement',
        'MontantDevise', 'Montant', 'NumeroFactureJour', 'VShopStatus',
        'EcoTaxeUnitaire_HT', 'EcoTaxeUnitaire_TTC',
        'Fournisseur', 'Avoir', 'Libelle'
    ]
    context_noise_drop = [
        'InfosComp', 'Origine', 'Reception', 'Echeance',
        'Designation2', 'Utilisateur', 'Motif', 'AdrLivraison'
    ]

    drop_columns = [
        col for col in (
            null_cols_to_drop + constants_drop + redundancy_drop + context_noise_drop
        )
        if col in df.columns
    ]
    df.drop(columns=drop_columns, inplace=True)

    key_cols = ['Vente', 'Ligne']
    if df[key_cols].isna().any().any():
        raise ValueError('Missing transaction-line identifiers')

    duplicate_mask = df.duplicated(subset=key_cols, keep=False)
    if duplicate_mask.any():
        other_cols = [col for col in df.columns if col not in key_cols]
        conflicts = (
            df.loc[duplicate_mask]
            .groupby(key_cols)[other_cols]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        if conflicts.any():
            raise ValueError(
                f'Found {conflicts.sum()} transaction lines '
                'with conflicting data. Review before deduplication.'
            )

    rows_before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep='first').copy()
    print(f'Step 2: Removed {rows_before - len(df):,} duplicate transaction-line rows.')

    # STEP 3: QUANTITY ENGINEERING AND INTEGRITY CHECK
    df.rename(columns={'Quantite': 'Basket_Total_Items'}, inplace=True)
    source_qty = pd.to_numeric(df['Quantite_1'], errors='coerce')
    if source_qty.isna().any():
        raise ValueError('Missing or invalid source quantities')
    if not np.isclose(source_qty, np.round(source_qty)).all():
        raise ValueError('Non-integer quantities detected')
    df['Line_Quantite'] = source_qty.astype('Int64')

    price = pd.to_numeric(df['Prix'], errors='coerce')
    discount = pd.to_numeric(df['Remise'], errors='coerce')
    actual_total = pd.to_numeric(df['Total_1'], errors='coerce')
    discounted_price = price * (1 - discount / 100.0)
    expected_total = discounted_price * df['Line_Quantite'].astype(float)
    valid = discounted_price.notna() & discounted_price.ne(0) & actual_total.notna()
    mismatches = valid & ~np.isclose(
        expected_total, actual_total, rtol=1e-6, atol=0.02
    )
    if mismatches.any():
        raise ValueError(f'Found {mismatches.sum()} financial inconsistencies')
    print('Step 3: Quantity validation completed.')

    # STEP 4: CLASSIFY MERCHANDISE, RETURNS, AND NONMERCHANDISE LINES
    conditions = [
        df['Famille'] == '~SHIPPING',
        df['Famille'] == '~COUPON',
        df['Famille'] == '~GIFTCARD',
        df['Famille'] == '~AVOIR',
        df['Nature'] == 'AVOIR'
    ]
    choices = ['SHIPPING', 'COUPON', 'GIFTCARD', 'MANUAL_ADJUSTMENT', 'RETURN']
    df['Line_Type'] = np.select(conditions, choices, default='ARTICLE')
    print('Step 4: Line types classified.')

    # STEP 5: ESTIMATE PURCHASE COSTS IN ILS (FIXED APPROXIMATE RATE)
    if 'PrixAchat' in df.columns:
        df['PrixAchat_ILS'] = df['PrixAchat'] * 4.0
    print('Step 5: Purchase costs converted using the approximate 4.0 rate.')

    # STEP 6: DATE VALIDATION AND TIME INTELLIGENCE
    if 'Date' not in df.columns:
        raise ValueError('Missing required column: Date')
    df['Clean_Date'] = pd.to_datetime(
        df['Date'], format='%d/%m/%Y', errors='coerce'
    )
    invalid_dates = df['Clean_Date'].isna()
    if invalid_dates.any():
        raise ValueError(f'Found {invalid_dates.sum()} invalid transaction dates')
    df['Day_Of_Week'] = df['Clean_Date'].dt.day_name()
    print('Step 6: Dates validated.')

    # STEP 7: VAT AND REVENUE EXCLUDING VAT (HT)
    vat_pct = pd.to_numeric(df['TVA'], errors='coerce')
    invalid_vat = vat_pct.isna() | ~vat_pct.isin([17, 18])
    if invalid_vat.any():
        raise ValueError(f'Found {invalid_vat.sum()} invalid VAT rates')
    vat_rate = vat_pct / 100.0

    unit_price_ttc = df['Prix'] * (1 - df['Remise'] / 100.0)
    df['Final_Unit_Price_HT'] = unit_price_ttc / (1 + vat_rate)
    df['Final_Total_Line_HT'] = df['Total_1'] / (1 + vat_rate)

    expected_line_ht = (
        df['Final_Unit_Price_HT'] * df['Line_Quantite'].astype(float)
    )
    mismatches = ~np.isclose(
        expected_line_ht, df['Final_Total_Line_HT'], rtol=1e-6, atol=0.02
    )
    if mismatches.any():
        raise ValueError(f'Found {mismatches.sum()} inconsistent HT totals')
    print('Step 7: VAT and HT calculations validated.')

    # STEP 8: EXPORT
    df.to_csv(staged_output_file, index=False, encoding='utf-8')
    print(f'Step 8: Pipeline completed. Output: {staged_output_file}')

except FileNotFoundError:
    print(f"Error: Target file '{original_file}' was not found. Please verify placement.")
    raise
except Exception as e:
    print(f'Pipeline stalled due to processing anomaly: {e}')
    raise
