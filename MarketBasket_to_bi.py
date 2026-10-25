"""Category-level market basket analysis and Power BI export.

One rule per row, with separate Omnichannel, Physical and Web populations.
Omnichannel overlaps the other two: do NOT sum basket counts or revenues across
channels, or across different rules (the same basket may satisfy many rules).

Eligibility: positive ARTICLE sale lines (including positive merchandise from
exchanges) with positive net-of-VAT sales; return lines and nonmerchandise are
excluded, without discarding other purchased articles on mixed transactions.

All financial values are *purchased rule-item revenue HT*, not net-of-returns
store revenue and not causal revenue uplift from cross-selling.
"""

from pathlib import Path
import inspect

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth


SOURCE = Path('fastmag_staged_v2.csv')
OUTPUT = Path('powerbi_tables_v2') / 'market_basket_rules_v2.csv'
MIN_SUPPORT = 0.005
MIN_CONFIDENCE = 0.15
MIN_LIFT = 1.2

OUTPUT_COLUMNS = [
    'Channel', 'Antecedents', 'Consequents', 'Rule_Size',
    'Support', 'Antecedent_Support', 'Consequent_Support',
    'Combo_Count', 'Total_Baskets', 'Confidence', 'Lift',
    'Rule_Items_Revenue_HT', 'Avg_Rule_Items_Revenue_HT',
    'Association_Score',
]

REQUIRED_COLUMNS = [
    'Vente', 'NumeroFacture', 'Ligne', 'Famille', 'Nature', 'Line_Type',
    'Line_Quantite', 'Final_Total_Line_HT', 'Final_Unit_Price_HT', 'Vendeur',
]


def validate_transaction_keys(df):
    """Assert a real one-to-one relationship between sale IDs and invoice IDs.

    All source rows (including returns and nonmerchandise) are checked so the
    basket ID cannot accidentally combine or split distinct invoices.
    """
    for column in ('Vente', 'NumeroFacture', 'Ligne'):
        if df[column].isna().any():
            raise ValueError(f'Missing {column} values in staged transactions.')

    if df.duplicated(['Vente', 'Ligne']).any():
        raise ValueError('Duplicate (Vente, Ligne) keys in staged transactions.')

    invoice_counts = df.groupby('Vente')['NumeroFacture'].nunique(dropna=False)
    sale_counts = df.groupby('NumeroFacture')['Vente'].nunique(dropna=False)
    if (invoice_counts > 1).any() or (sale_counts > 1).any():
        raise ValueError('Vente and NumeroFacture must identify the same single basket.')


def load_purchased_merchandise(path=SOURCE):
    """Keep purchased merchandise, never returns; preserve mixed baskets."""
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns in {path}: {missing}')

    validate_transaction_keys(df)

    # Validate numeric fields rather than silently interpreting malformed values.
    for col in ('Line_Quantite', 'Final_Total_Line_HT', 'Final_Unit_Price_HT'):
        values = pd.to_numeric(df[col], errors='coerce')
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f'Invalid monetary/quantity values in {col}.')
        df[col] = values

    if df['Famille'].isna().any() or df['Vendeur'].isna().any():
        raise ValueError('Missing product families or channel/seller identifiers.')

    mask = (
        df['Line_Type'].eq('ARTICLE')
        & df['Nature'].eq('VENTE')
        & df['Line_Quantite'].gt(0)
        & df['Final_Total_Line_HT'].gt(0)
        & ~df['Famille'].isin(['~SHIPPING', '~COUPON', '~GIFTCARD', '~AVOIR'])
    )
    purchased = df.loc[mask].copy()

    # The unit-price-derived amount is approximate because source unit prices
    # can be rounded. Use the validated *line total* for actual rule-item HT.
    if not purchased.empty:
        difference = (
            purchased['Final_Unit_Price_HT'] * purchased['Line_Quantite']
            - purchased['Final_Total_Line_HT']
        ).abs()
        if (difference > 0.01).any():
            raise ValueError('Purchase line HT differs from unit price × quantity by >0.01.')

    purchased['HT_Value'] = purchased['Final_Total_Line_HT']
    purchased['Channel_Label'] = np.where(
        purchased['Vendeur'].astype(str).str.strip().str.upper().eq('WEB'),
        'Web', 'Physical',
    )
    return purchased


def _association_rules_compatible(itemsets, total_baskets, min_lift):
    """Support mlxtend releases with and without the num_itemsets parameter."""
    kwargs = {'metric': 'lift', 'min_threshold': min_lift}
    if 'num_itemsets' in inspect.signature(association_rules).parameters:
        kwargs['num_itemsets'] = total_baskets
    return association_rules(itemsets, **kwargs)


def generate_rules(df_subset, channel, min_support=MIN_SUPPORT,
                   min_confidence=MIN_CONFIDENCE, min_lift=MIN_LIFT):
    """Generate association rules using exactly the given basket population."""
    if df_subset.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # One family is either present or absent in each Vente, regardless of units.
    basket = pd.crosstab(df_subset['Vente'], df_subset['Famille']).gt(0)
    total_baskets = len(basket)
    if not total_baskets or basket.shape[1] < 2:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
    if itemsets.empty or not (itemsets['itemsets'].map(len) >= 2).any():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    rules = _association_rules_compatible(itemsets, total_baskets, min_lift)
    if rules.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    rules = rules.loc[rules['confidence'] >= min_confidence].copy()
    if rules.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Category -> basket IDs and per-basket HT amounts. Build once, and reuse
    # for every directional rule. Cache the combined-item-set computations too.
    category_receipts = {
        family: set(rows['Vente'])
        for family, rows in df_subset.groupby('Famille', sort=False)
    }
    amount_by_family = {
        family: rows.groupby('Vente')['HT_Value'].sum()
        for family, rows in df_subset.groupby('Famille', sort=False)
    }
    combined_cache = {}

    def combined_financials(combined_set):
        if combined_set not in combined_cache:
            ordered = sorted(combined_set, key=lambda x: len(category_receipts[x]))
            matching = category_receipts[ordered[0]].copy()
            for family in ordered[1:]:
                matching.intersection_update(category_receipts[family])
                if not matching:
                    break
            count = len(matching)
            amount = sum(
                float(amount_by_family[family].reindex(list(matching), fill_value=0).sum())
                for family in ordered
            ) if count else 0.0
            combined_cache[combined_set] = (count, amount)
        return combined_cache[combined_set]

    records = []
    # Using iterrows avoids reliance on pandas-renamed tuple attributes.
    for _, rule in rules.iterrows():
        left = rule['antecedents']
        right = rule['consequents']
        combined = left | right
        count, amount = combined_financials(combined)
        left_count, _ = combined_financials(left)
        right_count, _ = combined_financials(right)

        support = count / total_baskets
        left_support = left_count / total_baskets
        right_support = right_count / total_baskets
        confidence = count / left_count if left_count else 0
        lift = confidence / right_support if right_support else 0
        # Assert that the exported metrics describe the same eligible baskets
        # as mlxtend; prevents mismatched denominators or counting errors.
        for metric_name, actual, mined in (
            ('support', support, rule['support']),
            ('confidence', confidence, rule['confidence']),
            ('lift', lift, rule['lift']),
        ):
            if not np.isclose(actual, mined, rtol=1e-8, atol=1e-10):
                raise ValueError(f'{channel}: {metric_name} failed basket reconciliation.')

        records.append({
            'Channel': channel,
            'Antecedents': ', '.join(sorted(left)),
            'Consequents': ', '.join(sorted(right)),
            'Rule_Size': len(combined),
            'Support': support,
            'Antecedent_Support': left_support,
            'Consequent_Support': right_support,
            'Combo_Count': count,
            'Total_Baskets': total_baskets,
            'Confidence': confidence,
            'Lift': lift,
            'Rule_Items_Revenue_HT': amount,
            'Avg_Rule_Items_Revenue_HT': amount / count if count else 0.0,
            'Association_Score': support * lift,
        })

    result = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    return result.sort_values(
        ['Association_Score', 'Antecedents', 'Consequents'],
        ascending=[False, True, True], kind='stable',
    ).reset_index(drop=True)


def print_summary(rules, channel, total_baskets):
    print(f'\n{channel}: {total_baskets:,} eligible baskets; {len(rules):,} rules')
    if rules.empty:
        print('  No association rules met the selected thresholds.')
        return
    display = rules.head(7).copy()
    for col in ('Rule_Items_Revenue_HT', 'Avg_Rule_Items_Revenue_HT'):
        display[col] = display[col].map(lambda x: f'₪{x:,.2f}')
    print(display[[
        'Antecedents', 'Consequents', 'Combo_Count', 'Support',
        'Confidence', 'Lift', 'Avg_Rule_Items_Revenue_HT',
        'Rule_Items_Revenue_HT',
    ]].to_string(index=False))


def main():
    print(f'Loading validated transactions from {SOURCE}...')
    purchased = load_purchased_merchandise(SOURCE)
    print(f'Eligible positive merchandise lines: {len(purchased):,}')
    print(f'Mining with support ≥ {MIN_SUPPORT}, confidence ≥ {MIN_CONFIDENCE}, lift ≥ {MIN_LIFT}.')

    results = []
    for channel in ('Omnichannel', 'Physical', 'Web'):
        subset = purchased if channel == 'Omnichannel' else purchased.loc[
            purchased['Channel_Label'] == channel
        ]
        rules = generate_rules(subset, channel)
        print_summary(rules, channel, subset['Vente'].nunique())
        results.append(rules)

    output = pd.concat(results, ignore_index=True).reindex(columns=OUTPUT_COLUMNS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, float_format='%.12g')
    print(f'\nExported {len(output):,} rules to {OUTPUT}.')
    if output.empty:
        print('No rules qualified. The CSV contains only column headings (zero data rows).')
    print('Note: Omnichannel overlaps Web/Physical; revenue may repeat across rules.')


if __name__ == '__main__':
    main()
