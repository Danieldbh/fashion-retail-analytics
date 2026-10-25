#!/usr/bin/env python3
"""Simulate fictional fashion-retail customers, products and purchases.

This module supplies the catalog, customer_documents and make_staged functions
used by generate_raw_fastmag.py, the entry point for the reproducible public
POS -> main.py -> DuckDB -> analytics pipeline. The simulation encodes stable
customer preferences and purchase frequencies, seasonality, product affinities,
promotions, Web/Store choices and returns linked to earlier purchases.

It also offers an OPTIONAL standalone shortcut that writes staged and BI CSVs
directly, bypassing main.py/DuckDB.py. Use generate_raw_fastmag.py for the
canonical end-to-end demonstration and MarketBasket_to_bi.py for FP-Growth.
The standalone shortcut's basket rules come from exhaustive small-set counting,
not FP-Growth; --run-analytics can execute the supplied analysis scripts.

All product names, IDs, pricing, probabilities and transactions are invented;
no company datasets, PBIX files or other transaction sources are read.

Standalone example:
    python generate_synthetic_retail.py --output-dir synthetic_preview --customers 1400 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STAGED_COLUMNS = [
    'Vente','Date','Heure','Total','Basket_Total_Items','Nature','Client','Vendeur',
    'NumeroFacture','Vente_1','Ligne','StockID','Famille','BarCode','Designation',
    'Couleur','Taille','PrixAchat','PrixVente','Prix','Quantite_1','Remise','Total_1',
    'Saison','TVA','SaisonOrigine','Line_Quantite','Line_Type','PrixAchat_ILS',
    'Clean_Date','Day_Of_Week','Final_Unit_Price_HT','Final_Total_Line_HT',
]
FACT_COLUMNS = [
    'Vente','Ligne','NumeroFacture','Clean_Date','Client','Vendeur','StockID',
    'Line_Quantite','PrixVente','Prix','Remise','Total_1','PrixAchat_ILS',
    'Line_Type','Final_Unit_Price_HT','Final_Total_Line_HT',
    'Estimated_Net_Margin_ILS',
]
PRODUCT_COLUMNS = [
    'StockID','BarCode','Designation','Famille','Couleur','Taille','Saison',
    'SaisonOrigine',
]
CALENDAR_COLUMNS = [
    'Clean_Date','Date','Day_Of_Week','Month_Number','Month_Name','Year',
    'Is_Weekend',
]

# All category labels, SKU designations, prices and weights below are fictional.
# Technical field names match the ETL contract; distributions are invented.
CATALOG = {
    'BAG': (('Metro Crossbody', 'Everyday Tote', 'Weekender Bag', 'Mini Shoulder Bag'), 710, 0.1),
    'ACCESSORIES': (('Keyring Set', 'Cardholder', 'Fabric Belt', 'Pouch'), 145, 0.12),
    'T-SHIRT': (('Basic Cotton Tee', 'Striped Crew Tee', 'Soft Jersey Tee', 'Graphic Cotton Tee'), 190, 0.16),
    'JEANS': (('Relaxed Denim', 'Straight Denim', 'Wide-Leg Denim', 'Classic Denim'), 440, 0.12),
    'SWEATER': (('Ribbed Knit', 'Crewneck Knit', 'Cotton Sweater', 'Lightweight Pullover'), 510, 0.1),
    'SCARF': (('Knitted Scarf', 'Soft Twill Scarf', 'Lightweight Wrap', 'Cotton Bandana'), 180, 0.08),
    'DRESS': (('Day Dress', 'Midi Dress', 'Shirt Dress', 'Pleated Dress'), 570, 0.09),
    'BLAZER': (('Classic Blazer', 'Relaxed Blazer', 'Single-Button Blazer', 'Linen Blazer'), 770, 0.06),
    'PANTS': (('Tailored Pants', 'Relaxed Trousers', 'Wide-Leg Pants', 'Cotton Chinos'), 430, 0.07),
    'SHIRT': (('Cotton Poplin Shirt', 'Linen Shirt', 'Oxford Shirt', 'Soft Collar Shirt'), 340, 0.05),
    'SKIRT': (('Midi Skirt', 'A-Line Skirt', 'Pleated Skirt', 'Denim Skirt'), 375, 0.03),
    'SNEAKERS': (('City Sneakers', 'Canvas Trainers', 'Low-Top Sneakers', 'Retro Trainers'), 480, 0.02),
}
# Explicit synthetic co-purchase tendencies; not estimates from external data.
AFFINITIES = {
    'BAG': ('ACCESSORIES', 0.68),
    'JEANS': ('T-SHIRT', 0.65),
    'DRESS': ('BLAZER', 0.58),
    'SWEATER': ('SCARF', 0.67),
    'PANTS': ('SHIRT', 0.50),
}
PERSONAS = {
    # probability, mean purchase interval (days), probability of return
    'Loyal': (0.22, 30, 0.07),
    'Regular': (0.29, 67, 0.10),
    'Occasional': (0.32, 170, 0.12),
    'Dormant': (0.17, 42, 0.13),
}
COLORS = ('Ink','Sand','Moss','Stone','Burgundy','Cream')
SIZES = ('S','M','L','U')
VAT_PERCENT = 18.0  # Fictional demonstration accounting rule, not a company rate.
COST_CONVERSION = 4.0  # Artificial structural compatibility assumption only.


@dataclass(frozen=True)
class Product:
    stock_id: int
    family: str
    designation: str
    barcode: str
    color: str
    size: str
    ttc_price: float
    cost_ils: float
    season: str


def catalog(rng: np.random.Generator) -> tuple[list[Product], dict[str, list[Product]]]:
    products: list[Product] = []
    by_family: dict[str,list[Product]] = defaultdict(list)
    sid = 610000
    for family, (labels, baseline, _) in CATALOG.items():
        for j, label in enumerate(labels):
            for variant in range(2):
                scale = float(rng.uniform(0.78, 1.30))
                price = float(round(baseline * scale / 5) * 5)
                cost_ils = round(price / (1 + VAT_PERCENT / 100) * float(rng.uniform(0.31, 0.53)), 2)
                product = Product(
                    stock_id=sid, family=family, designation=f'DEMO {label}',
                    barcode=f'DEMO-SKU-{sid}', color=COLORS[(j * 2 + variant) % len(COLORS)],
                    size=SIZES[(j + variant) % len(SIZES)], ttc_price=price,
                    cost_ils=cost_ils, season='DEMO-S' if j % 2 else 'DEMO-W',
                )
                by_family[family].append(product)
                products.append(product)
                sid += 1
    return products, dict(by_family)


def pick_family(rng, preferences: dict, excluded: set[str] | None = None) -> str:
    eligible = [c for c in CATALOG if not excluded or c not in excluded]
    weights = np.array([preferences[c] for c in eligible], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(eligible, p=weights))


def make_item(product: Product, quantity: int, discount: float = 0) -> dict:
    if quantity == 0:
        raise ValueError('Zero-quantity merchandise is not generated.')
    return {'product':product, 'qty':int(quantity), 'discount':float(discount), 'type':'ARTICLE'}


def new_document(date: pd.Timestamp, client: int, vendor: str, nature: str, items: list[dict]) -> dict:
    return {'date':pd.Timestamp(date).normalize(), 'client':int(client),
            'vendor':str(vendor), 'nature':str(nature), 'items':list(items)}


def customer_documents(rng, n_customers: int, start: pd.Timestamp, end: pd.Timestamp,
                       by_family: dict[str,list[Product]]) -> list[dict]:
    docs: list[dict] = []
    purchase_history: dict[int, list[tuple[pd.Timestamp,Product,float]]] = defaultdict(list)
    preferences_all = np.array([CATALOG[c][2] for c in CATALOG], dtype=float)
    preferences_all /= preferences_all.sum()
    days_span = (end - start).days
    persona_names = list(PERSONAS)
    persona_probs = np.array([PERSONAS[p][0] for p in persona_names])

    for person_no in range(n_customers):
        client_id = 800000 + person_no
        persona = str(rng.choice(persona_names, p=persona_probs))
        _, base_interval, return_rate = PERSONAS[persona]
        web_prob = float(rng.beta(3, 3))
        price_sensitivity = float(rng.beta(2, 5))
        personal_return_rate = min(.30, max(.02, return_rate * float(rng.uniform(.6, 1.5))))
        preferences = rng.dirichlet(preferences_all * 15)
        prefs = dict(zip(CATALOG, preferences))

        # Most customers start early; some enter later, giving entry and churn patterns.
        if persona == 'Dormant':
            activated_at = start + pd.Timedelta(days=int(rng.integers(0, min(250, days_span + 1))))
            stop_at = start + pd.Timedelta(days=int(rng.integers(360, min(760, days_span + 1))))
        else:
            activated_at = start + pd.Timedelta(days=int(rng.beta(1.4, 3.0) * min(days_span, 720)))
            stop_at = end
        activated_at = min(activated_at, end)
        at = activated_at
        while at <= min(end, stop_at):
            month = at.month
            # Demand intensifies during holiday/seasonal shopping periods.
            seasonal = 1.45 if month in (11, 12) else (1.18 if month in (5, 6) else 0.90)
            interval = max(4, int(round(float(rng.exponential(base_interval / seasonal)))))
            at = at + pd.Timedelta(days=interval)
            if at > end or at > stop_at:
                break
            # Weekend affects store vs web mix; channels are customer-dependent.
            adjusted_web = max(.04, min(.96, web_prob + (0.12 if at.dayofweek == 6 else 0)))
            vendor = 'WEB' if rng.random() < adjusted_web else str(rng.choice(['STORE_NORTH','STORE_CENTRAL']))
            count = int(rng.choice([1,2,3,4], p=[.47,.34,.15,.04]))
            first = pick_family(rng, prefs)
            families = [first]
            if count >= 2:
                affinity = AFFINITIES.get(first)
                if affinity and rng.random() < affinity[1]:
                    families.append(affinity[0])
                else:
                    families.append(pick_family(rng, prefs, set(families)))
            while len(families) < count:
                # Occasionally create a three-family outfit, not just a pair.
                next_family = pick_family(rng, prefs, set(families))
                families.append(next_family)

            items = []
            for family in families:
                product = by_family[family][int(rng.integers(len(by_family[family])))]
                qty = 2 if rng.random() < 0.075 else 1
                # Discount-prone customers accept more promotions; seasonal campaigns.
                promo_prob = min(.8, .12 + .48 * price_sensitivity + (.21 if month in (1, 11) else 0))
                discount = float(rng.choice([10, 20, 30, 40], p=[.38,.32,.23,.07])) if rng.random() < promo_prob else 0.0
                items.append(make_item(product, qty, discount))
                for _ in range(qty):
                    purchase_history[client_id].append((at,product,discount))

            # Occasional shipping fee and coupon rows are separate from merchandise.
            if vendor == 'WEB' and rng.random() < .45:
                items.append({'special':'SHIPPING','qty':1,'discount':0,'type':'SHIPPING','ttc':16.0})
            if rng.random() < .045:
                items.append({'special':'COUPON','qty':1,'discount':0,'type':'COUPON','ttc':-18.0})
            docs.append(new_document(at, client_id, vendor, 'VENTE', items))

        # Customer returns refer to merchandise purchased earlier; no independent
        # random returns for items never purchased.
        purchase_docs = sorted(
            (d for d in docs if d['client'] == client_id and d['nature'] == 'VENTE'),
            key=lambda d: d['date']
        )
        for purchase_date, product, discount in purchase_history[client_id]:
            if rng.random() >= personal_return_rate:
                continue
            return_date = purchase_date + pd.Timedelta(days=int(rng.integers(5, 37)))
            if return_date > end:
                continue
            future = [d for d in purchase_docs if 0 <= (d['date'] - return_date).days <= 23]
            attach = future and rng.random() < .58
            negative = make_item(product, -1, discount)
            if attach:
                receiving = future[int(rng.integers(len(future)))]
                receiving['items'].append(negative)
                # Some exchanges purchase a variation with same designation.
                if rng.random() < .35:
                    alternatives = [p for p in by_family[product.family]
                                    if p.designation == product.designation
                                    and p.stock_id != product.stock_id]
                    if alternatives:
                        receiving['items'].append(make_item(alternatives[0], 1, discount))
            else:
                docs.append(new_document(return_date, client_id,
                         'WEB' if rng.random() < web_prob else 'STORE_CENTRAL',
                         'AVOIR', [negative]))
    return docs


def make_staged(docs: list[dict], start: pd.Timestamp) -> pd.DataFrame:
    docs = sorted(docs, key=lambda d: (d['date'],d['client'],d['nature']))
    rows: list[dict] = []
    line_id = 2000000
    for tx_idx, doc in enumerate(docs):
        sale_id, invoice_id = 900000 + tx_idx, 500000 + tx_idx
        item_rows = []
        date = doc['date']
        hour = f'{9 + (tx_idx % 13):02d}:{(tx_idx * 7) % 60:02d}:00'
        for item in doc['items']:
            qty = item['qty']
            typ = item['type']
            if 'special' in item:
                family = '~' + item['special']
                product = None
                sid = {'SHIPPING':710001,'COUPON':710002,'GIFTCARD':710003}[item['special']]
                designation = 'DEMO '+item['special'].title()
                price = float(item['ttc'])
                cost_ils = 0.0
                barcode = 'DEMO-SYSTEM-'+item['special']
                color, size, season = '', '', ''
            else:
                product = item['product']
                family = product.family
                sid = product.stock_id
                designation = product.designation
                price = product.ttc_price
                cost_ils = product.cost_ils
                barcode = product.barcode
                color, size, season = product.color, product.size, product.season
            discount = float(item['discount'])
            line_ttc = round(price * qty * (1.0 - discount / 100.0), 2)
            # The line total is the canonical amount; HT uses the synthetic VAT.
            total_ht = round(line_ttc / (1.0 + VAT_PERCENT / 100.0), 6)
            unit_ht = round(total_ht / qty, 6)
            row = {
                'Vente':sale_id,'Date':date.strftime('%d/%m/%Y'),'Heure':hour,
                'Nature':doc['nature'],'Client':doc['client'],'Vendeur':doc['vendor'],
                'NumeroFacture':invoice_id,
                'Vente_1':sale_id if doc['nature']=='VENTE' else np.nan,
                'Ligne':line_id,'StockID':sid,'Famille':family,
                'BarCode':barcode,'Designation':designation,'Couleur':color,'Taille':size,
                'PrixAchat':round(cost_ils / COST_CONVERSION, 4),
                'PrixVente':price,'Prix':price,'Quantite_1':qty,'Remise':discount,
                'Total_1':line_ttc,'Saison':season,'TVA':VAT_PERCENT,
                'SaisonOrigine':season,'Line_Quantite':qty,'Line_Type':typ,
                'PrixAchat_ILS':cost_ils,'Clean_Date':date.strftime('%Y-%m-%d'),
                'Day_Of_Week':date.day_name(),
                'Final_Unit_Price_HT':unit_ht,'Final_Total_Line_HT':total_ht,
            }
            item_rows.append(row)
            line_id += 1
        invoice_total = round(sum(r['Total_1'] for r in item_rows), 2)
        total_positive = sum(r['Line_Quantite'] for r in item_rows
                             if r['Line_Type']=='ARTICLE' and r['Line_Quantite']>0)
        for row in item_rows:
            row['Total'] = invoice_total
            row['Basket_Total_Items'] = total_positive
        rows.extend(item_rows)
    if not rows:
        raise ValueError('No transactions generated. Use more customers or a wider date range.')
    return pd.DataFrame.from_records(rows, columns=STAGED_COLUMNS)


def build_bi_tables(staged: pd.DataFrame) -> dict[str,pd.DataFrame]:
    fact = staged[[c for c in FACT_COLUMNS if c != 'Estimated_Net_Margin_ILS']].copy()
    eligible = fact['Line_Type'].isin(['ARTICLE','RETURN'])
    fact['Estimated_Net_Margin_ILS'] = np.where(
        eligible,
        fact['Final_Total_Line_HT'] - fact['PrixAchat_ILS'] * fact['Line_Quantite'],
        np.nan,
    )
    products = staged[PRODUCT_COLUMNS].drop_duplicates('StockID',keep='first')
    customers = staged[['Client']].drop_duplicates('Client',keep='first')
    vendors = staged[['Vendeur']].drop_duplicates('Vendeur',keep='first')
    all_dates = pd.date_range(staged['Clean_Date'].min(), staged['Clean_Date'].max(), freq='D')
    cal = pd.DataFrame({
        'Clean_Date': all_dates.strftime('%Y-%m-%d'),
        'Date': all_dates.strftime('%d/%m/%Y'),
        'Day_Of_Week': all_dates.day_name(),
        'Month_Number': all_dates.month,
        'Month_Name': all_dates.month_name(),
        'Year': all_dates.year,
        # Consistent with the existing retail pipeline's Fri/Sat weekend flag.
        'Is_Weekend': all_dates.dayofweek.isin([4,5]).astype(int),
    })
    return {
        'fact_sales_items':fact[FACT_COLUMNS],
        'dim_products':products[PRODUCT_COLUMNS],
        'dim_customers':customers[['Client']],
        'dim_vendeurs':vendors[['Vendeur']],
        'dim_calendar':cal[CALENDAR_COLUMNS],
    }


def validate(staged: pd.DataFrame, tables: dict[str,pd.DataFrame]) -> dict:
    """Fail visibly if referential, financial or temporal invariants break."""
    if list(staged.columns) != STAGED_COLUMNS:
        raise AssertionError('Staged schema is not identical to expected schema')
    if staged.duplicated(['Vente','Ligne']).any():
        raise AssertionError('Duplicate Vente+Ligne')
    if staged[['Vente','NumeroFacture','Client','Vendeur']].isna().any().any():
        raise AssertionError('Null basket/customer/vendor/invoice key')
    for field_a,field_b in [('Vente','NumeroFacture'),('NumeroFacture','Vente')]:
        if staged.groupby(field_a)[field_b].nunique().gt(1).any():
            raise AssertionError('Sale and invoice keys are not 1:1')
    if staged.groupby('Vente')['Client'].nunique().gt(1).any():
        raise AssertionError('Transaction contains multiple customer IDs')
    if not (staged['Line_Quantite']==staged['Quantite_1']).all():
        raise AssertionError('Staged quantity fields disagree')
    if not (staged['Line_Quantite']!=0).all():
        raise AssertionError('Zero quantity line')
    if not np.allclose(staged['Final_Unit_Price_HT']*staged['Line_Quantite'],
                       staged['Final_Total_Line_HT'],atol=.00001):
        raise AssertionError('Unit vs line HT inconsistent')
    if not np.allclose(staged['Final_Total_Line_HT']*(1 + VAT_PERCENT/100),
                       staged['Total_1'],atol=.00001):
        raise AssertionError('HT vs TTC inconsistent')
    correct_invoice_totals = staged.groupby('Vente')['Total_1'].sum().round(2)
    declared_totals = staged.groupby('Vente')['Total'].first()
    if not np.allclose(correct_invoice_totals,declared_totals,atol=.00001):
        raise AssertionError('Invoice total mismatch')
    fact = tables['fact_sales_items']
    expected_margin = np.where(
        fact['Line_Type'].isin(['ARTICLE','RETURN']),
        fact['Final_Total_Line_HT'] - fact['PrixAchat_ILS']*fact['Line_Quantite'],
        np.nan,
    )
    if not np.allclose(fact['Estimated_Net_Margin_ILS'],expected_margin,equal_nan=True):
        raise AssertionError('Estimated margin formula mismatch')
    for name, col in [('dim_products','StockID'),('dim_customers','Client'),
                      ('dim_vendeurs','Vendeur'),('dim_calendar','Clean_Date')]:
        if tables[name][col].duplicated().any():
            raise AssertionError(f'{name} has duplicate primary keys')
    for dim,col in [('dim_products','StockID'),('dim_customers','Client'),
                    ('dim_vendeurs','Vendeur'),('dim_calendar','Clean_Date')]:
        if not fact[col].isin(tables[dim][col]).all():
            raise AssertionError(f'Broken fact-to-dimension relationship: {dim}')
    basket_lines = staged[staged['Line_Type'].isin(['ARTICLE','RETURN'])]
    positive = basket_lines[basket_lines['Line_Quantite'].gt(0)]
    negative = basket_lines[basket_lines['Line_Quantite'].lt(0)]
    if positive.empty or negative.empty:
        raise AssertionError('Need both purchases and returns in demo')
    mixed = basket_lines.groupby('Vente')['Line_Quantite'].agg(lambda v: v.gt(0).any() and v.lt(0).any())
    if not mixed.any():
        raise AssertionError('No mixed purchase/return invoices generated')
    no_purchase = basket_lines.loc[basket_lines['Nature'].eq('AVOIR') & basket_lines['Line_Quantite'].lt(0)]
    if no_purchase.empty:
        raise AssertionError('No pure refund transactions generated')
    # Reconciliation analogous to Power BI's validated gross/returns/net measures.
    gross_ht = float(positive['Final_Total_Line_HT'].sum())
    returned_ht = -float(negative['Final_Total_Line_HT'].sum())
    net_ht = float(basket_lines['Final_Total_Line_HT'].sum())
    if not math.isclose(gross_ht-returned_ht,net_ht,rel_tol=1e-10,abs_tol=.0001):
        raise AssertionError('Gross - returns != net merchandise revenue')
    return {
        'schema': '33 staged fields; 5 DuckDB-compatible exports',
        'transaction_rows':len(staged),
        'invoices':int(staged['Vente'].nunique()),
        'customers':int(staged['Client'].nunique()),
        'products':int(tables['dim_products']['StockID'].nunique()),
        'merchandise_lines':len(basket_lines),
        'negative_merchandise_lines':len(negative),
        'mixed_purchase_return_invoices':int(mixed.sum()),
        'refund_only_invoices':int(no_purchase['Vente'].nunique()),
        'calendar_days':len(tables['dim_calendar']),
        'gross_minus_returns_equals_net':True,
    }



BASKET_RULE_COLUMNS = [
    'Channel','Antecedents','Consequents','Rule_Size','Support',
    'Antecedent_Support','Consequent_Support','Combo_Count','Total_Baskets',
    'Confidence','Lift','Rule_Items_Revenue_HT','Avg_Rule_Items_Revenue_HT',
    'Association_Score',
]


def build_demo_association_rules(staged: pd.DataFrame) -> pd.DataFrame:
    """Calculate optional preview association rules by enumerating category sets.

    Basket support, confidence and lift are calculated from simulated purchases,
    not assigned by the generator. With 12 categories and baskets of up to four
    categories, brute-force enumeration is tractable. This is separate from the
    canonical FP-Growth analysis in MarketBasket_to_bi.py.
    """
    chosen = staged.loc[
        staged['Nature'].eq('VENTE')
        & staged['Line_Type'].eq('ARTICLE')
        & staged['Line_Quantite'].gt(0)
        & staged['Final_Total_Line_HT'].gt(0)
        & ~staged['Famille'].str.startswith('~')
    ].copy()
    chosen['Channel_Label'] = np.where(chosen['Vendeur'].eq('WEB'),'Web','Physical')
    results=[]
    for channel in ('Omnichannel','Physical','Web'):
        scope=chosen if channel=='Omnichannel' else chosen.loc[chosen['Channel_Label'].eq(channel)]
        if scope.empty:
            continue
        total_baskets=scope['Vente'].nunique()
        family_receipts={fam:set(frame['Vente']) for fam,frame in scope.groupby('Famille')}
        amount_by_family={fam:frame.groupby('Vente')['Final_Total_Line_HT'].sum().to_dict()
                          for fam,frame in scope.groupby('Famille')}
        cats=sorted(family_receipts)
        freq={}  # tuple of category names -> (number of matching receipts, amount)
        for k in range(1, min(4,len(cats))+1):
            for itemset in combinations(cats,k):
                receipt_sets=[family_receipts[cat] for cat in itemset]
                receipts=set.intersection(*receipt_sets)
                count=len(receipts)
                if count/total_baskets < 0.005:
                    continue
                amount=sum(sum(amount_by_family[cat].get(receipt,0.0) for receipt in receipts)
                           for cat in itemset)
                freq[itemset]=(count,amount)
        for combined,(joint_count,amount) in freq.items():
            if len(combined)<2:
                continue
            for left_size in range(1,len(combined)):
                for left in combinations(combined,left_size):
                    right=tuple(sorted(set(combined)-set(left)))
                    left_count=freq.get(left,(0,0))[0]
                    right_count=freq.get(right,(0,0))[0]
                    if not left_count or not right_count:
                        continue
                    supp=joint_count/total_baskets
                    conf=joint_count/left_count
                    lift=conf/(right_count/total_baskets)
                    if conf<0.15 or lift<1.2:
                        continue
                    results.append({
                        'Channel':channel,'Antecedents':', '.join(left),
                        'Consequents':', '.join(right),'Rule_Size':len(combined),
                        'Support':supp,'Antecedent_Support':left_count/total_baskets,
                        'Consequent_Support':right_count/total_baskets,
                        'Combo_Count':joint_count,'Total_Baskets':total_baskets,
                        'Confidence':conf,'Lift':lift,'Rule_Items_Revenue_HT':amount,
                        'Avg_Rule_Items_Revenue_HT':amount/joint_count,
                        'Association_Score':supp*lift,
                    })
    result=pd.DataFrame.from_records(results,columns=BASKET_RULE_COLUMNS)
    if not result.empty:
        result=result.sort_values(['Association_Score','Channel','Antecedents','Consequents'],
                                  ascending=[False,True,True,True]).reset_index(drop=True)
    return result

def run_analytics_scripts(outdir:Path, scripts_dir:Path)->None:
    """Run downstream analysis scripts from the synthetic output directory.

    Analysis programs use relative input/output paths. Setting cwd to outdir
    ensures their datasets are read and written in the synthetic output folder.
    The script directory is used only to locate the executable source files.
    """
    names=['Churn_Prep.py','Churn_Model.py','MarketBasket_to_bi.py']
    for name in names:
        script=scripts_dir/name
        if not script.is_file():
            raise FileNotFoundError(f'Missing analysis script: {script}')
    for name in names:
        print(f'Running {name} with synthetic output directory ...',flush=True)
        subprocess.run([sys.executable,str((scripts_dir/name).resolve())],
                       cwd=str(outdir.resolve()),check=True)
    expected=[outdir/'BI_Customer_Retention_Returning_Tuned_v2.csv',
              outdir/'powerbi_tables_v2'/'market_basket_rules_v2.csv']
    for path in expected:
        if not path.exists():
            raise ValueError(f'Analytics script failed to produce expected output: {path}')
    print('Analytics outputs created with XGBoost and FP-Growth scripts.')


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=Path('synthetic_retail_demo'))
    ap.add_argument('--customers',type=int,default=1400)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--start',default='2024-01-01')
    ap.add_argument('--end',default='2026-06-16')
    ap.add_argument('--run-analytics',action='store_true',
                    help='Run churn and market-basket scripts against the generated synthetic data.')
    ap.add_argument('--scripts-dir',type=Path,
                    help='Directory containing Churn_Prep.py, tuned Churn_Model.py and MarketBasket_to_bi.py')
    args=ap.parse_args()
    if args.customers < 100:
        ap.error('--customers must be at least 100 to create useful association/churn examples')
    start,end=pd.Timestamp(args.start),pd.Timestamp(args.end)
    if start>=end or (end-start).days<730:
        ap.error('The date span must cover at least two full years for 365-day churn labels')
    if args.run_analytics and args.scripts_dir is None:
        ap.error('--run-analytics requires --scripts-dir')
    outdir=args.output_dir.resolve()
    marker_file=outdir/'synthetic_manifest.json'
    if (outdir/'fastmag_staged_v2.csv').exists():
        try:
            prior=json.loads(marker_file.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            ap.error('Refusing to overwrite an existing staged CSV without a valid synthetic_manifest.json')
        if prior.get('synthetic_only') is not True:
            ap.error('Refusing to overwrite a folder not explicitly marked synthetic_only')
    if args.scripts_dir is not None and outdir==args.scripts_dir.resolve():
        ap.error('--output-dir and --scripts-dir must not be the same folder')
    outdir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    _,by_family=catalog(rng)
    docs=customer_documents(rng,args.customers,start,end,by_family)
    staged=make_staged(docs,start)
    tables=build_bi_tables(staged)
    summary=validate(staged,tables)
    # A re-run invalidates all previous ML scores/features. Remove only generated
    # model artifacts in this already-verified synthetic directory.
    for stale in ('customer_churn_features_returning_v2.csv',
                  'BI_Customer_Retention_Returning_Tuned_v2.csv'):
        (outdir/stale).unlink(missing_ok=True)
    # All generated datasets are written under the selected output directory.
    stagepath=outdir/'fastmag_staged_v2.csv'
    staged.to_csv(stagepath,index=False,float_format='%.8f')
    bidir=outdir/'powerbi_tables_v2'
    bidir.mkdir(exist_ok=True)
    for name,frame in tables.items():
        frame.to_csv(bidir/(name+'.csv'),index=False,float_format='%.8f')
    # Lightweight preview rules are calculated from baskets, never sampled or
    # invented; --run-analytics overwrites them with FP-Growth output.
    demo_rules=build_demo_association_rules(staged)
    demo_rules.to_csv(bidir/'market_basket_rules_v2.csv',index=False,float_format='%.12g')
    manifest={
        'synthetic_only':True,
        'source':'Generated from explicitly invented category catalog and probability rules; zero real records used',
        'seed':args.seed, 'requested_customers':args.customers,
        'requested_start':str(start.date()),'requested_end':str(end.date()),
        'demo_VAT_percent':VAT_PERCENT,'demo_cost_conversion':COST_CONVERSION,
        'designed_affinities':{k:{'associated_category':v[0],'conditional_inclusion_probability':v[1]}
                               for k,v in AFFINITIES.items()},
        'checks':summary,
        'preview_market_basket_rules':int(len(demo_rules)),
        'preview_rule_method':'Exact synthetic category-set enumeration (up to 4 categories); use MarketBasket_to_bi.py for FP-Growth output',
        'churn_output':'Generated only with --run-analytics, using Churn_Prep.py and Churn_Model.py',
    }
    (outdir/'synthetic_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Generated fully synthetic retail data at:',outdir)
    print('Checks:',json.dumps(summary,indent=2))
    if args.run_analytics:
        run_analytics_scripts(outdir,args.scripts_dir.resolve())
    else:
        print(f'Built {len(demo_rules)} exact synthetic association preview rules.')
        print('Optional: run --run-analytics --scripts-dir <analysis-scripts-directory>')
        print('to create actual synthetic-trained XGBoost and FP-Growth outputs.')

if __name__=='__main__':
    main()
