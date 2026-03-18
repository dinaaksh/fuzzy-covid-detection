"""
Full dataset analysis script.
Usage: python analyze_full_dataset.py --data_dir /path/to/your/csv/folder
"""
import os
import glob
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Folder containing all CSV files')
args = parser.parse_args()

# ── Load all CSVs ──────────────────────────────────────────────────────────────
all_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
print(f"Found {len(all_files)} CSV files\n")
df = pd.concat([pd.read_csv(f, low_memory=False) for f in all_files], ignore_index=True)

df = df[df['covid19_test_results'].isin(['Positive', 'Negative'])].copy()
df['target'] = df['covid19_test_results'].map({'Positive': 1, 'Negative': 0})

pos = df[df['target'] == 1]
neg = df[df['target'] == 0]

print("=" * 60)
print("1. DATASET SIZE & CLASS BALANCE")
print("=" * 60)
print(f"Total rows      : {len(df):,}")
print(f"Positives       : {len(pos):,}  ({len(pos)/len(df)*100:.2f}%)")
print(f"Negatives       : {len(neg):,}  ({len(neg)/len(df)*100:.2f}%)")
print(f"Imbalance ratio : {len(neg)/len(pos):.1f}:1")

# ── Binary features ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. BINARY FEATURE DISCRIMINABILITY (ratio = pos_rate / neg_rate)")
print("=" * 60)

binary_features = [
    'cough', 'fever', 'loss_of_smell', 'loss_of_taste', 'sob', 'fatigue',
    'headache', 'muscle_sore', 'sore_throat', 'runny_nose', 'diarrhea',
    'diabetes', 'high_risk_exposure_occupation', 'high_risk_interactions',
    'chd', 'htn', 'cancer', 'asthma', 'copd', 'autoimmune_dis', 'smoker'
]

rows = []
for col in binary_features:
    if col not in df.columns:
        continue
    pv = df[col].astype(str).str.strip().str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
    ).fillna(0)
    p_rate = pv[df['target'] == 1].mean()
    n_rate = pv[df['target'] == 0].mean()
    ratio = p_rate / (n_rate + 1e-9)
    rows.append((col, p_rate, n_rate, ratio))

rows.sort(key=lambda x: -x[3])
print(f"{'Feature':<35} {'Pos%':>6} {'Neg%':>6} {'Ratio':>7}")
print("-" * 58)
for col, p, n, r in rows:
    print(f"{col:<35} {p*100:>5.1f}% {n*100:>5.1f}% {r:>7.2f}x")

# ── Continuous features ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CONTINUOUS FEATURES (mean pos vs neg)")
print("=" * 60)

continuous = ['age', 'temperature', 'pulse', 'sats', 'rr', 'days_since_symptom_onset']
print(f"{'Feature':<30} {'Pos mean':>10} {'Neg mean':>10} {'Diff':>8}")
print("-" * 62)
for col in continuous:
    if col not in df.columns:
        continue
    pv = pd.to_numeric(df[col], errors='coerce')
    pm = pv[df['target'] == 1].mean()
    nm = pv[df['target'] == 0].mean()
    print(f"{col:<30} {pm:>10.2f} {nm:>10.2f} {pm-nm:>8.2f}")

# ── Severity features ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. SEVERITY FEATURES")
print("=" * 60)

sev_map = {'mild': 1, 'moderate': 2, 'severe': 3}
for col in ['cough_severity', 'sob_severity']:
    if col not in df.columns:
        continue
    pv = df[col].astype(str).str.strip().str.lower().map(sev_map).fillna(0)
    pm = pv[df['target'] == 1].mean()
    nm = pv[df['target'] == 0].mean()
    print(f"{col:<30} pos mean={pm:.3f}  neg mean={nm:.3f}  diff={pm-nm:.3f}")

    # Value distribution
    for grp, label in [(pos, 'POS'), (neg, 'NEG')]:
        vals = grp[col].astype(str).str.strip().str.lower().value_counts()
        print(f"  {label}: {dict(vals)}")

# ── Lung exam features ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. LUNG EXAM FEATURES")
print("=" * 60)

for col in ['ctab', 'rhonchi', 'wheezes', 'labored_respiration']:
    if col not in df.columns:
        continue
    pv = df[col].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0}).fillna(np.nan)
    p_rate = pv[df['target'] == 1].mean()
    n_rate = pv[df['target'] == 0].mean()
    p_null = pv[df['target'] == 1].isna().mean()
    n_null = pv[df['target'] == 0].isna().mean()
    print(f"{col:<30} pos={p_rate*100:.1f}%  neg={n_rate*100:.1f}%  "
          f"ratio={p_rate/(n_rate+1e-9):.2f}x  "
          f"null%: pos={p_null*100:.0f}% neg={n_null*100:.0f}%")

# ── Combination features ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. COMBINATION FEATURES (interaction terms)")
print("=" * 60)

def binarize(series):
    return series.astype(str).str.strip().str.lower().map(
        {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
    ).fillna(0).astype(int)

combos = [
    ('loss_of_smell', 'loss_of_taste'),
    ('fever', 'cough'),
    ('fever', 'loss_of_smell'),
    ('cough', 'loss_of_smell'),
    ('fever', 'muscle_sore'),
    ('fatigue', 'loss_of_smell'),
    ('fever', 'cough', 'loss_of_smell'),
    ('fever', 'cough', 'loss_of_taste'),
]

print(f"{'Combination':<45} {'Pos%':>6} {'Neg%':>6} {'Ratio':>7}")
print("-" * 68)
for combo in combos:
    if not all(c in df.columns for c in combo):
        continue
    mask = pd.Series([1]*len(df), index=df.index)
    for c in combo:
        mask = mask & binarize(df[c])
    p_rate = mask[df['target'] == 1].mean()
    n_rate = mask[df['target'] == 0].mean()
    ratio = p_rate / (n_rate + 1e-9)
    name = ' & '.join(combo)
    print(f"{name:<45} {p_rate*100:>5.1f}% {n_rate*100:>5.1f}% {ratio:>7.2f}x")

# ── Comorbidity check ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. COMORBIDITY — do any positives have them?")
print("=" * 60)

comorbidities = ['diabetes', 'chd', 'htn', 'cancer', 'asthma', 'copd', 'autoimmune_dis']
for col in comorbidities:
    if col not in df.columns:
        continue
    pv = binarize(df[col])
    pos_count = int(pv[df['target'] == 1].sum())
    pos_total = int((df['target'] == 1).sum())
    print(f"{col:<25} positives with it: {pos_count}/{pos_total}")

# ── Age distribution ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. AGE DISTRIBUTION IN POSITIVES")
print("=" * 60)

if 'age' in df.columns:
    age = pd.to_numeric(df['age'], errors='coerce')
    bins = [0, 18, 30, 40, 50, 60, 70, 120]
    labels = ['<18', '18-30', '30-40', '40-50', '50-60', '60-70', '70+']
    df['age_bin'] = pd.cut(age, bins=bins, labels=labels)
    for grp, label in [(pos, 'POS'), (neg, 'NEG')]:
        dist = df.loc[grp.index, 'age_bin'].value_counts().sort_index()
        pct = (dist / len(grp) * 100).round(1)
        print(f"{label}: {dict(pct)}")

# ── Missing value audit ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. NULL / MISSING VALUE AUDIT (columns > 5% null)")
print("=" * 60)

null_pct = df.isnull().mean() * 100
high_null = null_pct[null_pct > 5].sort_values(ascending=False)
for col, pct in high_null.items():
    print(f"{col:<35} {pct:.1f}% null")

print("\n" + "=" * 60)
print("DONE — paste this full output back.")
print("=" * 60)