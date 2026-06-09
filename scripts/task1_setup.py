"""
Task 1: Setup and Reproducibility
- Load PROMISE NFR dataset
- Apply preprocessing and class merging
- Create 70/10/20 stratified split with seed=42
- Save split indices to disk
"""
import pandas as pd
import numpy as np
import re
import json
import os
import sys

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 1. Load raw dataset
raw_path = os.path.join(os.path.dirname(BASE_DIR), "data", "raw", "promise_real.csv")
df = pd.read_csv(raw_path)
df.columns = ['RequirementText', 'Class']
print(f"[Task 1] Loaded raw dataset: {len(df)} rows, {df['Class'].nunique()} classes")
print(f"  Original distribution:\n{df['Class'].value_counts().to_string()}\n")

# 2. Merge rare classes
rare_classes = ['A', 'SC', 'MN', 'L', 'FT', 'PO']
df['Class'] = df['Class'].apply(lambda x: 'Other' if x in rare_classes else x)
print(f"  After merging rare classes into 'Other': {df['Class'].nunique()} classes")
print(f"  Merged distribution:\n{df['Class'].value_counts().to_string()}\n")

# 3. Clean text
df['cleaned_text'] = df['RequirementText'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))

# 4. Label encoding (sorted for reproducibility)
labels = sorted(df['Class'].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
df['label'] = df['Class'].map(label2id)

print(f"  Labels: {labels}")
print(f"  Label mapping: {label2id}\n")

# 5. Stratified split: 70/10/20
from sklearn.model_selection import train_test_split

train_val_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED, stratify=df['label']
)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.125, random_state=SEED, stratify=train_val_df['label']
)

print(f"  Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
print(f"  Test class distribution:\n{test_df['Class'].value_counts().to_string()}\n")

# 6. Save split indices to disk
splits = {
    'train_indices': train_df.index.tolist(),
    'val_indices': val_df.index.tolist(),
    'test_indices': test_df.index.tolist(),
    'labels': labels,
    'label2id': label2id,
    'id2label': {str(k): v for k, v in id2label.items()},
    'seed': SEED,
    'n_train': len(train_df),
    'n_val': len(val_df),
    'n_test': len(test_df),
}

splits_path = os.path.join(DATA_DIR, "splits.json")
with open(splits_path, 'w') as f:
    json.dump(splits, f, indent=2)
print(f"  Saved split indices to {splits_path}")

# 7. Save processed data
processed_path = os.path.join(DATA_DIR, "promise_processed.csv")
df.to_csv(processed_path, index=True)
print(f"  Saved processed data to {processed_path}")

# 8. Save individual splits as CSV for easy loading
train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=True)
val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=True)
test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=True)
print(f"  Saved train.csv, val.csv, test.csv to {DATA_DIR}")

print("\n[Task 1] COMPLETE. Splits saved and verified.")
