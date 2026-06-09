"""
Task 2: Baselines (Majority-Class + Logistic Regression)
Task 4: Multi-seed SVM with CV
Task 6: Bootstrap CIs + McNemar's test
Task 7: Full metrics for all classical models
"""
import pandas as pd
import numpy as np
import json, os, time, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                             confusion_matrix, f1_score)
from scipy.stats import chi2

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

# Load data
with open(os.path.join(DATA_DIR, "splits.json")) as f:
    splits = json.load(f)

labels = splits['labels']
label2id = splits['label2id']
id2label = {int(k): v for k, v in splits['id2label'].items()}

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0)
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col=0)
all_train = pd.concat([train_df, val_df])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"All train (for classical): {len(all_train)}")

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
X_train = tfidf.fit_transform(all_train['cleaned_text'])
X_test = tfidf.transform(test_df['cleaned_text'])
y_train = all_train['label'].values
y_test = test_df['label'].values

# ============================================================
# Helper functions
# ============================================================
def compute_full_metrics(y_true, y_pred, labels_list):
    """Compute weighted and macro metrics."""
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    per_class = classification_report(y_true, y_pred, target_names=labels_list, zero_division=0, output_dict=True)
    return {
        'weighted_precision': round(wp, 4), 'weighted_recall': round(wr, 4), 'weighted_f1': round(wf, 4),
        'macro_precision': round(mp, 4), 'macro_recall': round(mr, 4), 'macro_f1': round(mf, 4),
        'per_class': {k: per_class[k] for k in labels_list if k in per_class}
    }

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap CI for a metric."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return round(lower, 4), round(upper, 4)

def save_confusion_matrix(y_true, y_pred, labels_list, name, cm_dir):
    """Save confusion matrix as CSV and PNG."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    cm_df.to_csv(os.path.join(cm_dir, f"{name}_cm.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list, ax=ax)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, f"{name}_cm.png"), dpi=150)
    plt.close()
    return cm

def save_predictions(y_true, y_pred, name, pred_dir, labels_list):
    """Save predictions to CSV."""
    df = pd.DataFrame({'true_label': y_true, 'pred_label': y_pred,
                       'true_name': [labels_list[i] for i in y_true],
                       'pred_name': [labels_list[i] for i in y_pred]})
    df.to_csv(os.path.join(pred_dir, f"{name}_preds.csv"), index=False)

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test between two classifiers."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right
    if b + c == 0:
        return 1.0  # No disagreement
    # McNemar with continuity correction
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(stat, df=1)
    return round(p_value, 6)

all_results = {}

# ============================================================
# TASK 2a: Majority-Class Baseline
# ============================================================
print("\n" + "=" * 60)
print("TASK 2a: Majority-Class Baseline")
print("=" * 60)

majority = DummyClassifier(strategy='most_frequent')
majority.fit(X_train, y_train)
y_pred_maj = majority.predict(X_test)

metrics_maj = compute_full_metrics(y_test, y_pred_maj, labels)
wf1_ci = bootstrap_ci(y_test, y_pred_maj, lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0))
mf1_ci = bootstrap_ci(y_test, y_pred_maj, lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0))
metrics_maj['weighted_f1_ci'] = wf1_ci
metrics_maj['macro_f1_ci'] = mf1_ci

save_predictions(y_test, y_pred_maj, "majority_baseline", PRED_DIR, labels)
save_confusion_matrix(y_test, y_pred_maj, labels, "majority_baseline", CM_DIR)
all_results['majority_baseline'] = metrics_maj

print(f"  Weighted F1: {metrics_maj['weighted_f1']:.4f} [{wf1_ci[0]:.4f}, {wf1_ci[1]:.4f}]")
print(f"  Macro F1:    {metrics_maj['macro_f1']:.4f} [{mf1_ci[0]:.4f}, {mf1_ci[1]:.4f}]")

# ============================================================
# TASK 2b: Logistic Regression
# ============================================================
print("\n" + "=" * 60)
print("TASK 2b: Logistic Regression (TF-IDF)")
print("=" * 60)

start = time.time()
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='lbfgs')
lr.fit(X_train, y_train)
lr_train_time = time.time() - start

start = time.time()
y_pred_lr = lr.predict(X_test)
lr_inf_time = (time.time() - start) / len(y_test) * 1000  # ms per sample

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='f1_weighted')

metrics_lr = compute_full_metrics(y_test, y_pred_lr, labels)
wf1_ci = bootstrap_ci(y_test, y_pred_lr, lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0))
mf1_ci = bootstrap_ci(y_test, y_pred_lr, lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0))
metrics_lr['weighted_f1_ci'] = wf1_ci
metrics_lr['macro_f1_ci'] = mf1_ci
metrics_lr['cv_weighted_f1_mean'] = round(lr_cv_scores.mean(), 4)
metrics_lr['cv_weighted_f1_std'] = round(lr_cv_scores.std(), 4)
metrics_lr['train_time_s'] = round(lr_train_time, 4)
metrics_lr['inference_ms_per_sample'] = round(lr_inf_time, 4)

save_predictions(y_test, y_pred_lr, "logistic_regression", PRED_DIR, labels)
save_confusion_matrix(y_test, y_pred_lr, labels, "logistic_regression", CM_DIR)
all_results['logistic_regression'] = metrics_lr

print(f"  Weighted F1: {metrics_lr['weighted_f1']:.4f} [{wf1_ci[0]:.4f}, {wf1_ci[1]:.4f}]")
print(f"  Macro F1:    {metrics_lr['macro_f1']:.4f} [{mf1_ci[0]:.4f}, {mf1_ci[1]:.4f}]")
print(f"  5-Fold CV:   {metrics_lr['cv_weighted_f1_mean']:.4f} ± {metrics_lr['cv_weighted_f1_std']:.4f}")
print(f"  Train time:  {lr_train_time:.4f}s | Inference: {lr_inf_time:.4f} ms/sample")

# ============================================================
# TASK 4: SVM (test-set + 5-fold CV)
# ============================================================
print("\n" + "=" * 60)
print("TASK 4: SVM (TF-IDF, Linear, class_weight=balanced)")
print("=" * 60)

start = time.time()
svm = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)
svm_train_time = time.time() - start

start = time.time()
y_pred_svm = svm.predict(X_test)
svm_inf_time = (time.time() - start) / len(y_test) * 1000

# 5-fold CV
svm_cv_wf1 = cross_val_score(svm, X_train, y_train, cv=cv, scoring='f1_weighted')
svm_cv_mf1 = cross_val_score(svm, X_train, y_train, cv=cv, scoring='f1_macro')

metrics_svm = compute_full_metrics(y_test, y_pred_svm, labels)
wf1_ci = bootstrap_ci(y_test, y_pred_svm, lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0))
mf1_ci = bootstrap_ci(y_test, y_pred_svm, lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0))
metrics_svm['weighted_f1_ci'] = wf1_ci
metrics_svm['macro_f1_ci'] = mf1_ci
metrics_svm['cv_weighted_f1_mean'] = round(svm_cv_wf1.mean(), 4)
metrics_svm['cv_weighted_f1_std'] = round(svm_cv_wf1.std(), 4)
metrics_svm['cv_macro_f1_mean'] = round(svm_cv_mf1.mean(), 4)
metrics_svm['cv_macro_f1_std'] = round(svm_cv_mf1.std(), 4)
metrics_svm['train_time_s'] = round(svm_train_time, 4)
metrics_svm['inference_ms_per_sample'] = round(svm_inf_time, 4)

save_predictions(y_test, y_pred_svm, "svm", PRED_DIR, labels)
save_confusion_matrix(y_test, y_pred_svm, labels, "svm", CM_DIR)
all_results['svm'] = metrics_svm

print(f"  Weighted F1: {metrics_svm['weighted_f1']:.4f} [{wf1_ci[0]:.4f}, {wf1_ci[1]:.4f}]")
print(f"  Macro F1:    {metrics_svm['macro_f1']:.4f} [{mf1_ci[0]:.4f}, {mf1_ci[1]:.4f}]")
print(f"  5-Fold CV (weighted): {metrics_svm['cv_weighted_f1_mean']:.4f} ± {metrics_svm['cv_weighted_f1_std']:.4f}")
print(f"  5-Fold CV (macro):    {metrics_svm['cv_macro_f1_mean']:.4f} ± {metrics_svm['cv_macro_f1_std']:.4f}")
print(f"  Train: {svm_train_time:.4f}s | Inference: {svm_inf_time:.4f} ms/sample")

# ============================================================
# TASK 6 (partial): McNemar SVM vs LR
# ============================================================
print("\n" + "=" * 60)
print("TASK 6: McNemar's Test (SVM vs Logistic Regression)")
print("=" * 60)
p_svm_lr = mcnemar_test(y_test, y_pred_svm, y_pred_lr)
print(f"  McNemar p-value (SVM vs LR): {p_svm_lr}")
all_results['mcnemar_svm_vs_lr'] = p_svm_lr

# ============================================================
# Save intermediate results
# ============================================================
with open(os.path.join(RESULTS_DIR, "classical_results.json"), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\n  Saved classical results to {os.path.join(RESULTS_DIR, 'classical_results.json')}")

# Also save SVM predictions for later McNemar comparisons with BERT/LLM
np.save(os.path.join(PRED_DIR, "svm_y_pred.npy"), y_pred_svm)
np.save(os.path.join(PRED_DIR, "y_test.npy"), y_test)

print("\n[Tasks 2, 4, 6-partial, 7-partial] COMPLETE.")
