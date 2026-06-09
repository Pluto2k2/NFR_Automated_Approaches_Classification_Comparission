"""
Task 5: Improved LLM Evaluation (4 conditions)
  5a. Zero-shot, original prompt
  5b. Zero-shot, improved prompt (Other class explained)
  5c. Few-shot k=5
"""
import pandas as pd
import numpy as np
import json, os, time, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             classification_report, f1_score)
from groq import Groq
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")

# Load API key from .env file (project root) or environment variable
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not found. Create a .env file in the project root with:")
    print('  GROQ_API_KEY=your_key_here')
    sys.exit(1)

with open(os.path.join(DATA_DIR, "splits.json")) as f:
    splits = json.load(f)
labels = splits['labels']
label2id = splits['label2id']

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col=0)
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col=0)
y_test = test_df['label'].values

# ============================================================
# Prompt definitions
# ============================================================
PROMPT_ORIGINAL = """You are an expert Requirements Engineer. Your task is to classify a software requirement into exactly ONE of these categories:

F, LF, O, Other, PE, SE, US

Where:
- F = Functional Requirement (describes what the system should do)
- LF = Look and Feel (appearance, UI aesthetics)
- O = Operability (operational environment, platform constraints)
- Other = Other NFR (availability, scalability, maintainability, legal, fault tolerance)
- PE = Performance (speed, response time, throughput)
- SE = Security (encryption, authentication, access control)
- US = Usability (ease of use, learnability, accessibility)

Reply with ONLY the category label (e.g. "F" or "SE"). No explanation, no punctuation, just the label."""

PROMPT_IMPROVED = """You are an expert Requirements Engineer. Classify the following software requirement into exactly ONE of these 7 categories. Reply with ONLY the label.

CATEGORIES:
- F = Functional Requirement: describes a specific behavior, feature, or capability the system must provide.
- LF = Look and Feel: requirements about the system's appearance, visual design, or branding.
- O = Operability: requirements about the operational environment, platform support, or deployment constraints.
- PE = Performance: requirements about speed, response time, throughput, capacity, or resource efficiency.
- SE = Security: requirements about encryption, authentication, authorization, data protection, or access control.
- US = Usability: requirements about ease of use, learnability, accessibility, or user satisfaction.
- Other = Other Non-Functional Requirement. Use this category for ANY requirement related to:
    * Availability (uptime, redundancy, disaster recovery)
    * Scalability (handling growth in users, data, or transactions)
    * Maintainability (ease of modification, modular design, code quality)
    * Legal (compliance with laws, regulations, licenses, or standards)
    * Fault Tolerance (graceful degradation, error handling, recovery from failures)
    * Portability (ability to run on different platforms or environments)
  If the requirement matches ANY of these six subtypes, classify it as "Other".

IMPORTANT: Reply with exactly one of: F, LF, O, Other, PE, SE, US
Do not output anything else."""

def build_fewshot_examples(train_df, k_per_class, labels, seed=42):
    """Select k examples per class from training set."""
    rng = np.random.RandomState(seed)
    examples = []
    for label in labels:
        subset = train_df[train_df['Class'] == label]
        n = min(k_per_class, len(subset))
        sampled = subset.sample(n=n, random_state=seed)
        for _, row in sampled.iterrows():
            examples.append(f'Requirement: "{row["RequirementText"]}"\nLabel: {row["Class"]}')
    rng.shuffle(examples)
    return "\n\n".join(examples)

# ============================================================
# LLM classification function with retry
# ============================================================
def classify_with_llm(test_df, system_prompt, condition_name, user_prefix="Classify this requirement:"):
    """Classify all test requirements with exponential backoff and resume."""
    client = Groq(api_key=GROQ_API_KEY)
    valid_labels = set(labels)
    
    # Check for partial results to resume
    partial_path = os.path.join(PRED_DIR, f"llm_{condition_name}_partial.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        predictions = partial['predictions']
        start_idx = len(predictions)
        print(f"  Resuming from sample {start_idx}/{len(test_df)}")
    else:
        predictions = []
        start_idx = 0
    
    errors = 0
    total = len(test_df)
    rows = list(test_df.itertuples())
    
    for i in range(start_idx, total):
        row = rows[i]
        text = row.RequirementText
        
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f'{user_prefix} "{text}"'}
                    ],
                    temperature=0,
                    max_tokens=5
                )
                pred = response.choices[0].message.content.strip().upper()
                pred = pred.replace('"', '').replace("'", '').replace('.', '').strip()
                if pred not in valid_labels:
                    errors += 1
                    pred = 'F'  # Fallback
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                print(f"  API error (attempt {retries}/{max_retries}): {e}. Waiting {wait}s...")
                time.sleep(wait)
        else:
            pred = 'F'
            errors += 1
        
        predictions.append(pred)
        
        # Save partial results every 25 samples
        if (i + 1) % 25 == 0:
            with open(partial_path, 'w') as f:
                json.dump({'predictions': predictions}, f)
            print(f"  Classified {i + 1}/{total} (errors so far: {errors})")
        
        time.sleep(0.5)  # Rate limit respect
    
    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)
    
    return predictions, errors

def evaluate_llm_condition(predictions, errors, condition_name, y_test, labels, label2id):
    """Compute full metrics for an LLM condition."""
    y_pred = np.array([label2id.get(p, 0) for p in predictions])
    
    wp, wr, wf, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    
    # Bootstrap CIs
    def boot_wf1(yt, yp): return f1_score(yt, yp, average='weighted', zero_division=0)
    def boot_mf1(yt, yp): return f1_score(yt, yp, average='macro', zero_division=0)
    
    rng = np.random.RandomState(42)
    wf1_scores, mf1_scores = [], []
    for _ in range(1000):
        idx = rng.choice(len(y_test), len(y_test), replace=True)
        wf1_scores.append(boot_wf1(y_test[idx], y_pred[idx]))
        mf1_scores.append(boot_mf1(y_test[idx], y_pred[idx]))
    
    wf1_ci = (round(np.percentile(wf1_scores, 2.5), 4), round(np.percentile(wf1_scores, 97.5), 4))
    mf1_ci = (round(np.percentile(mf1_scores, 2.5), 4), round(np.percentile(mf1_scores, 97.5), 4))
    
    per_class = classification_report(y_test, y_pred, target_names=labels, zero_division=0, output_dict=True)
    
    # Save predictions
    pred_df = pd.DataFrame({'true_label': y_test, 'pred_label': y_pred,
                            'true_name': [labels[i] for i in y_test],
                            'pred_name': predictions})
    pred_df.to_csv(os.path.join(PRED_DIR, f"llm_{condition_name}_preds.csv"), index=False)
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(os.path.join(CM_DIR, f"llm_{condition_name}_cm.csv"))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Llama 3.3 70B — {condition_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(CM_DIR, f"llm_{condition_name}_cm.png"), dpi=150)
    plt.close()
    
    result = {
        'weighted_precision': round(wp, 4), 'weighted_recall': round(wr, 4), 'weighted_f1': round(wf, 4),
        'macro_precision': round(mp, 4), 'macro_recall': round(mr, 4), 'macro_f1': round(mf, 4),
        'weighted_f1_ci': wf1_ci, 'macro_f1_ci': mf1_ci,
        'parse_errors': errors, 'total_samples': len(y_test),
        'per_class': {k: per_class[k] for k in labels if k in per_class}
    }
    
    np.save(os.path.join(PRED_DIR, f"llm_{condition_name}_y_pred.npy"), y_pred)
    
    return result

# ============================================================
# Run all 4 conditions
# ============================================================
all_llm_results = {}

# Save all LLM results
with open(os.path.join(RESULTS_DIR, "llm_results.json"), 'w') as f:
    json.dump(all_llm_results, f, indent=2, default=str)
print(f"\n  Saved all LLM results to {os.path.join(RESULTS_DIR, 'llm_results.json')}")

print("\n[Task 5] COMPLETE.")
