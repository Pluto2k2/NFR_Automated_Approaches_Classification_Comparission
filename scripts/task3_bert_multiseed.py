"""
GOOGLE COLAB: Task 3 — Multi-seed BERT evaluation with class weighting
Upload: promise_real.csv + this file. Runtime → T4 GPU.
!pip install transformers datasets accelerate scikit-learn pandas
!python task3_bert_multiseed.py
"""
import pandas as pd
import numpy as np
import re, json, os, time, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             classification_report, f1_score)
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. Load & preprocess (identical to task1)
# ============================================================
df = pd.read_csv("promise_real.csv")
df.columns = ['RequirementText', 'Class']
rare = ['A', 'SC', 'MN', 'L', 'FT', 'PO']
df['Class'] = df['Class'].apply(lambda x: 'Other' if x in rare else x)
df['cleaned_text'] = df['RequirementText'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
labels = sorted(df['Class'].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for i, l in enumerate(labels)}
df['label'] = df['Class'].map(label2id)

# Same split as local
train_val_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['label'])
train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42, stratify=train_val_df['label'])
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 2. Compute class weights for weighted loss
# ============================================================
from torch.nn import CrossEntropyLoss

class_counts = train_df['label'].value_counts().sort_index()
total = len(train_df)
n_classes = len(labels)
class_weights = torch.tensor([total / (n_classes * class_counts[i]) for i in range(n_classes)], dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.tolist()}")

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_t = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels_t)
        return (loss, outputs) if return_outputs else loss

# ============================================================
# 3. Tokenize
# ============================================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_fn(examples):
    return tokenizer(examples['cleaned_text'], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize_fn, batched=True)

# ============================================================
# 4. Multi-seed evaluation
# ============================================================
SEEDS = [13, 42, 123, 456, 789]
# Optuna best hyperparameters (from previous Colab run)
LEARNING_RATE = 4.370e-05
EPOCHS = 4
BATCH_SIZE = 8

all_seed_results = {}
all_y_preds = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=n_classes, id2label=id2label, label2id=label2id
    ).to(device)
    
    args = TrainingArguments(
        output_dir=f'./results_seed_{seed}',
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer
    )
    
    start = time.time()
    trainer.train()
    train_time = time.time() - start
    
    start = time.time()
    predictions = trainer.predict(test_dataset)
    inf_time = time.time() - start
    
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_test = test_df['label'].values
    
    wp, wr, wf, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    per_class = classification_report(y_test, y_pred, target_names=labels, zero_division=0, output_dict=True)
    
    all_seed_results[seed] = {
        'weighted_f1': round(wf, 4), 'macro_f1': round(mf, 4),
        'weighted_precision': round(wp, 4), 'weighted_recall': round(wr, 4),
        'macro_precision': round(mp, 4), 'macro_recall': round(mr, 4),
        'train_time_s': round(train_time, 2), 'inference_time_s': round(inf_time, 2),
        'per_class': {k: per_class[k] for k in labels if k in per_class}
    }
    all_y_preds[seed] = y_pred.tolist()
    
    print(f"  Weighted F1: {wf:.4f} | Macro F1: {mf:.4f}")
    print(f"  Train: {train_time:.1f}s | Inference: {inf_time:.1f}s")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

# ============================================================
# 5. Aggregate across seeds
# ============================================================
wf1s = [all_seed_results[s]['weighted_f1'] for s in SEEDS]
mf1s = [all_seed_results[s]['macro_f1'] for s in SEEDS]

print(f"\n{'='*60}")
print(f"AGGREGATE RESULTS (5 seeds)")
print(f"{'='*60}")
print(f"  Weighted F1: {np.mean(wf1s):.4f} +/- {np.std(wf1s):.4f}")
print(f"  Macro F1:    {np.mean(mf1s):.4f} +/- {np.std(mf1s):.4f}")
print(f"  Individual weighted F1s: {wf1s}")
print(f"  Individual macro F1s:    {mf1s}")

# Save results
output = {
    'seeds': SEEDS,
    'hyperparameters': {'learning_rate': LEARNING_RATE, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE},
    'per_seed': all_seed_results,
    'predictions': all_y_preds,
    'aggregate': {
        'weighted_f1_mean': round(np.mean(wf1s), 4),
        'weighted_f1_std': round(np.std(wf1s), 4),
        'macro_f1_mean': round(np.mean(mf1s), 4),
        'macro_f1_std': round(np.std(mf1s), 4),
    }
}

with open('bert_multiseed_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nSaved results to bert_multiseed_results.json")
print("Download this file and place it in thesis_experiments/results/")
print("\n[Task 3] COMPLETE.")
