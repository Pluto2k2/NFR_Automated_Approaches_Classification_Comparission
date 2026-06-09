# Requirements Classification Pipeline

This repository contains the code and results for classifying software requirements into Non-Functional Requirement (NFR) categories using the PROMISE dataset. 

## Overview
The goal is to automatically classify requirements text into one of 7 categories (F, LF, O, Other, PE, SE, US). The project compares classical machine learning models, fine-tuned BERT models, and large language models (Llama 3.3 70B).

## Key Results Summary

The experiments evaluated multiple approaches on a holdout test set (125 samples). The main performance metric is Weighted F1 score.

* **SVM**: 0.7759
* **BERT (Average over 5 seeds)**: 0.7726
* **Logistic Regression**: 0.7569
* **Llama 3.3 70B (Zero-shot)**: 0.6507
* **Majority Baseline**: 0.2365

Classical methods like SVM and Logistic Regression performed very well and were much faster to train than neural networks. The fine-tuned BERT model achieved comparable results to SVM but required a GPU to run efficiently. The zero-shot LLM struggled to match the supervised baselines on this specific task.

## Repository Structure

* `data/` Contains the processed dataset and split indices.
* `scripts/` Python scripts to run the different approaches.
* `results/` Contains the final metrics, confusion matrices, and model predictions.
* `requirements.txt` Python dependencies needed to run the code locally.

## How to Run Locally

First, install the required packages:

```bash
pip install -r requirements.txt
```

To run the classical baselines (Majority, Logistic Regression, SVM):

```bash
python scripts/task2_4_classical.py
```

To run the LLM evaluation (requires a Groq API key):

```bash
python scripts/task5_llm.py
```

*Note: The BERT evaluation script (`task3_bert_multiseed.py`) is designed to be run in a GPU environment like Google Colab.*
