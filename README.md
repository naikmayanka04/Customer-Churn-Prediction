# Customer Churn Prediction — Milestone 01: Data Prep & Baseline

Supervised machine-learning project (binary classification) that predicts whether a telecom customer will churn.
This milestone covers data exploration, cleaning, a leakage-safe preprocessing pipeline, a simple baseline model, and its evaluation.
Hyperparameter tuning and more complex models are intentionally left for later milestones.

## Problem
* **Task:** binary classification — `Churn` (Yes/No) → encoded as 1/0, churn is the positive class
* **Business context:** identify customers likely to leave so retention offers can be targeted
* **Baseline model:** Logistic Regression, compared against a majority-class `DummyClassifier`

## Dataset
* **Name:** Telco Customer Churn (IBM sample dataset)
* **Where to get it:** search Kaggle for "Telco Customer Churn" and download `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
  Check the dataset page for its license and terms of use.
* **Where to put it:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
* The raw data is **not committed** to the repository (see `.gitignore`); download it yourself using the steps above.
* Use the standard 21-column CSV. Some expanded versions contain columns such as `Churn Label`, `Churn Score` or `Churn Reason`
  that leak the target; `src/data_preparation.py` drops them if present, but the standard file does not need this.

## Repository structure
```text
churn-project/
├── data/
│   ├── raw/                    # place the downloaded CSV here (git-ignored)
│   └── processed/              # cleaned data written by the notebook/script (git-ignored)
├── notebooks/
│   └── milestone_01_data_prep_baseline.ipynb   # EDA, cleaning, split, pipeline, baseline, evaluation
├── src/
│   ├── config.py               # paths, random seed, column names
│   ├── data_preparation.py     # load, rule-based cleaning, X/y creation
│   ├── preprocessing.py        # ColumnTransformer + baseline model pipelines
│   ├── evaluation.py           # metrics, cross-validation, plots
│   └── train_baseline.py       # end-to-end reproducible run
├── reports/
│   ├── figures/                # confusion matrix, ROC curve (created by the script)
│   └── baseline_metrics.json   # metrics from the last script run
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to reproduce
1. Place the CSV in `data/raw/` (see above).
2. **Notebook (full analysis):** run `jupyter lab`, open `notebooks/milestone_01_data_prep_baseline.ipynb`, and run all cells top to bottom.
3. **Script (baseline only):** from the repository root run
   ```bash
   python -m src.train_baseline
   ```
   This cleans the data, creates a stratified 80/20 split (seed 42), cross-validates on the training set, evaluates on the test set once,
   and writes `reports/baseline_metrics.json` and figures to `reports/figures/`.

## Method summary
| Stage | Approach |
| --- | --- |
| Cleaning | Rule-based only: whitespace stripping, `TotalCharges` to numeric, deterministic handling of blank values (see notebook Section 5), exact-duplicate removal |
| Split | 80/20, stratified on the target, `random_state=42`, done before any fitting |
| Preprocessing | `ColumnTransformer`: numeric → median impute + standard scale; categorical → most-frequent impute + one-hot |
| Leakage control | All learned transformations live inside a scikit-learn `Pipeline` fitted on training data only; cross-validation refits them per fold |
| Model selection | None in this milestone; the test set is evaluated once |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix (accuracy alone is misleading under class imbalance) |

## Results

| Metric (test set) | Dummy (majority) | Logistic Regression |
| --- | ---: | ---: |
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |
| ROC-AUC | | |
| PR-AUC | | |

Short interpretation (2–4 sentences, written from your results):

## Limitations
* Public sample dataset; may not reflect real-world churn behaviour.
* Snapshot data without timestamps, so no time-based validation.
* Class imbalance; a single 80/20 split gives one point estimate.
* Untuned linear baseline at the default 0.5 decision threshold.

## Next steps (later milestones)
Class weighting/resampling, threshold selection based on retention cost, regularisation tuning, tree-based models,
feature engineering, probability calibration, and more robust evaluation.
