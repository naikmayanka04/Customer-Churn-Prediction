"""Reproduce the Milestone 01 baseline end to end.

Usage (from the repository root):
    python -m src.train_baseline

Steps: load raw data -> clean -> X/y -> stratified train/test split ->
cross-validate baselines on TRAIN only -> fit on train -> evaluate on the
held-out test set ONCE -> save metrics and figures to reports/.
"""
import json

import matplotlib

matplotlib.use("Agg")  # no display needed when running as a script

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from .config import FIGURES_DIR, METRICS_PATH, RANDOM_STATE, TEST_SIZE
from .data_preparation import clean, load_raw, make_xy, save_processed
from .evaluation import (
    cross_validate_on_train,
    evaluate_on,
    plot_confusion_matrix,
    plot_roc,
)
from .preprocessing import build_dummy_baseline, build_logistic_baseline


def main() -> None:
    # 1. Load + clean (rule-based, nothing learned from data)
    raw = load_raw()
    df, cleaning_log = clean(raw)
    save_processed(df)
    print("Cleaning log:", json.dumps(cleaning_log, indent=2))

    # 2. Features / target
    X, y = make_xy(df)

    # 3. Split BEFORE any fitting; stratify because the classes are imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {X_train.shape}, churn rate {y_train.mean():.3f}")
    print(f"Test : {X_test.shape}, churn rate {y_test.mean():.3f}")

    # 4. Cross-validation on the training set only
    dummy = build_dummy_baseline(X_train)
    logreg = build_logistic_baseline(X_train)
    cv_dummy = cross_validate_on_train(dummy, X_train, y_train)
    cv_logreg = cross_validate_on_train(logreg, X_train, y_train)
    print("\nCV (train only) - dummy:\n", cv_dummy.round(4))
    print("\nCV (train only) - logistic regression:\n", cv_logreg.round(4))

    # 5. Final fit on the full training set, single evaluation on the test set
    dummy.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    dummy_metrics, _, _ = evaluate_on(dummy, X_test, y_test)
    test_metrics, y_pred, y_proba = evaluate_on(logreg, X_test, y_test)
    print("\nTest metrics - dummy:", {k: round(v, 4) for k, v in dummy_metrics.items()})
    print("Test metrics - logistic regression:", {k: round(v, 4) for k, v in test_metrics.items()})

    # 6. Save artefacts
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cleaning_log": cleaning_log,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
        "cv_train_dummy": cv_dummy.round(4).to_dict(),
        "cv_train_logreg": cv_logreg.round(4).to_dict(),
        "test_dummy": {k: round(float(v), 4) for k, v in dummy_metrics.items()},
        "test_logreg": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "confusion_matrix_logreg": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    plot_confusion_matrix(y_test, y_pred, "Baseline: confusion matrix (test)", FIGURES_DIR / "baseline_confusion_matrix.png")
    plot_roc(y_test, y_proba, "Baseline: ROC curve (test)", FIGURES_DIR / "baseline_roc_curve.png")
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
