"""Threshold selection on validation; final test reports and figures."""
import argparse
import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score)

from preprocess import ROOT, load_splits
from train import MODELS_DIR

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def bootstrap_ci(y, p, metric, n_bootstrap=200):
    y, p = np.asarray(y), np.asarray(p)
    rng, scores = np.random.default_rng(42), []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            scores.append(metric(y[idx], p[idx]))
    if not scores:
        raise ValueError("No bootstrap sample contains both classes.")
    lo, hi = np.quantile(scores, [.025, .975])
    return float(metric(y, p)), float(lo), float(hi)


def choose_threshold(y, p, target_recall=.8):
    # Largest threshold meeting a predeclared recall objective on validation.
    _, recall, thresholds = precision_recall_curve(y, p)
    feasible = thresholds[recall[:-1] >= target_recall]
    if not len(feasible):
        raise ValueError("No threshold meets the recall target.")
    return float(feasible.max())


def operating_point(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    rejected = p >= threshold
    approved = ~rejected
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, rejected, zero_division=0)),
        "recall": float(recall_score(y, rejected, zero_division=0)),
        "f1": float(f1_score(y, rejected, zero_division=0)),
        "approval_rate": float(approved.mean()),
        "observed_bad_rate_approved": float(y[approved].mean()) if approved.any() else None,
        "true_positives": int(((y == 1) & rejected).sum()),
        "false_positives": int(((y == 0) & rejected).sum()),
        "false_negatives": int(((y == 1) & approved).sum()),
        "true_negatives": int(((y == 0) & approved).sum()),
    }


def main(n_bootstrap=200, target_recall=.8):
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_splits()
    metadata = json.loads((MODELS_DIR / "run.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / "data/raw/cs-training.csv").read_bytes()).hexdigest()
    if digest != metadata["raw_csv_sha256"]:
        raise ValueError("Input CSV differs from the training run. Retrain first.")
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    names = ["logistic_regression", "random_forest", "xgboost"]
    if (MODELS_DIR / "xgboost_tuned.pkl").exists():
        names.append("xgboost_tuned")
    models = {name: joblib.load(MODELS_DIR / (name + ".pkl")) for name in names}
    validation = {name: model.predict_proba(X_valid)[:, 1] for name, model in models.items()}
    valid_rows = [{"model": name, "roc_auc": roc_auc_score(y_valid, p),
                   "average_precision": average_precision_score(y_valid, p)}
                  for name, p in validation.items()]
    valid_table = pd.DataFrame(valid_rows).sort_values("roc_auc", ascending=False)
    winner = str(valid_table.iloc[0]["model"])
    threshold = choose_threshold(y_valid, validation[winner], target_recall)
    predictions = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
    rows = []
    for name, p in predictions.items():
        row = {"model": name}
        for label, metric in [("roc_auc", roc_auc_score), ("average_precision", average_precision_score)]:
            point, lo, hi = bootstrap_ci(y_test, p, metric, n_bootstrap)
            row.update({label: point, label + "_ci_low": lo, label + "_ci_high": hi})
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORTS / "test_metrics.csv", index=False)
    valid_table.to_csv(REPORTS / "validation_metrics.csv", index=False)
    p = predictions[winner]
    policy = {"selected_model": winner, "selection_metric": "validation ROC-AUC",
              "target_validation_recall": target_recall,
              "validation": operating_point(y_valid, validation[winner], threshold),
              "test": operating_point(y_test, p, threshold),
              "bootstrap_replicates": n_bootstrap,
              "test_note": "Evaluate once after freezing model and threshold."}
    (REPORTS / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    # Paired bootstrap: same resampled borrowers for both model predictions.
    rng, differences = np.random.default_rng(42), []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_test), len(y_test))
        yb = y_test.to_numpy()[idx]
        if np.unique(yb).size == 2:
            differences.append(roc_auc_score(yb, p[idx]) -
                               roc_auc_score(yb, predictions["logistic_regression"][idx]))
    lo, hi = np.quantile(differences, [.025, .975])
    comparison = {"model": winner, "reference": "logistic_regression",
                  "delta_roc_auc": roc_auc_score(y_test, p) -
                      roc_auc_score(y_test, predictions["logistic_regression"]),
                  "ci_low": float(lo), "ci_high": float(hi),
                  "note": "Exploratory comparison; zero in CI means no clear difference on this split."}
    (REPORTS / "paired_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    # Scenarios use fixed validation-derived thresholds, not thresholds optimized on test.
    scenarios = []
    for target in [.5, .65, .8, .9]:
        t = choose_threshold(y_valid, validation[winner], target)
        scenarios.append({"target_validation_recall": target, **operating_point(y_test, p, t)})
    pd.DataFrame(scenarios).to_csv(REPORTS / "threshold_scenarios.csv", index=False)
    pd.DataFrame({"row_index": X_test.index, "target": y_test.to_numpy(),
                  **predictions}).to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titlesize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, proba in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        precision, recall, _ = precision_recall_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Prevalence")
    axes[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC · final test")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall · final test")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "discrimination.png", dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(predicted, observed, "o-", label=winner)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="Mean predicted score", ylabel="Observed event rate",
           title="Calibration diagnostic · weighted model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=160)
    plt.close(fig)

    # Risk-band cutoffs learned from validation scores and then frozen.
    cuts = np.unique(np.quantile(validation[winner], [.2, .4, .6, .8]))
    bands = np.searchsorted(cuts, p, side="right") + 1
    risk = pd.DataFrame({"risk_band": bands, "target": y_test.to_numpy(), "score": p})
    table = risk.groupby("risk_band").agg(clients=("target", "size"),
        bad_rate=("target", "mean"), mean_score=("score", "mean"))
    table.to_csv(REPORTS / "risk_bands.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table.index.astype(str), table["bad_rate"] * 100, color="#247e91")
    ax.set(xlabel="Risk band (low → high)", ylabel="Observed event rate, %",
           title="Portfolio segmentation · final test")
    fig.tight_layout()
    fig.savefig(FIGURES / "risk_bands.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(policy, indent=2))
    print("Reports saved to:", REPORTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--target-recall", type=float, default=.8)
    args = parser.parse_args()
    if args.bootstrap < 20 or not 0 < args.target_recall <= 1:
        parser.error("Use at least 20 bootstrap replicates and recall in (0, 1].")
    main(args.bootstrap, args.target_recall)
