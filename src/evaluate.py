from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import load_and_split

MODELS_DIR = Path("models")
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def bootstrap_ci(y_true, y_proba, metric_fn, n_bootstrap=N_BOOTSTRAP, ci=0.95):
    rng = np.random.RandomState(RANDOM_STATE)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_proba[idx]))

    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lower), float(upper)


def evaluate_model(name: str, proba, y_test, threshold: float = 0.5) -> dict:
    preds = (proba >= threshold).astype(int)

    roc_auc, roc_lo, roc_hi = bootstrap_ci(y_test, proba, roc_auc_score)
    pr_auc, pr_lo, pr_hi = bootstrap_ci(y_test, proba, average_precision_score)

    row = {
        "model": name,
        "roc_auc": roc_auc,
        "roc_auc_ci": f"[{roc_lo:.4f}, {roc_hi:.4f}]",
        "pr_auc": pr_auc,
        "pr_auc_ci": f"[{pr_lo:.4f}, {pr_hi:.4f}]",
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    print(f"\n{name}")
    for k, v in row.items():
        if k != "model":
            print(f"  {k}: {v}")
    return row


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()

    results = []

    logreg_bundle = joblib.load(MODELS_DIR / "logreg.pkl")
    proba = logreg_bundle["model"].predict_proba(logreg_bundle["scaler"].transform(X_test))[:, 1]
    results.append(evaluate_model("LogisticRegression", proba, y_test))

    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate_model("RandomForest", proba, y_test))

    tuned_path = MODELS_DIR / "xgboost_tuned.pkl"
    xgb_path = tuned_path if tuned_path.exists() else MODELS_DIR / "xgboost_baseline.pkl"
    xgb = joblib.load(xgb_path)
    proba = xgb.predict_proba(X_test)[:, 1]
    label = "XGBoost (tuned)" if tuned_path.exists() else "XGBoost (baseline)"
    results.append(evaluate_model(label, proba, y_test))

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODELS_DIR / "final_results.csv", index=False)
    print("\nИтоговая таблица сохранена в models/final_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
