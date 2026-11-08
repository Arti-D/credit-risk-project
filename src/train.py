"""Fit complete pipelines; compare candidates on validation only."""
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import ROOT, load_splits, preprocessing

MODELS_DIR = ROOT / "models/current"


def build_models(y_train):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    estimators = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=.05,
            scale_pos_weight=ratio, eval_metric="auc", random_state=42, n_jobs=-1),
    }
    return {name: Pipeline([("preprocess", preprocessing()), ("model", model)])
            for name, model in estimators.items()}


def main():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_splits()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # An optional model from a previous tuning run must not silently enter this run.
    tuned = MODELS_DIR / "xgboost_tuned.pkl"
    if tuned.exists():
        raise RuntimeError("Move the previous xgboost_tuned.pkl before starting a new baseline run.")
    rows = []
    for name, model in build_models(y_train).items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        rows.append({"model": name, "roc_auc": roc_auc_score(y_valid, proba),
                     "average_precision": average_precision_score(y_valid, proba)})
        joblib.dump(model, MODELS_DIR / (name + ".pkl"))
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    results.to_csv(MODELS_DIR / "validation_results.csv", index=False)
    raw = ROOT / "data/raw/cs-training.csv"
    metadata = {
        "seed": 42, "split": "60/20/20 stratified",
        "primary_metric": "roc_auc",
        "raw_csv_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "rows": {"train": len(y_train), "validation": len(y_valid), "test": len(y_test)},
        "target_rate": {"train": float(y_train.mean()), "validation": float(y_valid.mean()),
                        "test": float(y_test.mean())},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]},
    }
    (MODELS_DIR / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
