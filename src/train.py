from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import load_and_split

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def evaluate(name: str, model, X_test, y_test, proba) -> dict:
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"{name:22s} ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    return {"model": name, "roc_auc": roc_auc, "pr_auc": pr_auc}


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()

    results = []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_reg.fit(X_train_scaled, y_train)
    proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("LogisticRegression", log_reg, X_test, y_test, proba))
    joblib.dump({"model": log_reg, "scaler": scaler}, MODELS_DIR / "logreg.pkl")

    # --- Random Forest (бэггинг) ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("RandomForest", rf, X_test, y_test, proba))
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    proba = xgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("XGBoost", xgb, X_test, y_test, proba))
    joblib.dump(xgb, MODELS_DIR / "xgboost_baseline.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODELS_DIR / "baseline_results.csv", index=False)
    print("\nРезультаты сохранены в models/baseline_results.csv")
    print(results_df)


if __name__ == "__main__":
    main()
