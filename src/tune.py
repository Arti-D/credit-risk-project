from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from preprocess import load_and_split

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

N_TRIALS = 30


def objective(trial: optuna.Trial, X_train, y_train, scale_pos_weight: float) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def main() -> None:
    X_train, X_test, y_train, y_test = load_and_split()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\nЛучшие параметры:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Лучший CV ROC-AUC: {study.best_value:.4f}")

    best_params = study.best_params
    best_params.update(
        {"scale_pos_weight": scale_pos_weight, "eval_metric": "auc", "random_state": 42, "n_jobs": -1}
    )
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train)

    joblib.dump(best_model, MODELS_DIR / "xgboost_tuned.pkl")
    print("\nЛучшая модель сохранена в models/xgboost_tuned.pkl")


if __name__ == "__main__":
    main()
