"""Optional, computationally expensive. Preprocessing is fitted inside CV."""
import joblib
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from preprocess import load_splits, preprocessing
from train import MODELS_DIR


def main():
    X_train, _, _, y_train, _, _ = load_splits()
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    def make_model(params):
        return Pipeline([("preprocess", preprocessing()), ("model", XGBClassifier(
            **params, scale_pos_weight=ratio, eval_metric="auc", random_state=42, n_jobs=1))])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", .01, .2, log=True),
            "subsample": trial.suggest_float("subsample", .6, 1.),
            "colsample_bytree": trial.suggest_float("colsample_bytree", .6, 1.),
            "reg_lambda": trial.suggest_float("reg_lambda", .1, 10., log=True),
        }
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        return cross_val_score(make_model(params), X_train, y_train, cv=cv,
                               scoring="roc_auc", n_jobs=-1).mean()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)
    joblib.dump(make_model(study.best_params).fit(X_train, y_train),
                MODELS_DIR / "xgboost_tuned.pkl")
    study.trials_dataframe().to_csv(MODELS_DIR / "optuna_trials.csv", index=False)
    print("CV ROC-AUC:", study.best_value)


if __name__ == "__main__":
    main()
