"""Train-fitted preprocessing and reproducible stratified data splits."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/processed/credit.db"
TARGET = "serious_dlqin2yrs"
FEATURE_COLUMNS = [
    "revolving_utilization_of_unsecured_lines", "age",
    "number_of_time30_59_days_past_due_not_worse", "debt_ratio",
    "monthly_income", "number_of_open_credit_lines_and_loans",
    "number_of_times90_days_late", "number_real_estate_loans_or_lines",
    "number_of_time60_89_days_past_due_not_worse", "number_of_dependents",
]
LATE_COLUMNS = [c for c in FEATURE_COLUMNS if "past_due" in c or "90_days_late" in c]


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Learn clipping on train only; retain missingness/anomaly flags.

    Delinquency counts >=90 are treated as unknown, not literal counts.
    This is a preprocessing assumption to validate, not a known data definition.
    Unusual ages are flagged and imputed; inference never drops rows.
    """
    def fit(self, X, y=None):
        self.upper_ = X[["debt_ratio", "revolving_utilization_of_unsecured_lines"]].quantile(.99)
        return self

    def transform(self, X):
        frame = X.loc[:, FEATURE_COLUMNS].copy()
        for col in ["monthly_income", "number_of_dependents"]:
            frame[col + "_missing"] = frame[col].isna().astype(int)
        frame["age_unusual"] = ((frame["age"] < 18) | (frame["age"] > 100)).astype(int)
        frame.loc[frame["age_unusual"] == 1, "age"] = np.nan
        for col in LATE_COLUMNS:
            frame[col + "_unusual"] = (frame[col] >= 90).astype(int)
            frame.loc[frame[col] >= 90, col] = np.nan
        for col, upper in self.upper_.items():
            frame[col] = frame[col].clip(upper=upper)
        return frame


def preprocessing():
    return Pipeline([("clean", CreditCleaner()),
                     ("impute", SimpleImputer(strategy="median", keep_empty_features=True))])


def load_frame():
    if not DB_PATH.exists():
        raise FileNotFoundError("Run python src/load_data.py first.")
    with sqlite3.connect(DB_PATH) as conn:
        frame = pd.read_sql_query("SELECT * FROM credit_data", conn)
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1, without missing values.")
    return frame


def split_frame(frame, random_state=42):
    """60% train, 20% validation for model/threshold, 20% final test."""
    X, y = frame[FEATURE_COLUMNS], frame[TARGET].astype(int)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=.2, stratify=y, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=random_state)
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_splits():
    return split_frame(load_frame())


def load_and_split():
    X_train, _, X_test, y_train, _, y_test = load_splits()
    return X_train, X_test, y_train, y_test
