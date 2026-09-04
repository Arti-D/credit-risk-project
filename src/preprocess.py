import sqlite3
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DB_PATH = Path("data/processed/credit.db")
TARGET = "serious_dlqin2yrs"

FEATURE_COLUMNS = [
    "revolving_utilization_of_unsecured_lines",
    "age",
    "number_of_time30_59_days_past_due_not_worse",
    "debt_ratio",
    "monthly_income",
    "number_of_open_credit_lines_and_loans",
    "number_of_times90_days_late",
    "number_real_estate_loans_or_lines",
    "number_of_time60_89_days_past_due_not_worse",
    "number_of_dependents",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # медиана
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
    df["number_of_dependents"] = df["number_of_dependents"].fillna(0)

    # выбросы
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    df["debt_ratio"] = df["debt_ratio"].clip(upper=df["debt_ratio"].quantile(0.99))
    df["revolving_utilization_of_unsecured_lines"] = df[
        "revolving_utilization_of_unsecured_lines"
    ].clip(upper=df["revolving_utilization_of_unsecured_lines"].quantile(0.99))

    for col in [
        "number_of_time30_59_days_past_due_not_worse",
        "number_of_times90_days_late",
        "number_of_time60_89_days_past_due_not_worse",
    ]:
        median_val = df.loc[df[col] < 90, col].median()
        df.loc[df[col] >= 90, col] = median_val

    return df


def load_and_split(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM credit_data", conn)
    finally:
        conn.close()

    df = clean(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Доля дефолтов в train: {y_train.mean():.4f}")
    print(f"Доля дефолтов в test: {y_test.mean():.4f}")
