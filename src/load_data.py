import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = ROOT / "data/raw/cs-training.csv"
DB_PATH = ROOT / "data/processed/credit.db"
SCHEMA_PATH = ROOT / "sql/schema.sql"

COLUMN_MAPPING = {
    "Unnamed: 0": "id",
    "SeriousDlqin2yrs": "serious_dlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization_of_unsecured_lines",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "number_of_time30_59_days_past_due_not_worse",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "number_of_open_credit_lines_and_loans",
    "NumberOfTimes90DaysLate": "number_of_times90_days_late",
    "NumberRealEstateLoansOrLines": "number_real_estate_loans_or_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "number_of_time60_89_days_past_due_not_worse",
    "NumberOfDependents": "number_of_dependents",
}


def main() -> None:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {RAW_CSV_PATH}. Скачай датасет с Kaggle и положи сюда: "
            "https://www.kaggle.com/c/GiveMeSomeCredit/data"
        )

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Читаю {RAW_CSV_PATH} ...")
    df = pd.read_csv(RAW_CSV_PATH)
    df = df.rename(columns=COLUMN_MAPPING)

    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    from preprocess import FEATURE_COLUMNS, TARGET
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Borrower IDs must be unique and nonmissing.")
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError("Expected labeled training data with a binary target.")
    df = df[["id", TARGET] + FEATURE_COLUMNS]
    print(f"Строк: {len(df)}, колонок: {len(df.columns)}")

    conn = sqlite3.connect(DB_PATH)
    try:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        conn.executescript(schema_sql)
        df.to_sql("credit_data", conn, if_exists="append", index=False)
        conn.commit()
        print(f"Данные загружены в {DB_PATH}, таблица credit_data")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
