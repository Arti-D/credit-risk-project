import sqlite3
from pathlib import Path

import pandas as pd
from scipy import stats

DB_PATH = Path("data/processed/credit.db")
QUERIES_PATH = Path("sql/queries.sql")


def run_sql_queries(conn: sqlite3.Connection) -> None:
    raw = QUERIES_PATH.read_text(encoding="utf-8")
    statements = [s.strip() for s in raw.split(";") if s.strip() and not s.strip().startswith("--")]

    for i, stmt in enumerate(statements, 1):
        try:
            df = pd.read_sql_query(stmt, conn)
            print(f"\n--- Запрос {i} ---")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Запрос {i} пропущен (ошибка: {e})")


def hypothesis_tests(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("ПРОВЕРКА ГИПОТЕЗ")
    print("=" * 50)

    default_mask = df["serious_dlqin2yrs"] == 1

    # H1: возраст
    age_default = df.loc[default_mask, "age"].dropna()
    age_no_default = df.loc[~default_mask, "age"].dropna()
    t_stat, p_val = stats.ttest_ind(age_default, age_no_default, equal_var=False)
    print(f"\nH1 (возраст, Welch t-test): t={t_stat:.3f}, p={p_val:.4g}")
    print("  -> Значимо" if p_val < 0.05 else "  -> Не значимо", "(alpha=0.05)")

    # H2: возрастная группа vs дефолт (chi-square)
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 45, 60, 120], labels=["18-29", "30-44", "45-59", "60+"]
    )
    contingency = pd.crosstab(df["age_group"], df["serious_dlqin2yrs"])
    chi2, p_val_chi, dof, _ = stats.chi2_contingency(contingency)
    print(f"\nH2 (возрастная группа x дефолт, chi-square): chi2={chi2:.3f}, p={p_val_chi:.4g}")
    print("  -> Значимо" if p_val_chi < 0.05 else "  -> Не значимо", "(alpha=0.05)")

    # H3: debt_ratio (отфильтровала экстремальные выбросы)
    dr = df[df["debt_ratio"] < df["debt_ratio"].quantile(0.99)]
    dr_default = dr.loc[dr["serious_dlqin2yrs"] == 1, "debt_ratio"].dropna()
    dr_no_default = dr.loc[dr["serious_dlqin2yrs"] == 0, "debt_ratio"].dropna()
    t_stat2, p_val2 = stats.ttest_ind(dr_default, dr_no_default, equal_var=False)
    print(f"\nH3 (debt_ratio, Welch t-test, без топ-1% выбросов): t={t_stat2:.3f}, p={p_val2:.4g}")
    print("  -> Значимо" if p_val2 < 0.05 else "  -> Не значимо", "(alpha=0.05)")


def basic_overview(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("ОБЩИЙ ОБЗОР ДАННЫХ")
    print("=" * 50)
    print(df.info())
    print("\nПропуски по колонкам:")
    print(df.isnull().sum())
    print("\nОписательная статистика:")
    print(df.describe().T)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Не найден {DB_PATH}. Сначала запусти src/load_data.py")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM credit_data", conn)
        basic_overview(df)
        run_sql_queries(conn)
        hypothesis_tests(df)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
