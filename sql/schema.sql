DROP TABLE IF EXISTS credit_data;

CREATE TABLE credit_data (
    id                                          INTEGER PRIMARY KEY,
    serious_dlqin2yrs                           INTEGER,
    revolving_utilization_of_unsecured_lines    REAL,
    age                                         INTEGER,
    number_of_time30_59_days_past_due_not_worse REAL,
    debt_ratio                                  REAL,
    monthly_income                              REAL,
    number_of_open_credit_lines_and_loans       INTEGER,
    number_of_times90_days_late                 REAL,
    number_real_estate_loans_or_lines           INTEGER,
    number_of_time60_89_days_past_due_not_worse REAL,
    number_of_dependents                        REAL
);

CREATE INDEX idx_age ON credit_data(age);
CREATE INDEX idx_target ON credit_data(serious_dlqin2yrs);
