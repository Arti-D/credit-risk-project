# Experiment reports

Run `python src/train.py`, then `python src/evaluate.py` **on your own machine**, after downloading the labeled CSV.

The evaluation creates:
- `validation_metrics.csv`: model selection by validation ROC-AUC.
- `test_metrics.csv`: direct test ROC-AUC / average precision and bootstrap intervals.
- `policy.json`: validation-derived threshold, test confusion counts, approval rate and observed bad rate among approved applicants.
- `threshold_scenarios.csv`: test descriptions at thresholds frozen on validation.
- `risk_bands.csv`: validation-defined score bands and observed test event rates.
- `paired_comparison.json`: exploratory paired bootstrap difference against logistic regression.
- `figures/`: ROC/PR, calibration diagnostic and risk-band plots.

These are **not generated or populated with invented values** in the repository.
The first quick report uses 200 bootstrap replicates; `--bootstrap 1000` gives a less Monte Carlo-noisy interval and takes longer. Freeze the model and threshold before looking at test results.

Borrower-level predictions and fitted pipelines stay in ignored `models/current/`.
The old CSV files in `models/` refer to the previous preprocessing and split, not this protocol.
