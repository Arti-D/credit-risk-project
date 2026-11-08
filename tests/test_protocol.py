"""Small synthetic checks; no model training and no real data required."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import CreditCleaner, FEATURE_COLUMNS, TARGET, split_frame
from evaluate import choose_threshold, operating_point


class ProtocolTests(unittest.TestCase):
    def frame(self, n=100):
        frame = pd.DataFrame({col: np.arange(n, dtype=float) for col in FEATURE_COLUMNS})
        frame["age"] = 40.
        frame[TARGET] = np.arange(n) % 5 == 0
        return frame

    def test_test_extremes_do_not_change_fitted_bounds(self):
        train = self.frame()
        cleaner = CreditCleaner().fit(train)
        bound = cleaner.upper_.copy()
        test = self.frame(4)
        test.loc[0, "debt_ratio"] = 1e12
        transformed = cleaner.transform(test)
        pd.testing.assert_series_equal(bound, cleaner.upper_)
        self.assertEqual(transformed.loc[0, "debt_ratio"], bound["debt_ratio"])

    def test_splits_are_disjoint_and_preserve_rows(self):
        splits = split_frame(self.frame())
        a, b, c = [set(x.index) for x in splits[:3]]
        self.assertFalse(a & b or a & c or b & c)
        self.assertEqual(len(a | b | c), 100)
        self.assertEqual([len(a), len(b), len(c)], [60, 20, 20])

    def test_flags_preserve_unknown_counts_and_missing_income(self):
        frame = self.frame(4)
        frame.loc[0, "monthly_income"] = np.nan
        frame.loc[0, "number_of_times90_days_late"] = 98
        result = CreditCleaner().fit(frame).transform(frame)
        self.assertEqual(len(result), len(frame))
        self.assertEqual(result.loc[0, "monthly_income_missing"], 1)
        self.assertEqual(result.loc[0, "number_of_times90_days_late_unusual"], 1)
        self.assertTrue(np.isnan(result.loc[0, "number_of_times90_days_late"]))

    def test_threshold_meets_validation_recall_with_ties(self):
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([.1, .4, .4, .8, .9])
        threshold = choose_threshold(y, p, .8)
        self.assertGreaterEqual(operating_point(y, p, threshold)["recall"], .8)
        self.assertEqual(threshold, .4)

    def test_no_approved_borrowers_has_no_defined_bad_rate(self):
        result = operating_point(np.array([0, 1]), np.array([.5, .9]), 0)
        self.assertEqual(result["approval_rate"], 0)
        self.assertIsNone(result["observed_bad_rate_approved"])


if __name__ == "__main__":
    unittest.main()
