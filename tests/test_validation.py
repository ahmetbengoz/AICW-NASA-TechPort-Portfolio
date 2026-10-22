"""Scientific invariants for the retrospective validation extension."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_validation_extension import (rarity_transform, training_normalize,
                                      objective_profiles, selection_value,
                                      split_plan, learned_profile)
from run_analysis import FEATURES, prepare_data, marcos_scores


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        _, cls.data, _ = prepare_data(cls.root / "data/raw/techport_projects_batch1.csv")

    def test_source_sample_sizes_and_bounds(self):
        self.assertEqual(len(self.data), 201)
        self.assertTrue((self.data.realized_trl_progress <= self.data.planned_trl_gain).all())

    def test_chronological_dates_never_cross(self):
        for mode, repeat, fold, tr, te in split_plan(self.data):
            self.assertFalse(set(tr) & set(te))
            self.assertEqual(len(tr) + len(te), len(self.data))
            if mode == "chronological":
                self.assertLess(self.data.iloc[tr].startDate.max(), self.data.iloc[te].startDate.min())

    def test_programme_ids_never_cross(self):
        for mode, repeat, fold, tr, te in split_plan(self.data):
            if mode == "programme":
                self.assertFalse(set(self.data.iloc[tr].programTitle) & set(self.data.iloc[te].programTitle))

    def test_rarity_never_reads_candidate_counts(self):
        train = self.data.iloc[:120]
        candidate = self.data.iloc[120:125]
        a = rarity_transform(train, candidate)
        expanded = pd.concat([candidate, candidate] * 5, ignore_index=True)
        b = rarity_transform(train, expanded).iloc[:len(candidate)]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())

    def test_candidate_extreme_does_not_rescale_existing_candidate(self):
        train = rarity_transform(self.data.iloc[:120], self.data.iloc[:120])
        test = rarity_transform(self.data.iloc[:120], self.data.iloc[120:125])
        original = training_normalize(train, test)
        more = pd.concat([test, test.iloc[:1] * 100], ignore_index=True)
        expanded = training_normalize(train, more)[:len(test)]
        np.testing.assert_allclose(original, expanded)

    def test_objective_profiles_are_simplex_vectors(self):
        for w in objective_profiles(self.data[FEATURES]).values():
            self.assertTrue((w >= 0).all())
            self.assertAlmostEqual(w.sum(), 1)

    def test_all_tied_scores_equal_random_selection(self):
        y = np.array([0., 1., 2., 3., 4.])
        result = selection_value(y, np.ones(5), .2)
        self.assertAlmostEqual(result["selected_mean"], y.mean())
        self.assertAlmostEqual(result["lift"], 0)

    def test_fractional_tie_allocation(self):
        result = selection_value(np.array([0., 2., 4.]), np.array([0., 1., 1.]), 1/3)
        self.assertAlmostEqual(result["selected_mean"], 3)

    def test_perfect_order_has_zero_oracle_gap(self):
        y = np.array([0., 1., 2., 3., 4.])
        self.assertAlmostEqual(selection_value(y, y, .4)["oracle_gap"], 0)

    def test_unsigned_importance_does_not_identify_direction(self):
        x = np.linspace(0, 1, 21)
        y = 1-x
        # Perfect knowledge of f(x)=1-x still gives the one-feature weight 1.
        # Treating x as a benefit then selects the smallest observed outcome.
        self.assertLess(selection_value(y, x, .2)["lift"], 0)
        self.assertGreater(selection_value(y, y, .2)["lift"], 0)

    def test_marcos_has_weighted_sum_order_for_fixed_universe(self):
        rng = np.random.default_rng(83)
        matrix = rng.uniform(.01, 1, (50, 6))
        weights = rng.dirichlet(np.ones(6))
        np.testing.assert_array_equal(np.argsort(matrix @ weights),
                                      np.argsort(marcos_scores(matrix, weights)))

    def test_test_outcome_not_an_input_to_profile_learning(self):
        train = self.data.iloc[:50].copy()
        raw, w, fallback = learned_profile(train, "realized_trl_progress", "random", 99)
        self.assertEqual(len(w), len(FEATURES))
        self.assertTrue(np.isfinite(w).all())
        self.assertAlmostEqual(w.sum(), 1)

    def test_generated_selections_respect_oracle(self):
        path = self.root / "outputs/validation_extension/shortlist_fold_results.csv"
        if path.exists():
            x = pd.read_csv(path)
            self.assertTrue((x.oracle_gap >= -1e-10).all())
            self.assertTrue(((x.ndcg >= -1e-10) & (x.ndcg <= 1+1e-10)).all())


if __name__ == "__main__":
    unittest.main()
