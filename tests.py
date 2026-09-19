"""Public tests for CS260 HW03.

Run these tests from the HW03 folder with:

    python3 tests.py

These tests are only a sample of the Gradescope tests. Passing them does not
guarantee full credit, so also test your functions with examples of your own.
"""

import unittest

import numpy as np

from LinearRegression import add_ones, cost, fit, fit_SGD, predict
from run_regression import normalize


class TestLinearRegression(unittest.TestCase):
    def test_add_ones(self) -> None:
        """The intercept column should be added without changing the input."""
        X = np.array([[2.0, 3.0], [4.0, 5.0]])
        original = X.copy()

        actual = add_ones(X)
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])

        self.assertTrue(
            np.array_equal(actual, expected),
            "add_ones() should prepend one column of 1s.",
        )
        self.assertTrue(
            np.array_equal(X, original),
            "add_ones() should not modify its input array.",
        )

    def test_analytic_fit_predict_and_cost(self) -> None:
        """The analytic solution should recover a simple exact line."""
        X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        y = 3.0 + 2.5 * X[:, 0]
        print("Y in test afpc: " + str(y))

        weights = fit(X, y)

        self.assertEqual(
            np.asarray(weights).shape,
            (2,),
            "fit() should return the intercept followed by one feature weight.",
        )
        self.assertTrue(
            np.allclose(weights, [3.0, 2.5], atol=1e-8),
            "fit() did not recover the expected intercept and slope.",
        )
        self.assertTrue(
            np.allclose(predict(X, weights), y),
            "predict() should return one prediction for each row of X.",
        )
        self.assertAlmostEqual(
            cost(X, y, weights),
            0.0,
            places=10,
            msg="An exact model should have zero cost.",
        )

    def test_sgd_converges_on_a_small_dataset(self) -> None:
        """SGD should approach known weights on a centered dataset."""
        X = np.array([[-1.5], [-0.5], [0.5], [1.5]])
        y = 1.25 - 0.75 * X[:, 0]

        weights = np.asarray(
            fit_SGD(X, y, alpha=0.05, eps=1e-12, tmax=5_000)
        )

        self.assertEqual(
            weights.shape,
            (2,),
            "fit_SGD() should return one intercept and one feature weight.",
        )
        self.assertTrue(
            np.allclose(weights, [1.25, -0.75], atol=0.05),
            "SGD did not converge. Check the update sign, intercept, and stopping rule.",
        )
        self.assertLess(
            cost(X, y, weights),
            0.01,
            "The fitted SGD weights should give a small cost on this dataset.",
        )

    def test_normalize(self) -> None:
        """Every normalized feature should have mean 0 and standard deviation 1."""
        values = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])

        normalized, means, standard_deviations = normalize(values)

        self.assertTrue(
            np.allclose(means, [3.0, 14.0]),
            "normalize() returned incorrect column means.",
        )
        self.assertTrue(
            np.allclose(normalized.mean(axis=0), 0.0),
            "Normalized columns should have mean 0.",
        )
        self.assertTrue(
            np.allclose(normalized.std(axis=0), 1.0),
            "Normalized columns should have standard deviation 1.",
        )
        self.assertTrue(
            np.all(standard_deviations > 0),
            "Standard deviations should be positive for these nonconstant columns.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
