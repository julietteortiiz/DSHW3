"""Reusable linear-regression functions for CS260 HW03."""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

FloatArray = NDArray[np.float64]


def add_ones(X: FloatArray) -> FloatArray:
    """Return X with a leading intercept column of ones."""
    s = X.shape
    ones_col = np.ones((s[1], 1), dtype = float)
    return np.concatenate((ones_col, X), axis = 1)


def fit(X: FloatArray, y: FloatArray) -> FloatArray:
    """Return analytic least-squares weights, including the intercept."""
    # TODO: Add the intercept column and implement (X^T X)^+ X^T y.
    X = add_ones(X)
    xt = np.transpose(X)
    pseudoinverse = np.linalg.pinv(np.matmul(xt, X))
    return pseudoinverse + (np.matmul(xt,y))


def predict(X: FloatArray, w: FloatArray) -> FloatArray:
    """Return one prediction per row of X using weights w."""
    # TODO: Add the intercept column exactly once, then multiply by w.
    raise NotImplementedError


def cost(X: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """Return one-half the sum of squared prediction errors."""
    # TODO: Use predict() and vectorized NumPy operations.
    raise NotImplementedError


def fit_SGD(
    X: FloatArray,
    y: FloatArray,
    alpha: float,
    eps: float = 1e-10,
    tmax: int = 10_000,
) -> FloatArray:
    """Fit linear regression with SGD and return the final weights.

    Stop when consecutive epoch costs differ by less than eps or after tmax
    epochs. One loop over epochs and one nested loop over examples are allowed.
    """
    weights, _, _ = fit_SGD_with_history(X, y, alpha, eps, tmax)
    return weights


def fit_SGD_with_history(
    X: FloatArray,
    y: FloatArray,
    alpha: float,
    eps: float = 1e-10,
    tmax: int = 10_000,
) -> Tuple[FloatArray, List[float], int]:
    """Return SGD weights, the cost after every epoch, and epoch count."""
    # TODO:
    # 1. Add the intercept column and initialize all weights to zero.
    # 2. During each epoch, update the weights once per example.
    # 3. Append the cost after each epoch and check for convergence.
    # 4. Return the final weights, cost history, and completed epoch count.
    raise NotImplementedError
