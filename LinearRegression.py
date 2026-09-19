"""Reusable linear-regression functions for CS260 HW03."""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

FloatArray = NDArray[np.float64]


def add_ones(X: FloatArray) -> FloatArray:
    """Return X with a leading intercept column of ones."""
    #print("\nAdd Ones Testing\n")
    #print("X input: " + str(X) + "\n")
    s = X.shape
    #print("X shape: " + str(s) + "\n")
    ones_col = np.ones((s[0], 1), dtype = float)
    #print("ones col shape" + str(ones_col.shape) )
    new_x = np.concatenate((ones_col, X), axis = 1)
    #print("X with ones: " + str(new_x) + "\n")
    return new_x


def fit(X: FloatArray, y: FloatArray) -> FloatArray:
    """Return analytic least-squares weights, including the intercept."""
    #print("\nTesting fit\n")
    #print("X initially: " + str(X) + "\n")
    X = add_ones(X)
    xt = np.transpose(X)
    #print("xt: " + str(xt) + "\n")
    #print("y " + str(y) + "y shape: " + str(y.shape))
    pseudoinverse = np.linalg.pinv(np.matmul(xt, X))
    w = np.matmul(pseudoinverse, (np.matmul(xt,y)))
    #print("w: " + str(w) + "w shape: " + str(w.shape) + "\n")
    return w


def predict(X: FloatArray, w: FloatArray) -> FloatArray:
    """Return one prediction per row of X using weights w."""
    #print("\n Testing Predict \n")
    #print("w: " + str(w) + "\n")
    X_ones = add_ones(X)
    #print("X: " + str(X_ones) + "shape" + str(X_ones.shape)+ "\n")
    y_hat = np.matmul(X_ones, w)
    #print("y_hat"  + str(y_hat) + "shape" + str(y_hat.shape)+ "\n")
    return y_hat


def cost(X: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """Return one-half the sum of squared prediction errors."""
    # TODO: Use predict() and vectorized NumPy operations.
    y_hat = predict(X, w)
    a = np.subtract(y, y_hat) 
    b = a ** 2
    c = float(b.sum())
    d = 1/2 * c
    return d


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
