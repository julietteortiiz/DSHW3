"""Reusable linear-regression functions for CS260 HW03."""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

FloatArray = NDArray[np.float64]


def add_ones(X: FloatArray) -> FloatArray:
    """Return X with a leading intercept column of ones."""
    s = X.shape
    ones_vector = np.ones((s[0], 1), dtype = float)
    new_x = np.concatenate((ones_vector, X), axis = 1)
    return new_x


def fit(X: FloatArray, y: FloatArray) -> FloatArray:
    """Return analytic least-squares weights, including the intercept."""
    X = add_ones(X)
    xt = np.transpose(X)
    return np.matmul(np.linalg.pinv(np.matmul(xt, X)), (np.matmul(xt,y)))


def predict(X: FloatArray, w: FloatArray) -> FloatArray:
    """Return one prediction per row of X using weights w."""
    return np.matmul(add_ones(X), w)


def cost(X: FloatArray, y: FloatArray, w: FloatArray) -> float:
    """Return one-half the sum of squared prediction errors."""
    y_hat = predict(X, w)
    b = (y - y_hat) ** 2
    return 1/2 * float(b.sum())


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
    print("final weights: " + str(weights))
    print("should be [1.25, -0.75]")
    return weights


def fit_SGD_with_history(
    X: FloatArray,
    y: FloatArray,
    alpha: float,
    eps: float = 1e-10,
    tmax: int = 10_000,
) -> Tuple[FloatArray, List[float], int]:
    """Return SGD weights, the cost after every epoch, and epoch count."""
    # 1. Add the intercept column and initialize all weights to zero.

    X_ones = add_ones(X)
    s = X_ones.shape
    m = s[0] #number of rows in X_ones
    n = s[1] #number of columns in X_ones
    w = np.zeros((n), dtype = float) # vector with n rows

    # 2. During each epoch, update the weights once per example.
    costs = [0]
    for i in range(tmax):

        for j in range(m):
            x_i = X_ones[j]
            y_i = y[j]
            w = w - alpha * (np.matmul(w,x_i) - y_i) * x_i

        # 3. Append the cost after each epoch and check for convergence.
        c = cost(X, y, w)
        diff = abs((costs[-1]) - c)

        if diff < eps:
            costs.append(c)
            return w, costs[1:], i + 1
            
        costs.append(c)
                

    # 4. Return the final weights, cost history, and completed epoch count.
    return w, costs[1:], tmax
    
    
