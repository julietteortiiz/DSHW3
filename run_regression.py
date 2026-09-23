"""Command-line driver for CS260 HW03."""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from LinearRegression import cost, fit, fit_SGD_with_history

FloatArray = NDArray[np.float64]

FIGURES_DIR = Path("figures")


def parse_arguments(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="Linear Regression",
        description="Run analytic and stochastic linear regression",
    )
    parser.add_argument(
        "-d",
        "--data_filename",
        type=Path,
        required=True,
        help="path to a CSV data file",
    )
    return parser.parse_args(arguments)

# source https://www.datacamp.com/doc/numpy/reading-csv-files-into-numpy
def load_data(filename: Path) -> Tuple[FloatArray, FloatArray]:
    """Load a CSV and return its feature matrix and response vector."""
    data = np.genfromtxt(filename, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def normalize(values: FloatArray) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Return normalized values, their column means, and standard deviations."""
    # TODO: Compute means and standard deviations along axis 0.
    X = values
    X_mean = X.mean(axis = 0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std
    return X_normalized, X_mean, X_std



def plot_cost_history(cost_history: Sequence[float], output_path: Path) -> None:
    """Save a labeled plot of cost versus epoch."""
    # TODO: Create the parent directory and save the required figure.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    x = np.linspace(start = 1, stop = len(cost_history), num = len(cost_history))
    ax.plot(x, cost_history)
    ax.set(xlabel='epochs', ylabel='cost J(w)',
       title='SGD Cost History')
    ax.grid()
    fig.savefig(output_path)



def main() -> None:
    """Load data, fit both models, and report the required results."""
    args = parse_arguments()
    X, y = load_data(args.data_filename)

    # Normalize the housing data before fitting either model. You may choose a
    # clear, documented way to determine when normalization is appropriate.
    # TODO: Normalize X and y for the USA Housing experiment.
    
    normalized_X = normalize(X)
    normalized_y = normalize(y)

    X = normalized_X[0]
    y = normalized_y[0]


    analytic_weights = fit(X, y)
    analytic_cost = cost(X, y, analytic_weights)
    print(f"analytic cost: {analytic_cost}")
    print(f"analytic weights: {analytic_weights}")

    alpha = 0.0001
    eps = 1e-10
    tmax = 100

    # TODO: Tune these values for the normalized housing data.
    sgd_weights, cost_history, completed_epochs = fit_SGD_with_history(
        X, y, alpha=alpha, eps=eps, tmax=tmax
    )
    sgd_cost = cost(X, y, sgd_weights)

    print(f"alpha: {alpha}")
    print(f"epsilon: {eps}")
    print(f"maximum epochs: {tmax}")
    print(f"completed epochs: {completed_epochs}")
    print(f"SGD cost: {sgd_cost}")
    print(f"SGD weights: {sgd_weights}")

    plot_cost_history(cost_history, FIGURES_DIR / "cost_J.pdf")


if __name__ == "__main__":
    main()
