from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from .forecast.base import LitBase

def plot_rnn_forecasts(model: LitBase, actual: np.ndarray, predicted: np.ndarray):
    n_samples, n_forecasts = actual.shape

    # Create subplots with one row per forecast day
    fig, axes = plt.subplots(n_forecasts, 1, figsize=(12, 2 * n_forecasts), sharex=True)

    # Plot for each forecast day
    for i in range(n_forecasts):
        ax: Axes = axes[i]
        ax.plot(range(n_samples), actual[:, i], label=f'Actual t+{i+1}', color='b')
        ax.plot(range(n_samples), predicted[:, i], label=f'Predicted t+{i+1}', color='r')

        # Labels and legend
        ax.set_ylabel(f'Day {i+1} Close Price')
        ax.legend(loc='upper left')
        ax.grid()

    # Common x-axis label and title
    axes[-1].set_xlabel('Samples')
    plt.suptitle(f'Actual vs Predicted Close Prices for {model} Model ({n_forecasts} Days)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.show()