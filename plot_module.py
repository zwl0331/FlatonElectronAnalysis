import numpy as np
from scipy.stats import norm 
import matplotlib.pyplot as plt

def plot_residuals_with_gaussian(residual, bins=15, range=(-4, 4), filename="plot_mom_residuals_with_gaussian.png"):
    """
    Plot a histogram of residuals with an overlaid unit Gaussian and save the figure.

    Parameters
    ----------
    residual : array-like
        Array of residual values to histogram.
    bins : int or sequence, optional
        Number of bins or bin edges for the histogram (default: 15).
    range : tuple, optional
        Lower and upper range of the bins (default: (-4, 4)).
    filename : str, optional
        Path to save the output figure (default: "plot_mom_residuals_with_gaussian.png").

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Figure and axes objects for further customization if needed.
    """
    # histogram and bin edges
    counts, bin_edges = np.histogram(residual, bins=bins, range=range)
    bin_width = bin_edges[1] - bin_edges[0]

    # prepare Gaussian overlay
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, loc=0, scale=1)
    y = pdf * counts.sum() * bin_width

    # plotting
    fig, ax = plt.subplots()
    ax.hist(residual, bins=bins, range=range, histtype="step", color="gray", label="Residuals")
    ax.plot(x, y, color="red", linewidth=2, label="Gaussian (μ=0, σ=1)")
    ax.set_xlabel("Residuals [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Fit Residuals with Gaussian Overlay")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    return fig, ax