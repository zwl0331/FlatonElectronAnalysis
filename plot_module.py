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

def plot_fit_result(data_np, fit_range, PDF, N, log_scale=False, n_bins = 50):
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data_hist, data_binedge = np.histogram(data_np, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    ax1.hist(data_np, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    ax1.plot(data_bincenter, (PDF.pdf(data_bincenter, norm_range=fit_range) * N* scale).numpy(), '-r')

    # plot the residuals
    residuals = (data_hist - PDF.pdf(data_bincenter, norm_range=fit_range).numpy() * N * scale) / np.sqrt(data_hist)
    ax2.errorbar(data_bincenter, residuals, yerr=np.ones_like(residuals), fmt='o', color='black', markersize=3, capsize=3)
    ax2.set_xlim(ax1.get_xlim())

    if log_scale:
        ax1.set_yscale('log')
    
    return data_bincenter, residuals

def plotmom_data_only(data, fit_range):
    n_bins = 50
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, ax = plt.subplots()
    ax.hist(data, bins=n_bins, range=fit_range, histtype="step", color="black")
    ax.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), fmt="none",
                ecolor="black", capsize=3)
    ax.set_yscale("log")
    ax.set_xlim(fit_range)
    ax.set_xlabel("Reconstructed Momentum [MeV/c]")
    ax.set_ylabel("# events per bin")
    ax.grid(True)
    return fig, ax

def plt_mom_dist_mc_vs_reco(data_mc_np, data_np, filename="plot_mom_distribution.png"):
    fig, ax = plt.subplots()
    ax.hist(data_mc_np, bins=50, range=(93, 108), color="blue", alpha=0.5, label="MC")
    ax.hist(data_np,    bins=50, range=(93, 108), color="red",  alpha=0.5, label="Reco")
    ax.set_xlabel("Momentum [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Momentum Distribution")
    ax.legend()
    plt.savefig(filename)
    return fig, ax

def plot_mom_mc_vs_reco_2d(data_mc_np, data_np, filename="plot_mom_mc_vs_reco_2d.png"):
    fig, ax = plt.subplots()
    ax.hexbin(data_np, data_mc_np, gridsize=50, cmap="Blues", mincnt=1)
    ax.set_xlabel("Reco p [MeV/c]")
    ax.set_ylabel("MC p [MeV/c]")
    ax.set_title("Reco vs MC Momentum")
    plt.savefig(filename)
    return fig, ax

def plot_mom_diff(data_np, data_mc_np, fit_range=(-15, 1), filename="plot_mom_diff.png"):
    diff = data_np - data_mc_np
    fig, ax = plt.subplots()
    ax.hist(diff, bins=50, range=fit_range, histtype="step", color="black")
    ax.set_xlabel("Momentum Difference [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Momentum Difference Distribution")
    ax.grid(True)
    plt.savefig(filename)
    return fig, ax