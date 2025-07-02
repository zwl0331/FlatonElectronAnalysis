import numpy as np
from scipy.stats import chi2 as chi2_dist

from plot_module import plot_residuals_with_gaussian

def compute_chi2(residual, output_file="fit_mom_residuals.txt"):
    """
    Compute chi-squared of residuals against a zero-mean unit-variance Gaussian,
    print and serialize the results to file.

    Parameters
    ----------
    residual : array-like
        Array of residual values (can include NaNs or infinities).
    output_file : str, optional
        Path to write the chi2 summary (default: "fit_mom_residuals.txt").

    Returns
    -------
    chi2_val : float
    dof : int
    p_value : float
        Computed chi-squared, degrees of freedom, and p-value.
    """
    # filter finite entries
    finite_res = np.asarray(residual)[np.isfinite(residual)]
    chi2_val = np.sum(finite_res**2)
    dof = len(finite_res) - 1
    p_value = 1 - chi2_dist.cdf(chi2_val, dof)

    # output
    summary = f"Chi2: {chi2_val}\nDoF: {dof}\np value: {p_value}\n"
    print(summary)
    with open(output_file, "w") as f:
        f.write(summary)

    return chi2_val, dof, p_value

def analyze_residuals(residual, bins=15, range=(-4, 4), fig_file=None, chi2_file=None):
    """
    Convenience wrapper: plot histogram with Gaussian overlay and compute chi2.

    Parameters
    ----------
    residual : array-like
    bins : int or sequence, optional
    range : tuple, optional
    fig_file : str, optional
    chi2_file : str, optional

    Returns
    -------
    fig, ax, chi2_val, dof, p_value
    """
    if fig_file is None:
        fig_file = "plot_mom_residuals_with_gaussian.png"
    if chi2_file is None:
        chi2_file = "fit_mom_residuals.txt"

    fig, ax = plot_residuals_with_gaussian(residual, bins=bins, range=range, filename=fig_file)
    chi2_val, dof, p_value = compute_chi2(residual, output_file=chi2_file)
    return fig, ax, chi2_val, dof, p_value