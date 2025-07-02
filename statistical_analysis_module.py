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

def perform_goodness_of_fit_tests(data, PDF, fit_range, grid_size=2000, output_prefix=""):
    """
    Run Cramér–von Mises and Kolmogorov–Smirnov tests comparing ‘data’ to the model PDF.

    Parameters
    ----------
    data : array-like of float
        Your reconstructed momenta array.
    PDF : object
        The fitted PDF, must have an integrate(limits=(low, high), norm=(low, high)) → tf.Tensor API.
    fit_range : tuple (low, high)
        The interval over which you performed the fit.
    grid_size : int, optional
        Number of points to build the x‐grid for the CDF (default: 2000).
    output_prefix : str, optional
        Path prefix for saving the two “*_result.txt” files.

    Returns
    -------
    dict
        {
          "cramer_von_mises": {"statistic": float, "pvalue": float},
          "ks":              {"statistic": float, "pvalue": float},
          "model_cdf":       <callable CDF function>
        }
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.stats import cramervonmises, kstest

    low, high = fit_range

    # 1) Build fine x‐grid and compute CDF via PDF.integrate()
    x_grid = np.linspace(low, high, grid_size)
    cdf_vals = np.empty_like(x_grid)
    cdf_vals[0] = 0.0
    for i, xi in enumerate(x_grid[1:], start=1):
        cdf_vals[i] = float(PDF.integrate(limits=(low, xi), norm=(low, high)).numpy())
    model_cdf = interp1d(x_grid, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0))

    # 2) Restrict data to fit_range
    data_in = data[(data >= low) & (data <= high)]

    # 3) Cramér–von Mises
    cvm_res = cramervonmises(data_in, model_cdf)
    with open(f"{output_prefix}cramer_von_mises_result.txt", "w") as f:
        f.write(f"CvM stat: {cvm_res.statistic}\n")
        f.write(f"p-value: {cvm_res.pvalue}\n")

    # 4) Kolmogorov–Smirnov
    ks_res = kstest(data_in, model_cdf)
    with open(f"{output_prefix}kolmogorov_smirnov_result.txt", "w") as f:
        f.write(f"KS statistic D = {ks_res.statistic}\n")
        f.write(f"p-value = {ks_res.pvalue}\n")

    # 5) Return results
    return {
        "cramer_von_mises": {"statistic": cvm_res.statistic, "pvalue": cvm_res.pvalue},
        "ks":              {"statistic": ks_res.statistic,  "pvalue": ks_res.pvalue},
        "model_cdf":       model_cdf
    }
