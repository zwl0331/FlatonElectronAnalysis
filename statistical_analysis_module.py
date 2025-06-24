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
    
def anderson_darling_gof(data, pdf, n_toys):
    """
    Perform an unbinned Anderson–Darling goodness-of-fit test with Monte Carlo calibration,
    using zfit.pdf.integrate to compute the CDF (via cumulative integration).

    Parameters:
    -----------
    data : numpy.ndarray
        1D array of observed data points.
    pdf : zfit.pdf.BasePDF
        Fitted zfit PDF object with methods:
            - sample(n): returns samples of size n (Tensor or ndarray).
            - integrate(limits, norm=None): returns integral over limits, normalized.
            - space: zfit.Space giving the default normalization range.
    n_toys : int
        Number of toy Monte Carlo datasets to generate.

    Returns:
    --------
    A2_obs : float
        Observed Anderson–Darling statistic.
    p_value : float
        Monte Carlo p-value.
    fig : matplotlib.figure.Figure
        Figure showing the distribution of A² from toy experiments with a red line
        at the observed A2_obs.
    """
    # Helper to extract numpy arrays
    def to_numpy(x):
        try:
            return x.numpy()
        except AttributeError:
            return np.array(x)

    # Determine lower bound for CDF integration
    space = pdf.space
    lower = pdf.space.limits[0][0][0]
    upper = space.limits[1][0][0]
    print(f"Integration limits: {lower} to {upper}")
    print(f"Number of data points: {len(data)}")
    data = data[(data >= lower) & (data <= upper)]
    print(f"Number of data points after filtering: {len(data)}")
    
    # Compute observed PIT values via integrate
    u_obs = []
    for x in data:
        val = to_numpy(pdf.integrate((lower, float(x))))
        u_obs.append(val)
    u_obs = np.sort(np.array(u_obs))
    N = len(u_obs)
    i = np.arange(1, N + 1)
    A2_obs = -N - np.sum((2*i - 1) * (np.log(u_obs) + np.log(1 - u_obs[::-1]))) / N

    # Monte Carlo toys
    A2_toys = []
    for _ in range(n_toys):
        print(f"Generating toy dataset {_ + 1}/{n_toys}")
        toy = to_numpy(pdf.sample(n=N))
        u_toy = []
        for xt in toy:
            u_toy.append(to_numpy(pdf.integrate((lower, float(xt)))))
        u_toy = np.sort(np.array(u_toy))
        A2 = -N - np.sum((2*i - 1) * (np.log(u_toy) + np.log(1 - u_toy[::-1]))) / N
        A2_toys.append(A2)
    A2_toys = np.array(A2_toys)

    # p-value
    p_value = np.mean(A2_toys >= A2_obs)

    # Plot
    fig, ax = plt.subplots()
    ax.hist(A2_toys, bins=50, histtype='step')
    ax.axvline(A2_obs, color='red', linewidth=2)
    ax.set_xlabel('A$^2$ statistic')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo distribution of Anderson–Darling A$^2$')

    return A2_obs, p_value, fig
