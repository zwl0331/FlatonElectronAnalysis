import argparse
import numpy as np
import awkward as ak
import dill as pickle
from scipy.stats import norm 
import matplotlib.pyplot as plt

import fit_module
from cut_module import CutClass
from import_module import ImportClass

def stream_files(args):
    """Iterate through the file list one ROOT file at a time, apply all cuts, and
    return flat NumPy arrays of reconstructed momentum and the corresponding MC
    first‐surface momentum.  Only one Awkward array lives in memory at any
    moment, so peak RAM stays roughly the size of a single file."""

    # ---- resolve file list ---------------------------------------------
    if args.file.endswith(".txt"):
        with open(args.file) as f:
            file_list = [ln.strip() for ln in f if ln.strip()]
    else:
        file_list = [args.file]

    if args.verbose:
        print(f"[stream] Will iterate over {len(file_list)} file(s)")

    branches_trk = ["trk", "trksegs", "trksegpars_lh", "trkqual"]
    branches_crv = ["crvsummary", "crvcoincs"]
    branches_mc  = ["trkmcsim", "trksegsmc"]

    # prepare the cuts object once
    cuts = CutClass(str(args.cuts), True)

    reco_all, mc_all = [], []  # accumulators for final NumPy arrays

    # ---- loop over files -----------------------------------------------
    for fp in file_list:
        if args.verbose:
            print(f"[stream] Reading {fp}")

        mds = ImportClass(fp, args.dirname, args.treename)

        # read tracker branches and add |p| for each segment
        arr_trk = mds.Import(branches_trk)
        arr_trk = mds.AddVectorMag(arr_trk, "trksegs", "mom")

        # CRV branches
        arr_crv = mds.Import(branches_crv)

        # MC branches with |p| for the first‐surface segments
        arr_mc = mds.Import(branches_mc)
        arr_mc = mds.AddVectorMag(arr_mc, "trksegsmc", "mom")

        # ---- optional categorisation -----------------------------------
        if int(args.cat) == 1 and arr_mc is not None:
            cats = cuts.CategorizeTracks(arr_mc, args.mismatch)
            arr_trk["trksegs", "cat"] = ak.broadcast_arrays(
                arr_trk["trksegs", "time"], cats
            )[1]

        # ---- apply selection cuts --------------------------------------
        arr_cut = cuts.ApplyCut(arr_trk, arr_crv)

        # ---- flatten to NumPy and accumulate ---------------------------
        reco_np, mc_np = extract_with_loops(arr_cut, arr_mc)
        reco_all.append(reco_np)
        mc_all.append(mc_np)

        # help the GC between files
        del arr_trk, arr_crv, arr_mc, arr_cut

    # concatenate lists of 1-D NumPy arrays into single flat arrays
    data_np    = np.concatenate(reco_all) if reco_all else np.empty(0)
    data_mc_np = np.concatenate(mc_all)  if mc_all  else np.empty(0)
    return data_np, data_mc_np


def extract_with_loops(array_cut, array_mc):
    """Extract every reconstructed segment momentum and the first MC segment
    momentum of the corresponding track.  Returns two flat NumPy arrays of
    equal length."""

    reco_nested = array_cut["trksegs", "mom.mag"].to_list()
    mc_nested   = array_mc["trksegsmc", "mom.mag"].to_list()

    reco_vals, mc_vals = [], []

    for reco_evt, mc_evt in zip(reco_nested, mc_nested):
        for itrk, reco_trk in enumerate(reco_evt):
            mc_trk   = mc_evt[itrk] if itrk < len(mc_evt) else []
            mc_first = mc_trk[0] if (isinstance(mc_trk, list) and mc_trk) else None
            for seg_mom in reco_trk:
                if seg_mom is None:
                    continue
                reco_vals.append(seg_mom)
                mc_vals.append(mc_first)

    reco_np = np.asarray(reco_vals, dtype=float)
    mc_np   = np.asarray([m if m is not None else np.nan for m in mc_vals], dtype=float)
    return reco_np, mc_np


# -------------------------------------------------------------------------
# plotting helper (unchanged)
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# main analysis flow
# -------------------------------------------------------------------------

def main(args):
    """Run the full analysis with streaming input so memory never explodes."""

    # read & process files one at a time
    data_np, data_mc_np = stream_files(args)

    
    # momentum fit range mask (optional)
    fit_range = (args.fitrange_low[0], args.fitrange_hi[0])
    in_range  = (data_np >= fit_range[0]) & (data_np <= fit_range[1])
    #data_np   = data_np[in_range]
    #data_mc_np = data_mc_np[in_range]


    result, PDF, N = fit_module.Unbinned_fit_mom(data_np, (95, 104.97), "/exp/mu2e/app/users/wzhou2/FlatElectronAnalysis/output/flat_electron_no_cut/resolution_PDF.pkl")
    result.errors()  # calculate errors for the fit result
    print("Momentum fit result:", result)
    print("RESULT MESSAGE:", result.message)
    print("RESULT STATUS:", result.valid)
    # save the fit result
    with open("fit_result.txt", "w") as f:
        f.write(str(result))

    fit_module.plot_fit_result(data_np, (95, 104.97), PDF, N, log_scale=True)
    plt.savefig("plot_mom_fit_result.png")

    # statistical tests
    import numpy as np
    from scipy.interpolate import interp1d

    # --- 1) your fit range ---
    low, high = fit_range  # e.g. (95, 104.97)

    # --- 2) build a fine x-grid over [low, high] ---
    x_grid = np.linspace(low, high, 2000)

    # --- 3) allocate cdf_vals and set CDF=0 at the very first point ---
    cdf_vals = np.empty_like(x_grid)
    cdf_vals[0] = 0.0

    # --- 4) fill in the rest by integrating ---
    for i, xi in enumerate(x_grid[1:], start=1):
        # now xi > low, so integrate is legal
        integ = PDF.integrate(limits=(low, float(xi)), norm=(low, high))
        # `.integrate` returns a TensorFlow tensor → convert to numpy float
        cdf_vals[i] = integ.numpy()

    # (Optional sanity check: ensure the final CDF is ~1)
    # print("CDF at high:", cdf_vals[-1])

    # --- 5) build your “black-box” CDF function ---
    model_cdf = interp1d(
        x_grid,
        cdf_vals,
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )
    
    # limit data_mc_np to the fit range for the tests
    data_np = data_np[(data_np >= fit_range[0]) & (data_np <= fit_range[1])]

    from scipy.stats import cramervonmises
    print("Performing Cramér-von Mises test...")
    res = cramervonmises(data_np, model_cdf)
    print("CvM stat:", res.statistic, "p-value:", res.pvalue)

    # save the test result as text
    with open("cramer_von_mises_result.txt", "w") as f:
        f.write(f"CvM stat: {res.statistic}\np-value: {res.pvalue}\n")

    from scipy.stats import kstest

    # assume you’ve already built:
    #   x_grid, cdf_vals = ...  # via PDF.integrate + interp1d
    #   model_cdf = interp1d(x_grid, cdf_vals,
    #                        bounds_error=False,
    #                        fill_value=(0.0, 1.0))

    # ensure it's a plain callable CDF:
    # interp1d *is* callable, so you can pass it directly:
    dist_cdf = model_cdf

    # your data array (unsorted is fine; kstest will sort internally):
    data = data_np  

    # run KS test
    res = kstest(data, dist_cdf)

    print(f"KS statistic D = {res.statistic:.4f}")
    print(f"p-value       = {res.pvalue:.4f}")

    # save the test result as text
    with open("kolmogorov_smirnov_result.txt", "w") as f:
        f.write(f"KS statistic D = {res.statistic}\np-value = {res.pvalue}\n")

    '''
    # 2) attach a .cdf attribute pointing back to itself
    model_cdf.cdf = model_cdf

    # 3) now pass it directly
    from statsmodels.stats.diagnostic import anderson_statistic
    A2 = anderson_statistic(data_mc_np, dist=model_cdf, fit=False)
    def ad_pvalue_large_n(A2):
        """Approximate p-value for Anderson–Darling A², valid for large n."""
        if A2 < 0.2:
            return 1.0 - np.exp(-13.436 + 101.14*A2 - 223.73*A2**2)
        elif A2 < 0.34:
            return 1.0 - np.exp(-8.318 + 42.796*A2 - 59.938*A2**2)
        elif A2 < 0.6:
            return np.exp(0.9177 - 4.279*A2 - 1.38*A2**2)
        else:
            return np.exp(1.2937 - 5.709*A2 + 0.0186*A2**2)

    p_value = ad_pvalue_large_n(A2)
    print(f"A² = {A2:.3f},  p ≃ {p_value:.3f}")

    # save the test result as text
    with open("anderson_darling_result.txt", "w") as f:
        f.write(f"A² = {A2}\np-value ≃ {p_value}\n")'''
    
    '''
    # ------------------------------------------------------------------
    # Basic plots
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.hist(data_mc_np, bins=50, range=(93, 108), color="blue", alpha=0.5, label="MC")
    ax.hist(data_np,    bins=50, range=(93, 108), color="red",  alpha=0.5, label="Reco")
    ax.set_xlabel("Momentum [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Momentum Distribution")
    ax.legend()
    plt.savefig("plot_mom_distribution.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.hexbin(data_np, data_mc_np, gridsize=50, cmap="Blues", mincnt=1)
    ax.set_xlabel("Reco p [MeV/c]")
    ax.set_ylabel("MC first-surface p [MeV/c]")
    ax.set_title("Reco vs MC Momentum")
    plt.colorbar(ax.collections[0], ax=ax, label="Counts")
    plt.savefig("plot_reco_vs_mc.png")
    plt.show()

    # residuals and resolution fit
    diff = data_np - data_mc_np
    fig, ax = plt.subplots()
    ax.hist(diff, bins=50, range=(-20, 2), histtype="step", color="gray")
    ax.set_xlabel("Reco − MC [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Momentum Residuals")
    plt.savefig("plot_mom_residuals.png")
    plt.show()

    result, PDF, N = fit_module.Unbinned_fit_resolution_function(diff, (-15, 1), 0)

    # save the fit result
    ## freeze the parameters to avoid accidental changes
    for p in PDF.get_params(floating=True):
        p.float = False
    ## save the fit result to a pickle file
    with open("resolution_PDF.pkl", "wb") as f:
        pickle.dump(PDF, f)

    result.errors()  # calculate errors for the fit result
    print("Resolution fit:", result)
    print("RESULT MESSAGE:", result.message)
    print("RESULT STATUS:", result.valid)

    ## record the result to a text file
    with open("resolution_fit_result.txt", "w") as f:
        f.write(str(result))

    fit_module.plot_fit_result(diff, (-15, 1), PDF, N)
    plt.savefig("plot_fit_result.png")
    plt.show()
    '''

def PrintArgs(args):
    print("========= Analyzing with user opts: ===========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mu2e momentum analysis (streaming version)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--file", type=str, required=True, help="root file or txt list")
    parser.add_argument("--dirname", type=str, default="EventNtuple", help="TDirectory")
    parser.add_argument("--treename", type=str, default="ntuple", help="TTree name")
    parser.add_argument("--fittype", type=str, default="mom1D", help="mom1D | time1D | momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[95, 640], nargs="+")
    parser.add_argument("--fitrange_hi",  type=float, default=[115, 1650], nargs="+")
    parser.add_argument("--cuts", type=str, default="SU2020")
    parser.add_argument("--showMC", type=int, default=0)
    parser.add_argument("--cat", type=int, default=0)
    parser.add_argument("--mismatch", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--DIO_res", type=int, default=0)

    args = parser.parse_args()

    if args.verbose:
        PrintArgs(args)

    main(args)
