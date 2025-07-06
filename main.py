import argparse
from unittest import result
import numpy as np
import awkward as ak
import dill as pickle
from scipy.stats import norm 
import matplotlib.pyplot as plt

import fit_module
from cut_module import CutClass
from import_module import ImportClass
import plot_module
from statistical_analysis_module import perform_goodness_of_fit_tests

def save_to_pickle(obj, filename: str) -> None:
    """
    Save any picklable Python object to a pickle file.

    Parameters:
    -----------
    obj : any
        The Python object to save (must be picklable).
    filename : str
        Path to the output pickle file.
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_from_pickle(filename: str):
    """
    Load any Python object from a pickle file.

    Parameters:
    -----------
    filename : str
        Path to the pickle file.

    Returns:
    --------
    any
        The loaded Python object.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)

def stream_files(args):
    """Iterate through the file list one ROOT file at a time, apply all cuts, and
    return flat NumPy arrays of reconstructed momentum and the corresponding MC
    first‐surface momentum.  Only one Awkward array lives in memory at any
    moment, so peak RAM stays roughly the size of a single file."""

    # resolve the file list
    if args.file.endswith(".txt"):  # if a text file is given
        with open(args.file) as f:
            file_list = [ln.strip() for ln in f if ln.strip()]
    else:  # otherwise, assume a single ROOT file
        file_list = [args.file]

    if args.verbose:
        print(f"[stream] Will iterate over {len(file_list)} file(s)")

    branches_trk = ["trk", "trksegs", "trksegpars_lh", "trkqual"]
    branches_crv = ["crvsummary", "crvcoincs"]
    branches_mc  = ["trkmcsim", "trksegsmc"]

    # prepare the cuts object once
    cuts = CutClass(str(args.cuts), True)

    reco_all, mc_all = [], []  # accumulators for final NumPy arrays

    # loop over files
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

        # apply selection cuts
        arr_cut = cuts.ApplyCut(arr_trk, arr_crv)

        # flatten to NumPy and accumulate
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

def main(args):
    """Run the full analysis with streaming input so memory never explodes."""

    # read & process files one at a time
    #data_np, data_mc_np = stream_files(args)
    #save_to_pickle(data_np, "data_reco.pkl")
    #save_to_pickle(data_mc_np, "data_mc.pkl")

    data_np = load_from_pickle("data_reco.pkl")
    data_mc_np = load_from_pickle("data_mc.pkl")

    '''
    result, PDF, N = fit_module.Unbinned_fit_mom(data_mc_np, (95, 104.97)) ##, "/exp/mu2e/app/users/wzhou2/FlatElectronAnalysis/output/flat_electron_no_cut/resolution_PDF.pkl")
    result.errors()  # calculate errors for the fit result
    print("Momentum fit result:", result)
    print("RESULT MESSAGE:", result.message)
    print("RESULT STATUS:", result.valid)
    # save the fit result
    with open("fit_result.txt", "w") as f:
        f.write(str(result))

    plot_module.plot_fit_result(data_mc_np, (95, 104.97), PDF, N, log_scale=True)
    plt.savefig("plot_mom_fit_result.png")

    results = perform_goodness_of_fit_tests(
        data_mc_np,
        PDF,
        (95, 104.97),
        grid_size=2000,
        output_prefix="./"
    )
    print("CvM →", results["cramer_von_mises"])
    print("KS  →", results["ks"])

    # save the test result as text
    with open("kolmogorov_smirnov_result.txt", "w") as f:
        f.write(f"KS statistic D = {results['ks']['statistic']}\np-value = {results['ks']['pvalue']}\n")
    '''

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
    fig, ax = plot_module.plt_mom_dist_mc_vs_reco(data_mc_np, data_np)
    plt.show()

    fig, ax = plot_module.plot_mom_mc_vs_reco_2d(data_mc_np, data_np)
    plt.show()

    fig, ax = plot_module.plot_mom_diff(data_mc_np, data_np)
    plt.show()
    # ------------------------------------------------------------------
    '''
    
    # ----------------------------------------------------
    # Momentum resolution fit
    ## filter the data_np to be within the fit range
    data_mc_bound_low = 101.7
    data_mc_bound_high = 110
    mask = (data_mc_np >= data_mc_bound_low) & (data_mc_np <= data_mc_bound_high)
    data_np = data_np[mask]
    data_mc_np = data_mc_np[mask]
    ## plot the momentum distribution
    #fig, ax = plot_module.plotmom_data_only(data_mc_np, (95, 105))
    #plt.show()


    diff =  data_np - data_mc_np
    result, PDF, N = fit_module.Unbinned_fit_resolution_function(diff, (-15, 1), 0)

    # save the fit result
    for p in PDF.get_params(floating=True):
        p.float = False
    #save_to_pickle(result, "resolution_fit_result.pkl")

    #result.errors()  # calculate errors for the fit result
    print("Resolution fit:", result)
    print("RESULT MESSAGE:", result.message)
    print("RESULT STATUS:", result.valid)

    ## record the result to a text file
    with open("resolution_fit_result.txt", "w") as f:
        f.write(str(result))

    plot_module.plot_fit_result(diff, (-15, 1), PDF, N)
    #plt.savefig("plot_resolution_fit_result.png")
    plt.show()

    print("Performing statistical tests on the resolution fit...")
    # statistical analysis
    results = perform_goodness_of_fit_tests(
        diff,   
        PDF,
        (-15, 1),
        grid_size=2000,
    )
    print("CvM →", results["cramer_von_mises"])
    print("KS  →", results["ks"])
    # save the test result as text
    #with open("resolution_kolmogorov_smirnov_result.txt", "w") as f:
        #f.write(f"KS statistic D = {results['ks']['statistic']}\np-value = {results['ks']['pvalue']}\n")
    #with open("resolution_cramer_von_mises_result.txt", "w") as f:
        #f.write(f"CvM statistic = {results['cramer_von_mises']['statistic']}\np-value = {results['cramer_von_mises']['pvalue']}\n")
    # -----------------------------------------------------

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
    parser.add_argument("--fitrange_low", type=float, default=[95, 640], nargs="+")
    parser.add_argument("--fitrange_high", type=float, default=[115, 1650], nargs="+")
    parser.add_argument("--cuts", type=str, default="SU2020")
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    if args.verbose:
        PrintArgs(args)

    main(args)
