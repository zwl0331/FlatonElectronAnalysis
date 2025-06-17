import argparse
import numpy as np
import awkward as ak
import dill as pickle
import matplotlib.pyplot as plt

from import_module import ImportClass
from cut_module import CutClass
import fit_module

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

    
    result, PDF, N = fit_module.Unbinned_fit_mom(data_mc_np, (95, 104.97))
    result.errors()  # calculate errors for the fit result
    print("Momentum fit result:", result)
    print("RESULT MESSAGE:", result.message)
    print("RESULT STATUS:", result.valid)
    # save the fit result
    with open("fit_result.txt", "w") as f:
        f.write(str(result))

    data_bin_center, residual = fit_module.plot_fit_result(data_mc_np, (95, 104.97), PDF, N, log_scale=True)
    plt.savefig("plot_mom_fit_result.png")

    # plot the 1D distribution of residuals
    fig, ax = plt.subplots()
    ax.hist(residual, bins=50, range=(-3, 3), histtype="step", color="gray")

    ax.set_xlabel("Residuals [MeV/c]")
    ax.set_ylabel("Counts")
    ax.set_title("Fit Residuals")

    plt.savefig("plot_mom_residuals.png")
    plt.show()
    
    
    '''
    # ------------------------------------------------------------------
    # Efficiency fit (Chebyshev)
    # ------------------------------------------------------------------
    result_eff, PDF_eff, N_eff = fit_module.Unbinned_fit_efficiency(
        data_mc_np, (95, 104.97), degree=4)
    param_errors_eff, _ = result_eff.errors()
    
    # save the fit result
    ## freeze the parameters to avoid accidental changes
    for p in PDF_eff.get_params(floating = True):
        p.float = False
    ## save the fit result to a pickle file
    with open("efficiency_PDF.pkl", "wb") as f:
        pickle.dump(PDF_eff, f)

    print("Efficiency fit result:", result_eff)
    print("RESULT MESSAGE:", result_eff.message)
    print("RESULT STATUS:", result_eff.valid)

    ## record the result_eff to a text file
    with open("efficiency_fit_result.txt", "w") as f:
        f.write(str(result_eff))

    fit_module.plot_fit_result(data_mc_np, (95, 104.97), PDF_eff, N_eff)
    plt.savefig("plot_efficiency_fit_result.png")
    plt.show()
    '''

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
