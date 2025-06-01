import os
import argparse
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

from import_module import ImportClass
from cut_module import CutClass
import fit_module

def _load_single_file(file_path: str, args, list_branch_trk, list_branch_crv, list_branch_mc):
    """Read one ROOT file and return (trk, crv, mc|None) Awkward arrays."""
    mds = ImportClass(file_path, args.dirname, args.treename)

    arr_trk = mds.Import(list_branch=list_branch_trk)
    arr_trk = mds.AddVectorMag(arr_trk, 'trksegs', 'mom')


    arr_crv = mds.Import(list_branch=list_branch_crv)
    arr_mc = mds.Import(list_branch=list_branch_mc) 
    arr_mc = mds.AddVectorMag(arr_mc, 'trksegsmc', 'mom')
    #arr_trk = mds.AddVectorMag(arr_trk, 'trkmcsim', 'mom')

    return arr_trk, arr_crv, arr_mc

def load_data(args):
    """Read all requested files in a simple loop and concatenate."""
    # --- build file list ----------------------------------------------------
    if args.file.endswith('.txt'):
        with open(args.file, 'r') as f:
            file_list = [ln.strip() for ln in f if ln.strip()]
    else:
        file_list = [args.file]

    list_branch_trk = ["trk", "trksegs", "trksegpars_lh", "trkqual"]
    list_branch_crv = ["crvsummary", "crvcoincs"]
    list_branch_mc  = ["trkmcsim", "trksegsmc"]

    list_array_trk, list_array_crv, list_array_mc = [], [], []

    if args.verbose:
        print(f"Loading {len(file_list)} file(s) sequentially…")

    # --- sequential read ----------------------------------------------------
    for fp in file_list:
        if args.verbose:
            print(f"  Reading {fp} …")
        trk, crv, mc = _load_single_file(fp, args,
                                         list_branch_trk,
                                         list_branch_crv,
                                         list_branch_mc)
        list_array_trk.append(trk)
        list_array_crv.append(crv)
        if mc is not None:
            list_array_mc.append(mc)

    array_trk = ak.concatenate(list_array_trk)
    array_crv = ak.concatenate(list_array_crv)
    array_mc  = ak.concatenate(list_array_mc) if list_array_mc else None

    return array_trk, array_crv, array_mc, file_list

def plotmom_data_only(data, fit_range, cat=None):
    """Plot just the data histogram (no fits or residuals)."""
    n_bins = 50

    # build histogram
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    # single axes now
    fig, ax = plt.subplots()

    ax.hist(
        data,
        color='black',
        bins=n_bins,
        range=fit_range,
        histtype='step'
    )

    # error bars on the data points
    ax.errorbar(
        data_bincenter,
        data_hist,
        yerr=np.sqrt(data_hist),
        fmt='none',
        ecolor='black',
        capsize=3
    )

    # styling
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xlim(fit_range)
    ax.set_xlabel('Reconstructed Momentum [MeV/c]')
    ax.set_ylabel('# of events per bin')

    return fig, ax

def extract_with_loops(array_cut, array_mc):
    # 1) Convert to nested Python lists
    reco_nested = array_cut['trksegs','mom.mag'].to_list()
    mc_nested   = array_mc['trksegsmc','mom.mag'].to_list()

    reco_vals = []
    mc_vals   = []

    # 2) Loop over events
    for ievt, (reco_evt, mc_evt) in enumerate(zip(reco_nested, mc_nested)):
        # mc_evt is a list of tracks, each a list of surface‐momenta
        for itrk, reco_trk in enumerate(reco_evt):
            # get MC first‐surface momentum (None if no MC segments)
            mc_trk = mc_evt[itrk] if itrk < len(mc_evt) else []
            mc_first = mc_trk[0] if (isinstance(mc_trk, list) and mc_trk) else None

            # 3) Loop over segments in reco_trk
            for seg_mom in reco_trk:
                if seg_mom is None:
                    continue   # skip masked-out segments
                # record the reco & MC values
                reco_vals.append(seg_mom)
                mc_vals.append(mc_first)

    # 4) Convert to NumPy (using np.nan for missing MC truth, if you like)
    reco_np = np.array(reco_vals, dtype=float)
    mc_np   = np.array([m if m is not None else np.nan for m in mc_vals], dtype=float)
    return reco_np, mc_np

def main(args):
    array_trk, array_crv, array_mc, file_list = load_data(args)
    
    '''
    # print out the first few entries of the track segments and track segment MC to compare
    for i in range(1,2):
        for j in range(1):
            # how many real track‐segment entries there are
            n_segs = len(array_trk['trksegs', 'mom.mag'][i][j])
            for k in range(n_segs):
                sid     = array_trk['trksegs',   'sid'    ][i][j][k]
                mommag  = array_trk['trksegs',   'mom.mag'][i][j][k]
                print(f"the surface index is {k}, sid is {sid}, mom.mag is {mommag}")

            # now the MC version
            n_segs_mc = len(array_trk['trksegsmc', 'mom.mag'][i][j])
            for k in range(n_segs_mc):
                sid_mc    = array_trk['trksegsmc',   'sid'    ][i][j][k]
                mommag_mc = array_trk['trksegsmc',   'mom.mag'][i][j][k]
                print(f"the surface index is {k}, sid is {sid_mc}, mom.mag is {mommag_mc}")
    
    print(len(array_trk['trksegs', 'mom.mag'][1][0]))
    print(len(array_trk['trksegsmc', 'mom.mag'][1][0]))
    '''

    if args.verbose:
        print("Applying cut list", args.cuts)
    cuts = CutClass(str(args.cuts), True)

    if int(args.cat) == 1 and array_mc is not None:
        track_cat = cuts.CategorizeTracks(array_mc, args.mismatch)
        array_trk['trksegs', 'cat'] = ak.broadcast_arrays(
            array_trk['trksegs', 'time'], track_cat
        )[1]

    array_cut = cuts.ApplyCut(array_trk, array_crv)
    #data_np = ak.to_numpy(ak.flatten(array_cut['trksegs','mom.mag'], axis=None))
    #data_mc_np = ak.to_numpy(ak.flatten(array_cut['trksegsmc','mom.mag'], axis=None))
    # … whatever you do with data_np …

    data_np, data_mc_np = extract_with_loops(array_cut, array_mc)

    # apply a mask that selects only the reco momentum that is within the fit range
    fit_range = (args.fitrange_low[0], args.fitrange_hi[0])
    mask = (data_np >= fit_range[0]) & (data_np <= fit_range[1])
    #data_np = data_np[mask]
    #data_mc_np = data_mc_np[mask]

    # use Chebychev polynomials to fit the momentum distribution of the MC data
    result_eff, PDF_eff, N_eff = fit_module.Unbinned_fit_efficiency(data_mc_np, (95, 105), degree=4)
    print("Efficiency fit result:", result_eff)
    print("RESULT MESSAGE:", result_eff.message)
    print(result_eff.info["original"])
    print(result_eff.info.get("message"))
    print("RESULT STATUS:", result_eff.valid)
    fit_module.plot_fit_result(data_mc_np, (95, 105), PDF_eff, N_eff)
    plt.savefig("plot_efficiency_fit_result.png")
    plt.show()

    
    # make a 1D histogram of the MC momentum
    fig, ax = plt.subplots()
    ax.hist(data_mc_np, bins=50, range=(93, 108), color='blue', alpha=0.5, label='MC Momentum')
    ax.hist(data_np, bins=50, range=(93, 108), color='red', alpha=0.5, label='Reco Momentum')
    ax.set_xlabel('Momentum [MeV/c]')
    ax.set_ylabel('Counts')
    ax.set_title('Momentum Distribution')
    ax.legend()
    plt.savefig("plot_mom_distribution.png")
    plt.show()

    # make a 2D heat map of reco vs MC
    fig, ax = plt.subplots()
    ax.hexbin(data_np, data_mc_np, gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('Reconstructed Momentum [MeV/c]')
    ax.set_ylabel('MC First-Surface Momentum [MeV/c]')
    ax.set_title('Reco vs MC Momentum')
    plt.colorbar(ax.collections[0], ax=ax, label='Counts')
    plt.savefig("plot_reco_vs_mc.png")
    plt.show()

    # plot the momentum_reco - momentum_mc histogram
    diff = data_np - data_mc_np
    fig, ax = plt.subplots()
    ax.hist(diff, bins=50, range=(-20, 2), color='gray', histtype='step')
    ax.set_xlabel('Momentum Reco - Momentum MC [MeV/c]')
    ax.set_ylabel('Counts')
    ax.set_title('Momentum Residuals')
    plt.savefig("plot_mom_residuals.png")   
    plt.show()

    result, PDF, N = fit_module.Unbinned_fit_resolution_function(diff, (-15, 1), 0)
    print("Fit result:", result)
    print("RESULT MESSAGE:", result.message)
    print(result.info["original"])
    print(result.info.get("message"))
    print("RESULT STATUS:", result.valid)

    fit_module.plot_fit_result(diff, (-15, 1), PDF, N)
    plt.savefig("plot_fit_result.png")
    plt.show()
    
    

    '''
    from pprint import pprint
    
    # turn the first 5 records into a nested list/dict
    data = array_cut[0][0].to_list()

    # pprint with a controlled “depth” (how many levels it will recurse)
    pprint(data, depth=10, width=120)
    '''

    '''
    # plot the data 
    plotmom_data_only(data_np, fit_range=(args.fitrange_low[0], args.fitrange_hi[0]), cat=int(args.cat))
    plt.savefig("plot_mom_data_only.png")
    plt.show()
    '''

def PrintArgs(args):
    print("========= Analyzing with user opts: ===========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='command arguments', formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--file", type=str, required=True, help="root file or txt list")
    parser.add_argument("--dirname", type=str, default="EventNtuple", help="TDirectory")
    parser.add_argument("--treename", type=str, default="ntuple", help="TTree name")
    parser.add_argument("--fittype", type=str, default="mom1D",
                        help="mom1D | time1D | momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[95, 640], nargs='+')
    parser.add_argument("--fitrange_hi", type=float, default=[115, 1650], nargs='+')
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
