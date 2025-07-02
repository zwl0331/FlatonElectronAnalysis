import zfit
import numpy as np
import awkward as ak
import dill as pickle
import tensorflow as tf
import matplotlib.pyplot as plt

default_resolution_params = {
    'su2020' : {
        'delta_p_0': (-0.5527, -5, 5),
        'a_0': (0.3765, 0, 20),
        'b_0': (4.494, 1, 10),
        'alpha_1': (1.173, 0, 15),
        'alpha_2': (0.4416, 0, 10),
        'N_1': (1.728, 0, 100),
        'N_2': (6.507, 0, 100)
    },
}

default_mom_components = {
    'DIO' : {'a6'     : (1.603439e-17 / 6.395123e-17,   -100,     100), #(1.16874e-17 / 8.6434e-17,   -100,     100),
             'a7'     : (-7.063418e-19 / 6.395123e-17,   -1000,     1000), #(-1.87828e-19 / 8.6434e-17,   -1000,     1000),
             'a8'     : (9.863835e-20 / 6.395123e-17,   -1e-1,     1e-1), #(9.16327e-20 / 8.6434e-17,   -1e-1,     1e-1),
            }
}

class poly58(zfit.pdf.ZPDF):
    """DIO spectrum  ∝  Σ a_n · (δ/Eμ)^n   with n = 5…8
       Parameter names kept exactly as you defined them."""
    _N_OBS  = 1
    _PARAMS = ['a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)               # tensor of electron energies
        p = self.params           # shorthand

        # physics constants (keep double precision if your data requires it)
        E_mu = tf.constant(105.194, dtype=x.dtype)   # MeV
        m_Al = tf.constant(25133.,  dtype=x.dtype)   # MeV

        # standard DIO endpoint variable
        delta = tf.nn.relu(E_mu - x - 0.5 * x**2 / m_Al)

        # ***** 1‑line stabilisation trick *****
        u = delta         # 0 ≤ u ≤ 1  ⇒  all powers uⁿ stay ≤ 1
        # **************************************

        return (u**5 +
                p['a6'] * u**6 +
                p['a7'] * u**7 +
                p['a8'] * u**8)

class MomResolution(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = ['delta_p_0', 'a_0', 'b_0', 'alpha_1', 'alpha_2', 'N_1', 'N_2']

    def _unnormalized_pdf(self, x):
        z = zfit.z
        x = zfit.z.unstack_x(x)

        delta_p_0 = self.params['delta_p_0']
        a_0       = self.params['a_0']
        b_0       = self.params['b_0']
        alpha_1   = self.params['alpha_1']
        alpha_2   = self.params['alpha_2']
        N_1       = self.params['N_1']
        N_2       = self.params['N_2']

        shifted = x - delta_p_0

        B_1 = alpha_1 - N_1 / (a_0 * b_0 * (1 - z.exp(-b_0 * alpha_1)))
        A_1 = z.exp(a_0 * (-b_0 * alpha_1 - z.exp(-b_0 * alpha_1))) * (alpha_1 - B_1)**N_1

        B_2 = -alpha_2 - N_2 / (a_0 * b_0 * (1 - z.exp(b_0 * alpha_2)))
        A_2 = z.exp(a_0 * (b_0 * alpha_2 - z.exp(b_0 * alpha_2))) * (B_2 + alpha_2)**N_2

        core = z.exp(a_0 * (b_0 * shifted - z.exp(b_0 * shifted)))

        pdf = tf.where(shifted < -alpha_1,
                       A_1 / (- B_1 - shifted)**N_1,
                       tf.where(shifted < alpha_2,
                                core,
                                A_2 / (B_2 + shifted)**N_2))

        return pdf

def Unbinned_fit_resolution_function(data_np, fit_range, mom_resolution = 0):
    obs_mom_diff = zfit.Space('mom_diff', limits=fit_range)
    
    # build the zfit PDF for the momentum resolution
    N = zfit.Parameter('N', 10000, 0, 1e10)
    zpars_res = {'N': N}

    if mom_resolution == 0:
        for parameter in default_resolution_params['su2020']:
            zpars_res[parameter] = zfit.Parameter(parameter + '_res', 
                                                  default_resolution_params['su2020'][parameter][0], 
                                                  default_resolution_params['su2020'][parameter][1],
                                                  default_resolution_params['su2020'][parameter][2])
        PDF = MomResolution(obs=obs_mom_diff, extended = N, 
                            delta_p_0=zpars_res['delta_p_0'],
                                                      a_0=zpars_res['a_0'], 
                                                      b_0=zpars_res['b_0'], 
                                                      alpha_1=zpars_res['alpha_1'],
                                                      alpha_2=zpars_res['alpha_2'], 
                                                      N_1=zpars_res['N_1'], 
                                                      N_2=zpars_res['N_2'])
    
    # convert numpy array to zfit Data
    data_zfit =  zfit.Data.from_numpy(obs=obs_mom_diff, array=data_np)

    # do the fit
    loss = zfit.loss.ExtendedUnbinnedNLL(model = PDF, data = data_zfit)
    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=zpars_res.values())

    return result, PDF, N

def Unbinned_fit_efficiency(data_mc_np, fit_range, degree = 4):
    obs_mom = zfit.Space('mom', limits=fit_range)
    
    # build the zfit PDF for the efficiency
    N = zfit.Parameter('N', 10000, 0, 1e10)
    zpars_eff = {'N': N}

    # Chebychev polynomial coefficients
    coeffs = [zfit.Parameter(f'c_{i}', 0.1, -1, 1) for i in range(degree + 1)]
    for i, coeff in enumerate(coeffs):
        zpars_eff[f'c_{i}'] = coeff
    
    PDF = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N)

    # convert numpy array to zfit Data
    data_zfit =  zfit.Data.from_numpy(obs=obs_mom, array=data_mc_np)

    # do the fit
    loss = zfit.loss.ExtendedUnbinnedNLL(model=PDF, data=data_zfit)
    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        tol=1e-10,        # tighter EDM goal
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=zpars_eff.values())

    return result, PDF, N


def Unbinned_fit_mom(data_np, fit_range, resolution_path = None):
    efficiceny_result_path = "/exp/mu2e/app/users/wzhou2/FlatElectronAnalysis/output/flat_electron_no_cut/efficiency_PDF.pkl"
    obs_mom = zfit.Space('mom', limits=fit_range)

    # build pdf 
    ## build the parameter list for the theoretical PDF
    pars = []
    for parameter in default_mom_components['DIO']:
        pars.append(zfit.Parameter(parameter + "_DIO", default_mom_components['DIO'][parameter][0],
                                   default_mom_components['DIO'][parameter][1],
                                   default_mom_components['DIO'][parameter][2]))
    theoretical_pdf = poly58(obs=obs_mom, a6=pars[0], a7=pars[1], a8=pars[2])
    with open(efficiceny_result_path, "rb") as f:
        efficiency_pdf = pickle.load(f)
    ## product pdf
    N = zfit.Parameter('N', 10000, 0, 1e10)
    pars.append(N)

    if resolution_path is None:
        PDF = zfit.pdf.ProductPDF([theoretical_pdf, efficiency_pdf], extended=N)
    else:
        PDF = zfit.pdf.ProductPDF([theoretical_pdf, efficiency_pdf])
        # load the resolution PDF
        with open(resolution_path, "rb") as f:
            resolution_pdf = pickle.load(f)
        # convolution
        resolution_pdf = resolution_pdf.copy(obs=zfit.Space('mom', limits=(-8, 1)))
        PDF = zfit.pdf.FFTConvPDFV1(PDF, resolution_pdf, obs=obs_mom, extended=N, n=1000)

    # Convert data to zfit Data
    data_zfit = zfit.Data.from_numpy(array=data_np, obs=obs_mom)

    loss = zfit.loss.ExtendedUnbinnedNLL(model=PDF, data=data_zfit)

    minimizer = zfit.minimize.Minuit(
        gradient=True,    # use Minuit’s own gradient
        mode=2,           # full Hesse
        maxiter=2000      # more calls
    )
    result = minimizer.minimize(loss, params=[N])

    return result, PDF, N
