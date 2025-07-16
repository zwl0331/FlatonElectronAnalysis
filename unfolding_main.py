import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# 1) Register the omnifold loss so Keras can find it on clone_model
from omnifold.omnifold import weighted_binary_crossentropy
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="omnifold_losses",
                            name="weighted_binary_crossentropy")
def weighted_binary_crossentropy_serializable(y_true, y_pred):
    # wrap the original so it’s recognized by name
    return weighted_binary_crossentropy(y_true, y_pred)

import pickle, numpy as np
from omnifold import MultiFold, DataLoader, MLP, SetStyle, HistRoutine
from energyflow.archs import DNN
from matplotlib import pyplot as plt

# 1) Load your pickles
with open("test_flat_electron/data_reco.pkl", "rb") as f: reco1 = pickle.load(f)
with open("test_flat_electron/data_mc.pkl", "rb") as f: gen1  = pickle.load(f)
with open("test_7_1_2025/data_reco.pkl", "rb") as f: reco2 = pickle.load(f)
with open("test_7_1_2025/data_mc.pkl", "rb") as f: gen2 = pickle.load(f)

## small size data samples for testing
reco1 = reco1
gen1 = gen1
reco2 = reco2[:1000]  # limit size for testing
gen2 = gen2[:1000]  # limit size for testing


# 2) Create DataLoaders
loader_data_flat = DataLoader(reco=reco1, gen=gen1, normalize=True)
loader_data_DIO = DataLoader(reco=reco2, gen=gen2, normalize=True)

ndim = 1
model_reco = MLP(ndim)
model_gen  = MLP(ndim)

mf = MultiFold(
    name="DIO_unfolding",
    model_reco=model_reco,
    model_gen=model_gen,
    data=loader_data_DIO,
    mc=loader_data_flat,
    verbose=True
)

mf.Unfold()

SetStyle()
# 4) Plot unfolded vs truth
import numpy as np

# --- extract the arrays you want to compare ---
# unfolded samples are the MC gen you just reweighted
unfolded_vals   = loader_data_flat.gen[:]   # pick the 1D observable
unfolded_weights= mf.weights_push             # final weights after unfolding

# truth samples are the “data” gen
truth_vals      = loader_data_DIO.gen[:]

# --- choose a common binning over the full range ---
bins = np.linspace(
    min(unfolded_vals.min(), truth_vals.min()),
    max(unfolded_vals.max(), truth_vals.max()),
    50
)

# --- assemble dicts for HistRoutine ---
data_dict = {
    "Truth":    truth_vals,
    "Unfolded": unfolded_vals
}
weight_dict = {
    "Truth":    np.ones_like(truth_vals),
    "Unfolded": unfolded_weights
}

# --- call the built-in histogram routine (with ratio panel) ---
fig, ax = HistRoutine(
    data_dict,
    "Observable [units]",
    reference_name = "Truth",
    weights        = weight_dict,
    binning        = bins
)
plt.savefig("unfolded_vs_truth.png", dpi=300)