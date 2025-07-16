import pickle, numpy as np
from omnifold import MultiFold, DataLoader
from energyflow.archs import DNN

# 1) Load your pickles
with open("test_flat_electron/data_reco.pkl", "rb") as f: reco1 = pickle.load(f)
with open("test_flat_electron/data_mc.pkl", "rb") as f: gen1  = pickle.load(f)
with open("test_7_1_2025/data_reco.pkl", "rb") as f: reco2 = pickle.load(f)
with open("test_7_1_2025/data_mc.pkl", "rb") as f: gen2 = pickle.load(f)


# 2) Create DataLoaders
loader_data_flat = DataLoader(reco=reco1, gen=gen1, normalize=True)
loader_data_DIO = DataLoader(reco=reco2, gen=gen2, normalize=True)

# 3) Define neural networks
model_reco = DNN(input_dim=1, dense_sizes=[64,64])
model_gen  = DNN(input_dim=1, dense_sizes=[64,64])

mf = MultiFold("DIO_unfolding", model_reco, model_gen, loader_data_DIO, loader_data_flat, verbose=True)