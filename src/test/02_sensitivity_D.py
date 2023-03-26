"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT D
"""
import numpy as np
import umap
import sys
import pandas as pd

sys.path.append("../libs")
import sensitivity_helpers as sh


raw    = np.load("../../data/labeled-datasets/npy/coil20/data.npy")
labels = np.load("../../data/labeled-datasets/npy/coil20/label.npy")
emb    = umap.UMAP().fit_transform(raw) 

## generate randomized high-dimensional data
raw_skeleton = raw.copy()
raw_random   = np.random.rand(raw.shape[0], raw.shape[1])
for i in range(raw_skeleton.shape[1]):
	raw_random[:, i] = raw_random[:, i] * (np.max(raw_skeleton[:, i]) - np.min(raw_skeleton[:, i])) + np.min(raw_skeleton[:, i])

raw_arr = []
for i in range(25):
	random_rate = i / 25
	random_replacement = np.random.rand(raw_skeleton.shape[0], raw_skeleton.shape[1]) > random_rate
	random_replacement = random_replacement.astype(int)
	raw = raw_skeleton * random_replacement + raw_random * (1 - random_replacement)
	raw_arr.append(raw)

## calculate metrics
results = sh.compute_metrics(raw_arr, emb, labels)
results["Replacement prob."] = np.linspace(0, 100, 25)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_D.csv", index=False)


