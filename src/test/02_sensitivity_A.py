"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT A
"""
import numpy as np
import umap
import sys
import pandas as pd

sys.path.append("../libs")
import sensitivity_helpers as sh

raw    = np.load("../libs/labeled-datasets/npy/coil20/data.npy")
labels = np.load("../libs/labeled-datasets/npy/coil20/label.npy")

emb_skeleton = umap.UMAP().fit_transform(raw)
emb_random   = np.random.rand(emb_skeleton.shape[0], emb_skeleton.shape[1])
emb_random[:, 0] = emb_random[:, 0] * (np.max(emb_skeleton[:, 0]) - np.min(emb_skeleton[:, 0])) + np.min(emb_skeleton[:, 0])
emb_random[:, 1] = emb_random[:, 1] * (np.max(emb_skeleton[:, 1]) - np.min(emb_skeleton[:, 1])) + np.min(emb_skeleton[:, 1])

## generate randomized embeddings
emb_arr = []
for i in range(25):
	random_rate = i / 25
	random_replacement = np.random.rand(emb_skeleton.shape[0], emb_skeleton.shape[1]) > random_rate
	random_replacement = random_replacement.astype(int)
	emb = emb_skeleton * random_replacement + emb_random * (1 - random_replacement)
	emb_arr.append(emb)

## calculate metrics
results = sh.compute_metrics(raw, emb_arr, labels)
results["Replacement prob."] = np.linspace(0, 100, 25)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_A.csv", index=False)

