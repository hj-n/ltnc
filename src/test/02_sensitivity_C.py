"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT C
"""
import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd

sys.path.append("../libs")
import sensitivity_helpers as sh

raw    = np.load("../libs/labeled-datasets/npy/fashion_mnist/data.npy")
labels = np.load("../libs/labeled-datasets/npy/fashion_mnist/label.npy")

max_pc  = 10
emb_pca = PCA(n_components=max_pc).fit_transform(raw)

emb_arr = []
for i in range(max_pc):
	emb_arr.append(emb_pca[:, :i+1])

results = sh.compute_metrics(raw, emb_arr, labels)
results["Number of PCs"] = range(1, 11)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_C.csv", index=False)
