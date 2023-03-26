"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT F
"""
import numpy as np
from sklearn.decomposition import PCA
import sys
import pandas as pd

sys.path.append("../libs")
import sensitivity_helpers as sh


raw    = np.load("../../data/labeled-datasets/npy/fashion_mnist/data.npy")
emb    = np.load("../../data/labeled-datasets_embedding/fashion_mnist/umap.npy")
labels = np.load("../../data/labeled-datasets/npy/fashion_mnist/label.npy")

raw_pca = PCA(n_components=300).fit_transform(raw)

pc_interval = 10

raw_arr = []
for i in range(10):
  raw = raw_pca[:, i: i + pc_interval]
  raw_arr.append(raw)
  
## calculate metrics
results = sh.compute_metrics(raw_arr, emb, labels)
results["Starting index of PC"] = range(1, 11)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_F.csv", index=False)

