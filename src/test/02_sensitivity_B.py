"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT B
"""
import numpy as np
import pandas as pd
import sys

sys.path.append("../libs")
import sensitivity_helpers as sh


"""
Experiment B-1 
"""

raw    = np.load("../../data/spheres_data/data/raw.npy")
labels = np.load("../../data/spheres_data/data/label.npy")

## load embeddings
emb_arr = []
for i in range(25):
  emb_arr.append(np.load(f"../../data/spheres_data/data/overlapping/circle_{i}.npy"))

results = sh.compute_metrics(raw, emb_arr, labels)
results["Angle btw two discs"] = np.linspace(60, 0, 25)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_B_1.csv", index=False)

"""
Experiment B-2
"""

## load embeddings
emb_arr = []
for i in range(25):
	emb_arr.append(np.load(f"../../data/spheres_data/data/overlapping_more/circle_{i}.npy"))

results = sh.compute_metrics(raw, emb_arr, labels)
results["Dist. to the origin"] = np.linspace(4, 0, 25)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_B_2.csv", index=False)