"""
SENSITIVITY ANALYSIS (Section 5.1) - EXPERIMENT E
"""
import numpy as np
import sys
import pandas as pd

sys.path.append("../libs")
import sensitivity_helpers as sh


emb    = np.load("../../data/spheres_data/data/overlapping_raw/emb.npy")
labels = np.load("../../data/spheres_data/data/overlapping_raw/label.npy")

raw_arr = []
for i in range(25):
  raw_arr.append(np.load(f"../../data/spheres_data/data/overlapping_raw/raw_{i}.npy"))

## calculate metrics
results = sh.compute_metrics(raw_arr, emb, labels)
results["Dist. to the origin"] = np.linspace(12, 0, 25)

results_df = pd.DataFrame(results)
results_df.to_csv("./results/02_sensitivity_E.csv", index=False)