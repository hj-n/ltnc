"""
Run the scalability experiment for all metrics
metrics list:
- Label-T&C (DSC) 
- Label-T&C (CH$_{btwn}$
- T&C
- MRRE 
- CA-T&C
- S&C 
- KL Divergence 
- DTM 
- SC 
- DSC
"""

import numpy as np
import sys, os
sys.path.append('../libs')
## turn off warning
import warnings
warnings.filterwarnings("ignore")

from metrics import *
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_RAW = "../libs/labeled-datasets/npy/"
PATH_EMB = "../libs/labeled-datasets_embedding/"

DR_MEASURES = {
  "Label-T&C (DSC)": ltnc_dsc_time,
	"Label-T&C (CH$_{btwn}$)": ltnc_btw_ch_time,
  "T&C": trust_conti_time,
  "MRRE": mrre_time,
  "CA-T&C": class_aware_trust_conti_time,
  "S&C": stead_cohev_time,
  "KL Divergence": kl_div_time,
  "DTM": dtm_time,
  "SC": silhouette_time,
  "DSC": dsc_time
}

DATASETS = os.listdir(PATH_RAW)
DR_LIST  = ["tsne", "umap", "pca", "isomap", "random"]


if not os.path.exists("./results/scalability.csv"):
	dataset_arr = []
	dr_arr = []
	measure_arr = []
	time_arr = []
	for dataset in tqdm(DATASETS):
		raw = np.load(f"{PATH_RAW}/{dataset}/data.npy")
		label = np.load(f"{PATH_RAW}/{dataset}/label.npy")
		for dr in DR_LIST:
			emb = np.load(f"{PATH_EMB}/{dataset}/{dr}.npy")
			for dr_measure in DR_MEASURES:
				if dr_measure == "SC" or dr_measure == "DSC":
					time = DR_MEASURES[dr_measure](emb, label)
				elif dr_measure == "Label-T&C (DSC)" or dr_measure == "Label-T&C (CH$_{btwn}$)" or dr_measure == "CA-T&C":
					time = DR_MEASURES[dr_measure](raw, emb, label)
				else:
					time = DR_MEASURES[dr_measure](raw, emb)
				dataset_arr.append(dataset)
				dr_arr.append(dr)
				measure_arr.append(dr_measure)
				time_arr.append(time)
				

	df = pd.DataFrame({
		"dataset": dataset_arr,
		"dr": dr_arr,
		"Distortion Measures": measure_arr,
		"Time (s)": time_arr
	})

	df.to_csv("./results/03_scalability.csv", index=False)
else:
	df = pd.read_csv("./results/03_scalability.csv")

sns.set_style("whitegrid")
plt.figure(figsize=(6, 2.6))

sns.pointplot(x= "Time (s)", y="Distortion Measures", hue="Distortion Measures", data=df[df["dr"] == "tsne"])

plt.legend().set_visible(False)
plt.tight_layout()

plt.savefig("./plot/03_scalability.pdf", dpi=300)
plt.savefig("./plot/03_scalability.png", dpi=300)
  

