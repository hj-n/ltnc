"""
PRELIMINARY TEST (Section 3)
The preliminary experiment is conducted to compare the performance of 5 dimensionality reduction techqniques
using the current way of measuring the quality of the DR embedding based on label separability
Specifically, we measured the performance of the following 5 DR techniques:
 - LDA
 - t-SNE
 - UMAP
 - PCA
 - Isomap
using Silhouette and DSC
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys
sys.path.append("../libs")
from metrics import *
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PATH = "../libs/labeled-datasets_embedding/"
PATH_RAW = "../libs/labeled-datasets/npy"
DATASETS = os.listdir(PATH_RAW)

DR_LIST = ["lda", "umap", "tsne", "pca", "isomap"]
DR_NAME = ["LDA", "UMAP", "t-SNE", "PCA", "Isomap"]
METRICS = ["Silhouette", "1 - DSC"]
METRICS_EXECUTOR = {
	"silhouette": silhouette,
  "1 - dsc": lambda emb, labels: 1 - dsc(emb, labels)
}

if not os.path.exists("./results/01_preliminary.csv"):

	dataset_arr = []
	dr_arr = []
	metric_arr = []
	score_arr = []
	for dataset in tqdm(DATASETS):
		if not os.path.exists(f"{PATH}/{dataset}/lda.npy"):
			continue
		for dr in DR_LIST:
			emb = np.load(f"{PATH}/{dataset}/{dr}.npy")
			labels = np.load(f"{PATH_RAW}/{dataset}/label.npy")
			for metric in METRICS:
				score = METRICS_EXECUTOR[metric.lower()](emb, labels)
				dataset_arr.append(dataset)
				dr_arr.append(dr)
				metric_arr.append(metric)
				score_arr.append(score)

	df = pd.DataFrame({
		"Dataset": dataset_arr,
		"DR Technique": dr_arr,
		"Metric": metric_arr,
		"Score": score_arr
	})

	df.to_csv("./results/01_preliminary.csv", index=False)
else:
	df = pd.read_csv("./results/01_preliminary.csv")



sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
for i, metric in enumerate(METRICS):
  sns.pointplot(x="DR Technique", y="Score", hue="DR Technique", data=df[df["Metric"] == metric], ax=ax[i])
  ax[i].set_title(metric)
  ax[i].set_xticklabels(DR_NAME)

  ax[i].legend().remove()
  
plt.tight_layout()

plt.savefig("./plot/01_preliminary.png", dpi=300)
plt.savefig("./plot/01_preliminary.pdf", dpi=300)
