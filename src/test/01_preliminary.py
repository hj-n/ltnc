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

DR_LIST = ["lda", "umap", "tsne"]
DR_NAME = ["LDA", "UMAP", "t-SNE"]
METRICS = ["Silhouette"]
METRICS_EXECUTOR = {
	"silhouette": silhouette
}



DATASETS_HIGH = [
 'olivetti_faces', 'weather', 'ph_recognition', 'seeds',
 'pen_based_recognition_of_handwritten_digits', 'iris', 'coil20',
 'optical_recognition_of_handwritten_digits', 'wireless_indoor_localization','mnist64'
]

# DATASETS_INTER = [
#  'glass_identification', 'cifar10', 'wine_quality', 'extyaleb', 'flickr_material_database', 
#   'turkish_music_emotion', 'breast_tissue', 'birds_bones_and_living_habits', 'cnae9', 'yeast', 

# ]

DATASETS_LOW = [
 'epileptic_seizure_recognition', 'customer_classification','skillcraft1_master_table_dataset',
 'street_view_house_numbers', 'heart_disease', 'wine_quality', 'world12d', 
'orbit_classification_for_prediction_nasa', 'siberian_weather_stats', 'hate_speech', 

]

def run_preliminary(file_name, datasets):
	if not os.path.exists(f"./results/{file_name}.csv"):
		dataset_arr = []
		dr_arr = []
		metric_arr = []
		score_arr = []
		for dataset in tqdm(datasets):
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

		df.to_csv(f"./results/{file_name}.csv", index=False)
	else:
		df = pd.read_csv(f"./results/{file_name}.csv")
	return df

df_high = run_preliminary("01_preliminary_high", DATASETS_HIGH)
df_low = run_preliminary("01_preliminary_low", DATASETS_LOW)



sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 1, figsize=(6, 2.6), sharex=True)
for i, df in enumerate([df_high, df_low]):
	sns.pointplot(y="DR Technique", x="Score", hue="DR Technique", data=df, ax=ax[i])
	ax[i].set_title("Datasets with Good CLM" if i == 0 else "Datasets with Bad CLM")
	ax[i].set_yticklabels(DR_NAME)
	ax[i].legend().remove()

	ax[i].set_xlabel("Silhouette Coefficient Score")

plt.tight_layout()

plt.savefig("./plot/01_preliminary.png", dpi=300)
plt.savefig("./plot/01_preliminary.pdf", dpi=300)
