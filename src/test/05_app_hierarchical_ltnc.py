"""
APPLICATION - DR analysis with Hierarchical Label-TNC
"""

import sys
sys.path.append('../ltnc')
from hierarchical_ltnc import HierarchicalLTNC
import numpy as np
from tqdm import tqdm
import os
import json
import warnings
warnings.filterwarnings("ignore")

GRANULARITY = 20
MEASURES = ["dsc", "btw_ch"]

print("Loading dataset...")
datasets = []
for directory in os.listdir("../../data/labeled-datasets/npy/"):
	if (
		directory != ".DS_Store"
		and directory != "README.md"
		and directory != ".gitignore"
		and not directory.endswith(".zip")
	):
		datasets.append(directory)

## load the data
raw_list     = []
label_list   = []
tsne_list    = []
umap_list    = []
pca_list	   = []
iso_list	   = []
lle_list     = []
densmap_list = []


for dataset in tqdm(datasets):
	raw = np.nan_to_num(np.load(f"../../data/labeled-datasets/npy//{dataset}/data.npy"))
	raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
	raw = raw[:, ~np.isnan(raw).any(axis=0)]
	raw_list.append(raw)
	label_list.append(np.load("../../data/labeled-datasets/npy//{}/label.npy".format(dataset)))
	tsne_list.append(np.load("../../data/labeled-datasets_embedding/{}/tsne.npy".format(dataset)))
	umap_list.append(np.load("../../data/labeled-datasets_embedding/{}/umap.npy".format(dataset)))
	pca_list.append(np.load("../../data/labeled-datasets_embedding/{}/pca.npy".format(dataset)))
	iso_list.append(np.load("../../data/labeled-datasets_embedding/{}/isomap.npy".format(dataset)))
	lle_list.append(np.load("../../data/labeled-datasets_embedding/{}/lle.npy".format(dataset)))
	densmap_list.append(np.load("../../data/labeled-datasets_embedding/{}/densmap.npy".format(dataset)))

for MEASURE in MEASURES:



	def runEmbeddings(emb_list):
		frames = []
		for idx, (raw, emb, label) in enumerate(zip(raw_list, emb_list, label_list)):
			#if idx != 0: continue
			try:
				hltnc = HierarchicalLTNC(raw, emb, cvm=MEASURE)
				result = hltnc.run(granularity=GRANULARITY)
				frames.append({
								"dataset": idx,
								"ls": result["ls"],
								"lc": result["lc"],
								"unique_raw_label": np.unique(label).tolist()
					})
			except: 
				pass
		return frames

	dr_types = ['tsne', 'umap', 'pca', 'iso', 'lle', 'densmap']
	embs = [tsne_list, umap_list, pca_list, iso_list, lle_list, densmap_list]


	print("Running Hierarchical LTNC...")
	frames = []
	for dr_type, emb in zip(dr_types, embs):
		results = runEmbeddings(emb)
		frames.append({
				'result': results,
				'dr_type': dr_type,
		})

	with open(f"./results/05_app_hierarchical_ltnc_{GRANULARITY}_{MEASURE}.json", "w") as f:
		json.dump(frames, f)




print("Finished!!")