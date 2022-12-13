import numpy as np
import sys
sys.path.append('../')
## turn off warning
import warnings
warnings.filterwarnings("ignore")

from metrics import *
import os

from tqdm import tqdm


def run_all(dr_measure, datasets):
	"""
	compute the time duration of all datasets 
	"""
	results = []
	for dataset in tqdm(datasets):
		raw = np.load(f"./labeled-datasets/npy/{dataset}/data.npy")
		label = np.load(f"./labeled-datasets/npy/{dataset}/label.npy")
		tsne_emb = np.load(f"./labeled-datasets_embedding/{dataset}/tsne.npy")
		umap_emb = np.load(f"./labeled-datasets_embedding/{dataset}/umap.npy")
		pca_emb = np.load(f"./labeled-datasets_embedding/{dataset}/pca.npy")
		random_emb = np.load(f"./labeled-datasets_embedding/{dataset}/random.npy")

		tsne_time = run_single(dr_measure, raw, tsne_emb, label)
		umap_time = run_single(dr_measure, raw, umap_emb, label)
		pca_time = run_single(dr_measure, raw, pca_emb, label)
		random_time = run_single(dr_measure, raw, random_emb, label)
	
		results += [tsne_time, umap_time, pca_time, random_time]
	
	return results


def run_single(dr_measure, raw, emb, label):
	if dr_measure == "lsnc_btw_ch":
		return lsnc_btw_ch_time(raw, emb, label)
	elif dr_measure == "lsnc_dsc":
		return lsnc_dsc_time(raw, emb, label)
	elif dr_measure == "snc":
		return stead_cohev_time(raw, emb)
	elif dr_measure == "silhouette":
		return silhouette_time(emb, label)
	elif dr_measure == "tnc":
		return trust_conti_time(raw, emb)
	elif dr_measure == "mrre":
		return mrre_time(raw, emb)
	elif dr_measure == "kl_div":
		return kl_div_time(raw, emb)
	elif dr_measure == "dtm":
		return dtm_time(raw, emb)
	elif dr_measure == "dsc":
		return dsc_time(emb, label)

## generating the list of datasets
datasets = []
for directory in os.listdir("./labeled-datasets/npy/"):
	if (
		directory != ".DS_Store"
		and directory != "README.md"
		and directory != ".gitignore"
		and not directory.endswith(".zip")
	):
		datasets.append(directory)

dr_measure = "lsnc_dsc"
time_list = np.array(run_all(dr_measure, datasets))

np.save(f"./scalability_results/{dr_measure}_time.npy", time_list)




	
