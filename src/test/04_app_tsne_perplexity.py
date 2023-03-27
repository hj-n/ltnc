"""
APPLICATION - t-SNE Perplexity Analysis (Main experiment)
"""
from numba import cuda
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import sys

sys.path.append("../libs")
import sensitivity_helpers as sh

import warnings
warnings.filterwarnings("ignore")

CUDA_DEVICE_NUM = 1 ## SNC requires CUDA


PATH     = "../../data/labeled-datasets_embedding/"
PATH_RAW = "../../data/labeled-datasets/npy"
PP_LIST  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

DATASETS = os.listdir(PATH_RAW)



cuda.select_device(CUDA_DEVICE_NUM)

## load dataset and run metrics
if not os.path.exists("./results/04_app_tsne_perplexity.csv"):
	measure_arr = []
	dataset_arr = []
	perplexity_arr = []
	score_arr = []

	for dataset in tqdm(DATASETS):
		## load data
		raw = np.load(f"{PATH_RAW}/{dataset}/data.npy")
		label = np.load(f"{PATH_RAW}/{dataset}/label.npy")
		## standardize raw
		raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
		raw = raw[:, ~np.isnan(raw).any(axis=0)]
		for perplexity in PP_LIST:
			try:
				emb = np.load(f"{PATH}/{dataset}/tsne_{perplexity}.npy")
				pp_result = sh.run_all_metrics(raw, emb, label)
				for key in pp_result.keys():
					measure_arr.append(key)
					dataset_arr.append(dataset)
					perplexity_arr.append(perplexity)
					score_arr.append(pp_result[key])
			except:
				pass

	results = pd.DataFrame({
		"measure"    : measure_arr,
		"dataset"    : dataset_arr,
		"perplexity" : perplexity_arr,
		"score"			 : score_arr
	})

	results.to_csv("./results/04_app_tsne_perplexity.csv", index=False)
else:
	results = pd.read_csv("./results/04_app_tsne_perplexity.csv")

titles = [
	"(A) Label-T&C [CH$_{btwn}$]", 
	"(B) Label-T&C [DSC]", 
	"(A) Steadiness & Cohesivness", 
	"(B) Trustworthiness & Continuity", 
	"(C) MRREs",
	"(D) Steadiness & Cohesivness" 
	"(E) Global Measures"
	"(F) Class-Aware Trustworthiness & Continuity"
	"(G) CVMs"
]

sh.lineplot_agg(results, [(0, 1), (2, 3)], titles[0:2], "./results/04_app_tsne_perplexity")
sh.lineplot_agg(results, [(4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17)], titles[2:], "./results/04_app_tsne_perplexity_others")

