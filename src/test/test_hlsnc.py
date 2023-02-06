import sys
sys.path.append('../lsnc')
from hierarchical_lsnc import HierarchicalLSNC
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import itertools
import json

GRANULARITY = 20
MEASURE = "btw_ch"

if not os.path.exists(f'./shc_data/data_granularity_{GRANULARITY}_{MEASURE}.json'):
	## import the list of datasets
	datasets = []
	for directory in os.listdir("./labeled-datasets/npy/"):
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
		raw = np.nan_to_num(np.load(f"./labeled-datasets/npy/{dataset}/data.npy"))
		raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
		raw = raw[:, ~np.isnan(raw).any(axis=0)]
		raw_list.append(raw)
		label_list.append(np.load("./labeled-datasets/npy/{}/label.npy".format(dataset)))
		tsne_list.append(np.load("./labeled-datasets_embedding/{}/tsne.npy".format(dataset)))
		umap_list.append(np.load("./labeled-datasets_embedding/{}/umap.npy".format(dataset)))
		pca_list.append(np.load("./labeled-datasets_embedding/{}/pca.npy".format(dataset)))
		iso_list.append(np.load("./labeled-datasets_embedding/{}/isomap.npy".format(dataset)))
		lle_list.append(np.load("./labeled-datasets_embedding/{}/lle.npy".format(dataset)))
		densmap_list.append(np.load("./labeled-datasets_embedding/{}/densmap.npy".format(dataset)))


	def runEmbeddings(emb_list):
		frames = []
		for idx, (raw, emb, label) in enumerate(zip(raw_list, emb_list, label_list)):
			print(f'Dataset: {idx}')
			#if idx != 0: continue
			try:
				hlsnc = HierarchicalLSNC(raw, emb, cvm=MEASURE)
				result = hlsnc.run(granularity=GRANULARITY)
				frames.append({
								"dataset": idx,
								"ls": result["ls"],
								"lc": result["lc"],
								"unique_raw_label": np.unique(label).tolist()
					})
			except: 
				print(f"Error in: {idx}")
		return frames

	dr_types = ['tsne', 'umap', 'pca', 'iso', 'lle', 'densmap']
	embs = [tsne_list, umap_list, pca_list, iso_list, lle_list, densmap_list]

	frames = []
	for dr_type, emb in zip(dr_types, embs):
		results = runEmbeddings(emb)
		frames.append({
				'result': results,
				'dr_type': dr_type,
		})


	with open(f'./shc_data/data_granularity_{GRANULARITY}_{MEASURE}.json', 'w') as f:
		json.dump(frames, f)

else:
	with open(f'./shc_data/data_granularity_{GRANULARITY}_{MEASURE}.json', 'r') as f:
		frames = json.load(f)


def _reformat(nestedData: pd.DataFrame, column_name: str):
    decouples = nestedData[column_name].apply(pd.Series)
    decouples = pd.melt(decouples, value_vars=decouples.columns, var_name='level', value_name='score')
    decouples = decouples.loc[~decouples['score'].isna(), :]
    decouples['metric'] = column_name
    return decouples

stats = []
for each in frames:
    data = pd.DataFrame(each.get('result', None))
    data_ls = _reformat(data, 'ls')
    data_lc = _reformat(data, 'lc')
    data = pd.concat([data_ls, data_lc], ignore_index=True)
    data['type'] = each.get('dr_type', '')
    stats.append(data)

stats = pd.concat(stats, ignore_index=True)

lc_df = stats.loc[stats['metric'] == 'lc']


ls_df = stats.loc[stats['metric'] == 'ls']

lc_df = lc_df[lc_df['score'] <= 1]
lc_df = lc_df[lc_df['score'] >= 0]

ls_df = ls_df[ls_df['score'] <= 1]
ls_df = ls_df[ls_df['score'] >= 0]


sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.lineplot(x = "level", y = "score", hue = "type", data = ls_df, ax = ax[1], ci=0)
ax[0].set_title("Label-Continuity")
sns.lineplot(x = "level", y = "score", hue = "type", data = lc_df, ax = ax[0], ci=0, legend=False)
ax[1].set_title("Label-Trustworthiness")

plt.savefig(f"./shc_data/results_{GRANULARITY}_{MEASURE}.png")
plt.savefig(f"./shc_data/results_{GRANULARITY}_{MEASURE}.pdf")