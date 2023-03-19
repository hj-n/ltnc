import os 
import numpy as np
from metrics import *
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

path = "labeled-datasets_embedding"
datasets = os.listdir("labeled-datasets/npy/")


if not os.path.exists("./tnc_data/tnc.csv"):
	raw_list = []
	label_list = []
	tsne_list = []
	umap_list = []
	pca_list = []
	iso_list = []
	lle_list = []
	densmap_list = []




	for dataset in datasets:

		if dataset == ".DS_Store" or dataset == "README.md" or dataset == ".gitignore":
			continue
		if dataset[-4:] == ".zip":
			continue
		raw = np.nan_to_num(np.load(f"./labeled-datasets/npy/{dataset}/data.npy"))
		raw = (raw - raw.mean(axis=0)) / raw.std(axis=0)
		raw = raw[:, ~np.isnan(raw).any(axis=0)]
		raw_list.append(raw)
		label_list.append(np.load(f"./labeled-datasets/npy/{dataset}/label.npy"))
		tsne_list.append(np.load(f"./{path}/{dataset}/tsne.npy"))
		umap_list.append(np.load(f"./{path}/{dataset}/umap.npy"))
		pca_list.append(np.load(f"./{path}/{dataset}/pca.npy"))
		iso_list.append(np.load(f"./{path}/{dataset}/isomap.npy"))
		lle_list.append(np.load(f"./{path}/{dataset}/lle.npy"))
		densmap_list.append(np.load(f"./{path}/{dataset}/densmap.npy"))



	dr_list = []
	trust_list = []
	conti_list = []
	mrre_hl_list = []
	mrre_lh_list = []


	def run(raw, emb):
		trust = 0
		conti = 0
		mrre_hl = 0
		mrre_lh = 0
		for k in [5, 10, 15, 20, 25]:
			result = trust_conti_mrre(raw, emb, k)
			trust += result["trust"]
			conti += result["conti"]
			mrre_hl += result["mrre_hl"]
			mrre_lh += result["mrre_lh"]
		return {
			"trust": trust / 5,
			"conti": conti / 5,
			"mrre_hl": mrre_hl / 5,
			"mrre_lh": mrre_lh / 5
		}

	for i, raw in tqdm(enumerate(raw_list)):
		tsne_result = run(raw, tsne_list[i])
		dr_list.append("tsne")
		trust_list.append(tsne_result["trust"])
		conti_list.append(tsne_result["conti"])
		mrre_hl_list.append(tsne_result["mrre_hl"])
		mrre_lh_list.append(tsne_result["mrre_lh"])
		umap_result = run(raw, umap_list[i])
		dr_list.append("umap")
		trust_list.append(umap_result["trust"])
		conti_list.append(umap_result["conti"])
		mrre_hl_list.append(umap_result["mrre_hl"])
		mrre_lh_list.append(umap_result["mrre_lh"])
		pca_result = run(raw, pca_list[i])
		dr_list.append("pca")
		trust_list.append(pca_result["trust"])
		conti_list.append(pca_result["conti"])
		mrre_hl_list.append(pca_result["mrre_hl"])
		mrre_lh_list.append(pca_result["mrre_lh"])
		iso_result = run(raw, iso_list[i])
		dr_list.append("isomap")
		trust_list.append(iso_result["trust"])
		conti_list.append(iso_result["conti"])
		mrre_hl_list.append(iso_result["mrre_hl"])
		mrre_lh_list.append(iso_result["mrre_lh"])
		lle_result = run(raw, lle_list[i])
		dr_list.append("lle")
		trust_list.append(lle_result["trust"])
		conti_list.append(lle_result["conti"])
		mrre_hl_list.append(lle_result["mrre_hl"])
		mrre_lh_list.append(lle_result["mrre_lh"])
		densmap_result = run(raw, densmap_list[i])
		dr_list.append("densmap")
		trust_list.append(densmap_result["trust"])
		conti_list.append(densmap_result["conti"])
		mrre_hl_list.append(densmap_result["mrre_hl"])
		mrre_lh_list.append(densmap_result["mrre_lh"])



	df = pd.DataFrame({
		"dr": dr_list * 4,
		"measure": ["trust"] * len(dr_list) + ["conti"] * len(dr_list) + ["mrre_hl"] * len(dr_list) + ["mrre_lh"] * len(dr_list),
		"score": trust_list + conti_list + mrre_hl_list + mrre_lh_list,
	})
			
	df.to_csv("./tnc_data/tnc.csv", index=False)
else:
	df = pd.read_csv("./tnc_data/tnc.csv")





plt.figure(figsize=(7.5, 3.1))

sns.set_style("whitegrid")
sns.pointplot(x="measure", y="score", hue="dr", data=df, dodge=0.5, join=False, order=["trust", "conti", "mrre_lh", "mrre_hl"])

## change the legned into ["t-SNE", "UMAP", "PCA", "Isomap", "LLE", "Densmap"]
handles, labels = plt.gca().get_legend_handles_labels()
labels = ["t-SNE", "UMAP", "PCA", "Isomap", "LLE", "Densmap"]
plt.legend(handles, labels)

## change the x labels into ["Trustworthiness", "Continuity", "MRRE [LH]", "MRRE [HL]"]
xlabels = ["Trustworthiness", "Continuity", "MRRE [False]", "MRRE [Missing]"]
plt.gca().set_xticklabels(xlabels)

plt.gca().set_xlabel("")



## change x labels order

plt.tight_layout()


plt.savefig("./tnc_data/tnc.png", dpi=300)
plt.savefig("./tnc_data/tnc.pdf", dpi=300)