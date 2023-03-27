"""
APPLICATION - t-SNE Perplexity Analysis (Fashion-MNIST CLM analysis)
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../')
sys.path.append('../ltnc')
sys.path.append('../libs')
from ltnc import ltnc

## obtain matrices
raw   = np.load("../../data/labeled-datasets/npy/fashion_mnist/data.npy")
label = np.load("../../data/labeled-datasets/npy/fashion_mnist/label.npy")
emb   = np.load("../../data/labeled-datasets_embedding/fashion_mnist/tsne_64.npy")

ltnc_obj_dsc = ltnc.LabelTNC(raw, emb, label, "dsc")
dsc_raw_result = ltnc_obj_dsc.run()["raw_mat"]

ltnc_obj_btw_ch = ltnc.LabelTNC(raw, emb, label, "btw_ch")
btw_ch_raw_result = ltnc_obj_btw_ch.run()["raw_mat"]

## draw heatmap
fig, ax = plt.subplots(1, 2, figsize = (12, 5), gridspec_kw={"width_ratios": [1, 1.24]})
## margin between ax[0] and ax[1]
plt.subplots_adjust(wspace = 0.29)

for j in range(len(dsc_raw_result)):
  for i in range(len(dsc_raw_result)):
    if dsc_raw_result[i, j] == 0:
      dsc_raw_result[i, j] = dsc_raw_result[j, i]
    if btw_ch_raw_result[i, j] == 0:
      btw_ch_raw_result[i, j] = btw_ch_raw_result[j, i]
    if i == j:
      dsc_raw_result[i, j] = 1
      btw_ch_raw_result[i, j] = 1
    

sns.heatmap(dsc_raw_result, ax = ax[0], cmap = "RdBu", cbar = False, square = False, annot = True, fmt = ".2f", annot_kws = {"size": 8}, vmin = 0, vmax = 1)
ax[0].set_title("DSC", fontsize = 14)
sns.heatmap(btw_ch_raw_result, ax = ax[1], cmap = "RdBu", cbar = True, square = False, annot = True, fmt = ".2f", annot_kws = {"size": 8}, vmin = 0, vmax = 1)
ax[1].set_title("CH$_{btwn}$", fontsize = 14)



ax[0].set_xticklabels(["__;;"] * len(dsc_raw_result))
ax[0].set_yticklabels(["__;;"] * len(dsc_raw_result))
ax[1].set_yticklabels(["__;;"] * len(dsc_raw_result))
ax[1].set_xticklabels(["__;;"] * len(dsc_raw_result))
for ticklabels in [ax[0].get_xticklabels(), ax[0].get_yticklabels(), ax[1].get_xticklabels(), ax[1].get_yticklabels()]:
	for i, tick in enumerate(ticklabels):
		tick.set_backgroundcolor(sns.color_palette("tab10")[i])
		tick.set_alpha(0)
# plt.tight_layout()

## add circle patch
for i in [0, 1]:
	circle = plt.Circle((1.5, 3.5), 0.8, color = "black", fill = False, linewidth = 1.5)
	circle_red_1 = plt.Circle((2.5, 4.5), 0.8, color = "red", fill = False, linewidth = 1.5, linestyle = "--")
	circle_red_2 = plt.Circle((2.5, 6.5), 0.8, color = "red", fill = False, linewidth = 1.5, linestyle = "--")
	circle_red_3 = plt.Circle((4.5, 6.5), 0.8, color = "red", fill = False, linewidth = 1.5, 		linestyle = "--")
	if i == 0:
		ax[i].add_patch(circle)
	ax[i].add_patch(circle_red_1)
	ax[i].add_patch(circle_red_2)
	ax[i].add_patch(circle_red_3)
        
## add text as legends
for i, label in enumerate(["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]):
	ax[0].text(-1, i + 0.5, label, ha = "right", va = "center", fontsize = 10, color = "black")
	ax[1].text(-1, i + 0.5, label, ha = "right", va = "center", fontsize = 10, color = "black")
	ax[0].text(i + 0.75, 11, label, ha = "right", va = "top", fontsize = 10, color = "black", rotation = 45)
	ax[1].text(i + 0.75, 11, label, ha = "right", va = "top", fontsize = 10, color = "black", rotation = 45)

## add margin in bottom
plt.subplots_adjust(bottom = 0.2)

plt.savefig(f"./plot/04_app_tsne_perplexity_heatmap.png", dpi = 300)
plt.savefig(f"./plot/04_app_tsne_perplexity_heatmap.pdf", dpi = 300)
