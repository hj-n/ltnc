import sys
sys.path.append('../')
sys.path.append('../ltnc')
sys.path.append('../libs')

from ltnc import ltnc
import metrics as mts

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

from tqdm import tqdm

DR_MEASURES = [
  "lt_dsc", "lc_dsc", "lt_btw_ch", "lc_btw_ch", #### ours
  "trust", "conti", "mrre_lh", "mrre_hl", "stead", "cohev", "kl_div", "dtm", ### measures w/o labels
  "ca_trust", "ca_conti", "dsc", "sil", ### measures w/ labels
]

DR_MEASURES_NAME = [
  "Label-Trustworthiness [DSC]", "Label-Continuity [DSC]",
	"Label-Trustworthiness [CH$_{btwn}$]", "Label-Continuity [CH$_{btwn}$]", #### ours
  "Trustworthiness", "Continuity", "MRRE [False]", "MRRE [Missing]", 
	"Steadiness", "Cohesiveness", "1 - KL-Divergence", "1 - DTM", #### measures w/o labels
  "CA-Trustworthiness", "CA-Continuity", "1 - DSC", "Silhouette", #### measures w/ labels
]

DR_MEASURES_LINESTYLE = [
	"solid", "dashed", "solid", "dashed",    #### ours
	"solid", "dashed", "solid", "dashed", "solid", "dashed", #### measures w/o labels 
	(5, (10, 3)), (5, (10, 3)), "solid", "dashed", (5, (10, 3)), (5, (10, 3)) #### measures w/ labels
]
tab10  = sns.color_palette("tab10", 10)
dark10 = sns.color_palette("dark", 10)
DR_MEASURES_COLOR = [
	tab10[0], tab10[0], tab10[1], tab10[1],     #### ours
	tab10[2], tab10[2], tab10[3], tab10[3], tab10[4], tab10[4],  #### measures w/o labels
	tab10[5], tab10[6], tab10[7], tab10[7], tab10[8], tab10[9]   #### measures w/ labels
]
DR_MEASURES_PAIRCOLOR = [
	tab10[0], dark10[0], tab10[1], dark10[1],     #### ours
	tab10[2], dark10[2], tab10[3], dark10[3], tab10[4], dark10[4],  #### measures w/o labels
	tab10[5], tab10[6], tab10[7], dark10[7], tab10[8], tab10[9]   #### measures w/ labels
]

DR_MEASURES_TEXT_COLOR = ["red"] * 4  + ["blue"] * 8 + ["purple"] * 4
DR_MEASURES_MINUS_ONE = {"kl_div", "dtm"}

K_CANDIDATES = [5, 10, 15, 20, 25] ## k candiates for kNN
SIGMA_CANDIDATES = [0.01, 0.1, 1]  ## sigma candiates for Gaussian kernel

## run every measure for the given raw and emb data
def run_all_metrics(raw, emb, labels):
	## LT & LC [DSC]
	ltnc_dsc_obj        = ltnc.LabelTNC(raw, emb, labels, cvm="dsc")
	ltnc_dsc_results    = ltnc_dsc_obj.run() 
	## LT & LC [CH_BTW]
	ltnc_btw_ch_obj     = ltnc.LabelTNC(raw, emb, labels, cvm="btw_ch")
	ltnc_btw_ch_results = ltnc_btw_ch_obj.run()
	## S&C
	snc_results 				= mts.stead_cohev(raw, emb)
	## Silhoutte / DSC
	sil_result  				= mts.silhouette(emb, labels)	
	dsc_result				  = 1 - mts.dsc(emb, labels)
	## TNC, MRRE, CA-TNC
	trust_result, conti_result, mrre_lh_result, mrre_hl_result ,ca_trust_result, ca_conti_result = 0, 0, 0, 0, 0, 0
	for k in K_CANDIDATES:
		tnc_mrre_results = mts.trust_conti_mrre(raw, emb, k=k)
		ca_tnc_results   = mts.class_aware_trust_conti(raw, emb, labels, k=k)
		trust_result    += tnc_mrre_results["trust"]
		conti_result    += tnc_mrre_results["conti"]
		mrre_lh_result  += tnc_mrre_results["mrre_lh"]
		mrre_hl_result  += tnc_mrre_results["mrre_hl"]
		ca_trust_result += ca_tnc_results["ca_trust"]
		ca_conti_result += ca_tnc_results["ca_conti"]
	trust_result /= len(K_CANDIDATES)
	conti_result /= len(K_CANDIDATES)
	mrre_lh_result /= len(K_CANDIDATES)
	mrre_hl_result /= len(K_CANDIDATES)
	ca_trust_result /= len(K_CANDIDATES)
	ca_conti_result /= len(K_CANDIDATES)
	## KL, DTM
	kl_result, dtm_result = 0, 0
	for sigma in SIGMA_CANDIDATES:
		kl_dtm_results = mts.kl_div_rmse_dtm(raw, emb, sigma=sigma)
		kl_result += kl_dtm_results["kl_div"]
		dtm_result += kl_dtm_results["dtm"]
	kl_result /= len(SIGMA_CANDIDATES)
	dtm_result /= len(SIGMA_CANDIDATES)

	return {
		"lt_dsc": ltnc_dsc_results["lt"], "lc_dsc": ltnc_dsc_results["lc"],
		"lt_btw_ch": ltnc_btw_ch_results["lt"], "lc_btw_ch": ltnc_btw_ch_results["lc"],
		"trust": trust_result, "conti": conti_result, "mrre_lh": mrre_lh_result, "mrre_hl": mrre_hl_result,
		"stead": snc_results["stead"], "cohev": snc_results["cohev"], "kl_div": kl_result, "dtm": dtm_result,
		"ca_trust": ca_trust_result, "ca_conti": ca_conti_result, "dsc": dsc_result, "sil": sil_result
	}


def compute_metrics(raw_arr, emb_arr, labels):
	results = {}
	for measure in DR_MEASURES:
		results[measure] = []

	arr_len = None 
	if isinstance(raw_arr, list):
		arr_len = len(raw_arr)
	elif isinstance(emb_arr, list):
		arr_len = len(emb_arr)
	
	for i in tqdm(range(arr_len)):
		if isinstance(raw_arr, list):
			raw = raw_arr[i]
			emb = emb_arr
		else:
			raw = raw_arr
			emb = emb_arr[i]
		
		result_single = run_all_metrics(raw, emb, labels)
		for measure in DR_MEASURES:
			results[measure].append(result_single[measure])
	
	return results


def lineplot_ax(results, ax, index_range, x_label, y_label, title=None, invert_x_axis=False, show_x_label=False, show_y_label=False, show_title=False, xtick_int=False):
	x_list = results[x_label]
	for idx in range(index_range[0], index_range[1]):
		ax.plot(
			x_list, results[DR_MEASURES[idx]],
			label = DR_MEASURES_NAME[idx], linestyle = DR_MEASURES_LINESTYLE[idx], 
			color = DR_MEASURES_COLOR[idx],linewidth = 1.3
		)

	if invert_x_axis:
		ax.invert_xaxis()
	if show_x_label:
		ax.set_xlabel(x_label)
	if show_y_label:
		ax.set_ylabel(y_label)
	if xtick_int:
		ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
	if show_title:
		ax.set_title(title, fontsize=17)
	

def legend_ax(bbox_to_anchor, ncol, fontsize, ax):
	legend_elements = []
	for i, name in enumerate(DR_MEASURES_NAME):
		legend_elements.append(Line2D([0], [0], color=DR_MEASURES_COLOR[i], lw=1.5, label=name, linestyle=DR_MEASURES_LINESTYLE[i]))

	ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=bbox_to_anchor, ncol=ncol, fontsize=fontsize)
	for i, text in enumerate(ax.get_legend().get_texts()):
		text.set_color(DR_MEASURES_TEXT_COLOR[i])


def lineplot_agg(results, pair_idx_arr, titles, file_path):
	fig, axs = plt.subplots(1, len(pair_idx_arr), figsize=(3 * len(pair_idx_arr), 3))
	for i, pair in enumerate(pair_idx_arr):
		if type(pair) == tuple:
			filter_arr_first = results["measure"] == pair[0]
			filter_arr_second = results["measure"] == pair[1]
			if DR_MEASURES[pair[0]] in DR_MEASURES_MINUS_ONE:
				results.loc[filter_arr_first, "score"] = 1 - results.loc[filter_arr_first, "score"]
			if DR_MEASURES[pair[1]] in DR_MEASURES_MINUS_ONE:
				results.loc[filter_arr_second, "score"] = 1 - results.loc[filter_arr_second, "score"]
			filter_arr = np.logical_or(filter_arr_first, filter_arr_second)
			palette = [DR_MEASURES_PAIRCOLOR[pair[0]], DR_MEASURES_PAIRCOLOR[pair[1]]]
			hue_order = [DR_MEASURES_NAME[pair[0]], DR_MEASURES_NAME[pair[1]]]
		else:
			filter_arr = results["measure"] == pair
			if DR_MEASURES[pair] in DR_MEASURES_MINUS_ONE:
				results.loc[filter_arr, "score"] = 1 - results.loc[filter_arr, "score"]
			palette = [DR_MEASURES_PAIRCOLOR[pair]]
			hue_order = [DR_MEASURES_NAME[pair]]
		
		sns.lineplot(
			x = "perplexity", y="score", hue="measure", data=results[filter_arr], 
			ax=axs[i], palette=palette, hue_order=hue_order, legend=False
		)
		
		if type(pair) == tuple:
			axs[i].lines[0].set_linestyle(DR_MEASURES_LINESTYLE[pair[0]])
			axs[i].lines[1].set_linestyle(DR_MEASURES_LINESTYLE[pair[1]])
		
		axs[i].set_xscale("log")
		axs[i].set_xlabel("Perplexity ($\sigma$)")
		axs[i].set_ylabel("Score" if i == 0 else "")

		axs[i].legend(labels=hue_order, title=None)
		axs[i].set_title(titles[i])
	
	plt.tight_layout()


