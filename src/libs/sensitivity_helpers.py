import sys
sys.path.append('../')
sys.path.append('../ltnc')
sys.path.append('../libs')

from ltnc import ltnc
import metrics as mts

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

		