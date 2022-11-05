from btwim import calinski_harabasz as ch
import numpy as np


def run(raw, emb, labels, cvm="btw_ch", return_matrix=True):
	"""
	Run label-stretching and label-compression algorithm that evaluates
	the reliability of dimensionality reduction algorithms

	Parameters
	----------
	raw : numpy.ndarray: (n, d)
		Original data
	emb : numpy.ndarray: (m, d) where m < n
		Embedding of original data
	labels : numpy.ndarray: (n,)
		Labels of original data
	cvm: str, optional
		Cluster valididation measure to use. Default is btw_ch (Between-dataset calinski-harabasz index)
		Currently, only btw_ch is supported. 

	Returns
	-------
	lsnc_score : dict
		Label-stretching (ls) and label-compression (lc) score
		optional: return the intermediate results (matrix) of ls and lc by setting return_matrix=True
		e.g., {
			"ls": float
			"lc": float
			"ls_mat": np.ndarray: (n, n),  // only if return_matrix flag is set
			"lc_mat": np.ndarray: (n, n)   // only if return_matrix flag is set
			"f1": float
		}
	"""


	## change label into 0, 1, 2,....
	unique_labels = np.unique(labels)
	label_dict = {}
	for i, label in enumerate(unique_labels):
		label_dict[label] = i
	int_labels = np.array([label_dict[label] for label in labels])
	label_num = len(unique_labels)

	## set the cvm to be used (currently only btw_ch is supported)
	cvm = {
		"btw_ch": ch.btw
	}[cvm]

	## compute the label-pairwise cvm of the original data
	raw_cvm_mat = np.zeros((label_num, label_num))
	emb_cvm_mat = np.zeros((label_num, label_num))
	
	for label_i in range(label_num):
		for label_j in range(label_num):
			## raw data of a pair of labels
			raw_pair = raw[int_labels == label_i | int_labels == label_j]
			emb_pair = emb[int_labels == label_i | int_labels == label_j]
			## label of the raw data of a pair of labels
			raw_pair_label = int_labels[int_labels == label_i | int_labels == label_j]
			emb_pair_label = int_labels[int_labels == label_i | int_labels == label_j]
			## compute cvm
			raw_cvm_mat[label_i, label_j] = cvm(raw_pair, raw_pair_label)
			emb_cvm_mat[label_i, label_j] = cvm(emb_pair, emb_pair_label)
		
	## compute the label-stretching and label-compression score
	ls_mat = (emb_cvm_mat - raw_cvm_mat) 
	ls_mat[ls_mat < 0] = 0
	ls = np.sum(ls_mat) / (label_num * label_num)

	lc_mat = (raw_cvm_mat - emb_cvm_mat)
	lc_mat[lc_mat < 0] = 0
	lc = np.sum(lc_mat) / (label_num * label_num)

	## set the dictionary to return
	return_dict = {
		"ls": ls,
		"lc": lc,
		"f1": 2 * ls * lc / (ls + lc)
	}

	if return_matrix:
		return_dict["ls_mat"] = ls_mat
		return_dict["lc_mat"] = lc_mat
	
	return return_dict

