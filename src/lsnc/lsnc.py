from btwim import calinski_harabasz as ch
import numpy as np


class LSNC:
	def __init__(self, raw, emb, labels, cvm="btw_ch"):
		"""
		Initialize LSNC class.
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

		"""
		self.raw = raw
		self.emb = emb
		self.labels = labels
		self.cvm = cvm

		## change label into 0, 1, 2,....
		unique_labels = np.unique(self.labels)
		label_dict = {}
		for i, label in enumerate(unique_labels):
			label_dict[label] = i
		self.int_labels = np.array([label_dict[label] for label in labels])
		self.label_num = len(unique_labels)

		## set the cvm to be used (currently only btw_ch is supported)
		self.cvm = {
			"btw_ch": ch.btw
		}[cvm]

def run(self, return_matrix=True):
	"""
	run the label-stretching and label-compression algorithm
	return the score of ls and lc (return the intermediate matrix if the return_matrix flag is set)
	"""
	## compute the label-pairwise cvm of the original data
	raw_cvm_mat = np.zeros((self.label_num, self.label_num))
	emb_cvm_mat = np.zeros((self.label_num, self.label_num))
	
	for label_i in range(self.label_num):
		for label_j in range(self.label_num):
			## raw data of a pair of labels
			raw_pair = self.raw[self.int_labels == label_i | self.int_labels == label_j]
			emb_pair = self.emb[self.int_labels == label_i | self.int_labels == label_j]
			## label of the raw data of a pair of labels
			raw_pair_label = self.int_labels[self.int_labels == label_i | self.int_labels == label_j]
			emb_pair_label = self.int_labels[self.int_labels == label_i | self.int_labels == label_j]
			## compute cvm
			raw_cvm_mat[label_i, label_j] = self.cvm(raw_pair, raw_pair_label)
			emb_cvm_mat[label_i, label_j] = self.cvm(emb_pair, emb_pair_label)
		
	## compute the label-stretching and label-compression score
	ls_mat = (emb_cvm_mat - raw_cvm_mat) 
	ls_mat[ls_mat < 0] = 0
	ls = np.sum(ls_mat) / (self.label_num * self.label_num)

	lc_mat = (raw_cvm_mat - emb_cvm_mat)
	lc_mat[lc_mat < 0] = 0
	lc = np.sum(lc_mat) / (self.label_num * self.label_num)

	## set the dictionary to return
	self.return_dict = {
		"ls": ls,
		"lc": lc,
		"f1": 2 * ls * lc / (ls + lc)
	}

	if return_matrix:
		self.return_dict["ls_mat"] = ls_mat
		self.return_dict["lc_mat"] = lc_mat
	
	return self.return_dict
