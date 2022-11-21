from btwim import calinski_harabasz as ch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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
		self.raw = np.array(raw)
		self.emb = np.array(emb)
		self.labels = np.array(labels)
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

	def run(self):
		"""
		run the label-stretching and label-compression algorithm
		return the score of ls and lc 
		"""
		## compute the label-pairwise cvm of the original data
		raw_cvm_mat = np.zeros((self.label_num, self.label_num))
		emb_cvm_mat = np.zeros((self.label_num, self.label_num))
		
		for label_i in range(self.label_num):
			for label_j in range(label_i + 1, self.label_num):
				## raw data of a pair of labels
				filter_label = np.logical_or(self.int_labels == label_i, self.int_labels == label_j)
				raw_pair = self.raw[filter_label]
				emb_pair = self.emb[filter_label]
				## label of the raw data of a pair of labels
				raw_pair_label = self.int_labels[filter_label]
				emb_pair_label = self.int_labels[filter_label]

				## change the label to 0 and 1
				raw_pair_label[raw_pair_label == label_i] = 0
				raw_pair_label[raw_pair_label == label_j] = 1
				emb_pair_label[emb_pair_label == label_i] = 0
				emb_pair_label[emb_pair_label == label_j] = 1

				## compute cvm
				raw_cvm_mat[label_i, label_j] = self.cvm(raw_pair, raw_pair_label)
				emb_cvm_mat[label_i, label_j] = self.cvm(emb_pair, emb_pair_label)
			
		## compute the label-stretching and label-compression score


		ls_mat = (emb_cvm_mat - raw_cvm_mat) 

	

		ls_mat[ls_mat < 0] = 0
		ls = 1 - np.sum(ls_mat) / (self.label_num * (self.label_num - 1) / 2)

		lc_mat = (raw_cvm_mat - emb_cvm_mat)

		lc_mat[lc_mat < 0] = 0
		lc = 1 - np.sum(lc_mat) / (self.label_num * (self.label_num - 1) / 2)

		## set the dictionary to return
		self.return_dict = {
			"ls": ls,
			"lc": lc,
			"f1": 2 * ls * lc / (ls + lc),
			"raw_mat": raw_cvm_mat,
			"emb_mat": emb_cvm_mat,
			"ls_mat": ls_mat,
			"lc_mat": lc_mat
		}

		return self.return_dict

	def visualize_heatmap(self, mat=["raw_mat", "emb_mat", "ls_mat", "lc_mat"], figsize=(10, 10), save_path=None):
		"""
		visualize the heatmap of the label-stretching and label-compression score
		the self.run() function should be run before this function
		save the heatmap as a file if save_path is given
		"""

		## set the figure size
		plt.figure(figsize=(figsize[0] * len(mat), figsize[1]))

		fig, ax = plt.subplots(1, len(mat))

		for i, m in enumerate(mat):
			## set the title
			if m == "raw_mat":
				title = "Raw data"
			elif m == "emb_mat":
				title = "Embedding"
			elif m == "ls_mat":
				title = "Label-stretching score"
			elif m == "lc_mat":
				title = "Label-compression score"
			else:
				raise ValueError("mat should be one of [raw_mat, emb_mat, ls_mat, lc_mat]")

			## set the heatmap
			sns.heatmap(self.return_dict[m], ax=ax[i], cmap="Blues", annot=True, fmt=".2f")
			ax[i].set_title(title)

		## save the figure
		if save_path is not None:
			plt.savefig(save_path)
		
		plt.show()
