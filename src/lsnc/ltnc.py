import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from cvm import *


class LabelTNC:
	def __init__(self, raw, emb, labels, cvm="btw_ch"):
		"""
		Initialize Label-TNC class.
		Run label-trustworthiness and label-continuity algorithm that evaluates
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
			Current support: (btw_ch, DSC (Distance Consistency))
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
			"btw_ch": btw_ch,
			"dsc": dsc_normalize
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
		lt_mat = raw_cvm_mat - emb_cvm_mat
		lt_mat[lt_mat < 0] = 0
		lt = 1 - np.sum(lt_mat) / (self.label_num * (self.label_num - 1) / 2)

		lc_mat = emb_cvm_mat - raw_cvm_mat
		lc_mat[lc_mat < 0] = 0
		lc = 1 - np.sum(lc_mat) / (self.label_num * (self.label_num - 1) / 2)

		## set the dictionary to return
		self.return_dict = {
			"lt": lt,
			"lc": lc,
			"f1": 2 * lt * lc / (lt + lc),
			"raw_mat": raw_cvm_mat,
			"emb_mat": emb_cvm_mat,
			"lt_mat": lt_mat,
			"lc_mat": lc_mat
		}

		return self.return_dict