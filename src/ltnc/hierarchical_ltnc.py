'''
Application with hierarchical clustering (for Yun-Hsin)

'''

import scipy.cluster.hierarchy as shc
import numpy as np 
from ltnc import LabelTNC



class HierarchicalLTNC:
	def __init__(self, raw, emb, labels=[], cvm="btw_ch"):
		"""
		Initialize the instance
		"""
		self.raw    = np.array(raw)
		self.emb    = np.array(emb)
		#self.labels = np.array(labels)
		self.cvm    = cvm
		self.clustering = None
		self.dists = None

	def run(self, granularity=5):
		"""
		Step 1. Perform the hierarhical clustering algorithm
		Step 2. Compute the LTNC score for each "hierarchy"
		        while using the class labels as a clustering of the hierarchy
		Step 3. Return the list of LTNC scores from the lowest level (fine-grained) to the highest level (coarse-grained) 
					  (in the form of numpy array)
		"""
		self._perform_hierarchical()
		# fine-grained -> coarse
		# skip min and max (i.e., every member is a cluster and everyone is in one cluster) distances
		hierarchy = np.linspace(self.dists[0], self.dists[1], num=(granularity+1), endpoint=False) #11
		ltnc_lt = []
		ltnc_lc = []
		for threshold in hierarchy[1:]:
			# get labels, which started with 1 by default in scipy, thus minus by 1
			assignment = np.array(shc.fcluster(self.clustering, threshold, criterion='distance')) - 1
			raw, emb, labels = self._filter_one_point_cluster(assignment)
			result = self._compute_ltnc(raw, emb, labels)
			ltnc_lt.append(result.get('lt', -1))
			ltnc_lc.append(result.get('lc', -1))

		return {
			"lt": ltnc_lt,
			"lc": ltnc_lc
		}

	def _perform_hierarchical(self):
		self.clustering = shc.linkage(self.raw, method='ward')
		dists = self.clustering[:, 2]
		self.dists = np.array([np.min(dists), np.max(dists)])

	def _filter_one_point_cluster(self, assignment):
		keep_mask = np.full(np.size(assignment), True, dtype=bool)
		_, first_occur, counts = np.unique(assignment, return_index=True, return_counts=True)
		for occur_idx, frequency in zip(first_occur, counts):
			if frequency != 1: continue
			keep_mask[occur_idx] = False
		filtered_raw = self.raw[keep_mask, :]
		filtered_emb = self.emb[keep_mask, :]
		filtered_labels = assignment[keep_mask]
		# Reformat the labels
		mapping = dict([(each[1], each[0]) for each in enumerate(np.unique(filtered_labels))])
		reformat_labels = np.vectorize(mapping.get)(filtered_labels)
		return filtered_raw, filtered_emb, reformat_labels

	def _compute_ltnc(self, raw, emb, labels):
		lsnc_obj = LabelTNC(raw, emb, labels, cvm=self.cvm)
		result = lsnc_obj.run()
		return result
