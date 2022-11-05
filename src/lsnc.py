from btwim import calinski_harabasz as ch
import numpy as np


def run(raw, emb, labels, cvm="btw_ch"):
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
		e.g., {
			"ls": 0.7,
			"lc": 0.3,
			"f1": 0.5
		}
	"""