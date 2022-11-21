'''
Application with hierarchical clustering (for Yun-Hsin)

'''

from sklearn.cluster import AgglomerativeClustering
import numpy as np 
from lsnc.lsnc import LSNC



class HierarchicalLSNC:
	def __init__(self, raw, emb, labels, cvm="btw_ch"):
		"""
		Initialize the instance
		"""
		self.raw    = raw
		self.emb    = emb
		self.labels = labels
		self.cvm    = cvm
		pass
	def run(self):
		"""
		Step 1. Perform the hierarhical clustering algorithm
		Step 2. Compute the LSNC score for each "hierarchy"
		        while using the class labels as a clustering of the hierarchy
		Step 3. Return the list of LSNC scores from the lowest level (fine-grained) to the highest level (coarse-grained) 
					  (in the form of numpy array)
		"""
		pass	