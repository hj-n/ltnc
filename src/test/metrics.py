from snc.snc import SNC 
from metrics_provider import GlobalMeasure, LocalMeasure
from sklearn.metrics import silhouette_score
import time
import sys
sys.path.append('../')
sys.path.append("../lsnc")
from lsnc import lsnc
import numpy as np

def lsnc_btw_ch_time(raw, emb, labels):
	"""
	compute the time duration of betweenness and closeness
	"""
	start = time.time()
	lsnc_obj = lsnc.LSNC(raw, emb, labels, cvm="btw_ch")
	lsnc_obj.run()
	end = time.time()
	return end - start

def lsnc_dsc_time(raw, emb, labels):
	"""
	compute the time duration of distance consistency
	"""
	start = time.time()
	lsnc_obj = lsnc.LSNC(raw, emb, labels, cvm="dsc")
	lsnc_obj.run()
	end = time.time()
	return end - start

def silhouette(emb, labels):
	"""
	compute the silhouette score
	"""
	return silhouette_score(emb, labels)

def dsc(emb, labels):
	"""
	compute the distance consistency
	"""
	## contert labels to range from 0 to len(np.unique(labels)) - 1
	labels = np.array(labels)
	unique_labels = np.unique(labels)
	for i in range(len(unique_labels)):
		labels[labels == unique_labels[i]] = i

	## compute centroids
	centroids = []
	for i in range(len(np.unique(labels))):
		centroids.append(np.mean(emb[labels == i], axis = 0))
	
	## compute distance consistency
	consistent_num = 0
	for idx in range(emb.shape[0]):
		current_label = -1
		current_dist = 1e10
		for c_idx in range(len(centroids)):
			dist = np.linalg.norm(emb[idx] - centroids[c_idx])
			if dist < current_dist:
				current_dist = dist
				current_label = c_idx
		if current_label == labels[idx]:
			consistent_num += 1
	
	return 1 - consistent_num / emb.shape[0]

def dsc_time(emb, labels):
	"""
	compute the time duration of distance consistency 
	"""
	start = time.time()
	dsc(emb, labels)
	end = time.time()
	return end - start




def silhouette_time(emb, labels):
	"""
	compute the time duration of silhouette score
	"""
	start = time.time()
	silhouette_score(emb, labels)
	end = time.time()
	return end - start


def stead_cohev(raw, emb):
	"""
	compute the steadiness and cohesiveness 
	"""
	snc_obj = SNC(raw = raw, emb = emb, iteration = 300)
	snc_obj.fit()
	return {
		"stead": snc_obj.steadiness(), 
		"cohev": snc_obj.cohesiveness()
	}

def stead_cohev_time(raw, emb):
	"""
	compute the time duration of steadiness and cohesiveness
	"""
	start = time.time()
	snc_obj = SNC(raw = raw, emb = emb, iteration = 300)
	snc_obj.fit()
	end = time.time()
	return end - start

def trust_conti_mrre(raw, emb, k = 10):
	"""
	compute the trustworthiness and continuity
	"""
	lm = LocalMeasure(raw, emb, k = k)
	return {
		"trust": lm.trustworthiness(),
		"conti": lm.continuity(),
		"mrre_hl": lm.mrre_xz(),
		"mrre_lh": lm.mrre_zx()
	}

def trust_conti_time(raw, emb, k = 10):
	"""
	compute the time duration of trustworthiness and continuity
	"""
	start = time.time()
	lm = LocalMeasure(raw, emb, k = k)
	lm.trustworthiness()
	lm.continuity()
	end = time.time()
	return end - start

def mrre_time(raw, emb, k = 10):
	"""
	compute the time duration of mrre
	"""
	start = time.time()
	lm = LocalMeasure(raw, emb, k = k)
	lm.mrre_xz()
	lm.mrre_zx()
	end = time.time()
	return end - start


def kl_div_rmse_dtm(raw, emb, sigma = 0.1):
	"""
	computation of KL divergence
	"""
	gm = GlobalMeasure(raw, emb)

	return {
		"kl_div": gm.dtm_kl(sigma = sigma),
		"rmse": gm.rmse(),
		"dtm": gm.dtm(sigma = sigma)
	}

def kl_div_time(raw, emb, sigma = 0.1):
	"""
	compute the time duration of KL divergence
	"""
	start = time.time()
	gm = GlobalMeasure(raw, emb)
	gm.dtm_kl(sigma = sigma)
	end = time.time()
	return end - start

def dtm_time(raw, emb, sigma = 0.1):
	"""
	compute the time duration of KL divergence
	"""
	start = time.time()
	gm = GlobalMeasure(raw, emb)
	gm.dtm(sigma = sigma)
	end = time.time()
	return end - start
