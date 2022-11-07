from snc.snc import SNC 
from metrics_provider import GlobalMeasure, LocalMeasure
from sklearn.metrics import silhouette_score
import time
from lsnc import lsnc

def lsnc_btw_ch_time(raw, emb, labels):
	"""
	compute the time duration of betweenness and closeness
	"""
	start = time.time()
	lsnc_obj = lsnc.LSNC(raw, emb, labels)
	lsnc_obj.run()
	end = time.time()
	return end - start

def silhouette(emb, labels):
	"""
	compute the silhouette score
	"""
	return silhouette_score(emb, labels)

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
