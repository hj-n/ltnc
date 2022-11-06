from snc.snc import SNC 
from metrics_provider import GlobalMeasure, LocalMeasure
from sklearn.metrics import silhouette_score

def silhouette(emb, labels):
	"""
	compute the silhouette score
	"""
	return silhouette_score(emb, labels)


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
