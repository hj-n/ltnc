from btwim import calinski_harabasz as ch
import numpy as np

def btw_ch(data, labels):
	return ch.btw(data, labels)

def dsc(data, labels):
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
		centroids.append(np.mean(data[labels == i], axis = 0))
	
	## compute distance consistency
	consistent_num = 0
	for idx in range(data.shape[0]):
		current_label = -1
		current_dist = 1e10
		for c_idx in range(len(centroids)):
			dist = np.linalg.norm(data[idx] - centroids[c_idx])
			if dist < current_dist:
				current_dist = dist
				current_label = c_idx
		if current_label == labels[idx]:
			consistent_num += 1
	
	return 1 - consistent_num / data.shape[0]
