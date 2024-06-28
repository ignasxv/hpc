import numpy as np
import sklearn

import pandas as pd
import cuml

from cuml.cluster import KMeans as cuKmeans

from cuml.datasets import make_blobs
import time
import os
import dask_cudf
import cudf

if __name__=="__main__":
	n_samples = 100000000
	n_features = 5
	n_clusters = 8
	
	# Generate KMeans dataset
	X, Y = make_blobs(n_samples, 
                          n_features,
                          centers = 100, 
                          cluster_std=0.1)
	
	print(f'Number of Samples: {n_samples}')
	print(f'Number of features: {n_features}')
	print(f'Number of clusters: {n_clusters}')

	t1 = time.time()
	kmeans_cuml = cuKmeans(init="k-means||", 
                               n_clusters=n_clusters, 
                               random_state=100)

	kmeans_cuml.fit(X)
	t2 = time.time()
	total=t2-t1
	print(f'Time taken to fit kmeans model using cuML with 1 GPU: {total:.2f} secs')
