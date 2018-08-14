#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import numpy as np

MAX_K = 20
has_trained_model = False

if __name__=='__main__':

	# Load necessary pickle files
	with open('pickles/x_full.pickle', 'rb') as p:
		x_full = pickle.load(p)

	print('Calculating BIC scores...')

	bic_scores = []
	for k in range(2, MAX_K+1):
		print('K =', k)
		temp_model = GaussianMixture(n_components=k, covariance_type='full', init_params='kmeans').fit(x_full)
		bic_scores.append((k, temp_model.bic(x_full)))

	plt.plot(list(zip(*bic_scores))[0], list(zip(*bic_scores))[1])
	plt.title('BIC Scores For Different Values of K')
	plt.xlabel('Number of Clusters')
	plt.ylabel('BIC Score')
	plt.xticks(np.arange(0, MAX_K+1, step=2))
	plt.savefig('bic_scores')
	plt.show()


	if has_trained_model:
		print('Generating t-SNE plot...')
		# Using first 10,000 samples because full dataset takes too long

		with open('pickles/best_model_16.pickle', 'rb') as p:
			best_model = pickle.load(p)
		
		labels = best_model.predict(x_full[:10000,:])
		x_embedded = TSNE(n_components=2, perplexity=30, early_exaggeration=50).fit_transform(x_full[:10000,:])

		plt.scatter(x_embedded[:,0], x_embedded[:,1], c=labels)
		plt.title("t-SNE Analysis")
		plt.savefig('tsne')
		plt.show()