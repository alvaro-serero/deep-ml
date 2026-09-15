import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
		Y = X

	nb_samples, nb_features_x = X.shape
	nb_features_y = Y.shape[1]
	corr_matrix = np.zeros((nb_features_x, nb_features_y))
	for i in range(nb_features_x):
		for j in range(nb_features_y):
			x_i = X[:, i]
			y_j = Y[:, j]
			
			# Means
			mean_i = np.mean(x_i)
			mean_j = np.mean(y_j)

			# Deviations
			diff_i = x_i - mean_i
			diff_j = y_j - mean_j

			covariance = np.sum(diff_i * diff_j) / nb_samples
			corr_matrix[i, j] = covariance / (np.std(x_i) * np.std(y_j))

	return corr_matrix