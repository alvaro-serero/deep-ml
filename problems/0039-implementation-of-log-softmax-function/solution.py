import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores_shifted = scores - np.max(scores)
	return scores_shifted - np.log(np.sum(np.exp(scores_shifted)))