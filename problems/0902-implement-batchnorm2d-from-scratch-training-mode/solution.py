import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d
    c = x.shape[1]
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    variance = x.var(dim=(0, 2, 3), keepdim=True, correction=0)
    x_normalized = (x - mean) / torch.sqrt(variance + eps)

    gamma_reshaped = gamma.reshape((1, c, 1, 1))
    beta_reshaped = beta.reshape((1, c, 1, 1))

    y = gamma_reshaped * x_normalized + beta_reshaped

    return y
