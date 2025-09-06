import numpy as np

def rbf_kernel(X, h=-1):
    """
    Compute RBF (Gaussian) kernel matrix and its gradients.
    Args:
        X: particles, shape (n, d)
        h: bandwidth, if -1 use median trick
    Returns:
        K: kernel matrix
        grad_K: gradient of kernel wrt X
    """
    n, d = X.shape
    sq_dist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)

    if h <= 0:
        h = np.median(sq_dist) / np.log(n + 1.0)

    K = np.exp(-sq_dist / h)

    grad_K = np.zeros((n, n, d))
    for i in range(n):
        for j in range(n):
            grad_K[i, j, :] = -2 * (X[i, :] - X[j, :]) / h * K[i, j]

    return K, grad_K
