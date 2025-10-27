import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse

def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))

def expand_projection(P):
    # P: (n_fine, n_coarse)
    n_fine, n_coarse = P.shape
    zeros = np.zeros_like(P)

    # Stack into block-diagonal form
    top = np.hstack([P, zeros])
    bottom = np.hstack([zeros, P])
    P2 = np.vstack([top, bottom])

    return P2

def expand_elementwise_projection(P):
    # """Expand P so that each scalar maps to two coordinates [x, y]."""
    if sparse.issparse(P):
        P = P.tocoo()
        rows = np.concatenate([2 * P.row, 2 * P.row + 1])
        cols = np.concatenate([2 * P.col, 2 * P.col + 1])
        data = np.concatenate([P.data, P.data])
        P_expanded = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(2 * P.shape[0], 2 * P.shape[1])
        ).tocsr()
    else:
        # Dense version
        n_fine, n_coarse = P.shape
        P_expanded = np.zeros((2 * n_fine, 2 * n_coarse))
        P_expanded[0::2, 0::2] = P
        P_expanded[1::2, 1::2] = P
    return P_expanded