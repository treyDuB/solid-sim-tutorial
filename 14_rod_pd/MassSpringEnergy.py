import numpy as np
import utils
import scipy.sparse as sparse

def val(x, e, l2, k):
    sum = 0.0
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        sum += l2[i] * 0.5 * k[i] * (diff.dot(diff) / l2[i] - 1) ** 2
    return sum

def grad(x, e, l2, k, P = None):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        g_diff = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff
        g[e[i][0]] += g_diff
        g[e[i][1]] -= g_diff

    if P is None:
        return g
    else:
        # P2 = expand_elementwise_projection(P)
        g_proj = P.T @ g
        return g_proj

def hess(x, e, l2, k, P = None):
    IJV = [[0] * (len(e) * 16), [0] * (len(e) * 16), np.array([0.0] * (len(e) * 16))]
    for i in range(0, len(e)):
        [a,b] = e[i]
        diff = x[a] - x[b]
        H_diff = 2 * k[i] / l2[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - l2[i]) * np.identity(2))
        H_local = utils.make_PSD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))

        base = i * 16
        # add to global matrix
        for nI, a_node in enumerate([a, b]):
            for nJ, b_node in enumerate([a, b]):
                indStart = base + (nI * 2 + nJ) * 4
                for r in range(0, 2):
                    for c in range(0, 2):
                        idx = indStart + r * 2 + c
                        IJV[0][idx] = a_node * 2 + r
                        IJV[1][idx] = b_node * 2 + c
                        IJV[2][idx] = H_local[nI * 2 + r, nJ * 2 + c]
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    if P is None:
        return H
    else:
        temp = (H @ P)
        H_proj = P.T @ temp
        return H_proj