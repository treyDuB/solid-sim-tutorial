# ANCHOR: val_grad_hess
import math
import numpy as np
import utils

stiff = 100
thickness = 0.1
pi = 3.14159


def val(x, elems, length):
    sum = 0.0
    for i in range(len(elems)):
        vertices = elems[i]
        x0 = x[vertices[0]]
        x1 = x[vertices[1]]
        x2 = x[vertices[2]]
        e0 = x1 - x0
        e1 = x2 - x1

        alpha = stiff * math.pow(thickness,3) /16.0
        kappa = (e0[0] * e1[1] - e0[1] * e1[0]) * 2.0 / math.sqrt(np.linalg.norm(e0) * np.linalg.norm(e1)) + e0.dot(e1)
        sum += alpha * kappa * kappa / length[i]

    return sum

def grad(x, elems, length):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(len(elems)):
        vertices = elems[i]
        x0 = x[vertices[0]]
        x1 = x[vertices[1]]
        x2 = x[vertices[2]]

        g_kappa2 = utils.g_RB2D(x0[0], x0[1], x1[0], x1[1], x2[0], x2[1])
        weight = stiff * math.pow(thickness, 3) / (16.0 * length[i])

        for local_idx in range(3):
            for d in range(2):
                g[vertices[local_idx]][d] += g_kappa2[local_idx * 2 + d]
    return g

def hess(x, elems, length):
    IJV = [[0] * 0, [0] * 0, [0.0] * 0]
    for i in range(len(elems)):
        vertices = elems[i]
        [i0,i1,i2] = vertices
        x0 = x[vertices[0]]
        x1 = x[vertices[1]]
        x2 = x[vertices[2]]

        local_hess = utils.H_RB2D(x0[0], x0[1], x1[0], x1[1], x2[0], x2[1])
        weight = stiff * math.pow(thickness, 3) / (16.0 * length[i])

        ind_map = np.array([i0*2, i0*2+1, i1*2, i1*2+1, i2*2, i2*2+1])

        for r in range(6):
            for c in range(6):
                IJV[0].append(ind_map[r])
                IJV[1].append(ind_map[c])
                IJV[2].append(weight * local_hess[r, c])
    IJV[2] = np.array(IJV[2])
    return IJV
# ANCHOR_END: val_grad_hess