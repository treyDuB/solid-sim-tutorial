# ANCHOR: val_grad_hess
import math
import numpy as np
import utils

dhat = 0.01
kappa = 1e5
eps = 1e-5

def val(x, y_ground, box, contact_area):
    sum = 0.0
    for i in range(0, len(x)):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
        dx = max(eps, box[0][0] - x[i][0], x[i][0] - box[1][0])
        dy = max(eps, box[0][1] - x[i][1], x[i][1] - box[1][1])
        d = math.sqrt(dx**2 + dy**2)
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
    return sum

def grad(x, y_ground, box, contact_area):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(x)):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            g[i][1] += contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d))
        #box grad
        b = np.clip(x[i], box[0], box[1])
        d_vec = x[i] - b
        d = np.linalg.norm(d_vec)
        if d < dhat and d > eps:
            n = d_vec / d
            s = d / dhat
            w = contact_area[i]
            g[i] = contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d)) * n
    return g

def hess(x, y_ground, box, contact_area):
    IJV = [[0] * len(x), [0] * len(x), [0.0] * len(x)]
    for i in range(0, len(x)):
        IJV[0][i] = i * 2 + 1
        IJV[1][i] = i * 2 + 1
        d = x[i][1] - y_ground
        if d < dhat:
            IJV[2][i] = contact_area[i] * dhat * kappa / (2 * d * d * dhat) * (d + dhat)
        else:
            IJV[2][i] = 0.0
    #add box energy
    b = np.clip(x[i], box[0], box[1])
    d_vec = x[i] - b
    d = np.linalg.norm(d_vec)
    if d < dhat and d > eps:
        n = d_vec / d
        s = d / dhat
        f_prime = kappa / 2 * (math.log(s) + dhat * (s - 1) / d)
        f_double = kappa / 2 * (1 / s / dhat + (1 - s) * dhat / (d * d))
        local_hess = contact_area[i] * (f_double * np.outer(n, n) + f_prime / d * (np.eye(2) - np.outer(n, n)))
        for c in range(0, 2):
            for r in range(0, 2):
                IJV[0].append(i * 2 + r)
                IJV[1].append(i * 2 + c)
                IJV[2].append(local_hess[r, c])
    IJV[2] = np.array(IJV[2])
    return IJV
# ANCHOR_END: val_grad_hess

# ANCHOR: init_step_size
def init_step_size(x, y_ground, box, p):
    alpha = 1
    for i in range(0, len(x)):
        if p[i][1] < 0:
            alpha = min(alpha, 0.9 * (y_ground - x[i][1]) / p[i][1])
        if p[i][0] != 0 or p[i][1] != 0:
            hit, t = utils.ray_box_intersect(box[0], box[1], x[i], p[i])
            if hit and t >= 0:
                alpha = min(alpha, 0.9 * t)
    return alpha
# ANCHOR_END: init_step_size