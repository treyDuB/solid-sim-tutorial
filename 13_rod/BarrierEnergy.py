# ANCHOR: val_grad_hess
import math
import numpy as np
import utils

import distance.PointEdgeDistance as PE
import distance.CCD as CCD

dhat = 0.01
kappa = 1e5
eps = 1e-5

def val(x, y_ground, bp, be, contact_area):
    sum = 0.0
    #ground
    for i in range(0, len(x) - 4):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            sum += contact_area[i] * dhat * kappa / 2 * (s - 1) * math.log(s)
    # ANCHOR: value
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    sum += 0.5 * contact_area[xI] * dhat * kappa / 8 * (s - 1) * math.log(s)
    # ANCHOR_END: value
    return sum

def grad(x, y_ground, bp, be, contact_area):
    g = np.array([[0.0, 0.0]] * len(x))
    for i in range(0, len(x)-4):
        d = x[i][1] - y_ground
        if d < dhat:
            s = d / dhat
            g[i][1] += contact_area[i] * dhat * (kappa / 2 * (math.log(s) / dhat + (s - 1) / d))
    # ANCHOR: gradient
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_grad = 0.5 * contact_area[xI] * dhat * (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    g[xI] += local_grad[0:2]
                    g[eI[0]] += local_grad[2:4]
                    g[eI[1]] += local_grad[4:6]
    # ANCHOR_END: gradient
    return g

def hess(x, y_ground, bp, be, contact_area):
    IJV = [[0] * (len(x) - 4), [0] * (len(x) - 4), np.array([0.0] * (len(x) - 4))]
    for i in range(0, (len(x) - 4)):
        IJV[0][i] = i * 2 + 1
        IJV[1][i] = i * 2 + 1
        d = x[i][1] - y_ground
        if d < dhat:
            IJV[2][i] = contact_area[i] * dhat * kappa / (2 * d * d * dhat) * (d + dhat)
        else:
            IJV[2][i] = 0.0
    # ANCHOR: Hessian
    # self-contact
    dhat_sqr = dhat * dhat
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                d_sqr = PE.val(x[xI], x[eI[0]], x[eI[1]])
                if d_sqr < dhat_sqr:
                    d_sqr_grad = PE.grad(x[xI], x[eI[0]], x[eI[1]])
                    s = d_sqr / dhat_sqr
                    # since d_sqr is used, need to divide by 8 not 2 here for consistency to linear elasticity:
                    local_hess = 0.5 * contact_area[xI] * dhat * utils.make_PSD(kappa / (8 * d_sqr * d_sqr * dhat_sqr) * (d_sqr + dhat_sqr) * np.outer(d_sqr_grad, d_sqr_grad) \
                        + (kappa / 8 * (math.log(s) / dhat_sqr + (s - 1) / d_sqr)) * PE.hess(x[xI], x[eI[0]], x[eI[1]]))
                    index = [xI, eI[0], eI[1]]
                    for nI in range(0, 3):
                        for nJ in range(0, 3):
                            for c in range(0, 2):
                                for r in range(0, 2):
                                    IJV[0].append(index[nI] * 2 + r)
                                    IJV[1].append(index[nJ] * 2 + c)
                                    IJV[2] = np.append(IJV[2], local_hess[nI * 2 + r, nJ * 2 + c])
    # ANCHOR_END: Hessian
    return IJV
# ANCHOR_END: val_grad_hess

# ANCHOR: init_step_size
def init_step_size(x, y_ground, bp, be, p):
    alpha = 1
    for i in range(0, (len(x) - 4)):
        if p[i][1] < 0:
            alpha = min(alpha, 0.9 * (y_ground - x[i][1]) / p[i][1])
    # ANCHOR: line_search_filtering
    # self-contact
    for xI in bp:
        for eI in be:
            if xI != eI[0] and xI != eI[1]: # do not consider a point and its incident edge
                if CCD.bbox_overlap(x[xI], x[eI[0]], x[eI[1]], p[xI], p[eI[0]], p[eI[1]], alpha):
                    toc = CCD.narrow_phase_CCD(x[xI], x[eI[0]], x[eI[1]], p[xI], p[eI[0]], p[eI[1]], alpha)
                    if alpha > toc:
                        alpha = toc
    # ANCHOR_END: line_search_filtering
    return alpha
# ANCHOR_END: init_step_size