import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import InertiaEnergy
import MassSpringEnergy
import GravityEnergy
import BarrierEnergy

def step_forward(x, e, v, m, l2, k, y_ground, contact_area, is_DBC, h, tol, P = None, a_L = None, P2 = None, e_L = [], l2_L = [], k_L = []):
    x_tilde = x + v * h     # implicit Euler predictive position
    # TODO add progressive advancement
    x_n = copy.deepcopy(x)
    x_L = [] if P == None else np.column_stack([P @ x[:,0], P @ x[:,1]]) + a_L
    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, x_L, e_L, l2_L, k_L)
    p = search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h, P, P2, x_L, e_L, l2_L, k_L)
    while LA.norm(p, inf) / h > tol:
        # print('Iteration', iter, ':')
        # print('residual =', LA.norm(p, inf) / h)

        # ANCHOR: filter_ls
        # filter line search
        alpha = BarrierEnergy.init_step_size(x, y_ground, p)  # avoid interpenetration and tunneling
        x_alpha = x + alpha * p
        while IP_val(x_alpha, e, x_tilde, m, l2, k, y_ground, contact_area, h, x_L, e_L, l2_L, k_L) > E_last and alpha > 1e-6:
            x_alpha = x + alpha * p
            x_L = [] if P == None else np.column_stack([P @ x_alpha[:,0], P @ x_alpha[:,1]]) + a_L
            alpha /= 2
        # ANCHOR_END: filter_ls
        # print('step size =', alpha)

        x += alpha * p
        x_L = [] if P == None else np.column_stack([P @ x[:,0], P @ x[:,1]]) + a_L
        E_last = IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, x_L, e_L, l2_L, k_L)
        p = search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h, P, P2, x_L, e_L, l2_L, k_L)
        iter += 1
    if iter > 1: print('Took', iter, ' iterations')
    v = (x - x_n) / h   # implicit Euler velocity update
    return [x, v, iter]

def IP_val(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, x_L = [], e_L = [], l2_L = [], k_L = []):
    MS_val = MassSpringEnergy.val(x, e, l2, k) if len(x_L) == 0 else MassSpringEnergy.val(x_L, e_L, l2_L, k_L)
    return InertiaEnergy.val(x, x_tilde, m) + h * h * (MS_val + GravityEnergy.val(x, m) + BarrierEnergy.val(x, y_ground, contact_area))     # implicit Euler

def IP_grad(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, P = None, P2 = None, x_L = [], e_L = [], l2_L = [], k_L = []):
    MS_grad = MassSpringEnergy.grad(x, e, l2, k) if P is None else MassSpringEnergy.grad(x_L, e_L, l2_L, k_L, P)
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * (MS_grad + GravityEnergy.grad(x, m) + BarrierEnergy.grad(x, y_ground, contact_area))   # implicit Euler

def IP_hess(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, P = None, P2 = None, x_L = [], e_L = [], l2_L = [], k_L = []):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    H_MS = MassSpringEnergy.hess(x, e, l2, k) if P2 is None else MassSpringEnergy.hess(x_L, e_L, l2_L, k_L, P2)
    IJV_B = BarrierEnergy.hess(x, y_ground, contact_area)
    H_MS *= h * h    # implicit Euler
    IJV_B[2] *= h * h     # implicit Euler
    # IJV_In_MS = np.append(IJV_In, IJV_MS, axis=1)
    IJV = np.append(IJV_In, IJV_B, axis=1)
    H_C = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 2, len(x) * 2)).tocsr()
    # print('H_Course energy is ',H_C.shape,' H_MS is', H_MS.shape)
    H = H_C + H_MS
    return H

def search_dir(x, e, x_tilde, m, l2, k, y_ground, contact_area, is_DBC, h, P = None, P2 = None, x_L = [], e_L = [], l2_L = [], k_L = []):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, P, P2, x_L, e_L, l2_L, k_L)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, y_ground, contact_area, h, P, P2, x_L, e_L, l2_L, k_L).reshape(len(x) * 2, 1)
    # eliminate DOF by modifying gradient and Hessian for DBC:
    for i, j in zip(*projected_hess.nonzero()):
        if is_DBC[int(i / 2)] | is_DBC[int(j / 2)]:
            projected_hess[i, j] = (i == j)
    for i in range(0, len(x)):
        if is_DBC[i]:
            reshaped_grad[i * 2] = reshaped_grad[i * 2 + 1] = 0.0
    return spsolve(projected_hess, -reshaped_grad).reshape(len(x), 2)