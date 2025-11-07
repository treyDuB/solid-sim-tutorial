import numpy as np
import numpy.linalg as LA

def make_PSD(hess):
    [lam, V] = LA.eigh(hess)    # Eigen decomposition on symmetric matrix
    # set all negative Eigenvalues to 0
    for i in range(0, len(lam)):
        lam[i] = max(0, lam[i])
    return np.matmul(np.matmul(V, np.diag(lam)), np.transpose(V))

def ray_box_intersect(box_min, box_max, ray_origin, ray_dir):
    tmin = -float('inf')
    tmax = float('inf')
    for i in range(2):
        if abs(ray_dir[i]) < 1e-8:
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                return False, None
        else:
            t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
            t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            tmin = max(tmin, t_near)
            tmax = min(tmax, t_far)
            if tmin > tmax:
                return False, None
    if tmax < 0:
        return False, None
    return True, tmin if tmin >= 0 else tmax

def nearest_box_point(box_min, box_max, x):
    p = (0.0,0.0)
    for d in range(2):
        if x[d] < box_min[d]:
            p[d] = box_min[d]
        elif x[d] > box_max[d]:
            p[d] = box_max[d]
        else:
            p[d] = x[d]
    return p