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

import numpy as np

def g_RB2D(v01, v02, v11, v12, v21, v22):
    # Precompute temporaries
    t2 = v01 * 2.0
    t3 = v02 * 2.0
    t4 = v11 * 2.0
    t5 = v12 * 2.0
    t6 = v21 * 2.0
    t7 = v22 * 2.0

    t16 = -v11 + v01
    t17 = -v12 + v02
    t18 = -v21 + v11
    t19 = -v22 + v12
    t20 = t2 - t4
    t21 = t3 - t5
    t22 = t4 - t6
    t23 = t5 - t7

    t33 = t16 * t16 + t17 * t17
    t34 = t18 * t18 + t19 * t19
    t35 = t16 * t19 * 2.0 - t17 * t18 * 2.0

    t38 = np.sqrt(t33 * t34)
    t39 = 1.0 / t38
    t38 += t16 * t18 + t17 * t19
    t41 = 1.0 / (t38 * t38)

    g_tmp = t35 * t35 / (t38 ** 3)
    b_g_tmp = t20 * t34

    g = np.zeros(6)
    g[0] = t23 * t35 * t41 * 2.0 - g_tmp * (t18 + b_g_tmp * t39 / 2.0) * 2.0
    t34 *= t21
    g[1] = t22 * t35 * t41 * -2.0 - g_tmp * (t19 + t34 * t39 / 2.0) * 2.0
    t18 = t35 * t41
    t38 = t22 * t33
    g[2] = t18 * (t3 - t7) * -2.0 - g_tmp * (((-t4 + v01) + v21) - t39 * (b_g_tmp - t38) / 2.0) * 2.0
    b_g_tmp = t23 * t33
    g[3] = g_tmp * (((t5 - v02) - v22) + t39 * (t34 - b_g_tmp) / 2.0) * 2.0 + t18 * (t2 - t6) * 2.0
    g[4] = t21 * t35 * t41 * 2.0 + g_tmp * (t16 + t38 * t39 / 2.0) * 2.0
    g[5] = t20 * t35 * t41 * -2.0 + g_tmp * (t17 + b_g_tmp * t39 / 2.0) * 2.0

    return g

import numpy as np

def H_RB2D(v01, v02, v11, v12, v21, v22, eps=1e-12):
    # Pre-allocate H
    H = np.zeros((6, 6), dtype=float)

    t2 = v01 * 2.0
    t3 = v02 * 2.0
    t4 = v11 * 2.0
    t5 = v12 * 2.0
    t6 = v21 * 2.0
    t7 = v22 * 2.0

    t16 = -v11 + v01
    t17 = -v12 + v02
    t18 = -v21 + v11
    t19 = -v22 + v12
    t20 = t2 + -t4
    t21 = t3 + -t5
    t22 = t2 + -t6
    t23 = t3 + -t7
    t24 = t4 + -t6
    t25 = t5 + -t7

    t26 = t16 * t16
    t27 = t17 * t17
    t2 = t18 * t18
    t3 = t19 * t19
    t31 = t20 * t20
    t33 = t21 * t21
    t34 = t2 * 2.0
    t35 = t24 * t24
    t36 = t3 * 2.0
    t37 = t25 * t25
    t43 = t20 * t24
    t44 = t20 * t25
    t45 = t21 * t24
    t46 = t21 * t25
    t6 = t26 + t27
    t3 += t2
    t51 = t6 * t6
    t52 = t3 * t3
    t53 = t24 * t6
    t54 = t25 * t6
    t55 = t20 * t3
    t56 = t21 * t3
    t59 = t16 * t19 * 2.0 + -(t17 * t18 * 2.0)
    t7 = t59 * t59

    # sqrt(t6 * t3) guarded
    t2 = np.sqrt(max(t6 * t3, eps))
    t66 = 1.0 / t2
    t68 = t53 + -t55
    t69 = t54 + -t56
    t2 += t16 * t18 + t17 * t19
    t67 = t66 ** 3
    t70 = t6 * t66
    t71 = t3 * t66
    t73 = t44 * t66 / 2.0
    t74 = t45 * t66 / 2.0
    t82 = 1.0 / (t2 * t2)
    t83 = 1.0 / (t2 ** 3)
    t85 = t16 + t53 * t66 / 2.0
    t86 = t17 + t54 * t66 / 2.0
    t87 = t18 + t55 * t66 / 2.0
    t88 = t19 + t56 * t66 / 2.0
    t126 = t59 * t82 * 4.0
    t128 = ((-t4 + t66 * t68 / 2.0) + v01) + v21
    t2 = t24 * t25
    t3 = t20 * t21

    t243_tmp = t7 * (t82 * t82)
    b_t243_tmp = t21 * t59 * t83
    c_t243_tmp = t243_tmp * t85
    d_t243_tmp = t20 * t59 * t83

    t243 = (((-(t3 * t82 * 2.0) + t2 * t51 * t7 * t67 * t83 / 2.0) + b_t243_tmp * t86 * 4.0) +
            -(d_t243_tmp * t85 * 4.0)) + c_t243_tmp * t86 * 6.0

    t244_tmp = t25 * t59 * t83
    b_t244_tmp = t24 * t59 * t83
    c_t244_tmp = t243_tmp * t87

    t244 = (((-(t2 * t82 * 2.0) + t3 * t52 * t7 * t67 * t83 / 2.0) + b_t244_tmp * t87 * 4.0) +
            -(t244_tmp * t88 * 4.0)) + c_t244_tmp * t88 * 6.0

    t245_tmp = t7 * t83
    t16 = t43 * t66
    t245 = (((t46 * t82 * 2.0 + t244_tmp * t85 * 4.0) + -(b_t243_tmp * t87 * 4.0))
            + -(c_t243_tmp * t87 * 6.0)) + t245_tmp * (t16 / 4.0 + 1.0) * 2.0

    t246_tmp = t243_tmp * t86
    t6 = t46 * t66
    t246 = (((t43 * t82 * 2.0 + d_t243_tmp * t88 * 4.0) + -(b_t244_tmp * t86 * 4.0))
            + -(t246_tmp * t88 * 6.0)) + t245_tmp * (t6 / 4.0 + 1.0) * 2.0

    t247 = ((((-(t44 * t82 * 2.0) + -t126) + t244_tmp * t86 * 4.0) + d_t243_tmp * t87 * 4.0) +
            -(t246_tmp * t87 * 6.0)) + t7 * t73 * t83

    t248 = ((((-(t45 * t82 * 2.0) + t126) + -(b_t244_tmp * t85 * 4.0)) +
             -(b_t243_tmp * t88 * 4.0)) + -(c_t243_tmp * t88 * 6.0)) + t7 * t74 * t83

    t2 = t34 + t36
    t249_tmp = t23 * t59 * t83
    b_t249_tmp = t55 * t67
    t249 = (((-(t23 * t25 * t82 * 2.0) + t249_tmp * t87 * 4.0) +
             -(t244_tmp * t128 * 4.0)) + c_t244_tmp * t128 * 6.0) + t245_tmp * (((t2 + -t43) * t66 / 2.0 + b_t249_tmp * t68 / 4.0) - 1.0) * 2.0

    t5 = ((-t5 + t66 * t69 / 2.0) + v02) + v22
    t55 = t22 * t59 * t83
    t4 = t243_tmp * t88
    t18 = t56 * t67

    t83 = (((-(t22 * t24 * t82 * 2.0) + -(t55 * t88 * 4.0)) + b_t244_tmp * t5 * 4.0) +
           t4 * t5 * 6.0) + t245_tmp * (((t2 + -t46) * t66 / 2.0 + t18 * t69 / 4.0) - 1.0) * 2.0

    t19 = (((-(t22 * t23 * t82 * 2.0) + t249_tmp * t5 * 4.0) + -(t55 * t128 * 4.0)) +
           t243_tmp * t128 * t5 * 6.0) + t245_tmp * ((t44 + t45) * t66 / 2.0 + t67 * t68 * t69 / 4.0) * 2.0

    t3 = t54 * t67
    t17 = (((-(t20 * t22 * t82 * 2.0) + t55 * t86 * 4.0) + d_t243_tmp * t5 * 4.0) +
           -(t246_tmp * t5 * 6.0)) + -(t245_tmp * (((t6 / 2.0 + -t70) + t3 * t69 / 4.0) + 1.0) * 2.0)

    t2 = t53 * t67
    t16 = (((-(t21 * t23 * t82 * 2.0) + -(t249_tmp * t85 * 4.0)) + -(b_t243_tmp * t128 * 4.0)) +
           -(c_t243_tmp * t128 * 6.0)) + -(t245_tmp * (((t16 / 2.0 + -t70) + t2 * t68 / 4.0) + 1.0) * 2.0)

    t7 = ((((t20 * t23 * t82 * 2.0 + t126) + -(t249_tmp * t86 * 4.0)) + d_t243_tmp * t128 * 4.0) +
          -(t246_tmp * t128 * 6.0)) + -(t245_tmp * (t73 + t3 * t68 / 4.0) * 2.0)

    t6 = ((((t21 * t22 * t82 * 2.0 + -t126) + t55 * t85 * 4.0) + -(b_t243_tmp * t5 * 4.0)) +
          -(c_t243_tmp * t5 * 6.0)) + -(t245_tmp * (t74 + t2 * t69 / 4.0) * 2.0)

    t2 = ((((t23 * t24 * t82 * 2.0 + -t126) + t249_tmp * t88 * 4.0) + b_t244_tmp * t128 * 4.0) +
          t4 * t128 * 6.0) + -(t245_tmp * (t74 + -(t18 * t68 / 4.0)) * 2.0)

    t3 = ((((t22 * t25 * t82 * 2.0 + t126) + -(t55 * t87 * 4.0)) + -(t244_tmp * t5 * 4.0)) +
          c_t244_tmp * t5 * 6.0) + -(t245_tmp * (t73 + -(b_t249_tmp * t69 / 4.0)) * 2.0)

    H[0, 0] = ((t37 * t82 * 2.0 + t243_tmp * (t87 * t87) * 6.0) - t245_tmp * (t71 - t31 * t52 * t67 / 4.0) * 2.0) - t244_tmp * t87 * 8.0
    H[0, 1] = t244
    H[0, 2] = t249
    H[0, 3] = t3
    H[0, 4] = t245
    H[0, 5] = t247

    H[1, 0] = t244
    H[1, 1] = ((t35 * t82 * 2.0 + t243_tmp * (t88 * t88) * 6.0) - t245_tmp * (t71 - t33 * t52 * t67 / 4.0) * 2.0) + b_t244_tmp * t88 * 8.0
    H[1, 2] = t2
    H[1, 3] = t83
    H[1, 4] = t248
    H[1, 5] = t246

    H[2, 0] = t249
    H[2, 1] = t2
    t2 = ((t26 * 2.0 + t27 * 2.0) + t34) + t36
    H[2, 2] = ((t23 * t23 * t82 * 2.0 + t243_tmp * (t128 * t128) * 6.0) + t245_tmp * ((t67 * (t68 * t68) / 4.0 - t66 * (t2 - t43 * 2.0) / 2.0) + 2.0) * 2.0) + t249_tmp * t128 * 8.0
    H[2, 3] = t19
    H[2, 4] = t16
    H[2, 5] = t7

    H[3, 0] = t3
    H[3, 1] = t83
    H[3, 2] = t16
    H[3, 3] = ((t22 * t22 * t82 * 2.0 + t245_tmp * ((t67 * (t69 * t69) / 4.0 - t66 * (t2 - t46 * 2.0) / 2.0) + 2.0) * 2.0) + t243_tmp * (t5 * t5) * 6.0) - t55 * t5 * 8.0
    H[3, 4] = t6
    H[3, 5] = t17

    H[4, 0] = t245
    H[4, 1] = t248
    H[4, 2] = t16
    H[4, 3] = t6
    H[4, 4] = ((t33 * t82 * 2.0 + t243_tmp * (t85 * t85) * 6.0) - t245_tmp * (t70 - t35 * t51 * t67 / 4.0) * 2.0) + b_t243_tmp * t85 * 8.0
    H[4, 5] = t243

    H[5, 0] = t247
    H[5, 1] = t246
    H[5, 2] = t7
    H[5, 3] = t17
    H[5, 4] = t243
    H[5, 5] = ((t31 * t82 * 2.0 + t243_tmp * (t86 * t86) * 6.0) - t245_tmp * (t70 - t37 * t51 * t67 / 4.0) * 2.0) - d_t243_tmp * t86 * 8.0

    return H
