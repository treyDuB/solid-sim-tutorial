import numpy as np
import scipy.sparse as sparse
import os

def generate(side_length, n_seg, level = 0, x_0 = None, e_0 = None):
    # sample nodes uniformly on a square
    num_nodes = (n_seg + 1) ** 2
    if (level > 0):
        num_nodes += n_seg * n_seg
    if (level > 1):
        num_nodes += (n_seg * (n_seg+1)) * 2

    x = np.array([[0.0, 0.0]] * (num_nodes))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]

    # sample center of squares
    center_start = (n_seg + 1) * (n_seg + 1)
    if(level > 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                x[center_start + (i * n_seg)+j] = [-side_length / 2 + (i + 0.5) * step, -side_length / 2 + (j + 0.5) * step]

    # divide verical and horizontal edges at midpoint
    mid_h_start = center_start + n_seg * n_seg
    mid_v_start = mid_h_start + n_seg * (n_seg + 1)
    if(level > 1):
        # horizontal midpoint
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                x[mid_h_start + (i * (n_seg + 1)) + j] = (
                    0.5 * (x[i * (n_seg + 1) + j] + x[(i + 1) * (n_seg + 1) + j])
                    )
        # vertical midpoint
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                x[mid_v_start + (i * (n_seg)) + j] = (
                    0.5 * (x[i * (n_seg + 1) + j] + x[i * (n_seg + 1) + j + 1])
                )

    # connect the nodes with edges
    e = []
    # horizontal and vertical segments 0,1
    if(level <= 1):
        # horizontal edges
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j])
        # vertical edges
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                e.append([i * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    # diagonals 0
    if(level == 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    #connect centers 1,2
    if(level > 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                center_idx = center_start + (i * n_seg) + j
                e.append([i * (n_seg + 1) + j, center_idx])
                e.append([center_idx, (i + 1) * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, center_idx])
                e.append([center_idx, i * (n_seg + 1) + j + 1])
    #connect midpoints 2
    if(level > 1):
        # horizontal midpoint edges
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                #x[mid_h_start + (i * (n_seg + 1)) + j] = x[i * (n_seg + 1) + j] + [0.5, 0.0]
                idx_left = i * (n_seg + 1) + j
                idx_right = (i + 1) * (n_seg + 1) + j
                center_idx_up = center_start + (i * n_seg) + j
                center_idx_down = center_start + i * n_seg + (j - 1)
                mid_idx = mid_h_start + ((i * (n_seg + 1)) + j)
                #connect left, right and center(down), (center(up))
                e.append([idx_left, mid_idx])
                e.append([mid_idx, idx_right])
                # cell centers above (j < n_seg) and below (j > 0)
                if(j < n_seg):
                    e.append([center_idx_up, mid_idx])
                if(j > 0):
                    e.append([mid_idx, center_idx_down])

        # vertical midpoint edges
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                #x[mid_v_start + (i * (n_seg)) + j] = x[i * (n_seg + 1) + j] + [0.0, 0.5]
                idx_up = i * (n_seg + 1) + j
                idx_down = i * (n_seg + 1) + j + 1
                center_idx_left = center_start + (i - 1) * n_seg + j
                center_idx_right = center_start + i * n_seg + j
                mid_idx = mid_v_start + (i * n_seg) + j
                #connect up, down, and center(right), (center (left))
                e.append([idx_up, mid_idx])
                e.append([mid_idx, idx_down])
                # cell centers to the left (i > 0) and right (i < n_seg)
                if(i > 0):
                    e.append([center_idx_left, mid_idx])
                if(i < n_seg):
                    e.append([mid_idx,center_idx_right])
    print('at level l = ', level , ': size x=', len(x) , ' and size e=', len(e) )

    return [x, e]

def make_projection(x_0, x_1, e_1, eps=1e-6):
    start_x = len(x_0)
    end_x = len(x_1)

    rows, cols, data = [], [], [] # store individual projections


    # Identity projection for vertices in v_0
    for i in range(start_x):
        rows.append(i)
        cols.append(i)
        data.append(1.0)

    # Find projections for vertices in x_1
    for a, b in e_1:
        i, j = None, None
        if a >= start_x and b < start_x:
            i, j = a, b
        elif b >= start_x and a < start_x:
            i, j = b, a

        if i is not None:
            dist = np.linalg.norm(x_1[i] - x_0[j])
            weight = 1.0 / (dist + eps)
            rows.append(i)
            cols.append(j)
            data.append(weight)

    P = sparse.coo_matrix((data, (rows, cols)), shape=(end_x, start_x))
    P = P.tocsr()

    # Scale rows by inverse weight to normalize weights to sum to 1
    row_sums = np.array(P.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = 1.0 / row_sums
    D = sparse.diags(inv_row_sums)
    P = D @ P

    return P

def generate_with_projection(side_length, n_seg, level = 0, x_0 = None, e_0 = None ):
    eps =1e-6
    # sample nodes uniformly on a square
    num_nodes = (n_seg + 1) ** 2
    if (level > 0):
        num_nodes += n_seg * n_seg
    if (level > 1):
        num_nodes += (n_seg * (n_seg+1)) * 2

    x = np.array([[0.0, 0.0]] * (num_nodes))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]

    # sample center of squares
    center_start = (n_seg + 1) * (n_seg + 1)
    if(level > 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                x[center_start + (i * n_seg)+j] = [-side_length / 2 + (i + 0.5) * step, -side_length / 2 + (j + 0.5) * step]

    # divide verical and horizontal edges at midpoint
    mid_h_start = center_start + n_seg * n_seg
    mid_v_start = mid_h_start + n_seg * (n_seg + 1)
    if(level > 1):
        # horizontal midpoint
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                x[mid_h_start + (i * (n_seg + 1)) + j] = (
                    0.5 * (x[i * (n_seg + 1) + j] + x[(i + 1) * (n_seg + 1) + j])
                    )
        # vertical midpoint
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                x[mid_v_start + (i * (n_seg)) + j] = (
                    0.5 * (x[i * (n_seg + 1) + j] + x[i * (n_seg + 1) + j + 1])
                )

    # connect the nodes with edges
    e = []
    # horizontal and vertical segments 0,1
    if(level <= 1):
        # horizontal edges
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j])
        # vertical edges
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                e.append([i * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    # diagonals 0
    if(level == 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    #connect centers 1,2
    if(level > 0):
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                center_idx = center_start + (i * n_seg) + j
                e.append([i * (n_seg + 1) + j, center_idx])
                e.append([center_idx, (i + 1) * (n_seg + 1) + j + 1])
                e.append([(i + 1) * (n_seg + 1) + j, center_idx])
                e.append([center_idx, i * (n_seg + 1) + j + 1])
    #connect midpoints 2
    if(level > 1):
        # horizontal midpoint edges
        for i in range(0, n_seg):
            for j in range(0, n_seg + 1):
                #x[mid_h_start + (i * (n_seg + 1)) + j] = x[i * (n_seg + 1) + j] + [0.5, 0.0]
                idx_left = i * (n_seg + 1) + j
                idx_right = (i + 1) * (n_seg + 1) + j
                center_idx_up = center_start + (i * n_seg) + j
                center_idx_down = center_start + i * n_seg + (j - 1)
                mid_idx = mid_h_start + ((i * (n_seg + 1)) + j)
                #connect left, right and center(down), (center(up))
                e.append([idx_left, mid_idx])
                e.append([mid_idx, idx_right])
                # cell centers above (j < n_seg) and below (j > 0)
                if(j < n_seg):
                    e.append([center_idx_up, mid_idx])
                if(j > 0):
                    e.append([mid_idx, center_idx_down])

        # vertical midpoint edges
        for i in range(0, n_seg + 1):
            for j in range(0, n_seg):
                #x[mid_v_start + (i * (n_seg)) + j] = x[i * (n_seg + 1) + j] + [0.0, 0.5]
                idx_up = i * (n_seg + 1) + j
                idx_down = i * (n_seg + 1) + j + 1
                center_idx_left = center_start + (i - 1) * n_seg + j
                center_idx_right = center_start + i * n_seg + j
                mid_idx = mid_v_start + (i * n_seg) + j
                #connect up, down, and center(right), (center (left))
                e.append([idx_up, mid_idx])
                e.append([mid_idx, idx_down])
                # cell centers to the left (i > 0) and right (i < n_seg)
                if(i > 0):
                    e.append([center_idx_left, mid_idx])
                if(i < n_seg):
                    e.append([mid_idx,center_idx_right])


    ###  make projection from previous level ###
    P = None
    if level > 0:

        if(x_0 is None or e_0 is None):
            print('Creating new projection at level ', level)
            x_0, e_0 = generate_with_projection(side_length, n_seg, level - 1)[:2]
        x_1 = x
        e_1 = e
        start_x = len(x_0)
        end_x = len(x_1)

        rows, cols, data = [], [], [] # store individual projections


        # Identity projection for vertices in v_0
        for i in range(start_x):
            rows.append(i)
            cols.append(i)
            data.append(1.0)

        # Find projections for vertices in x_1
        for a, b in e_1:
            i, j = None, None
            if a >= start_x and b < start_x:
                i, j = a, b
            elif b >= start_x and a < start_x:
                i, j = b, a

            if i is not None:
                if (level == 2 and center_start <= j < center_start + n_seg * n_seg):
                    continue
                else:
                    dist = np.linalg.norm(x_1[i] - x_0[j])
                    weight = 1.0 / (dist + eps)
                    rows.append(i)
                    cols.append(j)
                    data.append(weight)

        P = sparse.coo_matrix((data, (rows, cols)), shape=(end_x, start_x))
        P = P.tocsr()

        # Scale rows by inverse weight to normalize weights to sum to 1
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        inv_row_sums = 1.0 / row_sums
        D = sparse.diags(inv_row_sums)
        P = D @ P

    return [x, e, P]



def write_to_file(frameNum, x, n_seg, level = 0):
    # Check if 'output_{level}' directory exists; if not, create it
    output = f"output_{level}"
    if not os.path.exists(output):
        os.makedirs(output)

    # create obj file
    filename = f"{output}/{frameNum}.obj"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in x:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n")
        # write vertex indices for each triangle
        for i in range(0, n_seg):
            for j in range(0, n_seg):
                #NOTE: each cell is exported as 2 triangles for rendering
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1}\n")
                f.write(f"f {i * (n_seg+1) + j + 1} {(i+1) * (n_seg+1) + j+1 + 1} {i * (n_seg+1) + j+1 + 1}\n")

def read_from_file(frameNum, num_nodes, level = 0):
    #read vertices out
    vertices = np.array([[0.0, 0.0]] * (num_nodes))
    output = f"output_{level}"
    filename = f"{output}/{frameNum}.obj"
    i = 0
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y = map(float, parts[1:3])  # just take x and y
                vertices[i] = [x, y]
                i += 1
            if i >= num_nodes:
                return vertices
    return vertices