import numpy as np
import scipy.sparse as sparse
import os

def generate(side_length, n_seg):
    # sample nodes uniformly on a square
    num_nodes = (n_seg + 1) ** 2

    x = np.array([[0.0, 0.0]] * (num_nodes))
    step = side_length / n_seg

    # sample square grid
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg + 1):
            x[i * (n_seg + 1) + j] = [-side_length / 2 + i * step, -side_length / 2 + j * step]

    # connect the nodes with edges
    e = []
    # horizontal edges
    for i in range(0, n_seg):
        for j in range(0, n_seg + 1):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j])
    # vertical edges
    for i in range(0, n_seg + 1):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    # diagonals
    for i in range(0, n_seg):
        for j in range(0, n_seg):
            e.append([i * (n_seg + 1) + j, (i + 1) * (n_seg + 1) + j + 1])
            e.append([(i + 1) * (n_seg + 1) + j, i * (n_seg + 1) + j + 1])
    print('size x=', len(x) , ' and size e=', len(e) )

    return [x, e]

def make_projection(x_0, x_1, e_1, eps=1e-6):

    num_nodes_0 = len(x_0)
    num_nodes_1 = len(x_1)

    rows, cols, data = [], [], [] # store individual projections


    # 10 - 1  - 10 -  1  - 10
    # |  \   /  |  \  |  / |
    # 1  - 1  - 1  -  1  - 10
    # |  /   \  |  /  |  \ |
    # 10 - 1  - 10 -  1  - 10

    # Collect neighbors of x_1 nodes in x_0
    neighbors = [set() for _ in range(num_nodes_1)]

    # Find direct correspondence for nodes in x_0
    direct = [None] * num_nodes_1
    for i in range(num_nodes_1):
        for j in range(num_nodes_0):
            if np.linalg.norm(x_1[i] - x_0[j]) < eps:
                direct[i] = j

    # Find neighbors through edges
    for a, b in e_1:
        if direct[a] is not None and direct[b] is not None:
            continue
        if direct[a] is not None:
            neighbors[b].add(direct[a])
        elif direct[b] is not None:
            neighbors[a].add(direct[b])
        else:
            print("Nothing to project from \n")
    # Add projection of neighbors or direct mapping
    for i in range(num_nodes_1):
        if direct[i] is not None:
            rows.append(i)
            cols.append(direct[i])
            data.append(1.0)
        else:
            for j in neighbors[i]:
                dist = np.linalg.norm(x_1[i] - x_0[j])
                weight = 1.0 / (dist + eps)
                rows.append(i)
                cols.append(j)
                data.append(weight)

    P = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes_1, num_nodes_0))
    P = P.tocsr()

    # Scale rows by inverse weight to normalize weights to sum to 1
    row_sums = np.array(P.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_row_sums = 1.0 / row_sums
    D = sparse.diags(inv_row_sums)
    P = D @ P

    return P

def offset(x_0, x_1, P):
    x_0_proj = P @ x_0
    offset = x_1 - x_0_proj
    return offset

def check_projection(P, a, x_0, x_1, eps=1e-6):
    x_0_proj = P @ x_0
    if a is not None:
        x_0_proj += a
    sum_offset = 0.0
    max_offset = 0.0
    for i in range(len(x_1)):
        offset = np.linalg.norm(x_1[i] - x_0_proj[i])
        sum_offset = sum_offset + offset
        max_offset = max(max_offset, offset)
        if offset > eps:
            print(f"Node {i} offset too large: {offset}")
    avg_offset = sum_offset / len(x_1)
    print(f"Max offset: {max_offset}")
    print(f"Avg offset: {avg_offset}")
    return max_offset, avg_offset



def write_to_file(frameNum, x, n_seg, level = 0, big_L = 0):
    # Check if 'output_{level}' directory exists; if not, create it
    output = f"output/sim_{level}"
    if(big_L != 0):
        output = f"output/sim_{level}_{big_L}"
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

def read_from_file(frameNum, num_nodes, level = 0, big_L = 0):
    #read vertices out
    vertices = np.array([[0.0, 0.0]] * (num_nodes))
    output = f"output/sim_{level}"
    if( big_L != 0):
        output = f"output/sim_{level}_{big_L}"
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