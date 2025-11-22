import numpy as np
import scipy.sparse as sparse
import os

#generate a rod mesh
def generate(side_length, n_seg, height=0.0):
    # sample nodes uniformly on line
    x = np.array([[0.0, 0.0]] * (n_seg + 1))
    step = side_length / n_seg
    for i in range(0, n_seg + 1):
        x[i] = [-side_length / 2 + i * step, height]

    # connect the nodes with edges
    e = []
    # horizontal edges
    for i in range(0, n_seg):
        e.append([i, (i + 1)])
    # add joints
    elem = np.array([[0,0,0]] * (n_seg-1))
    for i in range(0, n_seg-1):
        elem[i] = [i, (i + 1), (i + 2)]

    return [x, e, elem]

def make_projection(x_0, x_1, extra = 0, eps=1e-6):
    num_nodes_0 = len(x_0)
    num_nodes_1 = len(x_1)

    rows, cols, data = [], [], [] # store individual projections
    # 10 - 1  - 10 -  1  - 10

    # add first shared point
    rows.append(0)
    cols.append(0)
    data.append(1.0)

    # add the rest
    i = 1 # for x_1
    j = 0 # for x_2
    while i < num_nodes_1:
        #add middle and next
        rows.append(i)
        cols.append(j)
        data.append(0.5)
        rows.append(i)
        cols.append(j+1)
        data.append(0.5)

        i += 1
        j += 1

        rows.append(i)
        cols.append(j)
        data.append(1.0)

        i += 1
    #add extra
    for e in range(extra):
        rows.append(num_nodes_1 + e)
        cols.append(num_nodes_0 + e)
        data.append(1.0)

    P = sparse.coo_matrix((data, (rows, cols)), shape=(num_nodes_1 + extra, num_nodes_0 + extra))
    P = P.tocsr()

    return P

def check_projection(P, x_0, x_1, eps=1e-6):
    x_0_proj = P @ x_0
    sum_offset = 0.0
    max_offset = 0.0
    for i in range(len(x_1)):
        offset = np.linalg.norm(x_1[i] - x_0_proj[i])
        sum_offset = sum_offset + offset
        max_offset = max(max_offset, offset)
        if offset > eps:
            print(f"Node {i} Needs offset: {offset}")
    avg_offset = sum_offset / len(x_1)
    print(f"Max offset: {max_offset}")
    print(f"Avg offset: {avg_offset}")
    return max_offset, avg_offset

def write_to_file(frameNum, x, num_nodes, level = 0, big_L = 0):
     # Check if 'output_{level}' directory exists; if not, create it
    output = f"output/sim_{level}"
    if(big_L != 0):
        output = f"output/sim_{level}_{big_L}"
    if not os.path.exists(output):
        os.makedirs(output)

    # create obj file
    filename = f"{output}/{frameNum}_rod.txt"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in x:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n")

def read_from_file(frameNum, num_nodes, level = 0, big_L = 0):
    #read vertices out
    vertices = np.array([[0.0, 0.0]] * (num_nodes))
    output = f"output/sim_{level}"
    if( big_L != 0):
        output = f"output/sim_{level}_{big_L}"
    filename = f"{output}/{frameNum}_rod.txt"
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
