import numpy as np
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

    return [x, e]

def write_to_file(frameNum, x):
    # Check if 'output' directory exists; if not, create it
    if not os.path.exists('output'):
        os.makedirs('output')

    # create obj file
    filename = f"output/{frameNum}_rod.txt"
    with open(filename, 'w') as f:
        # write vertex coordinates
        for row in x:
            f.write(f"v {float(row[0]):.6f} {float(row[1]):.6f} 0.0\n")