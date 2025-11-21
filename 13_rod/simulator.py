# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()

import rod   # rod mesh
import time_integrator

# simulation setup
side_len = 2
rho = 1000      # density
k = 1e5         # spring stiffness
n_seg = 20       # num of segments of the rod
h = 0.01        # time step size in s
DBC = [0,1,2,3]    # box nodes need to be fixed
y_ground = -0.0   # height of the planar ground

box_size = 1

box_min = np.array([-box_size / 2, y_ground])
box_max = np.array([box_size / 2, y_ground + box_size])
box = [box_min, box_max]



# initialize simulation
[x_rod, e_rod, elem] = rod.generate(side_len, n_seg, 1.5)  # node positions and edge node indices
x_box = [
    [box_min[0], box_min[1]],
    [box_max[0], box_min[1]],
    [box_max[0], box_max[1]],
    [box_min[0], box_max[1]]
]
box_start = len(x_rod)
e_box = np.array([
    [0+box_start,1+box_start],
    [1+box_start,2+box_start],
    [2+box_start,3+box_start],
    [3+box_start,0+box_start]
])
x = np.vstack((x_rod, x_box))
e = np.vstack((e_rod, e_box))
v = np.array([[0.0, 0.0]] * len(x))             # velocity
m_rod = rho * side_len / (n_seg + 1)        # rod mass
m = np.array([m_rod] * len(x_rod) + [0.0] * len(x_box)) #all masses

# rest length squared
l2 = []
for i in range(0, len(e)):
    diff = x[e[i][0]] - x[e[i][1]]
    l2.append(diff.dot(diff))
length = []
for i in range(0,len(elem)):
    length.append(np.linalg.norm(x[elem[i][2]] - x[elem[i][0]]))
k = [k] * len(e)    # rod stiffness
# identify whether a node is Dirichlet
is_DBC = [False] * len(x)
for i in DBC:
    is_DBC[i+box_start] = True

# is_DBC[0] = True
# is_DBC[len(x_rod) - 1] = True
# ANCHOR: contact_area
contact_area = np.zeros(len(x))
for i in range(len(x_rod)):
    contact_area[i] = side_len / n_seg  # uniform per segment node
for i in range(len(x_box)):
    contact_area[i+box_start] = box_size
# ANCHOR_END: contact_area

bp = list(range(len(x)))
be = e.copy()

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

time_step = 0
rod.write_to_file(time_step, x)
screen = pygame.display.set_mode(resolution)
running = True
while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    print('### Time step', time_step, '###')

    # fill the background and draw the square
    screen.fill((255, 255, 255))
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, y_ground]), screen_projection([2, y_ground]))   # ground
    for eI in e_box: # box
        pygame.draw.aaline(screen, (255, 0, 0), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for eI in e_rod:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for xI in x:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.05 * side_len / n_seg * scale)


    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [x, v] = time_integrator.step_forward(x, e, elem, v, m, l2, length, k, y_ground, bp, be, contact_area, is_DBC, h, 1e-2)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    rod.write_to_file(time_step, x)

pygame.quit()