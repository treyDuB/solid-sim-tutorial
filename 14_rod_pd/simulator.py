# Rod Falling On Box Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()
import utils

import rod   # rod mesh
import time_integrator

# simulation setup
side_len = 2
rho = 1000      # density
k_0 = 1e5         # spring stiffness
n_seg = 20       # num of segments of the rod
h = 0.01        # time step size in s
DBC = [0,1,2,3]    # box nodes need to be fixed
y_ground = 0.0   # height of the planar ground

# setup levels
level = 0
big_L = 1

box_size = 1

box_min = np.array([-box_size / 2, y_ground])
box_max = np.array([box_size / 2, y_ground + box_size])
box = [box_min, box_max]



# initialize simulation
start_height = y_ground + box_size + 0.01
[x_rod_0, e_rod_0, elem_0] = rod.generate(side_len, n_seg, start_height)  # node positions and edge node indices
[x_rod_1, e_rod_1, elem_1] = rod.generate(side_len, n_seg * 2, start_height)
[x_rod, e_rod, elem] = [x_rod_0, e_rod_0, elem_0]
[x_rod_L, e_rod_L, elem_L] = [x_rod_1, e_rod_1, elem_1]

# Projections
P = rod.make_projection(x_rod_0, x_rod_1, 4)
P2 = utils.expand_elementwise_projection(P)

# rod.check_projection(P, x_rod_0, x_rod_1, 4)

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
num_nodes = len(x)

#Fine grained Level
box_start_L = len(x_rod_L)
e_box_L = np.array([
    [0+box_start_L,1+box_start_L],
    [1+box_start_L,2+box_start_L],
    [2+box_start_L,3+box_start_L],
    [3+box_start_L,0+box_start_L]
])
x_L = np.vstack((x_rod_L, x_box))
e_L = np.vstack((e_rod_L, e_box_L))
m_rod_L = rho * side_len / (n_seg * 2 + 1)        # rod mass
m_L = np.array([m_rod_L] * len(x_rod_L) + [0.0] * len(x_box)) #all masses

# rest length squared
l2 = []
for i in range(0, len(e)):
    diff = x[e[i][0]] - x[e[i][1]]
    l2.append(diff.dot(diff))
l2_L = []
for i in range(0, len(e_L)):
    diff = x_L[e_L[i][0]] - x_L[e_L[i][1]]
    l2_L.append(diff.dot(diff))
length = []
for i in range(0,len(elem)):
    length.append(np.linalg.norm(x[elem[i][2]] - x[elem[i][0]]))
length_L = []
for i in range(0,len(elem_L)):
    length_L.append(np.linalg.norm(x_L[elem_L[i][2]] - x_L[elem_L[i][0]]))
k = [k_0] * len(e)    # rod stiffness
k_L = [k_0] * len(e_L)
# identify whether a node is Dirichlet
is_DBC = [False] * len(x)
for i in DBC:
    is_DBC[i+box_start] = True

is_DBC_L = [False] * len(x_L)
for i in DBC:
    is_DBC_L[i+box_start_L] = True

# is_DBC[0] = True
# is_DBC[len(x_rod) - 1] = True
# ANCHOR: contact_area
contact_area = np.zeros(len(x))
for i in range(len(x_rod)):
    contact_area[i] = side_len / n_seg  # uniform per segment node
for i in range(len(x_box)):
    contact_area[i+box_start] = box_size
contact_area_L = np.zeros(len(x_L))
for i in range(len(x_rod_L)):
    contact_area_L[i] = side_len / (n_seg * 2)  # uniform per segment node
for i in range(len(x_box)):
    contact_area_L[i+box_start_L] = box_size

# ANCHOR_END: contact_area

# bp = list(range(len(x)))
# be = e.copy()

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

time_step = 0
end_time = 100
rod.write_to_file(time_step, x, len(x), level, big_L)
screen = pygame.display.set_mode(resolution)
running = True
while running and time_step < end_time:
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
    [x, v] = time_integrator.step_forward(x, None, e, elem, v, m, l2, length, k, y_ground, contact_area, is_DBC, h, 1e-2, P, P2, e_L, elem_L, l2_L, length_L, k_L)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    rod.write_to_file(time_step, x, len(x), level, big_L)

# Compute fine Level
level = 1
time_step = 0
x = rod.read_from_file(0, len(x))
x = np.column_stack([P @ x[:,0], P @ x[:,1]])
rod.write_to_file(0, x, len(x), level, big_L)

# Setup
e = e_L
e_rod = e_rod_L
e_box = e_box_L
prev_num_nodes = num_nodes
num_nodes = len(x)
v = np.array([[0.0, 0.0]] * num_nodes)             # velocity
m = m_L
l2 = l2_L
k = k_L
contact_area = contact_area_L
elem = elem_L
length = length_L
is_DBC = is_DBC_L
prev_P = P
P = None
P2 = None

while running and time_step < end_time:
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

    #calculate projected position and velocity
    x_hat = None
    if(time_step < end_time - 1):
        prev_x_l = rod.read_from_file(time_step, prev_num_nodes, level - 1, big_L)
        this_x_l = rod.read_from_file(time_step+1, prev_num_nodes, level - 1, big_L)
        diff_x_l = (this_x_l - prev_x_l)
        x_hat = x + np.column_stack([prev_P @ diff_x_l[:,0], prev_P @ diff_x_l[:,1]])

    # step forward simulation and wait for screen refresh
    [x, v] = time_integrator.step_forward(x, x_hat, e, elem, v, m, l2, length, k, y_ground, contact_area, is_DBC, h, 1e-2, P, P2, e_L, elem_L, l2_L, length_L, k_L)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    rod.write_to_file(time_step, x, len(x), level, big_L)



time_step = 0
running = True
print('### Replaying simulation ###')

while running:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    x = rod.read_from_file(time_step, len(x), level, big_L)


    # fill the background and draw the square
    screen.fill((255, 255, 255))
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, y_ground]), screen_projection([2, y_ground]))   # ground
    for eI in e_box: # box
        pygame.draw.aaline(screen, (255, 0, 0), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for eI in e_rod_1:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for xI in x:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.05 * side_len / n_seg * scale)


    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    pygame.time.wait(int(h * 2000))
    time_step += 1
    if(time_step >= end_time):
        time_step = 0


pygame.quit()