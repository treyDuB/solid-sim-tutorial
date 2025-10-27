# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
import utils
import time
import argparse
pygame.init()

import square_mesh_2 as square_mesh   # square mesh
import time_integrator
import timesheet


parser = argparse.ArgumentParser(
    description="Run simulation with optional integer levels for little_l and big_l."
)

parser.add_argument(
    "--little_l",
    type=int,
    default=0,
    help="Integer level for little_l (default: 0)"
)

parser.add_argument(
    "--big_l",
    type=int,
    default=2,
    help="Integer level for big_l (default: 2)"
)

args = parser.parse_args()

little_l = args.little_l
big_l = args.big_l

print(f"Running simulation from levels: little_l = {little_l}, big_l = {big_l}")

# simulation setup
side_len = 1.5
rho = 1000      # density of square
k = 2e5         # spring stiffness
n_seg = 4       # num of segments per side of the square
h = 0.01        # time step size in s
DBC = []   # no nodes need to be fixed
y_ground = -1   # height of the planar ground

# setup levels
level = 0 if little_l < 0 else (2 if little_l > 2 else little_l)
big_L = 2 if big_l > 2 else (little_l if big_l < big_l else big_l)

# initialize simulation # node positions and edge node indices
[x_0, e_0] = square_mesh.generate(side_len, n_seg)      # l = 0
[x_1, e_1] = square_mesh.generate(side_len, n_seg * 2)      # l = 1
[x_2, e_2] = square_mesh.generate(side_len, n_seg * 4)      # l = 2



# build projection matrices
P_0_1 = square_mesh.make_projection(x_0, x_1, e_1)
P_1_2 = square_mesh.make_projection(x_1, x_2, e_2)

a_0_1 = square_mesh.offset(x_0, x_1, P_0_1)
a_1_2 = square_mesh.offset(x_1, x_2, P_1_2)

P_0 = None
P_1 = None
a_0_2 = None


if big_L == 1:
    P_0 = P_0_1
    P_1 = None
if big_L == 2:
    P_0 = P_1_2 @ P_0_1
    P_1 = P_1_2
    a_0_2 = square_mesh.offset(x_0, x_2, P_0)


P = P_0 if level == 0 else (P_1 if level == 1 else None)
P2 = None if P == None else utils.expand_elementwise_projection(P)
if( P_0 is not None and P_1 is not None):
    print('Bullt projection P_0 ', P_0.shape)
    print('Bullt projection P_1 ', P_1.shape)

a_L = None
if level == 0:
    if big_L == 1:
        a_L = a_0_1
    if big_L == 2:
        a_L = a_0_2
if level == 1 and big_L == 2:
    a_L = a_1_2

x = x_0 if level == 0 else (x_1 if level == 1 else x_2)
e = e_0 if level == 0 else (e_1 if level == 1 else e_2)
x_L = x_0 if big_L == 0 else (x_1 if big_L == 1 else x_2)
e_L = e_0 if big_L == 0 else (e_1 if big_L == 1 else e_2)

if P is not None:
    square_mesh.check_projection(P, a_L, x, x_L)

num_nodes = len(x)
v = np.array([[0.0, 0.0]] * len(x))             # velocity
m_0 = [rho * side_len * side_len / ((n_seg + 1) * (n_seg + 1))] * len(x_0)  # calculate node mass evenly
m_1 = [rho * side_len * side_len / ((2 * n_seg + 1) * (2 * n_seg + 1))] * len(x_1)
m_2 = [rho * side_len * side_len / ((4 * n_seg + 1) * (4 * n_seg + 1))] * len(x_2)
m = m_0 if level == 0 else (m_1 if level == 1 else m_2)
m_L = m_0 if big_L == 0 else (m_1 if big_L == 1 else m_2)

# rest lengths squared
l2_0 = []
for i in range(0, len(e_0)):
    diff = x_0[e_0[i][0]] - x_0[e_0[i][1]]
    l2_0.append(diff.dot(diff))
l2_1 = []
for i in range(0, len(e_1)):
    diff = x_1[e_1[i][0]] - x_1[e_1[i][1]]
    l2_1.append(diff.dot(diff))
l2_2 = []
for i in range(0, len(e_2)):
    diff = x_2[e_2[i][0]] - x_2[e_2[i][1]]
    l2_2.append(diff.dot(diff))

l2 = l2_0 if level == 0 else (l2_1 if level == 1 else l2_2)
l2_L = l2_0 if big_L == 0 else (l2_1 if big_L == 1 else l2_2)


# spring stiffness
k_0 = [k] * len(e_0)
k_1 = [k] * len(e_1)
k_2 = [k] * len(e_2)
k = k_0 if level == 0 else (k_1 if level == 1 else k_2)
k_L = k_0 if big_L == 0 else (k_1 if big_L == 1 else k_2)
# identify whether a node is Dirichlet
is_DBC_0 = [False] * len(x_0)
is_DBC_1 = [False] * len(x_1)
is_DBC_2 = [False] * len(x_2)
# DBC on top two corners of each square
if False:
    is_DBC_0[n_seg] = True
    is_DBC_0[(n_seg + 1) * (n_seg + 1) - 1] = True
    is_DBC_1[2 * n_seg] = True
    is_DBC_1[(2 * n_seg + 1) * (2 * n_seg + 1) - 1] = True
    is_DBC_2[4 * n_seg] = True
    is_DBC_2[(4 * n_seg + 1) * (4 * n_seg + 1) - 1] = True
is_DBC = is_DBC_0 if level == 0 else (is_DBC_1 if level == 1 else is_DBC_2)
is_DBC_L = is_DBC_0 if big_L == 0 else (is_DBC_1 if big_L == 1 else is_DBC_2)




# ANCHOR: contact_area
contact_area_0 = [side_len / n_seg] * len(x_0)     # perimeter split to each node
contact_area_1 = [side_len / (2 * n_seg)] * len(x_1)
contact_area_2 = [side_len / (4 * n_seg)] * len(x_2)
contact_area = contact_area_0 if level == 0 else (contact_area_1 if level == 1 else contact_area_2)
# ANCHOR_END: contact_area

# simulation with visualization
resolution = np.array([900, 900])
offset = resolution / 2
scale = 200
def screen_projection(x):
    return [offset[0] + scale * x[0], resolution[1] - (offset[1] + scale * x[1])]

time_step = 0
end_time = 300
square_mesh.write_to_file(time_step, x, n_seg, level, big_L)
screen = pygame.display.set_mode(resolution)
running = True

# Time the simulation
times = timesheet.load_timesheet()
key = (str(level), str(big_L))
start = time.time()

max_iter = 0
iter_sum = 0

# Solve level 0
while running and time_step < end_time:
    # run until the user asks to quit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if pygame.key == 'p':
                running = False

    print('### Time step', time_step, '###')

    # fill the background and draw the square
    screen.fill((255, 255, 255))
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, y_ground]), screen_projection([2, y_ground]))   # ground
    for eI in e:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for i in range(len(x)):
        xI = x[i]
        if is_DBC[i]:
            pygame.draw.circle(screen, (255, 0, 0), screen_projection(xI), 0.12 * side_len / n_seg * scale)
        else:
            pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * side_len / n_seg * scale)

    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [x, v, iter] = time_integrator.step_forward(x, e, v, m, l2, k, y_ground, contact_area, is_DBC, h, 1e-2, P, a_L, P2, e_L, l2_L, k_L)
    max_iter = max(max_iter, iter)
    iter_sum += iter
    time_step += 1
    pygame.time.wait(int(h * 1000))
    square_mesh.write_to_file(time_step, x, n_seg, level, big_L)

# Save time
duration = time.time() - start
times[key] = duration
timesheet.save_timesheet(times)
print(f"Completed {key} in {duration:.2f}s")
print(' max iter = ', max_iter, ' avg iter = ', iter_sum / (time_step + 1))

end_time = time_step
time_step = 0



# print('### Replay Simulation ###')

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    #read position x
    x = square_mesh.read_from_file(time_step, num_nodes, level, big_L)

    x_L = x
    if(level == 0 and big_L >=1):
        x_L = np.column_stack([P_0 @ x[:,0], P_0 @ x[:,1]])
        if big_L == 1:
            x_L += a_0_1
        if big_L == 2:
            x_L += a_0_2
    if(level == 1 and big_L ==2):
        x_L = np.column_stack([P_1 @ x[:,0], P_1 @ x[:,1]]) + a_1_2


    # print('### replaying step', time_step, '###')
    # fill the background and draw the square
    screen.fill((255, 255, 255))
    pygame.draw.aaline(screen, (0, 0, 255), screen_projection([-2, y_ground]), screen_projection([2, y_ground]))   # ground
    for eI in e_L:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x_L[eI[0]]), screen_projection(x_L[eI[1]]))
    for i in range(len(x_L)):
        xI = x_L[i]
        if is_DBC_L[i]:
            pygame.draw.circle(screen, (255, 0, 0), screen_projection(xI), 0.12 * side_len / n_seg * scale)
        else:
            pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.1 * side_len / n_seg * scale)

    pygame.display.flip()   # flip the display

    time_step += 1
    if(time_step >= end_time): #loop time
        time_step = 0
    #simulate frametime
    pygame.time.wait(int(h * 1000))

pygame.quit()