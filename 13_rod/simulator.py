# Mass-Spring Solids Simulation

import numpy as np  # numpy for linear algebra
import pygame       # pygame for visualization
pygame.init()

import rod   # square mesh
import time_integrator

# simulation setup
side_len = 2
rho = 1000      # density of square
k = 4e4         # spring stiffness
n_seg = 20       # num of segments of the rod
h = 0.01        # time step size in s
DBC = []        # no nodes need to be fixed
y_ground = -0.0   # height of the planar ground

box_size = 1

box_min = np.array([-box_size / 2, y_ground])
box_max = np.array([box_size / 2, y_ground + box_size])
box = [box_min, box_max]

box_corners = [
    [box_max[0], box_min[1]],
    [box_max[0], box_min[1]],
    [box_max[0], box_max[1]],
    [box_min[0], box_max[1]],
    [box_min[0], box_min[1]]  # close the loop
]




# initialize simulation
[x, e, elem] = rod.generate(side_len, n_seg, 1.5)  # node positions and edge node indices
v = np.array([[0.0, 0.0]] * len(x))             # velocity
m = [rho * side_len * side_len / (n_seg + 1)] * len(x)  # calculate node mass evenly
# rest length squared
l2 = []
for i in range(0, len(e)):
    diff = x[e[i][0]] - x[e[i][1]]
    l2.append(diff.dot(diff))
length = []
for i in range(0,len(elem)):
    length.append(np.linalg.norm(x[elem[i][1]] - x[elem[i][0]]))

k = [k] * len(e)    # rod stiffness
# identify whether a node is Dirichlet
is_DBC = [False] * len(x)
for i in DBC:
    is_DBC[i] = True
# ANCHOR: contact_area
contact_area = [side_len / n_seg] * len(x)     # perimeter split to each node
# ANCHOR_END: contact_area

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
    for i in range(4): # box
        pygame.draw.aaline(
            screen, (255, 0, 0),
            screen_projection(box_corners[i]),
            screen_projection(box_corners[i + 1])
        )
    for eI in e:
        pygame.draw.aaline(screen, (0, 0, 255), screen_projection(x[eI[0]]), screen_projection(x[eI[1]]))
    for xI in x:
        pygame.draw.circle(screen, (0, 0, 255), screen_projection(xI), 0.05 * side_len / n_seg * scale)


    pygame.display.flip()   # flip the display

    # step forward simulation and wait for screen refresh
    [x, v] = time_integrator.step_forward(x, e, elem, v, m, l2, length, k, y_ground, box, contact_area, is_DBC, h, 1e-2)
    time_step += 1
    pygame.time.wait(int(h * 1000))
    rod.write_to_file(time_step, x)

pygame.quit()