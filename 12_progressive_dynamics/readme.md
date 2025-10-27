# Progressive Dynamics Mass-Spring Solids Simulation

Progressive Dynamics solves elastic energies of the course level at the fine level using a Projection
A square falling onto the ground under gravity is simulated with mass-spring elasticity potential and implicit Euler time integration.
Each time step is solved by minimizing the Incremental Potential with the projected Newton method.

## Dependencies

```
pip install numpy scipy pygame
```

## Run

```
python simulator.py --little_l 0 --big_l 2
```
