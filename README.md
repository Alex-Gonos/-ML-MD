# Recurrent Neural Networks for Molecular Dynamics

# Abstract
Systems in molecular dynamics are governed by ordinary differential equations, expressed through Hamilton's equations. As analytical solutions are not always possible for complex systems, simulations are required to gain intuition on these systems' behaviour. These simulated trajectories are typically generated using numerical methods, including integrators like Euler and Verlet. We explore the application of Neural Network based machine learning for the task of simulation. Applying deep, recurrent and Hamiltonian neural networks to simple and chaotic systems, we demonstrate the capability of these networks to produce fast, accurate results, that obey physical laws of conservation, in multiple example systems. 

# Contents of this Repository
This repository contains all the python code and notebooks used and referenced in my Masters project. These include the code necessary to construct and train neural networks of dense, recurrent and Hamiltonian architecture, 

In addition to the dynamical systems discussed in the paper, this repository also contains the code necessary to run the experiments on the 1D Lennard-Jones Oscillator and the 3D three-body chaotic system. It also contains the symplectic Euler integration scheme, alongside the symplectic Verlet, RK4 and Euler schema detailed in the paper.
