I am actively developing these classes and trying to add more examples. Please always pull the most recent changes.
# Languages
## Matlab
Original code is written in matlab. No installation is needed, the matlab files are located in the matlab directory.

## Python
To make this code more open source, I've also started porting it over to python as well
### Installation
This package makes use of numpy, sympy, and matplotlib. The python files are located in the python directory.
```
pip install numpy
pip install matplotlib
pip install sympy
```

# Main classes and functions
## ilqr
iLQR class which will solve for the optimal set of inputs and gains to get to a desired state

## ilqr_mpc
iLQR mpc class which will take a trajectory as an input, and solve for the optimal tracking input for a given horizon.
The mpc does not need to be ran on every timestep, and instead the feedback law and trajectory produced by the current solve can be used as a stabilizing controller for the rest of the horizon.

## ltvlqr
ltvlqr function solves for the optimal gain schedule givena trajectory. Use this function, if you already have a trajectory that you would like to track.

# Systems
## Acrobot
Double pendulum model without actuation on the first link.

## Car
A kinematic bicycle model where the inputs are steering velocity and linear acceleration.

## Cartpole
A cart pole model where the input is a thruster on the cart.

## Pendulum
Single pendulum with torque on the joint as input.

## Quadcopter
Quadcopter in full 3D space where the inputs are the 4 thrusters.