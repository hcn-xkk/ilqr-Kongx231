# ilqr
Matlab iLQR class which will solve for the optimal set of inputs and gains to get to a desired state

# ilqr_mpc
Matlab iLQR mpc class which will take a trajectory as an input, and solve for the optimal tracking input for a given horizon.

# Systems
## Acrobot

## Car
A kinematic bicycle model where the inputs are steering velocity and linear acceleration.

## Cartpole
A cart pole model where the input is a thruster on the cart.

## Pendulum
Single pendulum with torque on the joint as input.

## Quadcopter
Quadcopter in full 3D space where the inputs are the 4 thrusters.

# Instructions
To run ilqr examples, go to example_*.m to create a trajectory which gets to a target state.

To run mpc examples, go to example_*_ilqr_mpc.m to track the trajectory produced by example_*.m

For your own system, it is advised to create a similar file as the "symbolic_dynamics".



