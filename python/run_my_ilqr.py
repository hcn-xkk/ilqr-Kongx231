import numpy as np
import matplotlib.animation as animation
from my_ilqr import ILQR
from cost_function import CostFunction
from dynamics_base import DynamicsBase

# Import animator
from animate import animate_cart_pole

# @title  Dynamics Instance

from symbolic_dynamics import symbolic_cart_pole

class DynamicsCartPole(DynamicsBase):
  def __init__(self, parameters) -> None:
    self.par = parameters
    self.fd,self.Ad,self.Bd = symbolic_cart_pole()


dt = 0.005
N = 1000

Q = np.identity(4) * 0.0
Qf = np.diag([10.0, 0.0, 100.0, 30.0])
R = np.identity(1) * 0.001

dyn = DynamicsCartPole(np.array([0.5, 0.2, 9.8, 0.3]))
costs = CostFunction(Q, R, Qf)
init_state = np.zeros(4)
initial_guess = [np.ones(1)*0.1 for _ in range(N)]

ilqr = ILQR(dyn, costs, init_state, dt, initial_guess)
ilqr.plot()

ilqr.run(50)
ilqr.plot()


print("final states:{}".format(ilqr.x_traj[-1]) )

anim = animate_cart_pole(np.array(ilqr.x_traj),np.array(ilqr.u_traj),dt,ilqr.dyn.par)
anim