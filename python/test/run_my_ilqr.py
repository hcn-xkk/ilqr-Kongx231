import sys
sys.path.insert(0, '..')

from solver.my_ilqr import ILQR
from formulator.cost_function import CostTrackTarget
from formulator.dynamics_base import DynamicsBase
from common.animate import animate_cart_pole
from common.symbolic_dynamics import symbolic_cart_pole

import numpy as np
import matplotlib.animation as animation


# @title  Dynamics Instance
class DynamicsCartPole(DynamicsBase):
  def __init__(self, parameters) -> None:
    self.par = parameters
    self.fd,self.Ad,self.Bd = symbolic_cart_pole()


example_id = 1

if example_id == 1:

  # example 1
  dt = 0.005
  N = 1000

  Q = np.identity(4) * 0.0
  Qf = np.diag([10.0, 0.0, 100.0, 30.0])
  R = np.identity(1) * 0.001
  target_state = np.array([0.0,0.0,0.0,np.pi])

  dyn = DynamicsCartPole(np.array([0.5, 0.2, 9.8, 0.3]))
  costs = CostTrackTarget(Q, R, Qf, target_state)
  init_state = np.zeros(4)
  initial_guess = [np.ones(1)*0.1 for _ in range(N)]

elif example_id == 2:
  dt = 0.005
  N = 1000

  Q = np.identity(4) * 0.0
  Qf = np.diag([10.0, 0.0, 100.0, 30.0])
  R = np.identity(1) * 0.001
  target_state = np.array([0.0,0.0,0.0,np.pi])

  dyn = DynamicsCartPole(np.array([0.1, 0.5, 9.8, 0.3]))
  costs = CostTrackTarget(Q, R, Qf, target_state)
  init_state = np.zeros(4)
  initial_guess = [np.ones(1)*0.01 for _ in range(N)]

else:
  print('Please provide example id.')
  sys.exit()


ilqr = ILQR(dyn, costs, init_state, dt, initial_guess)
ilqr.plot()

ilqr.run(50)
ilqr.plot()


print("final states:{}".format(ilqr.x_traj[-1]) )

anim = animate_cart_pole(np.array(ilqr.x_traj),np.array(ilqr.u_traj),dt,ilqr.dyn.par)
anim

