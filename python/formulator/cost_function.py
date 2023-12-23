from formulator.cost_base import CostFunctionBase

import numpy as np
from typing import Callable

class CostTrackTarget(CostFunctionBase):
  """
  Consider cost function as tracking a target state with quadratic cost:
  l(x, u) = (x-x_target)^T * Q * (x-x_target) + u.T * R * u
  lf(xN) = (xN-x_target)^T * Qf * (xN-x_target)
  """
  
  _Q: np.array
  _Qf: np.array
  _R: np.array
  _target_x: np.array

  def __init__(self, Q: np.array, R:np.array, Qf:np.array, target_x) -> None:
    self._Q = Q
    self._R = R
    self._Qf = Qf
    self._target_x = target_x

    self.l = lambda x, u: (x - self._target_x).T @ self._Q @ (x - self._target_x)\
                         + u.T @ self._R @ u
    self.lf = lambda xN: (xN - self._target_x).T @ self._Qf @ (xN - self._target_x)

    self.l_x = lambda x_ref, u_ref: 2*self._Q @ (x_ref - self._target_x)
    self.l_u = lambda x_ref, u_ref: 2*self._R @ u_ref
    self.l_xx = lambda x_ref, u_ref: 2 * self._Q
    self.l_ux = lambda x_ref, u_ref: np.zeros((u_ref.shape[0], x_ref.shape[0]))
    self.l_uu = lambda x_ref, u_ref: 2 * self._R

    self.lf_x = lambda x_ref: 2*self._Qf @ (x_ref - self._target_x)
    self.lf_xx = lambda x_ref: 2*self._Qf

