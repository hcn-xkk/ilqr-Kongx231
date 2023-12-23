# @title Formulate cost function and jacobians/hessians
import numpy as np
from typing import Callable

class CostFunction:
  """
  Consider cost function being:
    min_u sum( l(xn, un) ) + lf(xN)
  """
  l: Callable[[np.array, np.array], float] # stage cost l(x,u)
  lf: Callable[[np.array], float] # terminal cost lf(xN)

  l_x: Callable[[np.array, np.array], np.array] # alpha_l / alpha_x 1d array, shape (Nx,)
  l_u: Callable[[np.array, np.array], np.array] # alpha_l / alpha_u, 1d array, shape (Nu,)
  l_xx: Callable[[np.array, np.array], np.array] # alpha^2_l / alpha_x^2 2d array, shape (Nx, Nx)
  l_ux: Callable[[np.array, np.array], np.array] # alpha^2_l / alpha_x^2 2d array, shape (Nu, Nx)
  l_uu: Callable[[np.array, np.array], np.array] # alpha^2_l / alpha_x^2 2d array, shape (Nu, Nu)

  lf_x: Callable[[np.array, np.array], np.array] # alpha_lf / alpha_x 1d array, shape (Nx,)
  lf_xx: Callable[[np.array, np.array], np.array] # alpha^2_lf / alpha_x^2 1d array, shape (Nx, Nx)

  _Q = np.identity(4) * 0.0
  _Qf = np.diag([10.0, 0.0, 100.0, 50.0])
  _R = np.identity(1) * 0.001
  _target_x = np.array([0.0,0.0,0.0,np.pi])

  def __init__(self, Q: np.array, R:np.array, Qf:np.array) -> None:
    self._Q = Q
    self._R = R
    self._Qf = Qf

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

