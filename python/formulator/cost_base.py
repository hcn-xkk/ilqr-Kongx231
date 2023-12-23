# @title Formulate cost function and jacobians/hessians
import numpy as np
from typing import Callable

class CostFunctionBase:
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

  def __init__(self) -> None:
    """
    Initialize all the members here
    """
    pass

