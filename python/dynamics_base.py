# @title Dynamics Base
import numpy as np
from typing import Callable

class DynamicsBase:
  """
  Base for system dynamics
  """
  par: np.array
  Ad: Callable[[np.array, np.array, float, np.array], np.array]
  Bd: Callable[[np.array, np.array, float, np.array], np.array]
  fd: Callable[[np.array, np.array, float, np.array], np.array]

  def __init__(self) -> None:
    """
    Initialize Ad, Bd, fd, par.
    """
    pass

  def DiscreteTimeLinearize(self, states:np.array, inputs:np.array, dt:float) -> tuple:
    """
    Linear system dynamics in discrete time. Sys eqn: x(k+1) = Ad(k)x(k) + Bd(k)u(k)

    Parameters
    ----------
    states: 1d array, states at current time
    inputs: 1d array, inputs at current time

    Returns:
    ----------
    Ad: 2d array
    Bd: 1d array
    """
    return self.Ad(states, inputs, dt, self.par), self.Bd(states, inputs, dt, self.par).flatten()

  def CalcXNext(self, states:np.array, inputs:np.array, dt:float) -> np.array:
    """
    Discrete time nonlinear dynamics, returns x(k+1) given x(k), u(k), dt.

    Parameters
    ----------
    states: 1d array, states at current time
    inputs: 1d array, inputs at current time
    dt: time delta

    Returns:
    ----------
    next states: 1d array, states at k+1
    """
    return self.fd(states, inputs, dt, self.par).flatten()
