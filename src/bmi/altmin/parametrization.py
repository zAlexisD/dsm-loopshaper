from typing import Protocol
import cvxpy as cp
import numpy as np

from zpk.parametrization import ZeroParams,PoleParams

class KStepParametrization(Protocol):
    """
    Defines how to construct (Af, Bf, Cf, Df) as CVXPY expressions
    for the K-step. Allows different parametrizations (free matrices,
    pole-coefficient-based, etc.) without changing BMIKStep itself.
    """
    def variables(self) -> list[cp.Variable]:
        """All CVXPY variables in this parametrization."""
        ...

    def build(self) -> tuple:
        """Return (Af, Bf, Cf, Df) as CVXPY-compatible expressions."""
        ...

    def extract(self) -> tuple:
        """Return (Af, Bf, Cf, Df) as numpy arrays after solve."""
        ...


class FreeFilterParametrization:
    """
    Default: all four matrices are independent CVXPY variables.
    Used by BMIAltMinProblem.
    """
    def __init__(self, filter_order: int, n_inputs: int, n_outputs: int):
        self.p = filter_order
        self.m_in = n_inputs
        self.m_out = n_outputs
        self._Af = None
        self._Bf = None
        self._Cf = None
        self._Df = None

    def variables(self, cp):
        self._Af = cp.Variable((self.p, self.p))
        self._Bf = cp.Variable((self.p, self.m_in))
        self._Cf = cp.Variable((self.m_out, self.p))
        self._Df = cp.Variable((self.m_out, self.m_in))
        return [self._Af, self._Bf, self._Cf, self._Df]

    def build(self):
        return self._Af, self._Bf, self._Cf, self._Df

    def extract(self):
        return self._Af.value, self._Bf.value, self._Cf.value, self._Df.value


class PoleCoeffParametrization:
    """
    K-step parametrization for the ZPK zero-fixing step.
    Af and Cf are both linear in the pole coefficient vector a_var.
    Bf and Df are fixed numpy arrays.
    """
    def __init__(self, zero_params: ZeroParams, pole_params: PoleParams):
        self.zero_params = zero_params
        self.pole_params = pole_params
        n = len(pole_params.coeffs)
        self._a_var = None

        # Precompute structural matrices
        self._n = n
        self._Bf = pole_params.to_Bf()
        b = zero_params.coeffs
        self._Df = np.array([[b[0]]])
        b_pad = np.concatenate([b[1:], np.zeros(max(0, n - (len(b) - 1)))])[:n]
        self._b_pad = b_pad
        self._Df_val = float(b[0])

        # Companion matrix structural part (shift register rows)
        self._Af_shift = np.zeros((n, n))
        if n > 1:
            self._Af_shift[:-1, 1:] = np.eye(n - 1)
        self._e_last = np.zeros((n, 1))
        self._e_last[-1, 0] = 1.0
        self._Rev = np.eye(n)[::-1]   # reversal matrix

    def variables(self, cp):
        self._a_var = cp.Variable(self._n)
        return [self._a_var]

    def build(self):
        import cvxpy as cp
        a = self._a_var
        # Af = shift part + last row linear in a
        Af = self._Af_shift - self._e_last @ cp.reshape(self._Rev @ a, (1, self._n))
        # Cf = b_pad - Df_val * a
        Cf = cp.reshape(self._b_pad - self._Df_val * a, (1, self._n))
        return Af, self._Bf, Cf, self._Df

    def extract(self):
        a = self._a_var.value
        Af = self._Af_shift - self._e_last @ (self._Rev @ a).reshape(1, self._n)
        Cf = (self._b_pad - self._Df_val * a).reshape(1, self._n)
        return Af, self._Bf, Cf, self._Df