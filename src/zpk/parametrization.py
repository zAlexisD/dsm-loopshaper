from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PoleParams:
    """
    Denominator coefficients a = [a_1, ..., a_n] of the monic polynomial
    A(z) = 1 + a_1*z^-1 + ... + a_n*z^-n.
    """
    coeffs: np.ndarray

    def to_Af(self) -> np.ndarray:
        """Companion matrix — entries are linear in coeffs."""
        n_p = len(self.coeffs)
        Af  = np.zeros((n_p, n_p))
        if n_p > 1:
            Af[:-1, 1:] = np.eye(n_p - 1)
        Af[-1, :] = -self.coeffs[::-1]
        return Af

    def to_Bf(self) -> np.ndarray:
        """Structurally fixed input matrix in controllable canonical form."""
        n_p = len(self.coeffs)
        Bf  = np.zeros((n_p, 1))
        Bf[-1, 0] = 1.0
        return Bf

    def to_poles(self) -> np.ndarray:
        """Roots of A(z) for inspection."""
        return np.roots(np.concatenate([[1.0], self.coeffs]))

    def is_stable(self) -> bool:
        return np.all(np.abs(self.to_poles()) < 1.0)


@dataclass
class ZeroParams:
    """
    Numerator coefficients b = [b_0, b_1, ..., b_m] of
    B(z) = b_0 + b_1*z^-1 + ... + b_m*z^-m.
    """
    coeffs: np.ndarray     

    def to_Cf(self, pole_params: PoleParams) -> np.ndarray:
        """
        Cf depends on both numerator and denominator coefficients:
            Cf_i = b_i - b_0 * a_i
        """
        a = pole_params.coeffs
        b = self.coeffs
        b[0] = 1.0                # enforce Df = b_0 fixed to 1 for strict causality constraint
        Df_val = b[0]  

        n_p = len(a)
        n_z = len(b) - 1

        b_pad  = np.concatenate([b[1:], np.zeros(max(0, n_p - n_z))])[:n_p]
        return (b_pad - Df_val * a).reshape(1, -1)

    def to_Df(self) -> np.ndarray:
        # enforce Df = b_0 fixed to 1 for strict causality constraint
        self.coeffs[0] = 1.0
        return np.array([[self.coeffs[0]]])

    def to_zeros(self) -> np.ndarray:
        """Roots of B(z) for inspection."""
        return np.roots(self.coeffs)

    @staticmethod
    def from_zeros(zeros: np.ndarray, gain: float = 1.0) -> ZeroParams:
        """Construct from zero locations via np.poly."""
        coeffs = gain * np.poly(zeros)
        return ZeroParams(coeffs=coeffs)


@dataclass
class ZPKParams:
    """
    Full parametrization of the feedback filter in coefficient space.
    Poles and zeros are represented via their polynomial coefficients,
    which map linearly to state-space matrices.
    """
    poles: PoleParams
    zeros: ZeroParams

    @property
    def n_poles(self) -> int:
        return len(self.poles.coeffs)

    @property
    def n_zeros(self) -> int:
        return len(self.zeros.coeffs) - 1

    @property
    def n_filter(self) -> int:
        return max(self.n_poles, self.n_zeros)

    @property
    def get_poles(self):
        return self.poles.to_poles()

    @property
    def get_zeros(self):
        return self.zeros.to_zeros()

    @property
    def n_filter(self) -> int:
        return max(len(self.poles.coeffs), len(self.zeros.coeffs) - 1)

    def is_proper(self) -> bool:
        return self.n_poles >= self.n_zeros

    def to_filter_matrices(self):
        """Return (Af, Bf, Cf, Df) for use in BMIProblem.build_system."""
        Af = self.poles.to_Af()
        Bf = self.poles.to_Bf()
        Cf = self.zeros.to_Cf(self.poles)
        Df = self.zeros.to_Df()
        return Af, Bf, Cf, Df