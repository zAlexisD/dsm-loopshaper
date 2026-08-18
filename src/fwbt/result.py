"""
FWBT Module: FWBTResult — holds reduced SS, HSVs, error bound
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class FWBTResult:
    # Reduced-order state-space
    A_r: np.ndarray
    B_r: np.ndarray
    C_r: np.ndarray
    D_r: np.ndarray

    # Full HSV array (length = order of full-order input)
    hsv: np.ndarray

    # Truncation metadata
    truncation_index: int
    hinf_error_bound: float
    is_stable: bool
    discrete: bool

    @property
    def order(self) -> int:
        return self.A_r.shape[0]

    def to_statespace(self) -> signal.StateSpace:
        return signal.StateSpace(
            self.A_r, self.B_r, self.C_r, self.D_r,
            dt=1 if self.discrete else None
        )

    def to_zpk(self):
        ss = self.to_statespace()
        return ss.to_zpk()

    def hsv_gap(self) -> float:
        """Ratio of first truncated HSV to last retained HSV — a truncation quality indicator."""
        r = self.truncation_index
        if r >= len(self.hsv):
            return np.inf
        return self.hsv[r] / self.hsv[r - 1]

    def summary(self) -> str:
        return (
            f"FWBTResult | order: {self.order} | "
            f"H∞ bound: {self.hinf_error_bound:.4e} | "
            f"HSV gap: {self.hsv_gap():.4f} | "
            f"stable: {self.is_stable}"
        )