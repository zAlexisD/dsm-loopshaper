"""
FWBT Module: FWBTResult — holds reduced SS, HSVs, error bound
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import ZerosPolesGain,StateSpace

@dataclass
class FWBTResult:
    # Reduced order state space
    A_r: np.ndarray
    B_r: np.ndarray
    C_r: np.ndarray
    D_r: np.ndarray

    # Diagnostics
    hsv: np.ndarray           # full Hankel singular values (length n)
    truncation_index: int     # r — number of states kept
    hinf_error_bound: float   # 2 * sum(sigma[r:])
    is_stable: bool

    def to_zpk(self) -> ZerosPolesGain: ...
    def to_statespace(self) -> StateSpace: ...