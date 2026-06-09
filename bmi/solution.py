"""
BMI Module - Solution definition
"""
from dataclasses import dataclass
import numpy as np
from scipy import signal

@dataclass
class BMISolution:
    P: np.ndarray
    A_f: np.ndarray
    B_f: np.ndarray
    C_f: np.ndarray
    D_f: np.ndarray
    gamma: float
    feasible: bool

    def to_statespace(self) -> signal.StateSpace:
        """Wrap filter matrices into a scipy StateSpace for analysis."""
        return signal.StateSpace(self.A_f, self.B_f, self.C_f, self.D_f)