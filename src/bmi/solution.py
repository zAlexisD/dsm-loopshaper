"""
BMI Module - Solution definition
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import signal

@dataclass
class BMISolution:
    P: np.ndarray
    Af: np.ndarray
    Bf: np.ndarray
    Cf: np.ndarray
    Df: np.ndarray
    gamma: float
    feasible: bool = field(default_factory=bool)

    def to_statespace(self) -> signal.StateSpace:
        """Wrap filter matrices into a scipy StateSpace for analysis."""
        return signal.StateSpace(self.A_f, self.B_f, self.C_f, self.D_f)