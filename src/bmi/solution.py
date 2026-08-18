"""
BMI Module - Solution definition
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import signal

@dataclass
class BMISolution:
    P: np.ndarray | None = None
    Pf: np.ndarray | None = None
    Af: np.ndarray | None = None
    Bf: np.ndarray | None = None
    Cf: np.ndarray | None = None
    Df: np.ndarray | None = None
    mu: float = float("inf") 
    feasible: bool = False

    def to_statespace(self,dt:float=1.0) -> signal.StateSpace:
        """Wrap filter matrices into a scipy StateSpace for analysis."""
        return signal.StateSpace(self.Af, self.Bf, self.Cf, self.Df,dt=dt)