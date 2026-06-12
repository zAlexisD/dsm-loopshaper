"""
BMI Module - Problem definition
"""

import numpy as np
from scipy import signal
import cvxpy as cp

from bmi.solution import BMISolution

class BMIProblem:
    def __init__(
        self,
        n_filter: int,
        n_input: int = 1,    # filter inputs
        n_output: int = 1,   # filter outputs
        gamma: float = 1.0,
    ):
        self.n_filter = n_filter
        self.n_input  = n_input
        self.n_output = n_output
        self.gamma    = gamma

        self._output_filter = self._build_output_filter()
        self._init_variables()
        self._build_system()

    # ── Output filter (internal, fixed for now) ────────────────────

    def _build_output_filter(self) -> signal.StateSpace:
        """
        Internally defined output filter.
        Hardcoded for now; later could accept a user-supplied StateSpace.
        """
        # Simple lowpass Butterworth filter, order 4, cutoff at pi/32
        order  = 4
        cutoff = np.pi / 32
        b, a   = signal.butter(order, cutoff / np.pi, btype='low')
        # Convert to state space reprezentation
        A_w, B_w, C_w, D_w = signal.tf2ss(b, a)
        return signal.StateSpace(A_w, B_w, C_w, D_w)

    # ── CVXPY variable initialization ──────────────────────────────

    def _init_variables(self) -> None:
        n_f = self.n_filter
        n_i = self.n_input
        n_o = self.n_output

        self.A_f = cp.Variable((n_f, n_f))
        self.B_f = cp.Variable((n_f, n_i))
        self.C_f = cp.Variable((n_o, n_f))
        self.D_f = cp.Variable((n_o, n_i))
        # Lyapunov matrix P will be initialized once system state space defined

    # ── Define BMI Involved Matrixes  ──────────────────────────────
    def _build_system(self):
        # First get state space output filter
        A_w, B_w, C_w, D_w = self._build_output_filter()
        
        # Assemble block matrixes
        self.A = np.block([
            [self.A_f, np.zeros(self.n_f,A_w.shape[1])],
            [B_w @ self.C_f, A_w]
        ])
        self.B = np.block([
            [self.B_f],
            [B_w @ (np.eye(self.n_u) + self.D_f)]
        ])
        self.C = np.block([
            [D_w @ self.C_f, C_w]
        ])
        self.D = D_w @ (np.eye(self.n_u) + self.D_f)

        # Initiate Lyapunov Matrix P as variable
        n = self.A.shape[0]
        self.P   = cp.Variable((n, n),symmetric=True)
    
    def build_BMI(self):
        # Define the BMI matrix
        bmi = np.block([
            [self.A.T @ self.P @ self.A - self.P, self.A.T @ self.P @ self. B, self.C.T],
            [self.B.T @ self.P @ self.A, self.B.T @ self.P @ self.B 
            - self.gamma * np.eye(self.D.T.shape[0]), self.D.T],
            [self.C, self.D, -self.gamma * np.eye(self.D.T.shape[0])]
        ])
        return bmi

    @abstract
    def solve(self) -> BMISolution:
        ...