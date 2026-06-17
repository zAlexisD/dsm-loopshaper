"""
BMI Module - Problem definition
"""
import numpy as np
from scipy import signal

class BMIProblem:
    def __init__(
        self,
        n_filter: int,
        n_input: int = 1,    # filter inputs
        n_output: int = 1   # filter outputs
        ):

        self.n_f = n_filter
        self.n_i = n_input
        self.n_o = n_output
        # Output filter order set to 4
        self.outFilterOrder = 4
        # Combined system order
        self.n = self.n_f + self.outFilterOrder

        self._output_filter = self._build_output_filter()

    # ── Output filter (internal, fixed for now) ────────────────────

    def _build_output_filter(self) -> signal.StateSpace:
        """
        Internally defined output filter.
        Hardcoded for now; later could accept a user-supplied StateSpace.
        """
        # Simple lowpass Butterworth filter, order 4, cutoff at pi/32
        self.outFilterOrder = 4
        cutoff = np.pi / 32
        b, a = signal.butter(self.outFilterOrder, cutoff / np.pi, btype='low')
        # Convert to state space reprezentation
        A_w, B_w, C_w, D_w = signal.tf2ss(b, a)
        return signal.StateSpace(A_w, B_w, C_w, D_w)

    # ── Dimension validation ───────────────────────────────────────

    @staticmethod
    def _raise_if_mismatch(expected: dict, actual: dict, context: str) -> None:
        """
        Format output error for dimension validation
        """
        mismatches = [
            f"{name}: expected {expected[name]}, got {actual[name]}"
            for name in expected
            if actual[name] != expected[name]
        ]
        if mismatches:
            raise ValueError(f"Dimension mismatch in {context}:\n  " + "\n  ".join(mismatches))

    def _check_filter_dims(self, A_f, B_f, C_f, D_f) -> None:
        expected = {
            "A_f": (self.n_f, self.n_f),
            "B_f": (self.n_f, self.n_i),
            "C_f": (self.n_o, self.n_f),
            "D_f": (self.n_o, self.n_i),
        }
        actual = {"A_f": A_f.shape, "B_f": B_f.shape, "C_f": C_f.shape, "D_f": D_f.shape}
        self._raise_if_mismatch(expected, actual, context="filter matrices")

    def _check_bmi_dims(self, P, A, B, C, D) -> None:
        expected = {
            "P": (self.n, self.n),
            "A": (self.n, self.n),
            "B": (self.n, self.n_i),
            "C": (self.n_o, self.n),
            "D": (self.n_o, self.n_i),
        }
        actual = {"P": P.shape, "A": A.shape, "B": B.shape, "C": C.shape, "D": D.shape}
        self._raise_if_mismatch(expected, actual, context="combined system / BMI")

    # ── Public API : BMI Involved Matrixes  ──────────────────────────
    def build_system(self, A_f, B_f, C_f, D_f):
        """
        Combined system state space realization building
        """
        # Ensure input arrays have right dimension
        self._check_filter_dims(A_f, B_f, C_f, D_f)
        
        # Extract state space output filter
        A_w = self._output_filter.A
        B_w = self._output_filter.B
        C_w = self._output_filter.C
        D_w = self._output_filter.D
        
        # Assemble block matrixes
        A = np.block([
            [A_f, np.zeros(self.n_f,self.outFilterOrder)],
            [B_w @ C_f, A_w]
        ])
        B = np.block([
            [B_f],
            [B_w @ (np.eye(self.n_o) + D_f)]
        ])
        C = np.block([
            [D_w @ C_f, C_w]
        ])
        D = D_w @ (np.eye(self.n_o) + D_f)

        return A, B, C, D

    
    def build_BMI(self, P, A, B, C, D, gamma):
        """
        Real bounded lemma BMI matrix building
        """
        # Ensure arrays have right dimensions
        self._check_bmi_dims(P, A, B, C, D)
        
        # Define the BMI matrix
        BMI_mat = np.block([
            [A.T @ P @ A - P, A.T @ P @ B, C.T],
            [B.T @ P @ A, B.T @ P @ B - gamma * np.eye(self.n_i), D.T],
            [C, D, - gamma * np.eye(self.n_i)]
        ])
        return BMI_mat