"""
BMI Module - Problem definition
"""
import numpy as np
from scipy import signal
import cvxpy as cp

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
        Combined system state space realization for plain numpy usage.
        All inputs must be numpy arrays.
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
            [A_f, np.zeros((self.n_f,self.outFilterOrder))],
            [B_w @ C_f, A_w]
        ])
        B = np.block([
            [B_f],
            [B_w @ (np.eye(self.n_i) + D_f)]
        ])
        C = np.block([
            [D_w @ C_f, C_w]
        ])
        D = D_w @ (np.eye(self.n_i) + D_f)

        return A, B, C, D
    
    def build_system_cvxpy(self, A_f, B_f, C_f, D_f):
        """
        Combined system state space realization for CVXPY usage.
        A_f, B_f, C_f, D_f may be CVXPY variables.
        Returns CVXPY expressions for A, B, C, D.
        """
        self._check_filter_dims(A_f, B_f, C_f, D_f)

        A_w = self._output_filter.A
        B_w = self._output_filter.B
        C_w = self._output_filter.C
        D_w = self._output_filter.D

        A = cp.bmat([
            [A_f,        np.zeros((self.n_f, self.outFilterOrder))],
            [B_w @ C_f,  A_w]
        ])
        B = cp.vstack([
            B_f,
            B_w @ (np.eye(self.n_i) + D_f)
        ])
        C = cp.hstack([
            D_w @ C_f,  C_w
        ])
        D = D_w @ (np.eye(self.n_i) + D_f)

        return A, B, C, D

    def build_BMI(self, P, A, B, C, D, gamma):
        """
        Quadratic Bounded real lemma BMI matrix for plain numpy usage
        All inputs must be numpy arrays.
        """
        self._check_bmi_dims(P, A, B, C, D)

        BMI_mat = np.block([
            [A.T @ P @ A - P, A.T @ P @ B, C.T],
            [B.T @ P @ A, B.T @ P @ B - gamma * np.eye(self.n_i), D.T],
            [C, D, -gamma * np.eye(self.n_o)]
        ])
        return BMI_mat
    
    def build_BMI_cvxpy_P_step(self, P, A, B, C, D, gamma):
        """
        Schur complement form of the discrete bounded real lemma for the P-step.
        P is a CVXPY variable; A, B, C, D are fixed numpy arrays.
        Linear in P.

        Equivalent to: ||H||_inf < gamma, i.e.

            [A'PA - P, A'PB, C']
            [B'PA, B'PB, D']       << 0

            [C, D, -γI]

        lifted via Schur complement to avoid quadratic terms.
        """
        self._check_bmi_dims(P, A, B, C, D)

        #TODO Check expression
        return cp.bmat([
            [-P, np.zeros((self.n, self.n_o)), P @ A, P @ B],
            [np.zeros((self.n_o, self.n)), -gamma*np.eye(self.n_o), C, D],
            [A.T @ P, C.T, -P,   np.zeros((self.n, self.n_i))],
            [B.T @ P, D.T, np.zeros((self.n_i, self.n)), -gamma*np.eye(self.n_i)]])
    
    def build_BMI_cvxpy_K_step(self, P, A, B, C, D, gamma):
        """
        Schur complement form of the discrete bounded real lemma for the K-step.
        P is a fixed numpy array; A, B, C, D are CVXPY expressions linear in
        the filter variables. Every block is linear in the decision variables.

        Equivalent to: ||H||_inf < gamma, i.e.
            [ A'PA - P   A'PB   C' ]
            [ B'PA       B'PB   D' ] << 0
            [ C          D      -γI]
        lifted via Schur complement to avoid quadratic terms.
        """
        self._check_bmi_dims(P, A, B, C, D)

        # P @ A and P @ B are linear in filter variables since P is fixed numpy
        PA = P @ A
        PB = P @ B

        #TODO: check this expression
        return cp.bmat([
            [-P, np.zeros((self.n, self.n_o)), PA, PB],
            [np.zeros((self.n_o, self.n)), -gamma*np.eye(self.n_o), C, D],
            [A.T @ P, C.T, -P,   np.zeros((self.n, self.n_i))],
            [B.T @ P, D.T, np.zeros((self.n_i, self.n)), -gamma*np.eye(self.n_i)]])