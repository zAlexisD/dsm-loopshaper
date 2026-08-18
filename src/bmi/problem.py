"""
BMI Module - Problem definition
"""
import numpy as np
from scipy import signal
import cvxpy as cp

#NOTE: Block matrixes are here defined for an IIR feedback filter in a standard DSM system, change if needed.

class BMIProblem:
    def __init__(
        self,
        n_filter: int,
        n_input: int = 1,    # filter inputs
        n_output: int = 1,   # filter outputs
        gamma: float = 1.5
        ):

        self.n_f   = n_filter
        self.n_i   = n_input
        self.n_o   = n_output
        self.gamma = gamma
        # Output filter order set to 4
        self.outFilterOrder = 4
        # Combined system order
        self.n = self.n_f + self.outFilterOrder

        self._output_filter = self._build_output_filter()

    # ── Output filter (internal, fixed for now) ────────────────────

    def _build_output_filter(self) -> signal.StateSpace:
        """
        Internally defined digital output filter.
        Hardcoded for now; later could accept a user-supplied StateSpace.
        """
        # Simple lowpass Butterworth filter, order 4, cutoff at pi/32
        self.outFilterOrder = 4
        cutoff = np.pi / 32
        samplingRate = 1
        b, a = signal.butter(self.outFilterOrder, cutoff / np.pi, btype='low', analog=False)
        # Convert to state space reprezentation
        A_w, B_w, C_w, D_w = signal.tf2ss(b, a)
        return signal.StateSpace(A_w, B_w, C_w, D_w, dt=samplingRate)

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
            [A_f, B_f @ C_w],
            [np.zeros((self.outFilterOrder,self.n_f)), A_w]
        ])
        B = np.block([
            [B_f @ D_w],
            [B_w]
        ])
        C = np.block([
            [C_f, C_w]
        ])
        D = D_w

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
            [A_f, B_f @ C_w],
            [np.zeros((self.outFilterOrder,self.n_f)), A_w]
        ])
        B = cp.vstack([
            B_f @ D_w,
            B_w
        ])
        C = cp.hstack([
            C_f, C_w
        ])
        D = D_w

        return A, B, C, D
    

    def build_LMI1_cvxpy(self, P, A, B):
        """
        Lyapunov LMI for H2 norm — must be > 0.

        [[P, P @ A,  P @ B],
        [A.T @ P, P, 0],
        [B.T @ P, 0, 1]]

        P is a CVXPY variable or fixed numpy array.
        A, B are fixed numpy arrays (closed-loop).
        """
        n, n_i = self.n, self.n_i

        return cp.bmat([
            [P, P @ A, P @ B],
            [A.T @ P, P, np.zeros((n, n_i))],
            [B.T @ P, np.zeros((n_i, n)), np.eye(n_i)],
        ])
    
    def build_LMI2_cvxpy(self, P, mu, C, D):
        """
        H2 norm upper bound — must be > 0.

        [[mu, C, D],
        [C.T, P, 0],
        [D.T, 0, 1]]

        mu is a CVXPY variable or scalar.
        C, D are fixed numpy arrays (closed-loop).
        """
        n, n_i, n_o = self.n, self.n_i, self.n_o

        return cp.bmat([
            [mu * np.eye(n_o),  C, D],
            [C.T, P, np.zeros((n, n_i))],
            [D.T, np.zeros((n_i, n)), np.eye(n_i)],
        ])
    
    def build_LMI3_cvxpy(self, P_f, A_f, B_f, C_f, D_f, gamma):
        """
        Bounded real lemma on feedback filter F(z) for H-inifinity constraint < 0.

        [[-Pf, Pf @ Af, Pf @ Bf, 0],
        [Af.T @ Pf, -Pf, 0, Cf.T],
        [Bf.T @ Pf, 0, -gamma^2, Df.T],
        [0, Cf, Df, -1]]

        Pf is a CVXPY variable or fixed numpy array.
        Af, Bf, Cf, Df may be CVXPY expressions (K-step) or numpy (P-step).
        """
        n_f, n_i, n_o = self.n_f, self.n_i, self.n_o

        return cp.bmat([
            [-P_f,           P_f @ A_f,            P_f @ B_f,              np.zeros((n_f, n_o))],
            [A_f.T @ P_f,   -P_f,                  np.zeros((n_f, n_i)),   C_f.T               ],
            [B_f.T @ P_f,    np.zeros((n_i, n_f)), -gamma**2 * np.eye(n_i), D_f.T              ],
            [np.zeros((n_o, n_f)), C_f,             D_f,                    -np.eye(n_o)        ],
        ])
    

    def build_LMI1(self, P, A, B):
        """Pure numpy version for inspection and post-hoc validation."""
        n, n_i = self.n, self.n_i

        return np.block([
            [P, P @ A, P @ B],
            [A.T @ P, P, np.zeros((n, n_i))],
            [B.T @ P, np.zeros((n_i, n)), np.eye(n_i)],
        ])
    
    def build_LMI2(self, P, mu, C, D):
        """Pure numpy version for inspection and post-hoc validation."""
        n, n_i, n_o = self.n, self.n_i, self.n_o

        return np.block([
            [mu * np.eye(n_o),  C, D],
            [C.T, P, np.zeros((n, n_i))],
            [D.T, np.zeros((n_i, n)), np.eye(n_i)],
        ])
    
    def build_LMI3(self, P_f, A_f, B_f, C_f, D_f, gamma):
        """
        Pure numpy version for inspection and post-hoc validation.
        Constraint on H_inf
        """
        n_f, n_i, n_o = self.n_f, self.n_i, self.n_o

        return np.block([
            [-P_f,           P_f @ A_f,            P_f @ B_f,              np.zeros((n_f, n_o))],
            [A_f.T @ P_f,   -P_f,                  np.zeros((n_f, n_i)),   C_f.T               ],
            [B_f.T @ P_f,    np.zeros((n_i, n_f)), -gamma**2 * np.eye(n_i), D_f.T              ],
            [np.zeros((n_o, n_f)), C_f,             D_f,                    -np.eye(n_o)        ],
        ])