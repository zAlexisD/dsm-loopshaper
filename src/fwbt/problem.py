"""
FWBT Module: FWBTProblem — user-facing entry point
"""
import warnings
import numpy as np
from scipy import signal

from fwbt.core import (
    _build_augmented_system,
    _solve_gramians,
    _balanced_transform,
    _truncate,
    _hinf_error_bound,
    _check_stability)
from fwbt.result import FWBTResult


class FWBTProblem:
    """
    Frequency-Weighted Balanced Truncation.

    Parameters
    ----------
    ss_full : scipy.signal.StateSpace
        Full-order system to reduce. Must be stable.
    order_target : int or None
        Target reduced order. If None, auto-selected from largest HSV gap.
    W_i : tuple (A, B, C, D) or None
        Input frequency weight state-space. None = identity.
    W_o : tuple (A, B, C, D) or None
        Output frequency weight state-space. None = identity.
    discrete : bool
        Whether the system is discrete-time.
    reg : float
        Regularization added to gramians before Cholesky.
    """

    def __init__(self,
                 ss_full: signal.StateSpace,
                 order_target: int = None,
                 W_i=None,
                 W_o=None,
                 discrete: bool = True,
                 reg: float = 1e-10,
                 verbose: bool=False):

        self.ss_full = ss_full
        self.order_target = order_target
        self.W_i = W_i
        self.W_o = W_o
        self.discrete = discrete
        self.reg = reg
        self.verbose = verbose

        self._hsv = None  # cached after first gramian solve

    @property
    def is_ready(self) -> bool:
        A = self.ss_full.A
        return _check_stability(A, self.discrete)

    def _auto_select_order(self, sigma) -> int:
        """Select r at the largest relative gap in the HSV spectrum."""
        if len(sigma) < 2:
            return 1
        gaps = sigma[:-1] / np.where(sigma[1:] > 0, sigma[1:], np.inf)
        r = int(np.argmax(gaps)) + 1
        warnings.warn(
            f"order_target not specified — auto-selected r={r} "
            f"from largest HSV gap ({gaps[r-1]:.3f}). "
            f"Inspect hsv_plot() to verify.",
            UserWarning
        )
        return r

    def hsv_preview(self):
        """
        Compute and cache HSVs without solving the full reduction.
        Useful for inspecting the spectrum before committing to an order.
        Returns sigma array.
        """
        A, B, C, D = (
            self.ss_full.A, self.ss_full.B,
            self.ss_full.C, self.ss_full.D
        )
        A_aug, B_aug, C_aug, D_aug, n_plant = _build_augmented_system(
            A, B, C, D, self.W_i, self.W_o
        )
        P, Q, _, _ = _solve_gramians(
            A_aug, B_aug, C_aug, D_aug, n_plant, self.discrete
        )
        _, _, sigma = _balanced_transform(P, Q, self.reg, verbose=self.verbose)
        self._hsv = sigma
        return sigma

    def solve(self, r: int = None) -> FWBTResult:
        """
        Run the full FWBT pipeline.

        Parameters
        ----------
        r : int or None
            Overrides order_target for this call only.
        """
        if not self.is_ready:
            raise ValueError(
                "Full-order system is not stable. "
                "Lyapunov solves require a stable A matrix."
            )

        A, B, C, D = (
            self.ss_full.A, self.ss_full.B,
            self.ss_full.C, self.ss_full.D
        )

        # Build augmented system
        A_aug, B_aug, C_aug, D_aug, n_plant = _build_augmented_system(
            A, B, C, D, self.W_i, self.W_o
        )

        # Solve gramians
        P, Q, _, _ = _solve_gramians(
            A_aug, B_aug, C_aug, D_aug, n_plant, self.discrete
        )

        # Balancing transform + HSVs
        T, T_inv, sigma = _balanced_transform(P, Q, self.reg, verbose=self.verbose)
        self._hsv = sigma

        # Choose order
        r_use = r if r is not None else self.order_target
        if r_use is None:
            r_use = self._auto_select_order(sigma)
        if r_use >= n_plant:
            raise ValueError(
                f"Requested order r={r_use} must be less than "
                f"plant order n={n_plant}."
            )

        # Truncate (operate on plant-only matrices, not augmented)
        A_r, B_r, C_r, D_r = _truncate(A, B, C, D, T, T_inv, r_use, verbose=self.verbose)

        return FWBTResult(
            A_r=A_r, B_r=B_r, C_r=C_r, D_r=D_r,
            hsv=sigma,
            truncation_index=r_use,
            hinf_error_bound=_hinf_error_bound(sigma, r_use),
            is_stable=_check_stability(A_r, self.discrete),
            discrete=self.discrete,
        )