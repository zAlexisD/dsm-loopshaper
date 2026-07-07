"""
FWBT Module: FWBTProblem — user-facing entry point
"""
from bmi.fwbt.result import FWBTResult

class FWBTProblem:
    def __init__(self,
                 ss_full,          # StateSpace of full-order NTF
                 order_target,     # int or None (None = auto from HSV gap)
                 W_i=None,         # input weight StateSpace, None = identity
                 W_o=None,         # output weight StateSpace, None = identity
                 discrete=True):
        ...

    @property
    def is_ready(self) -> bool:
        """Check A is stable, weights are compatible."""
        ...

    def hsv_plot(self):
        """Diagnostic: plot singular value decay before committing to an order."""
        ...

    def solve(self, r=None) -> FWBTResult:
        """
        r overrides order_target for quick experimentation.
        Calls core functions in sequence:
          _build_augmented_system → _solve_gramians → _balanced_transform → _truncate
        Wraps result in FWBTResult, computes error bound and stability flag.
        """
        ...