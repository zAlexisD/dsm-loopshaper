"""
BMI Module - Problem definition
"""

from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np
from scipy import signal
import cvxpy as cp

from bmi.solution import BMISolution

class OrderStatus(Enum):
    UNSET     = auto()   # not yet determined
    FIXED     = auto()   # user-specified (alt. min, CCL)
    DERIVED   = auto()   # computed from reduction step

class BMIProblem:
    def __init__(self, plant: signal.StateSpace, gamma: float = 1.0):
        self.plant  = plant
        self.gamma  = gamma
        
        self._order        = None
        self._order_status = OrderStatus.UNSET
        self._variables    = {}   # populated once order is known

    # ── Order binding ──────────────────────────────────────────────

    def set_order(self, n_f: int) -> "BMIProblem":
        """Explicit order — used by alternating min / CCL."""
        if self._order_status == OrderStatus.DERIVED:
            raise RuntimeError("Order already derived from reduction; cannot override.")
        self._order        = n_f
        self._order_status = OrderStatus.FIXED
        self._init_variables()
        return self   # allow chaining

    def _set_derived_order(self, n_f: int) -> None:
        """Called internally after balanced truncation or similar."""
        self._order        = n_f
        self._order_status = OrderStatus.DERIVED
        self._init_variables()

    @property
    def order(self) -> int:
        if self._order is None:
            raise RuntimeError("Filter order not set. Call set_order() or run a reduction step first.")
        return self._order

    # ── Variable initialization (deferred) ─────────────────────────

    def _init_variables(self) -> None:
        n   = self.plant.A.shape[0]
        n_f = self._order
        n_u = self.plant.B.shape[1]
        n_y = self.plant.C.shape[0]

        self._variables = {
            "P"  : cp.Variable((n, n),     symmetric=True),
            "A_f": cp.Variable((n_f, n_f)),
            "B_f": cp.Variable((n_f, n_y)),
            "C_f": cp.Variable((n_u, n_f)),
            "D_f": cp.Variable((n_u, n_y)),
        }

    # ── Solver interface ───────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._order is not None

    def solve(self) -> BMISolution:
        if not self.is_ready:
            raise RuntimeError(
                f"BMIProblem is not ready (order status: {self._order_status.name}). "
                "Call set_order() or run a reduction step first."
            )
        ...