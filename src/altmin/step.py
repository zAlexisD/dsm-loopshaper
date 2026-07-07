from __future__ import annotations
from typing import Protocol, Any

class AlternatingStep(Protocol):
    """
    One half of an alternating minimization pair.

    A "block" is opaque to the generic solver — it might be a tuple of
    matrices (BMI's Af,Bf,Cf,Df), a P matrix, a pair of pole/zero arrays,
    or anything else. The solver only ever passes blocks through.
    """

    def solve(self, fixed: Any) -> tuple[Any | None, float]:
        """
        Given the other block's fixed value, solve this block's subproblem.

        Returns (new_block_value, objective_value).
        Returns (None, inf) if infeasible.
        """
        ...