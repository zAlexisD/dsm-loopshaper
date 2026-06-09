"""
Alternating Minimization Module - Strategies definition to define different initial values
"""
from typing import Protocol
import numpy as np

class InitStrategy(Protocol):
    label: str
    def generate(self, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Return n candidate K matrices."""
        ...

class SynthesizeNTFInit:
    """Candidates from synthesizeNTF variants (OSR, order, type combos)."""
    label = "synthesize_ntf"
    def __init__(self, osr: int, order_range: range, ...): ...
    def generate(self, n, rng): ...

class WarmStartInit:
    """Candidates from a prior AltMinResult or BMISolution."""
    label = "warm_start"
    def __init__(self, source: BMISolution | AltMinResult, noise_scale: float = 0.05): ...
    def generate(self, n, rng): ...

class LHSInit:
    """Latin Hypercube Sampling over (A_f, B_f, C_f, D_f) space."""
    label = "lhs"
    def __init__(self, filter_order: int, radius_bound: float = 0.95): ...
    def generate(self, n, rng): ...

class ZeroInit:
    """K=0 as a single diversity point."""
    label = "zero"
    def generate(self, n, rng): ...

class MixedInit:
    """Weighted combination of strategies — the default you'll actually use."""
    def __init__(self, strategies: list[tuple[InitStrategy, float]]): ...
    def generate(self, n, rng) -> list[tuple[np.ndarray, str]]:
        # proportional allocation, returns (K, label) pairs
        ...