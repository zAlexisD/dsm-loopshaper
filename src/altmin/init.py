"""
Alternating Minimization Module - Strategies definition to define different initial values
"""
from typing import Protocol,runtime_checkable
import numpy as np
from scipy.stats.qmc import LatinHypercube

FilterMatrices = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

@runtime_checkable
class InitStrategy(Protocol):
    label: str

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[FilterMatrices, str]]:
        """Return n (K = (Af, Bf, Cf, Df), label) pairs."""
        ...

class WarmStartInit:
    """Prior (Af, Bf, Cf, Df) candidate with optional Gaussian noise perturbations."""
    label = "warm_start"

    #TODO: Code in python the matlab solution that we have with white noise
    def __init__(self, base: FilterMatrices, noise_scale: float = 0.05):
        self.base = base
        self.noise_scale = noise_scale

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[FilterMatrices, str]]:
        results = [(self.base, self.label)]
        for _ in range(n - 1):
            noisy = tuple(
                M + rng.normal(scale=self.noise_scale, size=M.shape)
                for M in self.base
            )
            results.append((noisy, self.label))
        return results

class LHSInit:
    """Latin Hypercube Sampling over (Af, Bf, Cf, Df) space."""
    label = "lhs"

    def __init__(self, filter_order: int, n_inputs: int, n_outputs: int, bound: float = 1.0):
        self.p = filter_order
        self.m_in = n_inputs
        self.m_out = n_outputs
        self.bound = bound

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[FilterMatrices, str]]:
        p, m_in, m_out = self.p, self.m_in, self.m_out
        dim = p * p + p * m_out + m_in * p + m_in * m_out
        sampler = LatinHypercube(d=dim, seed=rng.integers(0, 2**31))
        samples = (sampler.random(n=n) * 2 - 1) * self.bound

        results = []
        for row in samples:
            Af, Bf, Cf, Df = self._unpack(row)
            results.append(((Af, Bf, Cf, Df), self.label))
        return results

    def _unpack(self, flat: np.ndarray) -> FilterMatrices:
        p, m_in, m_out = self.p, self.m_in, self.m_out
        i = 0
        Af = flat[i:i + p * p].reshape(p, p); i += p * p
        Bf = flat[i:i + p * m_out].reshape(p, m_out); i += p * m_out
        Cf = flat[i:i + m_in * p].reshape(m_in, p); i += m_in * p
        Df = flat[i:i + m_in * m_out].reshape(m_in, m_out)
        return Af, Bf, Cf, Df

class ZeroInit:
    """All-zero feedback filter as a single diversity anchor."""
    label = "zero"

    def __init__(self, filter_order: int, n_inputs: int, n_outputs: int):
        self.p = filter_order
        self.m_in = n_inputs
        self.m_out = n_outputs
    
    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[FilterMatrices, str]]:
        p, m_in, m_out = self.p, self.m_in, self.m_out
        zero = (
            np.zeros((p, p)),
            np.zeros((p, m_out)),
            np.zeros((m_in, p)),
            np.zeros((m_in, m_out)),
        )
        return [(zero, self.label)]   

class MixedInit:
    """Weighted combination of strategies."""

    def __init__(self, strategies: list[tuple[InitStrategy, float]]): 
        self.strategies = strategies  # (strategy, weight) pairs

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, str]]:
        # proportional allocation, returns (K, label) pairs
        weights = np.array([w for _, w in self.strategies], dtype=float)
        weights /= weights.sum()
        allocations = np.round(weights * n).astype(int)
        # fix rounding so total == n
        allocations[-1] += n - allocations.sum()

        results = []
        for (strategy, _), count in zip(self.strategies, allocations):
            if count > 0:
                results.extend(strategy.generate(count, rng))
        return results