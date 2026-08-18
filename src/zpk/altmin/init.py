from __future__ import annotations
import numpy as np
from scipy.signal import ss2tf

from zpk.parametrization import PoleParams, ZeroParams, ZPKParams
from bmi.problem import BMIProblem
from dsm.core import synthesizeNTF
from dsm.initDesign import optIIR
from fwbt.problem import FWBTProblem


class InitStrategy:
    label: str

    def generate(self, n: int, rng: np.random.Generator, **kwargs) -> list[tuple[ZPKParams, str]]:
        ...


class UniformPoleInit:
    """
    Sample poles uniformly in polar form (r, θ) then convert to coefficients.
    Conjugate pairs + optional real poles.
    Stability enforced by r < radius_bound.
    """
    label = "uniform_poles"

    def __init__(
        self,
        n_poles: int,
        n_zeros: int,
        radius_bound: float = 0.95,
        zero_radius_bound: float = 1.0,
        dsm_bias: bool = False, 
        osr: int = 32,):

        self.n_poles = n_poles
        self.n_zeros = n_zeros
        self.radius_bound = radius_bound
        self.zero_radius_bound = zero_radius_bound
        self.dsm_bias = dsm_bias
        self.osr = osr

        if n_poles % 2 != 0:
            raise ValueError(
                "n_poles must be even (conjugate pairs only). "
                "Real pole support can be added later."
            )

    def _sample_poles(self, rng: np.random.Generator) -> np.ndarray:
        n_pairs = self.n_poles // 2
        radii = rng.uniform(0.0, self.radius_bound, size=n_pairs)
        if self.dsm_bias:
            # Bias angles toward signal band edge (π/OSR) 
            # so poles shape noise just above the signal band
            band_edge = np.pi / self.osr
            angles = rng.uniform(band_edge, 3 * band_edge, size=n_pairs)
        else:
            angles = rng.uniform(0.0, np.pi, size=n_pairs)
        pairs = radii * np.exp(1j * angles)
        return np.concatenate([pairs, pairs.conj()])

    def _sample_zeros(self, rng: np.random.Generator) -> np.ndarray:
        """Real zeros sampled uniformly in [-zero_radius_bound, zero_radius_bound]."""
        if self.dsm_bias:
            # Bias real zeros near z=+1 (DC) for noise shaping
            return rng.uniform(0.8, 1.0, size=self.n_zeros)
        else:
            return rng.uniform(
                -self.zero_radius_bound, 
                self.zero_radius_bound, 
                size=self.n_zeros
            )

    def generate(self, n: int, rng: np.random.Generator, **kwargs) -> list[tuple[ZPKParams, str]]:
        results = []
        for _ in range(n):
            poles = self._sample_poles(rng)
            zeros = self._sample_zeros(rng)
            pole_coeffs = np.poly(poles).real[1:]   # drop leading 1
            zero_coeffs = np.poly(zeros).real       # includes leading b_0
            params = ZPKParams(
                poles=PoleParams(coeffs=pole_coeffs),
                zeros=ZeroParams(coeffs=zero_coeffs),
            )
            results.append((params, self.label))
        return results


class WarmStartZPKInit:
    """
    Warm start from optIIR() MATLAB-designed filter, converted to ZPKParams,
    with Gaussian perturbations on the polynomial coefficients.
    """
    label = "warm_start_zpk"

    def __init__(self, bmi_problem: BMIProblem, noise_scale: float = 0.05):
        self.bmi_problem = bmi_problem
        self.fullorder   = bmi_problem.outFilterOrder
        self.noise_scale = noise_scale

    def _base_zpk(self, order: int, gamma: float) -> ZPKParams:

        bmi = self.bmi_problem
        Aw = bmi._output_filter.A
        Bw = bmi._output_filter.B
        Cw = bmi._output_filter.C
        Dw = bmi._output_filter.D
        R_full = optIIR(self.fullorder, gamma, Aw, Bw, Cw, Dw)

        # Reduce order via FWBT
        prob = FWBTProblem(R_full,order_target=order,W_o=(Aw,Bw,Cw,Dw),discrete=True)
        result = prob.solve()
        R_reduced = result.to_statespace()
        Af = R_reduced.A
        Bf = R_reduced.B
        Cf = R_reduced.C
        Df = R_reduced.D

        # Extract polynomial coefficients from state-space matrices
        num, den = ss2tf(Af, Bf, Cf, Df)
        pole_coeffs = den[1:]       # drop leading 1
        zero_coeffs = num.flatten()

        return ZPKParams(
            poles=PoleParams(coeffs=pole_coeffs),
            zeros=ZeroParams(coeffs=zero_coeffs))

    def generate(self,n: int,rng: np.random.Generator,order: int,gamma: float
                 ) -> list[tuple[ZPKParams, str]]:
        
        base = self._base_zpk(order, gamma)
        results = [(base, self.label)]

        for _ in range(n - 1):
            pole_coeffs = base.poles.coeffs + rng.normal(
                scale=self.noise_scale, size=base.poles.coeffs.shape
            )
            zero_coeffs = base.zeros.coeffs + rng.normal(
                scale=self.noise_scale, size=base.zeros.coeffs.shape
            )
            results.append((
                ZPKParams(
                    poles=PoleParams(coeffs=pole_coeffs),
                    zeros=ZeroParams(coeffs=zero_coeffs),
                ),
                self.label))
        return results


class SynthesizeNTFZPKInit:
    """
    Derive initial ZPKParams from synthesizeNTF.
    Poles come from NTF denominator; zeros from NTF numerator.
    """
    label = "synthesize_ntf_zpk"

    def __init__(
        self,
        order: int,
        osr: int = 32,
        opt_options: list[int] | None = None,
        H_inf_values: list[float] | None = None):

        self.order = order
        self.osr = osr
        self.opt_options = opt_options or [0, 1, 2]
        self.H_inf_values = H_inf_values or [1.3, 1.5, 1.8, 2.0]

    def _candidates(self):
        """
        Build (opt, H_inf) pairs.
        H_inf is only meaningful for opt=2; for opt=0,1 it is ignored.
        """
        candidates = []
        for opt in self.opt_options:
            if opt == 2:
                for H_inf in self.H_inf_values:
                    candidates.append((opt, H_inf))
            else:
                candidates.append((opt, None))
        return candidates

    def generate(self, n: int, rng: np.random.Generator, **kwargs) -> list[tuple[ZPKParams, str]]:
        
        results = []
        for opt, H_inf in self._candidates():
            if len(results) >= n:
                break
            try:
                args = [self.order, self.osr, opt]
                if H_inf is not None:
                    args.append(H_inf)

                ntf = synthesizeNTF(*args)
                ntf_tf = ntf.to_tf()
                # L(z) = NTF(z) - 1 → adjust numerator
                num = ntf_tf.num.copy()
                num[-1] -= ntf_tf.den[-1]
                pole_coeffs = ntf_tf.den[1:]           # drop leading 1
                zero_coeffs = num                       # full numerator
                params = ZPKParams(
                    poles=PoleParams(coeffs=pole_coeffs),
                    zeros=ZeroParams(coeffs=zero_coeffs),
                )
                results.append((params, self.label))
            except Exception:
                continue
        return results


class MixedZPKInit:
    """Proportional mix of ZPK init strategies."""

    def __init__(self, strategies: list[tuple[InitStrategy, float]]):
        self.strategies = strategies

    def generate(self, n: int, rng: np.random.Generator, **kwargs) -> list[tuple[ZPKParams, str]]:
        weights = np.array([w for _, w in self.strategies], dtype=float)
        weights /= weights.sum()
        allocations = np.round(weights * n).astype(int)
        allocations[-1] += n - allocations.sum()

        results = []
        for (strategy, _), count in zip(self.strategies, allocations):
            if count > 0:
                results.extend(strategy.generate(count, rng, **kwargs))
        return results