"""
Alternating Minimization Module - Strategies definition to define different initial values
"""
from __future__ import annotations
import numpy as np
from scipy.signal import tf2ss

from altmin.init import FilterMatrices
from bmi.problem import BMIProblem
from dsm.core import synthesizeNTF
from dsm.initDesign import optIIR
from fwbt.problem import FWBTProblem

class SynthesizeNTFInit:
    """Candidates from Shreier's synthesizeNTF variants across a grid of (opt,H_inf)"""
    label = "synthesize_ntf"

    def __init__(self,
                 order: int,
                 osr: int = 64,
                 opt: list[int] | None = [0, 1, 2],
                 Hinf: list[float] | None = [1.5, 2.0]):
        
        self.order        = order
        self.osr          = osr
        self.opt          = opt
        self.H_inf        = Hinf
    
    def _candidates(self) -> list[tuple[int, float | None]]:
        """
        Build (opt, H_inf) pairs.
        H_inf is only meaningful for opt=2; for opt=0,1 it is ignored.
        """
        candidates = []
        for opt in self.opt:
            if opt == 2:
                for H_inf in self.H_inf:
                    candidates.append((opt, H_inf))
            else:
                candidates.append((opt, None))
        return candidates

    def generate(self,n:int,rng:np.random.Generator, **kwargs) -> list[tuple[FilterMatrices, str]]:
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
                # L(z) = NTF(z) - 1  →  subtract 1 from the gain in transfer function form
                num = ntf_tf.num.copy()
                num[-1] -= ntf_tf.den[-1]   # subtract 1: L = NTF - 1
                Af, Bf, Cf, Df = tf2ss(num, ntf_tf.den)
                results.append(((Af, Bf, Cf, Df), self.label))
            except Exception:
                continue   # some (order, type, H_inf) combos are infeasible — skip silently
        return results
    

class WarmStartInit:
    """Prior (Af, Bf, Cf, Df) candidate with optional Gaussian noise perturbations."""
    label = "warm_start"

    def __init__(self, bmi_problem: BMIProblem, noise_scale: float = 0.05):
        self.bmi_problem = bmi_problem
        self.noise_scale = noise_scale
        self.fullorder   = bmi_problem.outFilterOrder

    def _base_matrices(self, order: int, gamma: float) -> FilterMatrices:
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
        
        return Af, Bf, Cf, Df

    def generate(self, n: int, rng: np.random.Generator, order: int, gamma: float
                 ) -> list[tuple[FilterMatrices, str]]:
        
        Af, Bf, Cf, Df = self._base_matrices(order, gamma)
        base = (Af, Bf, Cf, Df)

        results = [(base, self.label)]
        for _ in range(n - 1):
            noisy = tuple(
                M + rng.normal(scale=self.noise_scale, size=M.shape)
                for M in base
            )
            results.append((noisy, self.label))
        return results