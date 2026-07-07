"""
Alternating Minimization Module - Strategies definition to define different initial values
"""
from __future__ import annotations
import numpy as np
from itertools import product
from scipy.signal import tf2ss

from altmin.init import FilterMatrices
from core.core import synthesizeNTF

class SynthesizeNTFInit:
    """Candidates from Shreier's synthesizeNTF variants across a grid of (order,opt,H_inf)"""
    label = "synthesize_ntf"

    def __init__(self,osr:int,order:int,order_spread:int=1,opt:list[int]|None=[0,1,2],Hinf:list[float]|None=[1.5,2.0]):
        self.osr          = osr
        self.order        = order
        self.order_spread = order_spread
        self.opt          = opt
        self.H_inf        = Hinf
    
    def _candidates(self):
        """Lazily enumerate (order, opt, H_inf) combos."""
        orders = range(
            max(1, self.order - self.order_spread),
            self.order + self.order_spread + 1)
        return list(product(orders, self.opt, self.H_inf))

    def generate(self,n:int,rng:np.random.Generator) -> list[tuple[FilterMatrices, str]]:
        results = []
        for order, opt, H_inf in self._candidates():
            if len(results) >= n:
                break
            try:
                ntf = synthesizeNTF(order, self.osr, opt, H_inf)  # returns ZPK
                # L(z) = NTF(z) - 1  →  subtract 1 from the gain in transfer function form
                ntf.num[-1] -= ntf.den[-1]   # subtract 1: L = NTF - 1
                Af, Bf, Cf, Df = tf2ss(ntf.num, ntf.den)
                results.append(((Af, Bf, Cf, Df), self.label))
            except Exception:
                continue   # some (order, type, H_inf) combos are infeasible — skip silently
        return results