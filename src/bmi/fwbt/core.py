"""
FWBT Module: internal numerical routines
"""

def _build_augmented_system(A, B, C, D, W_i, W_o):
    """Assemble augmented (A_aug, B_aug, C_aug, D_aug) from plant + weights."""
    ...

def _solve_gramians(A_aug, B_aug, C_aug, D_aug, n_plant, discrete):
    """Solve Lyapunov equations on augmented system, extract P and Q via projection."""
    ...

def _balanced_transform(P, Q):
    """Cholesky + eigh pipeline. Returns T, T_inv, sigma array."""
    ...

def _truncate(A, B, C, D, T, T_inv, r):
    """Apply T, then keep first r states. Returns A_r, B_r, C_r, D_r."""
    ...