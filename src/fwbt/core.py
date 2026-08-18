"""
FWBT Module: internal numerical routines
"""
import numpy as np
from scipy import linalg


def _build_augmented_system(A, B, C, D, W_i=None, W_o=None):
    """
    Assemble augmented state-space from plant + optional input/output weights.
    
    Weight state-spaces are (A_w, B_w, C_w, D_w) tuples.
    Returns (A_aug, B_aug, C_aug, D_aug, n_plant).
    """
    n = A.shape[0]

    if W_i is None and W_o is None:
        return A, B, C, D, n

    # Apply input weight: u -> W_i -> plant
    if W_i is not None:
        Ai, Bi, Ci, Di = W_i
        # Augmented: x_aug = [x_plant; x_wi]
        A = np.block([
            [A,        B @ Ci],
            [np.zeros((Ai.shape[0], n)), Ai]
        ])
        B = np.vstack([B @ Di, Bi])
        C = np.hstack([C, D @ Ci])
        D = D @ Di
        n = A.shape[0]

    # Apply output weight: plant -> W_o -> y
    if W_o is not None:
        Ao, Bo, Co, Do = W_o
        n_o = Ao.shape[0]
        # Augmented: x_aug = [x_plant_wi; x_wo]
        A = np.block([
            [A,                          np.zeros((n, n_o))],
            [Bo @ C,  Ao]
        ])
        B = np.vstack([B, Bo @ D])
        C = np.hstack([Do @ C, Co])
        D = Do @ D

    return A, B, C, D, n


def _solve_gramians(A_aug, B_aug, C_aug, D_aug, n_plant, discrete=True):
    """
    Solve Lyapunov equations on augmented system.
    Extract P, Q for plant states via projection.
    Returns (P, Q, P_tilde, Q_tilde).
    """
    if discrete:
        P_tilde = linalg.solve_discrete_lyapunov(A_aug, B_aug @ B_aug.T)
        Q_tilde = linalg.solve_discrete_lyapunov(A_aug.T, C_aug.T @ C_aug)
    else:
        P_tilde = linalg.solve_continuous_lyapunov(A_aug, -B_aug @ B_aug.T)
        Q_tilde = linalg.solve_continuous_lyapunov(A_aug.T, -C_aug.T @ C_aug)

    # Enforce symmetry
    P_tilde = (P_tilde + P_tilde.T) / 2
    Q_tilde = (Q_tilde + Q_tilde.T) / 2

    # Project onto plant states
    P = P_tilde[:n_plant, :n_plant]
    Q = Q_tilde[:n_plant, :n_plant]

    return P, Q, P_tilde, Q_tilde


def _balanced_transform(P, Q, reg=1e-10,verbose:bool=False):
    """
    Compute balancing transform via SVD of the cross-gramian factor,
    avoiding Cholesky on ill-conditioned P.
    
    Uses the square-root method:
      1. SVD of P  →  P = Up Sp Up^T  →  Lp = Up sqrt(Sp)
      2. SVD of Lp^T Q Lp  →  U S V^T
      3. T = S^{-1/2} U^T Lp^{-1}  (but computed via pinv for stability)
    """
    n = P.shape[0]

    P = (P + P.T) / 2
    Q = (Q + Q.T) / 2

    # eigh of P
    eigvals_p, Up = linalg.eigh(P)
    eigvals_p = np.maximum(eigvals_p, 0)

    # Truncate near-zero eigenvalues of P for numerical stability
    tol = reg * eigvals_p[-1]
    mask = eigvals_p > tol
    rank = np.sum(mask)

    eigvals_p_r = eigvals_p[mask]
    Up_r = Up[:, mask]

    Lp = Up_r * np.sqrt(eigvals_p_r)   # (n, rank)

    # SVD of Lp^T Q Lp  — size (rank, rank)
    M = Lp.T @ Q @ Lp
    M = (M + M.T) / 2
    eigvals_m, Um = linalg.eigh(M)

    # Sort descending
    idx = np.argsort(eigvals_m)[::-1]
    eigvals_m = eigvals_m[idx]
    Um = Um[:, idx]

    sigma_r = np.sqrt(np.maximum(eigvals_m, 0))   # length == rank

    # Pad sigma back to length n with zeros for the discarded directions
    sigma = np.zeros(n)
    sigma[:rank] = sigma_r

    if verbose:
        print(f"  [DEBUG] sigma: {sigma}")
        print(f"  [DEBUG] cond(Lp): {np.linalg.cond(Lp):.2e}")

    # Balancing transform from the rank-r subspace
    sigma_inv_half = np.where(sigma_r > reg, 1.0 / np.sqrt(sigma_r), 0.0)
    Lp_pinv = np.linalg.pinv(Lp)
    T = (sigma_inv_half[:, None] * Um.T) @ Lp_pinv   # (rank, n)

    # Pad T and T_inv to (n, n) if rank < n
    if rank < n:
        T_full = np.zeros((n, n))
        T_full[:rank, :] = T
        # Fill remaining rows with orthogonal complement to make T invertible
        T_full[rank:, :] = linalg.null_space(T).T
        T = T_full

    T_inv = np.linalg.pinv(T)

    return T, T_inv, sigma


def _truncate(A, B, C, D, T, T_inv, r,verbose:bool=False):
    """
    Apply balancing transform and truncate to order r.
    Returns (A_r, B_r, C_r, D_r).
    """
    # Transform to balanced coordinates
    A_bal = T @ A @ T_inv
    B_bal = T @ B
    C_bal = C @ T_inv

    if verbose:
        print(f"  [DEBUG] A_bal diagonal: {np.diag(A_bal)}")
        print(f"  [DEBUG] C_bal[:, :r]: {C_bal[:, :r]}")

    # Truncate
    A_r = A_bal[:r, :r]
    B_r = B_bal[:r, :]
    C_r = C_bal[:, :r]
    D_r = D  # D is unchanged by state transformation

    return A_r, B_r, C_r, D_r


def _hinf_error_bound(sigma, r):
    """
    Compute the additive H-infinity error bound: 2 * sum(sigma[r:]).
    """
    return 2 * np.sum(sigma[r:])


def _check_stability(A, discrete=True):
    eigvals = linalg.eigvals(A)
    if discrete:
        return bool(np.all(np.abs(eigvals) < 1))
    else:
        return bool(np.all(np.real(eigvals) < 0))