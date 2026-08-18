"""
Initial DSM Design from Matlab code
"""
import cvxpy as cp
import numpy as np
from scipy.signal import StateSpace,TransferFunction
from scipy.linalg import solve_discrete_lyapunov

def optIIR(order,gamma,Ah,Bh,Ch,Dh,ts:float=1.0):
    # CVXPY variables
    Pf = cp.Variable((order, order), symmetric=True)
    Pg = cp.Variable((order, order), symmetric=True)
    Wf = cp.Variable((1,order))
    Wg = cp.Variable((order,1))
    L  = cp.Variable((order,order))
    mu = cp.Variable()

    # Changing variable matrixes
    Ma = cp.bmat([
        [Ah @ Pf + Bh @ Wf, Ah],
        [L, Pg @ Ah]
    ])
    Mb = cp.vstack([
        Bh,
        Wg
    ])
    Mc = cp.hstack([
        Ch @ Pf + Dh @ Wf, Ch
    ])
    Mp = cp.bmat([
        [Pf, np.eye(order)],
        [np.eye(order), Pg]
    ])
    Mc_tilde = cp.hstack([
        Wf, np.zeros((1,order))
    ])

    # LMIs
    Lmi1 = cp.bmat([
        [Mp, Ma, Mb],
        [Ma.T, Mp, np.zeros((2*order,1))],
        [Mb.T, np.zeros((1,2*order)), np.eye(1)]
    ])
    Lmi2 = cp.bmat([
        [mu*np.eye(1), Mc, Dh.T],
        [Mc.T, Mp, np.zeros((2*order,1))],
        [Dh, np.zeros((1,2*order)), np.eye(1)]
    ])
    Lmi3 = cp.bmat([
        [(gamma**2)*np.eye(1), Mc_tilde],
        [Mc_tilde.T, Mp]
    ])

    # Problem formulation
    constraints = [
        Pf >> 1e-6 * np.eye(order),
        Pg >> 1e-6 * np.eye(order),
        Lmi1 >> 0,
        Lmi2 >> 0,
        Lmi3 >> 0
    ]
    problem = cp.Problem(cp.Minimize(mu),constraints)
    problem.solve(solver=cp.SCS,eps=1e-6)

    if problem.status not in ("optimal", "optimal_inaccurate"):
            return None
    
    # Extract optimal values
    PfVal = Pf.value
    PgVal = Pg.value
    PgInv = np.linalg.inv(PgVal)
    WfVal = Wf.value
    WgVal = Wg.value
    L_Val = L.value

    # Reconstruct feedback filter components
    Sf = PfVal - PgInv
    SfInv = np.linalg.inv(Sf)
    Af = (Bh @ WfVal - PgInv @ (L_Val - PgVal @ Ah @ PfVal)) @ SfInv
    Bf = Bh - PgInv @ WgVal
    Cf = WfVal @ SfInv
    Df = np.eye(1)

    return StateSpace(Af,Bf,Cf,Df,dt=ts)

def computeMSE(H:StateSpace,R:StateSpace):
      # Multiply transfer functions H*R
    Hz = TransferFunction(H.to_tf())
    Rz = TransferFunction(R.to_tf())

    # Convolve numerators and denominators to get H*R
    num = np.polymul(Hz.num, Rz.num)
    den = np.polymul(Hz.den, Rz.den)
    RHz = TransferFunction(num, den)

    # H2 norm via state space
    RH_ss = RHz.to_ss()
    # Solve discrete Lyapunov equation: A*P*A' - P + B*B' = 0
    P = solve_discrete_lyapunov(RH_ss.A, RH_ss.B @ RH_ss.B.T)
    h2 = np.sqrt(np.trace(RH_ss.C @ P @ RH_ss.C.T + RH_ss.D @ RH_ss.D.T))

    # MSE computation
    MSE = 20 * np.log10(h2)
    return MSE