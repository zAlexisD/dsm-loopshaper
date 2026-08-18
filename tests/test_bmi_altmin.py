"""
test_bmi_altmin.py

Tests the BMI alternating minimization pipeline (P/K steps) with the
new H2 objective formulation (minimize mu subject to three LMIs).
"""
import numpy as np

from bmi.problem import BMIProblem
from bmi.altmin.problem import BMIAltMinProblem
from altmin.init import LHSInit, ZeroInit, MixedInit
from bmi.altmin.init import SynthesizeNTFInit


# ── 1. Instantiate BMIProblem ──────────────────────────────────────────────────

FILTER_ORDER = 2
N_INPUTS     = 1
N_OUTPUTS    = 1
GAMMA        = 1.5

bmi = BMIProblem(
    n_filter=FILTER_ORDER,
    n_input=N_INPUTS,
    n_output=N_OUTPUTS,
    gamma=GAMMA)

print(f"Filter order         : {FILTER_ORDER}")
print(f"Output filter order  : {bmi.outFilterOrder}")
print(f"Combined system order: {bmi.n}")
print(f"Gamma (fixed)        : {GAMMA}")


# ── 2. Sanity-check build_system ───────────────────────────────────────────────

Af = np.diag([0.5, 0.3])
Bf = np.array([[0.0], [1.0]])
Cf = np.array([[1.0, 0.0]])
Df = np.array([[0.0]])

A, B, C, D = bmi.build_system(Af, Bf, Cf, Df)

print(f"\nbuild_system output shapes:")
print(f"  A: {A.shape}  expected: ({bmi.n}, {bmi.n})")
print(f"  B: {B.shape}  expected: ({bmi.n}, {N_INPUTS})")
print(f"  C: {C.shape}  expected: ({N_OUTPUTS}, {bmi.n})")
print(f"  D: {D.shape}  expected: ({N_OUTPUTS}, {N_INPUTS})")


# ── 3. Sanity-check the three LMIs with plain numpy ───────────────────────────

n  = bmi.n
nf = bmi.n_f
P_test  = np.eye(n)
Pf_test = np.eye(nf)
mu_test = 2.0

LMI1 = bmi.build_LMI1(P_test, A, B)
LMI2 = bmi.build_LMI2(P_test, mu_test, C, D)
LMI3 = bmi.build_LMI3(Pf_test, Af, Bf, Cf, Df, GAMMA)

print(f"\nLMI shapes (numpy):")
print(f"  LMI1: {LMI1.shape}  expected: ({2*n+N_INPUTS}, {2*n+N_INPUTS})")
print(f"  LMI2: {LMI2.shape}  expected: ({N_OUTPUTS+n+N_INPUTS}, {N_OUTPUTS+n+N_INPUTS})")
print(f"  LMI3: {LMI3.shape}  expected: ({2*nf+N_INPUTS+N_OUTPUTS}, {2*nf+N_INPUTS+N_OUTPUTS})")

print(f"\nLMI1 min eigenvalue (should be > 0 for feasibility): "
      f"{np.min(np.linalg.eigvalsh(LMI1)):.4f}")
print(f"LMI2 min eigenvalue (should be > 0 for feasibility): "
      f"{np.min(np.linalg.eigvalsh(LMI2)):.4f}")
print(f"LMI3 max eigenvalue (should be < 0 for feasibility): "
      f"{np.max(np.linalg.eigvalsh(LMI3)):.4f}")


# ── 4. Run BMIAltMinProblem ────────────────────────────────────────────────────

OSR = 32

init_strategy = MixedInit([
    (LHSInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS, bound=0.5), 0.4),
    (ZeroInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS), 0.1),
    (SynthesizeNTFInit(FILTER_ORDER, osr=OSR), 0.5)
])

problem = BMIAltMinProblem(
    bmi_problem=bmi,
    osr=OSR,
    filter_order=FILTER_ORDER,
    n_inputs=N_INPUTS,
    n_outputs=N_OUTPUTS,
    init_strategy=init_strategy,
    n_starts=6,
    max_iter=20,
    tol=1e-3,
    n_jobs=1,
    verbose=True)

print("\nRunning BMI alternating minimization ...")
result = problem.solve(seed=42)


# ── 5. Inspect results ─────────────────────────────────────────────────────────

print(f"\nFeasible runs : {result.n_feasible} / {len(result.runs)}")

if result.n_feasible == 0:
    print("No feasible solution found.")
    print("Suggestions:")
    print("  - Increase gamma")
    print("  - Widen LHS bound")
    print("  - Increase n_starts")
else:
    sol = result.best
    print(f"Best mu              : {sol.mu:.4e}")
    print(f"Gamma (fixed)        : {GAMMA}")
    print(f"P  eigenvalues       : {np.linalg.eigvalsh(sol.P)}")
    print(f"Pf eigenvalues       : {np.linalg.eigvalsh(sol.Pf)}")
    print(f"Filter poles         : {np.linalg.eigvals(sol.Af)}")

    # Closed-loop stability check
    A_cl, _, _, _ = bmi.build_system(sol.Af, sol.Bf, sol.Cf, sol.Df)
    cl_poles = np.linalg.eigvals(A_cl)
    stable = np.all(np.abs(cl_poles) < 1.0)
    print(f"Closed-loop stable   : {stable}  "
          f"(max |pole| = {np.max(np.abs(cl_poles)):.4f})")

    # Verify LMIs at solution
    A_sol, B_sol, C_sol, D_sol = bmi.build_system(sol.Af, sol.Bf, sol.Cf, sol.Df)
    L1 = bmi.build_LMI1(sol.P, A_sol, B_sol)
    L2 = bmi.build_LMI2(sol.P, sol.mu, C_sol, D_sol)
    L3 = bmi.build_LMI3(sol.Pf, sol.Af, sol.Bf, sol.Cf, sol.Df, GAMMA)
    print(f"\nLMI verification at solution:")
    print(f"  LMI1 min eig (> 0): {np.min(np.linalg.eigvalsh(L1)):.4e}")
    print(f"  LMI2 min eig (> 0): {np.min(np.linalg.eigvalsh(L2)):.4e}")
    print(f"  LMI3 max eig (< 0): {np.max(np.linalg.eigvalsh(L3)):.4e}")

    # Convergence across runs
    print(f"\nPer-run summary:")
    print(f"  {'init':<16} {'iters':>5}  {'converged':>9}  {'final mu':>12}")
    for run in result.runs:
        if run.solution is not None:
            mu_final = run.residual_history[-1] if run.residual_history else float("inf")
            print(f"  {run.init_label:<16} {run.n_iterations:>5}  "
                  f"{str(run.converged):>9}  {mu_final:>12.4e}")
        else:
            print(f"  {run.init_label:<16} {'—':>5}  {'infeasible':>9}  {'—':>12}")