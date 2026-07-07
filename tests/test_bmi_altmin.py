"""
test_bmi_altmin.py

Minimal test of the BMI alternating minimization pipeline.
Uses a synthetic SISO DSM-like problem with a 2nd-order feedback filter.
"""
import numpy as np

from bmi.problem import BMIProblem
from bmi.altmin.problem import BMIAltMinProblem
from altmin.init import LHSInit, ZeroInit, MixedInit


# ── 1. Instantiate BMIProblem ──────────────────────────────────────────────────
#
# SISO: n_input=1, n_output=1
# Feedback filter order: 2
# Output filter is built internally (Butterworth order 4, cutoff pi/32)

FILTER_ORDER = 2
N_INPUTS     = 1
N_OUTPUTS    = 1
OSR          = 32

bmi = BMIProblem(n_filter=FILTER_ORDER, n_input=N_INPUTS, n_output=N_OUTPUTS)

print(f"Output filter order : {bmi.outFilterOrder}")
print(f"Combined system order: {bmi.n}")


# ── 2. Sanity-check build_system with a known stable filter ───────────────────
#
# Use a simple stable filter: A_f diagonal with small eigenvalues

A_f = np.diag([0.5, 0.3])
B_f = np.array([[1.0], [1.0]])
C_f = np.array([[1.0, 0.0]])
D_f = np.array([[0.0]])

A, B, C, D = bmi.build_system(A_f, B_f, C_f, D_f)

print(f"\nbuild_system output shapes:")
print(f"  A: {A.shape}  (expected ({bmi.n}, {bmi.n}))")
print(f"  B: {B.shape}  (expected ({bmi.n}, {N_INPUTS}))")
print(f"  C: {C.shape}  (expected ({N_OUTPUTS}, {bmi.n}))")
print(f"  D: {D.shape}  (expected ({N_OUTPUTS}, {N_INPUTS}))")


# ── 3. Sanity-check build_BMI with a dummy P ──────────────────────────────────

n = bmi.n
P_test  = np.eye(n)
gamma_test = 2.0

BMI_mat = bmi.build_BMI(P_test, A, B, C, D, gamma_test)
print(f"\nbuild_BMI output shape: {BMI_mat.shape}")
# Should be (n + n_i + n_o, n + n_i + n_o) = (n+1+1, n+1+1) for SISO
expected_bmi_size = n + N_INPUTS + N_OUTPUTS
print(f"  expected: ({expected_bmi_size}, {expected_bmi_size})")


# ── 4. Run BMIAltMinProblem ────────────────────────────────────────────────────
#
# Use a small n_starts and max_iter so the test runs quickly.
# Can test by changing parameters/init strategies

init_strategy = MixedInit([
    (LHSInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS, bound=0.5), 0.9),
    (ZeroInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS), 0.1),
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
    n_jobs=2,
    verbose=True
)

print("\nRunning alternating minimization ...")
result = problem.solve(seed=42)


# ── 5. Inspect results ────────────────────────────────────────────────────────

print(f"\nFeasible runs : {result.n_feasible} / {len(result.runs)}")

if result.n_feasible == 0:
    print("No feasible solution found — check BMI formulation or init bounds.")
else:
    sol = result.best
    print(f"Best gamma    : {sol.gamma:.2e}")
    print(f"P eigenvalues : {np.linalg.eigvalsh(sol.P)}")
    print(f"Filter poles  : {np.linalg.eigvals(sol.Af)}")

    # Closed-loop stability check
    A_cl, _, _, _ = bmi.build_system(sol.Af, sol.Bf, sol.Cf, sol.Df)
    cl_poles = np.linalg.eigvals(A_cl)
    stable = np.all(np.abs(cl_poles) < 1.0)
    print(f"Closed-loop stable: {stable}  (max |pole| = {np.max(np.abs(cl_poles)):.4f})")

    # Convergence summary across runs
    print("\nPer-run summary:")
    print(f"  {'init':<16} {'iters':>5}  {'converged':>9}  {'gamma':>10}")
    for run in result.runs:
        if run.solution is not None:
            _, _, g = run.solution
            print(f"  {run.init_label:<16} {run.n_iterations:>5}  {str(run.converged):>9}  {g:>10.4f}")
        else:
            print(f"  {run.init_label:<16} {'—':>5}  {'infeasible':>9}  {'—':>10}")