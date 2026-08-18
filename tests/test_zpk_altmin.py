"""
test_zpk_altmin.py

Tests the ZPK alternating minimization pipeline (pole/zero steps) with
the H2 objective formulation (minimize mu subject to three LMIs).
"""
import numpy as np

from bmi.problem import BMIProblem
from zpk.parametrization import PoleParams, ZeroParams, ZPKParams
from zpk.altmin.problem import ZPKAltMinProblem
from zpk.altmin.init import UniformPoleInit, MixedZPKInit, SynthesizeNTFZPKInit, WarmStartZPKInit


# ── 1. Instantiate BMIProblem ──────────────────────────────────────────────────

N_POLES  = 2     
N_ZEROS  = 2     
GAMMA    = 1.5
OSR      = 32
N_FILTER = max(N_POLES,N_ZEROS)

bmi = BMIProblem(
    n_filter=N_FILTER,
    n_input=1,
    n_output=1,
    gamma=GAMMA)

print(f"Filter poles         : {N_POLES}")
print(f"Filter zeros         : {N_ZEROS}")
print(f"Output filter order  : {bmi.outFilterOrder}")
print(f"Combined system order: {bmi.n}")
print(f"Gamma (fixed)        : {GAMMA}")


# ── 2. Sanity-check parametrization ───────────────────────────────────────────

# Known stable poles: two conjugate pairs
#pole_coeffs = np.poly([0.5+0.3j, 0.5-0.3j, 0.3+0.4j, 0.3-0.4j]).real[1:]   # 4 poles
pole_coeffs = np.poly([0.5+0.3j, 0.5-0.3j]).real[1:]                        # 2 poles
zero_coeffs = np.poly([-0.5, 0.5]).real

pole_params = PoleParams(coeffs=pole_coeffs)
zero_params = ZeroParams(coeffs=zero_coeffs)
zpk = ZPKParams(poles=pole_params, zeros=zero_params)

print(f"\nParametrization sanity check:")
print(f"  Poles (from coeffs) : {pole_params.to_poles()}")
print(f"  Zeros (from coeffs) : {zero_params.to_zeros()}")
print(f"  All poles stable    : {pole_params.is_stable()}")
print(f"  Is proper           : {zpk.is_proper()}")

Af, Bf, Cf, Df = zpk.to_filter_matrices()
print(f"  Af shape: {Af.shape}  expected: ({N_POLES}, {N_POLES})")
print(f"  Bf shape: {Bf.shape}  expected: ({N_POLES}, 1)")
print(f"  Cf shape: {Cf.shape}  expected: (1, {N_POLES})")
print(f"  Df shape: {Df.shape}  expected: (1, 1)")


# ── 3. Sanity-check pole and zero steps individually ──────────────────────────

from zpk.altmin.steps import ZPKPoleStep, ZPKZeroStep

pole_step = ZPKPoleStep(bmi,N_ZEROS)
zero_step = ZPKZeroStep(bmi)

print("\nTesting ZPKPoleStep individually ...")
new_zeros, P, Pf, mu_p = pole_step.solve(pole_params)
if new_zeros is None:
    print("  ZPKPoleStep: infeasible — try increasing gamma or adjusting poles")
else:
    print(f"  ZPKPoleStep: feasible  mu = {mu_p:.4e}")
    print(f"  P  eigenvalues: {np.linalg.eigvalsh(P)}")
    print(f"  Pf eigenvalues: {np.linalg.eigvalsh(Pf)}")
    print(f"  Recovered zeros: {new_zeros.to_zeros()}")

    print("\nTesting ZPKZeroStep individually ...")
    new_poles, sol_z, mu_z = zero_step.solve((new_zeros, pole_params, P, Pf))

    if new_poles is None:
        print("  ZPKZeroStep: infeasible")
    else:
        print(f"  ZPKZeroStep: feasible  mu = {mu_z:.4e}")
        print(f"  Recovered poles: {new_poles.to_poles()}")
        print(f"  All poles stable: {new_poles.is_stable()}")


# ── 4. Run ZPKAltMinProblem ────────────────────────────────────────────────────

init_strategy = MixedZPKInit([
    (UniformPoleInit(N_POLES, N_ZEROS, radius_bound=0.9), 0.5),
    (SynthesizeNTFZPKInit(order=N_FILTER,osr=OSR), 0.5),
    #(WarmStartZPKInit(bmi_problem=bmi),0.1)
])

problem = ZPKAltMinProblem(
    bmi_problem=bmi,
    n_poles=N_POLES,
    n_zeros=N_ZEROS,
    init_strategy=init_strategy,
    n_starts=20,
    max_iter=100,
    tol=1e-3,
    n_jobs=10,
    verbose=True)

print("\nRunning ZPK alternating minimization ...")
result = problem.solve(seed=123456)


# ── 5. Inspect results ─────────────────────────────────────────────────────────

print(f"\nFeasible runs : {result.n_feasible} / {len(result.runs)}")

if result.n_feasible == 0:
    print("No feasible solution found.")
    print("Suggestions:")
    print("  - Increase gamma")
    print("  - Increase radius_bound in UniformPoleInit")
    print("  - Increase n_starts or max_iter")
else:
    sol   = result.best_bmi
    zpk_r = result.best_zpk

    print(f"Best mu              : {sol.mu:.4e}")
    print(f"Filter poles         : {zpk_r.poles.to_poles()}")
    print(f"Filter zeros         : {zpk_r.zeros.to_zeros()}")
    print(f"All poles stable     : {zpk_r.poles.is_stable()}")
    print(f"Is proper            : {zpk_r.is_proper()}")

    # Verify LMIs at solution
    Af_s, Bf_s, Cf_s, Df_s = zpk_r.to_filter_matrices()
    A_s, B_s, C_s, D_s = bmi.build_system(Af_s, Bf_s, Cf_s, Df_s)
    L1 = bmi.build_LMI1(sol.P, A_s, B_s)
    L2 = bmi.build_LMI2(sol.P, sol.mu, C_s, D_s)
    L3 = bmi.build_LMI3(sol.Pf, Af_s, Bf_s, Cf_s, Df_s, GAMMA)

    print(f"\nLMI verification at solution:")
    print(f"  LMI1 min eig (> 0): {np.min(np.linalg.eigvalsh(L1)):.4e}")
    print(f"  LMI2 min eig (> 0): {np.min(np.linalg.eigvalsh(L2)):.4e}")
    print(f"  LMI3 max eig (< 0): {np.max(np.linalg.eigvalsh(L3)):.4e}")

    # Closed-loop stability
    cl_poles = np.linalg.eigvals(A_s)
    stable   = np.all(np.abs(cl_poles) < 1.0)
    print(f"Closed-loop stable   : {stable}  "
          f"(max |pole| = {np.max(np.abs(cl_poles)):.4f})")

    # Per-run summary
    print(f"\nPer-run summary:")
    print(f"  {'init':<20} {'iters':>5}  {'converged':>9}  {'final mu':>12}")
    for run in result.runs:

        # Results overview
        if run.zpk_params is not None:
            mu_final = run.residual_history[-1] if run.residual_history else float("inf")
            print(f"  {run.init_label:<20} {run.n_iterations:>5}  "
                  f"{str(run.converged):>9}  {mu_final:>12.4e}")
        else:
            print(f"  {run.init_label:<20} {'—':>5}  {'infeasible':>9}  {'—':>12}")


# ── 6. Plot results ─────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from utils.plot_metrics import bode_plot, residual_history_plot, pole_zero_plot

if result.n_feasible:
    # Run infos in title
    infoTitle = ": "+str(N_POLES)+" Poles,"+str(N_ZEROS)+" Zeros,"+str(bmi.outFilterOrder)+"-th Output filter order"

    # Frequency response
    bode_plot(solution=result.runs, output_filter=bmi._output_filter, title="ZPK Test Frequency Response"+infoTitle, 
              xlim=(1e-4,3), mag_only=False)

    # Plot MSE history
    residual_history_plot(runs=result.runs, title="ZPK Test MSE Plot"+infoTitle)

    # Pole zero plot : best vs others runs
    pole_zero_plot(solution=result.runs, title="ZPK Test Pole Zero Plot"+infoTitle)

    plt.show()