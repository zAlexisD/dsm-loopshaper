"""
Filename: main.py (Simulation Runner)
Author: Alexis DOAN
Date: 2026-08-09
Version: 1.0.0

Description:
    End-to-end simulation runner for the utility toolbox. This script demonstrates 
    the practical execution and integration of all core toolbox components. 

Simulated Workflows:
    1. AltMin Algorithm applied directly on State Space representation for lower order design
    2. AltMin Algorithm applied on poles and zeros for lower order design
    3. FWBT applied to design lower order initilizations for AltMin
    4. FWBT applied to reduce order of full order design using AltMin

Usage:
    Run directly from the root directory to execute all simulations sequentially:
    $ py main.py
"""
# =================================================================================================
# Common imports and values for simulations
# =================================================================================================
from bmi.problem import BMIProblem

GAMMA    = 1.5
OSR      = 32

# =================================================================================================
# AltMin applied on State-Space Representation
# =================================================================================================
# from altmin.init import LHSInit, ZeroInit, MixedInit
# from bmi.altmin.init import SynthesizeNTFInit,WarmStartInit
# from bmi.altmin.problem import BMIAltMinProblem

# FILTER_ORDER = 2
# N_INPUTS     = 1
# N_OUTPUTS    = 1

# bmi = BMIProblem(
#     n_filter=FILTER_ORDER,
#     n_input=N_INPUTS,
#     n_output=N_OUTPUTS,
#     gamma=GAMMA)

# init_bmi = MixedInit([
#     (LHSInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS, bound=0.5), 0.1),
#     (ZeroInit(FILTER_ORDER, N_INPUTS, N_OUTPUTS), 0.1),
#     (SynthesizeNTFInit(FILTER_ORDER, osr=OSR), 0.6),
#     (WarmStartInit(bmi_problem=bmi),0.2)
# ])

# problem = BMIAltMinProblem(
#     bmi_problem=bmi,
#     osr=OSR,
#     filter_order=FILTER_ORDER,
#     n_inputs=N_INPUTS,
#     n_outputs=N_OUTPUTS,
#     init_strategy=init_bmi,
#     n_starts=10,
#     max_iter=100,
#     tol=1e-3,
#     n_jobs=10,
#     verbose=True)

# print("\nRunning BMI alternating minimization ...")
# bmi_result = problem.solve(seed=42)

# =================================================================================================
# AltMin applied on Poles and Zeros
# =================================================================================================
from zpk.altmin.problem import ZPKAltMinProblem
from zpk.altmin.init import UniformPoleInit, MixedZPKInit, SynthesizeNTFZPKInit, WarmStartZPKInit
from utils.zpk_utils import inspectZPK,graphsZPK

N_POLES  = 2     
N_ZEROS  = 2
N_FILTER = max(N_POLES,N_ZEROS)

zpk_bmi = BMIProblem(
    n_filter=N_FILTER,
    n_input=1,
    n_output=1,
    gamma=GAMMA)

init_zpk = MixedZPKInit([
    (UniformPoleInit(N_POLES, N_ZEROS, radius_bound=0.9), 0.4),
    (SynthesizeNTFZPKInit(order=N_FILTER,osr=OSR), 0.5),
    (WarmStartZPKInit(bmi_problem=zpk_bmi),0.1)
    ])

problem = ZPKAltMinProblem(
    bmi_problem=zpk_bmi,
    n_poles=N_POLES,
    n_zeros=N_ZEROS,
    init_strategy=init_zpk,
    n_starts=20,
    max_iter=100,
    tol=1e-3,
    n_jobs=20,
    tol_nontrivial=0.5,
    verbose=True)

print("\nRunning ZPK alternating minimization ...")
zpk_result = problem.solve(seed=143297)

inspectZPK(zpk_result)
graphsZPK(result=zpk_result,N_POLES=N_POLES,N_ZEROS=N_ZEROS,bmi=zpk_bmi)


# =================================================================================================
# Full order design via Pole Zero AltMin then reduce it order with FWBT
# =================================================================================================
# from zpk.altmin.problem import ZPKAltMinProblem
# from zpk.altmin.init import UniformPoleInit, MixedZPKInit, SynthesizeNTFZPKInit, WarmStartZPKInit
# from utils.zpk_utils import inspectZPK,graphsZPK
# from fwbt.problem import FWBTProblem
# from bmi.solution import BMISolution
# from utils.plot_metrics import bode_plot,pole_zero_plot

# # Generate a Full order Pole Zero AltMin Design
# N_POLES  = 4     
# N_ZEROS  = 2
# N_FILTER = max(N_POLES,N_ZEROS)
# DESIRED_ORDER = 2

# fwbt_bmi = BMIProblem(
#     n_filter=N_FILTER,
#     n_input=1,
#     n_output=1,
#     gamma=GAMMA)

# init_fwbt = MixedZPKInit([
#     (UniformPoleInit(N_POLES, N_ZEROS, radius_bound=0.9), 0.6),
#     (SynthesizeNTFZPKInit(order=N_FILTER,osr=OSR), 0.5),
#     (WarmStartZPKInit(bmi_problem=fwbt_bmi),0.0)
#     ])

# problem = ZPKAltMinProblem(
#     bmi_problem=fwbt_bmi,
#     n_poles=N_POLES,
#     n_zeros=N_ZEROS,
#     init_strategy=init_fwbt,
#     n_starts=20,
#     max_iter=100,
#     tol=1e-3,
#     n_jobs=10,
#     tol_nontrivial=0.5,
#     verbose=True)

# print("\nRunning ZPK alternating minimization ...")
# fwbt_result = problem.solve(seed=143297)

# inspectZPK(fwbt_result)
# graphsZPK(fwbt_result,N_POLES,N_ZEROS,fwbt_bmi)

# # From this design, reduce the order with FWBT
# zpk_runs = fwbt_result.runs
# fwbt_runs = []

# for run in zpk_runs:
#     ss_run = run.bmi_solution.to_statespace()
#     prob = FWBTProblem(ss_run, order_target=DESIRED_ORDER, 
#                        W_o=fwbt_bmi._output_filter, discrete=True)
#     result = prob.solve()
#     if result.is_stable:
#         trunc_bmi = BMISolution(
#             Af=result.A_r,Bf=result.B_r,Cf=result.C_r,Df=result.D_r,feasible=True)
#         fwbt_runs.append(trunc_bmi)

# title = f"Zero Pole AltMin order {fwbt_bmi.n_f} truncated to order {DESIRED_ORDER}"

# bode_plot(solution=fwbt_runs,output_filter=fwbt_bmi._output_filter,mag_only=True,
#           title="Magnitude Response: "+title,xlim=(1e-4,3), ylim_mag=(-200,20))

# pole_zero_plot(solution=fwbt_runs, title="PZ plot: "+title)