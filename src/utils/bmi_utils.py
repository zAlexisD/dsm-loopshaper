import numpy as np
import matplotlib.pyplot as plt

from altmin.result import AltMinResult
from bmi.problem import BMIProblem
from utils.plot_metrics import bode_plot, residual_history_plot, pole_zero_plot

def inspectBMI(result:AltMinResult):
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
        print(f"P  eigenvalues       : {np.linalg.eigvalsh(sol.P)}")
        print(f"Pf eigenvalues       : {np.linalg.eigvalsh(sol.Pf)}")
        print(f"Filter poles         : {np.linalg.eigvals(sol.Af)}")

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

def graphsBMI(result:AltMinResult,bmi:BMIProblem,FILTER_ORDER:int):
    if result.n_feasible:
        # Run infos in title
        infoTitle = ": Target order "+str(FILTER_ORDER)+", "+str(bmi.outFilterOrder)+"-th Output filter order"

        # Frequency responses
        bode_plot(solution=result.best, output_filter=bmi._output_filter, title="BMI AltMin Frequency Response (best run)"+infoTitle, 
                        xlim=(1e-4,3), mag_only=True, ylim_mag=(-200,20))
        bode_plot(solution=result.runs, output_filter=bmi._output_filter, title="BMI AltMin Frequency Response"+infoTitle, 
                xlim=(1e-4,3), mag_only=True, ylim_mag=(-200,20))

        # Plot MSE history
        residual_history_plot(runs=result.runs, title="BMI AltMin MSE Plot"+infoTitle)

        # Pole zero plot : best vs others runs
        pole_zero_plot(solution=result.runs, title="BMI AltMin Pole Zero Plot"+infoTitle)

        plt.show()