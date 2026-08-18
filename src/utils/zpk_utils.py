import matplotlib.pyplot as plt

from zpk.altmin.result import ZPKResult
from bmi.problem import BMIProblem
from utils.plot_metrics import bode_plot, residual_history_plot, pole_zero_plot

def inspectZPK(result:ZPKResult):
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

def graphsZPK(result:ZPKResult,N_POLES,N_ZEROS,bmi:BMIProblem):
    if result.n_feasible:
        # Run infos in title
        infoTitle = ": "+str(N_POLES)+" Poles,"+str(N_ZEROS)+" Zeros,"+str(bmi.outFilterOrder)+"-th Output filter order"

        # Frequency responses
        bode_plot(solution=result.best_bmi, output_filter=bmi._output_filter, title="ZPK AltMin Frequency Response (best run)"+infoTitle, 
                        xlim=(1e-4,3), mag_only=True, ylim_mag=(-200,20))
        bode_plot(solution=result.runs, output_filter=bmi._output_filter, title="ZPK AltMin Frequency Response"+infoTitle, 
                xlim=(1e-4,3), mag_only=True, ylim_mag=(-200,20))

        # Plot MSE history
        residual_history_plot(runs=result.runs, title="ZPK AltMin MSE Plot"+infoTitle)

        # Pole zero plot : best vs others runs
        pole_zero_plot(solution=result.runs, title="ZPK AltMin Pole Zero Plot"+infoTitle)

        plt.show()