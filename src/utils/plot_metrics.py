"""
loopshaper/utils/plot_metrics.py

Plotting utilities for LoopShaper feedback-filter solutions.

    bode_plot(solution, output_filter=None, ...)
    residual_history_plot(runs, ...)
    pole_zero_plot(solution, ...)

`solution` accepted shapes (same convention across all three functions):
  - a single BMISolution, or an aggregate exposing `.best_bmi`/`.best`
    (ZPKResult / AltMinResult)                    -> treated as "the best".
  - a list of runs (ZPKRun / AltMinRun / ...), each exposing `.bmi_solution`
    or `.solution`                                 -> every run is used; the
                                                       best one (lowest `mu`
                                                       among feasible runs) is
                                                       identified automatically.

Everything funnels through one adapter, `_get_bmi`, which walks at most one
level of nesting (run -> .bmi_solution/.solution, or aggregate ->
.best_bmi/.best) to find the underlying BMISolution's Af/Bf/Cf/Df state-space
matrices. There is no deeper hierarchy-walking than that.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal
from adjustText import adjust_text

from bmi.solution import BMISolution
from zpk.altmin.result import ZPKResult,ZPKRun
from altmin.result import AltMinResult,AltMinRun


# ------------------------------------------------------------------------- #
# Core adapter
# ------------------------------------------------------------------------- #

def _get_bmi(obj: Any) -> BMISolution:
    """
    Extract a BMISolution from `obj`.

    Accepts:
      - `obj` itself, if already a BMISolution.
      - a run object exposing `.bmi_solution` (ZPKRun) or `.solution` (AltMinRun).
      - an aggregate exposing `.best_bmi` (ZPKResult) or `.best` (AltMinResult).
    """
    if isinstance(obj, BMISolution):
        return obj

    if not isinstance(obj,(ZPKRun,ZPKResult,AltMinRun,AltMinResult)):
        raise TypeError(
            f"{type(obj).__name__} object not accepted. Expected ZPKRun ZPKResult AltMinRun AltMinResult")
    
    for attr in ("bmi_solution", "solution", "best_bmi", "best"):
        nested = getattr(obj, attr, None)
        if nested is not None:
            return _get_bmi(nested)
        
    raise TypeError(
        f"Can't find a BMISolution on object of type {type(obj).__name__}. "
        f"Expected a BMISolution through .bmi_solution / .solution / .best_bmi / .best.")


def _to_dlti(obj: Any, dt: float = 1.0):
    """
    Build a discrete LTI system.

    If `obj` is already a scipy.signal system, or a raw (z, p, k) / (A, B, C, D)
    / (num, den) tuple, it's used directly (useful for passing a raw reference
    `output_filter`). Otherwise `obj` is resolved via `_get_bmi` and built from
    its Af/Bf/Cf/Df state-space matrices.
    """
    if isinstance(obj, signal.lti):
        return obj.to_discrete(dt=dt)
    if isinstance(obj, signal.dlti):
            return obj
    if isinstance(obj, tuple):
        if len(obj) == 3:
            z, p, k = obj
            return signal.ZerosPolesGain(z, p, k, dt=dt)
        if len(obj) == 4:
            A, B, C, D = obj
            return signal.StateSpace(A, B, C, D, dt=dt)
        if len(obj) == 2:
            num, den = obj
            return signal.TransferFunction(num, den, dt=dt)
    bmi = _get_bmi(obj)
    if bmi.feasible:
        return signal.StateSpace(bmi.Af, bmi.Bf, bmi.Cf, bmi.Df, dt=dt)
    else:
        return False


def _is_feasible(obj: Any) -> bool:
    try:
        bmi = _get_bmi(obj)
    except TypeError:
        return True
    return getattr(bmi, "feasible", True) is not False


def _mu_of(obj: Any) -> float:
    try:
        bmi = _get_bmi(obj)
    except TypeError:
        return float("inf")
    return getattr(bmi, "mu", float("inf"))


def _run_label(run: Any, index: int, showSeed: bool = False) -> str:
    if hasattr(run, "init_label") and hasattr(run, "seed"):
        base = getattr(run, "init_label") or f"Run {index}"
        if showSeed:
            return f"Run {index} ({base}\n seed={run.seed})"
        else:
            return f"Run {index} ({base})"
    return f"Run {index}"


def _best_run_index(runs: Sequence[Any]) -> int:
    """Index of the best run: lowest `mu` among feasible runs, falling back
    to lowest `mu` overall if none are feasible."""
    feasible_idxs = [i for i, r in enumerate(runs) if _is_feasible(r)]
    pool = feasible_idxs if feasible_idxs else range(len(runs))
    return min(pool, key=lambda i: _mu_of(runs[i]))


# ------------------------------------------------------------------------- #
# Bode plot
# ------------------------------------------------------------------------- #

def bode_plot(
    solution: Union[Any, list],
    output_filter: Optional[Any] = None,
    output_filter_label: str = "Output filter (reference)",
    fs: float = 1.0,
    n_points: int = 2**13,
    db: bool = True,
    mag_only: bool = False,
    unwrap_phase: bool = True,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
    xlim: Optional[tuple] = None,
    ylim_mag: Optional[tuple] = None,
    ylim_phase: Optional[tuple] = None,
):
    """
    Discrete-time Bode plot (magnitude + phase) of a LoopShaper feedback filter.

    Parameters
    ----------
    solution : BMISolution/aggregate, or list of runs
        A single BMISolution (or ZPKResult/AltMinResult aggregate) plots one
        curve labeled "Best". A list of runs (ZPKRun/AltMinRun/...) plots
        every feasible run, with the best one (lowest `mu`) tagged "(best)"
        in the legend; infeasible runs are skipped.
    output_filter : object, optional
        Reference/target output filter overlaid as a dashed black curve, to
        visually verify the designed filter's correctness. Accepts anything
        `_to_dlti` understands (a BMISolution, a scipy.signal system, or a
        raw (z,p,k)/(A,B,C,D)/(num,den) tuple).
    fs : float, default 1.0
        Sample rate, converts frequency axis to Hz. Leave at 1.0 to plot
        against normalized frequency (f / fs, range [0, 0.5]).
    n_points : int, default 1024
        Number of frequency points, log-spaced from 1e-4 rad/s up to the
        Nyquist frequency (pi/dt) — mirrors the MATLAB
        `logspace(-4, log10(pi/ts), n)` convention.
    db : bool, default True
        Plot magnitude in dB if True, else linear magnitude.
    unwrap_phase : bool, default True
        Unwrap phase to avoid artificial +-180 deg jumps.
    xlim, ylim_mag, ylim_phase : tuple, optional
        Axis limits. `xlim` is shared between the magnitude and phase axes
        (in the same units as the x-axis: Hz if `fs` is set, else f/fs).

    Returns
    -------
    fig, (ax_mag, ax_phase)
    """
    dt = 1.0 / fs
    # log-spaced grid from 1e-4 rad/s up to the Nyquist frequency (pi/dt)
    frq = np.logspace(-4, np.log10(np.pi / dt), n_points)  # rad/s
    w_norm = frq * dt  # dbode's `w` is normalized to Nyquist = pi
    f = frq / (2 * np.pi)  # Hz (or normalized cycles/sample if fs == 1.0)

    def mag_phase(sys_dlti):
        _, mag, phase = signal.dbode(sys_dlti, w=w_norm)  # mag already in dB, phase in deg
        if not db:
            mag = 10 ** (mag / 20)
        if unwrap_phase:
            phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))
        return mag, phase

    if mag_only:
        fig, ax_mag = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=figsize or (10, 8))

    if output_filter is not None:
        ref_mag, ref_phase = mag_phase(_to_dlti(output_filter,dt=dt))
        ax_mag.semilogx(frq, ref_mag, "r--", linewidth=1.5, label=output_filter_label)
        if not mag_only:
            ax_phase.semilogx(frq, ref_phase, "r--", linewidth=1.5, label=output_filter_label)

    if isinstance(solution, list):
        best_idx = _best_run_index(solution)
        for i, run in enumerate(solution):
            if _is_feasible(run):
                mag, phase = mag_phase(_to_dlti(run, dt=dt))
                label = _run_label(run, i + 1) + (" (best)" if i == best_idx else "")
                ax_mag.semilogx(frq, mag, label=label)
                if not mag_only:
                    ax_phase.semilogx(frq, phase, label=label)
            else:
                continue
        ax_mag.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        mag, phase = mag_phase(_to_dlti(solution, dt=dt))
        ax_mag.semilogx(frq, mag, label="Best")
        if not mag_only:
            ax_phase.semilogx(frq, phase, label="Best")
        ax_mag.legend(fontsize=8)

    mag_ylabel = "Magnitude [dB]" if db else "Magnitude"
    xlabel = "Frequency [Hz]" if fs != 1.0 else "Normalized frequency (f / fs)"

    ax_mag.set_ylabel(mag_ylabel)
    ax_mag.grid(True, which="both", alpha=0.3)

    if not mag_only:
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_xlabel(xlabel)
        ax_phase.grid(True, which="both", alpha=0.3)

    if xlim:
        ax_mag.set_xlim(xlim)
        if not mag_only:
            ax_phase.set_xlim(xlim)
    if ylim_mag:
        ax_mag.set_ylim(ylim_mag)
    if ylim_phase and not mag_only:
        ax_phase.set_ylim(ylim_phase)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if mag_only:
        output = fig, ax_mag
    else:
        output = fig, (ax_mag, ax_phase)
    return output


# ------------------------------------------------------------------------- #
# Residual (mu) history plot
# ------------------------------------------------------------------------- #

def residual_history_plot(
    runs: Union[Any, list],
    log_scale: bool = True,
    title: Optional[str] = "Residual (mu) convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Residual (mu)",
    figsize: Optional[tuple] = None,
    mark_final: bool = True,
):
    """
    Plot the residual/mu convergence history of every run in `runs`.

    Parameters
    ----------
    runs : list of ZPKRun/AltMinRun (or a single one)
        Each must expose `.residual_history` (ZPKRun) or `.mu_history`
        (AltMinRun). The best run (lowest `mu`) is tagged "(best)".
    log_scale : bool, default True
        Log-scale y-axis (residuals typically span orders of magnitude).
    mark_final : bool, default True
        Annotate each curve's final value.

    Returns
    -------
    fig, ax
    """
    if not isinstance(runs, list):
        runs = [runs]

    fig, ax = plt.subplots(figsize=figsize or (8, 6))
    best_idx = _best_run_index(runs)

    # Marker list to avoid text overlapping
    texts = []

    for i, run in enumerate(runs):
        hist = getattr(run, "residual_history", None)
        if hist is None:
            hist = getattr(run, "mu_history", None)
        if hist is None:
            continue
        hist = np.asarray(hist, dtype=float)
        iterations = np.arange(1, len(hist) + 1)
        label = _run_label(run, i + 1) + (" (best)" if i == best_idx else "")
        line, = ax.plot(iterations, hist, marker="o", markersize=3, label=label)
        if mark_final and len(hist) > 0:
            t = ax.text(
                iterations[-1], hist[-1], f"{hist[-1]:.4g}", 
                fontsize=8, color=line.get_color(), #va="center"
            )
            texts.append(t)

    # Auto adjust marker positionning
    if texts:
        adjust_text(
            texts, 
            ax=ax,
            #only_move={'text': 'y'},  # Keeps labels aligned at the end of the lines (X-axis stays put)
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5) # Optional: draws line if text moves far
        )

    if log_scale:
        ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5,-0.15), ncols=5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ------------------------------------------------------------------------- #
# Pole-zero plot
# ------------------------------------------------------------------------- #

def _scatter_zpk(ax, zpkparams, label: Optional[str] = None, alpha: float = 1.0, size: float = 70):
    """Scatter poles (x) and zeros (o, unfilled) of a ZerosPolesGain, both in
    the same auto-assigned color."""
    poles_scatter = ax.scatter(
        zpkparams.get_poles.real, zpkparams.get_poles.imag, marker="x", s=size, alpha=alpha,
        linewidths=1.5, label=f"{label} poles" if label else "poles",
    )
    if len(zpkparams.get_zeros):
        color = poles_scatter.get_facecolor()[0]
        ax.scatter(
            zpkparams.get_zeros.real, zpkparams.get_zeros.imag, marker="o", s=size, alpha=alpha,
            facecolors="none", edgecolors=[color], linewidths=1.5,
            label=f"{label} zeros" if label else "zeros",
        )


def _draw_unit_circle(ax):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_aspect("equal")


def _finish_pz_axes(ax, title=None, legend=True, ncols:int=1):
    _draw_unit_circle(ax)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10)
    if legend:
        if title == "Best":
            ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                       borderaxespad=0.0)
        else:
            ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0.0, ncols=ncols)


def pole_zero_plot(
    solution: Union[Any, list],
    fs: float = 1.0,
    title: Optional[str] = None,
    figsize: Optional[tuple] = None,
):
    """
    Plot poles (x) / zeros (o) on the complex plane, with the discrete-time
    unit circle for reference.

    Parameters
    ----------
    solution : BMISolution/aggregate, or list of runs
        A single BMISolution (or ZPKResult/AltMinResult aggregate) draws one
        figure with just the best filter's poles/zeros.
        A list of runs (ZPKRun/AltMinRun/...) draws two subplots side by
        side: left = best run, right = every other run.
    fs : float, default 1.0
        Sample rate, used to build the discrete system (dt = 1/fs); pole/zero
        placement relative to the unit circle doesn't depend on `fs`.

    Returns
    -------
    fig, ax           (single BMISolution/aggregate input)
    fig, (ax_best, ax_others)   (list-of-runs input)
    """
    dt = 1.0 / fs

    if isinstance(solution, list):
        runs = solution
        best_idx = _best_run_index(runs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (10, 6))

        zpk_best = runs[best_idx].zpk_params
        _scatter_zpk(ax1, zpk_best, label=_run_label(runs[best_idx], best_idx + 1))

        for i, run in enumerate(runs):
            if i == best_idx or not _is_feasible(run):
                continue
            zpk = run.zpk_params
            _scatter_zpk(ax2, zpk, label=_run_label(run, i + 1), alpha=0.6, size=45)

        _finish_pz_axes(ax1, title="Best")

        # Adapt legend columns according to amount of feasible runs
        n_cols = 2 if len(runs) >=10 else 1
        _finish_pz_axes(ax2, title="Other runs",ncols=n_cols)

        if title:
            fig.suptitle(title)
        fig.subplots_adjust(right=0.85, wspace=0.4, top=0.95, left=0.1)
        return fig, ax1, ax2

    # Single BMISolution / aggregate -> one figure, best only
    fig, ax = plt.subplots(figsize=figsize or (6, 6))
    zpk = _to_dlti(solution, dt=dt).to_zpk()
    _scatter_zpk(ax, zpk, label="Best")
    _finish_pz_axes(ax, title=title)
    fig.subplots_adjust(right=0.72)
    return fig, ax


# ------------------------------------------------------------------------- #
# Quick self-test / usage demo
# ------------------------------------------------------------------------- #

if __name__ == "__main__":

    def _rand_bmi(seed, feasible=True):
        rng = np.random.default_rng(seed)
        Af = np.diag([0.5, 0.6, 0.7]) + 0.02 * rng.standard_normal((3, 3))
        Bf = np.array([[1.0], [0.0], [0.0]])
        Cf = rng.standard_normal((1, 3)) * 0.5
        Df = np.array([[0.8]])
        return BMISolution(Af, Bf, Cf, Df, mu=abs(rng.standard_normal()), feasible=feasible)

    runs = [
        ZPKRun(_rand_bmi(1), [1.0, 0.5, 0.3, 0.15, 0.09], "SynthesizeNTFInit", seed=1),
        ZPKRun(_rand_bmi(2, feasible=True), [1.2, 0.8, 0.4, 0.1, 0.05], "WarmStartInit", seed=2),
        ZPKRun(_rand_bmi(3, feasible=False), [2.0, 1.6, 1.5], "MixedInit", seed=3),
    ]
    zpk_result = ZPKResult(best_bmi=min(
        (r.bmi_solution for r in runs if r.bmi_solution.feasible), key=lambda b: b.mu
    ), runs=runs)

    ref_filter = signal.ZerosPolesGain(
        [0.99 + 0.02j, 0.99 - 0.02j], [0.5, 0.55, 0.6], 0.85, dt=1.0
    )

    # Bode: best only, vs. reference
    bode_plot(zpk_result, output_filter=ref_filter, title="Demo — best vs reference")
    # Bode: every run, best tagged, vs. reference
    bode_plot(runs, output_filter=ref_filter, title="Demo — all runs")

    # Residual history: every run
    residual_history_plot(runs, title="Demo — residual history")

    # Pole-zero: aggregate -> single figure, best only
    pole_zero_plot(zpk_result, title="Demo — best poles/zeros")
    # Pole-zero: list of runs -> best vs others, side by side
    pole_zero_plot(runs, title="Demo — best vs other runs")

    plt.show()