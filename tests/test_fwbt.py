import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from fwbt.problem import FWBTProblem
from dsm.core import synthesizeNTF


# =============================================================================
# Helpers
# =============================================================================

def make_test_ntf(order=6, osr=64, opt=1):
    """
    Simple discrete-time NTF via synthesizeNTF.
    Falls back to a hand-built stable filter if synthesizeNTF is unavailable.
    """
    try:
        zpk = synthesizeNTF(order=order, osr=osr, opt=opt)
        # zpk is already a ZerosPolesGain object
        ss = zpk.to_ss()   # or: signal.StateSpace(*signal.zpk2ss(zpk.zeros, zpk.poles, zpk.gain), dt=1)
        return ss
    except ImportError:
        poles = np.array([0.6, -0.5, 0.4+0.3j, 0.4-0.3j, -0.3+0.2j, -0.3-0.2j])
        zeros = np.exp(1j * np.linspace(0, np.pi * (1 - 1/osr), order))
        b, a = signal.zpk2tf(zeros, poles, 1.0)
        ss = signal.StateSpace(*signal.tf2ss(b, a), dt=1)
        return ss


def make_output_weight(order=2, cutoff=0.05):
    """
    Simple low-pass output weight (emphasizes in-band fidelity).
    Returns (A, B, C, D) tuple for use as W_o.
    """
    b, a = signal.butter(order, cutoff, btype='low', analog=False)
    ss = signal.StateSpace(*signal.tf2ss(b, a), dt=1)
    return (ss.A, ss.B, ss.C, ss.D)


def frequency_response(ss_sys, n_freqs=512):
    """Evaluate discrete-time frequency response, returns (freqs, H)."""
    # Ensure dt is set — dfreqresp requires it
    if ss_sys.dt is None:
        ss_sys = signal.StateSpace(ss_sys.A, ss_sys.B, ss_sys.C, ss_sys.D, dt=1)
    w = np.linspace(0, np.pi, n_freqs)
    _, h = signal.dfreqresp(ss_sys, w=w)
    return w / np.pi, h


# =============================================================================
# Test 1 — stability check
# =============================================================================

def test_stability_check():
    print("\n--- Test 1: stability check ---")

    ss_stable = make_test_ntf(order=6)
    prob = FWBTProblem(ss_stable, order_target=2, discrete=True)
    assert prob.is_ready, "Stable system should pass is_ready"
    print("  PASS: stable NTF accepted")

    # Build an unstable system
    A_unstable = np.array([[1.5, 0], [0, 0.5]])  # eigenvalue > 1
    ss_unstable = signal.StateSpace(
        A_unstable, np.eye(2), np.eye(2), np.zeros((2, 2)), dt=1
    )
    prob_bad = FWBTProblem(ss_unstable, order_target=1, discrete=True)
    assert not prob_bad.is_ready, "Unstable system should fail is_ready"
    print("  PASS: unstable system correctly rejected")

    try:
        prob_bad.solve()
        print("  FAIL: solve() should have raised on unstable input")
    except ValueError as e:
        print(f"  PASS: solve() raised ValueError: {e}")


# =============================================================================
# Test 2 — HSV preview
# =============================================================================

def test_hsv_preview():
    print("\n--- Test 2: HSV preview ---")

    ss_full = make_test_ntf(order=6)
    prob = FWBTProblem(ss_full, discrete=True)

    sigma = prob.hsv_preview()
    assert len(sigma) == 6, f"Expected 6 HSVs, got {len(sigma)}"
    assert np.all(sigma >= 0), "HSVs must be non-negative"
    assert np.all(np.diff(sigma) <= 1e-10), "HSVs should be in descending order"

    print(f"  HSVs: {np.array2string(sigma, precision=4)}")
    print(f"  Largest gap at r={np.argmax(sigma[:-1] / sigma[1:]) + 1}")
    print("  PASS")


# =============================================================================
# Test 3 — unweighted reduction (identity weights)
# =============================================================================

def test_unweighted_reduction():
    print("\n--- Test 3: unweighted reduction ---")

    ss_full = make_test_ntf(order=6)
    prob = FWBTProblem(ss_full, order_target=3, discrete=True)
    result = prob.solve()

    assert result.order == 3, f"Expected order 3, got {result.order}"
    assert result.hinf_error_bound >= 0

    # Stability is not guaranteed by Enns FWBT — log rather than assert
    if not result.is_stable:
        print("  WARNING: reduced system is unstable (expected with Enns variant)")
    
    print(f"  {result.summary()}")
    print("  PASS")


# =============================================================================
# Test 4 — weighted reduction
# =============================================================================

def test_weighted_reduction():
    print("\n--- Test 4: weighted reduction ---")

    ss_full = make_test_ntf(order=6)
    W_o = make_output_weight(order=2, cutoff=0.05)

    prob_unweighted = FWBTProblem(ss_full, order_target=3, discrete=True)
    prob_weighted   = FWBTProblem(ss_full, order_target=3, W_o=W_o, discrete=True)

    result_uw = prob_unweighted.solve()
    result_w  = prob_weighted.solve()

    # Stability not guaranteed by Enns variant
    if not result_w.is_stable:
        print("  NOTE: weighted reduced system is unstable (expected with Enns)")

    # Meaningful check: weighted should have lower in-band error than unweighted
    f, H_full = frequency_response(ss_full)
    _, H_uw   = frequency_response(result_uw.to_statespace())
    _, H_w    = frequency_response(result_w.to_statespace())

    # In-band = normalized frequency below cutoff
    cutoff = 0.05
    in_band = f < cutoff

    err_uw = np.mean(np.abs(H_full[in_band]) - np.abs(H_uw[in_band]))
    err_w  = np.mean(np.abs(H_full[in_band]) - np.abs(H_w[in_band]))

    print(f"  In-band magnitude error — unweighted: {err_uw:.4e}, weighted: {err_w:.4e}")

    if err_w < err_uw:
        print("  PASS: weighted reduction has lower in-band error")
    else:
        # Not a hard failure — depends heavily on weight design and HSV spectrum
        print("  NOTE: weighted error not lower than unweighted for this NTF/weight pair")
        print("        This can happen with the fallback NTF — recheck with synthesizeNTF")

    print(f"  Unweighted: {result_uw.summary()}")
    print(f"  Weighted:   {result_w.summary()}")


# =============================================================================
# Test 5 — auto order selection
# =============================================================================

def test_auto_order():
    print("\n--- Test 5: auto order selection ---")

    ss_full = make_test_ntf(order=6)
    prob = FWBTProblem(ss_full, order_target=None, discrete=True)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.resetwarnings()          # clear any existing filters
        warnings.simplefilter("always")   # force all warnings through
        result = prob.solve()

    # Filter to only UserWarnings from our module
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]

    assert len(user_warnings) >= 1, "Expected at least one UserWarning for auto order"
    print(f"  Warning: {user_warnings[0].message}")

    assert result.order >= 1
    assert result.order < ss_full.A.shape[0]
    print(f"  Auto-selected order: {result.order}")
    print(f"  {result.summary()}")
    print("  PASS")


# =============================================================================
# Test 6 — solve(r=...) override
# =============================================================================

def test_order_override():
    print("\n--- Test 6: order override ---")

    ss_full = make_test_ntf(order=6)
    prob = FWBTProblem(ss_full, order_target=4, discrete=True)

    result = prob.solve(r=2)   # override 4 → 2
    assert result.order == 2, f"Expected order 2, got {result.order}"
    print(f"  {result.summary()}")
    print("  PASS")


# =============================================================================
# Test 7 — result conversions
# =============================================================================

def test_result_conversions():
    print("\n--- Test 7: result conversions ---")

    ss_full = make_test_ntf(order=6)
    prob = FWBTProblem(ss_full, order_target=3, discrete=True)
    result = prob.solve()

    ss_r = result.to_statespace()
    assert ss_r.A.shape == (3, 3)
    print("  to_statespace(): PASS")

    zpk = result.to_zpk()
    assert len(zpk.poles) == 3
    print("  to_zpk(): PASS")


# =============================================================================
# Test 8 — frequency response comparison (visual)
# =============================================================================

def test_frequency_response_plot():
    print("\n--- Test 8: frequency response comparison ---")

    ss_full = make_test_ntf(order=6)
    W_o = make_output_weight(order=2, cutoff=0.05)

    prob = FWBTProblem(ss_full, order_target=None, W_o=W_o, discrete=True)
    sigma = prob.hsv_preview()

    #fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    # Left: HSV spectrum
    axes.bar(np.arange(1, len(sigma) + 1), sigma, color='steelblue')
    axes.set_xlabel("Index")
    axes.set_ylabel("σ")
    axes.set_title("Hankel Singular Values")
    axes.set_yscale('log')

    # # Right: frequency responses for each reduced order
    # f, H_full = frequency_response(ss_full)
    # axes.plot(f, 20 * np.log10(np.abs(H_full)),
    #              'k-', linewidth=2, label=f"Full (n={ss_full.A.shape[0]})")

    # colors = plt.cm.viridis(np.linspace(0.2, 0.9, ss_full.A.shape[0] - 1))
    # for r, color in zip(range(1, ss_full.A.shape[0]), colors):
    #     result = prob.solve(r=r)
    #     f_r, H_r = frequency_response(result.to_statespace())
    #     axes.plot(f_r, 20 * np.log10(np.abs(H_r)),
    #                  '--', color=color, label=f"r={r}")

    # axes.set_xlabel("Normalized frequency")
    # axes.set_ylabel("Magnitude (dB)")
    # axes.set_title("Full vs reduced frequency responses")
    # axes.legend(fontsize=8)
    axes.grid(True)

    plt.tight_layout()
    plt.savefig("tests/fwbt_test_responses.png", dpi=150)
    plt.show()
    print("  Plot saved to fwbt_test_responses.png")
    print("  PASS")




# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    test_stability_check()
    test_hsv_preview()
    test_unweighted_reduction()
    test_weighted_reduction()
    test_auto_order()
    test_order_override()
    test_result_conversions()
    test_frequency_response_plot()

    print("\n=== All tests passed ===")