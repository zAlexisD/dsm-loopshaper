"""
dsm_core.py — self-contained DSM simulation primitives
# TODO: write clean doctring
"""
import numpy as np
from scipy import signal
from scipy.signal import zpk2tf, ZerosPolesGain, zpk2ss 
from scipy.optimize import minimize
import warnings

# ---------------------------------------------------------------------------
# Helper: _ds_optzeros
#   Returns the zeros which minimize the in-band noise power of
#   a delta-sigma modulator's NTF.
#   Python translation of _ds_optzeros from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _ds_optzeros(n, opt=1):
    """
    Returns the zeros which minimize the in-band noise power of
    a delta-sigma modulator's NTF.
    Python translation of _ds_optzeros from Schreier's Delta-Sigma Toolbox.
    """
    if opt == 0:
        opt_zeros = np.zeros(int(np.ceil(n / 2)))
    else:
        if n == 1:
            opt_zeros = np.array([0.0])
        elif n == 2:
            if opt == 1:
                opt_zeros = np.array([np.sqrt(1/3)])
            else:
                opt_zeros = np.array([0.0])
        elif n == 3:
            opt_zeros = np.array([np.sqrt(3/5), 0.0])
        elif n == 4:
            if opt == 1:
                discr = np.sqrt(9/49 - 3/35)
                tmp = 3/7
                opt_zeros = np.sqrt([tmp + discr, tmp - discr])
            else:
                opt_zeros = np.array([0.0, np.sqrt(5/7)])
        elif n == 5:
            discr = np.sqrt(25/81 - 5/21)
            tmp = 5/9
            opt_zeros = np.sqrt([tmp + discr, tmp - discr, 0])
        elif n == 6:
            if opt == 1:
                opt_zeros = np.array([0.23862059, 0.66120988, 0.9324696])
            else:
                discr = np.sqrt(56) / 33
                tmp = 7/11
                opt_zeros = np.sqrt([0, tmp + discr, tmp - discr])
        elif n == 7:
            opt_zeros = np.array([0.0, 0.40584371, 0.74153078, 0.94910785])
        elif n == 8:
            if opt == 1:
                opt_zeros = np.array([0.18343709, 0.52553345, 0.79666684, 0.96028993])
            else:
                opt_zeros = np.array([0.0, 0.50563161, 0.79017286, 0.95914731])
        elif n == 9:
            opt_zeros = np.array([0.0, 0.32425101, 0.61337056, 0.83603082, 0.9681602])
        elif n == 10:
            if opt == 1:
                opt_zeros = np.array([0.1834370913, 0.5255334458, 0.7966668433, 0.9602899327])
            else:
                opt_zeros = np.array([0.0, 0.41572267, 0.67208682, 0.86238894, 0.97342121])
        elif n == 11:
            opt_zeros = np.array([0.0, 0.26953955, 0.51909468, 0.73015137, 0.88706238, 0.97822864])
        elif n == 12:
            if opt == 1:
                opt_zeros = np.array([0.12523875, 0.36783403, 0.58731921, 0.7699033, 0.90411753, 0.9815607])
            else:
                opt_zeros = np.array([0.0, 0.35222363, 0.58006251, 0.76647993, 0.90281326, 0.98132047])
        elif n == 13:
            opt_zeros = np.array([0.0, 0.23045331, 0.44849063, 0.64234828, 0.8015776, 0.91759824, 0.98418306])
        elif n == 14:
            if opt == 1:
                opt_zeros = np.array([0.10806212, 0.31911586, 0.51525046, 0.68729392, 0.82720185, 0.92843513, 0.98628389])
            else:
                opt_zeros = np.array([0.0, 0.30524384, 0.50836649, 0.6836066, 0.82537239, 0.92772336, 0.98615167])
        else:
            print('Optimized zeros for n>14 are not available.')
            return None

    # Sort the zeros and replicate them
    z = np.sort(opt_zeros)
    opt_zeros_out = np.zeros(n)
    m = 0  # 0-indexed

    if n % 2 == 1:
        opt_zeros_out[0] = z[0]
        z = z[1:]
        m = 1

    for i in range(len(z)):
        opt_zeros_out[m]     =  z[i]
        opt_zeros_out[m + 1] = -z[i]
        m += 2

    return opt_zeros_out

# ---------------------------------------------------------------------------
# Helper: _cplxpair
#   Sort complex array into conjugate pairs (mimics MATLAB's _cplxpair)
# ---------------------------------------------------------------------------
def _cplxpair(a, tol=1e-6):
    """
    Sort complex array into conjugate pairs, real values last.
    Mimics MATLAB's _cplxpair behavior.
    """
    a = np.asarray(a, dtype=complex)
    reals = a[np.abs(a.imag) <= tol * np.abs(a)]
    complexes = a[np.abs(a.imag) > tol * np.abs(a)]

    # Pair up conjugates
    paired = []
    used = np.zeros(len(complexes), dtype=bool)
    for i in range(len(complexes)):
        if used[i]:
            continue
        diffs = np.abs(complexes - np.conj(complexes[i]))
        diffs[i] = np.inf
        j = np.argmin(diffs)
        if not used[j] and diffs[j] < tol:
            # Put the one with negative imag part first (MATLAB convention)
            if complexes[i].imag < 0:
                paired.extend([complexes[i], complexes[j]])
            else:
                paired.extend([complexes[j], complexes[i]])
            used[i] = True
            used[j] = True

    reals_sorted = np.sort(reals.real).astype(complex)
    return np.concatenate([np.array(paired), reals_sorted])

# ---------------------------------------------------------------------------
# Helper: _evalRPoly
#   Compute the value of a polynomial which is given in terms of its roots.
#   Python translation of _evalRPoly from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _evalRPoly(roots, x, k=1):
    """
    Compute the value of a polynomial which is given in terms of its roots.
    Python translation of _evalRPoly from Schreier's Delta-Sigma Toolbox.
    """
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    y = k * np.ones(x.shape, dtype=complex)

    roots = np.asarray(roots)
    roots = roots[~np.isinf(roots)]  # remove roots at infinity

    for r in roots:
        y = y * (x - r)

    return y

# ---------------------------------------------------------------------------
# Helper: _evalTF
#   Evaluates the rational transfer function described by tf at the point(s) z.
#   Python translation of _evalTF from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _evalTF(tf, z):
    """
    Evaluates the rational transfer function described by tf at the point(s) z.
    Python translation of _evalTF from Schreier's Delta-Sigma Toolbox.

    tf can be:
      - A scipy.signal.ZerosPolesGain (zpk) object
      - A dict with key 'form':
          'zp'    -> keys: 'zeros', 'poles', 'k'
          'coeff' -> keys: 'num', 'den'
      - A dict without 'form' (assumed zp): keys: 'zeros', 'poles', 'k'
    """
    z = np.atleast_1d(np.asarray(z, dtype=complex))

    # scipy ZerosPolesGain object
    try:
        from scipy.signal import ZerosPolesGain
        if isinstance(tf, ZerosPolesGain):
            h = tf.gain * _evalRPoly(tf.zeros, z) / _evalRPoly(tf.poles, z)
            return h
    except ImportError:
        pass

    # Dict-based tf
    if isinstance(tf, dict):
        if 'form' in tf:
            if tf['form'] == 'zp':
                h = tf['k'] * _evalRPoly(tf['zeros'], z) / _evalRPoly(tf['poles'], z)
            elif tf['form'] == 'coeff':
                h = np.polyval(tf['num'], z) / np.polyval(tf['den'], z)
            else:
                raise ValueError(f"_evalTF: Unknown form: {tf['form']}")
        else:
            # Assume zp form
            h = tf['k'] * _evalRPoly(tf['zeros'], z) / _evalRPoly(tf['poles'], z)
        return h

    raise TypeError(f"_evalTF: Unsupported tf type: {type(tf)}")

# ---------------------------------------------------------------------------
# Helper: _synthesizeNTF0
#   Synthesize a Noise Transfer Function for a delta-sigma modulator.
#   Python translation of _synthesizeNTF0 from Schreier's Delta-Sigma Toolbox.
#   Returns a scipy ZerosPolesGain object with the NTF zeros, poles and gain.
# ---------------------------------------------------------------------------
def _synthesizeNTF0(order, OSR, opt, H_inf, f0):
    """
    Synthesize a Noise Transfer Function for a delta-sigma modulator.
    Python translation of _synthesizeNTF0 from Schreier's Delta-Sigma Toolbox.

    Returns a scipy ZerosPolesGain object with the NTF zeros, poles and gain.
    """

    # --- Determine the zeros ---
    if f0 != 0:  # Bandpass design: halve the order temporarily
        order = order // 2
        dw = np.pi / (2 * OSR)
    else:
        dw = np.pi / OSR

    if np.isscalar(opt) or (hasattr(opt, '__len__') and len(opt) == 1):
        opt = int(np.atleast_1d(opt)[0])
        if opt == 0:
            z = np.zeros(order, dtype=complex)
        else:
            z = dw * _ds_optzeros(order, opt)
            if z is None or len(z) == 0:
                return None

        if f0 != 0:  # Bandpass design: shift and replicate zeros
            order = order * 2
            z = z + 2 * np.pi * f0
            z = np.concatenate([z, -z])

        z = np.exp(1j * z)
    else:
        z = np.asarray(opt, dtype=complex).flatten()

    ntf = ZerosPolesGain(z, np.zeros(order, dtype=complex), 1)
    itn_limit = 100

    # --- Iteratively determine the poles ---
    if f0 == 0:  # Lowpass design
        H_inf_limit = 2 ** order
        if H_inf >= H_inf_limit:
            print('_synthesizeNTF0 warning: Unable to achieve specified Hinf.')
            print('Setting all NTF poles to zero.')
            ntf = ZerosPolesGain(z, np.zeros(order, dtype=complex), 1)
        else:
            x = 0.3 ** (order - 1)  # starting guess
            converged = False
            fprev = 0.0
            delta_x = 0.0

            for itn in range(1, itn_limit + 1):
                me2 = -0.5 * (x ** (2.0 / order))
                w = (2 * np.arange(1, order + 1) - 1) * np.pi / order
                mb2 = 1 + me2 * np.exp(1j * w)
                p = mb2 - np.sqrt(mb2 ** 2 - 1)

                # Reflect poles inside the unit circle
                out = np.abs(p) > 1
                p[out] = 1.0 / p[out]

                ntf = ZerosPolesGain(z, _cplxpair(p), 1)
                f = np.real(_evalTF(ntf, -1)) - H_inf

                if itn == 1:
                    delta_x = -f / 100
                else:
                    delta_x = -f * delta_x / (f - fprev)

                xplus = x + delta_x
                x = xplus if xplus > 0 else x * 0.1
                fprev = f

                if abs(f) < 1e-10 or abs(delta_x) < 1e-10:
                    converged = True
                    break

                if x > 1e6:
                    print('_synthesizeNTF0 warning: Unable to achieve specified Hinf.')
                    print('Setting all NTF poles to zero.')
                    ntf = ZerosPolesGain(z, np.zeros(order, dtype=complex), 1)
                    break

            if itn == itn_limit and not converged:
                print('_synthesizeNTF0 warning: Danger! Iteration limit exceeded.')

    else:  # Bandpass design
        x = 0.3 ** (order / 2 - 1)  # starting guess
        z_inf = 1 if f0 > 0.25 else -1
        c2pif0 = np.cos(2 * np.pi * f0)
        fprev = 0.0
        delta_x = 0.0

        for itn in range(1, itn_limit + 1):
            e2 = 0.5 * x ** (2.0 / order)
            w = (2 * np.arange(1, order + 1) - 1) * np.pi / order
            mb2 = c2pif0 + e2 * np.exp(1j * w)
            p = mb2 - np.sqrt(mb2 ** 2 - 1)

            # Reflect poles inside the unit circle
            out = np.abs(p) > 1
            p[out] = 1.0 / p[out]

            ntf = ZerosPolesGain(z, _cplxpair(p), 1)
            f = np.real(_evalTF(ntf, z_inf)) - H_inf

            if itn == 1:
                delta_x = -f / 100
            else:
                delta_x = -f * delta_x / (f - fprev)

            xplus = x + delta_x
            x = xplus if xplus > 0 else x * 0.1
            fprev = f

            if abs(f) < 1e-10 or abs(delta_x) < 1e-10:
                break

            if x > 1e6:
                print('_synthesizeNTF0 warning: Unable to achieve specified Hinf.')
                print('Setting all NTF poles to zero.')
                ntf = ZerosPolesGain(z, np.zeros(order, dtype=complex), 1)
                break

            if itn == itn_limit:
                print('_synthesizeNTF0 warning: Danger! Iteration limit exceeded.')

    # --- Assemble final NTF ---
    z_sorted = _cplxpair(ntf.zeros)[::-1]
    p_sorted = _cplxpair(ntf.poles)[::-1]

    return ZerosPolesGain(z_sorted, p_sorted, 1)

# ---------------------------------------------------------------------------
# Helper: _padt
#   Pad a matrix x on the top to length n with value val(0)
#   The empty matrix is assumed to be have 1 empty column
# ---------------------------------------------------------------------------
def _padt(x, n, val=0):
    """
    Pad the top of a matrix x with val, so that the result has n rows.
    Python translation of _padt from Schreier's Delta-Sigma Toolbox.
    """
    x = np.atleast_2d(np.asarray(x))
    rows_to_pad = n - x.shape[0]
    cols = max(1, x.shape[1])
    padding = np.full((rows_to_pad, cols), val,dtype=x.dtype)
    return np.vstack([padding, x])

# ---------------------------------------------------------------------------
# Helper: _padb
#   Pad a matrix x on the bottom to length n with value val(0)
#   The empty matrix is assumed to be have 1 empty column
# ---------------------------------------------------------------------------
def _padb(x, n, val=0):
    """
    Pad the bottom of a matrix x with val, so that the result has n rows.
    Python translation of _padb from Schreier's Delta-Sigma Toolbox.
    """
    x = np.atleast_2d(np.asarray(x))
    rows_to_pad = n - x.shape[0]
    cols = max(1, x.shape[1])
    padding = np.full((rows_to_pad, cols), val,dtype=x.dtype)
    return np.vstack([x, padding])

# ---------------------------------------------------------------------------
# Helper: _rmsGain
#   Compute the root mean-square gain of the discrete-time tf H in the 
#   frequency band (f1,f2)
# ---------------------------------------------------------------------------
def _rmsGain(H, f1, f2, N=100):
    """
    Compute the RMS gain of a transfer function H in the frequency band [f1, f2].
    Python translation of _rmsGain from Schreier's Delta-Sigma Toolbox.
    """
    w = np.linspace(2 * np.pi * f1, 2 * np.pi * f2, N)
    return np.linalg.norm(_evalTF(H, np.exp(1j * w))) / np.sqrt(N)

# ---------------------------------------------------------------------------
# Helper: _ds_f1f2
#   Compute the frequency band [f1, f2] for a delta-sigma modulator.
#   Python translation of _ds_f1f2 from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _ds_f1f2(OSR=64, f0=0, complex_flag=0):
    """
    Compute the frequency band [f1, f2] for a delta-sigma modulator.
    Python translation of _ds_f1f2 from Schreier's Delta-Sigma Toolbox.
    """
    if complex_flag:
        f1 = f0 - 0.5 / OSR
        f2 = f0 + 0.5 / OSR
    else:
        if f0 > 0.25 / OSR:
            f1 = f0 - 0.25 / OSR
            f2 = f0 + 0.25 / OSR
        else:
            f1 = 0
            f2 = 0.5 / OSR

    return f1, f2

# ---------------------------------------------------------------------------
# Helper: _db
#   Compute gain in _dB.
# ---------------------------------------------------------------------------
def _db(x):
        return 20 * np.log10(np.abs(x))

# ---------------------------------------------------------------------------
# Helper: _ds_synNTFobj1
#   Objective function for synthesizeNTF().
# ---------------------------------------------------------------------------
def _ds_synNTFobj1(x, p, osr, f0):
    """
    Objective function for NTF synthesis.
    Python translation of _ds_synNTFobj1 from Schreier's Delta-Sigma Toolbox.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=complex))

    z = np.exp(2j * np.pi * (f0 + 0.5 / osr * x))

    if f0 > 0:
        z = _padt(z, len(p) // 2, np.exp(2j * np.pi * f0))

    z = np.concatenate([z.flatten(), np.conj(z).flatten()])

    if f0 == 0:
        z = _padb(z, len(p), 1)

    f1, f2 = _ds_f1f2(osr, f0)
    ntf = ZerosPolesGain(z.flatten(), p, 1)
    y = _db(_rmsGain(ntf, f1, f2))

    return y

# ---------------------------------------------------------------------------
# Helper: _synthesizeNTF1
#   Synthesize a Noise Transfer Function for a delta-sigma modulator,
#   with optional zero optimization.
#   Python translation of _synthesizeNTF1 from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _synthesizeNTF1(order, osr, opt, H_inf, f0):
    """
    Synthesize a Noise Transfer Function for a delta-sigma modulator,
    with optional zero optimization.
    Python translation of _synthesizeNTF1 from Schreier's Delta-Sigma Toolbox.
    """

    # --- Determine the zeros ---
    if f0 != 0:  # Bandpass design: halve the order temporarily
        order = order // 2
        dw = np.pi / (2 * osr)
    else:
        dw = np.pi / osr

    if np.isscalar(opt) or (hasattr(opt, '__len__') and len(opt) == 1):
        opt = int(np.atleast_1d(opt)[0])
        if opt == 0:
            z = np.zeros(order, dtype=complex)
        else:
            z = dw * _ds_optzeros(order, 1 + (opt - 1) % 2)
            if z is None or len(z) == 0:
                return None

        if f0 != 0:  # Bandpass design: shift and replicate zeros
            order = order * 2
            z = z + 2 * np.pi * f0
            z = np.concatenate([z, -z])

        z = np.exp(1j * z)
    else:
        z = np.asarray(opt, dtype=complex).flatten()

    # Initial zero optimization guess
    zp = z[np.angle(z) > 0]
    x0 = (np.angle(zp) - 2 * np.pi * f0) * osr / np.pi
    if opt == 4 and f0 != 0:
        # Do not optimize the zeros at f0
        x0 = x0[np.abs(x0) >= 1e-10]

    ntf = ZerosPolesGain(z, np.zeros(order, dtype=complex), 1)
    Hinf_itn_limit = 100
    ftol = 1e-10
    z_inf = 1 if f0 > 0.25 else -1

    opt_iteration = 5  # Max number of zero-optimizing/Hinf iterations
    p = np.zeros(order, dtype=complex)

    while opt_iteration > 0:

        # --- Iteratively determine the poles ---
        if f0 == 0:  # Lowpass design
            H_inf_limit = 2 ** order
            if H_inf >= H_inf_limit:
                print('_synthesizeNTF1 warning: Unable to achieve specified Hinf.')
                print('Setting all NTF poles to zero.')
                p = np.zeros(order, dtype=complex)
                ntf = ZerosPolesGain(z, p, 1)
            else:
                x = 0.3 ** (order - 1)  # starting guess
                converged = False
                fprev = 0.0
                delta_x = 0.0

                for itn in range(1, Hinf_itn_limit + 1):
                    me2 = -0.5 * (x ** (2.0 / order))
                    w = (2 * np.arange(1, order + 1) - 1) * np.pi / order
                    mb2 = 1 + me2 * np.exp(1j * w)
                    p = mb2 - np.sqrt(mb2 ** 2 - 1)

                    out = np.abs(p) > 1
                    p[out] = 1.0 / p[out]
                    p = _cplxpair(p)

                    ntf = ZerosPolesGain(z, p, 1)
                    f = np.real(_evalTF(ntf, z_inf)) - H_inf

                    if itn == 1:
                        delta_x = -f / 100
                    else:
                        delta_x = -f * delta_x / (f - fprev)

                    xplus = x + delta_x
                    x = xplus if xplus > 0 else x * 0.1
                    fprev = f

                    if abs(f) < ftol or abs(delta_x) < 1e-10:
                        converged = True
                        break

                    if x > 1e6:
                        print('_synthesizeNTF1 warning: Unable to achieve specified Hinf.')
                        print('Setting all NTF poles to zero.')
                        p = np.zeros(order, dtype=complex)
                        ntf = ZerosPolesGain(z, p, 1)
                        break

                if itn == Hinf_itn_limit and not converged:
                    print('_synthesizeNTF1 warning: Danger! Iteration limit exceeded.')

        else:  # Bandpass design
            x = 0.3 ** (order / 2 - 1)  # starting guess
            c2pif0 = np.cos(2 * np.pi * f0)
            fprev = 0.0
            delta_x = 0.0

            for itn in range(1, Hinf_itn_limit + 1):
                e2 = 0.5 * x ** (2.0 / order)
                w = (2 * np.arange(1, order + 1) - 1) * np.pi / order
                mb2 = c2pif0 + e2 * np.exp(1j * w)
                p = mb2 - np.sqrt(mb2 ** 2 - 1)

                out = np.abs(p) > 1
                p[out] = 1.0 / p[out]
                p = _cplxpair(p)

                ntf = ZerosPolesGain(z, p, 1)
                f = np.real(_evalTF(ntf, z_inf)) - H_inf

                if itn == 1:
                    delta_x = -f / 100
                else:
                    delta_x = -f * delta_x / (f - fprev)

                xplus = x + delta_x
                x = xplus if xplus > 0 else x * 0.1
                fprev = f

                if abs(f) < ftol or abs(delta_x) < 1e-10:
                    break

                if x > 1e6:
                    print('_synthesizeNTF1 warning: Unable to achieve specified Hinf.')
                    print('Setting all NTF poles to zero.')
                    p = np.zeros(order, dtype=complex)
                    ntf = ZerosPolesGain(z, p, 1)
                    break

                if itn == Hinf_itn_limit:
                    print('_synthesizeNTF1 warning: Danger! Hinf iteration limit exceeded.')

        # --- Zero optimization ---
        if opt < 3:  # Do not optimize the zeros
            opt_iteration = 0
        else:
            if f0 == 0:
                lb = np.zeros(len(x0))
                ub = np.ones(len(x0))
            else:
                ub = 0.5 * np.ones(len(x0))
                lb = -ub

            bounds = list(zip(lb, ub))
            result = minimize(
                fun=lambda x: _ds_synNTFobj1(x, p, osr, f0),
                x0=x0,
                method='SLSQP',  # closest to MATLAB's 'active-set'
                bounds=bounds,
                options={'ftol': 0.01, 'xatol': 0.001, 'maxiter': 100, 'disp': False}
            )
            x0 = result.x

            z = np.exp(2j * np.pi * (f0 + 0.5 / osr * x0))
            if f0 > 0:
                z = _padt(z, len(p) // 2, np.exp(2j * np.pi * f0))
            z = np.concatenate([z.flatten(), np.conj(z).flatten()])
            if f0 == 0:
                z = _padt(z, len(p), 1)

            z = z.flatten()
            ntf = ZerosPolesGain(z, p, 1)

            if abs(np.real(_evalTF(ntf, z_inf)) - H_inf) < ftol:
                opt_iteration = 0
            else:
                opt_iteration -= 1

    return ntf

# ---------------------------------------------------------------------------
# Main function: synthesizeNTF
#   Synthesize a Noise Transfer Function for a delta-sigma modulator.
#   Python translation of synthesizeNTF from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def synthesizeNTF(order=3, osr=64, opt=0, H_inf=1.5, f0=0):
    """
    Synthesize a Noise Transfer Function for a delta-sigma modulator.
    Python translation of synthesizeNTF from Schreier's Delta-Sigma Toolbox.

    Parameters
    ----------
    order   : int, modulator order (default 3)
    osr     : int, oversampling ratio (default 64)
    opt     : int or array, zero optimization (default 0)
                0 -> zeros at DC
                1 -> optimized zeros (Chebyshev)
                2 -> optimized zeros (same as 1 for even order)
                3 -> optimized zeros with fmincon
                4 -> same as 3 but with f0 zeros fixed
                array -> custom zeros
    H_inf   : float, max NTF gain (default 1.5)
    f0      : float, center frequency (0 for lowpass, default 0)

    Returns
    -------
    ntf : ZerosPolesGain object
    """

    if f0 > 0.5:
        raise ValueError('synthesizeNTF: f0 must be less than 0.5.')

    if f0 != 0 and f0 < 0.25 / osr:
        warnings.warn('synthesizeNTF: Creating a lowpass NTF.')
        f0 = 0

    if f0 != 0 and order % 2 != 0:
        raise ValueError('synthesizeNTF: order must be even for a bandpass modulator.')

    opt_array = np.atleast_1d(opt)
    if len(opt_array) > 1 and len(opt_array) != order:
        raise ValueError(f'synthesizeNTF: The opt vector must be of length {order} (=order).')

    # Use _synthesizeNTF1 if scipy.optimize is available (equivalent of fmincon),
    # otherwise fall back to _synthesizeNTF0
    try:
        from scipy.optimize import minimize
        ntf = _synthesizeNTF1(order, osr, opt, H_inf, f0)
    except ImportError:
        ntf = _synthesizeNTF0(order, osr, opt, H_inf, f0)

    return ntf

# ---------------------------------------------------------------------------
# Helper: _ds_quantize
#   Quantize y to:
#     an odd integer in  [-n+1, n-1] if n is even (mid-rise)
#     an even integer in [-n,   n  ] if n is odd  (mid-tread)
#   Python translation of _ds_quantize from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def _ds_quantize(y, n):
    """
    Quantize y to:
      an odd integer in  [-n+1, n-1] if n is even (mid-rise)
      an even integer in [-n,   n  ] if n is odd  (mid-tread)
    Python translation of _ds_quantize from Schreier's Delta-Sigma Toolbox.
    """
    n = np.atleast_1d(np.asarray(n, dtype=int))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    v = np.zeros_like(y)

    for qi in range(len(n)):
        if n[qi] % 2 == 0:  # mid-rise
            v[qi] = 2 * np.floor(0.5 * y[qi]) + 1
        else:                # mid-tread
            v[qi] = 2 * np.floor(0.5 * (y[qi] + 1))

        # Clip output
        L = n[qi] - 1
        v[qi] = np.clip(v[qi], -L, L)

    return v

# ---------------------------------------------------------------------------
# Main function: simulateDSM
#   Simulate a delta-sigma modulator.
#   Python translation of simulateDSM from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def simulateDSM(u, ntf, nlev=2, x0=None):
    """
    Simulate a delta-sigma modulator.
    Python translation of simulateDSM from Schreier's Delta-Sigma Toolbox.

    Parameters
    ----------
    u    : array, input signal, shape (nu, N) or (N,) for single input
    ntf : ZerosPolesGain, dict with 'zeros'/'poles', or ABCD numpy array
    nlev : int or array, number of quantizer levels (default 2)
    x0   : array, initial state (default zeros)

    Returns
    -------
    v    : quantizer output, shape (nq, N)
    xn   : state sequence, shape (order, N)
    xmax : max absolute state values, shape (order,)
    y    : quantizer input, shape (nq, N)
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    nu, N = u.shape
    nlev = np.atleast_1d(np.asarray(nlev, dtype=int))
    nq = len(nlev)

    # --- Determine form and extract system matrices ---
    if isinstance(ntf, ZerosPolesGain):
        zeros = np.asarray(ntf.zeros, dtype=complex).flatten()
        poles = np.asarray(ntf.poles, dtype=complex).flatten()
        form = 2
        order = len(zeros)

    elif isinstance(ntf, dict):
        if 'zeros' not in ntf:
            raise ValueError('simulateDSM: No zeros field in the NTF.')
        if 'poles' not in ntf:
            raise ValueError('simulateDSM: No poles field in the NTF.')
        zeros = np.asarray(ntf['zeros'], dtype=complex).flatten()
        poles = np.asarray(ntf['poles'], dtype=complex).flatten()
        form = 2
        order = len(zeros)

    elif isinstance(ntf, np.ndarray):
        if ntf.ndim == 2 and ntf.shape[1] > 2 and ntf.shape[1] == nu + ntf.shape[0]:
            form = 1
            ABCD = ntf
            order = ABCD.shape[0] - nq
        else:
            raise ValueError('simulateDSM: The ABCD argument does not have proper dimensions.')
    else:
        raise TypeError('simulateDSM: The second argument is neither an ABCD matrix nor an NTF.')

    # --- Initial state ---
    if x0 is None:
        x0 = np.zeros((order, 1), dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float).reshape((order, 1))

    # --- Build state-space matrices ---
    if form == 1:
        A  = ABCD[:order, :order]
        B  = ABCD[:order, order:order + nu + nq]
        C  = ABCD[order:order + nq, :order]
        D1 = ABCD[order:order + nq, order:order + nu]

    else:
        # A realization of 1/H via zpk2ss
        A, B2, C, D2 = zpk2ss(poles, zeros, -1)
        A  = np.atleast_2d(A).astype(complex)
        B2 = np.atleast_2d(B2).astype(complex)
        C  = np.atleast_2d(C).astype(complex)

        # Transform so that C = [1 0 0 ...]
        # Build orthonormal basis: C' augmented with identity, then take QR
        tmp  = np.hstack([C.T, np.eye(order)])          # shape (order, order+1)
        Q, _ = np.linalg.qr(tmp, mode='reduced')        # Q shape (order, order)
        Sinv = Q                                         # (order, order) orthonormal
        S    = np.linalg.inv(Sinv)

        # Check orientation: we want C*Sinv to have positive first element
        C_new = C @ Sinv                                 # shape (1, order)
        if np.real(C_new[0, 0]) < 0:
            S    = -S
            Sinv = -Sinv

        A  = np.real(S @ A @ Sinv)
        B2 = np.real(S @ B2)
        C  = np.zeros((1, order))
        C[0, 0] = 1
        D2 = 0

        # Assume STF = 1
        B1 = -B2
        D1 = np.ones((1, 1))
        B  = np.hstack([B1, B2])

    # --- Simulation loop ---
    v    = np.zeros((nq, N))
    y    = np.zeros((nq, N))
    xn   = np.zeros((order, N))
    xmax = np.abs(x0).flatten()

    for i in range(N):
        y[:, i:i+1]  = C @ x0 + D1 @ u[:, i:i+1]
        v[:, i:i+1]  = _ds_quantize(y[:, i:i+1], nlev)
        x0           = A @ x0 + B @ np.vstack([u[:, i:i+1], v[:, i:i+1]])
        xn[:, i]     = x0.flatten()
        xmax         = np.maximum(np.abs(x0).flatten(), xmax)

    return v, xn, xmax, y

# ---------------------------------------------------------------------------
# Helper: dbv
#   dbv(x) = 20*log10(abs(x)); the _dB equivalent of the voltage x.
#   Python translation of dbv from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def dbv(x):
    """
    dbv(x) = 20*log10(abs(x)); the _dB equivalent of the voltage x.
    Python translation of dbv from Schreier's Delta-Sigma Toolbox.
    """
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    y = -np.inf * np.ones(x.shape)
    nonzero = x != 0
    y[nonzero] = 20 * np.log10(np.abs(x[nonzero]))

    # Return a scalar if the input was scalar
    return float(y[0]) if y.size == 1 else y

# ---------------------------------------------------------------------------
# Main function: calculateSNR
#   Estimate the signal-to-noise ratio given the in-band bins of a
#   (Hann-windowed) FFT and the location of the input signal (f>0).
#   Python translation of calculateSNR from Schreier's Delta-Sigma Toolbox.
# ---------------------------------------------------------------------------
def calculateSNR(hwfft, f, nsig=1):
    """
    Estimate the signal-to-noise ratio given the in-band bins of a
    (Hann-windowed) FFT and the location of the input signal (f>0).
    For nsig=1, the input tone is contained in hwfft[f:f+2];
    this range is appropriate for a Hann-windowed FFT.
    Each increment in nsig adds a bin to either side.
    The SNR is expressed in _dB.
    Python translation of calculateSNR from Schreier's Delta-Sigma Toolbox.
    """
    hwfft = np.asarray(hwfft).flatten()
    n_bins = len(hwfft)

    # 0-indexed: MATLAB's f is 1-indexed, so subtract 1
    signal_bins = np.arange(f - nsig, f + nsig + 1, dtype=int)  # f-nsig+1 to f+nsig+1 in 1-indexed
    signal_bins = signal_bins[(signal_bins >= 0) & (signal_bins < n_bins)]

    s = np.linalg.norm(hwfft[signal_bins])

    noise_bins = np.ones(n_bins, dtype=bool)
    noise_bins[signal_bins] = False
    n = np.linalg.norm(hwfft[noise_bins])

    if n == 0:
        return np.inf
    else:
        return float(dbv(s / n))