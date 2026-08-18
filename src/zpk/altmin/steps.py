from __future__ import annotations
import numpy as np
import cvxpy as cp

from zpk.parametrization import PoleParams, ZeroParams
from bmi.problem import BMIProblem
from bmi.solution import BMISolution


class ZPKPoleStep:
    """
    Fix poles (Af, Bf) and Df=1 → solve single LMI for (Cf, P, Pf, mu).

    With Af, Bf, Df fixed:
      - A, B are fully fixed numpy arrays
      - C is linear in Cf
      - All three LMIs are jointly linear in (P, Pf, Cf, mu)
    Single SDP call — true LMI.
    """

    def __init__(self, bmi_problem: BMIProblem, n_zeros: int,
                 tol_nontrivial:float=0.1,force_dc_zero: bool = True):
        self.bmi_problem = bmi_problem
        self.gamma       = self.bmi_problem.gamma
        self.n_zeros     = int(n_zeros)
        self.tol_nontrivial = tol_nontrivial
        self.force_dc_zero = force_dc_zero

    def solve(self, pole_params: PoleParams
              ) -> tuple[ZeroParams | None, np.ndarray | None, np.ndarray | None, float]:
        """
        Returns (ZeroParams, P, Pf, mu) or (None, None, None, inf).
        """
        bmi = self.bmi_problem
        n, n_f, n_i, n_o = bmi.n, bmi.n_f, bmi.n_i, bmi.n_o
        n_poles = len(pole_params.coeffs)

        # Fixed filter matrices
        Af = pole_params.to_Af()
        Bf = pole_params.to_Bf()
        Df = np.array([[1.0]])

        # Fixed closed-loop A and B (Cf=0 placeholder to get A, B)
        A, B, _, D = bmi.build_system(Af, Bf, np.zeros((n_o, n_poles)), Df)

        # Output filter matrice needed to build C as function of Cf
        Cw = bmi._output_filter.C

        # Decision variables
        Cf  = cp.Variable((n_o, n_poles))
        P   = cp.Variable((n, n), symmetric=True)
        Pf  = cp.Variable((n_f, n_f), symmetric=True)
        mu  = cp.Variable(nonneg=True)

        # C linear in Cf: C = [Cf, Cw]
        C = cp.hstack([Cf, Cw])

        LMI1 = bmi.build_LMI1_cvxpy(P, A, B)
        LMI2 = bmi.build_LMI2_cvxpy(P, mu, C, D)
        LMI3 = bmi.build_LMI3_cvxpy(Pf, Af, Bf, Cf, Df, self.gamma)

        constraints = [
            # matrix inequalities constraints
            LMI1 >> 1e-6 * np.eye(LMI1.shape[0]),
            LMI2 >> 1e-6 * np.eye(LMI2.shape[0]),
            LMI3 << -1e-6 * np.eye(LMI3.shape[0]),
            P    >> 1e-6 * np.eye(n),
            Pf   >> 1e-6 * np.eye(n_f),
            # Constraint for non trivial solutions
            Cf[0, 0] >= self.tol_nontrivial,
        ]

        # DC at z=1 constraint for high-pass design
        if self.force_dc_zero:
            a = pole_params.coeffs
            Df_val = float(Df[0, 0])
            constraints.append(
                # Sum of numerator coefficients approximated to 0
                Df_val + cp.sum(Cf) + Df_val * np.sum(a) <= 0.01
            )

        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)
        # prob.solve(solver=cp.SCS, eps=1e-6,verbose=True)

        # Debug print
        print(f"  Pole-step status : {prob.status}")
        print(f"  Pole-step mu  : {mu.value}\n")

        if prob.status not in ("optimal", "optimal_inaccurate") or Cf.value is None:
            return None, None, None, float("inf")

        # Recover zero coefficients from solved Cf
        # Cf_i = b_pad_i - Df_val * a_i, with Df_val=0 → b_pad = Cf
        a = pole_params.coeffs
        Df_val = float(Df[0, 0])            
        b_pad  = Cf.value.flatten() + Df_val * a
        zero_coeffs = np.concatenate([[Df_val], b_pad[:self.n_zeros]])

        return (
            ZeroParams(coeffs=zero_coeffs),
            P.value,
            Pf.value,
            float(mu.value))


class ZPKZeroStep:
    """
    Fix zeros (Cf, Df) and (P, Pf) from pole step → solve single LMI for Af.

    With P, Pf, Cf, Df fixed and Bf structurally fixed:
      - A and C are both linear in a_var (pole coefficients)
      - B and D are fully fixed
      - All three LMIs are linear in a_var alone
    Single SDP call — true LMI.
    """

    def __init__(self, bmi_problem: BMIProblem, 
                 tol_nontrivial:float=0.1,force_dc_zero: bool = True):
        self.bmi_problem = bmi_problem
        self.gamma = self.bmi_problem.gamma
        self.tol_nontrivial = tol_nontrivial
        self.force_dc_zero = force_dc_zero

    def solve(self,
        fixed: tuple[ZeroParams, PoleParams, np.ndarray, np.ndarray]
        )-> tuple[PoleParams | None, BMISolution | None, float]:
        """
        fixed = (zero_params, pole_params, P, Pf)
        Returns (PoleParams, BMISolution, mu) or (None, None, inf).
        """

        zero_params, pole_params, P_fixed, Pf_fixed = fixed
        bmi = self.bmi_problem
        n_o = bmi.n_o
        n_poles = len(pole_params.coeffs)

        # Fixed quantities from zeros
        b = zero_params.coeffs                
        Df = zero_params.to_Df()
        Df_val = float(b[0])    
        Bf = pole_params.to_Bf()

        b_pad = np.concatenate([b[1:], np.zeros(max(0, n_poles - (len(b) - 1)))])[:n_poles]

        # Output filter matrices
        Aw = bmi._output_filter.A
        Bw = bmi._output_filter.B
        Cw = bmi._output_filter.C
        Dw = bmi._output_filter.D

        # Fixed closed-loop B and D
        B = np.block([[Bf @ Dw], [Bw]])
        D = Dw

        # Companion matrix structural parts — fixed
        Af_shift = np.zeros((n_poles, n_poles))
        if n_poles > 1:
            Af_shift[:-1, 1:] = np.eye(n_poles - 1)
        e_last = np.zeros((n_poles, 1))
        e_last[-1, 0] = 1.0

        # Square reversal matrix
        Rev = np.zeros((n_poles, n_poles))
        for k in range(n_poles):
            Rev[k, n_poles - 1 - k] = 1.0

        # Decision variable: pole coefficients
        a_var = cp.Variable(n_poles)

        # Af linear in a_var
        Af_var = Af_shift - e_last @ cp.reshape(Rev @ a_var, (1, n_poles), 'F')

        # Cf linear in a_var: Cf_i = b_pad_i - Df_val * a_i
        Cf_var = cp.reshape(b_pad - Df_val * (Rev @ a_var), (n_o, n_poles),'F')

        # A linear in a_var
        A_var = cp.bmat([
            [Af_var, Bf @ Cw],
            [np.zeros((bmi.outFilterOrder,n_poles)), Aw]])

        # C linear in a_var
        C_var = cp.hstack([Cf_var, Cw])

        # Recompute mu as a free variable too since C_cl changes with a_var
        mu = cp.Variable(nonneg=True)

        LMI1 = bmi.build_LMI1_cvxpy(P_fixed, A_var, B)
        LMI2 = bmi.build_LMI2_cvxpy(P_fixed, mu, C_var, D)
        LMI3 = bmi.build_LMI3_cvxpy(Pf_fixed, Af_var, Bf, Cf_var, Df, self.gamma)

        constraints = [
            LMI1 >> 1e-6 * np.eye(LMI1.shape[0]),
            LMI2 >> 1e-6 * np.eye(LMI2.shape[0]),
            LMI3 << -1e-6 * np.eye(LMI3.shape[0]),
            # LMI1 >> 0,                               
            # LMI2 >> 0,
            # LMI3 << 0, 
            Cf_var[0, 0] >= self.tol_nontrivial
        ]

        # DC at z=1 for high-pass design
        if self.force_dc_zero:
            b_sum = np.sum(b_pad)
            constraints.append(
                Df_val + b_sum + Df_val * cp.sum(a_var) <= 0.01
            )

        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)
        # prob.solve(solver=cp.SCS, eps=1e-6,verbose=True)

        # Debug print
        print(f"  Zero-step status : {prob.status}")
        print(f"  Zero-step mu  : {mu.value}\n")

        if prob.status not in ("optimal", "optimal_inaccurate") or a_var.value is None:
            return None, None, float("inf")

        a_opt  = a_var.value                              # (n_poles,)
        Af_opt = Af_shift - e_last @ (Rev @ a_opt).reshape(1, n_poles)
        Cf_opt = (b_pad - Df_val * (Rev @ a_opt)).reshape(n_o, n_poles)

        new_poles = PoleParams(coeffs=a_opt) 
        solution = BMISolution(
            P=P_fixed, Pf=Pf_fixed,
            Af=Af_opt, Bf=Bf,
            Cf=Cf_opt, Df=Df,
            mu=float(mu.value),
            feasible=True,
        )
        return new_poles, solution, float(mu.value)