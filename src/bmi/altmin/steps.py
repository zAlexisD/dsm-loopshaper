from __future__ import annotations
import numpy as np
import cvxpy as cp

from bmi.problem import BMIProblem
from bmi.altmin.parametrization import KStepParametrization, FreeFilterParametrization


#NOTE: change CVXPY solver depending on usage (CLARABEL,SCS,CVXOPT,OSQP,etc.)

class BMIPStep:
    """
    P-step: fix filter matrices (Af, Bf, Cf, Df), solve for (P, Pf, mu).
    All three LMIs are active. A, B, C, D are fixed numpy arrays.
    Linear in (P, Pf, mu) — true LMI, single SDP.
    """

    def __init__(self, bmi_problem: BMIProblem):
        self.bmi_problem = bmi_problem
        self.gamma = self.bmi_problem.gamma

    def solve(self, fixed) -> tuple[tuple | None, float]:
        Af, Bf, Cf, Df = fixed
        n, n_f = self.bmi_problem.n, self.bmi_problem.n_f

        # Closed-loop matrices — all numpy since filter matrices are fixed
        A, B, C, D = self.bmi_problem.build_system(Af, Bf, Cf, Df)

        # CVXPY Problem formulation
        P = cp.Variable((n, n), symmetric=True)
        P_f = cp.Variable((n_f, n_f), symmetric=True)
        mu  = cp.Variable(nonneg=True)

        LMI1 = self.bmi_problem.build_LMI1_cvxpy(P, A, B)
        LMI2 = self.bmi_problem.build_LMI2_cvxpy(P, mu, C, D)
        LMI3 = self.bmi_problem.build_LMI3_cvxpy(P_f, Af, Bf, Cf, Df, self.gamma)

        constraints = [
            LMI1 >> 1e-6 * np.eye(LMI1.shape[0]),
            LMI2 >> 1e-6 * np.eye(LMI2.shape[0]),
            LMI3 << -1e-6 * np.eye(LMI3.shape[0]),
            P    >> 1e-6 * np.eye(n),
            P_f  >> 1e-6 * np.eye(n_f),
        ]

        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)
        # prob.solve(solver=cp.SCS,eps=1e-6,verbose=True)

        # Debug print
        print(f"  P-step status : {prob.status}")
        print(f"  P-step mu  : {mu.value}\n")

        if prob.status not in ("optimal", "optimal_inaccurate") or mu.value is None:
            return None, float("inf")

        return (P.value, P_f.value), float(mu.value)


class BMIKStep:
    """
    K-step: fix (P, Pf), solve for filter matrices via parametrization.
    LMI1 and LMI2 depend on closed-loop (A,B,C,D) which depend on filter matrices.
    LMI3 depends directly on filter matrices.
    All LMIs linear in decision variables when P, Pf fixed — true LMI.
    """

    def __init__(self, bmi_problem: BMIProblem, parametrization: KStepParametrization):
        self.bmi_problem = bmi_problem
        self.gamma = self.bmi_problem.gamma
        self.parametrization = parametrization

    def solve(self, fixed) -> tuple[tuple | None, float]:
        P, P_f = fixed
        n, n_f = self.bmi_problem.n, self.bmi_problem.n_f

        # CVXPY problem formulation
        _ = self.parametrization.variables(cp)
        Af, Bf, Cf, Df = self.parametrization.build()
        mu = cp.Variable(nonneg=True)

        A, B, C, D = self.bmi_problem.build_system_cvxpy(Af, Bf, Cf, Df)

        LMI1 = self.bmi_problem.build_LMI1_cvxpy(P, A, B)
        LMI2 = self.bmi_problem.build_LMI2_cvxpy(P, mu, C, D)
        LMI3 = self.bmi_problem.build_LMI3_cvxpy(P_f, Af, Bf, Cf, Df, self.gamma)

        constraints = [
            LMI1 >> 1e-6 * np.eye(LMI1.shape[0]),
            LMI2 >> 1e-6 * np.eye(LMI2.shape[0]),
            LMI3 << -1e-6 * np.eye(LMI3.shape[0]),
        ]

        prob = cp.Problem(cp.Minimize(mu), constraints)
        prob.solve(solver=cp.SCS, eps=1e-6)
        # prob.solve(solver=cp.SCS,eps=1e-6,verbose=True)

        # Debug print
        print(f"  K-step status : {prob.status}")
        print(f"  K-step mu  : {mu.value}\n")

        if prob.status not in ("optimal", "optimal_inaccurate") or mu.value is None:
            return None, float("inf")

        return self.parametrization.extract(), float(mu.value)