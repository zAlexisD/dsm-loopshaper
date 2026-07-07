from __future__ import annotations
import numpy as np
import cvxpy as cp

from bmi.problem import BMIProblem

#NOTE: change CVXPY solver depending on usage (CLARABEL,SCS,CVXOPT,OSQP,etc.)

class BMIPStep:
    """P-step: fix the feedback filter (Af,Bf,Cf,Df), solve for P."""

    def __init__(self, bmi_problem: BMIProblem):
        self.bmi_problem = bmi_problem

    def solve(self, fixed):
        Af, Bf, Cf, Df = fixed
        A, B, C, D = self.bmi_problem.build_system(Af, Bf, Cf, Df)

        n = A.shape[0]
        P = cp.Variable((n, n), symmetric=True)
        gamma = cp.Variable(nonneg=True)

        # P is the only CVXPY variable — quadratic form is linear in P
        M = self.bmi_problem.build_BMI_cvxpy_P_step(P, A, B, C, D, gamma)
        # Add condition on gamma to avoid returning gamma = 0
        constraints = [P >> 1e-6 * np.eye(n), M << 0,gamma >= 1e-4]
        prob = cp.Problem(cp.Minimize(gamma), constraints)
        prob.solve(solver=cp.SCS,eps=1e-6)
        # prob.solve(solver=cp.SCS,eps=1e-6,verbose=True)

        # Debug print
        print(f"  P-step status : {prob.status}")
        print(f"  P-step gamma  : {gamma.value}")

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, float("inf")
        return P.value, float(gamma.value)


class BMIKStep:
    """K-step: fix P, solve for the feedback filter (Af,Bf,Cf,Df)."""

    def __init__(self, bmi_problem: BMIProblem, filter_order: int, n_inputs: int, n_outputs: int):
        self.bmi_problem = bmi_problem
        self.p = filter_order
        self.m_in = n_inputs
        self.m_out = n_outputs

    def solve(self, fixed):
        P = fixed
        p, m_in, m_out = self.p, self.m_in, self.m_out

        Af = cp.Variable((p, p))
        Bf = cp.Variable((p, m_out))
        Cf = cp.Variable((m_in, p))
        Df = cp.Variable((m_in, m_out))
        gamma = cp.Variable(nonneg=True)

        # Filter matrices are CVXPY variables — use Schur complement form
        A, B, C, D = self.bmi_problem.build_system_cvxpy(Af, Bf, Cf, Df)
        M = self.bmi_problem.build_BMI_cvxpy_K_step(P, A, B, C, D, gamma)
        # Add condition on gamma to avoid returning gamma = 0
        constraints = [M << 0,gamma >= 1e-4]
        prob = cp.Problem(cp.Minimize(gamma), constraints)
        prob.solve(solver=cp.SCS,eps=1e-6)
        # prob.solve(solver=cp.SCS,eps=1e-6,verbose=True)

        # Debug print
        print(f"  K-step status : {prob.status}")
        print(f"  K-step gamma  : {gamma.value}\n")

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, float("inf")
        return (Af.value, Bf.value, Cf.value, Df.value), float(gamma.value)