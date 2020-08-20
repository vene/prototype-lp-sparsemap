import numpy as np
from .sparsemap_fw import SparseMAPFW


class AdjSparseMAPFW(SparseMAPFW):

    def __init__(self, polytope, q, penalize_v=False, max_iter=100,
                 variant="pairwise", line_search='exact', tol=1e-6):
        """Generic implementation of SparseMAP via Frank-Wolfe variants.

        Parameters
        ----------

        polytope: object,
            A user-supplied object implementing the following methods:
             - `polytope.vertex(y)`, given a hashable structure representation
               `y`, must return a tuple [m_y, n_y] of vectors encoding the
               unaries and additionals of structure y. (n_y can be empty.).
               This is the `y`th column of the matrices M and N in our paper.
             - `polytope.map(eta_u, eta_v)` returns the y that solves
               `argmax_y <m_y, eta_u> + <n_y, eta_v>`.

        penalize_v : bool
            Whether to penalize v or just u

        max_iter: int,
            The number of FW iterations to run.

        variant: {'vanilla' | 'away-step' | 'pairwise'}
            FW variant to run. Pairwise seems to perform the best.

        line search: {'exact' | 'adaptive' | 'oblivious'}

        tol: float,
            Tolerance in the Wolfe gap, for convergence.
        """
        self.polytope = polytope
        self.penalize_v = penalize_v
        self.max_iter = max_iter
        self.variant = variant
        self.line_search = line_search
        self.tol = tol

        self.q = q

    def obj(self, u, v, eta_u, eta_v):
        """objective: Omega(u, v) -<u, eta_u> - <v, eta_v> """
        val = np.sum(u * eta_u) + np.sum(v * eta_v)
        # pen = np.sum(u ** 2)
        pen = np.dot(u, self.q * u)
        if self.penalize_v:
            pen += np.sum(v ** 2)
        return .5 * pen - val

    def grad(self, u, v, eta_u, eta_v):
        """Gradient of self.obj"""
        g_omega_u = self.q * u
        g_omega_v = v if self.penalize_v else 0
        g_u = g_omega_u - eta_u
        g_v = g_omega_v - eta_v

        return [g_u, g_v]

    def get_ls_denom(self, d_u, d_v):
        denom = np.dot(d_u, self.q * d_u)
        if self.penalize_v:
            denom += np.sum(d_v ** 2)
        return denom
