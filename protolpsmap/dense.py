# dense factor
import numpy as np

class DenseFactor(object):

    def __init__(self, m, n):
        self.m = m
        self.n = n

    def vertex(self, y):
        i, j = y
        um = np.zeros(self.m)
        un = np.zeros(self.n)
        um[i] = 1
        un[j] = 1

        U = np.concatenate([um, un])
        V = np.outer(um, un)

#        if self.scale_u is not None:
#            U *= self.scale_u

        return U, V

    def map_oracle(self, eta_u, eta_v):

        S = eta_v.copy()

        eta_um, eta_un = eta_u[:self.m], eta_u[self.m:]
        S += eta_um[:, np.newaxis]
        S += eta_un
        i, j = np.unravel_index(S.argmax(), S.shape)
        return i, j


    def jacobian(self, active_set):
        M = []
        N = []
        for y in active_set:
            m, n = self.vertex(y)
            M.append(m)
            N.append(n.ravel())

        M = np.column_stack(M)
        N = np.column_stack(N)
        Z = np.linalg.pinv(np.dot(M.T, M))
        MZM = M @ Z @ M.T

        d = len(Z)
        one = np.ones((d, d))
        eye = np.eye(d)
        Zsum = Z.sum()
        Zrow = Z.sum(axis=0)

        J = (eye - (Z @ one) / Zsum) @ Z

        JM = M @ J @ M.T
        JN = M @ J @ N.T

        return J, JM, JN, M, Z


if __name__ == '__main__':
    from .sparsemap_fw import SparseMAPFW

    m = 3
    n = 4

    eta_u = np.random.randn(m + n)
    eta_v = np.random.randn(m, n)

    df = DenseFactor(m, n)
    y = df.map_oracle(eta_u, eta_v)
    print(y)
    u, v, active_set = SparseMAPFW(df).solve(eta_u, eta_v)
    print(active_set)

    JM, JN = df.jacobian(active_set)

    from numdifftools import Jacobian
    def f(eta_u_prime):
        u, _, _ = SparseMAPFW(df).solve(eta_u_prime, eta_v)
        return u
    J = Jacobian(f)
    print(J(eta_u) - JM)
