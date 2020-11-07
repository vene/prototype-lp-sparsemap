# TODO: remove duplicated code from fw.py
import numpy as np

class PairwiseFactor(object):
    """A factor with two binary variables and a coupling between them."""

    def vertex(self, y):

        # y is a tuple (0, 0), (0, 1), (1, 0) or (1, 1)
        u = np.array(y, dtype=np.float)
        v = np.atleast_1d(np.prod(u))
        return u, v

    def map_oracle(self, eta_u, eta_v):

        best_score = -np.inf
        best_y = None
        for x1 in (0, 1):
            for x2 in (0, 1):
                y = (x1, x2)
                u, v = self.vertex(y)

                score = np.dot(u, eta_u) + np.dot(v, eta_v)
                if score > best_score:
                    best_score = score
                    best_y = y
        return best_y

    def qp(self, eta_u, eta_v, penalize_v=False):

        if penalize_v:
            # use cvxpy
            import cvxpy as cx
            c1, c2, c12 = eta_u[0], eta_u[1], eta_v[0]
            z1, z2, z12 = cx.Variable(), cx.Variable(), cx.Variable()
            obj = (z1 - c1) ** 2 + (z2 - c2) ** 2 + (z12 - c12) ** 2
            constraints = [
                z1 >= 0,
                z2 >= 0,
                z12 >= 0,
                z1 <= 1,
                z2 <= 1,
                z12 <= 1,
                z12 <= z1,
                z12 <= z2,
                z12 >= z1 + z2 - 1
            ]
            pb = cx.Problem(cx.Minimize(obj), constraints)
            pb.solve(eps_rel=1e-9, eps_abs=1e-9)
            z1 = np.atleast_1d(z1.value)
            z2 = np.atleast_1d(z2.value)
            u = np.concatenate([z1, z2])
            v = np.atleast_1d(z12.value)
            return u, v

        else:
            # Prop 6.5 in Andre Martins' thesis
            # closed form solution
            c1, c2, c12 = eta_u[0], eta_u[1], eta_v[0]

            flip_sign = False
            if c12 < 0:
                flip_sign = True
                c1, c2, c12 = c1 + c12, 1 - c2, -c12

            if c1 > c2 + c12:
                u = [c1, c2 + c12]
            elif c2 > c1 + c12:
                u = [c1 + c12, c2]
            else:
                uu = (c1 + c2 + c12) / 2
                u = [uu, uu]

            u = np.clip(np.array(u), 0, 1)
            v = np.atleast_1d(np.min(u))

            if flip_sign:
                u[1] = 1 - u[1]
                v[0] = u[0] - v[0]

            return u, v

class XORFactor(object):
    """A one-of-K factor"""

    def __init__(self, d):
        self.d = d

    def vertex(self, y):
        # y is an integer between 0 and k-1
        u = np.zeros(self.d)
        u[y] = 1
        v = np.array(())

        return u, v

    def vertex_dot(self, y1, y2):
        return 1 if y1 == y2 else 0

    def map_oracle(self, eta_u, eta_v):
        y = np.argmax(eta_u)
        return y, eta_u[y]

    def qp(self, eta_u, eta_v):
        """Projection onto the simplex"""
        z = 1
        v = np.array(eta_u)
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        uu = np.maximum(v - theta, 0)
        vv = np.array(())
        return uu, vv
