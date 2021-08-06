"""Active set algorithm for structured prediction."""

# author: vlad niculae <vlad.niculae@uva.nl>
# license: mit

from collections import OrderedDict
import numpy as np


class ActiveSet(object):
    def __init__(self, polytope, max_iter=100):
        self.polytope = polytope
        self.max_iter = max_iter

    def _pick_init(self, eta_u, eta_v):
        # for debugging purposes: we initialize with the worst atom.
        y, score = self.polytope.map_oracle(-eta_u, -eta_v)
        return y, -score

    def _reconstruct_guess(self, active_set):
        """Compute the current guess from the weights over the vertices:

            [u, v] = sum_{y in active_set} alpha[y] * [m_y, n_y]

        """
        u, v = [], []

        for y, alpha_y in active_set.items():
            m_y, n_y = self.polytope.vertex(y)
            u.append(alpha_y * m_y)
            v.append(alpha_y * n_y)

        return sum(u), sum(v)

    def _find_step(self, alpha, alpha_next):

        y = None
        gamma = 1

        for i, yy in enumerate(alpha.keys()):
            if alpha[yy] > alpha_next[i]:
                gamma_cand = alpha[yy] / (alpha[yy] - alpha_next[i])
                if gamma_cand < gamma:
                    y = yy
                    gamma = gamma_cand
        return y, gamma

    def solve(self, eta_u, eta_v, full_path=False):

        eta_u = np.asarray(eta_u, dtype=np.float)
        eta_v = np.asarray(eta_v, dtype=np.float)

        alpha = OrderedDict()
        scores = OrderedDict()

        y0, score_y0 = self._pick_init(eta_u, eta_v)
        alpha[y0] = 1
        scores[y0] = score_y0

        utu = self.polytope.vertex_dot(y0, y0)
        MtM = np.array([
            [0, 1],
            [1, utu]], dtype=np.double)

        for it in range(self.max_iter):
            print("solving", alpha)
            print(MtM)
            print()
            b = np.array([1] + list(scores.values()), dtype=np.double)
            res = np.linalg.solve(MtM, b)

            tau, alpha_next = res[0], res[1:]
            print("iter", it, tau, alpha_next)

            y_min, gamma = self._find_step(alpha, alpha_next)

            if gamma < 1:

                for i, yy in enumerate(alpha.keys()):
                    alpha[yy] *= (1 - gamma)
                    alpha[yy] += gamma * alpha_next[i]

                # drop the minimizer
                print(".. dropping", y_min)
                mask = np.ones(1 + len(alpha)).astype(np.bool)
                mask[1 + y_min] = 0
                del alpha[y_min]
                del scores[y_min]
                MtM = MtM[mask][:, mask]

            else:
                # get new point
                for i, yy in enumerate(alpha.keys()):
                    alpha[yy] = alpha_next[i]

                u_curr, v_curr = self._reconstruct_guess(alpha)
                y_next, score_y_next = self.polytope.map_oracle(eta_u - u_curr,
                                                                eta_v - v_curr)
                print(".. adding", y_next)

                gap = tau - score_y_next
                if gap >= -1e-9 or y_next in alpha:
                    print("Converged.", gap)
                    break

                Mu = np.array([1.] + [self.polytope.vertex_dot(y, y_next)
                                      for y in alpha.keys()]).reshape(-1, 1)
                utu = self.polytope.vertex_dot(y_next, y_next)

                alpha[y_next] = 0
                scores[y_next] = score_y_next
                MtM = np.block([[MtM, Mu],
                                [Mu.T, utu]])

        return self._reconstruct_guess(alpha)


def main():
    from polytopes import XORFactor

    d = 4
    xor = XORFactor(d)
    rng = np.random.RandomState(42)
    eta_u = .9 * rng.randn(d)
    print("input", eta_u)

    alg = ActiveSet(xor, max_iter=5)
    u, _ = alg.solve(eta_u, eta_v=None)
    print(u, np.sum(u), np.sum((u - eta_u) ** 2))

    u, _ = xor.qp(eta_u, eta_v=None)
    print(u, np.sum(u), np.sum((u - eta_u) ** 2))




if __name__ == '__main__':
    main()

