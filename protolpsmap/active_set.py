"""Active set algorithm for structured prediction."""

# author: vlad niculae <vlad.niculae@uva.nl>
# license: mit

from collections import defaultdict
import numpy as np


class ActiveSet(object):
    def __init__(self, polytope,  max_iter=100):
        self.polytope = polytope
        self.max_iter = max_iter

    def _pick_init(self, eta_u, eta_v):
        # for debugging purposes: we initialize with the worst atom.
        y, score = self.polytope.map_oracle(-eta_u, -eta_v)
        return y, -score

    def _reconstruct_guess(self, active_set, alpha):
        """Compute the current guess from the weights over the vertices:

            [u, v] = sum_{y in active_set} alpha[y] * [m_y, n_y]

        """
        u, v = self.polytope.vertex(active_set[0])
        u *= alpha[0]
        v *= alpha[0]

        for i in range(1, len(active_set)):
            m_y, n_y = self.polytope.vertex(active_set[i])
            u += alpha[i] * m_y
            v += alpha[i] * n_y

        return u, v

    def _find_step(self, p, p_next):

        y = None
        gamma = 1

        for yy in range(len(p)):
            if p[yy] > p_next[yy]:
                gamma_cand = p[yy] / (p[yy] - p_next[yy])
                if gamma_cand < gamma:
                    y = yy
                    gamma = gamma_cand
        return y, gamma

    def solve(self, eta_u, eta_v, full_path=False):

        eta_u = np.asarray(eta_u, dtype=np.float)
        eta_v = np.asarray(eta_v, dtype=np.float)

        y0, score_y0 = self._pick_init(eta_u, eta_v)

        active_set = [y0]
        scores = [1, score_y0]
        p = np.zeros(1)

        utu = self.polytope.vertex_dot(y0, y0)
        MtM = np.array([
            [0, 1],
            [1, utu]], dtype=np.double)

        for it in range(self.max_iter):
            print("solving", active_set, scores)
            print(MtM)
            print()
            res = np.linalg.solve(MtM,
                                  np.array(scores, dtype=np.double))
            # res = np.linalg.lstsq(MtM,
                                  # np.array(scores, dtype=np.double))
            # res, _, _, _ = res

            tau, p_next = res[0], res[1:]
            print("iter", it, tau, p_next)

            y_min, gamma = self._find_step(p, p_next)

            if gamma < 1:
                p *= (1 - gamma)
                p += gamma * p_next

                # drop the minimizer
                print(".. dropping", y_min)
                mask = np.ones(1 + len(active_set)).astype(np.bool)
                mask[1 + y_min] = 0
                print(mask)

                del active_set[y_min]
                del scores[1 + y_min]
                p = p[mask[1:]]
                MtM = MtM[mask][:, mask]

            else:
                # get new point
                p = p_next
                u_curr, v_curr = self._reconstruct_guess(active_set, p)
                y_next, score_y_next = self.polytope.map_oracle(eta_u - u_curr,
                                                                eta_v - v_curr)
                print(".. adding", y_next)

                gap = tau - score_y_next
                if gap >= -1e-9:
                    print("Converged.", gap)
                    break

                Mu = np.array([1.] + [self.polytope.vertex_dot(y, y_next)
                               for y in active_set]).reshape(-1, 1)
                utu = self.polytope.vertex_dot(y_next, y_next)
                active_set.append(y_next)
                MtM = np.block([[MtM, Mu],
                                [Mu.T, utu]])
                scores.append(score_y_next)
                p = np.concatenate([p, np.zeros(1)])

        return self._reconstruct_guess(active_set, p)


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

