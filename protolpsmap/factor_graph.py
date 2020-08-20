# generic implementation for a factor graph
# with only pairwise factors over multi-variables

import numpy as np
from scipy.sparse import block_diag
from scipy.linalg import pinv2 as pinv
from scipy.sparse import linalg as sl

import cvxpy as cx

from numdifftools import Jacobian
from block import block

from numpy.testing import assert_allclose

from .dense import DenseFactor
from .sparsemap_fw import SparseMAPFW
from .sparsemap_fw_adjusted import AdjSparseMAPFW


THR = 1e-6  # in jacobian, round off everything less than this to 0


def _block_diag(xs):
    return block_diag(list(xs)).toarray()


class FactorGraph(object):

    def __init__(self, n_vars, n_states, factors):

        self.n_vars = n_vars
        self.n_states = n_states

        if factors == 'seq':
            factors = [(i - 1, i) for i in range(1, n_vars)]
        elif factors == 'loop':
            factors = [(i - 1, i % n_vars) for i in range(1, n_vars + 1)]
        elif factors == 'star':
            factors = [(0, i) for i in range(1, n_vars)]

        self.factors = factors

        self.n_factors = len(factors)
        self.u_sz = n_vars * n_states
        self.pf_sz = n_states ** 2
        self.p_sz = self.pf_sz * self.n_factors
        self.uf_sz = n_states * 2

        self.df = DenseFactor(n_states, n_states)

        self.Mf = self.make_M()
        self.C = self.make_C()  # shape: (sum_f Mf_sz) x u_sz
        assert self.C.shape == (self.uf_sz * self.n_factors, self.u_sz)

        self.var_deg = np.diag(self.C.T @ self.C)
        self.B = _block_diag(np.ones(self.pf_sz) for _ in factors)

    def make_M(self):
        return np.column_stack([
            self.df.vertex((i, j))[0]
            for i in range(self.n_states)
            for j in range(self.n_states)])

    def make_C(self):
        # build the C correspondence matrix
        # satisfying C @ u = M @ p

        Z = np.zeros((self.n_states, self.n_states))
        I = np.identity(self.n_states)

        blocks = []
        for f, ixs in enumerate(self.factors):
            for i in ixs:
                row = [I if i == k else Z for k in range(self.n_vars)]
                blocks.append(row)
        C = np.block(blocks)
        return C

    def obj(self, p, x):

        pfs = p.reshape(-1, self.pf_sz)
        us = self.Mf @ pfs.T
        u = self.C.T @ np.ravel(us, order='F')
        u /= self.var_deg

        return np.dot(p, x) - 0.5 * np.sum(u ** 2)

    def get_full_potentials(self, eta_u, eta_v):

        eta_v_from_u = self.C @ (eta_u / self.var_deg)
        M = _block_diag(self.Mf for _ in self.factors)
        return M.T @ eta_v_from_u + eta_v

    def solve_cvxpy(self, x, qp=True):

        p = cx.Variable(self.p_sz)
        u = cx.Variable(self.u_sz)
        obj = p @ x
        if qp:
            obj -= 0.5 * cx.sum_squares(u)

        obj = cx.Maximize(obj)
        M = _block_diag(self.Mf for _ in self.factors)
        constr = [
            p >= 0,
            self.B @ p == np.ones(self.n_factors),
            self.C @ u == M @ p
        ]

        pb = cx.Problem(obj, constr)
        # pb.solve(solver='ECOS',
                 # max_iters=100000,
                 # abstol=1e-99,
                 # reltol=1e-99,
                 # feastol=1e-99),
        # pb.solve(solver='MOSEK',
                 # mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-99},
                 # verbose=False)
        pb.solve(max_iter=10000000,
                 eps_abs=1e-11,
                 eps_rel=1e-11,
                 eps_prim_inf=1e-11,
                 eps_dual_inf=1e-11,
                 scaling=10,
                 polish=10,
                 verbose=False)
        pp = p.value
#         pp[pp < THR] = 0
        return pp, u.value, obj.value

    def solve(self, x, n_iter=5000, t=.1):
        """ADMM"""
        lam = np.zeros((self.n_factors, 2 * self.n_states))

        u = np.zeros((self.n_vars, self.n_states))
        var_deg = self.var_deg.reshape(u.shape)

        xfs = x.reshape(-1, self.pf_sz)
        pfs = np.zeros_like(xfs)

        for it in range(n_iter):
            obj = 0

            for f, (i, j) in enumerate(self.factors):

                uf = u[[i, j]].ravel()
                deg = var_deg[[i, j]].ravel()

                eta_u = uf + lam[f] / t
                eta_v = xfs[f].reshape(self.n_states, self.n_states) / t

                q = 1 + 1 / (t * deg)
                fw = AdjSparseMAPFW(self.df, q=q, tol=1e-8, max_iter=50, line_search='exact')

                _, pf, _ = fw.solve(eta_u, eta_v)
                pfs[f] = pf.ravel()

            mp = self.Mf @ pfs.T
            mp = mp.ravel(order='F')
            u_next = self.C.T @ mp / self.var_deg

            diff = mp - self.C @ u_next
            diff = diff.reshape(self.n_factors, -1)
            viol = np.sum(diff ** 2)

            if np.sum(diff ** 2) == 0:
                break

            lam -= t * diff

        print('final violation', viol)

        print(lam)

        return pfs.ravel()

    def Jp_num(self, x):

        def solve_cvxpy_p(x):
            return self.solve_cvxpy(x)[0]

        jac = Jacobian(solve_cvxpy_p, step=1e-6)
        return jac(x)

    def Ju_num(self, x):

        def solve_cvxpy_u(x):
            return self.solve_cvxpy(x)[1]

        jac = Jacobian(solve_cvxpy_u, step=1e-6)
        return jac(x)

    def jacobian_solve(self, p):
        supp = p > THR
        nnz = supp.sum()

        supp_f = supp.reshape(self.n_factors, -1)

        B = self.B[:, supp]
        C = self.C
        # print(pinv(C.T @ C).sum(axis=0))

        deg_copies = (C @ self.var_deg).reshape(self.n_factors, -1)
        C = C / np.sqrt(self.var_deg)

        Ms = []
        for f in range(self.n_factors):
            M = self.Mf[:, supp_f[f]]
            M /= np.sqrt(deg_copies[f])[:, np.newaxis]
            Ms.append(M)

        M = _block_diag(Ms)
        MtM = M.T @ M

        lhs = block(
            arrtype=np.array,
           rows=(( MtM, 0, -M.T, B.T),
                 ( 0,   0,  C.T,   0),
#            rows=(( 0,   0, -M.T, B.T),
#                  ( 0, 'I',  C.T,   0),
                 (-M,   C,    0,   0),
                 ( B,   0,    0,   0))
        )

        lhs_inv = pinv(lhs)
        Jp_bar = lhs_inv[:nnz, :nnz]
        Ju_bar = lhs_inv[nnz:nnz + self.u_sz , :nnz]
        Jlam = lhs_inv[nnz + self.u_sz:nnz + self.u_sz + self.C.shape[0], :nnz]
        # Jlam_wrt_eta = (C.T @ M @ Jlam.T).T
        # print(Jlam_wrt_eta)
        Jlam = lhs_inv[nnz + self.u_sz:nnz + self.u_sz + self.C.shape[0],
                       nnz:nnz +self.u_sz]
        # print(Jlam)

        MtMs = [M_.T @ M_ for M_ in Ms]
        Zs = [pinv(MtM) for MtM in MtMs]
        Z_row_sum = [Z.sum(axis=0) for Z in Zs]
        Ss = [np.outer(zs, zs) / zs.sum() for zs in Z_row_sum]
        Qs = [Z_ - S_ for Z_, S_ in zip(Zs, Ss)]

        Q = _block_diag(Qs)

        # print(M @ Q @ M.T)
        # print(C.T @ Jlam)
        # print(C.T @ M @ Q @ M.T @ Jlam)

        wh = np.where(supp)[0]

        Jp = np.zeros((self.p_sz, self.p_sz))
        Ju = np.zeros((self.u_sz, self.p_sz))
        Jp[wh[:, np.newaxis], wh] = Jp_bar
        Ju[:, wh] = Ju_bar


        return Jp, Ju, Jp_bar, Ju_bar

    def jacobian_vec(self, p, du, n_iter=100000):
        """the iterative projections method used in the paper"""
        supp = p > THR
        nnz = supp.sum()
        supp_f = supp.reshape(self.n_factors, -1)

        C = self.C
        deg_copies = (C @ self.var_deg).reshape(self.n_factors, -1)
        #  C = C / np.sqrt(self.var_deg)
        # C = C / self.var_deg

        Ms = []
        Zs = []
        Qs = []

        for f in range(self.n_factors):

            M = self.Mf[:, supp_f[f]]
            M_div = M / np.sqrt(deg_copies[f])[:, np.newaxis]

            Z = pinv(M_div.T @ M_div)
            zs = Z.sum(axis=0)
            Q = Z - np.outer(zs, zs) / zs.sum()

            Ms.append(M)
            Zs.append(Z)
            Qs.append(Q)


        for t in range(n_iter):
            du_new = du / self.var_deg
            # du_new = du.copy()
            du_new = (C @ du_new).reshape(self.n_factors, -1)
            du_new = [M @ (Q @ (M.T @ pp)) for pp, M, Q in zip(du_new, Ms, Qs)]
            du_new = C.T @ np.concatenate(du_new)
            du_new = du_new / self.var_deg

            res = np.sum((du_new - du) ** 2)
            du = du_new

            if res < 1e-20:
                # print(t)
                break

        # du /= self.var_deg
        # du = (C @ du).reshape(self.n_factors, -1)
        # du = [Q @ (M.T @ pp) for pp, M, Q in zip(du, Ms, Qs)]

        return du

    def jacobian_fast(self, p, return_wrt_u=False, return_uu=False):
        """compute the full jacobian w eigendecomposition"""
        supp = p > THR
        nnz = supp.sum()
        supp_f = supp.reshape(self.n_factors, -1)

        C = self.C
        deg_copies = (C @ self.var_deg).reshape(self.n_factors, -1)
        C = C / np.sqrt(self.var_deg)

        Ms = []
        Zs = []
        Qs = []

        for f in range(self.n_factors):

            M = self.Mf[:, supp_f[f]]
            M /= np.sqrt(deg_copies[f])[:, np.newaxis]

            Z = pinv(M.T @ M)
            zs = Z.sum(axis=0)
            Q = Z - np.outer(zs, zs) / zs.sum()

            Ms.append(M)
            Zs.append(Z)
            Qs.append(Q)

        def matvec(x):
            # compute Ct @ M @ Q @ Mt @ C @ x
            p = (C @ x).reshape(self.n_factors, -1)
            p = [M @ (Q @ (M.T @ p)) for p, M, Q in zip(p, Ms, Qs)]
            return C.T @ np.concatenate(p)

        CMQMC = sl.LinearOperator(shape=(self.u_sz, self.u_sz), matvec=matvec)

        k = nnz - self.n_factors  # prove
        if k == 0:
            return np.zeros((nnz, nnz)), np.zeros((self.u_sz, nnz))

        if k >= self.u_sz:
            print("attention!")
        k = min(k, self.u_sz - 1)

        lams, U = sl.eigsh(CMQMC, k=k)
        U = U[:, lams >= .99]

        if return_uu:
            return U @ U.T

        CU = (C @ U).reshape(self.n_factors, self.uf_sz, -1)
        sqrt_blocks = [Zs[f] @ Ms[f].T @ CU[f]  for f in range(self.n_factors)]
        sqrt = np.row_stack(sqrt_blocks)
        Jp_bar = sqrt @ sqrt.T
        Ju_bar = U @ sqrt.T

        if return_wrt_u:
            M_blocks = _block_diag(Ms)
            J_smol = C.T @ M_blocks @ Jp_bar
            J_ful = np.zeros((C.shape[1], p.shape[0]))
            J_ful[:, supp] = J_smol
            return J_ful

        return Jp_bar, Ju_bar


def test_numeric(n_problems=10):

    for factors in ('loop', 'seq', 'star'):
        for n_vars in (3,):
            for n_states in (4,):

                rng = np.random.RandomState(0)
                fg = FactorGraph(n_vars=n_vars,
                                 n_states=n_states,
                                 factors=factors)
                for _ in range(n_problems):
                    x = rng.randn(fg.p_sz)

                    Jp_num = fg.Jp_num(x)
                    Ju_num = fg.Ju_num(x)
                    p, u, val = fg.solve_cvxpy(x)

                    Jp, Ju, _, _ = fg.jacobian_solve(p)

                    print(np.linalg.norm(Jp - Jp_num),
                          np.linalg.norm(Ju - Ju_num))
        print()


if __name__ == '__main__':

    # test_numeric()
    # exit()

    np.set_printoptions(precision=3, suppress=True)

    # rng = np.random.RandomState(31337)
    rng = np.random.RandomState()

    for factors in ('star', 'seq', 'loop'):
        print()
        print(factors)
        fg = FactorGraph(
                n_vars=3,
                n_states=2,
                factors=factors
        )

        for _ in range(1):
            x = rng.randn(fg.p_sz) * 0.001
            p, u, val = fg.solve_cvxpy(x)

            print(p)
            print(fg.obj(p, x))

            # print(fg.solve(x))

            Jp, Ju, Jp_bar, Ju_bar = fg.jacobian_solve(p)

            # print("real Jp\n", Jp_bar)
            # print("real Ju\n", Ju_bar)
            # print()
            Jp_, Ju_ = fg.jacobian_fast(p)

            print("Are we correct?", np.linalg.norm(Jp_ - Jp_bar), np.linalg.norm(Ju_ - Ju_bar))

            print()
            print()

            du = rng.randn(*u.shape)
            Ju = fg.jacobian_fast(p, return_uu=True)
            print(Ju.T @ du)
            print(fg.jacobian_vec(p, du))
