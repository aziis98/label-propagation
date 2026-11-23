"""
Harmonic Energy Minimization for Semi-Supervised Learning.

Reference:
    Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty (2003).
    "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions."
    Proceedings of the 20th International Conference on Machine Learning (ICML 2003).
"""

from __future__ import annotations

from typing import Any, Callable, Literal
import numpy as np
import scipy.linalg
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class HarmonicSolver:
    """
    Harmonic function solver for graph-based semi-supervised learning.

    Finds a real-valued harmonic function f on the graph minimizing the Dirichlet energy:
        E(f) = (1/2) * \\sum_{i,j} w_ij (f(i) - f(j))^2 = f^T \\Delta f
    subject to the boundary condition:
        f(x_i) = y_i for all labeled points x_i \\in L.
    """

    def __init__(
        self,
        sigma: float | np.ndarray = 1.0,
        metric: str = "rbf",
        k_neighbors: int | None = None,
        solver_type: Literal["cholesky", "lu", "solve"] = "cholesky",
        epsilon_reg: float = 1e-10,
        eps_pagerank: float = 0.0,
    ) -> None:
        self.sigma = sigma
        self.metric = metric
        self.k_neighbors = k_neighbors
        self.solver_type = solver_type
        self.epsilon_reg = epsilon_reg
        self.eps_pagerank = eps_pagerank

        # Learned attributes
        self.classes_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.L_idx_: np.ndarray | None = None
        self.U_idx_: np.ndarray | None = None
        self.F_: np.ndarray | None = None
        self.F_U_: np.ndarray | None = None
        self.F_L_: np.ndarray | None = None
        self.cond_number_: float | None = None
        self.n_components_: int | None = None
        self.is_singular_: bool = False
        self._last_fu: np.ndarray | None = None
        self._last_H: float | None = None

    def build_weight_matrix(
        self,
        X: np.ndarray,
        sigma: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Construct graph affinity matrix W.
        Supports dense isotropic/anisotropic RBF graphs and symmetrized k-NN graphs:
            w_ij = exp(- \\sum_{d=1}^m (x_id - x_jd)^2 / sigma_d^2), with w_ii = 0.
        """
        n, m = X.shape
        sig = self.sigma if sigma is None else sigma

        if np.isscalar(sig):
            sig_vec = np.full(m, float(sig), dtype=np.float64)
        else:
            sig_vec = np.asarray(sig, dtype=np.float64)
            if sig_vec.ndim == 0:
                sig_vec = np.full(m, float(sig_vec), dtype=np.float64)
            elif len(sig_vec) == 1:
                sig_vec = np.full(m, sig_vec[0], dtype=np.float64)
            elif len(sig_vec) != m:
                raise ValueError(f"sigma dimension {len(sig_vec)} does not match feature dimension {m}")

        if np.any(sig_vec <= 0):
            raise ValueError("All components of sigma must be strictly positive.")

        X_scaled = X / sig_vec

        if self.k_neighbors is None:
            sq_dists = cdist(X_scaled, X_scaled, metric="sqeuclidean")
            W = np.exp(-sq_dists)
            np.fill_diagonal(W, 0.0)
            return W

        k = min(self.k_neighbors, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        W = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            nbr_indices = indices[i, 1:]
            nbr_dists = distances[i, 1:]
            W[i, nbr_indices] = np.exp(-(nbr_dists ** 2))

        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        return W

    build_graph = build_weight_matrix

    def solve_harmonic(
        self,
        W: np.ndarray,
        L_idx: np.ndarray,
        U_idx: np.ndarray,
        F_L: np.ndarray,
        eps_pagerank: float | None = None,
        compute_cond: bool = False,
    ) -> np.ndarray:
        """
        Solve harmonic system on unlabeled points.
        """
        u = len(U_idx)
        n = W.shape[0]
        n_classes = F_L.shape[1]
        if u == 0:
            return np.empty((0, n_classes), dtype=np.float64)

        eps_pr = self.eps_pagerank if eps_pagerank is None else eps_pagerank
        d = np.sum(W, axis=1)

        if eps_pr == 0.0:
            D_UU = np.diag(d[U_idx])
            W_UU = W[np.ix_(U_idx, U_idx)]
            W_UL = W[np.ix_(U_idx, L_idx)]
            Delta_UU = D_UU - W_UU
            RHS = W_UL @ F_L

            if compute_cond:
                try:
                    self.cond_number_ = float(np.linalg.cond(Delta_UU))
                except Exception:
                    self.cond_number_ = np.nan
            else:
                self.cond_number_ = None

            if self.epsilon_reg > 0:
                Delta_UU = Delta_UU + self.epsilon_reg * np.eye(u, dtype=np.float64)

            self.is_singular_ = False
            try:
                c, lower = scipy.linalg.cho_factor(Delta_UU, lower=True, check_finite=False)
                F_U = scipy.linalg.cho_solve((c, lower), RHS, check_finite=False)
            except (scipy.linalg.LinAlgError, ValueError):
                self.is_singular_ = True
                try:
                    F_U = scipy.linalg.solve(Delta_UU, RHS, assume_a="sym")
                except Exception:
                    pinv_Delta = np.linalg.pinv(Delta_UU)
                    F_U = pinv_Delta @ RHS
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_d = np.where(d > 1e-14, 1.0 / d, 0.0)
            P = inv_d[:, np.newaxis] * W
            zero_degree_nodes = np.flatnonzero(d <= 1e-14)
            if len(zero_degree_nodes) > 0:
                P[zero_degree_nodes, :] = 1.0 / n

            P_tilde = eps_pr * (1.0 / n) + (1.0 - eps_pr) * P
            P_tilde_UU = P_tilde[np.ix_(U_idx, U_idx)]
            P_tilde_UL = P_tilde[np.ix_(U_idx, L_idx)]

            A = np.eye(u, dtype=np.float64) - P_tilde_UU
            RHS = P_tilde_UL @ F_L

            if compute_cond:
                try:
                    self.cond_number_ = float(np.linalg.cond(A))
                except Exception:
                    self.cond_number_ = np.nan
            else:
                self.cond_number_ = None

            self.is_singular_ = False
            try:
                lu, piv = scipy.linalg.lu_factor(A, check_finite=False)
                F_U = scipy.linalg.lu_solve((lu, piv), RHS, check_finite=False)
            except (scipy.linalg.LinAlgError, ValueError):
                self.is_singular_ = True
                F_U = scipy.linalg.solve(A, RHS)

        F_U = np.clip(F_U, 0.0, 1.0)
        row_sums = F_U.sum(axis=1, keepdims=True)
        valid = (row_sums.squeeze() > 1e-12)
        if np.any(valid):
            F_U[valid] = F_U[valid] / row_sums[valid]
        if np.any(~valid):
            F_U[~valid] = 1.0 / n_classes

        return F_U

    # =========================================================================
    # DECISION RULES AND CLASS MASS NORMALIZATION
    # =========================================================================

    def predict_thresh(self, F_U: np.ndarray, classes: np.ndarray | None = None) -> np.ndarray:
        """Standard threshold / argmax decision rule: argmax_c F_U(i, c)."""
        cls = self.classes_ if classes is None else classes
        if cls is None:
            cls = np.arange(F_U.shape[1])
        if len(F_U) == 0:
            return np.empty((0,), dtype=cls.dtype)
        best_col = np.argmax(F_U, axis=1)
        return cls[best_col]

    @staticmethod
    def estimate_priors(y_L: np.ndarray, n_classes: int) -> np.ndarray:
        """Estimate class priors q_c with Laplace smoothing: (l_c + 1) / (l + C)."""
        l_total = len(y_L)
        classes = np.arange(n_classes) if np.issubdtype(y_L.dtype, np.integer) else np.unique(y_L)
        priors = np.zeros(n_classes, dtype=np.float64)
        for c_idx, c_val in enumerate(classes):
            l_c = np.sum(y_L == c_val)
            priors[c_idx] = (l_c + 1.0) / (l_total + n_classes)
        return priors

    def predict_cmn(
        self,
        F_U: np.ndarray,
        y_L: np.ndarray,
        classes: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Class Mass Normalization (CMN) decision rule:
            argmax_c [ q_c * F_U(i, c) / sum_{j in U} F_U(j, c) ]
        """
        cls = self.classes_ if classes is None else classes
        if cls is None:
            cls = np.unique(y_L)
        if len(F_U) == 0:
            return np.empty((0,), dtype=cls.dtype)

        n_classes = len(cls)
        priors = np.zeros(n_classes, dtype=np.float64)
        l_total = len(y_L)
        for c_idx, c_val in enumerate(cls):
            l_c = np.sum(y_L == c_val)
            priors[c_idx] = (l_c + 1.0) / (l_total + n_classes)

        class_masses = np.sum(F_U, axis=0)
        eps = 1e-12
        safe_masses = np.where(class_masses > eps, class_masses, eps)
        scaled_F_U = F_U * (priors / safe_masses)

        best_col = np.argmax(scaled_F_U, axis=1)
        return cls[best_col]

    @staticmethod
    def cmn_probabilities_binary(f_u: np.ndarray, q: float) -> np.ndarray:
        """
        Compute binary CMN calibrated probabilities (Eq. 15 of Zhu et al. 2003):
            S = sum_{j in U} f_u(j), A = q(u - S), B = (1 - q)S
            f_tilde_i = (A * f_u(i)) / (A * f_u(i) + B * (1 - f_u(i)))
        """
        u = len(f_u)
        S = float(np.sum(f_u))
        S_clamped = np.clip(S, 1e-10, u - 1e-10)
        A = q * (u - S_clamped)
        B = (1.0 - q) * S_clamped

        denom = A * f_u + B * (1.0 - f_u)
        denom_safe = np.where(np.abs(denom) > 1e-12, denom, 1e-12)
        f_tilde = (A * f_u) / denom_safe
        return np.clip(f_tilde, 0.0, 1.0)

    @staticmethod
    def cmn_effective_threshold(f_u: np.ndarray, q: float) -> float:
        """
        Compute effective decision threshold tau_CMN for binary CMN:
            tau_CMN = ( (1 - q) * f_bar ) / ( q * (1 - f_bar) + (1 - q) * f_bar )
        where f_bar = (1/u) * sum_{j in U} f_u(j).
        """
        f_bar = float(np.mean(f_u))
        f_bar = np.clip(f_bar, 1e-10, 1.0 - 1e-10)
        num = (1.0 - q) * f_bar
        den = q * (1.0 - f_bar) + (1.0 - q) * f_bar
        if np.abs(den) < 1e-12:
            return 0.5
        return float(num / den)

    @staticmethod
    def compute_binary_entropy(p: np.ndarray) -> float:
        """
        Compute mean binary entropy H(p) in bits (base 2):
            H(p) = - (1/u) * sum_{i in U} [ p_i * log2(p_i) + (1 - p_i) * log2(1 - p_i) ]
        """
        p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
        h_vals = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))
        return float(np.mean(h_vals))

    # =========================================================================
    # TRANSDUCTIVE LEAVE-ONE-OUT CV VIA KRON REDUCTION
    # =========================================================================

    def kron_loo_cv(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        sigma: float | np.ndarray | None = None,
        W: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Exact Transductive Leave-One-Out Cross-Validation on L via Kron Reduction.
        Reuses W if provided.
        """
        y_arr = np.asarray(y_labeled)
        is_unlabeled = np.isnan(y_arr) | (y_arr == -1) if np.issubdtype(y_arr.dtype, np.floating) else (y_arr == -1)
        L_idx = np.flatnonzero(~is_unlabeled)
        U_idx = np.flatnonzero(is_unlabeled)
        l = len(L_idx)
        u = len(U_idx)

        y_L = y_arr[L_idx].astype(np.float64)
        W_mat = self.build_weight_matrix(X, sigma=sigma) if W is None else W
        d = np.sum(W_mat, axis=1)

        D_UU = np.diag(d[U_idx])
        W_UU = W_mat[np.ix_(U_idx, U_idx)]
        W_UL = W_mat[np.ix_(U_idx, L_idx)]
        W_LU = W_mat[np.ix_(L_idx, U_idx)]
        W_LL = W_mat[np.ix_(L_idx, L_idx)]
        D_LL = np.diag(d[L_idx])

        Delta_UU = D_UU - W_UU
        Delta_LL = D_LL - W_LL

        if self.epsilon_reg > 0:
            Delta_UU = Delta_UU + self.epsilon_reg * np.eye(u, dtype=np.float64)

        try:
            c, lower = scipy.linalg.cho_factor(Delta_UU, lower=True, check_finite=False)
            X_mat = scipy.linalg.cho_solve((c, lower), W_UL, check_finite=False)
        except Exception:
            X_mat = np.linalg.pinv(Delta_UU) @ W_UL

        S_mat = Delta_LL - W_LU @ X_mat

        f_U = X_mat @ y_L
        sum_f_U = float(np.sum(f_U))
        sum_X_cols = np.sum(X_mat, axis=0)

        Sy = S_mat @ y_L

        f_loo = np.zeros(l, dtype=np.float64)
        y_pred_thresh = np.zeros(l, dtype=int)
        y_pred_cmn = np.zeros(l, dtype=int)
        singular_mask = np.zeros(l, dtype=bool)

        for i in range(l):
            S_ii = S_mat[i, i]
            if S_ii <= 1e-12:
                singular_mask[i] = True
                f_loo[i] = 0.5
                y_pred_thresh[i] = int(y_L[i])
                y_pred_cmn[i] = int(y_L[i])
                continue

            f_hat_i = y_L[i] - Sy[i] / S_ii
            f_loo[i] = f_hat_i
            y_pred_thresh[i] = 1 if f_hat_i > 0.5 else 0

            M1_loo = sum_f_U + (f_hat_i - y_L[i]) * sum_X_cols[i] + f_hat_i
            M0_loo = (u + 1.0) - M1_loo
            M1_safe = max(M1_loo, 1e-12)
            M0_safe = max(M0_loo, 1e-12)

            l1_loo = np.sum(np.delete(y_L, i) == 1)
            q_loo = (l1_loo + 1.0) / (l - 1.0 + 2.0)

            score1 = q_loo * (f_hat_i / M1_safe)
            score0 = (1.0 - q_loo) * ((1.0 - f_hat_i) / M0_safe)
            y_pred_cmn[i] = 1 if score1 > score0 else 0

        valid_idx = np.flatnonzero(~singular_mask)
        if len(valid_idx) == 0:
            acc_thresh = 0.0
            acc_cmn = 0.0
            brier = 1.0
        else:
            acc_thresh = float(np.mean(y_pred_thresh[valid_idx] == y_L[valid_idx]))
            acc_cmn = float(np.mean(y_pred_cmn[valid_idx] == y_L[valid_idx]))
            brier = float(np.mean((f_loo[valid_idx] - y_L[valid_idx]) ** 2))

        return {
            "acc_loo_thresh": acc_thresh,
            "acc_loo_cmn": acc_cmn,
            "brier_loo": brier,
            "n_singular": int(np.sum(singular_mask)),
            "f_loo": f_loo,
        }

    # =========================================================================
    # EXACT ADJOINT GRADIENT AND SIGMA OPTIMIZATION
    # =========================================================================

    def entropy_and_gradient(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        theta: float | np.ndarray,
        eps: float = 0.01,
        use_cmn: bool = True,
        anisotropic: bool = False,
    ) -> tuple[float, float | np.ndarray]:
        """
        Compute mean binary entropy H and exact gradient w.r.t theta = log(sigma).
        """
        n, m = X.shape
        y_arr = np.asarray(y_labeled)
        is_unlabeled = np.isnan(y_arr) | (y_arr == -1) if np.issubdtype(y_arr.dtype, np.floating) else (y_arr == -1)
        L_idx = np.flatnonzero(~is_unlabeled)
        U_idx = np.flatnonzero(is_unlabeled)
        u = len(U_idx)

        if np.isscalar(theta):
            sig_scalar = float(np.exp(theta))
            sigma_vec = np.full(m, sig_scalar, dtype=np.float64)
        else:
            th_arr = np.asarray(theta, dtype=np.float64)
            if th_arr.ndim == 0 or len(th_arr) == 1:
                sig_scalar = float(np.exp(th_arr.item()))
                sigma_vec = np.full(m, sig_scalar, dtype=np.float64)
            else:
                sigma_vec = np.exp(th_arr)

        W = self.build_weight_matrix(X, sigma=sigma_vec)
        d = np.sum(W, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_d = np.where(d > 1e-14, 1.0 / d, 0.0)
        P = inv_d[:, np.newaxis] * W
        zero_d = np.flatnonzero(d <= 1e-14)
        if len(zero_d) > 0:
            P[zero_d, :] = 1.0 / n

        P_tilde = eps * (1.0 / n) + (1.0 - eps) * P
        P_tilde_UU = P_tilde[np.ix_(U_idx, U_idx)]
        P_tilde_UL = P_tilde[np.ix_(U_idx, L_idx)]

        A = np.eye(u, dtype=np.float64) - P_tilde_UU
        y_L = y_arr[L_idx].astype(np.float64)
        RHS = P_tilde_UL @ y_L

        lu, piv = scipy.linalg.lu_factor(A, check_finite=False)
        f_u = scipy.linalg.lu_solve((lu, piv), RHS, check_finite=False)
        f_u = np.clip(f_u, 1e-12, 1.0 - 1e-12)

        # Cache latest f_u for zero-cost callback tracking
        self._last_fu = f_u.copy()

        inv_ln2 = 1.0 / np.log(2.0)
        if not use_cmn:
            H = self.compute_binary_entropy(f_u)
            g = (inv_ln2 / u) * np.log((1.0 - f_u) / f_u)
        else:
            l1 = np.sum(y_L == 1)
            q = (l1 + 1.0) / (len(y_L) + 2.0)
            f_tilde = self.cmn_probabilities_binary(f_u, q)
            H = self.compute_binary_entropy(f_tilde)

            f_tilde_safe = np.clip(f_tilde, 1e-12, 1.0 - 1e-12)
            c = (inv_ln2 / u) * np.log((1.0 - f_tilde_safe) / f_tilde_safe) * f_tilde_safe * (1.0 - f_tilde_safe)
            sum_c = float(np.sum(c))

            S = float(np.sum(f_u))
            S_clamped = np.clip(S, 1e-10, u - 1e-10)
            denom_S = S_clamped * (u - S_clamped)

            g = c / (f_u * (1.0 - f_u)) - (u / denom_S) * sum_c

        self._last_H = H
        lambda_vec = scipy.linalg.lu_solve((lu, piv), g, trans=1, check_finite=False)

        f_full = np.zeros(n, dtype=np.float64)
        f_full[L_idx] = y_L
        f_full[U_idx] = f_u

        bar_f = P @ f_full
        bar_f_U = bar_f[U_idx]

        d_U = d[U_idx]
        safe_inv_d_U = np.where(d_U > 1e-14, 1.0 / d_U, 0.0)
        coeff = (1.0 - eps) * (lambda_vec * safe_inv_d_U)

        W_U = W[U_idx, :]
        diff_f = f_full[np.newaxis, :] - bar_f_U[:, np.newaxis]
        M = coeff[:, np.newaxis] * W_U * diff_f

        X_U = X[U_idx, :]
        row_sum = M.sum(axis=1)
        col_sum = M.sum(axis=0)

        term1 = row_sum @ (X_U ** 2)
        term2 = np.sum(X_U * (M @ X), axis=0)
        term3 = col_sum @ (X ** 2)

        grad_theta_vec = (2.0 / (sigma_vec ** 2)) * (term1 - 2.0 * term2 + term3)

        if not anisotropic:
            return H, float(np.sum(grad_theta_vec))
        return H, grad_theta_vec

    @staticmethod
    def verify_gradient(
        X: np.ndarray,
        y_labeled: np.ndarray,
        theta: float | np.ndarray | None = None,
        eps: float = 0.01,
        delta: float = 1e-6,
    ) -> bool:
        """
        Verify the exact adjoint gradient against numerical central differences.
        """
        solver = HarmonicSolver(epsilon_reg=1e-10)
        m = X.shape[1]

        configs = [
            ("Isotropic, Raw", False, False, 0.2),
            ("Isotropic, CMN", True, False, 0.2),
            ("Anisotropic, Raw", False, True, np.linspace(-0.2, 0.3, m)),
            ("Anisotropic, CMN", True, True, np.linspace(-0.2, 0.3, m)),
        ]

        all_passed = True
        for desc, use_cmn, anisotropic, th in configs:
            th_val = th if theta is None else theta
            H, g_analytic = solver.entropy_and_gradient(
                X, y_labeled, th_val, eps=eps, use_cmn=use_cmn, anisotropic=anisotropic
            )

            if not anisotropic:
                th_float = float(th_val)
                H_plus, _ = solver.entropy_and_gradient(
                    X, y_labeled, th_float + delta, eps=eps, use_cmn=use_cmn, anisotropic=False
                )
                H_minus, _ = solver.entropy_and_gradient(
                    X, y_labeled, th_float - delta, eps=eps, use_cmn=use_cmn, anisotropic=False
                )
                g_num = (H_plus - H_minus) / (2.0 * delta)
                rel_err = abs(float(g_analytic) - g_num) / (abs(float(g_analytic)) + abs(g_num) + 1e-12)
            else:
                th_arr = np.asarray(th_val, dtype=np.float64).copy()
                g_num = np.zeros(m, dtype=np.float64)
                for d_idx in range(m):
                    th_p = th_arr.copy()
                    th_m = th_arr.copy()
                    th_p[d_idx] += delta
                    th_m[d_idx] -= delta
                    Hp, _ = solver.entropy_and_gradient(
                        X, y_labeled, th_p, eps=eps, use_cmn=use_cmn, anisotropic=True
                    )
                    Hm, _ = solver.entropy_and_gradient(
                        X, y_labeled, th_m, eps=eps, use_cmn=use_cmn, anisotropic=True
                    )
                    g_num[d_idx] = (Hp - Hm) / (2.0 * delta)
                g_an_arr = np.asarray(g_analytic, dtype=np.float64)
                rel_err = np.linalg.norm(g_an_arr - g_num) / (
                    np.linalg.norm(g_an_arr) + np.linalg.norm(g_num) + 1e-12
                )

            passed = rel_err < 1e-4
            if not passed:
                all_passed = False
            print(f"  Gradient Check [{desc}]: rel_err={rel_err:.2e} -> {'PASS' if passed else 'FAIL'}")

        return all_passed

    def optimize_sigma(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        sigma_init: float | np.ndarray,
        eps: float = 0.01,
        use_cmn: bool = True,
        anisotropic: bool = False,
        max_iter: int = 50,
        bounds_scale: tuple[float, float] = (1e-3, 1e3),
        y_true: np.ndarray | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Optimize theta = log(sigma) via L-BFGS-B minimizing mean binary entropy.
        Uses cached f_u for fast O(1) callback tracking.
        """
        from scipy.optimize import minimize

        m = X.shape[1]
        y_arr = np.asarray(y_labeled)
        is_unlabeled = np.isnan(y_arr) | (y_arr == -1) if np.issubdtype(y_arr.dtype, np.floating) else (y_arr == -1)
        L_idx = np.flatnonzero(~is_unlabeled)
        U_idx = np.flatnonzero(is_unlabeled)

        if np.isscalar(sigma_init):
            sig0 = float(sigma_init)
            theta0 = np.array([np.log(sig0)]) if not anisotropic else np.full(m, np.log(sig0))
            ref_sig = sig0
        else:
            sig_arr = np.asarray(sigma_init, dtype=np.float64)
            if not anisotropic:
                sig_scalar = float(sig_arr[0] if len(sig_arr) > 0 else 1.0)
                theta0 = np.array([np.log(sig_scalar)])
                ref_sig = sig_scalar
            else:
                theta0 = np.log(sig_arr)
                ref_sig = float(np.median(sig_arr))

        lb = np.log(ref_sig * bounds_scale[0])
        ub = np.log(ref_sig * bounds_scale[1])
        bounds = [(lb, ub)] * len(theta0)

        history: list[dict[str, Any]] = []
        iter_counter = 0

        def loss_and_grad(th_vec: np.ndarray) -> tuple[float, np.ndarray]:
            th_input = th_vec[0] if not anisotropic else th_vec
            H_val, g_val = self.entropy_and_gradient(
                X, y_labeled, th_input, eps=eps, use_cmn=use_cmn, anisotropic=anisotropic
            )
            g_ret = np.array([g_val], dtype=np.float64) if not anisotropic else np.asarray(g_val, dtype=np.float64)
            return float(H_val), g_ret

        def step_callback(xk: np.ndarray) -> None:
            nonlocal iter_counter
            sig_k = np.exp(xk)
            H_k = self._last_H if self._last_H is not None else 0.5
            step_info: dict[str, Any] = {
                "iter": iter_counter,
                "entropy": H_k,
                "sigma": float(sig_k[0]) if not anisotropic else sig_k.copy(),
            }
            if y_true is not None and len(U_idx) > 0 and self._last_fu is not None:
                f_u = self._last_fu
                q = self.estimate_priors(y_arr[L_idx], 2)[1]
                f_tilde = self.cmn_probabilities_binary(f_u, q)
                pred_cmn_U = (f_tilde >= 0.5).astype(int)
                pred_thresh_U = (f_u >= 0.5).astype(int)
                step_info["acc_cmn"] = float(np.mean(pred_cmn_U == y_true[U_idx]))
                step_info["acc_thresh"] = float(np.mean(pred_thresh_U == y_true[U_idx]))

            history.append(step_info)
            if callback is not None:
                callback(step_info)
            iter_counter += 1

        # Initial point evaluation
        loss_and_grad(theta0)
        step_callback(theta0)

        opt_res = minimize(
            loss_and_grad,
            theta0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=step_callback,
            options={"maxiter": max_iter, "ftol": 1e-6, "gtol": 1e-5},
        )

        sig_opt = np.exp(opt_res.x)
        if not anisotropic:
            sig_opt = float(sig_opt[0])

        return {
            "sigma_opt": sig_opt,
            "entropy_opt": float(opt_res.fun),
            "history": history,
            "success": bool(opt_res.success),
            "message": str(opt_res.message),
        }

    # =========================================================================
    # FIT PREDICT
    # =========================================================================

    def fit_predict(
        self,
        X: np.ndarray,
        y_labeled: np.ndarray,
        decision_rule: Literal["thresh", "cmn"] = "cmn",
        sigma: float | np.ndarray | None = None,
        eps_pagerank: float | None = None,
        compute_cond: bool = False,
    ) -> np.ndarray:
        """
        Fit harmonic solver on (X, y_labeled) and predict labels for all samples.
        """
        n = X.shape[0]
        y_arr = np.asarray(y_labeled)

        if np.issubdtype(y_arr.dtype, np.floating):
            is_unlabeled = np.isnan(y_arr) | (y_arr == -1)
        else:
            is_unlabeled = (y_arr == -1)

        self.L_idx_ = np.flatnonzero(~is_unlabeled)
        self.U_idx_ = np.flatnonzero(is_unlabeled)

        if len(self.L_idx_) == 0:
            raise ValueError("y_labeled must contain at least one labeled point.")

        y_L = y_arr[self.L_idx_]
        self.classes_ = np.unique(y_L)
        n_classes = len(self.classes_)

        F_L = np.zeros((len(self.L_idx_), n_classes), dtype=np.float64)
        for c_idx, c_val in enumerate(self.classes_):
            F_L[y_L == c_val, c_idx] = 1.0
        self.F_L_ = F_L

        self.W_ = self.build_weight_matrix(X, sigma=sigma)

        self.F_U_ = self.solve_harmonic(
            self.W_, self.L_idx_, self.U_idx_, self.F_L_,
            eps_pagerank=eps_pagerank, compute_cond=compute_cond
        )

        self.F_ = np.zeros((n, n_classes), dtype=np.float64)
        self.F_[self.L_idx_] = self.F_L_
        if len(self.U_idx_) > 0:
            self.F_[self.U_idx_] = self.F_U_

        if decision_rule == "thresh":
            y_pred_U = self.predict_thresh(self.F_U_)
        elif decision_rule == "cmn":
            y_pred_U = self.predict_cmn(self.F_U_, y_L)
        else:
            raise ValueError(f"Unknown decision_rule '{decision_rule}'. Choose 'thresh' or 'cmn'.")

        y_pred = np.empty(n, dtype=self.classes_.dtype)
        y_pred[self.L_idx_] = y_L
        if len(self.U_idx_) > 0:
            y_pred[self.U_idx_] = y_pred_U

        return y_pred
