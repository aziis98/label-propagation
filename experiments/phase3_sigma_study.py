"""
Phase 3: Sigma Dependency, Automatic Selection & Adjoint Gradient Optimization.

Sub-experiments:
- 3A: 1D Parameter Scan & Physical Regimes (Isotropic sigma)
- 3B: Automatic Sigma Selection (PageRank Entropy vs Kron LOO-CV)
- 3C: Adjoint Gradient Descent Optimization (Isotropic & Anisotropic Feature Selection)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    compute_median_distance,
    load_dataset,
    load_mnist_binary,
    sample_stratified_split,
)
from src.harmonic_solver import HarmonicSolver


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_phase3_study() -> None:
    print("=" * 85)
    print("PHASE 3: Sigma Study, PageRank Entropy, Kron LOO-CV & Adjoint Gradient")
    print("Zhu, Ghahramani, Lafferty (ICML 2003)")
    print("=" * 85)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 0: Automatic Gradient Verification Check
    # -------------------------------------------------------------------------
    print("\n[VERIFY] Running automatic adjoint gradient verification (finite differences)...")
    X_test, y_test, _ = load_dataset("moons", n_samples=40, seed=42)
    L_test, _ = sample_stratified_split(y_test, l=6, u=34, seed=42)
    y_lab_test = np.full(40, -1, dtype=int)
    y_lab_test[L_test] = y_test[L_test]

    passed = HarmonicSolver.verify_gradient(X_test, y_lab_test, eps=0.01, delta=1e-6)
    assert passed, "Adjoint gradient verification failed! Aborting."
    print(">>> PASS: Adjoint gradient matches finite differences with relative error < 1e-4.\n")

    # Load 4 binary datasets
    print("Loading datasets for Phase 3...")
    X_spirals, y_spirals, d_spirals = load_dataset("spirals", n_samples=400, seed=42)
    X_moons, y_moons, d_moons = load_dataset("moons", n_samples=400, seed=42)
    X_circles, y_circles, d_circles = load_dataset("circles", n_samples=400, seed=42)
    X_mnist, y_mnist, d_mnist = load_dataset("mnist", n_samples=2200, seed=42)

    datasets_3a = [
        {"name": "Two Spirals", "X": X_spirals, "y": y_spirals, "d_med": d_spirals, "l": 8, "n_trials": 5},
        {"name": "Two Moons", "X": X_moons, "y": y_moons, "d_med": d_moons, "l": 8, "n_trials": 5},
        {"name": "Concentric Circles", "X": X_circles, "y": y_circles, "d_med": d_circles, "l": 8, "n_trials": 5},
        {"name": "MNIST_bin", "X": X_mnist[:2000], "y": y_mnist[:2000], "d_med": d_mnist, "l": 20, "n_trials": 3},
    ]

    c_grid = np.logspace(-2.0, 1.5, 30)

    # =========================================================================
    # SUB-EXPERIMENT 3A: 1D Parameter Scan & Physical Regimes
    # =========================================================================
    file_3a_csv = metrics_dir / "phase3a_scan_1d.csv"
    if file_3a_csv.exists() and (figures_dir / "phase3a_sigma_scan_1d.png").exists():
        print("=" * 80)
        print("SUB-EXPERIMENT 3A: Already completed, using cached 3A results.")
        print("=" * 80)
    else:
        print("=" * 80)
        print("SUB-EXPERIMENT 3A: 1D Sigma Parameter Scan (Isotropic)")
        print("=" * 80)

        records_3a = []
        plot_data_3a = {}

        for d_item in datasets_3a:
            d_name = d_item["name"]
            X, y, d_med = d_item["X"], d_item["y"], d_item["d_med"]
            l_cnt = d_item["l"]
            n_trials = d_item["n_trials"]
            n_samples = len(y)

            print(f"  Scanning {d_name} (n={n_samples}, l={l_cnt}, d_med={d_med:.4f})...")
            cmn_grid = np.zeros((len(c_grid), n_trials), dtype=np.float64)
            thr_grid = np.zeros((len(c_grid), n_trials), dtype=np.float64)
            ent_grid = np.zeros((len(c_grid), n_trials), dtype=np.float64)
            sing_grid = np.zeros((len(c_grid), n_trials), dtype=bool)

            for t in range(n_trials):
                seed = 42 + t * 500
                L_idx, U_idx = sample_stratified_split(y, l=l_cnt, u=None, seed=seed)
                y_labeled = np.full(n_samples, -1, dtype=int)
                y_labeled[L_idx] = y[L_idx]
                y_true_U = y[U_idx]

                for c_idx, c_val in enumerate(c_grid):
                    sig = c_val * d_med
                    solver = HarmonicSolver(sigma=sig, solver_type="cholesky", epsilon_reg=1e-10)
                    pred_thr = solver.fit_predict(X, y_labeled, decision_rule="thresh")
                    pred_cmn = solver.predict_cmn(solver.F_U_, y_labeled[L_idx])

                    f_u = solver.F_U_[:, 1]
                    q = HarmonicSolver.estimate_priors(y[L_idx], 2)[1]
                    f_tilde = HarmonicSolver.cmn_probabilities_binary(f_u, q)

                    cmn_grid[c_idx, t] = float(np.mean(pred_cmn == y_true_U))
                    thr_grid[c_idx, t] = float(np.mean(pred_thr[U_idx] == y_true_U))
                    ent_grid[c_idx, t] = HarmonicSolver.compute_binary_entropy(f_tilde)
                    sing_grid[c_idx, t] = solver.is_singular_

            for c_idx, c_val in enumerate(c_grid):
                records_3a.append({
                    "dataset": d_name,
                    "c": c_val,
                    "sigma": c_val * d_med,
                    "acc_cmn_mean": float(np.mean(cmn_grid[c_idx])),
                    "acc_thresh_mean": float(np.mean(thr_grid[c_idx])),
                    "entropy_cmn_mean": float(np.mean(ent_grid[c_idx])),
                    "singular_frac": float(np.mean(sing_grid[c_idx])),
                })

            plot_data_3a[d_name] = {
                "c_grid": c_grid,
                "cmn_m": np.mean(cmn_grid, axis=1) * 100,
                "thr_m": np.mean(thr_grid, axis=1) * 100,
                "ent_m": np.mean(ent_grid, axis=1),
                "sing_m": np.mean(sing_grid, axis=1),
            }

        _save_csv(file_3a_csv, records_3a)
        print(f"  [OK] Saved {file_3a_csv}")

        fig_3a, axes_3a = plt.subplots(2, 2, figsize=(14, 10))
        for idx, d_item in enumerate(datasets_3a):
            d_name = d_item["name"]
            ax = axes_3a.flatten()[idx]
            pdata = plot_data_3a[d_name]
            c_vals = pdata["c_grid"]

            ax.plot(c_vals, pdata["cmn_m"], "o-", color="darkblue", linewidth=2.0, label="Acc CMN (%)")
            ax.plot(c_vals, pdata["thr_m"], "s--", color="crimson", linewidth=1.8, label="Acc Thresh (%)")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\sigma / d_{\mathrm{med}}$ (log scale)")
            ax.set_ylabel("Accuracy on $U$ (%)", color="darkblue")
            ax.set_ylim(45.0, 105.0)

            ax_ent = ax.twinx()
            ax_ent.plot(c_vals, pdata["ent_m"], "^-.", color="green", alpha=0.7, label=r"Entropy $H_0(\sigma)$ (bits)")
            ax_ent.set_ylabel("Mean Entropy (bits)", color="green")
            ax_ent.set_ylim(-0.05, 1.05)

            underflow_mask = (c_vals < 0.04) | (pdata["sing_m"] > 0)
            if np.any(underflow_mask):
                c_under = c_vals[underflow_mask]
                ax.axvspan(c_under.min(), c_under.max(), color="grey", alpha=0.15, label="Numerical Underflow")

            ax.set_title(f"{d_name}: 1D Scan over $\\sigma$", fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_3a.savefig(figures_dir / "phase3a_sigma_scan_1d.png", dpi=300, bbox_inches="tight")
        fig_3a.savefig(figures_dir / "phase3a_sigma_scan_1d.pdf", bbox_inches="tight")
        plt.close(fig_3a)
        print(f"  [OK] Saved {figures_dir / 'phase3a_sigma_scan_1d.png'} and .pdf")

    # =========================================================================
    # SUB-EXPERIMENT 3B: PageRank Entropy vs Kron LOO-CV
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUB-EXPERIMENT 3B: PageRank Entropy Smoothing vs Kron LOO-CV")
    print("=" * 80)

    eps_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    plot_data_3b_eps = {}
    plot_data_3b_loo = {}
    records_3b = []

    for d_item in datasets_3a:
        d_name = d_item["name"]
        X, y, d_med = d_item["X"], d_item["y"], d_item["d_med"]
        n_samples = len(y)
        l_small = 6 if "MNIST" not in d_name else 10
        l_mod = 20 if "MNIST" not in d_name else 40

        print(f"  Analyzing {d_name} (reusing W per sigma)...")

        L_small, U_small = sample_stratified_split(y, l=l_small, u=None, seed=42)
        y_lab_small = np.full(n_samples, -1, dtype=int)
        y_lab_small[L_small] = y[L_small]
        y_L_s = y[L_small]
        F_L_s = np.zeros((l_small, 2), dtype=np.float64)
        for c_idx, c_val in enumerate([0, 1]):
            F_L_s[y_L_s == c_val, c_idx] = 1.0

        L_mod, U_mod = sample_stratified_split(y, l=l_mod, u=None, seed=42)
        y_lab_mod = np.full(n_samples, -1, dtype=int)
        y_lab_mod[L_mod] = y[L_mod]

        solver_loo = HarmonicSolver(epsilon_reg=1e-10)
        ent_by_eps = {eps_val: [] for eps_val in eps_list}
        acc_u_curve = []
        loo_small_curve = []
        loo_mod_curve = []

        q_small = HarmonicSolver.estimate_priors(y_L_s, 2)[1]

        for c_val in c_grid:
            sig = c_val * d_med
            # Build W ONCE per c_val and reuse across all eps and LOO-CVs!
            W = solver_loo.build_weight_matrix(X, sigma=sig)

            # 1. PageRank Entropy curves
            for eps_val in eps_list:
                f_u_mat = solver_loo.solve_harmonic(W, L_small, U_small, F_L_s, eps_pagerank=eps_val)
                f_u1 = f_u_mat[:, 1]
                f_tilde = HarmonicSolver.cmn_probabilities_binary(f_u1, q_small)
                ent_by_eps[eps_val].append(HarmonicSolver.compute_binary_entropy(f_tilde))

            # 2. Oracle accuracy on U
            f_u_std = solver_loo.solve_harmonic(W, L_small, U_small, F_L_s, eps_pagerank=0.0)
            pred_cmn = solver_loo.predict_cmn(f_u_std, y_L_s, classes=np.array([0, 1]))
            acc_u_curve.append(float(np.mean(pred_cmn == y[U_small])))

            # 3. Kron LOO on small and moderate (reusing W)
            res_s = solver_loo.kron_loo_cv(X, y_lab_small, W=W)
            loo_small_curve.append(res_s["acc_loo_cmn"])

            res_m = solver_loo.kron_loo_cv(X, y_lab_mod, W=W)
            loo_mod_curve.append(res_m["acc_loo_cmn"])

        plot_data_3b_eps[d_name] = ent_by_eps

        acc_u_arr = np.array(acc_u_curve)
        loo_s_arr = np.array(loo_small_curve)
        loo_m_arr = np.array(loo_mod_curve)
        ent_01_arr = np.array(ent_by_eps[0.01])

        opt_oracle_idx = int(np.argmax(acc_u_arr))
        valid_range = np.where(c_grid >= 0.05)[0]
        opt_ent_idx = valid_range[int(np.argmin(ent_01_arr[valid_range]))]
        opt_loo_idx = int(np.argmax(loo_s_arr))

        sigma_oracle = float(c_grid[opt_oracle_idx] * d_med)
        sigma_ent = float(c_grid[opt_ent_idx] * d_med)
        sigma_loo = float(c_grid[opt_loo_idx] * d_med)

        records_3b.append({
            "dataset": d_name,
            "sigma_star_ent": sigma_ent,
            "acc_u_at_sigma_ent": acc_u_arr[opt_ent_idx],
            "sigma_star_loo": sigma_loo,
            "acc_u_at_sigma_loo": acc_u_arr[opt_loo_idx],
            "sigma_star_oracle": sigma_oracle,
            "acc_u_at_sigma_oracle": acc_u_arr[opt_oracle_idx],
        })

        plot_data_3b_loo[d_name] = {
            "acc_u": acc_u_arr * 100,
            "loo_s": loo_s_arr * 100,
            "loo_m": loo_m_arr * 100,
            "ent_01": ent_01_arr,
            "opt_oracle_c": c_grid[opt_oracle_idx],
            "opt_ent_c": c_grid[opt_ent_idx],
            "opt_loo_c": c_grid[opt_loo_idx],
        }

    _save_csv(metrics_dir / "phase3b_selection_comparison.csv", records_3b)
    print(f"  [OK] Saved {metrics_dir / 'phase3b_selection_comparison.csv'}")

    # Plot 3B-1: PageRank Epsilon Curves
    fig_pr, axes_pr = plt.subplots(2, 2, figsize=(14, 10))
    eps_colors = ["black", "purple", "blue", "teal", "red"]
    for idx, d_item in enumerate(datasets_3a):
        d_name = d_item["name"]
        ax = axes_pr.flatten()[idx]
        ent_dict = plot_data_3b_eps[d_name]
        for ep_idx, ep_val in enumerate(eps_list):
            lbl = r"$\epsilon = 0$ (No Reg)" if ep_val == 0.0 else rf"$\epsilon = {ep_val}$"
            ax.plot(c_grid, ent_dict[ep_val], "o-", color=eps_colors[ep_idx], linewidth=1.7, label=lbl)

        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma / d_{\mathrm{med}}$ (log scale)")
        ax.set_ylabel(r"Mean Binary Entropy $H_\epsilon(\sigma)$ (bits)")
        ax.set_title(rf"{d_name}: Disappearance of Trivial Minima as $\epsilon > 0$", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_pr.savefig(figures_dir / "phase3b_pagerank_epsilon.png", dpi=300, bbox_inches="tight")
    fig_pr.savefig(figures_dir / "phase3b_pagerank_epsilon.pdf", bbox_inches="tight")
    plt.close(fig_pr)
    print(f"  [OK] Saved {figures_dir / 'phase3b_pagerank_epsilon.png'} and .pdf")

    # Plot 3B-2: Entropy vs Kron LOO-CV vs Acc_U
    fig_loo, axes_loo = plt.subplots(2, 2, figsize=(14, 10))
    for idx, d_item in enumerate(datasets_3a):
        d_name = d_item["name"]
        ax = axes_loo.flatten()[idx]
        p = plot_data_3b_loo[d_name]

        ax.plot(c_grid, p["acc_u"], "o-", color="darkblue", linewidth=2.2, label="Oracle Acc $U$ (%)")
        ax.plot(c_grid, p["loo_s"], "^--", color="crimson", linewidth=1.8, label="Kron LOO Acc ($l$ small)")
        ax.plot(c_grid, p["loo_m"], "s:", color="darkorange", linewidth=1.8, label="Kron LOO Acc ($l$ moderate)")

        ax.axvline(p["opt_oracle_c"], color="darkblue", linestyle="--", alpha=0.7, label=r"$\sigma^*_{\mathrm{oracle}}$")
        ax.axvline(p["opt_ent_c"], color="green", linestyle="-.", alpha=0.8, label=r"$\sigma^*_{\mathrm{ent}}$")
        ax.axvline(p["opt_loo_c"], color="crimson", linestyle=":", alpha=0.7, label=r"$\sigma^*_{\mathrm{LOO}}$")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma / d_{\mathrm{med}}$ (log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(40.0, 105.0)

        ax_h = ax.twinx()
        ax_h.plot(c_grid, p["ent_01"], "-", color="green", alpha=0.4, linewidth=1.5, label=r"$H_{0.01}(\sigma)$")
        ax_h.set_ylabel("Entropy (bits)", color="green")
        ax_h.set_ylim(-0.05, 1.05)

        ax.set_title(f"{d_name}: Selection Comparison (Entropy vs Kron LOO)", fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_loo.savefig(figures_dir / "phase3b_entropy_vs_kron_loo.png", dpi=300, bbox_inches="tight")
    fig_loo.savefig(figures_dir / "phase3b_entropy_vs_kron_loo.pdf", bbox_inches="tight")
    plt.close(fig_loo)
    print(f"  [OK] Saved {figures_dir / 'phase3b_entropy_vs_kron_loo.png'} and .pdf")

    # =========================================================================
    # SUB-EXPERIMENT 3C: Adjoint Gradient Descent & Feature Selection
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUB-EXPERIMENT 3C: Gradient Descent from Multiple Inits & Anisotropic Selection")
    print("=" * 80)

    # 1. Isotropic Initial Sensitivity (5 inits, max_iter=25)
    print("  1) Isotropic Sensitivity across 5 initializations (max_iter=25)...")
    c_inits = [0.15, 0.5, 1.0, 3.0, 6.0]
    records_3c_init = []
    plot_data_3c_init = {}

    for d_item in datasets_3a:
        d_name = d_item["name"]
        X, y, d_med = d_item["X"], d_item["y"], d_item["d_med"]
        l_cnt = d_item["l"]
        L_idx, _ = sample_stratified_split(y, l=l_cnt, u=None, seed=42)
        y_labeled = np.full(len(y), -1, dtype=int)
        y_labeled[L_idx] = y[L_idx]

        plot_data_3c_init[d_name] = {}
        for c_init in c_inits:
            sig0 = c_init * d_med
            solver_opt = HarmonicSolver()
            res_opt = solver_opt.optimize_sigma(
                X, y_labeled, sigma_init=sig0, eps=0.01, use_cmn=True, anisotropic=False,
                max_iter=25, y_true=y
            )

            hist = res_opt["history"]
            plot_data_3c_init[d_name][c_init] = hist
            for step_h in hist:
                records_3c_init.append({
                    "dataset": d_name,
                    "c_init": c_init,
                    "iter": step_h["iter"],
                    "entropy": step_h["entropy"],
                    "sigma": step_h["sigma"],
                    "sigma_ratio": step_h["sigma"] / d_med,
                    "acc_cmn": step_h.get("acc_cmn", np.nan),
                })

    _save_csv(metrics_dir / "phase3c_init_trajectories.csv", records_3c_init)
    print(f"  [OK] Saved {metrics_dir / 'phase3c_init_trajectories.csv'}")

    fig_init, axes_init = plt.subplots(2, 2, figsize=(14, 10))
    init_colors = ["blue", "teal", "green", "orange", "crimson"]

    for idx, d_item in enumerate(datasets_3a):
        d_name = d_item["name"]
        ax = axes_init.flatten()[idx]
        d_med = d_item["d_med"]
        hist_dict = plot_data_3c_init[d_name]

        for c_idx, c_init in enumerate(c_inits):
            hist = hist_dict[c_init]
            iters = [h["iter"] for h in hist]
            sig_ratios = [h["sigma"] / d_med for h in hist]
            ax.plot(iters, sig_ratios, "o-", color=init_colors[c_idx], linewidth=1.8,
                    label=rf"$\sigma^{{(0)}} = {c_init} d_{{med}}$")

        ax.set_yscale("log")
        ax.set_xlabel("Optimization Iteration $t$")
        ax.set_ylabel(r"$\sigma(t) / d_{\mathrm{med}}$ (log scale)")
        ax.set_title(f"{d_name}: Convergence to Optimal Bandwidth", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_init.savefig(figures_dir / "phase3c_init_sensitivity.png", dpi=300, bbox_inches="tight")
    fig_init.savefig(figures_dir / "phase3c_init_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig_init)
    print(f"  [OK] Saved {figures_dir / 'phase3c_init_sensitivity.png'} and .pdf")

    # -------------------------------------------------------------------------
    # 2. Anisotropic Feature Selection: Two Moons + Noise Feature z ~ N(0, 1)
    # -------------------------------------------------------------------------
    print("\n  2) Anisotropic Feature Selection on Two Moons + Noise Dimension z ~ N(0, 1)...")
    X_moons_std = (X_moons - np.mean(X_moons, axis=0)) / np.std(X_moons, axis=0)
    rng_noise = np.random.default_rng(42)
    z_noise = rng_noise.normal(0.0, 1.0, size=(len(X_moons), 1))
    X_moons_3d = np.hstack([X_moons_std, z_noise])

    d_med_3d = compute_median_distance(X_moons_3d, seed=42)
    L_moons, _ = sample_stratified_split(y_moons, l=8, u=None, seed=42)
    y_lab_moons = np.full(len(y_moons), -1, dtype=int)
    y_lab_moons[L_moons] = y_moons[L_moons]

    sig0_vec = np.full(3, 0.5 * d_med_3d)
    solver_ani = HarmonicSolver()
    res_ani = solver_ani.optimize_sigma(
        X_moons_3d, y_lab_moons, sigma_init=sig0_vec, eps=0.01, use_cmn=True, anisotropic=True,
        max_iter=35, y_true=y_moons
    )

    ani_hist = res_ani["history"]
    ani_records = []
    for step_h in ani_hist:
        s_vec = step_h["sigma"]
        ani_records.append({
            "iter": step_h["iter"],
            "entropy": step_h["entropy"],
            "sigma_x": s_vec[0],
            "sigma_y": s_vec[1],
            "sigma_z_noise": s_vec[2],
            "acc_cmn": step_h.get("acc_cmn", np.nan),
        })

    _save_csv(metrics_dir / "phase3c_toy_anisotropic.csv", ani_records)
    print(f"  [OK] Saved {metrics_dir / 'phase3c_toy_anisotropic.csv'}")

    fig_noise, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
    iters_ani = [r["iter"] for r in ani_records]
    ax_s.plot(iters_ani, [r["sigma_x"] for r in ani_records], "o-", color="blue", linewidth=2.0, label=r"Feature $x$ ($\sigma_x$)")
    ax_s.plot(iters_ani, [r["sigma_y"] for r in ani_records], "s-", color="teal", linewidth=2.0, label=r"Feature $y$ ($\sigma_y$)")
    ax_s.plot(iters_ani, [r["sigma_z_noise"] for r in ani_records], "^-", color="crimson", linewidth=2.5, label=r"Pure Noise $z$ ($\sigma_z \to \infty$)")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("Iteration $t$")
    ax_s.set_ylabel(r"Bandwidth $\sigma_d$ (log scale)")
    ax_s.set_title(r"Anisotropic Feature Selection: Divergence of Noise $\sigma_z$", fontsize=11, fontweight="bold")
    ax_s.legend(loc="center left", fontsize=10)
    ax_s.grid(True, alpha=0.3)

    ax_e.plot(iters_ani, [r["entropy"] for r in ani_records], "d-", color="green", linewidth=2.0, label=r"Entropy $H_{0.01}(t)$")
    ax_e_acc = ax_e.twinx()
    ax_e_acc.plot(iters_ani, [r["acc_cmn"] * 100 for r in ani_records], "o--", color="darkblue", linewidth=2.0, label="Acc CMN (%)")
    ax_e.set_xlabel("Iteration $t$")
    ax_e.set_ylabel("Entropy (bits)", color="green")
    ax_e_acc.set_ylabel("Accuracy on $U$ (%)", color="darkblue")
    ax_e.set_title("Optimization Objective and Generalization Accuracy", fontsize=11, fontweight="bold")
    ax_e.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_noise.savefig(figures_dir / "phase3c_anisotropic_toy_noise.png", dpi=300, bbox_inches="tight")
    fig_noise.savefig(figures_dir / "phase3c_anisotropic_toy_noise.pdf", bbox_inches="tight")
    plt.close(fig_noise)
    print(f"  [OK] Saved {figures_dir / 'phase3c_anisotropic_toy_noise.png'} and .pdf")

    # -------------------------------------------------------------------------
    # 3. Anisotropic Feature Selection on MNIST_bin (784 dimensions)
    # -------------------------------------------------------------------------
    print("\n  3) MNIST_bin Anisotropic 784-parameter optimization (Paper Exp 6, Table 1)...")
    n_mnist = 2200
    l_mnist = 92
    L_mn, U_mn = sample_stratified_split(y_mnist, l=l_mnist, u=None, seed=42)
    y_lab_mn = np.full(n_mnist, -1, dtype=int)
    y_lab_mn[L_mn] = y_mnist[L_mn]

    sig0_scalar = 0.50 * d_mnist
    sig0_vec_784 = np.full(784, sig0_scalar)

    print(f"     Starting 784-dimensional L-BFGS-B optimization from uniform sigma={sig0_scalar:.4f}...")
    solver_mnist = HarmonicSolver()

    pred_init = solver_mnist.fit_predict(X_mnist, y_lab_mn, decision_rule="cmn", sigma=sig0_vec_784, eps_pagerank=0.01)
    acc_cmn_init = float(np.mean(pred_init[U_mn] == y_mnist[U_mn]))
    acc_thr_init = float(np.mean(solver_mnist.predict_thresh(solver_mnist.F_U_) == y_mnist[U_mn]))
    f_init = solver_mnist.F_U_[:, 1]
    q_init = HarmonicSolver.estimate_priors(y_mnist[L_mn], 2)[1]
    H_init = HarmonicSolver.compute_binary_entropy(HarmonicSolver.cmn_probabilities_binary(f_init, q_init))

    print(f"     [Initial] Entropy: {H_init:.4f} bits | Acc CMN: {acc_cmn_init*100:.2f}% | Acc Thresh: {acc_thr_init*100:.2f}%")

    res_mnist_ani = solver_mnist.optimize_sigma(
        X_mnist, y_lab_mn, sigma_init=sig0_vec_784, eps=0.01, use_cmn=True, anisotropic=True,
        max_iter=25, y_true=y_mnist
    )

    sig_opt_784 = res_mnist_ani["sigma_opt"]
    pred_final = solver_mnist.fit_predict(X_mnist, y_lab_mn, decision_rule="cmn", sigma=sig_opt_784, eps_pagerank=0.01)
    acc_cmn_final = float(np.mean(pred_final[U_mn] == y_mnist[U_mn]))
    acc_thr_final = float(np.mean(solver_mnist.predict_thresh(solver_mnist.F_U_) == y_mnist[U_mn]))
    f_final = solver_mnist.F_U_[:, 1]
    H_final = HarmonicSolver.compute_binary_entropy(HarmonicSolver.cmn_probabilities_binary(f_final, q_init))

    print(f"     [Final]   Entropy: {H_final:.4f} bits | Acc CMN: {acc_cmn_final*100:.2f}% | Acc Thresh: {acc_thr_final*100:.2f}%")
    print(f"     Sigma range: min={sig_opt_784.min():.4f}, median={np.median(sig_opt_784):.4f}, max={sig_opt_784.max():.4f}")

    mnist_summary = [{
        "dataset": "MNIST_bin (1 vs 2)",
        "n": n_mnist,
        "l": l_mnist,
        "u": len(U_mn),
        "entropy_init_bits": H_init,
        "entropy_final_bits": H_final,
        "acc_cmn_init": acc_cmn_init,
        "acc_cmn_final": acc_cmn_final,
        "acc_thresh_init": acc_thr_init,
        "acc_thresh_final": acc_thr_final,
        "sigma_min": float(sig_opt_784.min()),
        "sigma_median": float(np.median(sig_opt_784)),
        "sigma_max": float(sig_opt_784.max()),
    }]
    _save_csv(metrics_dir / "phase3c_mnist_anisotropic_summary.csv", mnist_summary)
    print(f"  [OK] Saved {metrics_dir / 'phase3c_mnist_anisotropic_summary.csv'}")

    fig_mn, axes_mn = plt.subplots(1, 4, figsize=(18, 4.5))

    mean_digit1 = np.mean(X_mnist[y_mnist == 0], axis=0).reshape(28, 28)
    axes_mn[0].imshow(mean_digit1, cmap="gray")
    axes_mn[0].set_title('Mean Digit "1"', fontsize=11, fontweight="bold")
    axes_mn[0].axis("off")

    mean_digit2 = np.mean(X_mnist[y_mnist == 1], axis=0).reshape(28, 28)
    axes_mn[1].imshow(mean_digit2, cmap="gray")
    axes_mn[1].set_title('Mean Digit "2"', fontsize=11, fontweight="bold")
    axes_mn[1].axis("off")

    sigma_map_2d = sig_opt_784.reshape(28, 28)
    im_sig = axes_mn[2].imshow(sigma_map_2d, cmap="magma")
    axes_mn[2].set_title(r"Learned $\sigma_d$ Map (28$\times$28)" + "\nLow $\\sigma$ = Discriminative",
                         fontsize=11, fontweight="bold")
    fig_mn.colorbar(im_sig, ax=axes_mn[2], fraction=0.046, pad=0.04)
    axes_mn[2].axis("off")

    mn_hist = res_mnist_ani["history"]
    mn_iters = [h["iter"] for h in mn_hist]
    mn_ents = [h["entropy"] for h in mn_hist]
    mn_accs = [h.get("acc_cmn", np.nan) * 100 for h in mn_hist]

    axes_mn[3].plot(mn_iters, mn_ents, "o-", color="green", linewidth=2.0, label="Entropy $H(t)$ (bits)")
    ax_mn_acc = axes_mn[3].twinx()
    ax_mn_acc.plot(mn_iters, mn_accs, "s--", color="darkblue", linewidth=2.0, label="Acc CMN (%)")
    axes_mn[3].set_xlabel("Iteration $t$")
    axes_mn[3].set_ylabel("Entropy (bits)", color="green")
    ax_mn_acc.set_ylabel("Accuracy on $U$ (%)", color="darkblue")
    axes_mn[3].set_title("Optimization Dynamics on MNIST", fontsize=11, fontweight="bold")
    axes_mn[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_mn.savefig(figures_dir / "phase3c_mnist_sigma_map.png", dpi=300, bbox_inches="tight")
    fig_mn.savefig(figures_dir / "phase3c_mnist_sigma_map.pdf", bbox_inches="tight")
    plt.close(fig_mn)
    print(f"  [OK] Saved {figures_dir / 'phase3c_mnist_sigma_map.png'} and .pdf")

    print("\n" + "=" * 85)
    print("PHASE 3 COMPLETE.")
    print("=" * 85)


if __name__ == "__main__":
    run_phase3_study()
