"""
Phase 1: Binary Benchmark on 4 Datasets (Harmonic Function Energy Minimization).

Evaluates HarmonicSolver (dense RBF graph) comparing standard thresholding vs CMN
under fixed parameters across 10 random stratified trials.

Datasets:
- Two Spirals (n=400, l=6, sigma=0.15 * d_med)
- Two Moons (n=400, l=4, sigma=0.25 * d_med)
- Concentric Circles (n=400, l=4, sigma=0.25 * d_med)
- MNIST_bin (n=2000, sigma=0.50 * d_med, with l=10 and l=2)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Headless rendering for automated execution
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

# Shared fixed sigma scale factors: sigma = factor * d_med
FIXED_SIGMA_FACTORS = {
    "spirals": 0.15,
    "moons": 0.25,
    "circles": 0.25,
    "mnist": 0.50,
}


def _save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_phase1_benchmark() -> list[dict]:
    print("=" * 85)
    print("PHASE 1: Harmonic Energy Minimization Binary Benchmark (Thresh vs CMN)")
    print("Zhu, Ghahramani, Lafferty (ICML 2003)")
    print("=" * 85)

    n_trials = 10
    configs = [
        {"name": "Two Spirals", "key": "spirals", "n": 400, "l": 6, "factor": FIXED_SIGMA_FACTORS["spirals"]},
        {"name": "Two Moons", "key": "moons", "n": 400, "l": 4, "factor": FIXED_SIGMA_FACTORS["moons"]},
        {"name": "Concentric Circles", "key": "circles", "n": 400, "l": 4, "factor": FIXED_SIGMA_FACTORS["circles"]},
        {"name": "MNIST_bin (l=10)", "key": "mnist", "n": 2000, "l": 10, "factor": FIXED_SIGMA_FACTORS["mnist"]},
        {"name": "MNIST_bin (l=2)", "key": "mnist", "n": 2000, "l": 2, "factor": FIXED_SIGMA_FACTORS["mnist"]},
    ]

    # Pre-load datasets and compute d_med
    loaded_data = {}
    print("\n[1/4] Loading datasets and computing median distances d_med...")
    for cfg in configs:
        k = cfg["key"]
        if k not in loaded_data:
            X, y, d_med = load_dataset(k, n_samples=cfg["n"], seed=42)
            loaded_data[k] = (X, y, d_med)
            print(f"  Loaded {k}: X={X.shape}, d_med={d_med:.4f}")

    # Store exact sigma values to share with Phase 2
    sigma_dict = {}
    for cfg in configs:
        k = cfg["key"]
        _, _, d_med = loaded_data[k]
        sigma_val = cfg["factor"] * d_med
        sigma_dict[cfg["name"]] = {"sigma": sigma_val, "factor": cfg["factor"], "d_med": d_med}

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "fixed_sigmas.json", "w") as f:
        json.dump(sigma_dict, f, indent=2)

    summary_rows = []
    trial0_toy_data = {}
    mnist_hist_data = {}

    print("\n[2/4] Running 10 stratified trials per dataset...")
    for cfg in configs:
        c_name = cfg["name"]
        k = cfg["key"]
        X, y, d_med = loaded_data[k]
        l_count = cfg["l"]
        sigma = cfg["factor"] * d_med
        n_samples = len(y)
        u_count = n_samples - l_count

        acc_thresh_list = []
        acc_cmn_list = []
        pred1_thresh_list = []
        pred1_cmn_list = []
        cond_list = []

        for trial in range(n_trials):
            seed = 42 + trial * 1000
            L_idx, U_idx = sample_stratified_split(y, l=l_count, u=None, seed=seed)
            y_labeled = np.full(n_samples, -1, dtype=int)
            y_labeled[L_idx] = y[L_idx]
            y_true_U = y[U_idx]

            solver = HarmonicSolver(sigma=sigma, solver_type="cholesky", epsilon_reg=1e-10)
            y_pred_thresh_full = solver.fit_predict(X, y_labeled, decision_rule="thresh")
            f_u_class1 = solver.F_U_[:, 1]

            # Predictions on U
            y_pred_thresh_U = y_pred_thresh_full[U_idx]
            acc_t = accuracy_score(y_true_U, y_pred_thresh_U)

            y_pred_cmn_U = solver.predict_cmn(solver.F_U_, y_labeled[L_idx])
            acc_c = accuracy_score(y_true_U, y_pred_cmn_U)

            acc_thresh_list.append(acc_t)
            acc_cmn_list.append(acc_c)
            pred1_thresh_list.append(np.mean(y_pred_thresh_U == 1))
            pred1_cmn_list.append(np.mean(y_pred_cmn_U == 1))
            cond_list.append(solver.cond_number_ if solver.cond_number_ is not None else np.nan)

            # Save Trial 0 visual data
            if trial == 0:
                if k in ("spirals", "moons", "circles"):
                    trial0_toy_data[c_name] = {
                        "X": X,
                        "y": y,
                        "L_idx": L_idx,
                        "U_idx": U_idx,
                        "f_u": f_u_class1,
                        "y_pred_t": y_pred_thresh_U,
                        "y_pred_c": y_pred_cmn_U,
                        "acc_t": acc_t,
                        "acc_c": acc_c,
                    }
                elif "l=10" in c_name or "l=2" in c_name:
                    q = HarmonicSolver.estimate_priors(y[L_idx], 2)[1]
                    tau_cmn = HarmonicSolver.cmn_effective_threshold(f_u_class1, q)
                    mnist_hist_data[c_name] = {
                        "f_u": f_u_class1,
                        "y_true_U": y_true_U,
                        "tau_cmn": tau_cmn,
                        "acc_t": acc_t,
                        "acc_c": acc_c,
                    }

        row = {
            "dataset": c_name,
            "n": n_samples,
            "l": l_count,
            "u": u_count,
            "d_med": round(d_med, 4),
            "sigma": round(sigma, 4),
            "acc_thresh_mean": float(np.mean(acc_thresh_list)),
            "acc_thresh_std": float(np.std(acc_thresh_list)),
            "acc_cmn_mean": float(np.mean(acc_cmn_list)),
            "acc_cmn_std": float(np.std(acc_cmn_list)),
            "delta_cmn": float(np.mean(acc_cmn_list) - np.mean(acc_thresh_list)),
            "pred1_frac_thresh": float(np.mean(pred1_thresh_list)),
            "pred1_frac_cmn": float(np.mean(pred1_cmn_list)),
            "cond_number_mean": float(np.nanmean(cond_list)),
        }
        summary_rows.append(row)
        print(f"  {c_name:<18} | Thresh: {row['acc_thresh_mean']*100:5.2f}% +/- {row['acc_thresh_std']*100:4.2f}% "
              f"| CMN: {row['acc_cmn_mean']*100:5.2f}% +/- {row['acc_cmn_std']*100:4.2f}% "
              f"| Delta: {row['delta_cmn']*100:+5.2f}%")

    csv_path = metrics_dir / "phase1_summary.csv"
    _save_csv(csv_path, summary_rows)
    print(f"\n[OK] Metrics saved to {csv_path}")

    # =========================================================================
    # PLOT 1: Toy 2D 3x4 Grid (Trial 0)
    # =========================================================================
    print("\n[3/4] Generating Toy 2D 3x4 Grid Figure...")
    toy_order = ["Two Spirals", "Two Moons", "Concentric Circles"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    for row_idx, t_name in enumerate(toy_order):
        d = trial0_toy_data[t_name]
        X, y = d["X"], d["y"]
        L_idx, U_idx = d["L_idx"], d["U_idx"]

        # Col 1: Ground Truth + Labeled Seeds
        ax0 = axes[row_idx, 0]
        ax0.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=18, alpha=0.35, edgecolors="none")
        ax0.scatter(X[L_idx, 0], X[L_idx, 1], c=y[L_idx], cmap="coolwarm", s=130, marker="*",
                    edgecolors="black", linewidths=1.5, label=f"Seeds ($l={len(L_idx)}$)")
        ax0.set_title(f"{t_name}\nGround Truth & Seeds", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.set_aspect("equal")

        # Col 2: Continuous Harmonic Potential
        ax1 = axes[row_idx, 1]
        sc1 = ax1.scatter(X[U_idx, 0], X[U_idx, 1], c=d["f_u"], cmap="viridis", s=18, vmin=0.0, vmax=1.0)
        ax1.scatter(X[L_idx, 0], X[L_idx, 1], c="red", s=80, marker="o", edgecolors="white", linewidths=1.2)
        ax1.set_title(f"Harmonic Potential $f_u(x)$\nClass 1 Probability", fontsize=11, fontweight="bold")
        fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_aspect("equal")

        # Col 3: Threshold Decision
        ax2 = axes[row_idx, 2]
        ax2.scatter(X[U_idx, 0], X[U_idx, 1], c=d["y_pred_t"], cmap="coolwarm", s=18, alpha=0.75, edgecolors="k", linewidths=0.2)
        ax2.scatter(X[L_idx, 0], X[L_idx, 1], c=y[L_idx], cmap="coolwarm", s=130, marker="*", edgecolors="black", linewidths=1.5)
        ax2.set_title(f"Threshold Decision (0.5)\nAcc(U) = {d['acc_t']*100:.1f}%", fontsize=11, fontweight="bold")
        ax2.set_aspect("equal")

        # Col 4: CMN Decision
        ax3 = axes[row_idx, 3]
        ax3.scatter(X[U_idx, 0], X[U_idx, 1], c=d["y_pred_c"], cmap="coolwarm", s=18, alpha=0.75, edgecolors="k", linewidths=0.2)
        ax3.scatter(X[L_idx, 0], X[L_idx, 1], c=y[L_idx], cmap="coolwarm", s=130, marker="*", edgecolors="black", linewidths=1.5)
        ax3.set_title(f"CMN Decision\nAcc(U) = {d['acc_c']*100:.1f}%", fontsize=11, fontweight="bold")
        ax3.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(figures_dir / "phase1_toy2d_grid.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "phase1_toy2d_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {figures_dir / 'phase1_toy2d_grid.png'} and .pdf")

    # =========================================================================
    # PLOT 2: MNIST Potential Histograms (l=10 vs l=2)
    # =========================================================================
    print("\n[4/4] Generating MNIST Potential Histograms...")
    fig_hist, axes_h = plt.subplots(1, 2, figsize=(14, 5))
    h_configs = [("MNIST_bin (l=10)", axes_h[0]), ("MNIST_bin (l=2)", axes_h[1])]

    for title, ax in h_configs:
        hd = mnist_hist_data[title]
        f_u = hd["f_u"]
        y_u = hd["y_true_U"]
        tau_cmn = hd["tau_cmn"]

        bins = np.linspace(0.0, 1.0, 40)
        ax.hist(f_u[y_u == 0], bins=bins, alpha=0.6, color="blue", label="True Class 0 (Digit 1)", density=True)
        ax.hist(f_u[y_u == 1], bins=bins, alpha=0.6, color="red", label="True Class 1 (Digit 2)", density=True)

        ax.axvline(0.5, color="black", linestyle="--", linewidth=2.0, label="Trivial Threshold (0.5)")
        ax.axvline(tau_cmn, color="darkgreen", linestyle="-", linewidth=2.5,
                   label=f"CMN Effective $\\tau_{{CMN}} = {tau_cmn:.3f}$")

        ax.set_title(f"{title}\nAcc: Thresh={hd['acc_t']*100:.1f}% vs CMN={hd['acc_c']*100:.1f}%",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Harmonic Score $f_u(x)$ (Class 1)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper center", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_hist.savefig(figures_dir / "phase1_mnist_potential_hist.png", dpi=300, bbox_inches="tight")
    fig_hist.savefig(figures_dir / "phase1_mnist_potential_hist.pdf", bbox_inches="tight")
    plt.close(fig_hist)
    print(f"  Saved {figures_dir / 'phase1_mnist_potential_hist.png'} and .pdf")

    print("\n" + "=" * 85)
    print("PHASE 1 COMPLETE. Summary Table:")
    print("=" * 85)
    headers = ["Dataset", "n", "l", "u", "sigma", "Thresh Acc", "CMN Acc", "Delta CMN"]
    print(f"{headers[0]:<20} | {headers[1]:<4} | {headers[2]:<3} | {headers[3]:<5} | {headers[4]:<7} | {headers[5]:<10} | {headers[6]:<10} | {headers[7]:<9}")
    print("-" * 85)
    for r in summary_rows:
        t_acc = f"{r['acc_thresh_mean']*100:5.2f}%"
        c_acc = f"{r['acc_cmn_mean']*100:5.2f}%"
        d_acc = f"{r['delta_cmn']*100:+5.2f}%"
        print(f"{r['dataset']:<20} | {r['n']:<4} | {r['l']:<3} | {r['u']:<5} | {r['sigma']:<7.4f} | {t_acc:<10} | {c_acc:<10} | {d_acc:<9}")
    return summary_rows


if __name__ == "__main__":
    run_phase1_benchmark()
