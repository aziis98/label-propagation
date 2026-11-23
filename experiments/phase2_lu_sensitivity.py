"""
Phase 2: Joint (l, u) Sensitivity Study on 4 Binary Datasets.

Evaluates CMN Harmonic Function vs 1-NN supervised baseline across 2D (l, u) grids
with nested stratified sampling (generate_nested_splits) over 10 trials.

Pre-computes W_pool for optimal performance.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import (
    compute_median_distance,
    generate_nested_splits,
    load_dataset,
)
from src.harmonic_solver import HarmonicSolver

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


def run_phase2_sensitivity() -> list[dict]:
    print("=" * 85)
    print("PHASE 2: Joint (l, u) Sensitivity & 1-NN Baseline Comparison (Zhu et al. 2003)")
    print("=" * 85)

    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    n_trials = 10
    datasets_specs = [
        {
            "name": "Two Spirals",
            "key": "spirals",
            "n_pool": 600,
            "l_grid": [2, 4, 8, 16, 32],
            "u_grid": [40, 80, 160, 320, 500],
            "factor": FIXED_SIGMA_FACTORS["spirals"],
            "l_slice_curves": [2, 4, 8],
        },
        {
            "name": "Two Moons",
            "key": "moons",
            "n_pool": 600,
            "l_grid": [2, 4, 8, 16, 32],
            "u_grid": [40, 80, 160, 320, 500],
            "factor": FIXED_SIGMA_FACTORS["moons"],
            "l_slice_curves": [2, 4, 8],
        },
        {
            "name": "Concentric Circles",
            "key": "circles",
            "n_pool": 600,
            "l_grid": [2, 4, 8, 16, 32],
            "u_grid": [40, 80, 160, 320, 500],
            "factor": FIXED_SIGMA_FACTORS["circles"],
            "l_slice_curves": [2, 4, 8],
        },
        {
            "name": "MNIST_bin",
            "key": "mnist",
            "n_pool": 2200,
            "l_grid": [2, 6, 10, 20, 40, 80],
            "u_grid": [100, 250, 500, 1000, 2000],
            "factor": FIXED_SIGMA_FACTORS["mnist"],
            "l_slice_curves": [2, 6, 10],
        },
    ]

    all_grid_rows = []
    datasets_results = {}

    for d_spec in datasets_specs:
        d_name = d_spec["name"]
        d_key = d_spec["key"]
        n_pool = d_spec["n_pool"]
        l_grid = d_spec["l_grid"]
        u_grid = d_spec["u_grid"]
        factor = d_spec["factor"]

        print(f"\n--- Processing {d_name} (n_pool={n_pool}) ---")
        X_pool, y_pool, d_med = load_dataset(d_key, n_samples=n_pool, seed=42)
        sigma = factor * d_med
        print(f"  d_med = {d_med:.4f}, fixed sigma = {sigma:.4f}")

        # Precompute full weight matrix on the entire pool for optimal speed
        print("  Precomputing W_pool affinity matrix...")
        solver_base = HarmonicSolver(sigma=sigma)
        W_pool = solver_base.build_weight_matrix(X_pool)

        # Generate nested splits across 10 trials
        print(f"  Generating {n_trials} nested stratified trials...")
        trials_splits = generate_nested_splits(y_pool, l_grid, u_grid, n_trials=n_trials, seed=42)

        # Storage for this dataset: (len(l_grid), len(u_grid), n_trials)
        cmn_tensor = np.zeros((len(l_grid), len(u_grid), n_trials), dtype=np.float64)
        knn_tensor = np.zeros((len(l_grid), len(u_grid), n_trials), dtype=np.float64)

        for l_idx, l in enumerate(l_grid):
            for u_idx, u in enumerate(u_grid):
                for t in range(n_trials):
                    L_sub, U_sub = trials_splits[t][(l, u)]
                    sub_nodes = np.concatenate([L_sub, U_sub])
                    # Re-index to local 0 ... (l + u - 1)
                    local_L = np.arange(l)
                    local_U = np.arange(l, l + u)

                    y_L = y_pool[L_sub]
                    y_U = y_pool[U_sub]

                    # 1. CMN Harmonic Prediction using sliced W_pool sub-block
                    W_sub = W_pool[np.ix_(sub_nodes, sub_nodes)]
                    F_L = np.zeros((l, 2), dtype=np.float64)
                    for c_idx, c_val in enumerate([0, 1]):
                        F_L[y_L == c_val, c_idx] = 1.0

                    F_U = solver_base.solve_harmonic(W_sub, local_L, local_U, F_L)
                    # CMN decision
                    pred_cmn = solver_base.predict_cmn(F_U, y_L)
                    acc_cmn = float(np.mean(pred_cmn == y_U))
                    cmn_tensor[l_idx, u_idx, t] = acc_cmn

                    # 2. 1-NN Supervised Baseline (trained only on X[L_sub], evaluated on X[U_sub])
                    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
                    knn.fit(X_pool[L_sub], y_L)
                    pred_1nn = knn.predict(X_pool[U_sub])
                    acc_1nn = float(np.mean(pred_1nn == y_U))
                    knn_tensor[l_idx, u_idx, t] = acc_1nn

                mean_cmn = float(np.mean(cmn_tensor[l_idx, u_idx]))
                std_cmn = float(np.std(cmn_tensor[l_idx, u_idx]))
                mean_knn = float(np.mean(knn_tensor[l_idx, u_idx]))
                std_knn = float(np.std(knn_tensor[l_idx, u_idx]))

                all_grid_rows.append({
                    "dataset": d_name,
                    "l": l,
                    "u": u,
                    "acc_cmn_mean": mean_cmn,
                    "acc_cmn_std": std_cmn,
                    "acc_1nn_mean": mean_knn,
                    "acc_1nn_std": std_knn,
                })

            print(f"  l={l:2d} | u_max={u_grid[-1]} -> CMN: {np.mean(cmn_tensor[l_idx, -1])*100:5.2f}% "
                  f"vs 1-NN: {np.mean(knn_tensor[l_idx, -1])*100:5.2f}%")

        datasets_results[d_name] = {
            "l_grid": l_grid,
            "u_grid": u_grid,
            "cmn_tensor": cmn_tensor,
            "knn_tensor": knn_tensor,
            "l_slice_curves": d_spec["l_slice_curves"],
        }

    csv_out = metrics_dir / "phase2_lu_grid.csv"
    _save_csv(csv_out, all_grid_rows)
    print(f"\n[OK] Grid metrics saved to {csv_out}")

    # =========================================================================
    # FIGURE 1: 2x2 Heatmaps of Acc_CMN (l, u)
    # =========================================================================
    print("\n[Figure 1/3] Generating 2x2 Heatmaps of Acc_CMN...")
    fig_heat, axes_h = plt.subplots(2, 2, figsize=(14, 11))
    axes_list = axes_h.flatten()

    for idx, d_spec in enumerate(datasets_specs):
        d_name = d_spec["name"]
        ax = axes_list[idx]
        res = datasets_results[d_name]
        l_grid = res["l_grid"]
        u_grid = res["u_grid"]
        means = np.mean(res["cmn_tensor"], axis=2) * 100  # (len(l), len(u))
        stds = np.std(res["cmn_tensor"], axis=2) * 100

        im = ax.imshow(means, cmap="YlGnBu", aspect="auto", vmin=50.0, vmax=100.0)

        # Annotate cells with mean +/- std
        for i in range(len(l_grid)):
            for j in range(len(u_grid)):
                val_str = f"{means[i, j]:.1f}\n±{stds[i, j]:.1f}"
                color = "white" if means[i, j] > 80.0 else "black"
                ax.text(j, i, val_str, ha="center", va="center", color=color, fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(u_grid)))
        ax.set_xticklabels(u_grid)
        ax.set_yticks(range(len(l_grid)))
        ax.set_yticklabels(l_grid)
        ax.set_xlabel("Unlabeled Pool Size $u$", fontsize=10)
        ax.set_ylabel("Labeled Points $l$", fontsize=10)
        ax.set_title(f"{d_name}\nCMN Accuracy (%) on $U$", fontsize=11, fontweight="bold")
        fig_heat.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_heat.savefig(figures_dir / "phase2_heatmaps_lu.png", dpi=300, bbox_inches="tight")
    fig_heat.savefig(figures_dir / "phase2_heatmaps_lu.pdf", bbox_inches="tight")
    plt.close(fig_heat)
    print(f"  Saved {figures_dir / 'phase2_heatmaps_lu.png'} and .pdf")

    # =========================================================================
    # FIGURE 2: 2x2 Horizontal Sections at Fixed u = u_max (Acc vs l)
    # =========================================================================
    print("\n[Figure 2/3] Generating Acc vs l (u = u_max) curves...")
    fig_l, axes_l = plt.subplots(2, 2, figsize=(14, 10))
    axes_list_l = axes_l.flatten()

    for idx, d_spec in enumerate(datasets_specs):
        d_name = d_spec["name"]
        ax = axes_list_l[idx]
        res = datasets_results[d_name]
        l_grid = res["l_grid"]
        u_max = res["u_grid"][-1]

        cmn_u_max = res["cmn_tensor"][:, -1, :] * 100  # (len(l), n_trials)
        knn_u_max = res["knn_tensor"][:, -1, :] * 100

        cmn_m, cmn_s = np.mean(cmn_u_max, axis=1), np.std(cmn_u_max, axis=1)
        knn_m, knn_s = np.mean(knn_u_max, axis=1), np.std(knn_u_max, axis=1)

        ax.plot(l_grid, cmn_m, "o-", color="darkblue", linewidth=2.0, label=f"Harmonic CMN ($u={u_max}$)")
        ax.fill_between(l_grid, cmn_m - cmn_s, cmn_m + cmn_s, color="blue", alpha=0.15)

        ax.plot(l_grid, knn_m, "s--", color="crimson", linewidth=2.0, label="Supervised 1-NN Baseline")
        ax.fill_between(l_grid, knn_m - knn_s, knn_m + knn_s, color="red", alpha=0.15)

        ax.set_xscale("log", base=2)
        ax.set_xticks(l_grid)
        ax.set_xticklabels(l_grid)
        ax.set_xlabel("Number of Labeled Points $l$ (log scale)", fontsize=10)
        ax.set_ylabel("Accuracy on $U$ (%)", fontsize=10)
        ax.set_ylim(45.0, 102.0)
        ax.set_title(f"{d_name}: Acc vs $l$ (at $u={u_max}$)", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_l.savefig(figures_dir / "phase2_curves_vs_l.png", dpi=300, bbox_inches="tight")
    fig_l.savefig(figures_dir / "phase2_curves_vs_l.pdf", bbox_inches="tight")
    plt.close(fig_l)
    print(f"  Saved {figures_dir / 'phase2_curves_vs_l.png'} and .pdf")

    # =========================================================================
    # FIGURE 3: 2x2 Vertical Sections at Small Fixed l (Acc vs u)
    # =========================================================================
    print("\n[Figure 3/3] Generating Acc vs u (small l) curves...")
    fig_u, axes_u = plt.subplots(2, 2, figsize=(14, 10))
    axes_list_u = axes_u.flatten()
    colors = ["darkblue", "teal", "darkviolet"]

    for idx, d_spec in enumerate(datasets_specs):
        d_name = d_spec["name"]
        ax = axes_list_u[idx]
        res = datasets_results[d_name]
        l_grid = res["l_grid"]
        u_grid = res["u_grid"]
        l_slices = res["l_slice_curves"]

        for c_idx, l_target in enumerate(l_slices):
            l_pos = l_grid.index(l_target)
            cmn_slice = res["cmn_tensor"][l_pos, :, :] * 100  # (len(u), n_trials)
            knn_slice = res["knn_tensor"][l_pos, :, :] * 100

            cmn_m, cmn_s = np.mean(cmn_slice, axis=1), np.std(cmn_slice, axis=1)
            knn_overall_mean = float(np.mean(knn_slice))

            col = colors[c_idx % len(colors)]
            ax.plot(u_grid, cmn_m, "o-", color=col, linewidth=2.0, label=f"CMN ($l={l_target}$)")
            ax.fill_between(u_grid, cmn_m - cmn_s, cmn_m + cmn_s, color=col, alpha=0.12)
            # Dashed 1-NN reference horizontal line
            ax.axhline(knn_overall_mean, color=col, linestyle="--", alpha=0.75,
                       label=f"1-NN ($l={l_target}$, avg={knn_overall_mean:.1f}%)")

        ax.set_xlabel("Unlabeled Pool Size $u$", fontsize=10)
        ax.set_ylabel("Accuracy on $U$ (%)", fontsize=10)
        ax.set_title(f"{d_name}: Sensitivity to Unlabeled Size $u$", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_u.savefig(figures_dir / "phase2_curves_vs_u.png", dpi=300, bbox_inches="tight")
    fig_u.savefig(figures_dir / "phase2_curves_vs_u.pdf", bbox_inches="tight")
    plt.close(fig_u)
    print(f"  Saved {figures_dir / 'phase2_curves_vs_u.png'} and .pdf")

    print("\n" + "=" * 85)
    print("PHASE 2 COMPLETE.")
    print("=" * 85)
    return all_grid_rows


if __name__ == "__main__":
    run_phase2_sensitivity()
