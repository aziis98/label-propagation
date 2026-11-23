"""
Main entry point for Harmonic Energy Minimization Experiments.
Reference: Xiaojin Zhu, Zoubin Ghahramani, John Lafferty (ICML 2003).

Runs Phases 1, 2, and 3:
- Phase 1: Binary benchmark across 4 datasets (Thresholding vs CMN)
- Phase 2: Joint (l, u) sensitivity analysis and 1-NN supervised baseline
- Phase 3: Sigma parameter study (1D scan, PageRank entropy vs Kron LOO-CV, adjoint gradient optimization)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1_binary_benchmark import run_phase1_benchmark
from experiments.phase2_lu_sensitivity import run_phase2_sensitivity
from experiments.phase3_sigma_study import run_phase3_study


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Harmonic Energy Minimization benchmark and studies (Fasi 1-3)."
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["1", "2", "3", "all"],
        help="Specify which phase to run ('1', '2', '3', or 'all'). Default: 'all'.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 90)
    print(" HARMONIC ENERGY MINIMIZATION — SEMI-SUPERVISED LEARNING (ICML 2003)")
    print("=" * 90)

    if args.phase in ("1", "all"):
        print("\n>>> STARTING PHASE 1: Binary Benchmark (Thresh vs CMN)...")
        run_phase1_benchmark()

    if args.phase in ("2", "all"):
        print("\n>>> STARTING PHASE 2: (l, u) Sensitivity & 1-NN Comparison...")
        run_phase2_sensitivity()

    if args.phase in ("3", "all"):
        print("\n>>> STARTING PHASE 3: Sigma Study, PageRank Entropy, Kron LOO & Adjoint Gradient...")
        run_phase3_study()

    print("\n" + "=" * 90)
    print(" ALL REQUESTED PHASES COMPLETED SUCCESSFULLY.")
    print(" Results stored in 'results/metrics/' and 'results/figures/'.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
