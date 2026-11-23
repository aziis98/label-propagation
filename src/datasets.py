"""
Centralized dataset generation, loading, and splitting utilities.

Datasets:
1. Two Spirals
2. Two Moons
3. Concentric Circles
4. MNIST Binary ("1" vs "2")
(+ loader for future Phase 4: MNIST Multiclass)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import fetch_openml, make_circles, make_moons


def make_two_spirals(
    n_samples: int = 400,
    noise: float = 0.35,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two intertwined spirals dataset.

    Parameters
    ----------
    n_samples : int, default=400
        Total number of samples (split equally between the two spirals).
    noise : float, default=0.35
        Standard deviation of Gaussian noise added to coordinates.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2), dtype=float64
    y : np.ndarray of shape (n_samples,), dtype=int
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n)) * 3 * np.pi
    r_a = 2 * theta + np.pi
    data_a = np.column_stack([np.cos(theta) * r_a, np.sin(theta) * r_a]) + rng.normal(0, noise, (n, 2))
    data_b = np.column_stack([-np.cos(theta) * r_a, -np.sin(theta) * r_a]) + rng.normal(0, noise, (n, 2))
    X = np.vstack([data_a, data_b]).astype(np.float64)
    y = np.hstack([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def make_two_moons(
    n_samples: int = 400,
    noise: float = 0.08,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two interleaving half circles (Moons)."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(int)


def make_concentric_circles(
    n_samples: int = 400,
    factor: float = 0.45,
    noise: float = 0.06,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two concentric circles dataset."""
    X, y = make_circles(n_samples=n_samples, factor=factor, noise=noise, random_state=seed)
    return X.astype(np.float64), y.astype(int)


# Global cache for raw MNIST to avoid re-downloading across calls
_MNIST_RAW_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _get_raw_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Internal cached loader for full MNIST dataset."""
    global _MNIST_RAW_CACHE
    if _MNIST_RAW_CACHE is not None:
        return _MNIST_RAW_CACHE

    try:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    except Exception:
        # Fallback without explicit parser if parser is unsupported
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    y = y.astype(int)
    X = X.astype(np.float64) / 255.0
    _MNIST_RAW_CACHE = (X, y)
    return _MNIST_RAW_CACHE


def load_mnist_binary(
    digits: tuple[int, int] = (1, 2),
    n_samples: int = 2200,
    seed: int = 42,
    downsample_16x16: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load binary MNIST subset filtered to two specified digits.

    Parameters
    ----------
    digits : tuple[int, int], default=(1, 2)
        The two digits to extract. The first digit is remapped to 0, second to 1.
    n_samples : int, default=2200
        Total balanced samples to draw (n_samples // 2 per digit).
    seed : int, default=42
        Random seed for stratified selection.
    downsample_16x16 : bool, default=False
        If True, downsamples 28x28 images to 16x16 via bilinear interpolation.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 784 or 256), dtype=float64 in [0, 1]
    y : np.ndarray of shape (n_samples,), dtype=int in {0, 1}
    """
    X_full, y_full = _get_raw_mnist()
    d0, d1 = digits

    idx0 = np.flatnonzero(y_full == d0)
    idx1 = np.flatnonzero(y_full == d1)

    n_per_class = n_samples // 2
    if len(idx0) < n_per_class or len(idx1) < n_per_class:
        raise ValueError(
            f"Requested {n_per_class} per class, but available: {len(idx0)} for {d0}, {len(idx1)} for {d1}"
        )

    rng = np.random.default_rng(seed)
    chosen0 = rng.choice(idx0, size=n_per_class, replace=False)
    chosen1 = rng.choice(idx1, size=n_per_class, replace=False)

    X0, X1 = X_full[chosen0], X_full[chosen1]
    y0 = np.zeros(n_per_class, dtype=int)
    y1 = np.ones(n_per_class, dtype=int)

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])

    # Shuffle to interleave classes randomly but deterministically
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    if downsample_16x16:
        # Simple 16x16 downsampling via box-filtering / resizing if requested
        images = X.reshape(-1, 28, 28)
        # Rescale using scipy.ndimage if available, or area slicing
        from scipy.ndimage import zoom
        scale = 16.0 / 28.0
        downsampled = zoom(images, (1.0, scale, scale), order=1)
        X = downsampled.reshape(len(X), -1)

    return X.astype(np.float64), y.astype(int)


def load_mnist_multiclass(
    n_samples: int = 4000,
    class_counts: list[int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loader for multiclass MNIST (predisposition for future Phase 4).

    Parameters
    ----------
    n_samples : int, default=4000
        Total samples across all 10 classes.
    class_counts : list of int or None
        Custom distribution of samples per class. If None, balanced.
    seed : int, default=42
        Random seed.
    """
    X_full, y_full = _get_raw_mnist()
    rng = np.random.default_rng(seed)

    if class_counts is None:
        per_class = n_samples // 10
        counts = [per_class] * 10
    else:
        counts = class_counts

    indices = []
    for c in range(10):
        c_idx = np.flatnonzero(y_full == c)
        chosen = rng.choice(c_idx, size=counts[c], replace=False)
        indices.append(chosen)

    all_idx = np.concatenate(indices)
    perm = rng.permutation(len(all_idx))
    all_idx = all_idx[perm]

    return X_full[all_idx], y_full[all_idx]


def compute_median_distance(
    X: np.ndarray,
    max_samples: int = 2000,
    seed: int = 42,
) -> float:
    """
    Compute median Euclidean distance d_med among distinct pairs (i < j).

    Subsamples up to max_samples rows for computational efficiency when n is large.
    """
    n = X.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(seed)
        sub_idx = rng.choice(n, size=max_samples, replace=False)
        X_sub = X[sub_idx]
    else:
        X_sub = X

    dists = pdist(X_sub, metric="euclidean")
    # Exclude exact zeros (identical points) if any
    nonzero_dists = dists[dists > 1e-12]
    if len(nonzero_dists) == 0:
        return 1.0
    return float(np.median(nonzero_dists))


def sample_stratified_split(
    y: np.ndarray,
    l: int,
    u: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select l balanced labeled indices (guaranteeing >= 1 sample per class)
    and u disjoint unlabeled indices.

    If u is None, all remaining points are included in U.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    n_classes = len(classes)

    if l < n_classes:
        raise ValueError(f"l ({l}) must be >= number of classes ({n_classes})")

    # Allocate labeled counts per class as evenly as possible
    base_count = l // n_classes
    rem = l % n_classes

    l_indices_list = []
    available_u_pool = []

    for c_idx, c in enumerate(classes):
        c_all = np.flatnonzero(y == c)
        perm = rng.permutation(c_all)
        count_c = base_count + (1 if c_idx < rem else 0)
        l_indices_list.append(perm[:count_c])
        available_u_pool.append(perm[count_c:])

    L_idx = np.concatenate(l_indices_list)
    U_candidates = np.concatenate(available_u_pool)

    if u is None:
        U_idx = U_candidates
    else:
        if u > len(U_candidates):
            raise ValueError(f"Requested u={u}, but only {len(U_candidates)} remaining samples available.")
        # Draw balanced unlabeled if binary, or uniform permutation
        u_perm = rng.permutation(U_candidates)
        U_idx = u_perm[:u]

    # Deterministic sort for index stability
    return np.sort(L_idx), np.sort(U_idx)


def generate_nested_splits(
    y: np.ndarray,
    l_grid: list[int],
    u_grid: list[int],
    n_trials: int = 10,
    seed: int = 42,
) -> list[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]]:
    """
    Generate nested stratified splits for each trial.

    For every trial t = 0, ..., n_trials - 1, constructs a partitioned ordering of
    samples such that for any (l1, u1) and (l2, u2) with l1 < l2 and u1 < u2:
        L_{l1} subset L_{l2}
        U_{u1} subset U_{u2}
        L_max cap U_max = emptyset

    Returns
    -------
    trials_splits : list of dict
        Length n_trials. Each dict maps (l, u) -> (L_idx, U_idx).
    """
    l_max = max(l_grid)
    u_max = max(u_grid)
    n_total = len(y)
    if l_max + u_max > n_total:
        raise ValueError(f"l_max ({l_max}) + u_max ({u_max}) = {l_max + u_max} exceeds n_total ({n_total})")

    classes = np.unique(y)
    n_classes = len(classes)
    rng_master = np.random.default_rng(seed)

    trials_splits: list[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = []

    for trial in range(n_trials):
        trial_seed = int(rng_master.integers(0, 10_000_000))
        rng = np.random.default_rng(trial_seed)

        # For each class, create a randomized sequence
        l_seq_per_class: list[np.ndarray] = []
        u_seq_per_class: list[np.ndarray] = []

        l_req_per_class = int(np.ceil(l_max / n_classes))
        u_req_per_class = int(np.ceil(u_max / n_classes))

        for c in classes:
            c_indices = rng.permutation(np.flatnonzero(y == c))
            l_seq_per_class.append(c_indices[:l_req_per_class])
            u_seq_per_class.append(c_indices[l_req_per_class : l_req_per_class + u_req_per_class])

        # Interleave classes in the candidate sequences so any prefix of length l is balanced
        l_interleaved: list[int] = []
        for step in range(l_req_per_class):
            for c_idx in range(n_classes):
                if step < len(l_seq_per_class[c_idx]):
                    l_interleaved.append(l_seq_per_class[c_idx][step])

        u_interleaved: list[int] = []
        for step in range(u_req_per_class):
            for c_idx in range(n_classes):
                if step < len(u_seq_per_class[c_idx]):
                    u_interleaved.append(u_seq_per_class[c_idx][step])

        l_master_arr = np.array(l_interleaved)
        u_master_arr = np.array(u_interleaved)

        split_dict: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for l in l_grid:
            for u in u_grid:
                L_sub = np.sort(l_master_arr[:l])
                U_sub = np.sort(u_master_arr[:u])
                split_dict[(l, u)] = (L_sub, U_sub)

        trials_splits.append(split_dict)

    return trials_splits


def load_dataset(
    name: str,
    n_samples: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Central dispatcher returning (X, y, d_med) for any of the 4 benchmark datasets.

    Parameters
    ----------
    name : {'spirals', 'moons', 'circles', 'mnist'}
    n_samples : int or None
        Samples to generate/load.
    seed : int, default=42

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    d_med : float
    """
    name_clean = name.lower().strip()
    if name_clean in ("spirals", "two_spirals", "two spirals"):
        n = n_samples if n_samples is not None else 400
        X, y = make_two_spirals(n_samples=n, noise=0.35, seed=seed)
    elif name_clean in ("moons", "two_moons", "two moons"):
        n = n_samples if n_samples is not None else 400
        X, y = make_two_moons(n_samples=n, noise=0.08, seed=seed)
    elif name_clean in ("circles", "concentric_circles", "concentric circles"):
        n = n_samples if n_samples is not None else 400
        X, y = make_concentric_circles(n_samples=n, factor=0.45, noise=0.06, seed=seed)
    elif name_clean in ("mnist", "mnist_bin", "mnist binary"):
        n = n_samples if n_samples is not None else 2200
        X, y = load_mnist_binary(digits=(1, 2), n_samples=n, seed=seed)
    else:
        raise ValueError(f"Unknown dataset name: '{name}'. Choose from 'spirals', 'moons', 'circles', 'mnist'.")

    d_med = compute_median_distance(X, seed=seed)
    return X, y, d_med
