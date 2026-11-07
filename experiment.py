"""Reproducible alternating methods for Euclidean NMF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

for thread_variable in (
    "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parent
METHODS = ("MU", "PG", "HALS")


def normalize(W, H):
    """Fix the scaling ambiguity without changing the product."""
    scales = np.linalg.norm(W, axis=0)
    scales[scales == 0] = 1.0
    return W / scales, H * scales[:, None]


def initialize(V, rank, seed):
    rng = np.random.default_rng(seed)
    scale = np.sqrt(V.mean() / rank)
    W = scale * rng.uniform(0.1, 1.0, (V.shape[0], rank))
    H = scale * rng.uniform(0.1, 1.0, (rank, V.shape[1]))
    return normalize(W, H)


def objective(V, W, H):
    residual = W @ H - V
    return 0.5 * np.sum(residual * residual) / np.sum(V * V)


def gradients(V, W, H):
    residual = W @ H - V
    scale = np.sum(V * V)
    return (residual @ H.T) / scale, (W.T @ residual) / scale


def projected_gradient_norm(V, W, H):
    """Continuous unit-step gradient mapping; zero exactly at the KKT conditions."""
    gw, gh = gradients(V, W, H)
    pw = np.minimum(W, gw)
    ph = np.minimum(H, gh)
    return float(np.sqrt(np.sum(pw * pw) + np.sum(ph * ph)))


def step_mu(V, W, H):
    denominator = (W.T @ W) @ H
    # Multiplying before division avoids 0 * inf at zero-locked entries.
    H = np.divide(
        H * (W.T @ V), denominator, out=H.copy(), where=denominator > 0
    )
    denominator = W @ (H @ H.T)
    W = np.divide(
        W * (V @ H.T), denominator, out=W.copy(), where=denominator > 0
    )
    return W, H


def step_pg(V, W, H):
    # Alternating projected gradient; no momentum or acceleration.
    gram_w = W.T @ W
    lip_h = float(np.linalg.eigvalsh(gram_w)[-1])
    if lip_h > 0:
        H = np.maximum(H - (gram_w @ H - W.T @ V) / lip_h, 0.0)
    gram_h = H @ H.T
    lip_w = float(np.linalg.eigvalsh(gram_h)[-1])
    if lip_w > 0:
        W = np.maximum(W - (W @ gram_h - V @ H.T) / lip_w, 0.0)
    return W, H


def step_hals(V, W, H):
    gram_w, cross_h = W.T @ W, W.T @ V
    H = H.copy()
    for j in range(H.shape[0]):
        if gram_w[j, j] > 0:
            H[j] = np.maximum(
                H[j] + (cross_h[j] - gram_w[j] @ H) / gram_w[j, j], 0.0
            )
    gram_h, cross_w = H @ H.T, V @ H.T
    W = W.copy()
    for j in range(W.shape[1]):
        if gram_h[j, j] > 0:
            W[:, j] = np.maximum(
                W[:, j] + (cross_w[:, j] - W @ gram_h[:, j]) / gram_h[j, j],
                0.0,
            )
    return W, H


STEPS = {"MU": step_mu, "PG": step_pg, "HALS": step_hals}


def solve(V, W0, H0, method, max_iter=2000, tolerance=1e-4, check_every=10):
    W, H = W0.copy(), H0.copy()
    pg0 = projected_gradient_norm(V, W, H)
    history = [{
        "iteration": 0,
        "seconds": 0.0,
        "objective": float(objective(V, W, H)),
        "relative_pg": 1.0 if pg0 else 0.0,
    }]
    started = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        W, H = normalize(*STEPS[method](V, W, H))
        if iteration % check_every == 0 or iteration == max_iter:
            value = float(objective(V, W, H))
            pg = projected_gradient_norm(V, W, H)
            relative_pg = pg / pg0 if pg0 else pg
            if not np.isfinite(value + relative_pg):
                raise FloatingPointError(f"{method}: non-finite diagnostic")
            history.append({
                "iteration": iteration,
                "seconds": time.perf_counter() - started,
                "objective": value,
                "relative_pg": float(relative_pg),
            })
            if relative_pg <= tolerance:
                break
    last = history[-1]
    return {
        "W": W,
        "H": H,
        "history": history,
        "iterations": iteration,
        "seconds": last["seconds"],
        "objective": last["objective"],
        "relative_error": float(np.sqrt(2.0 * last["objective"])),
        "relative_pg": last["relative_pg"],
        "converged": bool(last["relative_pg"] <= tolerance),
        "checkpoint_monotone": bool(np.all(
            np.diff([item["objective"] for item in history]) <= 1e-12
        )),
    }


def make_datasets():
    rng = np.random.default_rng(20261107)
    W_true = np.maximum(rng.normal(0.3, 0.6, (80, 8)), 0.0)
    H_true = np.maximum(rng.normal(0.3, 0.6, (8, 500)), 0.0)
    clean = W_true @ H_true
    noisy = np.maximum(
        clean + rng.normal(0.0, 0.05 * clean.mean(), clean.shape), 0.0
    )
    digits = load_digits()
    V_digits = digits.data.T.astype(np.float64) / 16.0
    return {
        "synthetic_clean": (clean, 8),
        "synthetic_noisy": (noisy, 8),
        "digits_r16": (V_digits, 16),
        "digits_r8": (V_digits, 8),
        "digits_r32": (V_digits, 32),
    }, digits.target


def svd_lower_bound(V, rank):
    singular_values = np.linalg.svd(V, compute_uv=False)
    return float(0.5 * np.sum(singular_values[rank:] ** 2) / np.sum(V * V))


def summarize(rows):
    summaries = []
    for case in dict.fromkeys(row["case"] for row in rows):
        for method in METHODS:
            chosen = [
                row for row in rows
                if row["case"] == case and row["method"] == method
            ]
            summaries.append({
                "case": case,
                "method": method,
                "runs": len(chosen),
                "converged": sum(row["converged"] for row in chosen),
                **{
                    key + "_median": float(np.median([row[key] for row in chosen]))
                    for key in ("objective", "relative_error", "relative_pg",
                                "iterations", "seconds")
                },
                "objective_min": min(row["objective"] for row in chosen),
                "objective_max": max(row["objective"] for row in chosen),
                "all_checkpoint_monotone": all(
                    row["checkpoint_monotone"] for row in chosen
                ),
            })
    return summaries


def run(max_iter, seed_count):
    results_dir = ROOT / "results"
    data_dir = ROOT / "data"
    results_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    datasets, labels = make_datasets()
    np.savez_compressed(
        data_dir / "datasets.npz",
        synthetic_clean=datasets["synthetic_clean"][0],
        synthetic_noisy=datasets["synthetic_noisy"][0],
        digits=datasets["digits_r16"][0],
        digits_labels=labels,
    )
    rows, curves, details = [], {}, {}
    with threadpool_limits(limits=1):
        for case, (V, rank) in datasets.items():
            details[case] = {
                "shape": list(V.shape),
                "rank": rank,
                "svd_lower_bound": svd_lower_bound(V, rank),
                "sha256_float64_c_order": hashlib.sha256(
                    np.ascontiguousarray(V).tobytes()
                ).hexdigest(),
            }
            for seed in range(seed_count):
                W0, H0 = initialize(V, rank, seed)
                for method in METHODS:
                    result = solve(V, W0, H0, method, max_iter=max_iter)
                    row = {
                        "case": case, "method": method, "seed": seed,
                        **{
                            key: value for key, value in result.items()
                            if key not in ("W", "H", "history")
                        },
                    }
                    rows.append(row)
                    curves[f"{case}/{method}/{seed}"] = result["history"]
                    if case == "digits_r16" and seed == 0:
                        np.savez_compressed(
                            results_dir / f"digits_{method}_seed0.npz",
                            W=result["W"], H=result["H"],
                        )
                    print(
                        f"{case:18s} {method:4s} seed={seed} "
                        f"k={result['iterations']:4d} "
                        f"F={result['objective']:.6g} "
                        f"PG={result['relative_pg']:.3g} "
                        f"t={result['seconds']:.3f}s",
                        flush=True,
                    )

        # Deliberately erase one observed row in the initial basis.
        V, rank = datasets["synthetic_clean"]
        W0, H0 = initialize(V, rank, 0)
        row_to_lock = int(np.argmax(np.linalg.norm(V, axis=1)))
        W0[row_to_lock] = 0.0
        W0, H0 = normalize(W0, H0)
        floor = float(0.5 * np.sum(V[row_to_lock] ** 2) / np.sum(V ** 2))
        locking = {
            "locked_row": row_to_lock,
            "mu_objective_lower_bound_from_locked_row": floor,
            "runs": [],
        }
        for method in METHODS:
            result = solve(V, W0, H0, method, max_iter=max_iter)
            curves[f"zero_lock/{method}/0"] = result["history"]
            locking["runs"].append({
                "method": method,
                "objective": result["objective"],
                "relative_pg": result["relative_pg"],
                "iterations": result["iterations"],
                "locked_row_norm_at_end": float(
                    np.linalg.norm(result["W"][row_to_lock])
                ),
            })

        # A library solver supplies an additional feasible point, not F*.
        from sklearn.decomposition import NMF
        import warnings
        from sklearn.exceptions import ConvergenceWarning

        library = []
        V, rank = datasets["digits_r16"]
        for seed in range(seed_count):
            W0, H0 = initialize(V, rank, seed)
            estimator = NMF(
                n_components=rank, init="custom", solver="cd",
                tol=1e-6, max_iter=5000, random_state=seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                W = estimator.fit_transform(V, W=W0.copy(), H=H0.copy())
            W, H = normalize(W, estimator.components_)
            library.append({
                "seed": seed,
                "objective": float(objective(V, W, H)),
                "iterations": int(estimator.n_iter_),
                "relative_pg": (
                    projected_gradient_norm(V, W, H)
                    / projected_gradient_norm(V, W0, H0)
                ),
                "convergence_warning": any(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                ),
            })
        threads = threadpool_info()

    summary = summarize(rows)
    metadata = {
        "date": "2026-11-07",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "threadpools_during_run": threads,
        "requested_thread_environment": {
            key: os.environ.get(key) for key in (
                "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
            )
        },
        "thread_limit_note": (
            "Thread limits are requested before NumPy import and for supported "
            "pools via threadpoolctl. Apple Accelerate's effective thread count "
            "is not independently exposed by threadpoolctl on this machine."
        ),
        "max_iter": max_iter,
        "seeds": list(range(seed_count)),
        "relative_gradient_mapping_tolerance": 1e-4,
        "stationarity_measure": (
            "G_W=min(W,grad_W F), G_H=min(H,grad_H F); "
            "unit-step projected gradient mapping for normalized F. "
            "relative_pg=norm(G_k)/norm(G_0), after gauge normalization."
        ),
        "check_every": 10,
        "timing": (
            "One thread requested; wall time includes updates, gauge normalization "
            "and identical diagnostics every 10 sweeps; excludes loading, "
            "SVD bounds, file IO and plotting. Runs are sequential."
        ),
        "datasets": details,
        "zero_lock": locking,
        "library_reference": library,
        "main_run_count": len(rows),
    }
    for name, value in (
        ("runs.json", rows), ("summary.json", summary),
        ("curves.json", curves), ("metadata.json", metadata),
    ):
        (results_dir / name).write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    with (results_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} main runs, 3 zero-lock runs and "
          f"{len(library)} library reference runs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    arguments = parser.parse_args()
    run(arguments.max_iter, arguments.seeds)
