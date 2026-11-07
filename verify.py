"""Independent numerical checks for the NMF calculations."""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from experiment import (
    METHODS, STEPS, gradients, initialize, normalize, objective,
    projected_gradient_norm, solve,
)

ROOT = Path(__file__).resolve().parent


def check():
    rng = np.random.default_rng(1936)
    V = rng.uniform(0.2, 2.0, (12, 15))
    W, H = initialize(V, 4, 10)
    gw, gh = gradients(V, W, H)
    errors = []
    for A, grad in ((W, gw), (H, gh)):
        for _ in range(12):
            index = tuple(rng.integers(size) for size in A.shape)
            original = A[index]
            delta = 1e-6
            A[index] = original + delta
            plus = objective(V, W, H)
            A[index] = original - delta
            minus = objective(V, W, H)
            A[index] = original
            errors.append(abs((plus - minus) / (2 * delta) - grad[index]))
    assert max(errors) < 1e-8, max(errors)

    # Check algebraic consistency with the projection formula.
    mapping_reference = np.sqrt(
        np.sum((W - np.maximum(W - gw, 0.0)) ** 2)
        + np.sum((H - np.maximum(H - gh, 0.0)) ** 2)
    )
    mapping_error = abs(projected_gradient_norm(V, W, H) - mapping_reference)
    assert mapping_error < 1e-12
    near_zero = np.array([0.0, 1e-15, 1.0])
    positive_gradient = np.ones(3)
    assert np.allclose(
        np.minimum(near_zero, positive_gradient), near_zero, atol=0, rtol=0
    )

    scale = np.array([0.01, 0.2, 9.0, 300.0])
    W_scaled, H_scaled = normalize(W * scale, H / scale[:, None])
    gauge_error = np.linalg.norm(W_scaled @ H_scaled - W @ H)
    assert gauge_error < 1e-12

    maximum_increases = {}
    for method in METHODS:
        increases = []
        for seed in range(10):
            W, H = initialize(V, 4, seed)
            for _ in range(20):
                before = objective(V, W, H)
                W, H = normalize(*STEPS[method](V, W, H))
                assert np.min(W) >= 0 and np.min(H) >= 0
                increases.append(objective(V, W, H) - before)
        maximum_increases[method] = float(max(increases))
        assert max(increases) <= 1e-12

    # The fixed-W convex subproblem has independent SciPy NNLS solutions.
    W = rng.uniform(0.1, 2.0, (18, 4))
    V_block = rng.uniform(0.1, 2.0, (18, 9))
    H_reference = np.column_stack([
        nnls(W, V_block[:, j])[0] for j in range(V_block.shape[1])
    ])
    H = rng.uniform(0.1, 1.0, (4, 9))
    A, B = W.T @ W, W.T @ V_block
    for _ in range(5000):
        for j in range(4):
            H[j] = np.maximum(H[j] + (B[j] - A[j] @ H) / A[j, j], 0)
    fixed_block_error = np.linalg.norm(H - H_reference)
    assert fixed_block_error < 1e-8, fixed_block_error

    # A positive rank-one case admits an exact feasible factorization.
    u = rng.uniform(0.1, 2.0, (10, 1))
    v = rng.uniform(0.1, 2.0, (1, 13))
    V_one = u @ v
    W0, H0 = initialize(V_one, 1, 7)
    rank_one = {}
    for method in METHODS:
        result = solve(V_one, W0, H0, method, max_iter=30)
        rank_one[method] = result["objective"]
        assert result["objective"] < 1e-24

    W0, H0 = initialize(V, 4, 6)
    W0[0] = 0.0
    locked = {}
    for method in METHODS:
        result = solve(V, W0, H0, method, max_iter=100)
        locked[method] = float(np.linalg.norm(result["W"][0]))
    assert locked["MU"] == 0
    assert locked["PG"] > 0 and locked["HALS"] > 0

    # A constructed zero block exposes the unsafe divide-before-multiply rule.
    W_zero, H_zero = np.zeros((4, 2)), np.zeros((2, 5))
    V_zero_test = np.ones((4, 5))
    with np.errstate(divide="ignore", invalid="ignore"):
        unsafe_H = H_zero * (
            (W_zero.T @ V_zero_test) / ((W_zero.T @ W_zero) @ H_zero)
        )
    unsafe_nonfinite = bool(np.any(~np.isfinite(unsafe_H)))
    assert unsafe_nonfinite
    boundary_finite = {}
    for method in METHODS:
        W_new, H_new = STEPS[method](V_zero_test, W_zero, H_zero)
        boundary_finite[method] = bool(
            np.all(np.isfinite(W_new)) and np.all(np.isfinite(H_new))
        )
        assert boundary_finite[method]
        assert objective(V_zero_test, W_new, H_new) == 0.5

    result = {
        "finite_difference_max_absolute_error": float(max(errors)),
        "gradient_mapping_vs_projection_error": float(mapping_error),
        "gauge_product_error": float(gauge_error),
        "maximum_one_sweep_objective_increase": maximum_increases,
        "fixed_W_HALS_vs_scipy_nnls_factor_error": float(fixed_block_error),
        "rank_one_objectives": rank_one,
        "zero_lock_final_row_norms": locked,
        "unsafe_divide_before_multiply_nonfinite": unsafe_nonfinite,
        "guarded_zero_block_finite": boundary_finite,
        "passed": True,
    }
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        check()
