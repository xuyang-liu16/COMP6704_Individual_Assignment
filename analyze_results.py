"""Reproduce terminal diagnostics without replacing the original timings."""

import hashlib
import json

from experiment import (
    METHODS, ROOT, gradients, initialize, objective, projected_gradient_norm,
    solve, threadpool_limits,
)
import numpy as np


def h_row_gauge(W, H):
    scales = np.linalg.norm(H, axis=1)
    scales[scales == 0] = 1.0
    return W * scales[None, :], H / scales[:, None]


def lin_projected_gradient_norm(V, W, H):
    """Lin's active-set residual, distinct from the unit-step mapping."""
    gw, gh = gradients(V, W, H)
    pw = np.where(W > 0, gw, np.minimum(gw, 0.0))
    ph = np.where(H > 0, gh, np.minimum(gh, 0.0))
    return float(np.sqrt(np.sum(pw * pw) + np.sum(ph * ph)))


def matched_objectives(rows, curves):
    comparisons = []
    for case in dict.fromkeys(row["case"] for row in rows):
        target = float(np.median([
            row["objective"] for row in rows
            if row["case"] == case and row["method"] == "MU"
        ]))
        methods = {}
        for method in METHODS:
            hits = []
            for row in rows:
                if row["case"] != case or row["method"] != method:
                    continue
                history = curves[f"{case}/{method}/{row['seed']}"]
                first = next(
                    (point for point in history if point["objective"] <= target),
                    None,
                )
                if first is not None:
                    hits.append({"seed": row["seed"], **first})
            methods[method] = {
                "reached": len(hits),
                "runs": 5,
                "sweeps_median_over_reached": (
                    float(np.median([point["iteration"] for point in hits]))
                    if hits else None
                ),
                "seconds_median_over_reached": (
                    float(np.median([point["seconds"] for point in hits]))
                    if hits else None
                ),
                "first_saved_checkpoints": hits,
            }
        comparisons.append({"case": case, "objective_target": target, "methods": methods})
    return comparisons


def main():
    results = ROOT / "results"
    rows = json.loads((results / "runs.json").read_text())
    curves = json.loads((results / "curves.json").read_text())
    metadata = json.loads((results / "metadata.json").read_text())
    data = np.load(ROOT / "data/datasets.npz")
    assert len(rows) == 75
    records = []
    counts = {
        gauge: {method: 0 for method in METHODS}
        for gauge in ("W_columns", "H_rows")
    }
    with threadpool_limits(limits=1):
        for row in rows:
            case, method, seed = row["case"], row["method"], row["seed"]
            V = data["digits" if case.startswith("digits") else case]
            description = metadata["datasets"][case]
            digest = hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest()
            assert digest == description["sha256_float64_c_order"]
            W0, H0 = initialize(V, description["rank"], seed)
            result = solve(
                V, W0, H0, method, max_iter=metadata["max_iter"],
                tolerance=metadata["relative_gradient_mapping_tolerance"],
                check_every=metadata["check_every"],
            )
            assert result["iterations"] == row["iterations"], (case, method, seed)
            assert result["converged"] == row["converged"]
            np.testing.assert_allclose(result["objective"], row["objective"],
                                       rtol=1e-9, atol=1e-13)
            np.testing.assert_allclose(result["relative_pg"], row["relative_pg"],
                                       rtol=1e-7, atol=1e-12)
            W, H = result["W"], result["H"]
            initial_mapping = projected_gradient_norm(V, W0, H0)
            final_mapping = projected_gradient_norm(V, W, H)
            lin_initial = lin_projected_gradient_norm(V, W0, H0)
            lin_final = lin_projected_gradient_norm(V, W, H)
            Wh0, Hh0 = h_row_gauge(W0, H0)
            Wh, Hh = h_row_gauge(W, H)
            h_initial = projected_gradient_norm(V, Wh0, Hh0)
            h_final = projected_gradient_norm(V, Wh, Hh)
            product_error = float(np.linalg.norm(W @ H - Wh @ Hh))
            assert product_error <= 1e-12 * np.linalg.norm(V)
            np.testing.assert_allclose(objective(V, Wh, Hh), result["objective"],
                                       rtol=1e-12, atol=1e-14)
            h_relative = h_final / h_initial
            counts["W_columns"][method] += int(result["converged"])
            counts["H_rows"][method] += int(h_relative <= 1e-4)
            records.append({
                "case": case, "method": method, "seed": seed,
                "initial_mapping_norm": initial_mapping,
                "terminal_mapping_norm": final_mapping,
                "relative_mapping_W_columns": final_mapping / initial_mapping,
                "initial_lin_projected_gradient_norm": lin_initial,
                "terminal_lin_projected_gradient_norm": lin_final,
                "relative_lin_projected_gradient": lin_final / lin_initial,
                "initial_mapping_norm_H_rows": h_initial,
                "terminal_mapping_norm_H_rows": h_final,
                "relative_mapping_H_rows": h_relative,
                "gauge_product_error": product_error,
                "W_small_positive_entries": int(np.sum((W > 0) & (W < 1e-4))),
                "W_zero_entries": int(np.sum(W == 0)),
                "W_entries": W.size,
            })
            print(f"Rechecked {case}/{method}/{seed}", flush=True)

    stability = {}
    for method in METHODS:
        chosen = [
            row for row in rows if row["case"] == "digits_r16" and row["method"] == method
        ]
        objectives = [row["objective"] for row in chosen]
        stability[method] = {
            "objective_min": min(objectives),
            "objective_max": max(objectives),
            "objective_range_percent_of_median": float(
                100 * np.ptp(objectives) / np.median(objectives)
            ),
            "sweeps_min": min(row["iterations"] for row in chosen),
            "sweeps_max": max(row["iterations"] for row in chosen),
        }
    rank32 = {
        method: [row["objective"] for row in rows
                 if row["case"] == "digits_r32" and row["method"] == method]
        for method in METHODS
    }
    report = {
        "source_runs_sha256": hashlib.sha256((results / "runs.json").read_bytes()).hexdigest(),
        "source_curves_sha256": hashlib.sha256((results / "curves.json").read_bytes()).hexdigest(),
        "rerun_main_runs": len(records),
        "all_terminal_objectives_residuals_and_iterations_match": True,
        "original_timings_preserved": True,
        "gauge_sensitivity_note": (
            "Both initial and terminal factors are transformed. H-row values "
            "re-evaluate W-column stopping points; no H-row stopping experiment "
            "or alternative algorithm ranking is claimed."
        ),
        "gauge_counts_at_original_terminal_points": counts,
        "terminal_diagnostics": records,
        "digits_r16_initialization_sensitivity": stability,
        "matched_objectives": matched_objectives(rows, curves),
        "matched_objective_note": (
            "Each target is the median terminal MU objective for that setting. "
            "Counts include all five runs; medians include only runs reaching "
            "the target at a saved checkpoint. Checkpoints are ten sweeps apart. "
            "Timing values come from the original runs, not these diagnostic reruns."
        ),
        "digits_r32_best_relative_to_worst_HALS_percent": {
            method: 100 * (min(rank32[method]) / max(rank32["HALS"]) - 1)
            for method in ("MU", "PG")
        },
    }
    (results / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print("Saved diagnostics.json; original runs and timings were not replaced.")


if __name__ == "__main__":
    main()
