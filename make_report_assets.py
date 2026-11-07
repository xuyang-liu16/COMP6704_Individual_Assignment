"""Generate every report figure, table, and numeric macro from saved results."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiment import METHODS, ROOT

COLORS = {"MU": "#4285F4", "PG": "#DB4437", "HALS": "#0F9D58"}
NAMES = {
    "synthetic_clean": "Exact synthetic, rank 8",
    "synthetic_noisy": "Noisy synthetic, rank 8",
    "digits_r16": "Digits, rank 16",
}


def style():
    plt.rcParams.update({
        "font.family": ["serif", "DejaVu Sans"],
        "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{extension}", facecolor="white")
    plt.close(fig)


def legend(fig, location="upper center"):
    fig.legend(
        [Line2D([], [], color=COLORS[m], lw=2) for m in METHODS],
        METHODS, loc=location, ncol=3, frameon=False,
    )


def convergence(curves, metadata):
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.25), sharex="col")
    for column, case in enumerate(NAMES):
        lower_bound = metadata["datasets"][case]["svd_lower_bound"]
        axes[0, column].set_title(NAMES[case], pad=8)
        for method in METHODS:
            for seed in range(5):
                history = curves[f"{case}/{method}/{seed}"]
                iterations = [v["iteration"] for v in history]
                values = np.array([v["objective"] for v in history]) - lower_bound
                residuals = [v["relative_pg"] for v in history]
                alpha, width = (1.0, 1.6) if seed == 0 else (0.24, 0.9)
                axes[0, column].semilogy(
                    iterations, values, color=COLORS[method], alpha=alpha, lw=width
                )
                axes[1, column].semilogy(
                    iterations, residuals, color=COLORS[method],
                    alpha=alpha, lw=width,
                )
        axes[1, column].axhline(1e-4, color="#555555", ls=":", lw=1)
        axes[1, column].set_xlabel("Sweeps")
        for row in range(2):
            axes[row, column].set_xlim(0, 2000)
            axes[row, column].set_xticks([0, 1000, 2000])
    axes[0, 0].set_ylabel(r"$F-F_{\mathrm{SVD}}$")
    axes[1, 0].set_ylabel(r"Relative mapping $R_G$")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5, h_pad=1.4)
    save(fig, "convergence")


def rank_sensitivity(rows, metadata):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    offsets = {"MU": -0.16, "PG": 0.0, "HALS": 0.16}
    for index, rank in enumerate((8, 16, 32)):
        case = f"digits_r{rank}"
        bound = metadata["datasets"][case]["svd_lower_bound"]
        axes[0].plot([index - 0.27, index + 0.27], [bound, bound],
                     color="#555555", ls=":", lw=1.5)
        for method in METHODS:
            selected = [
                row for row in rows if row["case"] == case and row["method"] == method
            ]
            x = index + offsets[method] + np.linspace(-0.035, 0.035, len(selected))
            for axis, key in zip(axes, ("objective", "seconds")):
                values = [row[key] for row in selected]
                axis.scatter(x, values, c=COLORS[method], s=15, alpha=0.75,
                             edgecolors="none", zorder=3)
                midpoint = index + offsets[method]
                axis.plot([midpoint - 0.06, midpoint + 0.06],
                          [np.median(values)] * 2, color=COLORS[method], lw=2.4)
    for axis in axes:
        axis.set_xticks(range(3), ["8", "16", "32"])
        axis.set_xlabel("Factorization rank")
        axis.set_yscale("log")
        axis.set_xlim(-0.42, 2.42)
    axes[0].set_ylabel("Final normalized objective F")
    axes[1].set_ylabel("Wall time to termination (s)")
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2)
    save(fig, "rank_sensitivity")


def zero_lock(curves, metadata):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    for method in METHODS:
        history = curves[f"zero_lock/{method}/0"]
        iterations = [v["iteration"] for v in history]
        for axis, key in zip(axes, ("objective", "relative_pg")):
            axis.semilogy(iterations, [v[key] for v in history],
                          color=COLORS[method])
    floor = metadata["zero_lock"]["mu_objective_lower_bound_from_locked_row"]
    axes[0].axhline(floor, color="#555555", ls=":", lw=1)
    axes[0].annotate(
        "Locked-row error floor", xy=(1100, floor), xytext=(700, floor * 7),
        fontsize=8, arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
    )
    axes[1].axhline(1e-4, color="#555555", ls=":", lw=1)
    axes[0].set_ylabel("Normalized objective F")
    axes[1].set_ylabel(r"Relative mapping $R_G$")
    for axis in axes:
        axis.set_xlabel("Sweeps")
        axis.set_xlim(0, 2000)
        axis.xaxis.set_major_locator(MaxNLocator(4))
    legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.89), w_pad=2)
    save(fig, "zero_lock")


def image_reconstruction():
    data = np.load(ROOT / "data" / "datasets.npz")
    V, labels = data["digits"], data["digits_labels"]
    selected = [int(np.flatnonzero(labels == digit)[0]) for digit in range(10)]
    factors = {
        method: np.load(ROOT / "results" / f"digits_{method}_seed0.npz")
        for method in METHODS
    }
    images = {"Input": V[:, selected]}
    for method, factor in factors.items():
        images[method] = (factor["W"] @ factor["H"])[:, selected]
    common_maximum = max(float(v.max()) for v in images.values())
    fig = plt.figure(figsize=(7.3, 2.65))
    grid = fig.add_gridspec(
        4, 15, left=0.07, right=0.99, top=0.87, bottom=0.04, wspace=0.12, hspace=0.1
    )
    for row, (label, values) in enumerate(images.items()):
        for column in range(10):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(values[:, column].reshape(8, 8), cmap="gray_r",
                        vmin=0, vmax=common_maximum, interpolation="nearest")
            axis.set_axis_off()
            if row == 0:
                axis.set_title(str(column), fontsize=8, pad=3)
            if column == 0:
                axis.text(-0.22, 0.5, label, transform=axis.transAxes,
                          ha="right", va="center", fontsize=8)
    basis = factors["HALS"]["W"]
    for component in range(16):
        row, column = divmod(component, 4)
        axis = fig.add_subplot(grid[row, column + 11])
        image = basis[:, component].reshape(8, 8)
        maximum = float(image.max()) or 1.0
        axis.imshow(image / maximum, cmap="gray_r", vmin=0, vmax=1,
                    interpolation="nearest")
        axis.set_axis_off()
    fig.text(0.405, 0.98, "Digits and rank-16 reconstructions, seed 0",
             ha="center", va="top", fontsize=10)
    fig.text(0.87, 0.98, "HALS basis", ha="center", va="top", fontsize=10)
    save(fig, "reconstruction")


def scientific(value, digits=2):
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def write_tables(summaries, metadata, verification, diagnostics):
    indexed = {(v["case"], v["method"]): v for v in summaries}
    rows = []
    for case, name in (
        ("synthetic_clean", "Exact"),
        ("synthetic_noisy", "Noisy"),
        ("digits_r16", "Digits 16"),
    ):
        for method in METHODS:
            value = indexed[case, method]
            milliseconds = float(f"{1000 * value['seconds_median']:.2g}")
            row = (
                f"{name} & {method} & ${scientific(value['objective_median'])}$ & "
                f"${100 * value['relative_error_median']:.2f}$ & "
                f"${value['iterations_median']:.0f}$ & "
                f"${milliseconds:g}$ & "
                f"${value['converged']}/5$"
            )
            rows.append(row + r" \\")
        rows.append(r"\midrule")
    (ROOT / "tables" / "main_results.tex").write_text(
        "\n".join(rows[:-1]) + "\n", encoding="utf-8"
    )
    rows = []
    for rank in (8, 32):
        for method in METHODS:
            value = indexed[f"digits_r{rank}", method]
            rows.append(
                f"${rank}$ & {method} & ${value['objective_median']:.5f}$ & "
                f"$[{value['objective_min']:.5f},\\ {value['objective_max']:.5f}]$ & "
                f"${value['iterations_median']:.0f}$ & "
                f"${value['seconds_median']:.2g}$ & "
                f"${value['converged']}/5$" + r" \\"
            )
        rows.append(r"\midrule")
    (ROOT / "tables" / "rank_results.tex").write_text(
        "\n".join(rows[:-1]) + "\n", encoding="utf-8"
    )
    names = {
        "synthetic_clean": "Exact",
        "synthetic_noisy": "Noisy",
        "digits_r16": "Digits 16",
        "digits_r8": "Digits 8",
        "digits_r32": "Digits 32",
    }
    rows = []
    for case in diagnostics["matched_objectives"]:
        cells = [
            names[case["case"]],
            f"${scientific(case['objective_target'])}$",
        ]
        for method in METHODS:
            outcome = case["methods"][method]
            sweeps = outcome["sweeps_median_over_reached"]
            cells.append(
                f"${outcome['reached']}/5;\\ {sweeps:g}$"
                if sweeps is not None else "$0/5;\\ {-}$"
            )
        rows.append(" & ".join(cells) + r" \\")
    (ROOT / "tables" / "matched_objectives.tex").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    clean_hals = indexed["synthetic_clean", "HALS"]
    clean_pg = indexed["synthetic_clean", "PG"]
    noisy_hals = indexed["synthetic_noisy", "HALS"]
    noisy_bound = metadata["datasets"]["synthetic_noisy"]["svd_lower_bound"]
    library = metadata["library_reference"]
    macros = {
        "MainRunCount": str(metadata["main_run_count"]),
        "CleanHALSK": f"{clean_hals['iterations_median']:.0f}",
        "CleanPGK": f"{clean_pg['iterations_median']:.0f}",
        "NoisyBound": f"{noisy_bound:.7f}",
        "NoisyBest": f"{noisy_hals['objective_median']:.7f}",
        "NoisyExcessPercent": f"{100 * (noisy_hals['objective_median'] / noisy_bound - 1):.2f}",
        "DigitsBound": f"{metadata['datasets']['digits_r16']['svd_lower_bound']:.7f}",
        "LibraryBest": f"{min(v['objective'] for v in library):.8f}",
        "LibraryWorst": f"{max(v['objective'] for v in library):.8f}",
        "LockFloor": f"{metadata['zero_lock']['mu_objective_lower_bound_from_locked_row']:.7f}",
        "LockFinal": f"{metadata['zero_lock']['runs'][0]['objective']:.7f}",
        "GradientCheck": scientific(verification["finite_difference_max_absolute_error"], 1),
        "NNLSCheck": scientific(verification["fixed_W_HALS_vs_scipy_nnls_factor_error"], 1),
        "RankHALSTime": f"{indexed['digits_r32', 'HALS']['seconds_median']:.2f}",
        "RankMUTime": f"{indexed['digits_r32', 'MU']['seconds_median']:.2f}",
        "LockExcess": scientific(
            metadata["zero_lock"]["runs"][0]["objective"]
            - metadata["zero_lock"]["mu_objective_lower_bound_from_locked_row"], 2
        ),
    }
    counts = diagnostics["gauge_counts_at_original_terminal_points"]
    for method in METHODS:
        macros[f"WColumn{method}Met"] = str(counts["W_columns"][method])
        macros[f"HRow{method}Met"] = str(counts["H_rows"][method])
        stable = diagnostics["digits_r16_initialization_sensitivity"][method]
        for suffix, key in (
            ("Minimum", "objective_min"), ("Maximum", "objective_max")
        ):
            macros[f"Digits{method}{suffix}"] = f"{stable[key]:.7f}"
        macros[f"Digits{method}Spread"] = (
            f"{stable['objective_range_percent_of_median']:.2f}"
        )
        macros[f"Digits{method}MinSweeps"] = str(stable["sweeps_min"])
        macros[f"Digits{method}MaxSweeps"] = str(stable["sweeps_max"])
    examples = {
        row["method"]: row for row in diagnostics["terminal_diagnostics"]
        if row["case"] == "digits_r16" and row["seed"] == 0
    }
    macros.update({
        "MUUnitRelative": f"{examples['MU']['relative_mapping_W_columns']:.4f}",
        "MULinRelative": f"{examples['MU']['relative_lin_projected_gradient']:.3f}",
        "MUSmallEntries": str(examples["MU"]["W_small_positive_entries"]),
        "GaugeWExample": scientific(examples["HALS"]["relative_mapping_W_columns"]),
        "GaugeHExample": scientific(examples["HALS"]["relative_mapping_H_rows"]),
        "RankMUGapPercent": (
            f"{diagnostics['digits_r32_best_relative_to_worst_HALS_percent']['MU']:.1f}"
        ),
        "RankPGGapPercent": (
            f"{diagnostics['digits_r32_best_relative_to_worst_HALS_percent']['PG']:.1f}"
        ),
    })
    (ROOT / "tables" / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in macros.items())
        + "\n", encoding="utf-8",
    )


def main():
    style()
    for folder in ("figures", "tables"):
        (ROOT / folder).mkdir(exist_ok=True)
    results = ROOT / "results"
    values = {
        name: json.loads((results / f"{name}.json").read_text())
        for name in ("runs", "curves", "summary", "metadata", "verification", "diagnostics")
    }
    assert values["verification"]["passed"]
    for source in ("runs", "curves"):
        digest = hashlib.sha256((results / f"{source}.json").read_bytes()).hexdigest()
        assert digest == values["diagnostics"][f"source_{source}_sha256"]
    assert len(values["runs"]) == 75
    assert all(row["checkpoint_monotone"] for row in values["runs"])
    convergence(values["curves"], values["metadata"])
    rank_sensitivity(values["runs"], values["metadata"])
    zero_lock(values["curves"], values["metadata"])
    image_reconstruction()
    write_tables(
        values["summary"], values["metadata"], values["verification"],
        values["diagnostics"],
    )
    print("Generated four figures, three result tables and numeric macros.")


if __name__ == "__main__":
    main()
