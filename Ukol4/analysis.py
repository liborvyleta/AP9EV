"""Doplňková analýza k pso.py: diverzita roje, částice mimo oblast v čase a citlivost na NP.

Ověřuje tvrzení, která jsou v README formulována jako vysvětlení:
  - kruhová topologie déle udržuje diverzitu roje než hvězda,
  - PSO-1 (c1 + c2 = 4) opouští oblast nejčastěji, zejména na začátku běhu,
a ukazuje, zda pořadí variant platí i pro NP = 20 a NP = 100.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

import pso

NP_VALUES = (20, 50, 100)
NP_SWEEP_CSV_PATH = "np_sweep.csv"
DIVERSITY_PLOT_PATH = "diversity.png"
OUTSIDE_PLOT_PATH = "outside_over_time.png"
NP_SWEEP_PLOT_PATH = "np_sweep.png"

OUTSIDE_BINS = 50  # počet intervalů FES pro průměrování podílu částic mimo oblast


def run_all(pop_size: int) -> dict:
    """Stejný experiment jako pso.run_experiment, ale s volitelným NP."""
    run_seeds = pso.make_run_seeds(pso.SEED)
    results: dict = {}
    for name, (func, low, high) in pso.PROBLEMS.items():
        results[name] = {v: [] for v in pso.ALL_VARIANTS}
        for run_seed in run_seeds[name]:
            init_pos = np.random.default_rng(run_seed).uniform(low, high, (pop_size, pso.DIM))
            variant_seeds = run_seed.spawn(len(pso.ALL_VARIANTS))
            for (variant_name, variant), variant_seed in zip(pso.ALL_VARIANTS.items(), variant_seeds):
                rng = np.random.default_rng(variant_seed)
                results[name][variant_name].append(pso.run_pso(func, low, high, variant, init_pos, rng))
    return results


def value_at(result: dict, key: str, fes_grid: np.ndarray) -> np.ndarray:
    """Hodnota časové řady `key` po `fes_grid` ohodnoceních (poslední iterace s fes <= bodu)."""
    idx = np.searchsorted(result["fes_history"], fes_grid, side="right") - 1
    return result[key][idx]


def outside_binned(result: dict) -> np.ndarray:
    """Průměrný podíl částic mimo oblast v OUTSIDE_BINS stejně dlouhých intervalech FES."""
    edges = np.linspace(0, pso.MAXFES, OUTSIDE_BINS + 1)
    which = np.clip(np.searchsorted(edges, result["fes_history"], side="right") - 1, 0, OUTSIDE_BINS - 1)
    return np.array([result["outside_history"][which == b].mean() if np.any(which == b) else np.nan for b in range(OUTSIDE_BINS)])


def _four_panels(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=pso.SURFACE)
    fig.suptitle(title, color=pso.INK, fontsize=12)
    return fig, axes.ravel()


def _save(fig: plt.Figure, handles: list, path: str) -> None:
    pso._legend(fig, handles)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=pso.SURFACE)
    plt.close(fig)


def _line_handles(variants: list[str]) -> list:
    return [
        plt.Line2D([], [], color=pso.COLORS[v], linestyle=pso.LINESTYLES[v], linewidth=2, label=f"{v}: {pso.DESCRIPTIONS[v]}")
        for v in variants
    ]


def plot_diversity(results: dict) -> None:
    """Medián diverzity roje přes FES (logaritmická osa)."""
    grid = np.linspace(pso.POP_SIZE, pso.MAXFES, 500)
    fig, axes = _four_panels(f"Diverzita roje: medián z {pso.RUNS} běhů (průměrná vzdálenost od těžiště / úhlopříčka oblasti)")
    for ax, (problem, by_variant) in zip(axes, results.items()):
        for variant, runs in by_variant.items():
            curves = np.maximum([value_at(r, "diversity_history", grid) for r in runs], pso.LOG_FLOOR)
            ax.plot(grid, np.median(curves, axis=0), color=pso.COLORS[variant], linestyle=pso.LINESTYLES[variant], linewidth=2)
        ax.set_yscale("log")
        ax.set_xlim(0, pso.MAXFES)
        pso._sparse_log_ticks(ax)
        pso._style_axes(ax, f"{problem} (D={pso.DIM})", "Počet ohodnocení (FES)", "Diverzita roje")
    _save(fig, _line_handles(list(pso.ALL_VARIANTS)), DIVERSITY_PLOT_PATH)


def plot_outside(results: dict) -> None:
    """Průměrný podíl částic mimo oblast v jednotlivých úsecích běhu (jen varianty ze zadání)."""
    centers = (np.arange(OUTSIDE_BINS) + 0.5) * pso.MAXFES / OUTSIDE_BINS
    fig, axes = _four_panels(f"Podíl částic mimo prohledávanou oblast: průměr z {pso.RUNS} běhů")
    for ax, (problem, by_variant) in zip(axes, results.items()):
        for variant in pso.VARIANTS:
            ratio = 100.0 * np.nanmean([outside_binned(r) for r in by_variant[variant]], axis=0)
            ax.plot(centers, ratio, color=pso.COLORS[variant], linestyle=pso.LINESTYLES[variant], linewidth=2)
        ax.set_xlim(0, pso.MAXFES)
        ax.set_ylim(bottom=0)
        pso._style_axes(ax, f"{problem} (D={pso.DIM})", "Počet ohodnocení (FES)", "Částic mimo oblast [%]")
    _save(fig, _line_handles(list(pso.VARIANTS)), OUTSIDE_PLOT_PATH)


def np_sweep() -> list[dict]:
    """Medián, průměr a směrodatná odchylka finální fitness pro NP = 20, 50, 100."""
    rows = []
    for pop_size in NP_VALUES:
        results = run_all(pop_size)
        for problem, by_variant in results.items():
            for variant, runs in by_variant.items():
                finals = np.array([r["best_f"] for r in runs])
                rows.append(
                    {
                        "function": problem,
                        "variant": variant,
                        "np": pop_size,
                        "median": float(np.median(finals)),
                        "mean": float(finals.mean()),
                        "std": float(finals.std()),
                    }
                )
    with open(NP_SWEEP_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    return rows


def plot_np_sweep(rows: list[dict]) -> None:
    fig, axes = _four_panels(f"Vliv velikosti populace NP: medián finální fitness z {pso.RUNS} běhů (MAXFES = {pso.MAXFES:,})".replace(",", " "))
    for ax, problem in zip(axes, pso.PROBLEMS):
        for variant in pso.VARIANTS:
            sel = [r for r in rows if r["function"] == problem and r["variant"] == variant]
            ax.plot(
                [r["np"] for r in sel],
                np.maximum([r["median"] for r in sel], pso.LOG_FLOOR),
                color=pso.COLORS[variant],
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor=pso.SURFACE,
                markeredgewidth=1.5,
            )
        ax.set_yscale("log")
        ax.set_xticks(NP_VALUES)
        pso._sparse_log_ticks(ax)
        pso._style_axes(ax, f"{problem} (D={pso.DIM})", "NP", "Medián finální fitness")
    _save(fig, _line_handles(list(pso.VARIANTS)), NP_SWEEP_PLOT_PATH)


def main() -> None:
    results = run_all(pso.POP_SIZE)
    plot_diversity(results)
    plot_outside(results)

    rows = np_sweep()
    plot_np_sweep(rows)

    print("Medián finální fitness podle NP:")
    print(f"{'Funkce':<12}{'Varianta':<8}" + "".join(f"{'NP=' + str(n):>12}" for n in NP_VALUES))
    for problem in pso.PROBLEMS:
        for variant in pso.VARIANTS:
            vals = {r["np"]: r["median"] for r in rows if r["function"] == problem and r["variant"] == variant}
            print(f"{problem:<12}{variant:<8}" + "".join(f"{vals[n]:>12.3g}" for n in NP_VALUES))


if __name__ == "__main__":
    main()
