"""Particle Swarm Optimization (PSO): tři varianty ze zadání (minimalizace).

  PSO-1: lineárně klesající setrvačnost w = 0,8 -> 0,3, c1 = c2 = 2, globální topologie (hvězda)
  PSO-2: konstantní w = 0,7, c1 = c2 = 1,49618, globální topologie (hvězda)
  PSO-3: konstantní w = 0,7, c1 = c2 = 1,49618, kruhová topologie (sousedství N = 3)

Doplňkově (mimo zadání) kruh s N = 1, protože "velikost sousedství 3" jde vyložit i jako
jedinec + 2 sousedé.

Společně: v_max = 0,2 * Range, v(0) = 0, částice smí vylétnout z prohledávané oblasti,
ale mimo ni se neohodnocují (a nezapočítávají se do MAXFES). Ukončení: MAXFES.

Experiment: Sphere, Rosenbrock, Rastrigin, Schwefel; D = 10; NP = 50; MAXFES = 100 000;
30 nezávislých běhů. Všechny varianty startují ze stejné počáteční populace.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

# Společné parametry
DIM = 10
POP_SIZE = 50
MAXFES = 10_000 * DIM
RUNS = 30
SEED = 42

VMAX_RATIO = 0.2  # v_max = 0,2 * (high - low)

# Práh pro "počet ohodnocení do dosažení cíle"
TARGET = 1e-8

STATISTICS_CSV_PATH = "statistics.csv"
SIGNIFICANCE_CSV_PATH = "significance.csv"
CONVERGENCE_PLOT_PATH = "convergence.png"
BOXPLOT_PATH = "final_boxplot.png"

# ---------------------------------------------------------------------------
# Účelové funkce (vstup: pole (n, D), výstup: pole (n,))
# ---------------------------------------------------------------------------


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere: sum(x_i^2); minimum 0 v bodě 0."""
    return np.sum(x**2, axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock: sum(100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2); minimum 0 v bodě 1."""
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    """Rastrigin: 10 D + sum(x_i^2 - 10 cos(2 pi x_i)); minimum 0 v bodě 0."""
    return 10.0 * x.shape[1] + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)


SCHWEFEL_OFFSET = 418.9828872724337  # přesná konstanta -> minimum je 0 (v Ukolu 2 zaokrouhleno na 418,9829)


def schwefel(x: np.ndarray) -> np.ndarray:
    """Schwefel: 418,98... D - sum(x_i sin(sqrt|x_i|)); minimum 0 v bodě 420,9687."""
    return SCHWEFEL_OFFSET * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


# název -> (funkce, dolní mez, horní mez)
PROBLEMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float, float]] = {
    "Sphere": (sphere, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -5.0, 10.0),
    "Rastrigin": (rastrigin, -5.12, 5.12),
    "Schwefel": (schwefel, -500.0, 500.0),
}

# ---------------------------------------------------------------------------
# Varianty PSO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    w_start: float
    w_end: float  # w_start == w_end -> konstantní setrvačnost
    c1: float
    c2: float
    neighborhood: int | None  # None = globální topologie (hvězda); N = kruh, N sousedů vlevo i vpravo


C_CLERC = 1.49618

# Varianty ze zadání (pro statistické testy) a doplňková varianta
VARIANTS: dict[str, Variant] = {
    "PSO-1": Variant(0.8, 0.3, 2.0, 2.0, None),
    "PSO-2": Variant(0.7, 0.7, C_CLERC, C_CLERC, None),
    "PSO-3": Variant(0.7, 0.7, C_CLERC, C_CLERC, 3),
}
SUPPLEMENTARY: dict[str, Variant] = {
    "PSO-3 (N=1)": Variant(0.7, 0.7, C_CLERC, C_CLERC, 1),
}
ALL_VARIANTS = {**VARIANTS, **SUPPLEMENTARY}

DESCRIPTIONS = {
    "PSO-1": "w 0,8→0,3, c=2, hvězda",
    "PSO-2": "w=0,7, c=1,496, hvězda",
    "PSO-3": "w=0,7, c=1,496, kruh N=3",
    "PSO-3 (N=1)": "doplňkově: kruh N=1",
}

# ---------------------------------------------------------------------------
# Jádro PSO
# ---------------------------------------------------------------------------


def inertia_weight(variant: Variant, fes: int, maxfes: int) -> float:
    """w = w_start - (w_start - w_end) * i / n; i/n se měří spotřebovanými ohodnoceními.

    Bez částic mimo oblast je fes = i * NP a maxfes = n * NP, tedy shodné se vzorcem ze slidu;
    měření přes FES zaručuje, že w dojde přesně na w_end i když se část částic neohodnocuje.
    """
    return variant.w_start - (variant.w_start - variant.w_end) * fes / maxfes


def ring_neighbors(pop_size: int, n: int) -> np.ndarray:
    """Sousedství kruhové topologie podle ID: pro jedince i indexy i-n .. i+n (modulo NP).

    Tvar (pop_size, 2n + 1), jedinec sám je součástí svého sousedství.
    """
    return (np.arange(pop_size)[:, None] + np.arange(-n, n + 1)[None, :]) % pop_size


def guide_positions(pbest_x: np.ndarray, pbest_f: np.ndarray, ring: np.ndarray | None) -> np.ndarray:
    """Pozice 'gBest' pro každou částici: nejlepší pBest v celé populaci, resp. v jejím sousedství."""
    if ring is None:
        return np.broadcast_to(pbest_x[np.argmin(pbest_f)], pbest_x.shape)
    winner = ring[np.arange(len(pbest_f)), np.argmin(pbest_f[ring], axis=1)]
    return pbest_x[winner]


def swarm_diversity(pos: np.ndarray, low: float, high: float) -> float:
    """Diverzita roje: průměrná vzdálenost částic od těžiště, normovaná úhlopříčkou oblasti."""
    return float(np.mean(np.linalg.norm(pos - pos.mean(axis=0), axis=1)) / ((high - low) * math.sqrt(pos.shape[1])))


def run_pso(
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    variant: Variant,
    init_pos: np.ndarray,
    rng: np.random.Generator,
    maxfes: int = MAXFES,
) -> dict:
    """Spustí jednu variantu PSO z dané počáteční populace; končí po `maxfes` ohodnoceních.

    Vrací slovník: fes_history a best_history (nejlepší dosud nalezená fitness po každé
    iteraci včetně počáteční populace), best_x, best_f, fes (spotřebovaná ohodnocení),
    outside_ratio (podíl pozic částic mimo oblast přes všechny iterace) a max_speed_ratio
    (největší |v| / v_max, musí být <= 1), diversity_history a outside_history (diverzita roje
    a podíl částic mimo oblast po každé iteraci, délka jako fes_history).
    """
    pop_size, dim = init_pos.shape
    vmax = VMAX_RATIO * (high - low)
    ring = None if variant.neighborhood is None else ring_neighbors(pop_size, variant.neighborhood)

    pos = init_pos.copy()
    vel = np.zeros_like(pos)  # v(0) = 0
    pbest_x = pos.copy()
    pbest_f = func(pos)
    fes = pop_size

    fes_history = [fes]
    best_history = [pbest_f.min()]
    diversity_history = [swarm_diversity(pos, low, high)]
    outside_history = [0.0]
    outside = 0
    particle_steps = 0
    max_speed_ratio = 0.0

    # horní mez iterací jen jako pojistka proti smyčce, kdyby se částice nevracely do oblasti
    for _ in range(50 * maxfes // pop_size):
        if fes >= maxfes:
            break
        w = inertia_weight(variant, fes, maxfes)
        guide = guide_positions(pbest_x, pbest_f, ring)
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        vel = w * vel + variant.c1 * r1 * (pbest_x - pos) + variant.c2 * r2 * (guide - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        max_speed_ratio = max(max_speed_ratio, float(np.abs(vel).max()) / vmax)

        # částice mimo oblast zůstávají v roji, ale neohodnocují se (pBest se neaktualizuje)
        inside = np.flatnonzero(np.all((pos >= low) & (pos <= high), axis=1))
        outside += pop_size - inside.size
        outside_history.append((pop_size - inside.size) / pop_size)
        particle_steps += pop_size
        inside = inside[: maxfes - fes]  # poslední iterace nesmí překročit rozpočet
        if inside.size:
            f_new = func(pos[inside])
            fes += inside.size
            better = f_new < pbest_f[inside]
            upd = inside[better]
            pbest_x[upd] = pos[upd]
            pbest_f[upd] = f_new[better]

        fes_history.append(fes)
        best_history.append(pbest_f.min())
        diversity_history.append(swarm_diversity(pos, low, high))

    best = int(np.argmin(pbest_f))
    return {
        "fes_history": np.array(fes_history),
        "best_history": np.array(best_history),
        "best_x": pbest_x[best].copy(),
        "best_f": float(pbest_f[best]),
        "fes": fes,
        "outside_ratio": outside / particle_steps if particle_steps else 0.0,
        "max_speed_ratio": max_speed_ratio,
        "diversity_history": np.array(diversity_history),
        "outside_history": np.array(outside_history),
    }


def best_at(result: dict, fes_grid: np.ndarray) -> np.ndarray:
    """Nejlepší fitness po `fes_grid` ohodnoceních (poslední iterace s fes <= zadanému bodu)."""
    idx = np.searchsorted(result["fes_history"], fes_grid, side="right") - 1
    return result["best_history"][idx]


def fes_to_target(result: dict, target: float = TARGET) -> float:
    """Počet ohodnocení první iterace s nejlepší fitness <= target; NaN, pokud cíl nebyl dosažen."""
    hit = np.nonzero(result["best_history"] <= target)[0]
    return float(result["fes_history"][hit[0]]) if hit.size else float("nan")


# ---------------------------------------------------------------------------
# Kontrola funkčnosti
# ---------------------------------------------------------------------------


def self_check() -> None:
    """Ověří pravidla ze zadání na malém rozpočtu: žádné ohodnocení mimo oblast, přesně MAXFES
    ohodnocení, |v| <= v_max, w od 0,8 do 0,3 a sousedství kruhu podle slidu."""
    # sousedství podle ID: jedinec 40, N = 2 -> 38, 39, 41, 42 (+ on sám); přetečení přes okraj
    assert sorted(ring_neighbors(100, 2)[40]) == [38, 39, 40, 41, 42]
    assert sorted(ring_neighbors(10, 2)[0]) == [0, 1, 2, 8, 9]
    assert sorted(ring_neighbors(10, 3)[9]) == [0, 1, 2, 6, 7, 8, 9]

    lin = VARIANTS["PSO-1"]
    assert math.isclose(inertia_weight(lin, 0, 1000), 0.8)
    assert math.isclose(inertia_weight(lin, 500, 1000), 0.55)
    assert math.isclose(inertia_weight(lin, 1000, 1000), 0.3)
    assert inertia_weight(VARIANTS["PSO-2"], 700, 1000) == 0.7

    low, high, budget = -5.12, 5.12, 5000
    left_area = 0.0
    for i, (name, variant) in enumerate(ALL_VARIANTS.items()):
        evaluated: list[np.ndarray] = []

        def spy(x: np.ndarray) -> np.ndarray:
            evaluated.append(x.copy())
            return sphere(x)

        rng = np.random.default_rng(1000 + i)
        res = run_pso(spy, low, high, variant, rng.uniform(low, high, (POP_SIZE, DIM)), rng, maxfes=budget)
        points = np.vstack(evaluated)
        assert len(points) == res["fes"] == budget, f"{name}: špatný počet ohodnocení"
        assert points.min() >= low and points.max() <= high, f"{name}: ohodnocení mimo oblast"
        assert res["max_speed_ratio"] <= 1.0 + 1e-12, f"{name}: překročeno v_max"
        assert np.all(np.diff(res["best_history"]) <= 0), f"{name}: nejlepší fitness se zhoršila"
        left_area = max(left_area, res["outside_ratio"])
    assert left_area > 0, "částice nikdy nevylétly z oblasti, pravidlo se netestuje"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def make_run_seeds(seed: int) -> dict[str, list[np.random.SeedSequence]]:
    """Pro každou funkci RUNS seedů; stejný seed -> stejné počáteční populace."""
    problem_seeds = np.random.SeedSequence(seed).spawn(len(PROBLEMS))
    return {name: ps.spawn(RUNS) for name, ps in zip(PROBLEMS, problem_seeds)}


def run_experiment() -> dict:
    """Pro každou funkci a běh: stejná počáteční populace pro všechny varianty."""
    run_seeds = make_run_seeds(SEED)
    results: dict = {}
    for name, (func, low, high) in PROBLEMS.items():
        results[name] = {v: [] for v in ALL_VARIANTS}
        for run_seed in run_seeds[name]:
            init_pos = np.random.default_rng(run_seed).uniform(low, high, (POP_SIZE, DIM))
            variant_seeds = run_seed.spawn(len(ALL_VARIANTS))
            for (variant_name, variant), variant_seed in zip(ALL_VARIANTS.items(), variant_seeds):
                # každá varianta má vlastní generátor, ale stejnou počáteční populaci
                rng = np.random.default_rng(variant_seed)
                results[name][variant_name].append(run_pso(func, low, high, variant, init_pos, rng))
    return results


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Oboustranný Mann-Whitneyův U test (normální aproximace s opravou na shody, bez scipy)."""
    n1, n2 = len(a), len(b)
    n = n1 + n2
    _, inverse, counts = np.unique(np.concatenate([a, b]), return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]  # průměrné pořadí při shodách
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    tie_term = np.sum(counts**3 - counts) / (n * (n - 1))
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term))
    if sigma == 0:
        return 1.0
    z = (abs(u1 - n1 * n2 / 2.0) - 0.5) / sigma  # s opravou na spojitost
    return math.erfc(max(z, 0.0) / math.sqrt(2.0))


def write_statistics(results: dict) -> list[dict]:
    rows = []
    for problem, by_variant in results.items():
        for variant, runs in by_variant.items():
            finals = np.array([r["best_f"] for r in runs])
            to_target = np.array([fes_to_target(r) for r in runs])
            reached = int(np.sum(~np.isnan(to_target)))
            rows.append(
                {
                    "function": problem,
                    "variant": variant,
                    "best": finals.min(),
                    "median": float(np.median(finals)),
                    "mean": finals.mean(),
                    "std": finals.std(),
                    "worst": finals.max(),
                    "reached_target": f"{reached}/{RUNS}",
                    "mean_fes_to_target": float(np.nanmean(to_target)) if reached else float("nan"),
                    "outside_percent": 100.0 * float(np.mean([r["outside_ratio"] for r in runs])),
                    "maxfes": MAXFES,
                }
            )
    _write_csv(STATISTICS_CSV_PATH, rows)
    return rows


# páry pro Mann-Whitneyův test: tři varianty ze zadání navzájem + kruh N = 3 vs N = 1
SIGNIFICANCE_PAIRS = [*combinations(VARIANTS, 2), ("PSO-3", "PSO-3 (N=1)")]


def write_significance(results: dict) -> list[dict]:
    """Párové porovnání variant (finální nejlepší fitness, 30 běhů)."""
    rows = []
    for problem, by_variant in results.items():
        finals = {v: np.array([r["best_f"] for r in by_variant[v]]) for v in ALL_VARIANTS}
        for a, b in SIGNIFICANCE_PAIRS:
            med_a, med_b = np.median(finals[a]), np.median(finals[b])
            rows.append(
                {
                    "function": problem,
                    "variant_a": a,
                    "variant_b": b,
                    "median_a": float(med_a),
                    "median_b": float(med_b),
                    "p_value": mann_whitney_p(finals[a], finals[b]),
                    "lower_median": a if med_a < med_b else b if med_b < med_a else "=",
                }
            )
    # Holmova korekce přes všech 16 testů (zachovává pořadí p-hodnot, zajišťuje monotonii)
    order = np.argsort([r["p_value"] for r in rows])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[idx]["p_value"]))
        rows[idx]["p_holm"] = running
    _write_csv(SIGNIFICANCE_CSV_PATH, rows)
    return rows


def _write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


# ---------------------------------------------------------------------------
# Grafy
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
COLORS = {"PSO-1": "#2a78d6", "PSO-2": "#eb6834", "PSO-3": "#1baf7a", "PSO-3 (N=1)": INK_SECONDARY}
LINESTYLES = {"PSO-1": "-", "PSO-2": "-", "PSO-3": "-", "PSO-3 (N=1)": (0, (4, 3))}
LOG_FLOOR = 1e-300  # kvůli logaritmické ose (Sphere může podtéct na přesnou 0)
LOG_BOXPLOT = {"Sphere", "Rosenbrock"}  # Rastrigin a Schwefel: lineární osa (Rastrigin se trefuje přesně do 0)


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, which="major", color=GRID, linewidth=1.0, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _sparse_log_ticks(ax: plt.Axes, max_ticks: int = 7) -> None:
    """Při velmi širokém rozsahu osy (desítky řádů) popisky po každých několika řádech."""
    low, high = ax.get_ylim()
    decades = math.log10(high / low)
    if decades > max_ticks:
        ax.yaxis.set_major_locator(LogLocator(base=10.0 ** math.ceil(decades / max_ticks)))
        ax.yaxis.set_minor_locator(NullLocator())


def _legend(fig: plt.Figure, handles: list) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        labelcolor=INK,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )


def plot_convergence(results: dict) -> None:
    """Medián nejlepší fitness (a mezikvartilové pásmo) přes FES, logaritmická osa."""
    grid = np.linspace(POP_SIZE, MAXFES, 500)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=SURFACE)
    for ax, (problem, by_variant) in zip(axes.ravel(), results.items()):
        for variant, runs in by_variant.items():
            curves = np.maximum(np.array([best_at(r, grid) for r in runs]), LOG_FLOOR)
            q1, med, q3 = np.percentile(curves, [25, 50, 75], axis=0)
            ax.fill_between(grid, q1, q3, color=COLORS[variant], alpha=0.10, linewidth=0)
            ax.plot(grid, med, color=COLORS[variant], linestyle=LINESTYLES[variant], linewidth=2)
        ax.set_yscale("log")
        ax.set_xlim(0, MAXFES)
        _sparse_log_ticks(ax)
        _style_axes(ax, f"{problem} (D={DIM})", "Počet ohodnocení (FES)", "Nejlepší fitness")
    _legend(
        fig,
        [
            plt.Line2D([], [], color=COLORS[v], linestyle=LINESTYLES[v], linewidth=2, label=f"{v}: {DESCRIPTIONS[v]}")
            for v in ALL_VARIANTS
        ],
    )
    fig.suptitle(f"Konvergence PSO: medián a mezikvartilové pásmo z {RUNS} běhů", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(CONVERGENCE_PLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_boxplot(results: dict) -> None:
    """Rozdělení finální nejlepší fitness z RUNS běhů (jednotlivé běhy jako body)."""
    jitter = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor=SURFACE)
    for ax, (problem, by_variant) in zip(axes.ravel(), results.items()):
        for pos, (variant, runs) in enumerate(by_variant.items(), start=1):
            finals = np.maximum([r["best_f"] for r in runs], LOG_FLOOR)
            ax.boxplot(
                finals,
                positions=[pos],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": INK, "linewidth": 2},
                whiskerprops={"color": COLORS[variant], "linewidth": 1.5},
                capprops={"color": COLORS[variant], "linewidth": 1.5},
                boxprops={"facecolor": COLORS[variant], "alpha": 0.25, "edgecolor": COLORS[variant]},
            )
            ax.scatter(
                pos + jitter.uniform(-0.15, 0.15, len(finals)),
                finals,
                s=40,
                color=COLORS[variant],
                edgecolor=SURFACE,
                linewidth=1.2,
                zorder=3,
            )
        if problem in LOG_BOXPLOT:
            ax.set_yscale("log")
            _sparse_log_ticks(ax)
        ax.set_xticks(range(1, len(by_variant) + 1), list(by_variant))
        _style_axes(ax, f"{problem} (D={DIM})", "", f"Finální nejlepší fitness po {MAXFES:,} FES".replace(",", " "))
    _legend(
        fig,
        [
            plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=COLORS[v], label=f"{v}: {DESCRIPTIONS[v]}")
            for v in ALL_VARIANTS
        ],
    )
    fig.suptitle(f"Finální řešení PSO: {RUNS} nezávislých běhů (box = kvartily, čára = medián)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(BOXPLOT_PATH, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    self_check()
    print("Kontrola funkčnosti: OK")

    results = run_experiment()
    for by_variant in results.values():
        for runs in by_variant.values():
            assert all(r["fes"] == MAXFES for r in runs), "některý běh nespotřeboval přesně MAXFES"
    rows = write_statistics(results)
    sig_rows = write_significance(results)
    plot_convergence(results)
    plot_boxplot(results)

    print(
        f"{'Funkce':<12}{'Varianta':<13}{'best':>11}{'median':>11}{'mean':>11}{'std':>11}"
        f"{'cíl':>8}{'FES do cíle':>13}{'mimo %':>8}"
    )
    for r in rows:
        print(
            f"{r['function']:<12}{r['variant']:<13}{r['best']:>11.3e}{r['median']:>11.3e}"
            f"{r['mean']:>11.3e}{r['std']:>11.3e}{r['reached_target']:>8}"
            f"{r['mean_fes_to_target']:>13.0f}{r['outside_percent']:>8.2f}"
        )
    print("\nMann-Whitney (finální fitness, oboustranný p, Holmova korekce):")
    for s in sig_rows:
        print(f"{s['function']:<12}{s['variant_a']} vs {s['variant_b']}: p = {s['p_value']:.2g}, Holm {s['p_holm']:.2g}  (nižší medián: {s['lower_median']})")


if __name__ == "__main__":
    main()
