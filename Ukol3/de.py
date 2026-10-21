"""Diferenciální evoluce: DE/rand/1/bin, DE/best/1/bin a jDE (minimalizace).

Experiment: Sphere, Rosenbrock, Rastrigin; D = 10; NP = 50; 500 generací;
10 nezávislých běhů. Všechny tři varianty startují ze stejné počáteční populace.
"""

from __future__ import annotations

import csv
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# Společné parametry
DIM = 10
POP_SIZE = 50
GENERATIONS = 500
RUNS = 10
SEED = 42

# Pevné parametry Rand/1 a Best/1
CR_FIXED = 0.8
F_FIXED = 0.5

# jDE
TAU1 = 0.1
TAU2 = 0.1
F_INIT, F_MIN, F_MAX = 0.5, 0.1, 0.9  # F_new = F_MIN + (F_MAX - F_MIN) * U(0,1) -> [0.1, 0.9]
CR_INIT = 0.9

# Práh pro "počet generací do dosažení cíle"
TARGET = 1e-6

STATISTICS_CSV_PATH = "statistics.csv"
CONVERGENCE_PLOT_PATH = "convergence.png"
SINGLE_RUN_PLOT_PATH = "convergence_single_run.png"
JDE_PARAMS_PLOT_PATH = "jde_parameters.png"

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


# název -> (funkce, dolní mez, horní mez)
PROBLEMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float, float]] = {
    "Sphere": (sphere, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -5.0, 10.0),
    "Rastrigin": (rastrigin, -5.12, 5.12),
}

# ---------------------------------------------------------------------------
# Jádro DE
# ---------------------------------------------------------------------------


def distinct_indices(rng: np.random.Generator, pop_size: int, count: int) -> np.ndarray:
    """Pro každého jedince i vrátí `count` navzájem různých indexů různých od i.

    Výstup má tvar (pop_size, count). Používá se náhodná permutace indexů, u každého
    řádku se vynechá i, takže stačí vzít prvních `count` zbylých.
    """
    perm = np.argsort(rng.random((pop_size, pop_size)), axis=1)
    own = np.arange(pop_size)[:, None]
    # přesun vlastního indexu na konec řádku -> prvních `count` sloupců je bez i
    keys = (perm == own).astype(int)
    order = np.argsort(keys, axis=1, kind="stable")
    return np.take_along_axis(perm, order, axis=1)[:, :count]


def binomial_crossover(
    rng: np.random.Generator, target: np.ndarray, mutant: np.ndarray, cr: np.ndarray
) -> np.ndarray:
    """Binomické křížení: gen z mutanta s pravděpodobností CR, alespoň jeden vždy (j_rand)."""
    pop_size, dim = target.shape
    mask = rng.random((pop_size, dim)) < cr[:, None]
    mask[np.arange(pop_size), rng.integers(0, dim, pop_size)] = True
    return np.where(mask, mutant, target)


def run_de(
    variant: str,
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    init_pop: np.ndarray,
    rng: np.random.Generator,
    generations: int = GENERATIONS,
    f_fixed: float = F_FIXED,
    cr_fixed: float = CR_FIXED,
) -> dict:
    """Spustí jednu variantu DE ('rand1', 'best1' nebo 'jde') z dané počáteční populace.

    Vrací slovník: history (nejlepší fitness po každé generaci včetně počáteční, délka
    generations + 1), best_x, best_f a u jDE průměrné F a CR v čase. f_fixed a cr_fixed
    se používají jen u rand1 a best1 (výchozí hodnoty ze zadání).
    """
    pop = init_pop.copy()
    fit = func(pop)
    pop_size = pop.shape[0]

    # jDE: parametry F a CR si nese každý jedinec
    f_i = np.full(pop_size, F_INIT)
    cr_i = np.full(pop_size, CR_INIT)

    history = [fit.min()]
    f_mean, cr_mean = [f_i.mean()], [cr_i.mean()]

    for _ in range(generations):
        if variant == "jde":
            # samoadaptace: s pravděpodobností tau se parametry jedince přegenerují
            f_trial = np.where(rng.random(pop_size) < TAU1, F_MIN + (F_MAX - F_MIN) * rng.random(pop_size), f_i)
            cr_trial = np.where(rng.random(pop_size) < TAU2, rng.random(pop_size), cr_i)
        else:
            f_trial = np.full(pop_size, f_fixed)
            cr_trial = np.full(pop_size, cr_fixed)

        # mutace
        if variant == "best1":
            r1, r2 = distinct_indices(rng, pop_size, 2).T
            base = pop[np.argmin(fit)][None, :]
            mutant = base + f_trial[:, None] * (pop[r1] - pop[r2])
        else:  # rand1 i jDE
            r1, r2, r3 = distinct_indices(rng, pop_size, 3).T
            mutant = pop[r1] + f_trial[:, None] * (pop[r2] - pop[r3])

        # ošetření mezí: souřadnice mimo rozsah se vrátí náhodně dovnitř rozsahu
        out = (mutant < low) | (mutant > high)
        mutant = np.where(out, rng.uniform(low, high, mutant.shape), mutant)

        trial = binomial_crossover(rng, pop, mutant, cr_trial)
        trial_fit = func(trial)

        # selekce: potomek nahradí rodiče, pokud není horší
        better = trial_fit <= fit
        pop[better] = trial[better]
        fit[better] = trial_fit[better]
        if variant == "jde":
            # úspěšné parametry se zachovají, neúspěšné se zahodí
            f_i[better] = f_trial[better]
            cr_i[better] = cr_trial[better]

        history.append(fit.min())
        f_mean.append(f_i.mean())
        cr_mean.append(cr_i.mean())

    best = int(np.argmin(fit))
    return {
        "history": np.array(history),
        "best_x": pop[best].copy(),
        "best_f": float(fit[best]),
        "f_mean": np.array(f_mean),
        "cr_mean": np.array(cr_mean),
    }


VARIANTS = {
    "DE/rand/1/bin": "rand1",
    "DE/best/1/bin": "best1",
    "jDE": "jde",
}


def generations_to_target(history: np.ndarray, target: float = TARGET) -> float:
    """Index první generace s nejlepší fitness <= target; NaN, pokud cíl nebyl dosažen."""
    hit = np.nonzero(history <= target)[0]
    return float(hit[0]) if hit.size else float("nan")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def make_run_seeds(seed: int) -> dict[str, list[np.random.SeedSequence]]:
    """Pro každou funkci RUNS seedů; stejný seed -> stejné počáteční populace."""
    problem_seeds = np.random.SeedSequence(seed).spawn(len(PROBLEMS))
    return {name: ps.spawn(RUNS) for name, ps in zip(PROBLEMS, problem_seeds)}


def run_experiment() -> dict:
    """Pro každou funkci a běh: stejná počáteční populace pro všechny 3 varianty."""
    run_seeds = make_run_seeds(SEED)
    results: dict = {}
    for name, (func, low, high) in PROBLEMS.items():
        results[name] = {v: [] for v in VARIANTS}
        for run_seed in run_seeds[name]:
            init_rng = np.random.default_rng(run_seed)
            init_pop = init_rng.uniform(low, high, (POP_SIZE, DIM))
            for variant_name, variant in VARIANTS.items():
                # každá varianta má vlastní generátor, ale stejnou počáteční populaci
                rng = np.random.default_rng(run_seed.spawn(1)[0])
                results[name][variant_name].append(run_de(variant, func, low, high, init_pop, rng))
    return results


def write_statistics(results: dict) -> list[dict]:
    rows = []
    for problem, by_variant in results.items():
        for variant, runs in by_variant.items():
            finals = np.array([r["best_f"] for r in runs])
            gens = np.array([generations_to_target(r["history"]) for r in runs])
            reached = int(np.sum(~np.isnan(gens)))
            rows.append(
                {
                    "function": problem,
                    "variant": variant,
                    "best": finals.min(),
                    "mean": finals.mean(),
                    "std": finals.std(),
                    "reached_target": f"{reached}/{RUNS}",
                    "mean_gens_to_target": float(np.nanmean(gens)) if reached else float("nan"),
                    "generations": GENERATIONS,
                }
            )
    with open(STATISTICS_CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    return rows


COLORS = {"DE/rand/1/bin": "tab:blue", "DE/best/1/bin": "tab:red", "jDE": "tab:green"}


def plot_convergence(results: dict) -> None:
    """Průměrná konvergence přes všechny běhy (log osa)."""
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(5 * len(PROBLEMS), 4))
    for ax, (problem, by_variant) in zip(np.atleast_1d(axes), results.items()):
        for variant, runs in by_variant.items():
            mean_hist = np.mean([r["history"] for r in runs], axis=0)
            ax.plot(mean_hist, label=variant, color=COLORS[variant])
        ax.set_yscale("log")
        ax.set_title(f"{problem} (D={DIM}, průměr {RUNS} běhů)")
        ax.set_xlabel("Generace")
        ax.set_ylabel("Nejlepší fitness")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(CONVERGENCE_PLOT_PATH, dpi=150)
    plt.close(fig)


def plot_single_run(results: dict) -> None:
    """Konvergence prvního běhu (jedna společná počáteční populace) - ukázka pro protokol."""
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(5 * len(PROBLEMS), 4))
    for ax, (problem, by_variant) in zip(np.atleast_1d(axes), results.items()):
        for variant, runs in by_variant.items():
            ax.plot(runs[0]["history"], label=variant, color=COLORS[variant])
        ax.set_yscale("log")
        ax.set_title(f"{problem} (D={DIM}, jeden běh)")
        ax.set_xlabel("Generace")
        ax.set_ylabel("Nejlepší fitness")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(SINGLE_RUN_PLOT_PATH, dpi=150)
    plt.close(fig)


def plot_jde_parameters(results: dict) -> None:
    """Vývoj průměrného F a CR v populaci jDE (průměr přes běhy)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for problem, by_variant in results.items():
        runs = by_variant["jDE"]
        axes[0].plot(np.mean([r["f_mean"] for r in runs], axis=0), label=problem)
        axes[1].plot(np.mean([r["cr_mean"] for r in runs], axis=0), label=problem)
    axes[0].set_title("jDE: průměrné F v populaci")
    axes[1].set_title("jDE: průměrné CR v populaci")
    for ax in axes:
        ax.set_xlabel("Generace")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(JDE_PARAMS_PLOT_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    results = run_experiment()
    rows = write_statistics(results)
    plot_convergence(results)
    plot_single_run(results)
    plot_jde_parameters(results)

    print(f"{'Funkce':<12}{'Varianta':<16}{'best':>12}{'mean':>12}{'std':>12}{'cíl dosažen':>13}{'gen. do cíle':>14}")
    for r in rows:
        print(
            f"{r['function']:<12}{r['variant']:<16}{r['best']:>12.3e}{r['mean']:>12.3e}"
            f"{r['std']:>12.3e}{r['reached_target']:>13}{r['mean_gens_to_target']:>14.1f}"
        )


if __name__ == "__main__":
    main()
