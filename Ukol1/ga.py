"""Genetický algoritmus s binární reprezentací pro OneMax a LeadingOnes.

Operátory:
- Elitismus (podíl nejlepších jedinců přenesených beze změny)
- Selekce: ruletová (roulette) nebo pořadová (rank)
- Jednobodové křížení (one-point crossover)
- Bitová mutace (bit-flip) s danou pravděpodobností

Experiment:
- Úlohy: OneMax, LeadingOnes
- Dimenze D: 10, 30, 100
- Rozpočet ohodnocení účelové funkce: 100 * D
- Počet nezávislých běhů: 10
- Výstup: průměrný konvergenční graf + základní statistiky
"""

from __future__ import annotations

import csv
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

FitnessFunc = Callable[[np.ndarray], int]
SelectorFunc = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]

# Výchozí parametry GA (viz README pro zdůvodnění).
DEFAULT_POP_SIZE = 30
DEFAULT_ELITE_FRACTION = 0.15
DEFAULT_SELECTION = "rank"
DIMENSIONS = (10, 30, 100)
PROBLEM_NAMES = ("OneMax", "LeadingOnes")
RUNS_PER_EXPERIMENT = 10
EVAL_BUDGET_MULTIPLIER = 100
CONVERGENCE_PLOT_PATH = "convergence.png"
STATISTICS_CSV_PATH = "statistics.csv"


# ---------------------------------------------------------------------------
# Účelové funkce
# ---------------------------------------------------------------------------


def one_max(individual: np.ndarray) -> int:
    """Spočítá počet jedniček v binárním řetězci.

    Args:
        individual: Binární vektor (0/1) reprezentující jedince.

    Returns:
        Počet jedniček — hodnota účelové funkce OneMax.
    """
    return int(np.sum(individual))


def leading_ones(individual: np.ndarray) -> int:
    """Spočítá délku souvislé řady jedniček od levého konce řetězce.

    Args:
        individual: Binární vektor (0/1) reprezentující jedince.

    Returns:
        Počet počátečních jedniček — hodnota účelové funkce LeadingOnes.
    """
    if np.all(individual == 1):
        return len(individual)
    first_zero_index = np.argmin(individual)
    return int(first_zero_index)


FITNESS_FUNCS: dict[str, FitnessFunc] = {
    "OneMax": one_max,
    "LeadingOnes": leading_ones,
}


# ---------------------------------------------------------------------------
# Genetické operátory
# ---------------------------------------------------------------------------


def init_population(pop_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Vygeneruje náhodnou počáteční populaci binárních jedinců.

    Args:
        pop_size: Počet jedinců v populaci.
        dim: Délka binárního řetězce (D).
        rng: Generátor náhodných čísel.

    Returns:
        Pole tvaru (pop_size, dim) s hodnotami 0/1.
    """
    return rng.integers(0, 2, size=(pop_size, dim))


def evaluate(population: np.ndarray, fitness_func: FitnessFunc) -> np.ndarray:
    """Ohodnotí každého jedince v populaci danou účelovou funkcí.

    Args:
        population: Pole jedinců tvaru (pop_size, dim).
        fitness_func: Účelová funkce jednoho jedince.

    Returns:
        Pole hodnot fitness, jedna na jedince.
    """
    return np.array([fitness_func(individual) for individual in population])


def select_roulette(
    population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Vybere jednoho rodiče ruletovou selekcí (pravděpodobnost ~ fitness).

    Args:
        population: Aktuální populace, tvar (pop_size, dim).
        fitness: Hodnoty fitness odpovídající populaci.
        rng: Generátor náhodných čísel.

    Returns:
        Vybraný jedinec (kopie řádku z population).
    """
    # Posun na kladné hodnoty, aby šlo počítat pravděpodobnost i při záporném fitness.
    shifted_fitness = fitness - fitness.min() + np.finfo(float).eps
    probabilities = shifted_fitness / shifted_fitness.sum()
    selected_index = rng.choice(len(population), p=probabilities)
    return population[selected_index]


def select_rank(
    population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Vybere jednoho rodiče pořadovou (rank) selekcí.

    Pravděpodobnost výběru závisí na pořadí jedince podle fitness, ne na
    absolutní hodnotě — odolnější vůči extrémním rozdílům ve fitness.

    Args:
        population: Aktuální populace, tvar (pop_size, dim).
        fitness: Hodnoty fitness odpovídající populaci.
        rng: Generátor náhodných čísel.

    Returns:
        Vybraný jedinec (kopie řádku z population).
    """
    ascending_order = np.argsort(fitness)
    ranks = np.empty(len(fitness), dtype=float)
    ranks[ascending_order] = np.arange(1, len(fitness) + 1)
    probabilities = ranks / ranks.sum()
    selected_index = rng.choice(len(population), p=probabilities)
    return population[selected_index]


SELECTORS: dict[str, SelectorFunc] = {
    "roulette": select_roulette,
    "rank": select_rank,
}


def one_point_crossover(
    parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Provede jednobodové křížení dvou rodičů.

    Args:
        parent1: První rodič, binární vektor délky D.
        parent2: Druhý rodič, binární vektor délky D.
        rng: Generátor náhodných čísel.

    Returns:
        Dvojice potomků vzniklých výměnou částí za náhodným bodem řezu.
    """
    dim = len(parent1)
    crossover_point = rng.integers(1, dim)  # 1..dim-1, vyloučí krajní body
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2


def mutate(
    individual: np.ndarray, mutation_probability: float, rng: np.random.Generator
) -> np.ndarray:
    """Provede bitovou mutaci — každý bit se s danou pravděpodobností invertuje.

    Args:
        individual: Binární vektor, který se má zmutovat.
        mutation_probability: Pravděpodobnost inverze jednoho bitu.
        rng: Generátor náhodných čísel.

    Returns:
        Nová (zmutovaná) kopie jedince.
    """
    flip_mask = rng.random(len(individual)) < mutation_probability
    mutated = individual.copy()
    mutated[flip_mask] = 1 - mutated[flip_mask]
    return mutated


# ---------------------------------------------------------------------------
# Vlastní GA
# ---------------------------------------------------------------------------


def _build_next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    pop_size: int,
    n_elite: int,
    selector: SelectorFunc,
    mutation_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sestaví novou generaci: elitismus + selekce/křížení/mutace potomků.

    Args:
        population: Aktuální populace, tvar (pop_size, dim).
        fitness: Hodnoty fitness odpovídající populaci.
        pop_size: Cílová velikost nové populace.
        n_elite: Počet nejlepších jedinců přenesených beze změny.
        selector: Funkce pro výběr jednoho rodiče.
        mutation_probability: Pravděpodobnost inverze jednoho bitu.
        rng: Generátor náhodných čísel.

    Returns:
        Nová populace, tvar (pop_size, dim).
    """
    descending_order = np.argsort(fitness)[::-1]
    elite_indices = descending_order[:n_elite]
    new_population = [population[i].copy() for i in elite_indices]

    while len(new_population) < pop_size:
        parent1 = selector(population, fitness, rng)
        parent2 = selector(population, fitness, rng)
        child1, child2 = one_point_crossover(parent1, parent2, rng)
        child1 = mutate(child1, mutation_probability, rng)
        child2 = mutate(child2, mutation_probability, rng)
        new_population.append(child1)
        if len(new_population) < pop_size:
            new_population.append(child2)

    return np.array(new_population[:pop_size])


def run_ga(
    fitness_func: FitnessFunc,
    dim: int,
    max_evals: int,
    pop_size: int = DEFAULT_POP_SIZE,
    elite_fraction: float = DEFAULT_ELITE_FRACTION,
    mutation_probability: float | None = None,
    selection: str = DEFAULT_SELECTION,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Provede jeden běh genetického algoritmu.

    Args:
        fitness_func: Účelová funkce jednoho jedince (OneMax/LeadingOnes).
        dim: Délka binárního řetězce (D).
        max_evals: Maximální počet ohodnocení účelové funkce.
        pop_size: Velikost populace.
        elite_fraction: Podíl nejlepších jedinců přenesených beze změny.
        mutation_probability: Pravděpodobnost inverze bitu; `None` použije 1/D.
        selection: Typ selekce, `"rank"` nebo `"roulette"`.
        seed: Seed generátoru náhodných čísel pro reprodukovatelnost.

    Returns:
        Dvojice polí `(evals, best_so_far)` sledujících vývoj nejlepšího
        nalezeného fitness v závislosti na počtu ohodnocení.
    """
    rng = np.random.default_rng(seed)
    if mutation_probability is None:
        mutation_probability = 1.0 / dim  # doporučené rozmezí ~0.5-1 % pro D~100-200

    selector = SELECTORS[selection]
    n_elite = max(1, round(elite_fraction * pop_size))

    population = init_population(pop_size, dim, rng)
    fitness = evaluate(population, fitness_func)
    evals = pop_size

    best_so_far = fitness.max()
    evals_history = [evals]
    best_history = [best_so_far]

    while evals < max_evals:
        population = _build_next_generation(
            population, fitness, pop_size, n_elite, selector, mutation_probability, rng
        )
        fitness = evaluate(population, fitness_func)
        evals += pop_size
        best_so_far = max(best_so_far, fitness.max())

        evals_history.append(evals)
        best_history.append(best_so_far)

    return np.array(evals_history), np.array(best_history)


# ---------------------------------------------------------------------------
# Experiment: N běhů, průměrná konvergence + interpolace na společnou osu
# ---------------------------------------------------------------------------


def run_experiment(
    problem_name: str,
    dim: int,
    n_runs: int = RUNS_PER_EXPERIMENT,
    budget_multiplier: int = EVAL_BUDGET_MULTIPLIER,
    **ga_kwargs: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Spustí opakované běhy GA nad jednou úlohou a shrne výsledky.

    Args:
        problem_name: Klíč do `FITNESS_FUNCS` (`"OneMax"` nebo `"LeadingOnes"`).
        dim: Délka binárního řetězce (D).
        n_runs: Počet nezávislých běhů.
        budget_multiplier: Rozpočet ohodnocení účelové funkce jako násobek D.
        **ga_kwargs: Další parametry předané do `run_ga`.

    Returns:
        Trojice `(common_evals, all_curves, stats)`, kde `common_evals` je
        společná osa počtu ohodnocení, `all_curves` obsahuje interpolovanou
        konvergenční křivku každého běhu a `stats` obsahuje základní
        statistiky finálního fitness (best/worst/mean/median/std).
    """
    fitness_func = FITNESS_FUNCS[problem_name]
    max_evals = budget_multiplier * dim
    common_evals = np.arange(0, max_evals + 1, max(1, dim))

    all_curves = []
    final_bests = []
    for run_index in range(n_runs):
        run_seed = 1000 * run_index + dim
        evals, best = run_ga(fitness_func, dim, max_evals, seed=run_seed, **ga_kwargs)
        all_curves.append(np.interp(common_evals, evals, best))
        final_bests.append(best[-1])

    all_curves_array = np.array(all_curves)
    final_bests_array = np.array(final_bests)

    stats = {
        "best": final_bests_array.max(),
        "worst": final_bests_array.min(),
        "mean": final_bests_array.mean(),
        "median": np.median(final_bests_array),
        "std": final_bests_array.std(),
    }

    return common_evals, all_curves_array, stats


def _plot_convergence(
    ax: plt.Axes, evals: np.ndarray, curves: np.ndarray, problem: str, dim: int
) -> None:
    """Vykreslí průměrnou konvergenční křivku s pásmem ±1 std do daných os.

    Args:
        ax: Matplotlib osy, do kterých se graf vykreslí.
        evals: Společná osa počtu ohodnocení účelové funkce.
        curves: Interpolované konvergenční křivky, tvar (n_runs, len(evals)).
        problem: Název úlohy (pro titulek grafu).
        dim: Délka binárního řetězce (pro titulek a linku optima).
    """
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    ax.plot(evals, mean_curve, label="průměr")
    ax.fill_between(
        evals, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="±1 std"
    )
    ax.axhline(dim, color="gray", linestyle="--", linewidth=0.8, label="optimum")
    ax.set_title(f"{problem}, D={dim}")
    ax.set_xlabel("počet ohodnocení ÚF")
    ax.set_ylabel("nejlepší fitness")
    ax.legend(fontsize=7)


def main() -> None:
    """Spustí experiment pro OneMax a LeadingOnes napříč dimenzemi a uloží výstupy."""
    ga_settings = {
        "pop_size": DEFAULT_POP_SIZE,
        "elite_fraction": DEFAULT_ELITE_FRACTION,
        "selection": DEFAULT_SELECTION,
        "mutation_probability": None,
    }

    all_stats = []
    _, axes = plt.subplots(len(PROBLEM_NAMES), len(DIMENSIONS), figsize=(15, 8), squeeze=False)

    for row, problem in enumerate(PROBLEM_NAMES):
        for col, dim in enumerate(DIMENSIONS):
            evals, curves, stats = run_experiment(problem, dim, **ga_settings)
            _plot_convergence(axes[row][col], evals, curves, problem, dim)

            all_stats.append({"problem": problem, "dim": dim, **stats})
            print(
                f"{problem} D={dim}: best={stats['best']:.2f} worst={stats['worst']:.2f} "
                f"mean={stats['mean']:.2f} median={stats['median']:.2f} std={stats['std']:.3f}"
            )

    plt.tight_layout()
    plt.savefig(CONVERGENCE_PLOT_PATH, dpi=150)
    print(f"\nGraf uložen do {CONVERGENCE_PLOT_PATH}")

    fieldnames = ["problem", "dim", "best", "worst", "mean", "median", "std"]
    with open(STATISTICS_CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"Statistiky uloženy do {STATISTICS_CSV_PATH}")
    print("\n".join(", ".join(f"{key}={row[key]}" for key in fieldnames) for row in all_stats))


if __name__ == "__main__":
    main()
