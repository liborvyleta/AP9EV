"""Genetický algoritmus pro optimalizaci spojitých funkcí (minimalizace).

Varianty:
- binární kódování, 32 bitů na dimenzi: IEEE 754 (float32), Fixed-Point, BCD
  (jednobodové křížení přes celý chromozom + bit-flip mutace),
- Real-valued GA s Gaussovou mutací (x + N(0, sigma^2)),
- Real-valued GA s rovnoměrnou mutací z celého rozsahu.

Experiment: Sphere, Rosenbrock, Schwefel; D = 10; 10 000 ohodnocení; 10 běhů.
"""

from __future__ import annotations

import csv
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# Společné parametry GA
POP_SIZE = 50
ELITE_FRACTION = 0.1
CROSSOVER_PROBABILITY = 0.9
DIM = 10
BITS_PER_DIM = 32
EVAL_BUDGET = 10_000
RUNS = 10
SEED = 42

# Výchozí mutace real-valued GA (vybráno z parameter_sweep, viz README)
DEFAULT_SIGMA = 0.05  # relativně k šířce rozsahu
DEFAULT_GENE_MUTATION = 0.1

STATISTICS_CSV_PATH = "statistics.csv"
SWEEP_CSV_PATH = "sigma_sweep.csv"
CONVERGENCE_PLOT_PATH = "convergence.png"

# ---------------------------------------------------------------------------
# Účelové funkce (vstup: pole (n, D), výstup: pole (n,))
# ---------------------------------------------------------------------------


def sphere(x: np.ndarray) -> np.ndarray:
    """Sphere: sum(x_i^2); minimum 0 v bodě 0."""
    return np.sum(x**2, axis=1)


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """Rosenbrock: sum(100 (x_{i+1}-x_i^2)^2 + (1-x_i)^2); minimum 0 v bodě 1."""
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2, axis=1)


def schwefel(x: np.ndarray) -> np.ndarray:
    """Schwefel: 418.9829 D - sum(x_i sin(sqrt|x_i|)); minimum ~0 v bodě 420.9687."""
    return 418.9829 * x.shape[1] - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


# název -> (funkce, dolní mez, horní mez)
PROBLEMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], float, float]] = {
    "Sphere": (sphere, -5.12, 5.12),
    "Rosenbrock": (rosenbrock, -5.0, 10.0),
    "Schwefel": (schwefel, -500.0, 500.0),
}

# ---------------------------------------------------------------------------
# Dekódování bitů na reálná čísla; vstup (n, D, 32) bitů, výstup (n, D)
# ---------------------------------------------------------------------------

_BIT_WEIGHTS = 2 ** np.arange(BITS_PER_DIM - 1, -1, -1, dtype=np.uint64)


def _bits_to_uint(bits: np.ndarray) -> np.ndarray:
    """Převede bity (MSB první) na nezáporná celá čísla."""
    return (bits.astype(np.uint64) * _BIT_WEIGHTS[-bits.shape[-1]:]).sum(axis=-1)


def _sanitize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Ošetří NaN/Inf a ořízne hodnoty do povoleného rozsahu."""
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=hi, neginf=lo)
    return np.clip(values, lo, hi)


def decode_ieee754(bits: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """IEEE 754 float32: 1 bit znaménko, 8 bitů exponent, 23 bitů mantisa."""
    raw = _bits_to_uint(bits).astype(np.uint32)
    with np.errstate(all="ignore"):
        return _sanitize(raw.view(np.float32), lo, hi)


def decode_fixed_point(bits: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fixed-point: 1 bit znaménko, 10 bitů celá část, 21 bitů desetinná část."""
    sign = np.where(bits[..., 0] == 1, -1.0, 1.0)
    integer = _bits_to_uint(bits[..., 1:11]).astype(np.float64)
    fraction = _bits_to_uint(bits[..., 11:]).astype(np.float64) / 2.0**21
    return _sanitize(sign * (integer + fraction), lo, hi)


def decode_bcd(bits: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """BCD: 8 číslic po 4 bitech = [znaménko] [3 celé číslice] [4 desetinné].

    Nibble > 9 není platná číslice -> modulo 10. Znaménko: číslice >= 5 je záporné.
    """
    nibbles = bits.reshape(*bits.shape[:-1], 8, 4)
    digits = (nibbles * np.array([8, 4, 2, 1])).sum(axis=-1) % 10
    sign = np.where(digits[..., 0] >= 5, -1.0, 1.0)
    magnitude = (digits[..., 1:] * 10.0 ** np.arange(2, -5, -1)).sum(axis=-1)
    return _sanitize(sign * magnitude, lo, hi)


def decode_fixed_point_tuned(bits: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Fixed-point s dělením bitů přizpůsobeným rozsahu úlohy.

    1 bit znaménko, `int_bits` = nejmenší počet bitů pro celou část max(|lo|, |hi|),
    zbylé bity (31 - int_bits) desetinná část.
    """
    int_bits = int(np.ceil(np.log2(max(abs(lo), abs(hi)) + 1.0)))
    frac_bits = BITS_PER_DIM - 1 - int_bits
    sign = np.where(bits[..., 0] == 1, -1.0, 1.0)
    integer = _bits_to_uint(bits[..., 1 : 1 + int_bits]).astype(np.float64)
    fraction = _bits_to_uint(bits[..., 1 + int_bits :]).astype(np.float64) / 2.0**frac_bits
    return _sanitize(sign * (integer + fraction), lo, hi)


def decode_bcd_tuned(bits: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """BCD s dělením číslic přizpůsobeným rozsahu: [znaménko] [celé] [desetinné].

    Počet celých číslic odpovídá max(|lo|, |hi|), zbylé číslice jsou desetinné.
    """
    int_digits = len(str(int(max(abs(lo), abs(hi)))))
    nibbles = bits.reshape(*bits.shape[:-1], 8, 4)
    digits = (nibbles * np.array([8, 4, 2, 1])).sum(axis=-1) % 10
    sign = np.where(digits[..., 0] >= 5, -1.0, 1.0)
    exponents = 10.0 ** np.arange(int_digits - 1, int_digits - 8, -1)
    magnitude = (digits[..., 1:] * exponents).sum(axis=-1)
    return _sanitize(sign * magnitude, lo, hi)


DECODERS = {
    "IEEE754": decode_ieee754,
    "FixedPoint": decode_fixed_point,
    "BCD": decode_bcd,
    "FixedPoint-tuned": decode_fixed_point_tuned,
    "BCD-tuned": decode_bcd_tuned,
}

# ---------------------------------------------------------------------------
# GA — společná kostra
# ---------------------------------------------------------------------------


def rank_select(fitness: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pořadová selekce pro minimalizaci; vrací n indexů rodičů."""
    order = np.argsort(-fitness)  # nejhorší první -> nejhorší má pořadí 1
    ranks = np.empty(len(fitness))
    ranks[order] = np.arange(1, len(fitness) + 1)
    return rng.choice(len(fitness), size=n, p=ranks / ranks.sum())


def one_point_crossover(parents: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Jednobodové křížení po dvojicích (řádky 2k a 2k+1) podél osy genů.

    Pole `parents` má tvar (n, L); L je počet bitů (binární GA) nebo dimenzí (real GA).
    """
    children = parents.copy()
    length = parents.shape[1]
    for i in range(0, len(parents) - 1, 2):
        if rng.random() < CROSSOVER_PROBABILITY:
            point = rng.integers(1, length)
            children[i, point:] = parents[i + 1, point:]
            children[i + 1, point:] = parents[i, point:]
    return children


def run_ga(
    init: Callable[[int, np.random.Generator], np.ndarray],
    mutate: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Spustí GA s elitismem na přesně EVAL_BUDGET ohodnocení.

    Returns:
        Pole délky EVAL_BUDGET: nejlepší dosud nalezená hodnota po k-tém ohodnocení.
    """
    population = init(POP_SIZE, rng)
    fitness = evaluate(population)
    history = list(fitness)
    n_elite = max(1, int(round(ELITE_FRACTION * POP_SIZE)))

    while len(history) < EVAL_BUDGET:
        elite_idx = np.argsort(fitness)[:n_elite]
        elite, elite_fitness = population[elite_idx], fitness[elite_idx]
        n_children = POP_SIZE - n_elite
        parents = population[rank_select(fitness, n_children + n_children % 2, rng)]
        children = mutate(one_point_crossover(parents, rng), rng)[:n_children]
        remaining = EVAL_BUDGET - len(history)
        children = children[:remaining]  # poslední generace může být neúplná
        child_fitness = evaluate(children)
        history.extend(child_fitness)
        population = np.vstack([elite, children])
        fitness = np.concatenate([elite_fitness, child_fitness])

    return np.minimum.accumulate(np.array(history[:EVAL_BUDGET]))


# ---------------------------------------------------------------------------
# Konkrétní varianty
# ---------------------------------------------------------------------------


def make_binary_variant(encoding: str, problem: str):
    """Sestaví (init, mutate, evaluate) pro binární GA se zvoleným kódováním."""
    func, lo, hi = PROBLEMS[problem]
    decode = DECODERS[encoding]
    length = DIM * BITS_PER_DIM
    p_bit = 1.0 / length

    def init(n, rng):
        return rng.integers(0, 2, size=(n, length), dtype=np.uint8)

    def mutate(pop, rng):
        flips = rng.random(pop.shape) < p_bit
        return np.where(flips, 1 - pop, pop).astype(np.uint8)

    def evaluate(pop):
        return func(decode(pop.reshape(len(pop), DIM, BITS_PER_DIM), lo, hi))

    return init, mutate, evaluate


def make_real_variant(
    mutation: str, problem: str, sigma: float = DEFAULT_SIGMA, p_gene: float = DEFAULT_GENE_MUTATION
):
    """Sestaví (init, mutate, evaluate) pro real-valued GA.

    Args:
        mutation: "gauss" (x + N(0, (sigma*(hi-lo))^2)) nebo "uniform" (celý rozsah).
        sigma: směrodatná odchylka Gaussovy mutace relativně k šířce rozsahu.
        p_gene: pravděpodobnost mutace jednoho genu (dimenze).
    """
    func, lo, hi = PROBLEMS[problem]
    width = hi - lo

    def init(n, rng):
        return rng.uniform(lo, hi, size=(n, DIM))

    def mutate(pop, rng):
        mask = rng.random(pop.shape) < p_gene
        if mutation == "gauss":
            new = np.clip(pop + rng.normal(0.0, sigma * width, pop.shape), lo, hi)
        else:
            new = rng.uniform(lo, hi, pop.shape)
        return np.where(mask, new, pop)

    return init, mutate, func


def variants_for(problem: str) -> dict[str, tuple]:
    """Všech pět variant GA pro danou úlohu."""
    variants = {f"Binary-{name}": make_binary_variant(name, problem) for name in DECODERS}
    variants["Real-Gauss"] = make_real_variant("gauss", problem)
    variants["Real-Uniform"] = make_real_variant("uniform", problem)
    return variants


def run_variant(variant: tuple, seed: int) -> np.ndarray:
    """Provede RUNS nezávislých běhů; vrací pole (RUNS, EVAL_BUDGET)."""
    init, mutate, evaluate = variant
    seeds = np.random.SeedSequence(seed).spawn(RUNS)
    return np.array([run_ga(init, mutate, evaluate, np.random.default_rng(s)) for s in seeds])


def summarize(histories: np.ndarray) -> dict[str, float]:
    """Statistiky finálních hodnot přes běhy."""
    final = histories[:, -1]
    return {
        "best": final.min(),
        "worst": final.max(),
        "mean": final.mean(),
        "median": float(np.median(final)),
        "std": final.std(ddof=1),
    }


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parameter_sweep() -> list[dict]:
    """Vliv sigma a pravděpodobnosti mutace genu na Real-Gauss (všechny úlohy)."""
    rows = []
    for problem in PROBLEMS:
        for sigma in (0.01, 0.05, 0.2):
            for p_gene in (0.05, 0.1, 0.3):
                variant = make_real_variant("gauss", problem, sigma, p_gene)
                stats = summarize(run_variant(variant, SEED))
                rows.append({"problem": problem, "sigma": sigma, "p_gene": p_gene, **stats})
                print(f"sweep {problem} sigma={sigma} p={p_gene}: mean={stats['mean']:.4g}")
    return rows


def main() -> None:
    rows = []
    _, axes = plt.subplots(1, len(PROBLEMS), figsize=(16, 4.5))
    grid = np.arange(1, EVAL_BUDGET + 1)

    for ax, problem in zip(axes, PROBLEMS):
        for name, variant in variants_for(problem).items():
            histories = run_variant(variant, SEED)
            stats = summarize(histories)
            rows.append({"problem": problem, "variant": name, **stats})
            print(f"{problem:11s} {name:15s} mean={stats['mean']:.4g} std={stats['std']:.4g}")
            ax.plot(grid, histories.mean(axis=0), label=name)
        ax.set_title(f"{problem} ({DIM}D)")
        ax.set_xlabel("počet ohodnocení")
        ax.set_ylabel("průměr nejlepších hodnot")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(CONVERGENCE_PLOT_PATH, dpi=150)

    write_csv(STATISTICS_CSV_PATH, rows, ["problem", "variant", "best", "worst", "mean", "median", "std"])
    write_csv(
        SWEEP_CSV_PATH,
        parameter_sweep(),
        ["problem", "sigma", "p_gene", "best", "worst", "mean", "median", "std"],
    )


if __name__ == "__main__":
    main()
