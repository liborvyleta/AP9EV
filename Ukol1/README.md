# Úkol 1 — Genetický algoritmus (OneMax, LeadingOnes)

## Zadání
Implementace genetického algoritmu s binární reprezentací jedinců, elitismem,
selekcí (ruletová / pořadová), jednobodovým křížením a bitovou mutací.
Otestováno na úlohách **OneMax** a **LeadingOnes** v dimenzích D = 10, 30, 100,
vždy 10 nezávislých běhů s rozpočtem 100×D ohodnocení účelové funkce.

## Implementace (`ga.py`)

- **Reprezentace**: binární vektor délky D (numpy pole 0/1).
- **Elitismus**: `elite_frac` (výchozí 15 %) nejlepších jedinců přechází beze
  změny do nové populace.
- **Selekce rodičů**: `rank` (pořadová) nebo `roulette` (ruletová) — volitelné
  parametrem `selection`.
- **Křížení**: jednobodové (`one_point_crossover`) — náhodný bod dělení,
  výměna prvních částí obou rodičů → 2 potomci.
- **Mutace**: bitová inverze každého bitu s pravděpodobností `p_mut`
  (výchozí `1/D`, což odpovídá doporučenému rozmezí ~0.5–1 % pro D okolo
  100–200).
- **Cyklus**: elitní jedinci + opakovaná selekce → křížení → mutace, dokud
  nová populace nedosáhne velikosti `pop_size` (výchozí 30).
- **Rozpočet**: běh se zastaví po dosažení `100 × D` ohodnocení účelové
  funkce; sleduje se průběžně nejlepší nalezené řešení (konvergenční
  křivka).

Konvergenční křivky z jednotlivých běhů jsou interpolovány na společnou osu
počtu ohodnocení a zprůměrovány přes 10 běhů (`run_experiment`).

## Spuštění

```bash
python3 ga.py
```

Vygeneruje:
- `convergence.png` — průměrné konvergenční grafy (± směrodatná odchylka)
  pro obě úlohy a všechny tři dimenze,
- `statistics.csv` — základní statistiky finálního fitness přes 10 běhů
  (nejlepší, nejhorší, průměr, medián, směrodatná odchylka).

## Výsledky (výchozí nastavení: pop=30, elitismus 15 %, pořadová selekce, p_mut=1/D)

| problem      | D   | best | worst | mean  | median | std   |
|--------------|-----|------|-------|-------|--------|-------|
| OneMax       | 10  | 10   | 10    | 10.0  | 10.0   | 0.000 |
| OneMax       | 30  | 30   | 30    | 30.0  | 30.0   | 0.000 |
| OneMax       | 100 | 100  | 100   | 100.0 | 100.0  | 0.000 |
| LeadingOnes  | 10  | 10   | 10    | 10.0  | 10.0   | 0.000 |
| LeadingOnes  | 30  | 30   | 29    | 29.9  | 30.0   | 0.300 |
| LeadingOnes  | 100 | 80   | 55    | 71.9  | 76.0   | 8.264 |

## Diskuse nastavení parametrů

- **OneMax** je pro GA snadná úloha (fitness je aditivní, separabilní) —
  s daným nastavením se optimum spolehlivě najde ve všech dimenzích, a to i
  se značnou rezervou v rozpočtu.
- **LeadingOnes** je výrazně náročnější, protože zlepšení fitness vyžaduje
  "uhodnout" správnou hodnotu dalšího bitu zleva a mutace snadno kazí již
  nalezenou souvislou sekvenci jedniček. U D=100 s rozpočtem 10 000
  ohodnocení GA nestíhá dojít k optimu — to odpovídá teoretické složitosti
  LeadingOnes (řádově O(D²) ohodnocení pro (1+1)-EA), zatímco máme k
  dispozici jen O(100·D).
- Elitismus 15 % a pořadová selekce (rank) dávaly stabilnější a mírně
  rychlejší konvergenci než ruletová selekce, protože rank selekce méně
  trpí "dominancí" jedinců s výrazně vyšším fitness v raných fázích (u
  LeadingOnes se fitness soustředí na nízkých hodnotách, ruleta pak vybírá
  téměř náhodně).
- Pravděpodobnost mutace `1/D` se v testech ukázala jako rozumný kompromis
  — nižší hodnoty (blíže 0.5 %) zpomalují explorate u LeadingOnes, vyšší
  (blíže 1 %) zvyšují riziko destrukce dobrých řešení u velkého D.
- Pro experimentování lze v `main()` v `ga.py` snadno měnit `pop_size`,
  `elite_frac`, `selection` a `p_mut` a porovnávat výsledné konvergenční
  křivky a statistiky.
