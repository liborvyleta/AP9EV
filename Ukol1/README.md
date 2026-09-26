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
- **Křížení**: v hlavních experimentech jednobodové (`one_point_crossover`) —
  náhodný bod dělení a výměna částí rodičů. Pro srovnání parametrů je dostupné
  také dvoubodové křížení.
- **Mutace**: bitová inverze každého bitu s pravděpodobností `p_mut`
  (výchozí 1 % pro všechny dimenze; ve srovnání se testuje také 0,5 %).
- **Cyklus**: elitní jedinci + opakovaná selekce → křížení → mutace, dokud
  nová populace nedosáhne velikosti `pop_size` (výchozí 30).
- **Rozpočet**: běh provede přesně `100 × D` ohodnocení účelové funkce.
  Pokud zbývá méně než celá generace, vyhodnotí se jen potřebný počet jedinců.
  Sleduje se průběžně nejlepší nalezené řešení.

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
- `parameter_comparison.csv` — srovnání šesti nastavení na náročném případu
  LeadingOnes D=100, každé přes 10 běhů.

## Výsledky (pop=30, elitismus 15 %, pořadová selekce, jednobodové křížení, p_mut=1 %)

| problem      | D   | best | worst | mean  | median | std   |
|--------------|-----|------|-------|-------|--------|-------|
| OneMax       | 10  | 10   | 10    | 10.0  | 10.0   | 0.000 |
| OneMax       | 30  | 30   | 30    | 30.0  | 30.0   | 0.000 |
| OneMax       | 100 | 100  | 100   | 100.0 | 100.0  | 0.000 |
| LeadingOnes  | 10  | 10   | 10    | 10.0  | 10.0   | 0.000 |
| LeadingOnes  | 30  | 30   | 22    | 27.9  | 30.0   | 3.239 |
| LeadingOnes  | 100 | 80   | 55    | 71.8  | 76.0   | 8.352 |

## Porovnání parametrů

Parametry byly porovnány na LeadingOnes D=100 s rozpočtem 10 000 evaluací a
10 běhy pro každou konfiguraci. Vždy se měnila jedna vlastnost vůči základnímu
nastavení; křížení bylo testováno jednobodové i dvoubodové.

| konfigurace | populace | elitismus | selekce | křížení | mutace | průměr | std |
|-------------|---------:|----------:|---------|---------|-------:|-------:|----:|
| základní | 30 | 15 % | rank | jednobodové | 1 % | 71,8 | 8,35 |
| větší populace | 60 | 15 % | rank | jednobodové | 1 % | 56,4 | 6,68 |
| nižší elitismus | 30 | 10 % | rank | jednobodové | 1 % | 69,7 | 5,81 |
| ruletová selekce | 30 | 15 % | roulette | jednobodové | 1 % | 67,3 | 7,46 |
| nižší mutace | 30 | 15 % | rank | jednobodové | 0,5 % | 58,7 | 9,96 |
| dvoubodové křížení | 30 | 15 % | rank | dvoubodové | 1 % | 79,6 | 8,06 |

V tomto omezeném srovnání vyšlo nejlépe dvoubodové křížení; hlavní požadované
experimenty ale zůstávají s jednobodovým křížením podle zadání. Populace 30,
elitismus 15 %, pořadová selekce a mutace 1 % jsou rozumný výchozí kompromis
pro základní běhy. Výsledky jsou z pouhých 10 běhů na jedné dimenzi, proto je
ber jako orientační, ne jako univerzální optimum.

## Diskuse výsledků

- **OneMax** je pro GA snadná úloha (fitness je aditivní, separabilní) —
  s daným nastavením se optimum spolehlivě najde ve všech dimenzích, a to i
  se značnou rezervou v rozpočtu.
- **LeadingOnes** je výrazně náročnější, protože zlepšení fitness vyžaduje
  "uhodnout" správnou hodnotu dalšího bitu zleva a mutace snadno kazí již
  nalezenou souvislou sekvenci jedniček. U D=100 s rozpočtem 10 000
  ohodnocení GA nestíhá dojít k optimu — to odpovídá teoretické složitosti
  LeadingOnes (řádově O(D²) ohodnocení pro (1+1)-EA), zatímco máme k
  dispozici jen O(100·D).
- V konkrétním srovnání na LeadingOnes D=100 dosáhla pořadová selekce vyššího
  průměru než ruletová. Vyšší velikost populace ani mutace 0,5 % v tomto
  rozpočtu nepomohly.
- Mutace 1 % odpovídá přímo doporučenému rozmezí v zadání a používá se stejně
  pro všechna D; nejde o dřívější nastavení `1/D`, které pro nižší dimenze
  znamenalo podstatně vyšší pravděpodobnost mutace.
