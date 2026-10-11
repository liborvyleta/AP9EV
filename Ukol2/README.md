# Úkol 2 — GA pro spojité (reálné) optimalizační úlohy

## Zadání
Porovnat varianty genetického algoritmu na reálných funkcích: tři binární
kódování (IEEE 754, Fixed-Point, BCD; vždy 32 bitů na dimenzi) a dvě varianty
real-valued GA. Funkce **Sphere**, **Rosenbrock**, **Schwefel**, každá v 10D,
10 000 ohodnocení, 10 nezávislých běhů (minimalizace).

| Funkce     | Rozsah          | Optimum                |
|------------|-----------------|------------------------|
| Sphere     | [-5.12, 5.12]   | f(0,…,0) = 0           |
| Rosenbrock | [-5, 10]        | f(1,…,1) = 0           |
| Schwefel   | [-500, 500]     | f(420.9687,…) ≈ 0      |

## Implementace (`ga_real.py`)
Společná kostra: populace 50, elitismus 10 %, pořadová selekce, jednobodové
křížení (p = 0,9), rozpočet přesně 10 000 ohodnocení (elita se znovu
neohodnocuje).

- **IEEE 754**: 32 bitů = float32 (1 znaménko, 8 exponent, 23 mantisa). NaN/Inf
  se ošetří, hodnoty se ořežou do rozsahu úlohy.
- **Fixed-Point**: 1 bit znaménko, 10 bitů celá část, 21 bitů desetinná část.
- **BCD**: 8 číslic po 4 bitech = znaménko (číslice ≥ 5 → záporné), 3 celé a
  4 desetinné číslice; nibble > 9 se bere modulo 10.
- Binární varianty: křížení přes celý 320bitový chromozom, bit-flip mutace s
  p = 1/320.
- **Real-Gauss**: křížení mezi dimenzemi, mutace `x + N(0, sigma²)` každého genu
  s pravděpodobností p_gene, `sigma` zadána relativně k šířce rozsahu (výchozí
  sigma = 0,05, p_gene = 0,1), výsledek oříznut do rozsahu.
- **Real-Uniform**: křížení mezi dimenzemi, mutace genu (p_gene = 0,1) novou
  hodnotou z celého rozsahu.

## Spuštění
```bash
python3 ga_real.py
```
Vytvoří `statistics.csv`, `sigma_sweep.csv`, `convergence.png`.

## Výsledky (finální nejlepší hodnota, 10 běhů)

| Funkce     | Varianta        | best    | mean     | std      |
|------------|-----------------|---------|----------|----------|
| Sphere | Binary-IEEE754 | 1.83e-79 | 1.57e-78 | 1.62e-78 |
| Sphere | Binary-FixedPoint | 78.6 | 163 | 36.7 |
| Sphere | Binary-BCD | 52.5 | 97.4 | 29.2 |
| Sphere | Real-Gauss | 0.000169 | 0.000635 | 0.000386 |
| Sphere | Real-Uniform | 0.00949 | 0.025 | 0.0179 |
| Rosenbrock | Binary-IEEE754 | 8.76 | 8.94 | 0.0664 |
| Rosenbrock | Binary-FixedPoint | 3.23e+05 | 5.38e+05 | 1.16e+05 |
| Rosenbrock | Binary-BCD | 2.66e+05 | 3.43e+05 | 6.81e+04 |
| Rosenbrock | Real-Gauss | 4.34 | 33.4 | 71.7 |
| Rosenbrock | Real-Uniform | 12.6 | 48.5 | 30.8 |
| Schwefel | Binary-IEEE754 | 2.03e+03 | 2.27e+03 | 139 |
| Schwefel | Binary-FixedPoint | 373 | 967 | 376 |
| Schwefel | Binary-BCD | 186 | 544 | 236 |
| Schwefel | Real-Gauss | 119 | 386 | 176 |
| Schwefel | Real-Uniform | 5.82 | 15.5 | 6.37 |

Průměrné konvergenční křivky: `convergence.png` (logaritmická osa).

### Vliv sigma a pravděpodobnosti mutace (Real-Gauss, `sigma_sweep.csv`)
Průměr finální hodnoty, sigma ∈ {0,01; 0,05; 0,2}, p_gene ∈ {0,05; 0,1; 0,3}:

| Funkce     | nejlepší nastavení (mean)        | nejhorší nastavení (mean)       |
|------------|----------------------------------|---------------------------------|
| Sphere     | sigma 0,01, p 0,05 (4·10⁻⁵)      | sigma 0,2, p 0,3 (0,26)         |
| Rosenbrock | sigma 0,05, p 0,3 (16,6)         | sigma 0,2, p 0,3 (129)          |
| Schwefel   | sigma 0,2, p 0,05 (193)          | sigma 0,2, p 0,3 (625)          |

Malá sigma a nízká p_gene vyhovují hladké unimodální Sphere (jemné ladění);
multimodální Schwefel naopak profituje z větších skoků. Vysoká p_gene spolu s
velkou sigma ničí dobré jedince (příliš mnoho náhodnosti). Experiment je
jednoduchý, nejde o plně vyladěné nastavení; pro hlavní srovnání je použito
společné výchozí nastavení.

## Zdůvodnění výsledků

Aby zdůvodnění nebylo jen odhad, změřil jsem, jak vypadají hodnoty dekódované
z náhodných bitových řetězců (20 000 × 10 genů; reprodukovatelné voláním
dekodérů z `ga_real.py`):

| Kódování   | oříznuto na hranici (Sphere/Rosenbrock) | \|x\| < 10⁻³ | oříznuto (Schwefel) | x ∈ (300; 500) (Schwefel) |
|------------|------|------|------|-------|
| IEEE 754   | 49 % | 46 % | 46 % | 0,2 % |
| Fixed-Point| 99 % | 0 %  | 51 % | 9,7 % |
| BCD        | 99 % | 0 %  | 38 % | 15,5 % |

**Sphere.** Vyhrává IEEE 754 (~10⁻⁷⁸), ale kvůli vlastnosti kódování, ne kvůli
kvalitě GA: v float32 vede ~46 % náhodných genů na hodnotu téměř nulovou
(malý exponent) a každý bit exponentu mění řád hodnoty, takže GA „zmenšuje“
čísla po řádech. Optimum Sphere leží v 0, kam kódování samo „přitahuje“. U
Fixed-Pointu a BCD je to opačně: 10 bitů celé části (±1023), resp. 3 celé
číslice, jsou pro rozsah ±5,12 příliš široké, takže 99 % náhodných genů se
ořízne na hranici a počáteční populace je prakticky celá v rozích. Ke správné
hodnotě je pak nutné trefit horní bity (znaménko, celá část), které mění
hodnotu skokem; mutace těchto bitů a jednobodové křížení dobré řešení často
rozbíjí (Hammingovy útesy). Výsledek (~10²) tedy z velké části odráží špatné
přizpůsobení pevného formátu doméně, ne nutně slabost kódování obecně. Real GA
pracuje přímo s hodnotami, Gaussova mutace dělá malé kroky, a proto s malou
sigma dobře konverguje (6·10⁻⁴, při sigma 0,01 až 4·10⁻⁵); uniformní mutace
konverguje hůř (0,025), protože nová hodnota z celého rozsahu jen vzácně
zlepší téměř optimálního jedince.

**Rosenbrock.** Úzké zakřivené údolí s optimem v (1,…,1); geny jsou provázané.
Fixed-Point a BCD dopadly velmi špatně (10⁵) ze stejného důvodu jako u Sphere:
geny jsou většinou na hranici rozsahu [-5, 10], kde má Rosenbrock obrovské
hodnoty. IEEE 754 se za ~300 ohodnocení dostane na ≈ 9 a pak se přestane
zlepšovat (std 0,07). Hodnota odpovídá f(0,…,0) = 9, tedy řešení s geny téměř v nule
(přímo jsem to neověřoval); k dalšímu zlepšení by bylo třeba přesně doladit
souřadnice k 1, což kódování při jednobodovém křížení a bit-flip mutaci
nenajde. Real GA je pomalejší,
ale některým během se do údolí dostane (nejlepší 4,3, medián 8,1 u Gauss), jiným
ne (průměr 33 vs. 49 u Uniform, velký rozptyl). Podle průměru je tak nejlepší
IEEE 754 (8,9), podle mediánu Real-Gauss (8,1) — rozdíl je malý.

**Schwefel.** Silně multimodální funkce, optimum je blízko hranice rozsahu
(≈ 421). Real-Uniform vyhrává výrazně (≈ 15): náhodný skok na libovolnou hodnotu
je nezávislé zkoušení každé souřadnice a protože je Schwefel separabilní,
křížení skládá dobré souřadnice z různých jedinců. Gaussova mutace zůstává
v okolí rodiče a uvízne v lokálním minimu (386). IEEE 754 je nejhorší (2274):
téměř nikdy netrefí oblast 300–500 (0,2 % genů), protože hodnoty jsou buď
≈ 0, nebo oříznuté na hranici. Fixed-Point a BCD mají v této oblasti 10–15 %
náhodných genů a hranici ořezávají méně, proto jsou lepší (967 a 545).

**Celkově.** Neexistuje univerzálně nejlepší varianta (no free lunch): výkon
závisí na shodě kódování/operátorů se strukturou funkce. Real-valued GA je
spolehlivý napříč funkcemi (na Schwefelu nejlepší Uniform, na Rosenbrocku
druhý resp. v mediánu první Gauss, na Sphere s velkým náskokem před
Fixed-Point a BCD), ale na Sphere ho výrazně poráží IEEE 754. Binární kódování
trpí nerovnoměrným významem bitů a Hammingovými útesy; IEEE 754 je silně
zkreslené k nule, což pomáhá u funkcí s optimem v nule a škodí u Schwefela.
Fixed-Point a BCD by v lépe přizpůsobeném formátu (méně bitů celé části pro
malé rozsahy) pravděpodobně dopadly lépe; to jsem neověřoval. Výsledky jsou
z 10 běhů na jednom nastavení parametrů, takže rozdíly mezi blízkými variantami
nejsou statisticky podložené.
