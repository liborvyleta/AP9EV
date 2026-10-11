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
- **FixedPoint-tuned / BCD-tuned**: stejná kódování, ale dělení bitů (číslic)
  na celou a desetinnou část přizpůsobené rozsahu úlohy: Fixed-Point má
  1 + ⌈log₂(max|mez| + 1)⌉ bitů pro znaménko a celou část (Sphere 3, Rosenbrock 4,
  Schwefel 9 bitů celé části), BCD tolik celých číslic, kolik má max|mez|
  (1, 2, 3). Původní varianty (10 bitů / 3 číslice celé části pro všechny
  úlohy) zůstávají jako základní srovnání; všech 7 variant má 32 bitů na dimenzi.
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
| Sphere | Binary-FixedPoint-tuned | 3.17e-08 | 5.04e-07 | 6.68e-07 |
| Sphere | Binary-BCD-tuned | 4.18e-06 | 0.472 | 1.3 |
| Sphere | Real-Gauss | 0.000169 | 0.000635 | 0.000386 |
| Sphere | Real-Uniform | 0.00949 | 0.025 | 0.0179 |
| Rosenbrock | Binary-IEEE754 | 8.76 | 8.94 | 0.0664 |
| Rosenbrock | Binary-FixedPoint | 3.23e+05 | 5.38e+05 | 1.16e+05 |
| Rosenbrock | Binary-BCD | 2.66e+05 | 3.43e+05 | 6.81e+04 |
| Rosenbrock | Binary-FixedPoint-tuned | 5.3 | 63.3 | 53.7 |
| Rosenbrock | Binary-BCD-tuned | 1.5e+03 | 5.12e+04 | 3.5e+04 |
| Rosenbrock | Real-Gauss | 4.34 | 33.4 | 71.7 |
| Rosenbrock | Real-Uniform | 12.6 | 48.5 | 30.8 |
| Schwefel | Binary-IEEE754 | 2.03e+03 | 2.27e+03 | 139 |
| Schwefel | Binary-FixedPoint | 373 | 967 | 376 |
| Schwefel | Binary-BCD | 186 | 544 | 236 |
| Schwefel | Binary-FixedPoint-tuned | 212 | 524 | 211 |
| Schwefel | Binary-BCD-tuned | 186 | 544 | 236 |
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
z náhodných bitových řetězců (20 000 × 10 genů, `decode_*` z `ga_real.py`):

| Kódování         | oříznuto na hranici: Sphere / Rosenbrock / Schwefel | \|x\| < 10⁻³ | x ∈ (300; 500) (Schwefel) |
|------------------|------------------------|------|-------|
| IEEE 754         | 49 % / 49 % / 46 %     | 46 % | 0,2 % |
| Fixed-Point      | 99 % / 99 % / 51 %     | 0 %  | 9,7 % |
| BCD              | 99 % / 99 % / 38 %     | 0 %  | 15,5 % |
| Fixed-Point-tuned| 36 % / 53 % / 2 %      | 0 %  | 19,5 % |
| BCD-tuned        | 36 % / 89 % / 38 %     | 0 %  | 15,5 % |

(BCD-tuned pro Schwefel je shodné s BCD, protože 3 celé číslice jsou tam už
původně správně.)

**Ověření IEEE 754 na Rosenbrocku.** U všech 10 běhů jsem vypsal nejlepší
nalezené řešení: všechny geny jsou v absolutní hodnotě ≤ 0,13 (typicky
10⁻² až 10⁻³⁸), tedy f ≈ f(0,…,0) = 9. GA se za ~300 ohodnocení dostane do
okolí nuly (kam kódování samo „táhne“) a odtud se nedokáže dostat k optimu
(1,…,1): k tomu by bylo potřeba přesně trefit hodnotu 1 (v float32 je to
konkrétní exponent + nulová mantisa) ve všech 10 dimenzích najednou.

**Vliv přizpůsobení formátu rozsahu (Fixed-Point, BCD).** Původní formát
(10 bitů celé části, resp. 3 celé číslice) je pro rozsahy ±5,12 a [-5, 10]
příliš široký: 99 % náhodných genů se ořízne na hranici, počáteční populace
je prakticky v rozích a horní bity (znaménko, celá část) mění hodnotu skokem,
takže mutace a jednobodové křížení dobrá řešení rozbíjejí (Hammingovy útesy).
Po přizpůsobení formátu se výsledky dramaticky zlepšily:
Fixed-Point na Sphere ze 163 na 5·10⁻⁷, na Rosenbrocku z 5·10⁵ na 63, na
Schwefelu z 967 na 524. BCD-tuned zlepšil Sphere (97 → 0,47) i Rosenbrock
(3·10⁵ → 5·10⁴); Schwefel je beze změny (formát byl už správný). Zlepšení
je tedy z velké části zásluhou volby formátu, ne kódování samotného.
BCD-tuned zůstává na Rosenbrocku slabé, protože pro rozsah do 10 potřebuje
2 celé číslice a desítková číslice je smysluplná jen pro 0 a 1 — 89 % genů
je oříznuto na horní mez 10. BCD navíc plýtvá kódovým prostorem (nibble > 9 je
neplatný) a sousední hodnoty (např. 0,9999 → 1,0000) se liší ve všech bitech.

**Sphere.** Vyhrává IEEE 754 (~10⁻⁷⁸), ale kvůli vlastnosti kódování, ne
kvůli kvalitě GA: v float32 vede ~46 % náhodných genů na téměř nulovou hodnotu
(malý exponent) a každý bit exponentu mění řád hodnoty, takže GA „zmenšuje“
čísla po řádech. Optimum Sphere leží v 0, kam kódování samo „přitahuje“.
Fixed-Point-tuned (5·10⁻⁷) je překvapivě lepší než Real-Gauss (6·10⁻⁴):
Gaussova mutace s pevným sigma = 5 % rozsahu nemůže zpřesňovat pod tuto
úroveň, kdežto u fixed-pointu mutace nízkých bitů jemně dolaďuje hodnotu a
mutace různých bitů dává kroky všech velikostí najednou (od 2⁻²⁸ po jednotky).
Při sigma 0,01 dosahuje Real-Gauss 4·10⁻⁵, stále hůř než Fixed-Point-tuned.
Uniformní mutace konverguje hůř (0,025), protože nová hodnota z celého rozsahu
jen vzácně zlepší téměř optimálního jedince.

**Rosenbrock.** Úzké zakřivené údolí s optimem v (1,…,1); geny jsou provázané.
IEEE 754 uvízne na ≈ 9 (viz výše; std 0,07). Real GA je pomalejší, ale
některým během se do údolí dostane (nejlepší 4,3, medián 8,1 u Gauss), jiným
ne (průměr 33 vs. 49 u Uniform, velký rozptyl). Podle průměru je tak
nejlepší IEEE 754 (8,9), podle mediánu Real-Gauss (8,1) — rozdíl je malý.
Fixed-Point-tuned (průměr 63) je srovnatelný s real-valued variantami.

**Schwefel.** Silně multimodální funkce, optimum je blízko hranice rozsahu
(≈ 421). Real-Uniform vyhrává výrazně (≈ 15): náhodný skok na libovolnou hodnotu
je nezávislé zkoušení každé souřadnice a protože je Schwefel separabilní,
křížení skládá dobré souřadnice z různých jedinců. Gaussova mutace zůstává
v okolí rodiče a uvízne v lokálním minimu (386). IEEE 754 je nejhorší (2274):
téměř nikdy netrefí oblast 300–500 (0,2 % genů), protože hodnoty jsou buď
≈ 0, nebo oříznuté na hranici. Fixed-Point (524–967) a BCD (545) mají v této
oblasti 10–20 % náhodných genů, proto jsou lepší, ale hůř než Real-Uniform:
změna horních bitů je skok a nízké bity doladění neřeší problém lokálních minim.

**Celkově.** Neexistuje univerzálně nejlepší varianta (no free lunch): výkon
závisí na shodě kódování/operátorů se strukturou funkce. Real-valued GA je
spolehlivý napříč funkcemi (nejlepší Uniform na Schwefelu, Gauss na Rosenbrocku
v mediánu), ale na Sphere ho poráží IEEE 754 i správně nastavený Fixed-Point.
Binární kódování je silně závislé na návrhu formátu: špatně zvolené dělení
bitů zhorší výsledek o několik řádů, IEEE 754 je zkreslené k nule (pomáhá u
funkcí s optimem v nule, škodí u Schwefela). Výsledky jsou z 10 běhů na
jednom nastavení parametrů, takže rozdíly mezi blízkými variantami nejsou
statisticky podložené.
