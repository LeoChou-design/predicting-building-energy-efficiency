# Data: UCI "Energy Efficiency" Dataset

768 samples simulating the heating and cooling load of 12 building shapes under
different glazing/orientation configurations, generated with Ecotect.

| File | Content | Size |
|---|---|---|
| `energy_efficiency.csv` | Features (X1–X8) + targets (Y1, Y2), cached from the UCI repository | 768 rows × 10 columns |

Run `src/data_loader.py` to (re-)download the dataset via `ucimlrepo` and
regenerate this CSV.

## Columns

| Column | Meaning |
|---|---|
| X1 | Relative Compactness |
| X2 | Surface Area |
| X3 | Wall Area |
| X4 | Roof Area |
| X5 | Overall Height |
| X6 | Orientation |
| X7 | Glazing Area |
| X8 | Glazing Area Distribution |
| Y1 | Heating Load (target) |
| Y2 | Cooling Load (target) |

## Source

- Dataset page: https://archive.ics.uci.edu/dataset/242/energy+efficiency
- Citation: Tsanas, A. & Xifara, A. (2012). Energy Efficiency [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C51307
- Original study: Tsanas, A., & Xifara, A. (2012). Accurate quantitative
  estimation of energy performance of residential buildings using statistical
  machine learning tools. *Energy and Buildings*, 49, 560–567.
- License: CC BY 4.0 (UCI Machine Learning Repository standard license) —
  reuse and redistribution permitted with attribution.
