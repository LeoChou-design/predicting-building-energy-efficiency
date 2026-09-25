# Predicting Building Energy Efficiency using Ensemble Learning: Model Comparison and Key Parameter Analysis

Data processing, model training, and figure-generation code for a conference paper on predicting building heating/cooling loads with ensemble learning.

## 1. Paper & Conference

| Item | Detail |
|---|---|
| Title | Predicting Building Energy Efficiency using Ensemble Learning: Model Comparison and Key Parameter Analysis |
| Author | Li Yang Chou (周理陽) |
| Affiliation | Department of Mechanical Engineering, National Central University, Taiwan |
| E-mail | 112303573@cc.ncu.edu.tw |
| Keywords | Heating and cooling loads, Feature importance, Machine learning, Ensemble learning, LightGBM |

| Conference | Detail |
|---|---|
| Full name | 2025 International Conference on AI for a Sustainable Society — Frontier Research, Policy, and Practice |
| Date | 13–14 November 2025 |
| Venue | Academia Sinica, Taipei, Taiwan |
| Co-organizers | Center for Sustainability Science (Academia Sinica), College of Management (National Central University), and partner institutes |
| Proof of presentation | Certificate of Presentation issued to Li Yang Chou, signed by Prof. Chien-wen Shen (Digital Age W.G., Future Earth Taipei; Dept. of Business Administration, NCU) |

### Abstract

Improving building energy efficiency is essential for reducing environmental impact and supporting sustainable development. Heating and cooling loads, which dominate building energy consumption, are strongly influenced by architectural and physical parameters such as wall area, overall height, glazing distribution, and relative compactness. While linear correlation analysis provides an initial understanding of these relationships, it cannot fully capture the nonlinear interactions among features. To address this gap, this study integrates traditional statistical methods with advanced machine learning models to enhance predictive performance and interpretability.

Using the "Energy Efficiency" dataset, the research followed a structured process: data preprocessing, correlation analysis, baseline modeling, and evaluation of four ensemble algorithms — Random Forest, Extra Trees, Gradient Boosting, and LightGBM. Results indicate that all ensemble models significantly outperform linear regression, particularly in predicting heating load (Y1), where LightGBM achieved the highest accuracy. Feature importance analysis revealed that relative compactness (X1) is the most critical predictor, surpassing variables with high linear correlation, such as wall area (X3) and overall height (X5). This finding highlights the capability of nonlinear models to identify deeper patterns overlooked by linear approaches.

The study demonstrates that machine learning provides reliable tools for analyzing building energy performance and offers practical insights for energy-efficient building design.

## 2. Data

UCI Machine Learning Repository — **Energy Efficiency** dataset (id=242): 768 samples of simulated building shapes (Ecotect), 8 input features (X1–X8) and 2 targets (heating load Y1, cooling load Y2).

Details, column definitions, and citation: [`data/README_data.md`](data/README_data.md).

## 3. Method

All models share the same preprocessing (`src/data_loader.py`): features are standardized with `StandardScaler` fit on the training split only, and an 80/20 train/test split is used (`random_state=42`).

- **Baseline**: Linear Regression, one model per target.
- **Ensemble models** (`src/models.py`): Random Forest, Gradient Boosting, Extra Trees, LightGBM — each trained separately for Y1 and Y2, evaluated with MSE, RMSE, and R², and compared against the Linear Regression baseline via RMSE improvement (%).
- **Hyperparameter search** (`src/optuna_tuning.py`): Optuna (TPE sampler, 100 trials) searches `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` for a Random Forest on each target.
- **Feature importance** (`src/feature_importance.py`): the best model per target (by test RMSE) is retrained and its native `feature_importances_` are extracted.
- **Figures** (`src/make_figures.py`): correlation heatmap, RMSE-improvement bar chart, actual-vs-predicted scatter plots, and feature-importance bar charts.

## 4. Results

Test-set performance (`results/model_comparison.csv`), sorted by RMSE within each target:

| Model | Target | RMSE | R² | RMSE Improvement vs. LR |
|---|---|---|---|---|
| LightGBM | Y1 | 0.479 | 0.998 | 84.2% |
| Random Forest | Y1 | 0.497 | 0.998 | 83.6% |
| Extra Trees | Y1 | 0.506 | 0.998 | 83.3% |
| Gradient Boosting | Y1 | 0.515 | 0.997 | 83.0% |
| Linear Regression | Y1 | 3.025 | 0.912 | — |
| LightGBM | Y2 | 1.124 | 0.986 | 64.3% |
| Gradient Boosting | Y2 | 1.514 | 0.975 | 51.9% |
| Extra Trees | Y2 | 1.721 | 0.968 | 45.3% |
| Random Forest | Y2 | 1.726 | 0.968 | 45.1% |
| Linear Regression | Y2 | 3.145 | 0.893 | — |

LightGBM is the best model for both targets. Feature importance (`results/feature_importance.csv`) confirms the paper's key finding: **relative compactness (X1)** dominates for both Y1 and Y2, ahead of wall area (X3) and overall height (X5) despite the latter two showing stronger *linear* correlation with the targets in the heatmap.

Optuna's 100-trial Random Forest search (`results/rf_optuna_best_params.csv`) converges to a heavily regularized, worse-than-default tree (R² ≈ 0.90 for both targets) because its search space allows `min_samples_split`/`min_samples_leaf` up to 0.5 of the training set — this is kept in the repo as a documented negative result, not cleaned up, since it is a genuine property of that search space rather than a bug.

Figures: `figures/fig1_correlation_heatmap.png`, `fig2_rmse_improvement.png`, `fig3_actual_vs_predicted_{y1,y2}.png`, `fig4_feature_importance_{y1,y2}.png`.

## 5. File Structure

```
predicting-building-energy-efficiency/
├─ data/
│  ├─ energy_efficiency.csv     Cached UCI dataset (regenerated by data_loader.py)
│  └─ README_data.md            Column definitions, source, license
├─ src/
│  ├─ data_loader.py            Fetch/cache dataset, scaled train/test split
│  ├─ models.py                 Train & evaluate 5 models on Y1/Y2
│  ├─ optuna_tuning.py          Optuna RF hyperparameter search
│  ├─ feature_importance.py     Feature importance from the best model per target
│  └─ make_figures.py           All figures
├─ results/                     model_comparison.csv, rf_optuna_best_params.csv, feature_importance.csv
├─ figures/                     fig1–fig4
├─ references/                  Bibliography (see references/README.md)
└─ requirements.txt
```

## 6. How to Run

```bash
pip install -r requirements.txt

python src/data_loader.py        # fetch + cache the dataset
python src/models.py             # train & evaluate 5 models, write results/model_comparison.csv
python src/optuna_tuning.py --trials 100   # RF hyperparameter search
python src/feature_importance.py # feature importance of the best model per target
python src/make_figures.py       # write all figures to figures/
```

## 7. References

See [`references/README.md`](references/README.md) for the full bibliography and licensing notes. Full-text PDFs of third-party papers are not redistributed in this repository.

## 8. License

Code and documentation authored for this project (`src/`, this README) are released under the MIT License — see [`LICENSE`](LICENSE).

The following are **not** covered by that license and remain under their own terms:

- **Dataset** (`data/`): UCI Machine Learning Repository, Energy Efficiency dataset, CC BY 4.0 — cite Tsanas & Xifara (2012), see `data/README_data.md`.
- **References** (`references/`): copyright of the original authors/publishers — see `references/README.md`.
