<a id="zh"></a>

**中文** | [English](#english)

# 以集成學習預測建築能源效率：模型比較與關鍵參數分析

本專案為一篇研討會論文的資料處理、模型訓練與圖表產生程式，主題是以集成學習預測建築的暖氣／冷氣負荷。

## 一、論文與研討會資料

| 項目 | 內容 |
|---|---|
| 論文題目 | Predicting Building Energy Efficiency using Ensemble Learning: Model Comparison and Key Parameter Analysis（以集成學習預測建築能源效率：模型比較與關鍵參數分析，論文以英文撰寫） |
| 作者 | 周理陽（Li Yang Chou） |
| 單位 | 國立中央大學機械工程學系 |
| E-mail | 112303573@cc.ncu.edu.tw |
| 關鍵字 | 暖冷氣負荷、特徵重要性、機器學習、集成學習、LightGBM |

| 研討會 | 內容 |
|---|---|
| 全名 | 2025 International Conference on AI for a Sustainable Society — Frontier Research, Policy, and Practice（2025 永續社會人工智慧國際研討會：前沿研究、政策與實務） |
| 日期 | 2025 年 11 月 13 日至 14 日 |
| 地點 | 中央研究院（臺北市） |
| 合辦單位 | 中央研究院永續科學中心、國立中央大學管理學院及其他合作單位 |
| 發表證明 | 大會核發之 Certificate of Presentation（發表證書），受證人 Li Yang Chou，由沈建文教授簽署（Future Earth Taipei 數位時代工作組；國立中央大學企業管理學系） |

### 論文摘要（中文為譯文，原文見下方英文部分）

提升建築能源效率對於降低環境衝擊與支持永續發展至關重要。暖氣與冷氣負荷主導建築能耗，並受牆面積、整體高度、玻璃配置與相對緊湊度等建築與物理參數強烈影響。線性相關分析能提供初步理解，卻無法完整捕捉特徵間的非線性交互作用。為填補此缺口，本研究結合傳統統計方法與先進機器學習模型，以提升預測表現與可解釋性。

以「Energy Efficiency」資料集為對象，研究依序進行資料前處理、相關性分析、基準模型建立，並評估四種集成演算法——隨機森林（Random Forest）、極限隨機樹（Extra Trees）、梯度提升（Gradient Boosting）與 LightGBM。結果顯示所有集成模型皆明顯優於線性迴歸，尤其在預測暖氣負荷（Y1）時，LightGBM 達到最高準確度。特徵重要性分析發現，相對緊湊度（X1）是最關鍵的預測變數，超越了線性相關性很高的牆面積（X3）與整體高度（X5）。此發現凸顯非線性模型能辨識線性方法所忽略的深層模式。

本研究證明機器學習能提供可靠的建築能源表現分析工具，並為節能建築設計提供實務洞見。

## 二、資料

UCI Machine Learning Repository 的 **Energy Efficiency** 資料集（id=242）：768 筆以 Ecotect 模擬的建築形體樣本，8 個輸入特徵（X1–X8）與 2 個目標（暖氣負荷 Y1、冷氣負荷 Y2）。

欄位定義與引用方式見 [`data/README_data.md`](data/README_data.md)。

## 三、方法

所有模型共用相同前處理（`src/data_loader.py`）：特徵以 `StandardScaler` 標準化（僅以訓練集配適），並以 80/20 切分訓練與測試集（`random_state=42`）。

- **基準模型**：線性迴歸，每個目標各一個模型。
- **集成模型**（`src/models.py`）：隨機森林、梯度提升、Extra Trees、LightGBM，分別對 Y1、Y2 訓練，以 MSE、RMSE、R² 評估，並以 RMSE 相對線性迴歸的改善百分比比較。
- **超參數搜尋**（`src/optuna_tuning.py`）：以 Optuna（TPE，100 次試驗）為每個目標搜尋隨機森林的 `n_estimators`、`max_depth`、`min_samples_split`、`min_samples_leaf`、`max_features`。
- **特徵重要性**（`src/feature_importance.py`）：以測試 RMSE 最佳的模型重新訓練並取出原生 `feature_importances_`。
- **圖表**（`src/make_figures.py`）：相關性熱圖、RMSE 改善長條圖、實際 vs. 預測散佈圖、特徵重要性長條圖。

## 四、執行結果

測試集表現（`results/model_comparison.csv`，各目標內依 RMSE 排序）：

| 模型 | 目標 | RMSE | R² | 相對線性迴歸的 RMSE 改善 |
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

LightGBM 在兩個目標上都是最佳模型。特徵重要性（`results/feature_importance.csv`）印證論文的核心發現：**相對緊湊度（X1）** 在 Y1 與 Y2 都居首，領先牆面積（X3）與整體高度（X5），儘管後兩者在熱圖中與目標的線性相關更強。

Optuna 對隨機森林做 100 次試驗的搜尋（`results/rf_optuna_best_params.csv`）收斂到高度正則化、反而比預設值差的樹（兩個目標 R² 約 0.90），原因是搜尋空間允許 `min_samples_split`／`min_samples_leaf` 高達訓練集的 0.5。此結果作為已記錄的負面結果保留，因為這是該搜尋空間本身的特性，並非程式錯誤。

圖表：`figures/fig1_correlation_heatmap.png`、`fig2_rmse_improvement.png`、`fig3_actual_vs_predicted_{y1,y2}.png`、`fig4_feature_importance_{y1,y2}.png`。

## 五、檔案結構

```
predicting-building-energy-efficiency/
├─ data/
│  ├─ energy_efficiency.csv     UCI 資料集快取（由 data_loader.py 重新產生）
│  └─ README_data.md            欄位定義、來源、授權
├─ src/
│  ├─ data_loader.py            下載／快取資料、標準化與切分
│  ├─ models.py                 訓練並評估 5 個模型
│  ├─ optuna_tuning.py          Optuna 隨機森林超參數搜尋
│  ├─ feature_importance.py     各目標最佳模型的特徵重要性
│  └─ make_figures.py           所有圖表
├─ results/                     model_comparison.csv、rf_optuna_best_params.csv、feature_importance.csv
├─ figures/                     fig1–fig4
├─ references/                  參考文獻（見 references/README.md）
└─ requirements.txt
```

## 六、如何執行

```bash
pip install -r requirements.txt

python src/data_loader.py        # 下載並快取資料集
python src/models.py             # 訓練與評估 5 個模型，輸出 results/model_comparison.csv
python src/optuna_tuning.py --trials 100   # 隨機森林超參數搜尋
python src/feature_importance.py # 各目標最佳模型的特徵重要性
python src/make_figures.py       # 產生所有圖表到 figures/
```

## 七、參考文獻

完整書目與授權說明見 [`references/README.md`](references/README.md)。第三方論文全文不隨本 repo 散布。

## 八、授權

本專案自行撰寫的程式碼（`src/`）與文件以 MIT License 釋出，詳見 [`LICENSE`](LICENSE)。

以下內容不在本授權範圍內，各自沿用原本的條款：

- **資料集**（`data/`）：UCI Machine Learning Repository 的 Energy Efficiency 資料集，CC BY 4.0，引用 Tsanas & Xifara (2012)，見 `data/README_data.md`。
- **參考文獻**（`references/`）：著作權歸各作者與出版方所有，見 `references/README.md`。

## 九、AI 使用揭露

所有研究設計、方法與結論皆由本人獨立主導。AI 工具作為輔助，用於英文文法潤飾、對本人撰寫之程式進行除錯與重構、將實驗筆記本整理為可執行腳本，以及撰寫與翻譯 repo 文件。本人已逐行驗證所有代碼、結果與文稿，對研究真實性負完全責任。

---

<a id="english"></a>

[中文](#zh) | **English**

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

## 9. AI Use Disclosure

All research design, methods, and conclusions were led and completed independently by the author. AI tools were used as an aid for English grammar polishing, debugging and refactoring of code written by the author, organizing experiment notebooks into runnable scripts, and drafting and translating the documentation in this repository. The author has verified all code, results, and manuscripts line by line and takes full responsibility for the authenticity of the research. (English translation of the Chinese text above.)
