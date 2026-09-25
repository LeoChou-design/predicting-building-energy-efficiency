"""Train Linear Regression (baseline) and four ensemble regressors
(Random Forest, Gradient Boosting, Extra Trees, LightGBM) on both targets
(Y1 = heating load, Y2 = cooling load), then write a comparison table to
results/model_comparison.csv."""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import RANDOM_STATE, load_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Extra Trees": lambda: ExtraTreesRegressor(random_state=RANDOM_STATE),
    "LightGBM": lambda: lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
}


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Fit every model in MODEL_FACTORIES on both targets and return a long
    DataFrame with one row per (model, target) plus the fitted estimators."""
    rows = []
    fitted = {}
    for target in ("Y1", "Y2"):
        for name, factory in MODEL_FACTORIES.items():
            model = factory()
            model.fit(X_train, y_train[target])
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test[target], y_pred)
            rows.append(
                {
                    "Model": name,
                    "Target": target,
                    "MSE": mse,
                    "RMSE": np.sqrt(mse),
                    "R2": r2_score(y_test[target], y_pred),
                }
            )
            fitted[(name, target)] = model
    return pd.DataFrame(rows), fitted


def add_rmse_improvement(df):
    """Add an RMSE_Improvement_pct column relative to Linear Regression,
    per target."""
    df = df.copy()
    baseline = df[df["Model"] == "Linear Regression"].set_index("Target")["RMSE"]
    df["RMSE_Improvement_pct"] = df.apply(
        lambda r: 100 * (baseline[r["Target"]] - r["RMSE"]) / baseline[r["Target"]],
        axis=1,
    )
    return df


def main():
    X_train, X_test, y_train, y_test = load_split()
    comparison, fitted = train_and_evaluate(X_train, X_test, y_train, y_test)
    comparison = add_rmse_improvement(comparison)
    comparison = comparison.sort_values(["Target", "RMSE"]).reset_index(drop=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(comparison.to_string(index=False))

    for target in ("Y1", "Y2"):
        sub = comparison[comparison["Target"] == target]
        best = sub.loc[sub["RMSE"].idxmin()]
        print(f"\nBest model for {target}: {best['Model']} (RMSE={best['RMSE']:.3f}, R2={best['R2']:.3f})")

    return comparison, fitted, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
