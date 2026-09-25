"""Retrain the best-performing model for each target (per results/model_comparison.csv)
and extract its feature importances."""

import os

import pandas as pd

from data_loader import load_split
from models import MODEL_FACTORIES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def best_model_per_target(comparison):
    best = {}
    for target in ("Y1", "Y2"):
        sub = comparison[comparison["Target"] == target]
        best[target] = sub.loc[sub["RMSE"].idxmin(), "Model"]
    return best


def main():
    comparison = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    best = best_model_per_target(comparison)

    X_train, X_test, y_train, y_test = load_split()

    rows = []
    for target, model_name in best.items():
        model = MODEL_FACTORIES[model_name]()
        model.fit(X_train, y_train[target])
        if not hasattr(model, "feature_importances_"):
            print(f"{model_name} has no feature_importances_, skipping {target}")
            continue
        for feature, importance in zip(X_train.columns, model.feature_importances_):
            rows.append(
                {
                    "Target": target,
                    "Model": model_name,
                    "Feature": feature,
                    "Importance": importance,
                }
            )

    df = pd.DataFrame(rows).sort_values(["Target", "Importance"], ascending=[True, False])
    df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
