"""Generate the figures used in the paper from the cached data and the CSVs
in results/: correlation heatmap, RMSE improvement chart, actual-vs-predicted
scatter for the best model per target, and feature importance bar charts."""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loader import load_split
from models import MODEL_FACTORIES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

sns.set_style("whitegrid")


def fig1_correlation_heatmap(X_train, y_train):
    combined = pd.concat([X_train, y_train.loc[X_train.index]], axis=1)
    plt.figure(figsize=(11, 9))
    sns.heatmap(combined.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Features and Targets")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_correlation_heatmap.png"), dpi=150)
    plt.close()


def fig2_rmse_improvement(comparison):
    plotted = comparison[comparison["Model"] != "Linear Regression"]
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="RMSE_Improvement_pct", hue="Target", data=plotted, palette="viridis")
    plt.title("RMSE Improvement vs. Linear Regression (%)")
    plt.ylabel("RMSE Improvement (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_rmse_improvement.png"), dpi=150)
    plt.close()


def fig3_actual_vs_predicted(comparison, X_train, X_test, y_train, y_test):
    for target in ("Y1", "Y2"):
        sub = comparison[comparison["Target"] == target]
        best_name = sub.loc[sub["RMSE"].idxmin(), "Model"]
        model = MODEL_FACTORIES[best_name]()
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)

        plt.figure(figsize=(7, 6))
        sns.scatterplot(x=y_test[target], y=y_pred)
        lims = [y_test[target].min(), y_test[target].max()]
        plt.plot(lims, lims, "r--")
        plt.title(f"Actual vs. Predicted {target} — Best Model: {best_name}")
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"fig3_actual_vs_predicted_{target.lower()}.png"), dpi=150)
        plt.close()


def fig4_feature_importance(importance_df):
    for target in ("Y1", "Y2"):
        sub = importance_df[importance_df["Target"] == target]
        if sub.empty:
            continue
        model_name = sub["Model"].iloc[0]
        plt.figure(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=sub)
        plt.title(f"Feature Importance for {target} — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"fig4_feature_importance_{target.lower()}.png"), dpi=150)
        plt.close()


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    comparison = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    importance_df = pd.read_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"))

    fig1_correlation_heatmap(X_train, y_train)
    fig2_rmse_improvement(comparison)
    fig3_actual_vs_predicted(comparison, X_train, X_test, y_train, y_test)
    fig4_feature_importance(importance_df)
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
