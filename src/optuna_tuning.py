"""Optuna hyperparameter search for the Random Forest regressor, run
independently for Y1 (heating load) and Y2 (cooling load)."""

import argparse
import os

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import RANDOM_STATE, load_split

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_objective(X_train, X_test, y_train, y_test, target):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 32),
            min_samples_split=trial.suggest_float("min_samples_split", 0.1, 1.0, log=True),
            min_samples_leaf=trial.suggest_float("min_samples_leaf", 0.1, 0.5, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        )
        model = RandomForestRegressor(random_state=RANDOM_STATE, **params)
        model.fit(X_train, y_train[target])
        y_pred = model.predict(X_test)
        return -r2_score(y_test[target], y_pred)

    return objective


def tune_target(X_train, X_test, y_train, y_test, target, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(X_train, X_test, y_train, y_test, target), n_trials=n_trials
    )

    best_model = RandomForestRegressor(random_state=RANDOM_STATE, **study.best_trial.params)
    best_model.fit(X_train, y_train[target])
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test[target], y_pred)
    r2 = r2_score(y_test[target], y_pred)

    return {
        "Target": target,
        "Best_R2": r2,
        "Best_MSE": mse,
        **study.best_trial.params,
    }, best_model


def main(n_trials=100):
    X_train, X_test, y_train, y_test = load_split()

    rows = []
    for target in ("Y1", "Y2"):
        result, _ = tune_target(X_train, X_test, y_train, y_test, target, n_trials)
        print(f"[{target}] best params: {result}")
        rows.append(result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "rf_optuna_best_params.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    main(n_trials=args.trials)
