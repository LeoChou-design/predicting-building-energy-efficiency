"""Fetch the UCI Energy Efficiency dataset, cache it locally, and produce a
scaled train/test split shared by every model in this project."""

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "energy_efficiency.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw():
    """Return (X, y) as pandas DataFrames, downloading from UCI on first run
    and caching to data/energy_efficiency.csv afterwards."""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        feature_cols = [c for c in df.columns if c not in ("Y1", "Y2")]
        return df[feature_cols], df[["Y1", "Y2"]]

    import ucimlrepo

    energy_efficiency = ucimlrepo.fetch_ucirepo(id=242)
    X = energy_efficiency.data.features
    y = energy_efficiency.data.targets

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.concat([X, y], axis=1).to_csv(CSV_PATH, index=False)
    return X, y


def load_split():
    """Return X_train, X_test, y_train, y_test with features standardized
    (scaler is fit on the training split only)."""
    X, y = load_raw()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test_raw), columns=X.columns, index=X_test_raw.index
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X, y = load_raw()
    print(f"Features: {X.shape}, Targets: {y.shape}")
    print(X.describe())
