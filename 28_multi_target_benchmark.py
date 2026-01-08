import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Optional models
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

# Load full feature dataset
df = pd.read_csv(os.path.join(data_dir, "mp_full_features.csv"))

# Target properties to model
targets = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "density",
    "volume",
]

# Feature columns (everything except ID, formula, and targets)
# Drop non-numeric columns automatically
numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])

# Remove target columns from numeric features
feature_cols = [c for c in numeric_df.columns if c not in targets]
X = numeric_df[feature_cols].fillna(0.0)

X = df[feature_cols].fillna(0.0)

# Define ML models
def get_models():
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            random_state=42,
        )

    return models


# Loop over each target property
for target in targets:
    print(f"\n==============================")
    print(f"Training models for: {target}")
    print(f"==============================")

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "model": name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        })

    # Save CSV
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(data_dir, f"mp_model_scores_{target}.csv")
    results_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Bar plot for R²
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    plt.bar(x, results_df["R2"], color="C0")
    plt.xticks(x, results_df["model"], rotation=45)
    plt.ylabel("R²")
    plt.title(f"Model Comparison for {target}")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"model_scores_{target}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
