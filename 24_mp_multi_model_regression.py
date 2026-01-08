import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Try XGBoost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
df = pd.read_csv(mp_csv)

target_col = "energy_per_atom"

# Feature columns
frac_cols = [c for c in df.columns if c.startswith("frac_")]
struct_cols = ["density", "volume", "band_gap"]
feature_cols = frac_cols + struct_cols

X = df[feature_cols].fillna(0.0)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

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

results_df = pd.DataFrame(results)
out_csv = os.path.join(data_dir, "mp_model_scores.csv")
results_df.to_csv(out_csv, index=False)
print("Saved model scores to:", out_csv)

# ---- Bar plot ----
plt.figure(figsize=(8, 5))

x = np.arange(len(results_df))
width = 0.25

plt.bar(x - width, results_df["R2"], width, label="R²")
plt.bar(x, results_df["MAE"], width, label="MAE")
plt.bar(x + width, results_df["RMSE"], width, label="RMSE")

plt.xticks(x, results_df["model"], rotation=45)
plt.ylabel("Score")
plt.title("Model Comparison on MP Dataset")
plt.legend()
plt.tight_layout()

fig_path = os.path.join(fig_dir, "model_scores_barplot.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print("Saved:", fig_path)
