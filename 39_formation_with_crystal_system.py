import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

full_csv = os.path.join(data_dir, "mp_full_features_with_crystal.csv")
df = pd.read_csv(full_csv)

target = "formation_energy_per_atom"

# One-hot encode crystal_system
if "crystal_system" in df.columns:
    df = pd.get_dummies(df, columns=["crystal_system"], dummy_na=False)

# Keep numeric columns only
numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
feature_cols = [c for c in numeric_df.columns if c != target]

X = numeric_df[feature_cols].fillna(0.0)
y = numeric_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_dist = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

base_model = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring="r2",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

print("Tuning XGB with crystal_system features...")
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best params:", search.best_params_)

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test R2  : {r2:.3f}")
print(f"Test MAE : {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

params_out = os.path.join(data_dir, "formation_xgb_with_crystal_best_params.json")
with open(params_out, "w") as f:
    json.dump(search.best_params_, f, indent=2)
print("Saved best params to:", params_out)

pred_out = os.path.join(data_dir, "formation_with_crystal_test_predictions.csv")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(pred_out, index=False)
print("Saved test predictions to:", pred_out)

plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, s=10, alpha=0.6, c="tab:blue")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
plt.xlabel("True formation_energy_per_atom (eV/atom)")
plt.ylabel("Predicted formation_energy_per_atom (eV/atom)")
plt.tight_layout()
parity_fig = os.path.join(fig_dir, "formation_with_crystal_parity.png")
plt.savefig(parity_fig, dpi=1200)
plt.close()
print("Saved:", parity_fig)
