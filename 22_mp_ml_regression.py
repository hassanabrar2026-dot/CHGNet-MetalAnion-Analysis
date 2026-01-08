import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
print("Reading:", mp_csv)
df = pd.read_csv(mp_csv)

target_col = "energy_per_atom"
if target_col not in df.columns:
    raise SystemExit(f"{target_col} not in mp_dataset.csv")

# Feature columns: all frac_* plus basic structural descriptors
frac_cols = [c for c in df.columns if c.startswith("frac_")]
struct_cols = ["density", "volume", "band_gap"]
feature_cols = frac_cols + struct_cols

X = df[feature_cols].fillna(0.0)
y = df[target_col]

# Save feature matrix for reference
features_out = os.path.join(data_dir, "mp_ml_features.csv")
X.assign(target=y).to_csv(features_out, index=False)
print("Saved feature matrix to:", features_out)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 on test set: {r2:.3f}")
print(f"MAE on test set: {mae:.4f} (eV/atom)")

# Save predictions
pred_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
})
pred_out = os.path.join(data_dir, "mp_energy_predictions.csv")
pred_df.to_csv(pred_out, index=False)
print("Saved predictions to:", pred_out)

# Parity plot
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, s=10, alpha=0.6)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
plt.xlabel("True energy_per_atom (eV)")
plt.ylabel("Predicted energy_per_atom (eV)")
plt.tight_layout()
parity_fig = os.path.join(fig_dir, "energy_parity_plot.png")
plt.savefig(parity_fig, dpi=300)
plt.close()
print("Saved:", parity_fig)
