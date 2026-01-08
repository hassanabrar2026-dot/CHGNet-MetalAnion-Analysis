import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

full_csv = os.path.join(data_dir, "mp_full_features.csv")
params_json = os.path.join(data_dir, "formation_xgb_best_params.json")

df = pd.read_csv(full_csv)
target = "formation_energy_per_atom"

# numeric-only
numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
feature_cols = [c for c in numeric_df.columns if c != target]

X = numeric_df[feature_cols].fillna(0.0)
y = numeric_df[target]

with open(params_json, "r") as f:
    best_params = json.load(f)

model = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    **best_params,
)

print("Training XGB model with best params for SHAP...")
model.fit(X, y)

# Use a subset for SHAP if dataset is large
max_samples = min(1000, X.shape[0])
X_sample = X.sample(n=max_samples, random_state=42)

explainer = shap.TreeExplainer(model)
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_sample)

# SHAP bar plot (global importance)
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)
bar_fig = os.path.join(fig_dir, "formation_shap_bar.png")
plt.tight_layout()
plt.savefig(bar_fig, dpi=1200, bbox_inches="tight")
plt.close()
print("Saved:", bar_fig)

# SHAP beeswarm plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)
bees_fig = os.path.join(fig_dir, "formation_shap_beeswarm.png")
plt.tight_layout()
plt.savefig(bees_fig, dpi=1200, bbox_inches="tight")
plt.close()
print("Saved:", bees_fig)
