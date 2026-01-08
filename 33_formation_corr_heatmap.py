import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

print("Training XGB model for correlation analysis...")
model.fit(X, y)

importances = model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
fi_df = fi_df.sort_values("importance", ascending=False)

top10 = fi_df.head(10)["feature"].tolist()
print("Top 10 features:", top10)

# Save list of top 10 features
top10_out = os.path.join(data_dir, "formation_top10_features.csv")
fi_df.head(10).to_csv(top10_out, index=False)
print("Saved:", top10_out)

# Correlation matrix
corr = numeric_df[top10 + [target]].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("Correlation heatmap (top 10 features + formation_energy_per_atom)")
plt.tight_layout()
heat_fig = os.path.join(fig_dir, "formation_top10_corr_heatmap.png")
plt.savefig(heat_fig, dpi=1200)
plt.close()
print("Saved:", heat_fig)
