import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

full_csv = os.path.join(data_dir, "mp_full_features.csv")
print("Reading:", full_csv)
df = pd.read_csv(full_csv)

targets = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "density",
    "volume",
]

numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
feature_cols = [c for c in numeric_df.columns if c not in targets]
X_all = numeric_df[feature_cols].fillna(0.0)


def get_model():
    if HAS_XGB:
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        return RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        )

for target in targets:
    if target not in df.columns:
        print(f"Target {target} not in dataframe, skipping.")
        continue

    print(f"\n=== Feature importance for {target} ===")
    y_all = df[target]

    # just to be safe, keep a hold-out, but train importance model on full data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    model = get_model()
    print("Training model...")
    model.fit(X_all, y_all)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # fallback: use variance of predictions when perturbing features (not ideal)
        importances = np.zeros(len(feature_cols))

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    out_csv = os.path.join(data_dir, f"mp_feature_importance_{target}.csv")
    fi_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # Plot top 20
    top_n = fi_df.head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top_n["feature"][::-1], top_n["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top 20 features for {target}")
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"feature_importance_{target}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved:", fig_path)
