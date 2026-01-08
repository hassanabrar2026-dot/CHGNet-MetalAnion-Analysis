import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

full_csv = os.path.join(data_dir, "mp_full_features.csv")
static_csv = os.path.join(data_dir, "static_features.csv")

print("Reading MP full features:", full_csv)
df = pd.read_csv(full_csv)

print("Reading starting crystal features:", static_csv)
start_df = pd.read_csv(static_csv)
start_row = start_df.iloc[0]

targets = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
    "density",
    "volume",
]

# Keep only numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])

# Remove targets from features
feature_cols = [c for c in numeric_df.columns if c not in targets]

X_all = numeric_df[feature_cols].fillna(0.0)


# Build feature vector for starting structure; missing features -> 0
start_features = {}
for col in feature_cols:
    if col in start_df.columns:
        start_features[col] = start_row[col]
    else:
        start_features[col] = 0.0

X_start = pd.DataFrame([start_features])[feature_cols]

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

rows = []

for target in targets:
    if target not in df.columns:
        print(f"Target {target} not in dataframe, skipping.")
        continue

    print(f"\n=== Training model for {target} ===")
    y_all = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    model = get_model()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)

    # Fit final model on all data for best prediction
    model.fit(X_all, y_all)
    y_start_pred = float(model.predict(X_start)[0])

    # Percentile of prediction in MP distribution
    mp_vals = y_all.values
    percentile = float((mp_vals < y_start_pred).sum() / len(mp_vals) * 100.0)

    rows.append({
        "target": target,
        "predicted_value": y_start_pred,
        "mp_mean": float(mp_vals.mean()),
        "mp_std": float(mp_vals.std()),
        "percentile": percentile,
        "test_R2": float(r2),
    })

out_csv = os.path.join(data_dir, "start_predictions_vs_mp.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Saved:", out_csv)
