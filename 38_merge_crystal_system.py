import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

full_csv = os.path.join(data_dir, "mp_full_features.csv")
crys_csv = os.path.join(data_dir, "mp_crystal_systems.csv")

df_full = pd.read_csv(full_csv)
df_crys = pd.read_csv(crys_csv)

df_merged = df_full.merge(df_crys[["material_id", "crystal_system"]], on="material_id", how="left")

out_csv = os.path.join(data_dir, "mp_full_features_with_crystal.csv")
df_merged.to_csv(out_csv, index=False)
print("Saved:", out_csv)
