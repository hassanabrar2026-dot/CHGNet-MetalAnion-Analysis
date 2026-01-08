import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
xrd_csv = os.path.join(data_dir, "mp_xrd_descriptors.csv")

df_mp = pd.read_csv(mp_csv)
df_xrd = pd.read_csv(xrd_csv)

df = df_mp.merge(df_xrd, on="material_id", how="inner")

out_csv = os.path.join(data_dir, "mp_full_features.csv")
df.to_csv(out_csv, index=False)

print("Saved merged dataset to:", out_csv)
