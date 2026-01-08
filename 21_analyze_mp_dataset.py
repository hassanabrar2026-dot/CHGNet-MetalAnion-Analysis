import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
print("Reading:", mp_csv)
df = pd.read_csv(mp_csv)

# Basic summary statistics for main properties
props = [
    "density",
    "volume",
    "energy_per_atom",
    "formation_energy_per_atom",
    "band_gap",
]

summary = df[props].describe().T
summary.to_csv(os.path.join(data_dir, "mp_summary_stats.csv"))
print("Saved summary stats to mp_summary_stats.csv")

# Histograms
for col in props:
    if col not in df.columns:
        continue
    plt.figure(figsize=(5, 4))
    df[col].dropna().hist(bins=40)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    out_fig = os.path.join(fig_dir, f"hist_{col}.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print("Saved:", out_fig)
