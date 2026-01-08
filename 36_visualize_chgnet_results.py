import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "chgnet_analysis")
os.makedirs(fig_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "mp_chgnet_predictions.csv")
df = pd.read_csv(csv_path)

# -----------------------------
# Energy vs Forces
# -----------------------------
plt.figure(figsize=(5, 4))
plt.scatter(df["chgnet_energy_per_atom"], df["chgnet_forces_norm"], s=10, alpha=0.6)
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Force norm")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "energy_vs_forces.png"), dpi=1200)
plt.close()

# -----------------------------
# Energy vs Stress
# -----------------------------
plt.figure(figsize=(5, 4))
plt.scatter(df["chgnet_energy_per_atom"], df["chgnet_stress_norm"], s=10, alpha=0.6, color="tab:red")
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Stress norm")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "energy_vs_stress.png"), dpi=1200)
plt.close()

# -----------------------------
# KDE joint plot
# -----------------------------
plt.figure(figsize=(5, 4))
sns.kdeplot(
    x=df["chgnet_energy_per_atom"],
    y=df["chgnet_forces_norm"],
    fill=True,
    cmap="viridis",
    thresh=0.05,
)
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Force norm")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "kde_energy_forces.png"), dpi=1200)
plt.close()

# -----------------------------
# Histograms
# -----------------------------
plt.figure(figsize=(5, 4))
df["chgnet_forces_norm"].hist(bins=40, color="tab:green")
plt.xlabel("Force norm")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "forces_hist.png"), dpi=1200)
plt.close()

plt.figure(figsize=(5, 4))
df["chgnet_stress_norm"].hist(bins=40, color="tab:purple")
plt.xlabel("Stress norm")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "stress_hist.png"), dpi=1200)
plt.close()

# -----------------------------
# Correlation heatmap
# -----------------------------
corr = df[["chgnet_energy_per_atom", "chgnet_forces_norm", "chgnet_stress_norm"]].corr()

plt.figure(figsize=(4, 3))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "chgnet_corr_heatmap.png"), dpi=1200)
plt.close()

print("All CHGNet visualization figures saved.")
