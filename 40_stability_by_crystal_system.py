import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

full_csv = os.path.join(data_dir, "mp_full_features_with_crystal.csv")
df = pd.read_csv(full_csv)

if "crystal_system" not in df.columns:
    raise ValueError("crystal_system column missing. Run 38_merge_crystal_system.py first.")

# Boxplot: formation energy by crystal system
plt.figure(figsize=(7, 4))
sns.boxplot(
    data=df,
    x="crystal_system",
    y="formation_energy_per_atom",
    order=sorted(df["crystal_system"].dropna().unique())
)
plt.xticks(rotation=45)
plt.ylabel("Formation energy per atom (eV)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "formation_by_crystal_system_boxplot.png"), dpi=1200)
plt.close()

# Barplot: fraction of stable structures by crystal system (is_stable column assumed)
if "is_stable" in df.columns:
    group = df.groupby("crystal_system")["is_stable"].mean().reset_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=group,
        x="crystal_system",
        y="is_stable",
        order=sorted(group["crystal_system"].dropna().unique()),
        color="tab:blue",
    )
    plt.xticks(rotation=45)
    plt.ylabel("Fraction stable")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "stability_fraction_by_crystal_system.png"), dpi=1200)
    plt.close()

print("Stability by crystal system figures saved.")
