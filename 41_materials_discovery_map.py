import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "mp_analysis")
os.makedirs(fig_dir, exist_ok=True)

full_csv = os.path.join(data_dir, "mp_full_features_with_crystal.csv")
df = pd.read_csv(full_csv)

# One-hot encode crystal_system for features, but keep original labels for color
df_feat = df.copy()
if "crystal_system" in df_feat.columns:
    df_feat = pd.get_dummies(df_feat, columns=["crystal_system"], dummy_na=False)

numeric_df = df_feat.select_dtypes(include=["int64", "float64", "bool"]).fillna(0.0)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(numeric_df)

df_map = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "formation_energy_per_atom": df["formation_energy_per_atom"],
    "crystal_system": df.get("crystal_system", "unknown"),
})

# Map colored by formation energy
plt.figure(figsize=(6, 5))
sc = plt.scatter(
    df_map["PC1"],
    df_map["PC2"],
    c=df_map["formation_energy_per_atom"],
    cmap="viridis",
    s=12,
    alpha=0.7,
)
plt.colorbar(sc, label="Formation energy (eV/atom)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "discovery_map_formation_energy.png"), dpi=1200)
plt.close()

# Map colored by crystal system
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df_map,
    x="PC1",
    y="PC2",
    hue="crystal_system",
    s=12,
    alpha=0.8,
    palette="tab10",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "discovery_map_crystal_system.png"), dpi=1200)
plt.close()

print("Materials discovery map figures saved.")
