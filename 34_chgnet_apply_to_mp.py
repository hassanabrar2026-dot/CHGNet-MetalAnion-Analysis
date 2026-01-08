import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from chgnet.model import CHGNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "chgnet_analysis")
os.makedirs(fig_dir, exist_ok=True)

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
df = pd.read_csv(mp_csv)

model = CHGNet.load()

rows = []

print("Running CHGNet on MP structures...")

for idx, row in df.iterrows():
    try:
        struct = Structure.from_dict(eval(row["structure"]))
    except:
        continue

    pred = model.predict_structure(struct)

    rows.append({
        "material_id": row["material_id"],
        "mp_energy_per_atom": row["energy_per_atom"],
        "chgnet_energy_per_atom": float(pred["e"] / len(struct)),
        "chgnet_forces_norm": float(np.linalg.norm(pred["f"])),
        "chgnet_stress_norm": float(np.linalg.norm(pred["s"])),
    })

out_csv = os.path.join(data_dir, "mp_chgnet_predictions.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Saved:", out_csv)

# Load for plotting
dfp = pd.DataFrame(rows)

# Scatter: CHGNet vs MP
plt.figure(figsize=(5, 5))
plt.scatter(dfp["mp_energy_per_atom"], dfp["chgnet_energy_per_atom"], s=10, alpha=0.6)
minv = min(dfp["mp_energy_per_atom"].min(), dfp["chgnet_energy_per_atom"].min())
maxv = max(dfp["mp_energy_per_atom"].max(), dfp["chgnet_energy_per_atom"].max())
plt.plot([minv, maxv], [minv, maxv], "k--")
plt.xlabel("MP energy_per_atom (eV)")
plt.ylabel("CHGNet energy_per_atom (eV)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "chgnet_vs_mp_energy.png"), dpi=1200)
plt.close()

# Residuals
res = dfp["chgnet_energy_per_atom"] - dfp["mp_energy_per_atom"]
plt.figure(figsize=(5, 4))
plt.scatter(dfp["chgnet_energy_per_atom"], res, s=10, alpha=0.6)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Residual (CHGNet - MP)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "chgnet_residuals.png"), dpi=1200)
plt.close()

# Histogram
plt.figure(figsize=(5, 4))
dfp["chgnet_energy_per_atom"].hist(bins=40)
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "chgnet_energy_hist.png"), dpi=1200)
plt.close()

print("All CHGNet figures saved.")
