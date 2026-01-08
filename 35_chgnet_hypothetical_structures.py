import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from chgnet.model import CHGNet

# -----------------------------
# Paths
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "chgnet_analysis")
os.makedirs(fig_dir, exist_ok=True)

# -----------------------------
# Load reference structure
# -----------------------------
ref_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
ref_struct = Structure.from_file(ref_path)

# -----------------------------
# Load CHGNet model
# -----------------------------
model = CHGNet.load()

# -----------------------------
# Generate hypothetical structures
# -----------------------------
rows = []
perturb_levels = np.linspace(0.01, 0.20, 10)  # 1% to 20% perturbation

for p in perturb_levels:
    # Copy reference structure
    struct = ref_struct.copy()

    # Generate fractional perturbations (safe)
    frac_disp = np.random.normal(scale=p, size=(len(struct), 3))
    new_frac = struct.frac_coords + frac_disp

    # Build new structure
    struct = Structure(struct.lattice, struct.species, new_frac)

    # Predict with CHGNet
    pred = model.predict_structure(struct)

    rows.append({
        "perturbation": p,
        "chgnet_energy_per_atom": float(pred["e"] / len(struct)),
        "forces_norm": float(np.linalg.norm(pred["f"])),
        "stress_norm": float(np.linalg.norm(pred["s"])),
    })

# -----------------------------
# Save results
# -----------------------------
dfh = pd.DataFrame(rows)
out_csv = os.path.join(data_dir, "chgnet_hypothetical_results.csv")
dfh.to_csv(out_csv, index=False)
print("Saved:", out_csv)

# -----------------------------
# Plot: Energy vs Perturbation
# -----------------------------
plt.figure(figsize=(5, 4))
plt.plot(dfh["perturbation"], dfh["chgnet_energy_per_atom"], "o-", color="tab:blue")
plt.xlabel("Perturbation magnitude")
plt.ylabel("CHGNet energy_per_atom (eV)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hypothetical_energy_vs_perturbation.png"), dpi=1200)
plt.close()

# -----------------------------
# Plot: Histogram of energies
# -----------------------------
plt.figure(figsize=(5, 4))
dfh["chgnet_energy_per_atom"].hist(bins=20, color="tab:green")
plt.xlabel("CHGNet energy_per_atom (eV)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hypothetical_energy_hist.png"), dpi=1200)
plt.close()

print("Hypothetical structure figures saved.")
