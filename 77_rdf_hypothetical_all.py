import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.analysis.local_env import VoronoiNN
from chgnet.model import CHGNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
hyp_dir = os.path.join(data_dir, "hypothetical_structures")
fig_dir = os.path.join(ROOT, "figs", "structure_analysis")
os.makedirs(fig_dir, exist_ok=True)

# ---------------------------------------------------------
# Load original structure
# ---------------------------------------------------------
orig_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
orig_struct = Structure.from_file(orig_path)

# ---------------------------------------------------------
# XRD pattern
# ---------------------------------------------------------
calc = XRDCalculator(wavelength="CuKa")
pattern = calc.get_pattern(orig_struct)
xrd_x = np.array(pattern.x)
xrd_y = np.array(pattern.y)

# ---------------------------------------------------------
# RDF (original)
# ---------------------------------------------------------
rdf_data = np.loadtxt(os.path.join(data_dir, "rdf_original.txt"))
rdf_r = rdf_data[:, 0]
rdf_g = rdf_data[:, 1]

# ---------------------------------------------------------
# Coordination (Voronoi)
# ---------------------------------------------------------
vnn = VoronoiNN()
coord_numbers = []
for i, site in enumerate(orig_struct):
    try:
        neigh = vnn.get_nn_info(orig_struct, i)
        coord_numbers.append(len(neigh))
    except:
        pass

coord_counts = pd.Series(coord_numbers).value_counts().sort_index()

# ---------------------------------------------------------
# CHGNet predictions
# ---------------------------------------------------------
model = CHGNet.load()

rows = []

# Original
pred = model.predict_structure(orig_struct)
rows.append({
    "name": "original",
    "energy_per_atom": float(pred["e"] / len(orig_struct)),
    "forces_norm": float(np.linalg.norm(pred["f"])),
})

# Hypotheticals
for fname in sorted(os.listdir(hyp_dir)):
    if fname.endswith(".vasp"):
        struct = Structure.from_file(os.path.join(hyp_dir, fname))
        pred = model.predict_structure(struct)
        rows.append({
            "name": fname.replace(".vasp", ""),
            "energy_per_atom": float(pred["e"] / len(struct)),
            "forces_norm": float(np.linalg.norm(pred["f"])),
        })

df = pd.DataFrame(rows)

# ---------------------------------------------------------
# Plot 2×2 panel
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Panel 1: XRD
ax1 = axes[0, 0]
ax1.vlines(xrd_x, 0, xrd_y, color="C0")
ax1.set_title("XRD Pattern (Cu Kα)")
ax1.set_xlabel("2θ (degrees)")
ax1.set_ylabel("Intensity")

# Panel 2: RDF
ax2 = axes[0, 1]
ax2.plot(rdf_r, rdf_g, color="C1")
ax2.set_title("Radial Distribution Function")
ax2.set_xlabel("r (Å)")
ax2.set_ylabel("g(r)")

# Panel 3: Coordination histogram
ax3 = axes[1, 0]
ax3.bar(coord_counts.index.astype(str), coord_counts.values, color="C2")
ax3.set_title("Coordination Environment (Voronoi)")
ax3.set_xlabel("Coordination number")
ax3.set_ylabel("Count")

# Panel 4: CHGNet energy & forces
ax4 = axes[1, 1]
ax42 = ax4.twinx()

x = np.arange(len(df))
ax4.bar(x - 0.2, df["energy_per_atom"], width=0.4, color="C3", label="Energy/atom")
ax42.plot(x, df["forces_norm"], "o-", color="C4", label="Force norm")

ax4.set_xticks(x)
ax4.set_xticklabels(df["name"], rotation=45, ha="right", fontsize=7)
ax4.set_ylabel("Energy per atom (eV)")
ax42.set_ylabel("Force norm")
ax4.set_title("CHGNet Energetics")

plt.tight_layout()
out_path = os.path.join(fig_dir, "structure_analysis_panel_1200dpi.png")
plt.savefig(out_path, dpi=1200)
plt.close()

print("Saved:", out_path)
