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
# 1) Load original structure
# ---------------------------------------------------------
orig_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
orig_struct = Structure.from_file(orig_path)

# ---------------------------------------------------------
# 2) XRD pattern
# ---------------------------------------------------------
calc = XRDCalculator(wavelength="CuKa")
pattern = calc.get_pattern(orig_struct)

xrd_x = np.array(pattern.x)
xrd_y = np.array(pattern.y)

# ---------------------------------------------------------
# 3) RDF (original) - from saved file
# ---------------------------------------------------------
rdf_orig_path = os.path.join(data_dir, "rdf_original.txt")
rdf_data = np.loadtxt(rdf_orig_path)
rdf_r = rdf_data[:, 0]
rdf_g = rdf_data[:, 1]

# ---------------------------------------------------------
# 4) Coordination environment (Voronoi)
# ---------------------------------------------------------
vnn = VoronoiNN()
coord_numbers = []
for i, site in enumerate(orig_struct):
    try:
        neigh = vnn.get_nn_info(orig_struct, i)
        coord_numbers.append(len(neigh))
    except Exception:
        pass

# Histogram of coordination
coord_counts = pd.Series(coord_numbers).value_counts().sort_index()

# ---------------------------------------------------------
# 5) CHGNet energy/forces/stress (original + hypothetical)
# ---------------------------------------------------------
model = CHGNet.load()

rows = []

# Original
pred_orig = model.predict_structure(orig_struct)
rows.append({
    "name": "original",
    "energy_per_atom": float(pred_orig["e"] / len(orig_struct)),
    "forces_norm": float(np.linalg.norm(pred_orig["f"])),
    "stress_norm": float(np.linalg.norm(pred_orig["s"])),
})

# Hypotheticals
for fname in sorted(os.listdir(hyp_dir)):
    if not fname.endswith(".vasp"):
        continue

    fpath = os.path.join(hyp_dir, fname)
    struct = Structure.from_file(fpath)
    pred = model.predict_structure(struct)
    rows.append({
        "name": os.path.splitext(fname)[0],
        "energy_per_atom": float(pred["e"] / len(struct)),
        "forces_norm": float(np.linalg.norm(pred["f"])),
        "stress_norm": float(np.linalg.norm(pred["s"])),
    })

df_chg = pd.DataFrame(rows)

# ---------------------------------------------------------
# 6) Multi-panel figure
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Panel 1: XRD
ax1 = axes[0, 0]
ax1.stem(xrd_x, xrd_y, basefmt=" ", linefmt="C0-", markerfmt=" ", use_line_collection=True)
ax1.set_xlabel("2θ (degrees)")
ax1.set_ylabel("Intensity (arb. units)")
ax1.set_title("XRD pattern (Cu Kα)")

# Panel 2: RDF (original)
ax2 = axes[0, 1]
ax2.plot(rdf_r, rdf_g, color="C1")
ax2.set_xlabel("r (Å)")
ax2.set_ylabel("g(r)")
ax2.set_title("Radial Distribution Function")

# Panel 3: Coordination histogram
ax3 = axes[1, 0]
ax3.bar(coord_counts.index.astype(str), coord_counts.values, color="C2")
ax3.set_xlabel("Coordination number")
ax3.set_ylabel("Count")
ax3.set_title("Coordination environment (Voronoi)")

# Panel 4: CHGNet energy & forces (original vs hypothetical)
ax4 = axes[1, 1]
ax42 = ax4.twinx()

x = np.arange(len(df_chg))
ax4.bar(x - 0.2, df_chg["energy_per_atom"], width=0.4, label="Energy/atom (eV)", color="C3")
ax42.plot(x, df_chg["forces_norm"], "o-", label="Force norm", color="C4")

ax4.set_xticks(x)
ax4.set_xticklabels(df_chg["name"], rotation=45, ha="right", fontsize=7)
ax4.set_ylabel("Energy per atom (eV)")
ax42.set_ylabel("Force norm")
ax4.set_title("CHGNet energy & forces")

# Combine legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax42.get_legend_handles_labels()
ax42.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

plt.tight_layout()
out_path = os.path.join(fig_dir, "structure_analysis_panel_1200dpi.png")
plt.savefig(out_path, dpi=1200)
plt.close()

print("Structure analysis panel saved to:", out_path)
