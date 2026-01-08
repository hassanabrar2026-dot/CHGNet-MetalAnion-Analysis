import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.geometry.analysis import Analysis
from chgnet.model import CHGNet

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "interatomic_analysis")
os.makedirs(fig_dir, exist_ok=True)

cif_path = os.path.join(data_dir, "2109450.cif")
struct = Structure.from_file(cif_path)
atoms = struct.to_ase_atoms()

# ---------------------------------------------------------
# 1. CHGNet predictions (energy, forces, stress)
# ---------------------------------------------------------
model = CHGNet.load()
pred = model.predict_structure(struct)

energy_pa = float(pred["e"] / len(struct))
forces = pred["f"]
stress = pred["s"]

print("CHGNet energy per atom:", energy_pa)
print("Force norm:", np.linalg.norm(forces))
print("Stress norm:", np.linalg.norm(stress))

# Save results
df = pd.DataFrame({
    "energy_per_atom": [energy_pa],
    "forces_norm": [np.linalg.norm(forces)],
    "stress_norm": [np.linalg.norm(stress)],
})
df.to_csv(os.path.join(data_dir, "2109450_chgnet_results.csv"), index=False)

# ---------------------------------------------------------
# 2. Symmetry + crystal system
# ---------------------------------------------------------
sga = SpacegroupAnalyzer(struct, symprec=1e-3)
spacegroup = sga.get_space_group_symbol()
crystal_system = sga.get_crystal_system()

print("Space group:", spacegroup)
print("Crystal system:", crystal_system)

# ---------------------------------------------------------
# 3. XRD pattern
# ---------------------------------------------------------
xrd = XRDCalculator(wavelength="CuKa")
pattern = xrd.get_pattern(struct)

plt.figure(figsize=(7,4))
plt.vlines(pattern.x, 0, pattern.y, color="C0")
plt.xlabel("2θ (degrees)")
plt.ylabel("Intensity")
plt.title("XRD Pattern (Cu Kα)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "xrd_2109450.png"), dpi=1200)
plt.close()

# ---------------------------------------------------------
# 4. Coordination environment (Voronoi)
# ---------------------------------------------------------
vnn = VoronoiNN()
coord_numbers = []

for i, site in enumerate(struct):
    try:
        neigh = vnn.get_nn_info(struct, i)
        coord_numbers.append(len(neigh))
    except:
        pass

plt.figure(figsize=(5,4))
plt.hist(coord_numbers, bins=range(1,15), color="C2", edgecolor="black")
plt.xlabel("Coordination number")
plt.ylabel("Count")
plt.title("Coordination Environment (Voronoi)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "coordination_2109450.png"), dpi=1200)
plt.close()

# ---------------------------------------------------------
# 5. RDF (using supercell)
# ---------------------------------------------------------
atoms_super = atoms.repeat((3,3,3))
analysis = Analysis(atoms_super)
rdf_result = analysis.get_rdf(rmax=10.0, nbins=300)

# Handle ASE return types
if isinstance(rdf_result, list):
    rdf_arrays = [np.array(r) for r in rdf_result]
    rdf = np.mean(rdf_arrays, axis=0)
elif isinstance(rdf_result, np.ndarray):
    rdf = rdf_result
else:
    rdf = np.array(rdf_result[0])

bins = np.linspace(0, 10.0, len(rdf))

plt.figure(figsize=(7,4))
plt.plot(bins, rdf, color="C1")
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function (CHGNet structure)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "rdf_2109450.png"), dpi=1200)
plt.close()

# ---------------------------------------------------------
# 6. Elastic-like response (finite strain test)
# ---------------------------------------------------------
strains = np.linspace(-0.03, 0.03, 9)
energies = []

for eps in strains:
    strained = struct.copy()
    strained.scale_lattice(strained.volume * (1 + eps))
    pred_s = model.predict_structure(strained)
    e_pa = float(pred_s["e"] / len(strained))
    energies.append(e_pa)

plt.figure(figsize=(6,4))
plt.plot(strains, energies, "o-", color="C3")
plt.xlabel("Volume strain")
plt.ylabel("Energy per atom (eV)")
plt.title("CHGNet Elastic-like Response")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "elastic_response_2109450.png"), dpi=1200)
plt.close()

print("All interatomic potential analysis complete.")
