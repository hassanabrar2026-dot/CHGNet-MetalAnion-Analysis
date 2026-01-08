import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    RadialDistributionFunction,
)
from chgnet.model import CHGNet

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
os.makedirs(data_dir, exist_ok=True)

cif_path = os.path.join(data_dir, "2109450.cif")
struct = Structure.from_file(cif_path)

# ---------------------------------------------------------
# Basic structural info
# ---------------------------------------------------------
sga = SpacegroupAnalyzer(struct, symprec=1e-3)
spacegroup = sga.get_space_group_symbol()
spg_num = sga.get_space_group_number()
crystal_system = sga.get_crystal_system()

lattice = struct.lattice
a, b, c = lattice.a, lattice.b, lattice.c
alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma

density = struct.density
volume = struct.volume
n_atoms = len(struct)

# ---------------------------------------------------------
# Composition features (matminer)
# ---------------------------------------------------------
comp = struct.composition
elem_feat = ElementProperty.from_preset("magpie")
comp_features = elem_feat.featurize(comp)

comp_feature_labels = elem_feat.feature_labels()

# ---------------------------------------------------------
# Density & symmetry features
# ---------------------------------------------------------
df_feat = DensityFeatures()
density_features = df_feat.featurize(struct)
density_labels = df_feat.feature_labels()

sym_feat = GlobalSymmetryFeatures()
sym_features = sym_feat.featurize(struct)
sym_labels = sym_feat.feature_labels()

# ---------------------------------------------------------
# RDF features (matminer)
# ---------------------------------------------------------
rdf_feat = RadialDistributionFunction(cutoff=10.0, bin_size=0.1)
rdf_features = rdf_feat.featurize(struct)
rdf_labels = rdf_feat.feature_labels()

# ---------------------------------------------------------
# XRD features
# ---------------------------------------------------------
xrd_calc = XRDCalculator(wavelength="CuKa")
pattern = xrd_calc.get_pattern(struct)

# Extract top 10 peaks
top_n = 10
xrd_peaks = sorted(zip(pattern.x, pattern.y), key=lambda x: -x[1])[:top_n]

xrd_positions = [p[0] for p in xrd_peaks]
xrd_intensities = [p[1] for p in xrd_peaks]

# Pad to fixed length
while len(xrd_positions) < top_n:
    xrd_positions.append(0.0)
    xrd_intensities.append(0.0)

# ---------------------------------------------------------
# CHGNet predictions
# ---------------------------------------------------------
model = CHGNet.load()
pred = model.predict_structure(struct)

energy_pa = float(pred["e"] / n_atoms)
forces_norm = float(np.linalg.norm(pred["f"]))
stress_norm = float(np.linalg.norm(pred["s"]))

# ---------------------------------------------------------
# Combine all features into a single row
# ---------------------------------------------------------
data = {
    "material_id": "2109450",
    "spacegroup": spacegroup,
    "spacegroup_number": spg_num,
    "crystal_system": crystal_system,
    "a": a, "b": b, "c": c,
    "alpha": alpha, "beta": beta, "gamma": gamma,
    "density": density,
    "volume": volume,
    "n_atoms": n_atoms,
    "chgnet_energy_per_atom": energy_pa,
    "chgnet_forces_norm": forces_norm,
    "chgnet_stress_norm": stress_norm,
}

# Add composition features
for label, value in zip(comp_feature_labels, comp_features):
    data[f"comp_{label}"] = value

# Add density features
for label, value in zip(density_labels, density_features):
    data[f"density_{label}"] = value

# Add symmetry features
for label, value in zip(sym_labels, sym_features):
    data[f"sym_{label}"] = value

# Add RDF features
for label, value in zip(rdf_labels, rdf_features):
    data[f"rdf_{label}"] = value

# Add XRD features
for i in range(top_n):
    data[f"xrd_peak_pos_{i+1}"] = xrd_positions[i]
    data[f"xrd_peak_int_{i+1}"] = xrd_intensities[i]

# ---------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------
df = pd.DataFrame([data])
out_csv = os.path.join(data_dir, "2109450_ml_features.csv")
df.to_csv(out_csv, index=False)

print("ML dataset saved to:", out_csv)
print("Total features:", len(df.columns))
