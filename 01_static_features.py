import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
relaxed_path = os.path.join(ROOT, "data", "2109450_relaxed_chgnet.vasp")
out_csv = os.path.join(ROOT, "data", "static_features.csv")

print("Reading:", relaxed_path)
structure = Structure.from_file(relaxed_path)

cn = CrystalNN()
comp = structure.composition
lat = structure.lattice

features = {
    "formula": comp.reduced_formula,
    "num_atoms": len(structure),
    "density": structure.density,
    "a": lat.a, "b": lat.b, "c": lat.c,
    "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
    "volume": lat.volume,
}

# Composition fractions
for el, frac in comp.fractional_composition.items():
    features[f"frac_{el.symbol}"] = frac

# Coordination numbers
coord_nums = []
for i in range(len(structure)):
    try:
        neighs = cn.get_nn_info(structure, i)
        coord_nums.append(len(neighs))
    except:
        pass

features["avg_coord_num"] = float(np.mean(coord_nums))
features["std_coord_num"] = float(np.std(coord_nums))

pd.DataFrame([features]).to_csv(out_csv, index=False)
print("Saved:", out_csv)
