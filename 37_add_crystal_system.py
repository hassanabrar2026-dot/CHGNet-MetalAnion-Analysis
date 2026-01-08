import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
df = pd.read_csv(mp_csv)

rows = []

print("Extracting crystal systems...")

for idx, row in df.iterrows():
    try:
        struct = Structure.from_dict(eval(row["structure"]))
        sga = SpacegroupAnalyzer(struct, symprec=1e-3)

        spacegroup = sga.get_space_group_symbol()
        spg_num = sga.get_space_group_number()
        crystal_sys = sga.get_crystal_system()

        rows.append({
            "material_id": row["material_id"],
            "formula": row["formula"],
            "spacegroup_symbol": spacegroup,
            "spacegroup_number": spg_num,
            "crystal_system": crystal_sys,
        })

    except Exception as e:
        rows.append({
            "material_id": row["material_id"],
            "formula": row["formula"],
            "spacegroup_symbol": None,
            "spacegroup_number": None,
            "crystal_system": None,
        })

out_csv = os.path.join(data_dir, "mp_crystal_systems.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Saved:", out_csv)
