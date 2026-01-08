import os
import pandas as pd
from pymatgen.core import Structure
from mp_api.client import MPRester

# -----------------------------
# Paths
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
relaxed_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")

print("Reading reference structure:", relaxed_path)
ref_struct = Structure.from_file(relaxed_path)

# -----------------------------
# Extract dominant elements
# -----------------------------
comp = ref_struct.composition.fractional_composition
sorted_elements = sorted(comp.items(), key=lambda x: x[1], reverse=True)

# Keep top 1–2 most abundant elements
main_elements = [el.symbol for el, frac in sorted_elements[:2]]
print("Using main elements for MP search:", main_elements)

# -----------------------------
# Materials Project API
# -----------------------------
API_KEY = "oYSGoP03VNw8QRjN8Udv02PUC9z33Vbg"   # <-- Replace with your MP API key
mpr = MPRester(API_KEY)

print("Querying Materials Project for structures containing ANY of:", main_elements)

results = mpr.summary.search(
    elements=main_elements,   # match ANY of these elements
    fields=[
        "material_id",
        "formula_pretty",
        "structure",                     # <-- REQUIRED for XRD
        "density",
        "volume",
        "energy_per_atom",
        "formation_energy_per_atom",
        "band_gap",
        "is_stable",
    ],
)

print(f"Total hits found: {len(results)}")

# -----------------------------
# Keep top 500 entries
# -----------------------------
top = results[:500]
print(f"Keeping {len(top)} structures.")

rows = []

for r in top:
    s = r.structure
    comp = s.composition

    row = {
        "material_id": r.material_id,
        "formula": r.formula_pretty,
        "structure": s.as_dict(),        # <-- Save structure as dict
        "density": r.density,
        "volume": r.volume,
        "energy_per_atom": r.energy_per_atom,
        "formation_energy_per_atom": r.formation_energy_per_atom,
        "band_gap": r.band_gap,
        "is_stable": r.is_stable,
    }

    # Composition fractions
    for el, frac in comp.fractional_composition.items():
        row[f"frac_{el.symbol}"] = frac

    rows.append(row)

df = pd.DataFrame(rows)

out_csv = os.path.join(data_dir, "mp_dataset.csv")
df.to_csv(out_csv, index=False)

print("Saved dataset to:", out_csv)
