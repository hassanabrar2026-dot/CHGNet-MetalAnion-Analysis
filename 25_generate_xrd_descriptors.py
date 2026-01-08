import os
import numpy as np
import pandas as pd
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

mp_csv = os.path.join(data_dir, "mp_dataset.csv")
df = pd.read_csv(mp_csv)

xrd_calc = XRDCalculator(wavelength="CuKa")

# We will store descriptors here
rows = []

print("Generating XRD descriptors...")

for idx, row in df.iterrows():
    try:
        struct = Structure.from_dict(eval(row["structure"]))
    except:
        continue

    pattern = xrd_calc.get_pattern(struct, two_theta_range=(5, 80))

    tt = np.array(pattern.x)
    intens = np.array(pattern.y)

    # Normalize intensities
    if intens.max() > 0:
        intens = intens / intens.max()

    # Extract top 10 peaks
    top_idx = np.argsort(intens)[-10:]
    top_tt = tt[top_idx]
    top_int = intens[top_idx]

    # Statistical descriptors
    desc = {
        "material_id": row["material_id"],
        "xrd_mean_intensity": float(np.mean(intens)),
        "xrd_std_intensity": float(np.std(intens)),
        "xrd_skew_intensity": float(pd.Series(intens).skew()),
        "xrd_kurt_intensity": float(pd.Series(intens).kurt()),
    }

    # Add top peak positions and intensities
    for i in range(10):
        if i < len(top_tt):
            desc[f"peak_{i}_tt"] = float(top_tt[i])
            desc[f"peak_{i}_int"] = float(top_int[i])
        else:
            desc[f"peak_{i}_tt"] = 0.0
            desc[f"peak_{i}_int"] = 0.0

    rows.append(desc)

out_csv = os.path.join(data_dir, "mp_xrd_descriptors.csv")
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Saved XRD descriptors to:", out_csv)
