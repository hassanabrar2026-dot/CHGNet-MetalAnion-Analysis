import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.trainer import Relaxer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
fig_dir = os.path.join(ROOT, "figs", "chgnet_analysis")
os.makedirs(fig_dir, exist_ok=True)

hyp_dir = os.path.join(data_dir, "hypothetical_structures")
files = glob.glob(os.path.join(hyp_dir, "*.vasp"))

if not files:
    raise FileNotFoundError("No hypothetical structures found. Run 42_generate_hypothetical_crystals.py first.")

relaxer = Relaxer()
rows = []

for fpath in files:
    name = os.path.splitext(os.path.basename(fpath))[0]
    struct = Structure.from_file(fpath)
    print("Relaxing:", name)
    result = relaxer.relax(struct)

    final_struct = result["final_structure"]
    final_e = result["trajectory"].energies[-1]  # eV
    n_atoms = len(final_struct)
    e_pa = float(final_e / n_atoms)

    rows.append({
        "name": name,
        "final_energy_per_atom": e_pa,
        "n_atoms": n_atoms,
    })

df = pd.DataFrame(rows)
out_csv = os.path.join(data_dir, "hypothetical_relaxed_chgnet.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)

plt.figure(figsize=(5, 4))
plt.bar(df["name"], df["final_energy_per_atom"], color="tab:orange")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Final energy per atom (eV)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hypothetical_relaxed_energies.png"), dpi=1200)
plt.close()

print("Hypothetical relaxation figure saved.")
