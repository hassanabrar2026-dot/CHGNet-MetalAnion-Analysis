import os
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
struct = Structure.from_file(path)

calc = XRDCalculator(wavelength="CuKa")
pattern = calc.get_pattern(struct)

plt.figure(figsize=(8,4))
plt.stem(pattern.x, pattern.y, basefmt=" ")
plt.xlabel("2θ (degrees)")
plt.ylabel("Intensity")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "xrd_pattern_1200dpi.png"), dpi=1200)
plt.close()

print("Saved XRD pattern.")
