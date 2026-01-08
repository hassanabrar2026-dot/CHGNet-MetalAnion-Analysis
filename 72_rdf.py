import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from ase.geometry.analysis import Analysis

# ---------------------------------------------------------
# Load structure
# ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

vasp_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
struct = Structure.from_file(vasp_path)

# Convert to ASE atoms
atoms = struct.to_ase_atoms()

# ---------------------------------------------------------
# Build a 3×3×3 supercell (large enough for rmax=10 Å)
# ---------------------------------------------------------
atoms_super = atoms.repeat((3, 3, 3))

# ---------------------------------------------------------
# Compute RDF
# ---------------------------------------------------------
analysis = Analysis(atoms_super)

rdf_result = analysis.get_rdf(rmax=10.0, nbins=300)

# ---------------------------------------------------------
# Handle ALL possible ASE return types
# ---------------------------------------------------------

# Case A: ASE returns a list of RDF arrays (your case)
if isinstance(rdf_result, list):
    # Convert each element to numpy array
    rdf_arrays = [np.array(r) for r in rdf_result]

    # Average all RDF curves
    rdf = np.mean(rdf_arrays, axis=0)

    # Generate bins manually
    bins = np.linspace(0, 10.0, len(rdf))

# Case B: ASE returns a single RDF array
elif isinstance(rdf_result, np.ndarray):
    rdf = rdf_result
    bins = np.linspace(0, 10.0, len(rdf))

# Case C: ASE returns (rdf, bins)
elif isinstance(rdf_result, (tuple, list)) and len(rdf_result) == 2:
    rdf, bins = rdf_result

else:
    raise RuntimeError(f"Unhandled RDF return type: {type(rdf_result)}")

# ---------------------------------------------------------
# Plot RDF
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(bins, rdf, color="blue", linewidth=1.5)
plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function (3×3×3 Supercell)")
plt.tight_layout()

out_path = os.path.join(data_dir, "rdf_1200dpi.png")
plt.savefig(out_path, dpi=1200)
plt.close()

print("RDF saved to:", out_path)
