import os
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
struct = Structure.from_file(path)

print("Lattice parameters:")
print(struct.lattice)

print("\nAtomic species:")
print(struct.species)

sga = SpacegroupAnalyzer(struct, symprec=1e-3)
print("\nSpace group:", sga.get_space_group_symbol(), sga.get_space_group_number())
print("Crystal system:", sga.get_crystal_system())
