from pymatgen.core import Structure
from pathlib import Path

def main():
    # Path to your CIF file
    cif_path = Path(__file__).resolve().parents[1] / "data" / "2109450.cif"

    # Load structure
    structure = Structure.from_file(str(cif_path))

    print("=== Basic information ===")
    print("Formula:", structure.composition.formula)
    print("Number of atoms in cell:", len(structure))
    print("Lattice parameters (a, b, c):", structure.lattice.abc)
    print("Lattice angles (α, β, γ):", structure.lattice.angles)
    print("Space group (from CIF, if present) is NOT directly stored;")
    print("  but CIF metadata should reflect your P 1 (2) cell.")

    print("\n=== Atomic sites ===")
    for site in structure:
        print(f"Element: {site.specie}, frac coords: {site.frac_coords}")

if __name__ == "__main__":
    main()
