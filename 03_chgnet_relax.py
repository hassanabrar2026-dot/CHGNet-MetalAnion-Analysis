from pathlib import Path
from pymatgen.core import Structure
from chgnet.model import CHGNet, CHGNetCalculator
from ase.optimize import BFGS
from ase import Atoms

def pmg_to_ase(structure):
    symbols = [str(site.specie) for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms

def ase_to_pmg(atoms):
    lattice = atoms.cell.array
    species = atoms.get_chemical_symbols()
    coords = atoms.get_positions()
    return Structure(lattice, species, coords, coords_are_cartesian=True)

def main():
    # Load CIF
    cif_path = Path(__file__).resolve().parents[1] / "data" / "2109450.cif"
    structure = Structure.from_file(str(cif_path))

    # Convert to ASE
    atoms = pmg_to_ase(structure)

    # Load CHGNet model
    model = CHGNet.load(use_device="cpu")
    atoms.calc = CHGNetCalculator(model=model)

    print("Starting CHGNet relaxation...")

    # Run relaxation
    dyn = BFGS(atoms, logfile="relax.log")
    dyn.run(fmax=0.05)

    # Convert back to Pymatgen
    final_structure = ase_to_pmg(atoms)

    # Save results
    out_dir = Path(__file__).resolve().parents[1] / "data"

    # Save CIF
    final_structure.to(filename=str(out_dir / "2109450_relaxed_chgnet.cif"))

    # Save POSCAR-style VASP file
    final_structure.to(
        filename=str(out_dir / "2109450_relaxed_chgnet.vasp"),
        fmt="poscar"
    )

    print("\nRelaxation complete.")
    print("Saved:")
    print("  data/2109450_relaxed_chgnet.cif")
    print("  data/2109450_relaxed_chgnet.vasp")

if __name__ == "__main__":
    main()
