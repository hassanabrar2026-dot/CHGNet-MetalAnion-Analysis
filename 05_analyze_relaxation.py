from pathlib import Path
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from chgnet.model import CHGNet, CHGNetCalculator
from ase.optimize import BFGS
from ase import Atoms
import numpy as np

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
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figs_dir = project_root / "figs"
    figs_dir.mkdir(exist_ok=True)

    # Load structures
    exp_path = data_dir / "2109450.cif"
    rel_path = data_dir / "2109450_relaxed_chgnet.cif"

    exp_struct = Structure.from_file(str(exp_path))
    rel_struct = Structure.from_file(str(rel_path))

    # Convert experimental structure to ASE
    atoms = pmg_to_ase(exp_struct)

    # Load CHGNet calculator
    model = CHGNet.load(use_device="cpu")
    atoms.calc = CHGNetCalculator(model=model)

    # Recompute energies along a linear interpolation path
    n_steps = 50
    energies = []
    steps = list(range(n_steps))

    exp_cart = np.array(exp_struct.cart_coords)
    rel_cart = np.array(rel_struct.cart_coords)

    print("Computing energy interpolation path...")

    for i in steps:
        t = i / (n_steps - 1)
        interp_positions = (1 - t) * exp_cart + t * rel_cart
        atoms.set_positions(interp_positions)
        e = atoms.get_potential_energy()
        energies.append(e)

    #