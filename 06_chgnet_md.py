from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from chgnet.model import CHGNet, CHGNetCalculator
from ase import Atoms
from ase.md.langevin import Langevin
from ase.io import write
from ase import units

def pmg_to_ase(structure):
    symbols = [str(site.specie) for site in structure]
    positions = structure.cart_coords
    cell = structure.lattice.matrix
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    return atoms

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figs_dir = project_root / "figs"
    figs_dir.mkdir(exist_ok=True)

    # Load relaxed structure
    rel_path = data_dir / "2109450_relaxed_chgnet.cif"
    structure = Structure.from_file(str(rel_path))

    # Convert to ASE
    atoms = pmg_to_ase(structure)

    # Attach CHGNet calculator
    model = CHGNet.load(use_device="cpu")
    atoms.calc = CHGNetCalculator(model=model)

    # MD parameters
    temperature_K = 300
    timestep_fs = 1.0
    total_steps = 2000  # ~2 ps

    dyn = Langevin(
        atoms,
        timestep_fs * units.fs,
        temperature_K * units.kB,
        friction=0.01
    )

    # Storage arrays
    energies = []
    temperatures = []
    steps = []

    traj_path = data_dir / "md_300K.traj"
    xyz_path = data_dir / "md_300K.xyz"
    csv_path = data_dir / "md_300K.csv"

    print("Running CHGNet MD at 300 K...")

    for step in range(total_steps):
        dyn.run(1)

        e = atoms.get_potential_energy()
        t = atoms.get_temperature()

        energies.append(e)
        temperatures.append(t)
        steps.append(step)

        # Save XYZ frame
        write(xyz_path, atoms, format="xyz", append=True)

    # Save CSV
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("step,energy_eV,temperature_K\n")
        for s, e, t in zip(steps, energies, temperatures):
            f.write(f"{s},{e:.8f},{t:.4f}\n")

    print(f"MD data saved to: {csv_path}")
    print(f"XYZ trajectory saved to: {xyz_path}")

    # Plot temperature
    plt.figure(figsize=(6, 4))
    plt.plot(steps, temperatures, color="tab:blue")
    plt.xlabel("MD step")
    plt.ylabel("Temperature (K)")
    plt.title("CHGNet MD: Temperature vs Time (300 K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    temp_fig = figs_dir / "md_300K_temperature.png"
    plt.savefig(temp_fig, dpi=300)
    plt.close()

    print(f"Temperature plot saved to: {temp_fig}")

    # Plot energy
    plt.figure(figsize=(6, 4))
    plt.plot(steps, energies, color="tab:red")
    plt.xlabel("MD step")
    plt.ylabel("Energy (eV)")
    plt.title("CHGNet MD: Energy vs Time (300 K)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    energy_fig = figs_dir / "md_300K_energy.png"
    plt.savefig(energy_fig, dpi=300)
    plt.close()

    print(f"Energy plot saved to: {energy_fig}")

if __name__ == "__main__":
    main()
