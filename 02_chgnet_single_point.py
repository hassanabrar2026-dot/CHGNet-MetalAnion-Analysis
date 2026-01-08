from pathlib import Path
from pymatgen.core import Structure
from chgnet.model import CHGNet

def main():
    # Load structure
    cif_path = Path(__file__).resolve().parents[1] / "data" / "2109450.cif"
    structure = Structure.from_file(str(cif_path))

    # Load pretrained CHGNet model
    model = CHGNet.load()

    # Single-point prediction
    prediction = model.predict_structure(structure)

    print("=== CHGNet Single-Point Results ===")
    energy_total = prediction["e"].item()
    energy_per_atom = energy_total / len(structure)

    print("Total energy (eV):", energy_total)
    print("Energy per atom (eV/atom):", energy_per_atom)

    forces = prediction["f"]
    print("\nForces on first 5 atoms (eV/Å):")
    for i in range(min(5, len(forces))):
        fx, fy, fz = forces[i].tolist()
        print(f"Atom {i:3d}: ({fx:9.5f}, {fy:9.5f}, {fz:9.5f})")

if __name__ == "__main__":
    main()
