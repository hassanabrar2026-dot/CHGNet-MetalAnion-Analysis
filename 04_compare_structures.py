from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure

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

    if len(exp_struct) != len(rel_struct):
        raise ValueError(f"Atom count mismatch: exp={len(exp_struct)}, relaxed={len(rel_struct)}")

    print("=== Lattice comparison ===")
    a_exp, b_exp, c_exp = exp_struct.lattice.abc
    a_rel, b_rel, c_rel = rel_struct.lattice.abc

    vol_exp = exp_struct.lattice.volume
    vol_rel = rel_struct.lattice.volume

    print(f"Experimental a, b, c (Å): {a_exp:.4f}, {b_exp:.4f}, {c_exp:.4f}")
    print(f"Relaxed     a, b, c (Å): {a_rel:.4f}, {b_rel:.4f}, {c_rel:.4f}")

    print(f"Experimental volume (Å^3): {vol_exp:.4f}")
    print(f"Relaxed     volume (Å^3): {vol_rel:.4f}")
    print(f"Volume change (%): {(vol_rel - vol_exp) / vol_exp * 100:.3f}")

    # Per-atom displacement in Cartesian space
    exp_cart = np.array(exp_struct.cart_coords)
    rel_cart = np.array(rel_struct.cart_coords)

    displacements = np.linalg.norm(rel_cart - exp_cart, axis=1)

    print("\n=== Atomic displacement statistics ===")
    print(f"Mean displacement (Å): {displacements.mean():.4f}")
    print(f"Max displacement  (Å): {displacements.max():.4f}")
    print(f"Min displacement  (Å): {displacements.min():.4f}")

    # Save CSV with displacements
    csv_path = data_dir / "2109450_relaxation_displacements.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("index,element,disp_angstrom\n")
        for i, (site, disp) in enumerate(zip(exp_struct, displacements)):
            f.write(f"{i},{site.specie},{disp:.6f}\n")
    print(f"\nDisplacement data saved to: {csv_path}")

    # Plot 1: Lattice parameter comparison
    labels = ["a", "b", "c"]
    exp_vals = [a_exp, b_exp, c_exp]
    rel_vals = [a_rel, b_rel, c_rel]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, exp_vals, width, label="Experimental")
    plt.bar(x + width/2, rel_vals, width, label="Relaxed")
    plt.xticks(x, labels)
    plt.ylabel("Lattice parameter (Å)")
    plt.title("Lattice parameters: experimental vs CHGNet-relaxed")
    plt.legend()
    plt.tight_layout()
    lattice_fig_path = figs_dir / "2109450_lattice_comparison.png"
    plt.savefig(lattice_fig_path, dpi=300)
    plt.close()
    print(f"Lattice comparison figure saved to: {lattice_fig_path}")

    # Plot 2: Displacement histogram
    plt.figure(figsize=(6, 4))
    plt.hist(displacements, bins=30, color="tab:blue", edgecolor="black")
    plt.xlabel("Atomic displacement (Å)")
    plt.ylabel("Count")
    plt.title("Distribution of atomic displacements upon CHGNet relaxation")
    plt.tight_layout()
    disp_fig_path = figs_dir / "2109450_displacement_histogram.png"
    plt.savefig(disp_fig_path, dpi=300)
    plt.close()
    print(f"Displacement histogram figure saved to: {disp_fig_path}")

if __name__ == "__main__":
    main()
