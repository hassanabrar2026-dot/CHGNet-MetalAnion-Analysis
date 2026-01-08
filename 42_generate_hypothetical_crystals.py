import os
import random
from copy import deepcopy
from pymatgen.core import Structure, Element

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

ref_path = os.path.join(data_dir, "2109450_relaxed_chgnet.vasp")
ref_struct = Structure.from_file(ref_path)

out_dir = os.path.join(data_dir, "hypothetical_structures")
os.makedirs(out_dir, exist_ok=True)

# 1) Random strained variants
for i, strain in enumerate([0.98, 1.02, 1.05]):
    s = ref_struct.copy()
    s.scale_lattice(s.volume * strain)
    s.to(fmt="poscar", filename=os.path.join(out_dir, f"hyp_strained_{i+1}.vasp"))

# 2) Simple alloy: substitute one species
species = list(set(str(ref) for ref in ref_struct.species))
if len(species) >= 1:
    # pick first species, substitute with a nearby element in periodic table
    el0 = Element(species[0])
    # crude neighbor: plus one atomic number if exists
    try:
        new_el = Element.from_Z(el0.Z + 1)
    except Exception:
        new_el = el0
    s_alloy = ref_struct.copy()
    for i_site, sp in enumerate(s_alloy.species):
        if str(sp) == species[0]:
            s_alloy[i_site] = new_el
    s_alloy.to(fmt="poscar", filename=os.path.join(out_dir, "hyp_alloy_1.vasp"))

# 3) Random substitution on subset of sites
s_rand = ref_struct.copy()
unique_species = list({str(sp) for sp in s_rand.species})
if len(unique_species) >= 2:
    # swap ~20% of sites of species 0 to species 1
    from_sp = unique_species[0]
    to_sp = unique_species[1]
    indices = [i for i, sp in enumerate(s_rand.species) if str(sp) == from_sp]
    n_swap = max(1, int(0.2 * len(indices)))
    swap_indices = random.sample(indices, n_swap)
    for idx in swap_indices:
        s_rand[idx] = Element(to_sp)
    s_rand.to(fmt="poscar", filename=os.path.join(out_dir, "hyp_random_sub_1.vasp"))

print("Hypothetical structures written to:", out_dir)
