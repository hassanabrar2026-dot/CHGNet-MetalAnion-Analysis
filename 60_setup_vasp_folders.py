import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")

# ID of the system
mid = "2109450"

base_dir = os.path.join(data_dir, mid)
os.makedirs(base_dir, exist_ok=True)

subdirs = ["01_relax", "02_static", "03_dos", "04_band", "05_phonon"]
for sd in subdirs:
    os.makedirs(os.path.join(base_dir, sd), exist_ok=True)

# Copy CHGNet-relaxed structure as POSCAR for 01_relax
src = os.path.join(data_dir, f"{mid}_relaxed_chgnet.vasp")
dst = os.path.join(base_dir, "01_relax", "POSCAR")

shutil.copy(src, dst)
print("Created folder structure and copied POSCAR to 01_relax.")
