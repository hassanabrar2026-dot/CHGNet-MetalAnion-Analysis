import os
import numpy as np
import pandas as pd
from ase.io import read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
traj_path = os.path.join(ROOT, "data", "md_300K.xyz")
out_csv = os.path.join(ROOT, "data", "md_timeseries.csv")

print("Reading:", traj_path)
atoms_list = read(traj_path, ":")

n_frames = len(atoms_list)
n_atoms = len(atoms_list[0])

dt_fs = 1.0  # assume 1 fs timestep
time_fs = np.arange(n_frames) * dt_fs

# Reference positions for displacement
pos0 = atoms_list[0].get_positions()

avg_disp = []
rms_disp = []
com_drift = []
bbox_volume = []

for at in atoms_list:
    pos = at.get_positions()

    # Displacement from initial frame
    disp = pos - pos0
    dist = np.linalg.norm(disp, axis=1)

    avg_disp.append(np.mean(dist))
    rms_disp.append(np.sqrt(np.mean(dist**2)))

    # Center of mass drift
    com = np.mean(pos, axis=0)
    com0 = np.mean(pos0, axis=0)
    com_drift.append(np.linalg.norm(com - com0))

    # Bounding box volume (approximate)
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    bbox = maxs - mins
    bbox_volume.append(float(bbox[0] * bbox[1] * bbox[2]))

df = pd.DataFrame({
    "time_fs": time_fs,
    "avg_displacement_A": avg_disp,
    "rms_displacement_A": rms_disp,
    "com_drift_A": com_drift,
    "bbox_volume_A3": bbox_volume,
})

df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
