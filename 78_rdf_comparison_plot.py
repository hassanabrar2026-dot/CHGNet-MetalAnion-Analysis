import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT, "data")
rdf_out_dir = os.path.join(data_dir, "rdf_hypothetical")
fig_dir = os.path.join(ROOT, "figs", "structure_analysis")
os.makedirs(fig_dir, exist_ok=True)

# Load original RDF
orig_path = os.path.join(data_dir, "rdf_original.txt")
orig_data = np.loadtxt(orig_path)
r = orig_data[:, 0]
g_orig = orig_data[:, 1]

# Load hypothetical RDFs
rdf_files = sorted(f for f in os.listdir(rdf_out_dir) if f.startswith("rdf_") and f.endswith(".txt"))

plt.figure(figsize=(7, 4))
plt.plot(r, g_orig, color="black", linewidth=2.0, label="Original")

colors = plt.cm.tab10.colors
for i, fname in enumerate(rdf_files):
    fpath = os.path.join(rdf_out_dir, fname)
    data = np.loadtxt(fpath)
    g = data[:, 1]
    label = os.path.splitext(fname)[0].replace("rdf_", "")
    plt.plot(r, g, color=colors[i % len(colors)], alpha=0.7, linewidth=1.0, label=label)

plt.xlabel("r (Å)")
plt.ylabel("g(r)")
plt.title("RDF Comparison: Original vs Hypothetical Structures")
plt.legend(fontsize=7, ncol=2)
plt.tight_layout()

out_path = os.path.join(fig_dir, "rdf_comparison_1200dpi.png")
plt.savefig(out_path, dpi=1200)
plt.close()

print("RDF comparison plot saved to:", out_path)
