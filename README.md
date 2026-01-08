Interatomic‑Potential‑Driven Structural and Energetic Analysis of Multicomponent Metal–Anion Crystalline Compounds Using CHGNet and Machine‑Learned Descriptors
This repository contains the full workflow, scripts, and analysis pipeline used to study multicomponent metal–anion crystalline compounds using the CHGNet machine‑learned interatomic potential. The project integrates structural analysis, energetic evaluation, PXRD‑based descriptors, RDF fingerprints, coordination environments, and symmetry features into a unified, ML‑ready framework.

📌 Project Overview
This project demonstrates how machine‑learned interatomic potentials can be used to:

Predict energies, forces, and stresses of crystalline materials

Analyze bonding environments and local structure

Generate PXRD‑based descriptors for ML models

Compute radial distribution functions (RDFs)

Extract symmetry, lattice, and coordination features

Evaluate elastic‑like strain–energy responses

Compare a reference structure with hundreds of hypothetical variants

The workflow is fully automated and does not require DFT calculations.

🧠 Scientific Motivation
Multicomponent metal–anion crystalline compounds (oxides, chalcogenides, pnictides, halides, and mixed‑anion materials) form a vast and technologically important class of inorganic solids. Traditional DFT calculations are too expensive for large‑scale screening.

CHGNet, a universal graph neural network interatomic potential, enables:

near‑DFT accuracy

orders‑of‑magnitude faster evaluation

scalable analysis of large structural datasets

This repository provides a complete pipeline for generating ML‑ready descriptors and interatomic‑potential‑based insights for these materials.

📂 Repository Structure
Code
project/
│
├── data/
│   ├── 2109450.cif                     # Reference crystal
│   ├── hypothetical_structures/        # ~500 related structures
│   ├── rdf_original.txt                # RDF of reference structure
│   ├── rdf_hypothetical/               # RDFs for all hypothetical structures
│   └── 2109450_ml_features.csv         # ML feature vector
│
├── scripts/
│   ├── 77_rdf_hypothetical_all.py      # RDF generation for all structures
│   ├── 78_rdf_comparison_plot.py       # RDF comparison figure
│   ├── 79_structure_analysis_panel.py  # XRD + RDF + coordination + CHGNet panel
│   └── 81_interatomic_potential_analysis_2109450.py
│
├── figs/
│   └── interatomic_analysis/           # All generated figures
│
└── README.md
⚙️ Key Features
1. CHGNet Interatomic Potential Analysis
Energy per atom

Force and stress norms

Strain–energy curves

Stability indicators

2. Structural Descriptors
PXRD peak‑based descriptors (ANN‑ready)

Radial distribution functions (300‑dimensional)

Voronoi coordination statistics

Symmetry + space group

Lattice parameters + density

3. Visualization Tools
XRD patterns

RDF curves

Coordination histograms

Energetic comparison plots

Multi‑panel structural analysis figures

📊 Machine‑Learning Descriptor Generation
This repository converts structural data into ML‑ready numerical vectors:

PXRD → ANN Descriptor
Compute PXRD pattern

Extract top N peaks

Encode as:

Code
[2θ1, I1, 2θ2, I2, ..., 2θN, IN]
Produces a fixed‑length 2N‑dimensional vector

RDF Descriptor
300‑bin RDF fingerprint

Captures short‑ and medium‑range order

Coordination Descriptor
Distribution of coordination numbers

Local bonding environment

Symmetry Descriptor
Space group

Crystal system

Lattice metrics

CHGNet Energetic Descriptor
Energy per atom

Force norm

Stress norm

🚀 Getting Started
Install dependencies
bash
pip install pymatgen ase matminer chgnet matplotlib numpy pandas
Run the analysis
bash
python scripts/81_interatomic_potential_analysis_2109450.py
Generate RDFs
bash
python scripts/77_rdf_hypothetical_all.py
Create comparison plots
bash
python scripts/78_rdf_comparison_plot.py
Generate the full analysis panel
bash
python scripts/79_structure_analysis_panel.py
📘 Citation
If you use this repository, please cite:

CHGNet:
Deng et al., “CHGNet: A Universal Neural Network Interatomic Potential for Crystalline Solids,” 2023.

This repository:
Interatomic‑Potential‑Driven Structural and Energetic Analysis of Multicomponent Metal–Anion Crystalline Compounds Using CHGNet and Machine‑Learned Descriptors
