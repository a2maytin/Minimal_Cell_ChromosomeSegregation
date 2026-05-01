# Chromosome Segregation in a Minimal Bacterial Cell Driven by SMC Protein Complexes

This repository contains the scripts and analysis code used for the simulations and figures for the article:

**Chromosome segregation in a minimal bacterial cell driven by SMC protein complexes (Protein Science) DOI: 10.1002/pro.70604.**

Simulation runs in this project call btree_chromo via a Docker container (Dockerfile and dependencies are available on Zenodo at: 10.5281/zenodo.19959420).

The code for the version of btree_chromo used for this study can be found at: [https://github.com/Luthey-Schulten-Lab/btree_chromo_gpu/tree/protein_science](https://github.com/Luthey-Schulten-Lab/btree_chromo_gpu/tree/protein_science).

Cell growth behavior in the model is based on the 4D whole-cell model (4DWCM) from DOI: 10.1016/j.cell.2026.02.009 (Thornburg et al. 2026). Code and representative set of runs from than model: [https://zenodo.org/records/15579159](https://zenodo.org/records/15579159)

Initial chromosome configuration is built using sc_chain_generation (Gilbert et al. 2023). Code: [https://github.com/Luthey-Schulten-Lab/sc_chain_generation](https://github.com/Luthey-Schulten-Lab/sc_chain_generation)

## Repository contents

- `LAMMPS_DNA_model_kk/`: various parameters used for calling LAMMPS 
- `scripts/`: simulation input templates and run drivers
- `shell_scripts/`: job-submission and batch drivers
- `analysis/`: analysis and plotting notebooks and scripts

Run outputs used for the study are excluded from this repository, but a selection of run outputs are available on Zenodo at: 10.5281/zenodo.19959420.
