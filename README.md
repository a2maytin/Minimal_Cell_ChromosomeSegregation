# Chromosome Segregation in a Minimal Bacterial Cell Driven by SMC Protein Complexes

This repository contains the scripts and analysis code used for the simulations and figures for DOI: 10.1002/pro.70604.

Simulation runs in this project call `btree-chromo-dev` via a Docker container (Dockerfile included here).

Cell growth behavior in the model is based on DOI: 10.1016/j.cell.2026.02.009 (Thornburg et al. 2026).

Initial chromosome configuration is built using sc_chain_generation (Gilbert et al. 2023).

## Repository contents

- `scripts/`: simulation input templates and run drivers
- `shell_scripts/`: job-submission and batch drivers
- `analysis/`: analysis and plotting notebooks and scripts
- `data/`: various input data required by runs

Run outputs used for the study are excluded from this repository, but a selection of run outputs are available on Zenodo at: PLACEHOLDER.
