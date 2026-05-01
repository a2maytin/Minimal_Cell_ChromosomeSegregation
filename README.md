# Protein Science Code (Public)

This repository contains the scripts and analysis code used for the protein science simulations and figures.

Simulation runs in this project call `btree-chromo-dev` via a containerized workflow.

Cell growth behavior in the model is based on Thornburg et al. (2026).

## Repository contents

- `scripts/`: simulation input templates and Python run drivers
- `shell_scripts/`: job-submission and run orchestration scripts
- `analysis/`: analysis notebooks and helper scripts
- `data/`: static/reference input data required by runs

Large generated run outputs are intentionally excluded from this public package.
