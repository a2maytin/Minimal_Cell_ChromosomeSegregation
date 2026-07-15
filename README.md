# Bare-bones energy minimization 

In /shell_scripts/, there is a `run.sh` which
1) Runs sc_chain_gen to get starting (jagged) coordinates
2) Runs a bare bones btree_chromo script which loads in the sc_chain_gen coordinates (and does whatever other commands are necessary to make btree_chromo happy), outputs the LAMMPS system data file pre-minimization, runs an energy minimization, and outputs the LAMMPS system data file post-minimization.

**The LAMMPS data files pre and post minimization, as well the log file, are in the /data/ folder**.

The outputs here should be useful reference for those who would like to **run, or generate one's own, LAMMPS input scripts and data files** without the need for calling btree_chromo as a "wrapper".
