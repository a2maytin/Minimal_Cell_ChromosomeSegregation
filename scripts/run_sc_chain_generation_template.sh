#!/bin/bash

cd "$(dirname "$0")"

# Set variables for scchainDirectory and working_directory
scchainDirectory="/Software/"
inputDirectory="./"
outputDirectory="../data/coords/"

# Define the input and log file paths
# input_fname="${inputDirectory}Syn3A_chromosome_init.inp"
input_fname="${inputDirectory}Syn3A_chromosome_init.inp"
cp $input_fname $outputDirectory
log_fname="${outputDirectory}log_init.log"

# Construct the DNA executable command
DNA_executable="${scchainDirectory}sc_chain_generation/src/gen_sc_chain --i_f=${input_fname} --o_d=${outputDirectory} --o_l=Syn3A_chromosome_init --s=10 --l=${log_fname} --n_t=8 --bin --xyz"

# Print the command to check it
echo "Executing command: $DNA_executable"

# Execute the command
$DNA_executable

# Use as initial dna monomer coordinates
cp ${outputDirectory}x_chain_Syn3A_chromosome_init_rep00001.bin ${outputDirectory}dna_{DATE}_{REPL}_0.bin
cp ${outputDirectory}x_obst_Syn3A_chromosome_init_rep00001.bin ${outputDirectory}ribo_{DATE}_{REPL}_0.bin

