#!/bin/bash

./improve_voxelize_ligand.py --input_dir="/gscratch/pfaendtner/Chowdhury/PDBbind/split-general-set/collection$SLURM_LOCALID/" --output_dir="/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/test-output$SLURM_LOCALID/" --affinity_data="./affinity_data_cleaned$SLURM_LOCALID.csv" 
