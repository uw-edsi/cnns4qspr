#!/bin/bash
#SBATCH -J 1.0CPUscale
#SBATCH --account=pfaendtner
#SBATCH --partition=pfaendtner-gpu
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=02:10:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

source activate cnn4qspr
source ~/.login
echo $CUDA_VISIBLE_DEVICES
#echo $SLURM_LOCALID

srun ./improve_voxelize_ligand.py --input_dir="/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/test-set0/" --output_dir="/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/small_set/" --affinity_data="./affinity_data_cleaned0.csv"
#srun ./improve_voxelize_ligand.py --input_dir="/gscratch/pfaendtner/Chowdhury/PDBbind/split-general-set/collection$SLURM_LOCALID/" --output_dir="/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/test-output$SLURM_LOCALID/" --affinity_data="./affinity_data_cleaned$SLURM_LOCALID.csv"

#CUDA_VISIBLE_DEVICES=1 ./improve_voxelize_ligand.py --input_dir='/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/test-set1/' --output_dir='/gscratch/pfaendtner/Chowdhury/cnns4qspr/test_codes/test-output1/' &


echo "waiting for job to finish"
wait
