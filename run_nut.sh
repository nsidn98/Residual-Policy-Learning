#!/bin/bash

# Slurm sbatch options
#SBATCH -o run.sh.log-%j
#SBATCH -n 16
#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda/2020a 
module load mpi/openmpi-4.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gridsan/sidnayak/.mujoco/mujoco200/bin

# Run the script        
# script to iterate through different hyperparameters
envs="NutAssembly"
seeds=0
exp_names="rl"
# Run the script
mkdir "${exp_names}_${envs}_${seed}"
# echo "${exp_names}_${envs}_${seeds}"
mpirun python -m RL.ddpg_mpi --env_name ${envs} --seed ${seeds} --n_epochs 2000  --exp_name ${exp_names} --actor_lr=0.0001 2>&1 | tee ${exp_names}_${envs}/out_${envs}_${seeds}