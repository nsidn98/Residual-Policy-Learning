#!/bin/bash

# Slurm sbatch options
# add -g volta:1 -s 20 for 1 GPU
#SBATCH -o run.sh.log-%j

# Loading the required module
source /etc/profile
module load anaconda/2020a 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/gridsan/sidnayak/.mujoco/mujoco200/bin

# Run the script        
# script to iterate through different envs
# Get the baselines for just pure control
envs=("FetchSlideFrictionControl" "FetchSlideImperfectControl" "FetchSlideSlapControl" "FetchPickAndPlacePerfect" "FetchPickAndPlaceSticky" "FetchPushImperfect" "FetchPushSlippery")
mkdir baseline_outputs
# Run the script
for j in ${!envs[@]}; do
    echo "Env Name: ${envs[$j]} | baseline_outputs/out_${envs[$j]}"
    python -m RL.baseline --exp_name 'baseline' --env_name ${envs[$j]} --n_epochs 500 2>&1 | tee baseline_outputs/out_${envs[$j]}
done
