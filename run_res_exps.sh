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
# script to check for zero actor lr
# this one is to check what value critic converges to with perfect controller, zero actor_lr
#mpirun python -m RL.ddpg_mpi --env_name='FetchPickAndPlacePerfect' --exp_name='res' --n_epochs 500 --actor_lr=0 --coin_flipping_prob=1 2>&1 | tee exp_res_outputs/out_0_1
# this one is to check what value critic converges to with perfect controller, zero actor_lr and some randomness in actor actions
#mpirun python -m RL.ddpg_mpi --env_name='FetchPickAndPlacePerfect' --exp_name='res' --n_epochs 500 --actor_lr=0 --coin_flipping_prob=0.5 2>&1 | tee exp_res_outputs/out_0_0.5
# this one is to check if residual learning happens by monitoring actor loss for burn-in
#mpirun python -m RL.ddpg_mpi --env_name='FetchPickAndPlacePerfect' --exp_name='res' --beta=0.005 --beta_monitor='actor' --n_epochs 500 2>&1 | tee exp_outputs/out_actor_0.005
# this one is to check if residual learning happens by monitoring critic loss for burn-in
#mpirun python -m RL.ddpg_mpi --env_name='FetchPickAndPlacePerfect' --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs 500 2>&1 | tee exp_outputs/out_critic_0.005
