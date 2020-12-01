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

# change environment name here
env_names="FetchPush"
#env_names="FetchPushImperfect"
#env_names="FetchPushSlippery"
echo $env_names
# make folder for storing the log file
mkdir $env_names

# Run the script   
# this one is to check if residual learning happens by monitoring critic loss for burn-in
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=0 200 2>&1 | tee $env_names/out_critic_0
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=1 200 2>&1 | tee $env_names/out_critic_1
mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=2 200 2>&1 | tee $env_names/out_critic_2
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=3 200 2>&1 | tee $env_names/out_critic_3
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=4 200 2>&1 | tee $env_names/out_critic_4