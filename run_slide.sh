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
env_names="FetchSlideSlapControl"
# env_names="FetchSlideFrictionControl"
# env_names="FetchSlide"
echo $env_names
# make folder for storing the log file
mkdir $env_names

# Run the script   
# this one is to check if residual learning happens by monitoring critic loss for burn-in
mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=0 500 2>&1 | tee $env_names/out_critic_0
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=1 500 2>&1 | tee $env_names/out_critic_1
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=2 500 2>&1 | tee $env_names/out_critic_2
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=3 500 2>&1 | tee $env_names/out_critic_3
# mpirun python -m RL.ddpg_mpi --env_name=$env_names --exp_name='res' --beta=0.005 --beta_monitor='critic' --n_epochs --seed=4 500 2>&1 | tee $env_names/out_critic_4