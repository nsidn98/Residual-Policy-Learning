# Residual-Policy-Learning

6.881 Robotic Manipulation Course Project, MIT Fall 2020 - Implementation of Residual Policy Learning

## Usage:

### First install the requirements using:

`pip install -r requirements.txt` or `pip3 install -r requirements.txt`

### Make a wandb account:

All the experiments will be logged with a library called "[Wandb](https://www.wandb.com/)". So make sure that you first make an account in wandb [here](https://app.wandb.ai/login?signup=true) and login your terminal using `wandb login`

### Non-MPI version:
`python RL/ddpg/ddpg.py --n_cycles=10 --env_name='FetchReach'`

### MPI version:
`mpirun -np 1 python -u RL/ddpg/ddpg_mpi.py --env_name='FetchReach' --n_cycles=10`

### SAC
`python RL/sac/sac.py --env_name="robosuiteNutAssemblyDense`

## TODO:
* Improve README with more plots and project information.
* Add information about all files.
* Make SAC modular.
* Add report, video links.
* If possible add wandb report and link to network weights.

