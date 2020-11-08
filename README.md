# Residual-Policy-Learning
Implementation of Residual Policy Learning

## Contents
* `controllers/basic_controller.py`: Sample controllers for 'FetchPickAndPlace-v1' and 'FetchSlide-v1'
* `her/her.py`: Sampling class for Hindsight Experience Replay(HER)
* `mpi/mpi_utils.py`: Utility methods if we want to use [MPI](https://mpi4py.readthedocs.io/en/stable/) (Don't need to worry about this, it is just an implementational thing to improve efficiency)
* `mpi/normalizer.py`: A normalizer for observations and goals if we use MPI. (Don't need to worry about this, it is just an implementational thing to improve efficiency)
* `RL/models.py`: Contains the description for the actor and critic neural networks.
* `RL/replay_buffer.py`: A repplay buffer for storing the `< state, goal, achieved_goal, action >` transitions.
* `RL/replay_buffer_mpi.py`: Same as `RL/replay_buffer.py` but with some modifications to accomodate for the MPI version.
* `RL/ddpg.py`: The DDPG agent and the file to run to train a DDPG+HER agent from scratch on the environment.
* `RL/ddpg_mpi.py`: Same as `RL/ddpg.py` but with some modifications to accomodate for the MPI version.
* `RL/test.py`: Once training is done, use this to load the weights and test the agent by rendering the environment.
* `config.py`: The configuration paramters used in `RL/ddpg.py` and `RL/ddpg_mpi.py`

## Usage:

### First install the requirements using:

`pip install -r requirements.txt` or `pip3 install -r requirements.txt`

### Make a wandb account:

All the experiments will be logged with a library called "[Wandb](https://www.wandb.com/)". So make sure that you first make an account in wandb [here](https://app.wandb.ai/login?signup=true) and login your terminal using `wandb login`

### Non-MPI version:
`python RL/ddpg.py --n_cycles=10 --env_name='FetchReach-v1'`

### MPI version:
`mpirun -np 1 python -u RL/ddpg_mpi.py --env_name='FetchReach-v1' --n_cycles=10`

## Modifications we could try:
* Sensor noise: add some noise/bias to the observations
* Imperfect control: have an imperfect controller
* Make modifications in the environment. Like friction, obstacles etc.

