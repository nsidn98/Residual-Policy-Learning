import torch
import gym
import numpy as np

# from RL.models import actor
from config import args

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.
"""

# define the actor network
class actor(nn.Module):
    def __init__(self, args, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# process the inputs
def process_inputs_mpi(o, g, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

def process_inputs(o, g, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    inputs = np.concatenate([o_clip, g_clip])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    import os
    # load weights
    # weight_path = 'wandb/run-20201108_210931-3jb4qca6/files/model.ckpt'
    weight_path = 'wandb/offline-run-20201115_145020-27v13tns/files/model.ckpt'
    if os.path.exists(weight_path):
        # load the model checkpoints
        ckpt = torch.load(weight_path)
    else:
        print('Path to weights does not exist. Download it from wandb')
    args_model = ckpt['args']   # load the arguments used to train that particular model
    args.mpi = args_model['mpi']
    args.env_name = args_model['env_name']
    args.clip_range = args_model['clip_range']
    args.clip_obs = args_model['clip_obs']
    #################################################################
    # check if MPI was used or not and accordingly set preprocess() method
    mpi_mode = args.mpi    # whether MPI was used to train the agent
    if mpi_mode:
        preprocess = process_inputs_mpi
        o_mean, o_std, g_mean, g_std = ckpt['normalizer_feats']
    else:
        preprocess = process_inputs
    #################################################################
    # create the environment
    env = gym.make(args.env_name)
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                    'goal': observation['desired_goal'].shape[0], 
                    'action': env.action_space.shape[0], 
                    'action_max': env.action_space.high[0],
                    }
    #################################################################
    # create the actor network and load weights in the network
    actor_network = actor(args, env_params)
    actor_network.load_state_dict(ckpt['actor_state_dict'])
    actor_network.eval()
    #################################################################
    for i in range(10):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            inputs = preprocess(obs, g, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))