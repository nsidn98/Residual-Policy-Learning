"""
    Networks for actor and critic
    NOTE: Input should be [observation, goal]
"""
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTS = {
    'relu': F.relu,
    'sigmoid':torch.sigmoid,
    'tanh':torch.tanh,
}

class actor(nn.Module):
    """
            Actor Network
            Parameters:
            ----------
            args: argparse.Namespace
                Should at least have the following arguments:
                hidden_dims: list
                    List of hidden layer dimensions
                activation: str
                    Which activation function to use
                    Choices: 'relu', 'sigmoid', 'tanh'
            env_params: Dict
                a dictionary containing the following:
                {
                    'obs': obs['observation'].shape[0],
                    'goal': obs['desired_goal'].shape[0],
                    'action': env.action_space.shape[0],
                    'action_max': env.action_space.high[0],
                    'max_timesteps': env._max_episode_steps
                }
        """
    def __init__(self, args:argparse.Namespace, env_params:Dict):
        super(actor, self).__init__()

        hidden_dims = args.hidden_dims
        hidden_dims = [env_params['obs'] + env_params['goal']] + hidden_dims
        self.activation = ACTS[args.activation]
        self.max_action = env_params['action_max']
        self.layers = []
        for i, j in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(i,j))    # append linear layers
        self.action_out = nn.Linear(hidden_dims[-1], env_params['action'])

    def forward(self, x:torch.Tensor):
        for layer in self.layers:
            x = self.activation(layer(x))
        actions = self.max_action * torch.tanh(self.action_out(x))
        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        """
            Critic Network
            Parameters:
            ----------
            args: argparse.Namespace
                Should at least have the following arguments:
                hidden_dims: list
                    List of hidden layer dimensions
                activation: str
                    Which activation function to use
                    Choices: 'relu', 'sigmoid', 'tanh'
            env_params: Dict
                a dictionary containing the following:
                {
                    'obs': obs['observation'].shape[0],
                    'goal': obs['desired_goal'].shape[0],
                    'action': env.action_space.shape[0],
                    'action_max': env.action_space.high[0],
                    'max_timesteps': env._max_episode_steps
                }
            NOTE: Normalisation of actions happens in self.forward()
        """
        super(critic, self).__init__()

        hidden_dims = args.hidden_dims
        hidden_dims = [env_params['obs'] + env_params['goal']] + hidden_dims
        self.activation = ACTS[args.activation]
        self.max_action = env_params['action_max']
        self.layers = []
        for i, j in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers.append(nn.Linear(i,j))    # append linear layers
        self.q_out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x:torch.Tensor, actions:torch.Tensor):

        x = torch.cat([x, actions / self.max_action], dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        q_value = self.q_out(x)
        return q_value