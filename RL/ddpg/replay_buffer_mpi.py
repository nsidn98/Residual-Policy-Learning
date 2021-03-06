"""
    Replay buffer for storing transitions
    Code taken from OpenAI baselines code
"""
import numpy as np
import threading
from typing import Dict

class replay_buffer:
    def __init__(self, env_params:Dict, buffer_size:int, sample_func):
        """
            Replay Buffer to store transitions and goals for HER
            Parameters:
            -----------
            env_params: Dict
                {
                    'obs': obs['observation'].shape[0],
                    'goal': obs['desired_goal'].shape[0],
                    'action': env.action_space.shape[0],
                    'action_max': env.action_space.high[0],
                    'max_timesteps': env._max_episode_steps
                }
            buffer_size: int
                Maximum size of the buffer
            sample_func: sampling function for transitions
        """
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch:list):
        """
            Store the transitions
            Parameters:
            -----------
            episode_batch: list of numpy arrays
                [obs, ag, g, actions]
        """
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        """
            Sample transitions using self.sample_func
        """
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        """
            Get the indices to store at in the buffer
        """
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx