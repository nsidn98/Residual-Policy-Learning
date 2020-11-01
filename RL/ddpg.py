"""
    DDPG with HER
"""
import gym
import argparse
import torch
from torch import nn
from torch import optim
import os
from datetime import datetime
import numpy as np

from RL.models import actor, critic
from RL.replay_buffer import replay_buffer
from her.her import her_sampler

OPTIMIZERS = {
    'adam': optim.Adam,
    'adamax': optim.Adamax,
    'rmsprop': optim.RMSprop,
}

LOSS_FN = {
    'mse': nn.MSELoss(),
    'smooth_l1': nn.SmoothL1Loss(),
    'l1': nn.L1Loss()
}

class OUNoise(object):
    """
        Ornstein Uhlenbeck Noise for DDPG
        https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process
        dx = theta * (mu - x) + sigma * random()
        x <- x + dx
        sigma = max_sigma - (max_sigma - min_sigma) * min(1, t/decay)
        at =  at + x    # with clipping
    """
    def __init__(self, action_space:gym.spaces.box.Box, args:argparse.Namespace, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        """
            Parameters:
            -----------
            action_space : gym.spaces.box.Box
                action space for the environment
            args : argparse.Namespace
                args should contain the following
                OU_mu : float
                    mean for OU Noise
                OU_theta : float
                    theta for OU Noise
                OU_max_sigma : float
                    maximum value for sigma in OU Noise
                OU_min_sigma : float
                    minimum value for sigma in OU Noise
                OU_decay_period : int
                    number of timesteps to decay sigma from max to min
        """
        self.mu           = args.OU_mu
        self.theta        = args.OU_theta
        self.sigma        = args.OU_max_sigma
        self.max_sigma    = args.OU_max_sigma
        self.min_sigma    = args.OU_min_sigma
        self.decay_period = args.OU_decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class DDPG_Agent:
    def __init__(self, args:argparse.Namespace, env, env_params:Dict, device:str, writer=None):
        """
            Module for the DDPG agent along with HER
            Parameters:
            -----------
            args: argparse.Namespace
                args should contain the following:
            env: gym.Env
                OpenAI type gym environment
            device: str
                device to run the training process on
                Choices: 'cpu', 'cuda'
            writer: tensorboardX
                tensorboardX to log metrics like losses, rewards, success_rates, etc.
        """
        self.args = args
        self.env = env
        self.env_params = env_params
        self.device = device
        self.writer = writer
        self.train_steps = 0   # a count to keep track of number of training steps

        # create the network
        self.actor_network = actor(args, env_params).to(device)
        self.critic_network = critic(args, env_params).to(device)
        # build up the target network
        self.actor_target_network = actor(args, env_params).to(device)
        self.critic_target_network = critic(args, env_params).to(device)

        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        self.actor_optim  = OPTIMIZERS[args.actor_optim](self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = OPTIMIZERS[args.critic_optim](self.critic_network.parameters(), lr=self.args.lr_critic)

        # loss function for DDPG
        self.criterion = LOSS_FN[args.loss_fn]

        # her sampler
        self.her_module = her_sampler(replay_stategy = self.args.replay_strategy, \
                                      replay_k = self.args.replay_k, \
                                      reward_func = self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, \
                                    self.args.buffer_size, \
                                    self.her_module.sample_her_transitions)
        

    def learn(self):
        """
            Run the episodes for training
        """
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):

                ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                # reset the environment
                observation = self.env.reset()
                obs = observation['observation']
                ag = observation['achieved_goal']
                g = observation['desired_goal']

                for t in range(self.env_params['max_timesteps']):
                    # take actions 
                    # NOTE/TODO: add controller here OR make a wrapper OR add it in self.select_actions()
                    with torch.no_grad():
                        state = self.preprocess_inputs(obs, g)
                        pi = self.actor_network(input_tensor)
                        action = self.select_actions(pi)
                    # give the action to the environment
                    observation_new, _, _, info = self.env.step(action)
                    obs_new = observation_new['observation']
                    ag_new = observation_new['achieved_goal']

                    # append rollouts
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())
                    ep_actions.append(action.copy())
                    # re-assign the observation
                    obs = obs_new
                    ag = ag_new
                # convert to np arrays
                ep_obs = np.array(ep_obs)
                ep_ag = np.array(ep_ag)
                ep_g = np.array(ep_g)
                ep_actions = np.array(ep_actions)

                # store them in buffer
                self.buffer.store_episode([ep_obs, ep_ag, ep_g, ep_actions])

                for batch in range(self.args.n_batches):
                    # train the network with 'n_batches' number of batches
                    self.update_network(self.train_steps)
                    self.train_steps += 1
                self.polyak_update_networks(self.actor_target_network, self.actor_network)
                self.polyak_update_networks(self.critic_target_network, self.critic_network)
            # evaluate the agent
            success_rate = self.eval_agent()
            # TODO @nsidn98 save weights and log success rates

    def preprocess_inputs(self, obs:np.ndarray, g:np.ndarray):
        """
            Concatenate state and goal
            and convert them to torch tensors
            and then transfer them to either CPU of GPU
        """
        # concatenate the stuffs
        inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        inputs = inputs.to(self.device)
        return inputs
    
    def select_actions(self, pi: torch.Tensor):
        """
            Take action
            with a probability of self.args.random_eps, it will take random actions
            otherwise, this will add a gaussian noise to the action along with clipping
        """
        # transfer action from CUDA to CPU is using GPU and make numpy array out of it
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])

        # random actions
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose whether to take random actions or not
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    def preprocess_og(self, o:np.ndarray, g:np.ndarray):
        """
            Perform observation clipping
        """
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def polyak_update_networks(self, target, source):
        """
            Polyak averaging of target and main networks; Also known as soft update of networks
            target_net_params = (1 - polyak) * main_net_params + polyak * target_net_params
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def update_network(self, step:int):
        """
            The actual DDPG training
        """

        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self.preprocess_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self.preprocess_og(o_next, g)

        # concatenate obs and goal
        states = np.concatenate([transitions['obs'], transitions['g']], axis=1)
        next_states = np.concatenate([transitions['obs_next'], transitions['g_next']], axis=1)

        # convert to tensor
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(transitions['actions'], dtype=torch.float32).to(self.device)
        rewards = torch.tensor(transitions['r'], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            actions_next = self.actor_target_network(next_states)
            q_next_value = self.critic_target_network(next_states, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = rewards + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the returns
            clip_return = 1 / (1-self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        
        # critic loss
        q_value = self.critic_network(states, actions)
        critic_loss = self.criterion(target_q_value, q_value)   # loss (mostly MSE)

        # actor loss
        actions_pred = self.actor_network(states)
        actor_loss = -self.critic_network(states, actions_pred).mean()
        actor_loss = actor_loss + self.args.action_l2 * (actions_pred / self.env_params['action_max']).pow(2).mean()

        if self.writer:
            writer.add_scalar('losses/actor_loss', actor_loss.item(), step)
            writer.add_scalar('losses/critic_loss', critic_loss.item(), step)

        # backpropagate
        self.actor_optim.zero_grad()    # zero the gradients
        actor_loss.backward()           # backward prop
        self.actor_optim.step()         # take step towards gradient direction

        self.critic_optim.zero_grad()    # zero the gradients
        critic_loss.backward()           # backward prop
        self.critic_optim.step()         # take step towards gradient directions
        
    def eval_agent(self):
        """
            Evaluate the agent using the trained policy
            performs n_test_rollouts in the environment
            and returns
        """
        successes = []
        for _ in range(self.args.n_test_rollouts):
            success = np.zeros(self.env_params['max_timesteps'])
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for i in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self.preprocess_inputs(obs,g)
                    pi = self.actor_network(input_tensor)
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                success[i] = info['is_success']
            successes.append(success)
        successes = np.array(successes)
        return np.mean(successes[:,-1]) # return mean of only final steps success

    def save_checkpoint(self, path:str):
        """
            Saves the model in the wandb experiment run directory
            This will store the 
                * model state_dict
                * optimizer state_dict
                * args/hparams
            param:
                path: str
                    path to the wandb run directory
                    Example: os.path.join(wandb.run.dir, "model.ckpt")
        """
        checkpoint = {}
        checkpoint['args'] = vars(self.args)
        checkpoint['actor_state_dict'] = self.actor_network.state_dict()
        checkpoint['critic_state_dict'] = self.critic_network.state_dict()
        checkpoint['actor_optimizer_dict'] = self.actor_optim.state_dict()
        checkpoint['critic_optimizer_dict'] = self.critic_optim.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path:str):
        """
            Load the trained model weights
            param:
                path: str
                    path to the saved weights file
        """
        use_cuda = torch.cuda.is_available()
        device   = torch.device("cuda" if use_cuda else "cpu")
        checkpoint_dict = torch.load(path, map_location=device)
        self.actor_network.load_state_dict(checkpoint_dict['actor_state_dict'])
