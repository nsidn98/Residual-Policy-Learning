import os
import math
import numpy as np
import itertools

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sac_config import args
from RL.sac_replay_buffer import replay_buffer
from RL.sac_models import GaussianActor, DeterministicActor, Critic


class SAC:
    def __init__(self, args:argparse.Namespace, env, save_dir:str, writer=None):
        """
            Module for the SAC agent
            Parameters:
            -----------
            args: argparse.Namespace
                args should contain the following:
            env: gym.Env
                OpenAI type gym environment
            save_dir: str
                Path to save the network weights, checkpoints
            device: str
                device to run the training process on
                Choices: 'cpu', 'cuda'
            writer: tensorboardX
                tensorboardX to log metrics like losses, rewards, success_rates, etc.
        """
        self.args = args
        self.env = env
        self.env_params = get_env_params(env)
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = Critic(self.env_params['obs'], self.env_params['action'], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(self.env_params['obs'], self.env_params['action'], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.actor = GaussianActor(self.env_params['obs'], self.env_params['action'], args.hidden_size, self.env_params['action_space']).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicActor(self.env_params['obs'], self.env_params['action'], args.hidden_size, self.env_params['action_space']).to(self.device)
            self.actor_optim = Adam(self.actor.parameters(), lr=args.lr)

        self.burn_in_done = False   # to check if burn-in is done or not
        self.buffer = replay_buffer(self.args.buffer_size, self.args.seed)

    def get_env_params(self, env):
        """
            Get the environment parameters
        """
        obs = env.reset()
        # close the environment
        params = {'obs': obs['observation'].shape[0],
                'action': env.action_space.shape[0],
                'action_space': env.action_space,
                }
        try:
            params['max_timesteps'] = env._max_episode_steps
        # for custom envs
        except:
            params['max_timesteps'] = env.max_episode_steps
        return params

    def train(self):
        # Training Loop
        total_numsteps = 0
        updates = 0
        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()

            while not done:
                if self.args.start > total_numsteps:
                    # TODO add random actions depending on res/rl
                    pass
                else:
                    action = self.select_action(state)  # sample action policy

                if len(self.buffer) > self.args.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        critic_1_loss, critic_2_loss, actor_loss, ent_loss, alpha = self.update_parameters(self.buffer, self.args.batch_size, updates)

                        if self.writer:
                            self.writer.add_scalar('Loss/Critic_1', critic_1_loss, updates)
                            self.writer.add_scalar('Loss/Critic_2', critic_2_loss, updates)
                            self.writer.add_scalar('Loss/Actor', policy_loss, updates)
                            self.writer.add_scalar('Loss/Entropy Loss', ent_loss, updates)
                            self.writer.add_scalar('Entropy Temprature/Alpha', alpha, updates)
                        updates += 1
                # TODO add info success rate metric
                next_state, reward, done, info = self.env.step(action)
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward

                # ignore the done signal if we hit time horizon
                # Refer: # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.env_params['max_timesteps'] else float(not done)

                self.buffer.push(state, action, reward, next_state, mask) # Append transition to memory
                state = next_state

            if total_numsteps > self.args.num_step:
                break

            if self.writer:
                self.writer.add_scalar('reward/train', episode_reward, i_episode)
            print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {episode_reward:.3f}")

            if i_episode % self.args.eval_freq == 0:
                avg_reward = 0
                for _ in range(self.args.num_eval_episodes):
                    state = self.env.reset()
                    episode_reward = 0
                    done = False
                    for not done:
                        action = self.select_action(state, evaluate=True)
                        next_state, reward, done, info = self.env.step(action)
                        # TODO add info success rate metric
                        episode_reward += reward

                        state = next_state
                    avg_reward += episode_reward
                avg_reward /= self.args.num_eval_episodes

                if self.writer:
                    self.writer.add_scalar('Avg Reward/Test', avg_reward, i_episode)

                print("----------------------------------------")
                print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
                print("----------------------------------------")

    def select_action(self, state, evaluate=False):
        # TODO  change here
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(state)
        else:
            _, _, action = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size:int, updates:int):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
    
    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
