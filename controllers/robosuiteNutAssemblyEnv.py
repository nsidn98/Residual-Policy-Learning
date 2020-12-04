import gym
from gym.utils import seeding
import numpy as np
from typing import Dict

# Additional libraries needed for robosuite
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

import robosuite.utils.transform_utils as T
from math import *

class NutAssembly(gym.Env):
    """
        NutAssembly:
        NutAssembly task from robosuite with no controller. Can be used for learning from scratch.
    """
    def __init__(self, *args, **kwargs):
        # self.fetch_env = gym.make('FetchPickAndPlace-v1')
        # self.metadata = self.fetch_env.metadata

        self.env = GymWrapper(
            suite.make(
                "NutAssemblyRound",             # Nut Assembly task with the round peg
                robots="IIWA",                  # use IIWA robot
                #**options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=True,              # make sure we can render to the screen
                reward_shaping=False,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )
        self.max_episode_steps = 100 #self.fetch_env._max_episode_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation = env.env._get_observation()
        observation['goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = env.env._get_observation()
        observation['goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        return observation

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.env.seed(seed=seed)

    def render(self, mode="human", *args, **kwargs):
        return self.env.render()

    def close(self):
        return self.=env.close()

    def compute_reward(self, *args, **kwargs):
        return self.env.reward(*args, **kwargs)

class NutAssemblyHand(gym.Env):
    """
        NutAssemblyHand:
            'FetchPickAndPlace-v1' with a perfect controller
            Action taken as:
                pi_theta(s) = pi(s) + f_theta(s)
        Parameters:
        -----------
        kp: float
            Scaling factor for position control
    """
    def __init__(self, kp:float=20, *args, **kwargs):
        #self.fetch_env = gym.make('FetchPickAndPlace-v1')
        #self.metadata = self.fetch_env.metadata

        options = {}
        controller_name = 'OSC_POSE'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

        self.env = GymWrapper(
            suite.make(
                "NutAssemblyRound",             # Nut Assembly task with the round peg
                robots="IIWA",                  # use IIWA robot
                **options,                      # controller options
                use_object_obs = True,
                use_camera_obs=False,           # do not use pixel observations
                has_offscreen_renderer=False,   # not needed since not using pixel obs
                has_renderer=True,              # make sure we can render to the screen
                reward_shaping=False,            # use dense rewards
                control_freq=20,                # control should happen fast enough so that simulation looks smooth
            )
        )

        self.max_episode_steps = 100 #self.fetch_env._max_episode_steps
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, residual_action:np.ndarray):
        controller_action = np.array(self.controller_action(self.last_observation))
        if (controller_action>1).any() or (controller_action<-1).any():
            print(controller_action)
        action = np.add(controller_action, residual_action)
        action = np.clip(action, -1, 1)
        observation, reward, done, info = self.env.step(action)
        observation = env.env._get_observation()
        observation['goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        self.last_observation = observation.copy()
        return observation, reward, done, info

    def reset(self):
        self.env.reset() # reset according to task defaults
        observation = env.env._get_observation()
        observation['goal'] = np.array(self.env.sim.data.body_xpos[self.env.peg2_body_id])
        self.last_observation = observation.copy()
        self.object_in_hand = False
        self.object_below_hand = False
        self.gripper_reoriented = 0
        return observation

    def seed(self, seed:int=0):
        self.np_random, seed = seeding.np_random(seed)
        return self.env.seed(seed=seed)

    def render(self, mode:str="human", *args, **kwargs):
        return self.env.render()

    def close(self):
        return self.env.close()

    def compute_reward(self, *args, **kwargs):
        return self.env.reward(*args, **kwargs)

    def controller_action(self, obs:dict, take_action:bool=True, DEBUG:bool=False):
        gripper_pos = obs['robot0_eef_pos']
        gripper_quat = obs['robot0_eef_quat']
        object_pos  = obs['RoundNut0_pos']
        object_quat = obs['RoundNut0_quat']
        goal_pos = obs['goal']
        z_table = 0.8610982

        object_axang = T.quat2axisangle(object_quat)
        gripper_axang = T.quat2axisangle(gripper_quat)

        if not self.object_below_hand or self.gripper_reoriented < 5:
            self.nut_p = T.quat2axisangle(object_quat)[-1]
            if not self.object_below_hand:
                action = 20 * (object_pos[:2] - gripper_pos[:2])
            else:
                action = [0,0]
            frac = 0.2 # Rate @ which to rotate gripper about z. Negative because z axes of object and gripper are antiparallel
            ang_goal = frac*self.nut_p # Nut p is the nut's random intial pertubation about z.
            if self.gripper_reoriented < 5: # Gripper should be aligned with nut after 5 action steps
                action_angle= [0,0,ang_goal]
                self.gripper_reoriented+=1
            else: # After 5 action steps, don't rotate gripper any more
                action_angle=[0,0,0]
            action = np.hstack((action, [0], action_angle, [-1]))
            if np.linalg.norm((object_pos[:2] - gripper_pos[:2])) < 0.01:
                self.object_below_hand = True

        elif not self.object_in_hand: # Close gripper
            action = [0,0,-1,0,0,0,-1]
            if np.linalg.norm((object_pos[2] - gripper_pos[2])) < 0.01:
                print('in here too')
                action = [0,0,0,0,0,0,1]
                self.object_in_hand = True
        
        else: # Move gripper up and toward goal position
            action = [0,0,1,0,0,0,1]
            if object_pos[2] - z_table > 0.1:
                action = 20 * (goal_pos[:2] - object_pos[:2])
                action = np.hstack((action,[0,0,0,0,1]))
                if np.linalg.norm((goal_pos[:2] - object_pos[:2])) < 0.025:
                    action = [0,0,0,0,0,0,-1] # Drop nut once it's close enough to the peg

        action = np.clip(action, -1, 1)
        return action

if __name__ == "__main__":
    env_name = 'NutAssemblyHand'
    env = globals()[env_name]() # this will initialise the class as per the string env_name
    # env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    successes = []
    # set the seeds
    env.seed(1)
    env.action_space.seed(1)
    env.observation_space.seed(1)
    failed_eps = []
    for ep in range(10):
        success = np.zeros(env.max_episode_steps)
        # print('_'*50)
        obs = env.reset()
        action = [0,0,0,0,0,0,0]  # give zero action at first time step
        for i in (range(env.max_episode_steps)):
            env.render()
            obs, rew, done, info = env.step(action)
            print(rew)
            success[i] = rew#info['is_success']
        ep_success = rew#info['is_success']
        if not ep_success:
            failed_eps.append(ep)
        successes.append(ep_success)
        print('this is successes ' + str(successes))
        # print(f'Episode:{ep} Success:{success}')
    print(f'Success Rate:{sum(successes)/len(successes)}')
    print(f'Failed Episodes:{failed_eps}')
    env.close()
