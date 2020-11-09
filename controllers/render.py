"""
    Testing video recording for rendering:
"""
import gym
import numpy as np
import time
from tqdm import tqdm
from typing import Dict


def pick_place_controller(obs:Dict, object_in_hand:bool):
    dist_threshold = 0.002
    height_threshold = 0.003
    kp = 2
    action_pos =  list(kp * obs['observation'][6:9])  # vector joining gripper and object 

    if not object_in_hand:
        # try to hover above the object
        if np.linalg.norm(action_pos[:2])>dist_threshold:
            action = action_pos[:2] + [0,0]
        # once above the object, move down while opening the gripper
        else:
            action = action_pos[:3] + [1]   # open the gripper
            # if we are close to the object close the gripper
            if np.linalg.norm(action_pos) < height_threshold:
                action = action_pos[:3] + [-1]  # close the gripper
                object_in_hand = True
    # once object is in hand, move towards goal
    else:
        p_rel = obs['desired_goal'] - obs['achieved_goal']
        action_pos =  list(kp * p_rel)
        action = action_pos + [-1]
    return action, object_in_hand


# time.sleep(5)

if __name__ == "__main__":
    env_name = 'FetchPickAndPlace-v1'
    # env_name = 'FetchSlide-v1'
    env = gym.make(env_name)
    # this will record video of the render and store it in video/ folder
    # NOTE this is the new stuff for rendering in @rhjiang's sytem
    env = gym.wrappers.Monitor(env, 'video/' + env_name, force=True)
    obs = env.reset()

    hand_above = False
    hand_behind = False
    hand_higher = False
    hand_down = False
    object_in_hand = False
    action = [0,0,0,0]   # give zero action at first time step
    # time.sleep(5)
    for i in tqdm(range(50)):
        env.render(mode='rgb_array')
        obs, rew, done, info = env.step(action)
        if env_name == 'FetchPickAndPlace-v1':
            action, object_in_hand = pick_place_controller(obs, object_in_hand)
        elif env_name == 'FetchSlide-v1':
            action, hand_higher, hand_above, hand_behind, hand_down = imperfect_slide_controller(obs, hand_higher, hand_above, hand_behind, hand_down)

    print('Done')
