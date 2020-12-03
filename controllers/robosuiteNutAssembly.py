import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config

import robosuite.utils.transform_utils as T
from math import *


"""
[0] JOINT_VELOCITY - Joint Velocity
[1] JOINT_TORQUE - Joint Torque
[2] JOINT_POSITION - Joint Position
[3] OSC_POSITION - Operational Space Control (Position Only)
[4] OSC_POSE - Operational Space Control (Position + Orientation)
[5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)
"""

"""
TO DOS:
1. Figure out whats going wrong with the z pose

"""
options = {}
controller_name = 'OSC_POSE'
options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

env = GymWrapper(
    suite.make(
        "NutAssemblyRound",                # pickPLaceMilk task
        robots="IIWA",                  # use IIWA robot
        **options,                      # controller options
        use_object_obs = True,
        use_camera_obs=False,           # do not use pixel observations
        has_offscreen_renderer=False,   # not needed since not using pixel obs
        has_renderer=True,              # make sure we can render to the screen
        reward_shaping=True,            # use dense rewards
        control_freq=20,                # control should happen fast enough so that simulation looks smooth
    )
)

def controller(obs:dict, nut_p, object_below_hand:bool=False, gripper_reoriented:int=0, object_in_hand:bool=False,):
    gripper_pos = obs['robot0_eef_pos']
    gripper_quat = obs['robot0_eef_quat']
    object_pos  = obs['RoundNut0_pos']
    object_quat = obs['RoundNut0_quat']
    z_table = 0.8610982

    object_axang = T.quat2axisangle(object_quat)
    gripper_axang = T.quat2axisangle(gripper_quat)

    if not object_below_hand or gripper_reoriented < 5:
        if not object_below_hand:
            action = 20 * (object_pos[:2] - gripper_pos[:2])
        else:
            action = [0,0]
        frac = -0.2
        ang_goal = frac*nut_p
        if gripper_reoriented < 5:
            action_angle= [0,0,ang_goal]
            gripper_reoriented+=1
        else:
            action_angle=[0,0,0]
        action = np.hstack((action, [0], action_angle, [-1]))
        if np.linalg.norm((object_pos[:2] - gripper_pos[:2])) < 0.01:
            object_below_hand = True

    elif not object_in_hand:
        action = [0,0,-1,0,0,0,-1]
        if np.linalg.norm((object_pos[2] - gripper_pos[2])) < 0.01:
            action = [0,0,0,0,0,0,1]
            object_in_hand = True
    
    else:
        action = [0,0,1,0,0,0,1]
        if object_pos[2] - z_table > 0.1:
            action = 20 * (goal_pos[:2] - object_pos[:2])
            action = np.hstack((action,[0,0,0,0,1]))
            if np.linalg.norm((goal_pos[:2] - object_pos[:2])) < 0.025:
                action = [0,0,0,0,0,0,-1]

    return action, object_below_hand, gripper_reoriented, object_in_hand

for i_episode in range(20):
    observation = env.reset()
    object_below_hand = False
    gripper_reoriented = 0
    object_in_hand = False
    peg2_pos = np.array(env.sim.data.body_xpos[env.peg2_body_id])
    goal_pos = peg2_pos
    #print(goal_pos)
    for t in range(100):
        env.render()
        if t == 0: # Record some initial info about the scene
            initial_obs = env.env._get_observation()
            initial_obj = initial_obs['RoundNut0_quat']
            nut_p = T.quat2axisangle(initial_obj)[-1]
        action, object_below_hand, gripper_reoriented, object_in_hand = controller(env.env._get_observation(), nut_p, object_below_hand, gripper_reoriented, object_in_hand)
        observation, reward, done, info = env.step(action)
        if reward == 1:
            print("Episode finished after {} timesteps".format(t + 1))
            break