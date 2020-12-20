"""
    Robosuite environment for the peg in hole task
    NOTE: Under contruction
    TODO: add documentation
"""
import gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
import imageio

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import robosuite.utils.transform_utils as T


"""
[0] JOINT_VELOCITY - Joint Velocity
[1] JOINT_TORQUE - Joint Torque
[2] JOINT_POSITION - Joint Position
[3] OSC_POSITION - Operational Space Control (Position Only)
[4] OSC_POSE - Operational Space Control (Position + Orientation)
[5] IK_POSE - Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)
"""


class keypointOptPegInHole(gym.Env):

    def __init__(self, *args, **kwargs):
        options = {}
        controller_name = 'OSC_POSE'
        options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)
        # options["controller_configs"]['control_delta'] = False
        # options["controller_configs"]['control_ori'] = True
        # options["controller_configs"]['uncouple_pos_ori'] = False

        self.cameraName = "frontview"
        skip_frame = 2
        self.peg_env = suite.make(
            "TwoArmPegInHole",
            robots=["IIWA","IIWA"],
            **options,
            has_renderer=False,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=self.cameraName,
            camera_heights=512,
            camera_widths=512,
        )
        posTol = 0.001
        rotTol = 0.001

        observation = self.peg_env.reset()

        # apparently observation["peg_to_hole"] is the vector FROM the hole TO the peg
        peg_pos0 = observation["hole_pos"] + observation["peg_to_hole"]
        self.peg_pos0 = peg_pos0
        self.hole_pos0 = observation["hole_pos"]

        # qPegRelRob0 = T.quat_multiply(observation["peg_quat"],T.quat_conjugate(observation["robot0_eef_quat"]))

        # positions of robots 0 and 1 rel peg and hole, in peg and hole frames.  should be constant forever.
        pRob0RelPeg = np.matmul(T.quat2mat(T.quat_inverse(observation["peg_quat"])) ,  observation["robot0_eef_pos"] - (peg_pos0))
        pRob1RelHole = np.matmul(T.quat2mat(T.quat_inverse(observation["hole_quat"])) ,  observation["robot1_eef_pos"] - observation["hole_pos"])
        # qHoleRelRob1 = T.quat_multiply(observation["hole_quat"],T.quat_conjugate(observation["robot1_eef_quat"]))

        # pRob0RelPeg = observation["robot0_eef_pos"] - peg_pos0
        # pRob1RelHole = observation["robot1_eef_pos"] - observation["hole_pos"]
        qRob0RelPeg = T.quat_multiply(T.quat_inverse(observation["robot0_eef_quat"]),observation["peg_quat"])
        qRob1RelHole = T.quat_multiply(T.quat_inverse(observation["robot1_eef_quat"]),observation["hole_quat"])

        model = self.peg_env.model.get_model()
        pegDim = model.geom_size[15]
        rPeg = pegDim[0]
        lPeg = pegDim[2]

        # define 3 keypoints: peg higher than hole, peg centered above hole with hole facing up, and peg in hole.
        nonlinear_constraint_1 = NonlinearConstraint(self.cons_1, lPeg, np.inf, jac='2-point', hess=BFGS())
        nonlinear_constraint_2 = NonlinearConstraint(self.cons_2, np.array([-posTol, -posTol, lPeg,-rotTol,-rotTol,-np.inf]), np.array([posTol,posTol,np.inf,rotTol,rotTol,np.inf]), jac='2-point', hess=BFGS())
        nonlinear_constraint_3 = NonlinearConstraint(self.cons_3, np.array([-posTol, -posTol, lPeg,-rotTol,-rotTol,-np.inf,-rotTol,-rotTol,-np.inf]), np.array([posTol,posTol,np.inf,rotTol,rotTol,np.inf,rotTol,rotTol,np.inf]), jac='2-point', hess=BFGS())
        nonlinear_constraint_4 = NonlinearConstraint(self.cons_unit_quat, np.array([1,1,1,1,1,1]), np.array([1,1,1,1,1,1]), jac='2-point', hess=BFGS())
        x0 = np.tile(np.hstack((peg_pos0,observation["hole_pos"],observation["peg_quat"],observation["hole_quat"])),3)

        res = minimize(self.traj_obj, x0, method='trust-constr', jac='2-point', hess=BFGS(),
                   constraints=[nonlinear_constraint_1, nonlinear_constraint_2,nonlinear_constraint_3,nonlinear_constraint_4],
                   options={'verbose': 1})

        x = res.x
        print(x0)
        print("x")
        print(x)
        # extract optimization results
        # x = [p_peg_1, p_hole_1, q_peg_1, q_hole_1, p_peg_2, ... q_peg_3, q_hole_3]
        ind_offset_1 = 14
        ind_offset_2 = 28

        p_peg_1 = x[0:3]
        p_hole_1 = x[3:6]
        p_peg_2 = x[ind_offset_1 + 0:ind_offset_1 + 3]
        p_hole_2 = x[ind_offset_1 + 3:ind_offset_1 + 6]
        p_peg_3 = x[ind_offset_2 + 0:ind_offset_2 + 3]
        p_hole_3 = x[ind_offset_2 + 3:ind_offset_2 + 6]

        q_peg_1 = x[6:10]
        q_hole_1 = x[10:14]
        q_peg_2 = x[ind_offset_1 + 6:ind_offset_1 + 10]
        q_hole_2 = x[ind_offset_1 + 10:ind_offset_1 + 14]
        q_peg_3 = x[ind_offset_2 + 6:ind_offset_2 + 10]
        q_hole_3 = x[ind_offset_2 + 10:ind_offset_2 + 14]


        # robot rel world = robot rel peg * peg rel world
        q_rob0_1 = T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_1))
        q_rob1_1 = T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_1))
        q_rob0_2 = T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_2))
        q_rob1_2 = T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_2))
        q_rob0_3 = T.quat_multiply(qRob0RelPeg,T.quat_inverse(q_peg_3))
        q_rob1_3 = T.quat_multiply(qRob1RelHole,T.quat_inverse(q_hole_3))

        # rob rel world = peg rel world + robot rel peg
        # robot rel peg in world frame = (q world frame rel peg frame) * (robot rel peg in peg frame)

        print("p robot 1 rel hole in hole frame:")
        print(pRob1RelHole)
        print("p robot 0 rel peg in peg frame:")
        print(pRob0RelPeg)

        self.p_rob0_1 = p_peg_1 + np.matmul(T.quat2mat(q_peg_1),pRob0RelPeg)
        self.p_rob1_1 = p_hole_1 + np.matmul(T.quat2mat(q_hole_1),pRob1RelHole)
        self.p_rob0_2 = p_peg_2 + np.matmul(T.quat2mat(q_peg_2),pRob0RelPeg)
        self.p_rob1_2 = p_hole_2 + np.matmul(T.quat2mat(q_hole_2),pRob1RelHole)
        self.p_rob0_3 = p_peg_3 + np.matmul(T.quat2mat(q_peg_3),pRob0RelPeg)
        self.p_rob1_3 = p_hole_3 + np.matmul(T.quat2mat(q_hole_3),pRob1RelHole)

        self.axang_rob0_1 = T.quat2axisangle(q_rob0_1)
        self.axang_rob1_1 = T.quat2axisangle(q_rob1_1)
        self.axang_rob0_2 = T.quat2axisangle(q_rob0_2)
        self.axang_rob1_2 = T.quat2axisangle(q_rob1_2)
        self.axang_rob0_3 = T.quat2axisangle(q_rob0_3)
        self.axang_rob1_3 = T.quat2axisangle(q_rob1_3)

        self.max_episode_steps = 200
        self.kpp = 3
        self.kpr = 0.1
        # self.action_space = self.peg_env.action_space
        # self.observation_space = self.peg_env.observation_space

        print(self.p_rob0_1)

    def reset(self):
        self.poseNum = 0
    def controller(self,obs:dict):



        posePosTol = 0.001
        poseAxangTol = 0.01

        if self.poseNum == 0:
            # posActionRob0 = self.kpp*(self.p_rob0_1 - obs["robot0_eef_pos"])
            # axangActionRob0 = self.kpr*(self.axang_rob0_1 - T.quat2axisangle(obs["robot0_eef_quat"]))
            # posActionRob1 = self.kpp*(self.p_rob1_1 - obs["robot1_eef_pos"])
            # axangActionRob1 = self.kpr*(self.axang_rob1_1 - T.quat2axisangle(obs["robot1_eef_quat"]))

            # q goal rel current = q goal rel world * inverse(q current rel world)
            # not totally sure I flipped the following lines correctly when flipping everything after realizing quats are inverted
            qDeltaRob0 = T.quat_multiply(T.quat_inverse(obs["robot0_eef_quat"]),T.axisangle2quat(self.axang_rob0_1))
            qDeltaRob1 = T.quat_multiply(T.quat_inverse(obs["robot1_eef_quat"]),T.axisangle2quat(self.axang_rob1_1))
            # delta axis angle rotation, in current frame
            axAngDelta0 = T.quat2axisangle(qDeltaRob0)
            axAngDelta1 = T.quat2axisangle(qDeltaRob1)

            # rotate axis angle to world frame for command
            axangActionRob0 = self.kpr*(np.matmul(T.quat2mat(obs["robot0_eef_quat"]) ,axAngDelta0))
            axangActionRob1 = self.kpr*(np.matmul(T.quat2mat(obs["robot1_eef_quat"]) ,axAngDelta1))

            posActionRob0 = self.kpp*(self.p_rob0_1 - obs["robot0_eef_pos"])
            posActionRob1 = self.kpp*(self.p_rob1_1 - obs["robot1_eef_pos"])

            # # position delta in current frame, for IK control
            # posActionRob0 = self.kpp*(np.matmul(T.quat2mat(obs["robot0_eef_quat"]) ,self.p_rob0_1 - obs["robot0_eef_pos"]))
            # posActionRob1 = self.kpp*(np.matmul(T.quat2mat(obs["robot0_eef_quat"]) ,self.p_rob1_1 - obs["robot1_eef_pos"]))

            # for absolute control:
            # posActionRob0 = self.p_rob0_1
            # posActionRob1 = self.p_rob1_1
            # axangActionRob0 = self.axang_rob0_1
            # axangActionRob1 = self.axang_rob1_1

            if np.linalg.norm(posActionRob0/self.kpp)<posePosTol and np.linalg.norm(axangActionRob0/self.kpr)<poseAxangTol and np.linalg.norm(posActionRob1/self.kpp)<posePosTol and np.linalg.norm(axangActionRob1/self.kpr)<poseAxangTol:
                print("here")
                self.poseNum = 1

        # elif self.poseNum == 1:
        #     skdj
        # elif self.poseNum == 2:
        #     slkfd
        # elif self.poseNum == 3:
        #     jkj

        # return (np.hstack((posActionRob0,axangActionRob0)).tolist(),np.hstack((posActionRob1,axangActionRob1)).tolist())
        return np.hstack((posActionRob0,axangActionRob0,posActionRob1,axangActionRob1)).tolist()
        # return np.hstack((posActionRob0,axangActionRob0,np.array([0,0,0,0,np.pi,0]))).tolist()

    def cons_1(self,x):
        # first pose: constrain peg further than lPeg in z direction relative to hole
        p_peg = x[0:3]
        p_hole = x[3:6]
        q_peg = x[6:10]
        q_hole = x[10:14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)


        return p_peg_in_hole_frame[2]

    def cons_2(self,x):
        # second  pose: constrain peg further than lPeg in z direction relative to hole
        # also constrain peg x and y in hole frame to be below a tolerance
        # also constrain rotation error
        ind_offset = 14 # ind at which to start looking for pose 2 info
        p_peg = x[ind_offset + 0:ind_offset + 3]
        p_hole = x[ind_offset + 3:ind_offset + 6]
        q_peg = x[ind_offset + 6:ind_offset + 10]
        q_hole = x[ind_offset + 10:ind_offset + 14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)

        q_error = T.get_orientation_error(q_hole,q_peg)

        return np.hstack((p_peg_in_hole_frame,q_error))

    def cons_3(self,x):
        # third  pose: constrain peg less than lPeg/2 in z direction relative to hole
        # also constrain peg x and y in hole frame to be below a tolerance
        # also constrain same orientations as in pose 2
        last_ind_offset = 14 # ind at which to start looking for pose 2 info
        ind_offset = 28 # ind at which to start looking for pose 3 info
        p_peg = x[ind_offset + 0:ind_offset + 3]
        p_hole = x[ind_offset + 3:ind_offset + 6]
        q_peg = x[ind_offset + 6:ind_offset + 10]
        q_hole = x[ind_offset + 10:ind_offset + 14]
        p_peg_in_hole_frame = np.matmul(T.quat2mat(T.quat_inverse(q_hole)),p_peg - p_hole)

        q_error_peg_3_2 = T.get_orientation_error(q_peg , x[last_ind_offset + 6:last_ind_offset + 10])
        q_error_hole_3_2 = T.get_orientation_error(q_hole, x[last_ind_offset + 10:last_ind_offset + 14])

        return np.hstack((p_peg_in_hole_frame,q_error_peg_3_2,q_error_hole_3_2))

    def cons_unit_quat(self,x):
        # constrain quaternions to be unit
        ind_offset_1 = 14
        ind_offset_2 = 28

        q_peg_1 = x[6:10]
        q_hole_1 = x[10:14]
        q_peg_2 = x[ind_offset_1 + 6:ind_offset_1 + 10]
        q_hole_2 = x[ind_offset_1 + 10:ind_offset_1 + 14]
        q_peg_3 = x[ind_offset_2 + 6:ind_offset_2 + 10]
        q_hole_3 = x[ind_offset_2 + 10:ind_offset_2 + 14]

        return np.array([np.linalg.norm(q_peg_1), np.linalg.norm(q_hole_1), np.linalg.norm(q_peg_2), np.linalg.norm(q_hole_2), np.linalg.norm(q_peg_3), np.linalg.norm(q_hole_3)])

    def traj_obj(self,x):
        peg_pos0 = self.peg_pos0
        hole_pos0 = self.hole_pos0
        ind_offset_1 = 14
        ind_offset_2 = 28
        p_peg_1 = x[0:3]
        p_hole_1 = x[3:6]
        p_peg_2 = x[ind_offset_1 + 0:ind_offset_1 + 3]
        p_hole_2 = x[ind_offset_1 + 3:ind_offset_1 + 6]
        p_peg_3 = x[ind_offset_2 + 0:ind_offset_2 + 3]
        p_hole_3 = x[ind_offset_2 + 3:ind_offset_2 + 6]
        # cost on motion between adjacent poses
        return np.linalg.norm(p_peg_1 - peg_pos0) + np.linalg.norm(p_peg_2 - p_peg_1) + np.linalg.norm(p_peg_3 - p_peg_2) + np.linalg.norm(p_hole_1 - hole_pos0) + np.linalg.norm(p_hole_2 - p_hole_1) + np.linalg.norm(p_hole_3 - p_hole_2)

    # def seed(self, seed=0):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return self.peg_env.seed(seed=seed)

if __name__ == "__main__":
    env_name = 'keypointOptPegInHole'
    env = globals()[env_name]()
    successes = []

    # env.seed(1)
    # env.action_space.seed(1)
    # env.observation_space.seed(1)
    failed_eps = []
    writer = imageio.get_writer('video/TwoArmPegInHole.mp4', fps=20)
    frames = []
    skip_frame = 2
    for i_episode in range(1):
        success = np.zeros(env.max_episode_steps)
        obs = env.reset()
        # print(obs.keys())
        action = ([0,0,0,0,0,0],[0,0,0,0,0,0])  # give zero action at first time step
        for t in range(100):
            action = env.controller(env.peg_env._get_observation())
            observation, reward, done, info = env.peg_env.step(action)
            # print(np.matmul(T.quat2mat(T.quat_inverse(observation["hole_quat"])) ,  observation["robot1_eef_pos"] - observation["hole_pos"]))
            if t % skip_frame == 0:
                frame = observation[env.cameraName + "_image"][::-1]
                writer.append_data(frame)

            if reward == 1:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        print("final hole pos")
        print(observation["hole_pos"])
    env.peg_env.close()
    writer.close()
