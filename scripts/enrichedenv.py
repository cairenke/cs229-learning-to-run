from osim.env.run import RunEnv
import opensim
import math
import numpy as np
import os
import random
import string
from itertools import chain

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

class EnrichedRunEnv(RunEnv):
    STATE_PELVIS_V_X = 4
    STATE_PELVIS_V_Y = 5

    ninput = 41 + 6 + 10 #61

    def __init__(self, visualize=True, max_obstacles=3, original_reward=False):
        RunEnv.__init__(self, visualize, max_obstacles)
        self.original_reward = original_reward

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        if self.original_reward:
            reward = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]
        else:
            v_x = self.current_state[self.STATE_PELVIS_V_X]
            v_y = self.current_state[self.STATE_PELVIS_V_Y]
            y = self.current_state[self.STATE_PELVIS_Y]

            # reward = self.current_state[self.STATE_PELVIS_X] - self.last_state[self.STATE_PELVIS_X]
            # print('original reward {0}'.format(reward))
            # print('speed x {0} combo {1} y square {2}'.format(abs(v_x), 0.005 * (v_x * v_x + v_y * v_y), 0.05 * y * y))

            reward = min(v_x, 4) - 0.005 * (v_x * v_x + v_y * v_y) - 0.05 * y * y + 0.02
            # print('final reward {0}'.format(reward))
            # reward += 0.01  # small reward for still standing
            # # use velocity
            # # reward = self.current_state[self.STATE_PELVIS_V_X] * 0.01
            reward += min(0, self.current_state[22]) * 0.01  # penalty for head behind pelvis
            # reward -= sum([max(0.0, k - 0.1) for k in
            #                [self.current_state[7], self.current_state[10]]]) * 0.02  # penalty for straight legs

        return reward - math.sqrt(lig_pen) * 10e-8

    def process_observation(self, input):
        _stepsize = 0.01

        o = list(input)  # an array
        '''
        observation:
        0 pelvis r
        1 x
        2 y
        3 pelvis vr
        4 vx
        5 vy
        '''
        pr = o[0]
        px = o[1]
        py = o[2]
        pvr = o[3]
        pvx = o[4]
        pvy = o[5]

        # 'hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l'
        # angle, and angle velocity
        # for i in range(6, 18):
        #     o[i] /= 4

        # append body parts velocity information
        o = o + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if self.istep == 0:
            o[24] = 0
            o[25] = 0
        else:
            # Head velocity
            o[24] = (o[22] - self.last_state[22]) / _stepsize - pvx
            o[25] = (o[23] - self.last_state[23]) / _stepsize - pvy

            for i in range(5):
                offset = i * 2 + 0
                o[47 + offset] = (o[26 + offset] - self.last_state[26 + offset]) / _stepsize - pvx
                offset += 1
                o[47 + offset] = (o[26 + offset] - self.last_state[26 + offset]) / _stepsize - pvy

        # change x and y relative to pelvis
        # 'head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r'
        for i in range(7):
            # skip pelvis
            if i == 1:
                continue
            o[22 + i * 2 + 0] -= px
            o[22 + i * 2 + 1] -= py

        # mass
        o[18] -= px  # mass pos xy made relative
        o[19] -= py
        o[20] -= pvx  # mass vel xy made relative
        o[21] -= pvy

        # o[0] /= 2  # divide pr by 4
        # o[1] = 0  # abs value of pel x should not be included
        # o[2] -= 0.9  # minus py by 0.5
        # o[3] /= 4  # divide pvr by 4
        # o[4] /= 8  # divide pvx by 10
        # o[5] /= 1  # pvy is okay

        return o


    def find_obstacles(self, x, limit):
        obstacles = self.env_desc['obstacles']
        counter = 0
        ret = []
        for obstacle in obstacles:
            if counter > limit:
                break

            info = list(obstacle)
            if (obstacle[0] + obstacle[2] * 2) < x:
                info[0] = 0
                info[1] = 0
                info[2] = 0
            else:
                info[0] -= x
                ret.append(info)
            counter += 1

        if counter < limit:
            ret.append([0, 0, 0])
            counter += 1

        return ret

    last_step = None
    last_body_transformation = None
    all_obstacles = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    def get_observation(self):

        bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        jnts = ['hip_r','knee_r','ankle_r','hip_l','knee_l','ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [[self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for body in bodies]

        muscles = [ self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R] ]

        # see the next obstacle
        obstacle = list(flatten(self.find_obstacles(pelvis_pos[1], 3)))

        # feet = [opensim.HuntCrossleyForce.safeDownCast(self.osim_model.forceSet.get(j)) for j in range(20,22)]
        info = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(flatten(body_transforms)) + muscles + obstacle

        self.current_state = self.process_observation(info)

        # print('len is {0} step {1}'.format(len(self.current_state), self.istep))
        # print(self.current_state)
        return self.current_state
