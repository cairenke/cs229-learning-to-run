from osim.env.run import RunEnv
import opensim
import math
from itertools import chain


def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

class EnrichedRunEnv(RunEnv):
    STATE_PELVIS_V_X = 4
    STATE_PELVIS_V_Y = 5
    STATE_HIP_R_A = 6
    STATE_HIP_R_V_A = 12
    STATE_HEAD_X = 22
    STATE_HEAD_Y = 23
    STATE_HEAD_V_X = 24
    STATE_HEAD_V_Y = 25
    STATE_TORSO_X = 26
    STATE_TORSO_Y = 27
    STATE_TORSO_V_X = 47
    STATE_TORSO_V_Y = 48

    ninput = 41 + 6 + 10  # 61

    def __init__(self, visualize=True, max_obstacles=3, reward_type=0):
        RunEnv.__init__(self, visualize, max_obstacles)
        self.reward_type = reward_type
        self.last_position = 0
        self.current_position = 0

    def compute_reward(self):
        # Compute ligaments penalty
        lig_pen = 0
        # Get ligaments
        for j in range(20, 26):
            lig = opensim.CoordinateLimitForce.safeDownCast(self.osim_model.forceSet.get(j))
            lig_pen += lig.calcLimitForce(self.osim_model.state) ** 2

        if self.reward_type == 0:
            reward = self.current_position - self.last_position
        elif self.reward_type == 1:
            # use velocity
            reward = self.current_state[self.STATE_PELVIS_V_X] * 0.01
            reward += 0.01  # small reward for still standing
            reward += min(0, self.current_state[self.STATE_HEAD_X]) * 0.01  # penalty for head behind pelvis
        elif self.reward_type == 2:
            reward = self.current_position - self.last_position
        else:
            v_x = self.current_state[self.STATE_PELVIS_V_X]
            v_y = self.current_state[self.STATE_PELVIS_V_Y]
            y = self.current_state[self.STATE_PELVIS_Y]
            reward = min(v_x, 4) - 0.005 * (v_x * v_x + v_y * v_y) - 0.05 * y * y + 0.02

        if self.reward_type == 2:
            return 10 * (reward - math.sqrt(lig_pen) * 10e-8)
        else:
            return reward - math.sqrt(lig_pen) * 10e-8

    def process_observation(self, input):
        _stepsize = 0.01

        output = list(input)
        pr = output[0]
        pvr = output[3]
        px = output[self.STATE_PELVIS_X]
        py = output[self.STATE_PELVIS_Y]
        pvx = output[self.STATE_PELVIS_V_X]
        pvy = output[self.STATE_PELVIS_V_Y]

        for i in range(6):
            output[self.STATE_HIP_R_A + i] -= pr
            output[self.STATE_HIP_R_V_A + i] -= pvr

        # 'head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r'
        # change x and y relative to pelvis
        for i in range(7):
            output[self.STATE_HEAD_X + i * 2 + 0] -= px
            output[self.STATE_HEAD_X + i * 2 + 1] -= py

        # compute body parts velocity information
        output = output + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if self.istep == 0:
            output[self.STATE_HEAD_V_X] = 0
            output[self.STATE_HEAD_V_Y] = 0
        else:
            # Head velocity
            output[self.STATE_HEAD_V_X] = (output[self.STATE_HEAD_X] - self.last_state[self.STATE_HEAD_X]) / _stepsize
            output[self.STATE_HEAD_V_Y] = (output[self.STATE_HEAD_Y] - self.last_state[self.STATE_HEAD_Y]) / _stepsize

            # print('speed vx {0} {1}  curr {2} last {3}'.format(output[self.STATE_HEAD_V_X], output[self.STATE_HEAD_V_Y], output[23], self.last_state[23]))

            for i in range(5):
                offset = i * 2
                output[self.STATE_TORSO_V_X + offset] = (output[self.STATE_TORSO_X + offset] - self.last_state[
                    self.STATE_TORSO_X + offset]) / _stepsize
                output[self.STATE_TORSO_V_Y + offset] = (output[self.STATE_TORSO_Y + offset] - self.last_state[
                    self.STATE_TORSO_Y + offset]) / _stepsize

        # mass
        output[18] -= px  # mass pos xy made relative
        output[19] -= py
        output[20] -= pvx  # mass vel xy made relative
        output[21] -= pvy

        # set pelvis x position to 0
        output[1] = 0

        return output

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

        while counter < limit:
            ret.append([0, 0, 0])
            counter += 1

        return ret

    def get_observation(self):

        bodies = ['head', 'pelvis', 'torso', 'toes_l', 'toes_r', 'talus_l', 'talus_r']

        pelvis_pos = [self.pelvis.getCoordinate(i).getValue(self.osim_model.state) for i in range(3)]
        pelvis_vel = [self.pelvis.getCoordinate(i).getSpeedValue(self.osim_model.state) for i in range(3)]

        if self.istep == 0:
            self.last_position = 0
            self.current_position = pelvis_pos[1]
        else:
            self.last_position = self.current_position
            self.current_position = pelvis_pos[1]

        jnts = ['hip_r', 'knee_r', 'ankle_r', 'hip_l', 'knee_l', 'ankle_l']
        joint_angles = [self.osim_model.get_joint(jnts[i]).getCoordinate().getValue(self.osim_model.state) for i in
                        range(6)]
        joint_vel = [self.osim_model.get_joint(jnts[i]).getCoordinate().getSpeedValue(self.osim_model.state) for i in
                     range(6)]

        mass_pos = [self.osim_model.model.calcMassCenterPosition(self.osim_model.state)[i] for i in range(2)]
        mass_vel = [self.osim_model.model.calcMassCenterVelocity(self.osim_model.state)[i] for i in range(2)]

        body_transforms = [
            [self.osim_model.get_body(body).getTransformInGround(self.osim_model.state).p()[i] for i in range(2)] for
            body in bodies]

        muscles = [self.env_desc['muscles'][self.MUSCLES_PSOAS_L], self.env_desc['muscles'][self.MUSCLES_PSOAS_R]]

        # see the next obstacle
        obstacle = list(flatten(self.find_obstacles(pelvis_pos[1], 3)))

        # print('obstacle {0}'.format(obstacle))

        # feet = [opensim.HuntCrossleyForce.safeDownCast(self.osim_model.forceSet.get(j)) for j in range(20,22)]
        info = pelvis_pos + pelvis_vel + joint_angles + joint_vel + mass_pos + mass_vel + list(
            flatten(body_transforms)) + muscles + obstacle
        self.current_state = self.process_observation(info)

        # print('step {0} head pos is {1} {2}  pelvis {3}'.format(self.istep, self.current_state[self.STATE_HEAD_X], self.current_state[self.STATE_HEAD_Y], self.current_state[self.STATE_PELVIS_Y]))
        # print(self.current_state)
        return self.current_state
