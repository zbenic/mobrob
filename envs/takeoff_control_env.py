import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")


class TakeoffFlightControlEnv(GazeboEnv):
    def compute_reward(self):
        """ Compute the reward """
        return -np.clip(np.sum(np.abs(self.error))/(self.omega_bounds[1]*6), 0, 1)

    def sample_position_target(self):
        """ Sample a x, y, z position and roll, pitch and yaw angle """
        return  np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
class TakeoffTestFlightControlEnv(TakeoffFlightControlEnv):
    def __init__(self, world="takeoff-iris.world",
                 omega_bounds = [-math.pi, math.pi], 
                 max_sim_time = 2.9,
                 motor_count = 4, 
                 memory_size=1,): 
        
        self.omega_bounds = omega_bounds
        self.max_sim_time = max_sim_time
        self.memory_size = memory_size
        self.motor_count = motor_count
        self.observation_history = []
        super(TakeoffTestFlightControlEnv, self).__init__(motor_count=motor_count, world=world)
        self.position_target = self.sample_position_target()
        self.target_z = 1.0
        self.position_tolerance = 0.1
        self.velocity_tolerance = 0.1
        self.done = False
        self.last_target_distance = np.inf

    def step(self, action):
        # TODO: should self.done be reset to False?
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Step the sim
        self.quadState = self.step_sim(action)
        self.error = self.position_target - self.quadState  # TODO: Check this out further
        self.observation_history.append(np.concatenate([self.error]))
        # state = self.state()
        state = self.quadState
        reward = 0
        reward += self.compute_reward()
        if self.collision:
            reward += -100
            done = True
        elif self.sim_time >= self.max_sim_time:
            # reward += -50
            done = True
        elif np.linalg.norm(self.position_target[0:3] - self.quadState[0:3]) >= 5:
            reward += -10
            self.set_collision()
            done = True
        else:
            target_distance = np.linalg.norm(self.position_target[0:3] - self.quadState[0:3])
            velocity_distance = np.linalg.norm(self.position_target[6:9] - self.quadState[6:9])
            if target_distance <= self.position_tolerance and velocity_distance <= self.velocity_tolerance:  # agent has crossed the target height
                info = {"sim_time": self.sim_time, "target_xyz_pos": self.position_target,
                        "current_xyz_pos": self.position_actual}
                done = True
                return state, reward, done, info
            # elif self.sim_time > self.max_sim_time:
            #     reward += -50
            #     done = False
            else:
                done = False

        info = {"sim_time": self.sim_time, "target_xyz_pos": self.position_target, "current_xyz_pos": self.position_actual}

        return state, reward, done, info


    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      ((12 * self.memory_size) - memory.size, 0),
                      'constant', constant_values=(0))

    # def compute_reward(self):
    #     """ Compute the reward """
    #     target_distance = np.linalg.norm(self.position_target[0:3] - self.quadState[0:3])
    #     velocity_distance = np.linalg.norm(self.position_target[6:9] - self.quadState[6:9])
    #     distance_reward = 1 - np.tanh(target_distance) ** 0.5
    #     # distance_reward = 1 - (target_distance/5) ** 0.4
    #     velocity_discount = 1-np.maximum(velocity_distance, self.velocity_tolerance)**(1/(np.maximum(target_distance, self.position_tolerance)))
    #     reward = distance_reward * velocity_discount * 100
    #     if target_distance <= self.position_tolerance:  # agent has crossed the target height
    #         reward += 50.0  # bonus reward
    #         # print("*********** Position achieved! ***********")
    #         if velocity_distance <= self.velocity_tolerance:
    #             reward += 50.0  # bonus reward
    #             print("*********** Desired state achieved! ***********")
    #
    #     return reward

    def compute_reward(self):
        """ Compute the reward """
        target_distance = np.linalg.norm(self.position_target[0:3] - self.quadState[0:3])
        velocity_distance = np.linalg.norm(self.position_target[6:9] - self.quadState[6:9])
        reward = np.exp(-np.abs(target_distance - 0) / (0.1 * 5)) * np.exp(
            -np.abs(velocity_distance - 0) / (0.1 * 12))
        if self.last_target_distance <= target_distance:
            reward -= 5
        self.last_target_distance = target_distance
        if target_distance <= self.position_tolerance:  # agent has crossed the target height
            reward += 100.0  # bonus reward
            print("*********** Position achieved! ***********")
            if velocity_distance <= self.velocity_tolerance:
                reward += 200.0  # bonus reward
                print("*********** Desired state achieved! ***********")

        return reward


# class GyroErrorESCVelocityFeedbackEnv(GazeboEnv):
#     def __init__(self, world="attitude-iris.world",
#                  omega_bounds =[-math.pi, math.pi],
#                  max_sim_time = 1.,
#                  motor_count = 4,
#                  memory_size=1,):
#
#         self.omega_bounds = omega_bounds
#         self.max_sim_time = max_sim_time
#         self.memory_size = memory_size
#         self.motor_count = motor_count
#         self.observation_history = []
#         super(GyroErrorESCVelocityFeedbackEnv, self).__init__(motor_count = motor_count, world=world)
#         self.omega_target = self.sample_target()
#
#     def step(self, action):
#         action = np.clip(action, self.action_space.low, self.action_space.high)
#         # Step the sim
#         self.obs = self.step_sim(action)
#         self.error = self.omega_target - self.obs.angular_velocity_rpy
#         self.observation_history.append(np.concatenate([self.error, self.obs.motor_velocity]))
#         state = self.state()
#         done = self.sim_time >= self.max_sim_time
#         reward = self.compute_reward()
#         info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
#
#         return state, reward, done, info
#
#     def compute_reward(self):
#         """ Compute the reward """
#         return -np.clip(np.sum(np.abs(self.error))/(self.omega_bounds[1]*3), 0, 1)
#
#     def sample_target(self):
#         """ Sample a random angular velocity """
#         return  self.np_random.uniform(self.omega_bounds[0], self.omega_bounds[1], size=3)
#
#     def state(self):
#         """ Get the current state """
#         # The newest will be at the end of the array
#         memory = np.array(self.observation_history[-self.memory_size:])
#         return np.pad(memory.ravel(),
#                       (( (3+self.motor_count) * self.memory_size) - memory.size, 0),
#                       'constant', constant_values=(0))
#
#
# class GyroErrorESCVelocityFeedbackContinuousEnv(GyroErrorESCVelocityFeedbackEnv):
#     def __init__(self, command_time_off=[], command_time_on=[], **kwargs):
#         self.command_time_off = command_time_off
#         self.command_time_on = command_time_on
#         self.command_off_time = None
#         super(GyroErrorESCVelocityFeedbackContinuousEnv, self).__init__(**kwargs)
#
#     def step(self, action):
#         """ Sample a random angular velocity """
#         ret = super(GyroErrorESCVelocityFeedbackContinuousEnv, self).step(action)
#
#         # Update the target angular velocity
#         if not self.command_off_time:
#             self.command_off_time = self.np_random.uniform(*self.command_time_on)
#         elif self.sim_time >= self.command_off_time: # Issue new command
#             # Commands are executed as pulses, always returning to center
#             if (self.omega_target == np.zeros(3)).all():
#                 self.omega_target = self.sample_target()
#                 self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_on)
#             else:
#                 self.omega_target = np.zeros(3)
#                 self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_off)
#
#         return ret


