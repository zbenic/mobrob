import numpy as np
import logging

import math, random

from collections import deque

import torch
import torch.nn as nn
# import torch.optim as optim
import torch.autograd as autograd
# import torch.nn.functional as F


USE_CUDA = torch.cuda.is_available()
USE_CUDA = False  # TODO: Fix this to be able to use CUDA
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

    def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X


class DQNAgent(object):
    def __init__(self, env, num_inputs, num_actions, action_lim):
        super(DQNAgent, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_actions)
        )
        self.motorRangeMin = 1000
        self.motorRangeMax = 2000
        self.env = env
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim=num_actions)
        self.action_lim = action_lim


    def forward(self, state):
        return self.layers(state)

    def get_exploitation_action(self, state):
        """
        gets the action from target actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):
        """
        gets the action from actor added with exploration noise
        :param state: state (Numpy array)
        :return: sampled action (Numpy array)
        """
        state = Variable(torch.from_numpy(state).float())
        action = self.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def get_actions(self, state, epsilon):
        if random.random() > epsilon:
            action = self.get_exploitation_action(state)
        else:
            action = self.get_exploration_action(state)
        return action



    # def calculate_motor_values(self, state, epsilon):
    #     motorOutput = self.constrainf(motorOutput, self.motorRangeMin, self.motorRangeMax);
    #     if random.random() > epsilon:
    #         state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
    #         q_value = self.forward(state)
    #         action = q_value.max(1)[1].data[0]
    #     else:
    #         action = random.randrange(env.action_space.n)
    #     return action


    def constrainf(self, amt, low, high):
        # From BF src/main/common/maths.h
        if amt < low:
            return low
        elif amt > high:
            return high
        else:
            return amt


    def is_airmode_active(self):
        return True

    def reset(self):
        for pid in self.pid_rpy:
            pid.clear()
