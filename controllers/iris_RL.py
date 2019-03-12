import argparse
import gym
import gymfc
from gymfc.controllers.DQN import DQNAgent, ReplayBuffer, OrnsteinUhlenbeckActionNoise
import model
# from gymfc.controllers.positionPID import PIDControllerPosition
import matplotlib.pyplot as plt
import numpy as np
# from mpi4py import MPI
import math
import time
import torch


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MAX_EPISODES = 10
MAX_STEPS = 20 * 1000 #unit is step whose duratin is 1ms in real world
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: torch.autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else torch.autograd.Variable(*args, **kwargs)

def plot_step_response(desired, actual,
                 end=1., title=None,
                 step_size=0.001, threshold_percent=0.1):
    """
        Args:
            threshold (float): Percent of the start error
    """

    #actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    #desired = desired[:end]
    threshold = threshold_percent * desired

    plot_min = -math.radians(350)
    plot_max = math.radians(350)

    subplot_index = 3
    num_subplots = 6

    f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
    f.set_size_inches(10, 5)
    if title:
        plt.suptitle(title)
    ax[0].set_xlim([0, end_time])
    res_linewidth = 2
    linestyles = ["c", "m", "b", "g"]
    reflinestyle = "k--"
    error_linestyle = "r--"

    # Always
    ax[0].set_ylabel("X (m)")
    ax[1].set_ylabel("Y (m)")
    ax[2].set_ylabel("Z (m)")
    ax[3].set_ylabel("Roll (rad)")
    ax[4].set_ylabel("Pitch (rad)")
    ax[5].set_ylabel("Yaw (rad)")

    ax[-1].set_xlabel("Time (s)")


    """ X """
    # Highlight the starting x axis
    ax[0].axhline(0, color="#AAAAAA")
    ax[0].plot(t, desired[:, 0], reflinestyle)
    ax[0].plot(t, desired[:, 0] - threshold[:, 0], error_linestyle, alpha=0.5)
    ax[0].plot(t, desired[:, 0] + threshold[:, 0], error_linestyle, alpha=0.5)

    x = actual[:, 0]
    ax[0].plot(t[:len(x)], x, linewidth=res_linewidth)

    ax[0].grid(True)

    """ Y """
    # Highlight the starting x axis
    ax[1].axhline(0, color="#AAAAAA")
    ax[1].plot(t, desired[:, 1], reflinestyle)
    ax[1].plot(t, desired[:, 1] - threshold[:, 1], error_linestyle, alpha=0.5)
    ax[1].plot(t, desired[:, 1] + threshold[:, 1], error_linestyle, alpha=0.5)

    y = actual[:, 1]
    ax[1].plot(t[:len(y)], y, linewidth=res_linewidth)

    ax[1].grid(True)


    """ Z """
    # Highlight the starting x axis
    ax[2].axhline(0, color="#AAAAAA")
    ax[2].plot(t, desired[:, 2], reflinestyle)
    ax[2].plot(t, desired[:, 2] - threshold[:, 2], error_linestyle, alpha=0.5)
    ax[2].plot(t, desired[:, 2] + threshold[:, 2], error_linestyle, alpha=0.5)

    z = actual[:, 2]
    ax[2].plot(t[:len(z)], z, linewidth=res_linewidth)

    ax[2].grid(True)


    """ ROLL """
    # Highlight the starting x axis
    ax[3].axhline(0, color="#AAAAAA")
    ax[3].plot(t, desired[:,3], reflinestyle)
    ax[3].plot(t, desired[:,3] -  threshold[:,3] , error_linestyle, alpha=0.5)
    ax[3].plot(t, desired[:,3] +  threshold[:,3] , error_linestyle, alpha=0.5)
 
    r = actual[:,3]
    ax[3].plot(t[:len(r)], r, linewidth=res_linewidth)

    ax[3].grid(True)



    """ PITCH """

    ax[4].axhline(0, color="#AAAAAA")
    ax[4].plot(t, desired[:,4], reflinestyle)
    ax[4].plot(t, desired[:,4] -  threshold[:,4] , error_linestyle, alpha=0.5)
    ax[4].plot(t, desired[:,4] +  threshold[:,4] , error_linestyle, alpha=0.5)
    p = actual[:,1]
    ax[4].plot(t[:len(p)],p, linewidth=res_linewidth)
    ax[4].grid(True)


    """ YAW """
    ax[5].axhline(0, color="#AAAAAA")
    ax[5].plot(t, desired[:,5], reflinestyle)
    ax[5].plot(t, desired[:,5] -  threshold[:,5] , error_linestyle, alpha=0.5)
    ax[5].plot(t, desired[:,5] +  threshold[:,5] , error_linestyle, alpha=0.5)

    y = actual[:,5]
    ax[5].plot(t[:len(y)],y , linewidth=res_linewidth)
    ax[5].grid(True)

    plt.show()

class Policy(object):
    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):
        pass
    def reset(self):
        pass


class DQNPolicyPosition(Policy):
    def __init__(self, env, state_vector_size, action_num, action_limit, ram):
        """
        :param env: Gym environment
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param ram: replay memory buffer object
        :return:
        """
        self.env = env
        self.state_dim = state_vector_size
        self.action_dim = action_num
        self.action_lim = action_limit
        self.ram = ram
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = model.Critic(self.state_dim, self.action_dim)
        self.target_critic = model.Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)


        self.state_vector_size = state_vector_size
        self.action_num = action_num
        self.action_limit = action_limit
        self.controller = DQNAgent(env, state_vector_size, action_num, action_limit)

    def action(self, state, episode_step):
        epsilon = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * episode_step / EPS_DECAY)
        return self.controller.get_actions(state, epsilon)

    def reset(self):
        self.controller = DQNAgent(self.env, self.state_vector_size, self.action_num, self.action_limit)  # TODO: check this out

def evalPolicy(env, pi):
    quadStates = []
    pi.reset()
    for episode in range(0, MAX_EPISODES):
        ob = env.reset()
        quadState = env.get_quad_state()
        for episodeStep in range(0, MAX_STEPS):
            action = pi.action(quadState, episodeStep)
            ob, reward, done, info = env.step(action)
            quadStates.append(quadState)
            # TODO: optimizing step
        if done:
            break
    env.close()
    return quadStates

def main(env_id, seed):
    env = gym.make(env_id)

    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = env.action_space.high[0]

    env.render()
    time.sleep(5)

    # rank = MPI.COMM_WORLD.Get_rank()
    rank = 0
    workerseed = seed + 1000000 * rank
    env.seed(workerseed)
    pi = DQNPolicyPosition(env, S_DIM, A_DIM, A_MAX)
    actuals = evalPolicy(env, pi)
    title = "DQN Step Response in Environment {}".format(env_id)
    # plot_step_response(np.array(desireds), np.array(actuals), title=title)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=17)

    args = parser.parse_args()

    main(args.env_id, args.seed)
