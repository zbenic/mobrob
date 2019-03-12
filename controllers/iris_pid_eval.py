import argparse
import gym
import gymfc
# from .pid import PIDController
from gymfc.controllers.positionPID import PIDControllerPosition
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import math
import time

"""
This script evaluates a PID controller in the GymFC environment. This can be
used as a baseline for comparing to other control algorihtms in GymFC and also
to confirm the GymFC environment is setup and installed correctly.

The PID and mix settings reflect what was used in the following paper,

Koch, William, Renato Mancuso, Richard West, and Azer Bestavros. 
"Reinforcement Learning for UAV Attitude Control." arXiv 
preprint arXiv:1804.04154 (2018).

For reference, PID

PID Roll = [2, 10, 0.005]
PID PITCH = [10, 10, 0.005]
PID YAW = [4, 50, 0.0]

and mix for values throttle, roll, pitch, yaw,

rear right motor = [ 1.0, -1.0,  0.598, -1.0 ]  
front rear motor = [ 1.0, -0.927, -0.598,  1.0 ]
rear left motor  = [ 1.0,  1.0,  0.598,  1.0 ]
front left motor = [ 1.0,  0.927, -0.598, -1.0 ]

PID terms were found first using the Ziegler–Nichols method and then manually tuned
for increased response. The Iris quadcopter does not have an X frame therefore 
a custom mixer is required. Using the mesh files found in the Gazebo models they were
imported into a CAD program and the motor constraints were measured. Using these
values the mix calculater found here, https://www.iforce2d.net/mixercalc, was
used to derive the values. The implmementation of the PID controller can be found here,
https://github.com/ivmech/ivPID/blob/master/PID.py, windup has been removed so 
another variable was not introduced.
"""


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

class PIDPolicy(Policy):
    def __init__(self):
        self.r = [2, 10, 0.005]
        self.p = [10, 10, 0.005]
        self.y = [4, 50, 0.0]
        self.controller = PIDController(pid_roll = self.r, pid_pitch = self.p, pid_yaw =self.y )

    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):
        # Convert to degrees
        desired = list(map(math.degrees, desired))
        actual = list(map(math.degrees, actual))
        motor_values = np.array(self.controller.calculate_motor_values(sim_time, desired, actual))
        # Need to scale from 1000-2000 to -1:1
        return np.array( [ (m - 1000)/500  - 1 for m in motor_values])

    def reset(self):
        self.controller = PIDController(pid_roll = self.r, pid_pitch = self.p, pid_yaw = self.y )

def eval(env, pi):
    actuals = []
    desireds = []
    pi.reset()
    ob = env.reset()
    while True:
        desired = env.omega_target
        actual = env.omega_actual
        # PID only needs to calculate error between desired and actual y_e
        ac = pi.action(ob, env.sim_time, desired, actual)
        ob, reward, done, info = env.step(ac)
        actuals.append(actual)
        desireds.append(desired)
        if done:
            break
    env.close()
    return desireds, actuals

class PIDPolicyPosition(Policy):
    def __init__(self):
        self.x = [40, 5, 0.05]
        self.y = [40, 5, 0.05]
        self.h = [40, 5, 0.05]
        self.r = [45, 5, 0.5]
        self.p = [45, 5, 0.5]
        self.y = [40, 5, 0.05]
        self.controller = PIDControllerPosition(pid_x=self.x, pid_y=self.y, pid_z=self.h, pid_roll=self.r, pid_pitch=self.p, pid_yaw=self.y)

    def action(self, state, sim_time=0, desired=np.zeros(6), actual=np.zeros(6)):
        motor_values = np.array(self.controller.calculate_motor_values(sim_time, desired, actual))
        # Need to scale from 1000-2000 to -1:1
        # motor_values = [1600, 1600, 1600, 1600]
        return np.array( [ (m - 1000)/500  - 1 for m in motor_values])

    def reset(self):
        self.controller = PIDControllerPosition(pid_x=self.x, pid_y=self.y, pid_z=self.h, pid_roll=self.r, pid_pitch=self.p, pid_yaw=self.y)

def evalPosition(env, pi):
    actuals = []
    desireds = []
    pi.reset()
    ob = env.reset()
    while True:
        desired = env.position_target
        print(env.position_target[4])
        actual = env.quadState
        # PID only needs to calculate error between desired and actual y_e
        ac = pi.action(ob, env.sim_time, desired, actual)
        ob, reward, done, info = env.step(ac)
        actuals.append(actual)
        desireds.append(desired)
        if done:
            break
    env.close()
    return desireds, actuals

def main(env_id, seed):
    env = gym.make(env_id)
    env.render()
    time.sleep(5)
    rank = MPI.COMM_WORLD.Get_rank()
    workerseed = seed + 1000000 * rank
    env.seed(workerseed)
    pi = PIDPolicyPosition()
    desireds, actuals = evalPosition(env, pi)
    title = "PID Step Response in Environment {}".format(env_id)
    plot_step_response(np.array(desireds), np.array(actuals), title=title)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=17)

    args = parser.parse_args()

    main(args.env_id, args.seed)
