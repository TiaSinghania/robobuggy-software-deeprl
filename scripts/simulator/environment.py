"""
Gymansium Environment API
Check the documentation: https://gymnasium.farama.org/introduction/create_custom_env/

Controls two buggies:
SC - (policy controlled) : Buggy
NAND - Classical Control (Stanley) : Buggy
"""
import numpy as np
import gymnasium as gym
from util.buggy import Buggy, BuggyObs

class BuggyCourse(gym.Env):
    def __init__(self):
        # HILL1_NAND from old sim code
        self.starting_utm = (589760.46, 4477322.07)
        self.starting_theta = -110

        # NAND_WHEELBASE = 1.3
        # SC_WHEELBASE = 1.104

        self.buggy = Buggy(n_utm=self.starting_utm[0], e_utm=self.starting_utm[1], speed=0, theta=self.starting_theta, wheelbase=1.3)

    def _get_obs(self):
        return BuggyObs(self.buggy.n_utm, self.buggy.e_utm, self.buggy.speed, self.buggy.theta, self.buggy.wheelbase, self.buggy.angle_clip)

    def reset(self):
        self.buggy = Buggy(n_utm=self.starting_utm[0], e_utm=self.starting_utm[1], speed=0, theta=self.starting_theta, wheelbase=1.3)

    def dynamics(self, state, v):
        l = self.buggy.wheelbase
        _, _, theta, delta = state

        return np.array([v * np.cos(theta),
                         v * np.sin(theta),
                         v / l * np.tan(delta),
                         0])

    def step(self, steering_angle):
        heading = self.buggy.theta
        e_utm = self.buggy.e_utm
        n_utm = self.buggy.n_utm
        velocity = self.buggy.speed

        h = 1/self.rate
        state = np.array([e_utm, n_utm, np.deg2rad(heading), np.deg2rad(steering_angle)])
        k1 = self.dynamics(state, velocity)
        k2 = self.dynamics(state + h/2 * k1, velocity)
        k3 = self.dynamics(state + h/2 * k2, velocity)
        k4 = self.dynamics(state + h/2 * k3, velocity)

        final_state = state + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        e_utm_new, n_utm_new, heading_new, _ = final_state
        heading_new = np.rad2deg(heading_new)

        self.buggy.e_utm = e_utm_new
        self.buggy.n_utm = n_utm_new
        self.buggy.theta = heading_new
