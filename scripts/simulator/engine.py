"""
Previous file that did most of the simulation

Should delete soon! Will all be transferred to physics engine.
Should turn into physics engine with one important function:

buggy_step(buggy : Buggy, dt : float) -> None:
    Updates buggy state by a certain timestep

"""

#! /usr/bin/env python3
from util.buggy import Buggy
from stanley_controller import StanleyController
from rl_controller import RL_Controller
import threading
import time
from collections import deque
import numpy as np
import utm

class Simulator():
    def __init__(self, starting_buggy: Buggy, controller_name="stanley"):
        super().__init__('engine')

        self.buggy = starting_buggy
        self.rate = 100  # Hz

    def dynamics(self, state, v):
        l = self.buggy.wheelbase
        _, _, theta, delta = state

        return np.array([v * np.cos(theta),
                         v * np.sin(theta),
                         v / l * np.tan(delta),
                         0])

    # updates buggy state based on physics and some action (transition function)
    def buggy_step(self, steering_angle):
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

    def loop(self):
        for _ in range(500):
            time.sleep(0.01)
            steering_angle = self.controller.compute_control()
            sim.buggy_step()

def main(controller_name="stanley"):
    buggy = Buggy()
    sim = Simulator(buggy, controller_name)
    sim.loop()

if __name__ == "__main__":
    main()
