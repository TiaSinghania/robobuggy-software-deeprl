from dataclasses import dataclass

import numpy as np

"""
Dataclass representing a buggy, mainly should just be holding buggy attributes

State Vector:
e_utm - position easting (utm)
n_utm - position northing (utm)
speed - speed (m/s)
theta - heading (degrees)

Constants:
wheelbase (length from center of buggy to front wheel) (m)
angle_clip - the max/min value that each buggy can steer (degrees)

"""


@dataclass
class Buggy:
    # Buggy State
    e_utm: float  # m
    n_utm: float  # m
    speed: float  # m/s
    theta: float  # rad

    # Buggy Constants
    wheelbase: float  # m
    angle_clip: float = np.pi / 9  # rad

    # Buggy Control
    delta: float = 0  # rad

    def get_state(self) -> np.ndarray:
        return np.array([self.e_utm, self.n_utm, self.speed, self.theta]).reshape(-1)

    def get_control(self) -> np.ndarray:
        return np.array([self.delta]).reshape(-1)

    def get_constants(self) -> np.ndarray:
        return np.array([self.wheelbase, self.angle_clip]).reshape(-1)

    def set_state(self, state: np.ndarray):
        assert state.shape == (4,)
        self.e_utm = state[0]
        self.n_utm = state[1]
        self.speed = state[2]
        self.theta = state[3]

    def get_self_obs(self) -> np.ndarray:
        return np.array([self.e_utm, self.n_utm, self.speed, self.theta, self.delta])

    def get_other_obs(self) -> np.ndarray:
        return np.array([self.e_utm, self.n_utm])
