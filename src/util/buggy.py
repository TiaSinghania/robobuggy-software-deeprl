from dataclasses import dataclass

import numpy as np

"""
Dataclass representing a buggy, mainly should just be holding buggy attributes

State Vector:
e_utm - position easting (utm)
n_utm - position northing (utm)
x_speed - longitudinal speed (m/s) (along length of buggy)
y_speed - lateral speed in (m/s)
theta - heading (radians)
omega - heading rate (rad/s)

Constants:
wheelbase_f (length from center of gravity of buggy to front wheel) (m)
wheelbase_r (length from center of gravity of buggy to rear wheel) (m)
angle_clip - the max/min value that each buggy can steer (degrees)
mass:
inertia:
cornering stiffness:
mu_friction: coefficient of friction between tires and road (unitless)
r: wheel radius (including tire)
"""


@dataclass
class Buggy:
    # Buggy State
    e_utm: float  # m
    n_utm: float  # m
    x_speed: float  # m/s
    y_speed: float  # m/s
    theta: float  # rad
    omega: float # rad/s

    # Buggy Constants
    wheelbase_f: float = 1.104 / 2 # m
    wheelbase_r: float = 1.104 / 2# m
    angle_clip: float = np.pi / 9  # rad
    mass: float = 41 # kg
    inertia: float = 4.05 # kg * m^2
    cornering_stiffness: float # this should be randomized: depends on surface conditions, tire wear, too hard to know exactly basically.
    mu_friction: float # this should be randomized too 

    # Buggy Control
    delta: float = 0  # rad

    def get_state(self) -> np.ndarray:
        return np.array([self.e_utm, self.n_utm, self.x_speed, self.y_speed, self.theta, self.omega]).reshape(-1)

    def get_control(self) -> np.ndarray:
        return np.array([self.delta]).reshape(-1)

    def get_constants(self) -> np.ndarray:
        return np.array([self.wheelbase_f, self.wheelbase_r, self.angle_clip, self.mass, self.inertia, self.cornering_stiffness, self.mu_friction]).reshape(-1)

    def set_state(self, state: np.ndarray):
        assert state.shape == (6,)
        self.e_utm = state[0]
        self.n_utm = state[1]
        self.x_speed = state[2]
        self.y_speed = state[3]
        self.theta = state[4]
        self.omega = state[5]


    def get_full_obs(self) -> np.ndarray:
        return np.array(
            [self.e_utm, self.n_utm, self.x_speed, self.y_speed, self.theta, self.omega, self.delta], np.float32
        )

    def get_no_pos_obs(self) -> np.ndarray:
        return np.array([self.x_speed, self.y_speed, self.theta, self.omega, self.delta], np.float32)
