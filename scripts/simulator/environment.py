"""
Gymansium Environment API
Check the documentation: https://gymnasium.farama.org/introduction/create_custom_env/

TODO: https://gymnasium.farama.org/introduction/create_custom_env/#using-wrappers
Can create multiple similar environments!!!

Controls two buggies:
SC - (policy controlled) : Buggy
NAND - Classical Control (Stanley) : Buggy
"""
import numpy as np
import gymnasium as gym
from util.buggy import Buggy
from typing import Optional

NAND_WHEELBASE = 1.3
SC_WHEELBASE = 1.104

class BuggyCourseEnv(gym.Env):
    def __init__(self, rate : int = 100, steer_scale : float = np.pi/9):
        """
        Initialize a Buggy Course Environmnet. 

        Arguments:
        rate (Hz) - Simulation Rate
        steer_scale - Scale action space to full steering range
        
        """
        # HILL1_NAND from old sim code
        self.sc_init_state = (589761.40, 4477321.07, -1.91986) #easting, northing, heading
        self.nand_init_state = (589751.46, 4477322.07, -1.91986) #easting, northing, heading

        
        self.reset() # Sets up the buggies

        self.dt = 1 / rate
        self.steer_scale = steer_scale

        self.observation_space = gym.spaces.Box(-float('inf'), float('inf'), shape=(7,))
        self.action_space = gym.spaces.Box(-1, 1)
        

    def _get_obs(self) -> np.ndarray:
        """
        Observation Space:
        
        SC:
            - easting
            - northing
            - theta
            - speed
            - delta
        
        NAND:
            - easting
            - northing
        """
        return np.concatenate(arrays=[self.sc.get_self_obs(), self.nand.get_other_obs()])

    def _get_info(self) -> None:
        """
        Return environment info
        """
        return None

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, None]:
        """
        Starts a new episode
        Args:
            seed: Random seed for reproducible episodes
        """
        super().reset(seed=seed)

        self.sc = Buggy(
            e_utm=self.sc_init_state[0], 
            n_utm=self.sc_init_state[1],
            speed=0, 
            theta=self.sc_init_state[2], 
            wheelbase=SC_WHEELBASE
        )

        self.nand = Buggy(
            e_utm=self.nand_init_state[0], 
            n_utm=self.nand_init_state[1],
            speed=0, 
            theta=self.nand_init_state[2], 
            wheelbase=NAND_WHEELBASE
        )

        return self._get_obs(), self._get_info()

    def _dynamics(self, state : np.ndarray, control : np.ndarray, constants : np.ndarray):
        """
        Finds the derivative of the buggy state to help understand movement

        state - Buggy State
        control - Buggy Control
        constants - Buggy Constants
        """
        assert state.shape == (4,)
        assert control.shape == (1,)
        assert constants.shape == (2,)

        speed = state[2]
        theta = state[3]
        delta = control[0]
        wheelbase = constants[0]

        return np.array([speed * np.cos(theta),
                         speed * np.sin(theta),
                         0.0,
                         speed / wheelbase * np.tan(delta),
                         ])
    
    def _update_buggy(self, buggy: Buggy, dt : float) -> None:
        """
        Uses the `dynamics` function to update the buggies state based on the state and steering angle
        """
        state, control, constants = buggy.get_state(), buggy.get_control(), buggy.get_constants()

        k1 = self._dynamics(state, control, constants)
        k2 = self._dynamics(state + k1 * dt/2, control, constants)
        k3 = self._dynamics(state + k2 * dt/2, control, constants)
        k4 = self._dynamics(state + k3 * dt, control, constants)

        buggy.set_state(state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6)


    def step(self, sc_steering_percentage):
        """
        Executes one timestep within environment

        Args:
            action: The action to take ([-1 to 1] for steering percentage)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.sc.delta = sc_steering_percentage * self.steer_scale
        self._update_buggy(self.sc, self.dt)

        # TODO: Update NAND State using Stanley

        reward = 0 # TODO: Make a real reward

        terminated = False # TODO: Check when we cross the buggy finish line or crashes?

        # Simple environment doesn't have max step limit
        truncated = False 

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the environment for human viewing. Maybe just a plot of the trajectories?"""
        raise NotImplementedError()
