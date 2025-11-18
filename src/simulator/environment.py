"""
Gymansium Environment API
Check the documentation: https://gymnasium.farama.org/introduction/create_custom_env/

TODO: https://gymnasium.farama.org/introduction/create_custom_env/#using-wrappers
Can create multiple similar environments!!!

Controls two buggies:
SC - (policy controlled) : Buggy


"""

from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register

from src.controller.stanley_controller import StanleyController
from src.util.buggy import Buggy
from src.util.trajectory import Trajectory

SC_WHEELBASE = 1.104

UTM_EAST_ZERO = 589761.40
UTM_NORTH_ZERO = 4477321.07

OBS_SIZE = 6


class BuggyCourseEnv(gym.Env):
    def __init__(
        self,
        rate: int = 100,
        steer_scale: float = np.pi / 9,
        target_path: str = "src/util/buggycourse_sc.json",
        left_curb_path: str = "src/util/left_curb.json",
        right_curb_path: str = "src/util/right_curb.json",
        render_every_n_steps: int = 5,
    ):
        """
        Initialize a Buggy Course Environmnet.

        Arguments:
        rate (Hz) - Simulation Rate
        steer_scale - Scale action space to full steering range

        """
        # Positions
        self.sc_init_state = (
            589761.40 - UTM_EAST_ZERO,
            4477321.07 - UTM_NORTH_ZERO,
            -1.91986,
        )  # easting, northing, heading

        self.dt = 1 / rate
        self.steer_scale = steer_scale
        self.render_every_n_steps = render_every_n_steps

        self.target_traj = Trajectory(target_path)
        self.left_curb = Trajectory(left_curb_path)
        self.right_curb = Trajectory(right_curb_path)

        target_traj_idx = self.target_traj.get_closest_index_on_path(
            589693.75 - UTM_EAST_ZERO, 4477191.05 - UTM_NORTH_ZERO
        )
        self.target_traj_dist = self.target_traj.get_distance_from_index(
            target_traj_idx
        )

        self.observation_space = gym.spaces.Box(
            low=np.ones((OBS_SIZE,), dtype=np.float32) * -float("inf"),
            high=np.ones((OBS_SIZE,), dtype=np.float32) * float("inf"),
            shape=(OBS_SIZE,),
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1)

        # Visualization
        self.fig = None
        self.ax = None
        self.step_count = 0
        self.window_closed = False

        self.reset()  # Sets up the buggies

    def _get_privileged_obs(self) -> np.ndarray:
        sc_x, sc_y = self.sc.e_utm, self.sc.n_utm
        left_dist = self.left_curb.get_distance_to_path(sc_x, sc_y)
        right_dist = self.right_curb.get_distance_to_path(sc_x, sc_y)
        return np.array([left_dist - right_dist], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """
        Observation Space:

        SC:
            - easting
            - northing
            - speed
            - theta
            - delta

        PRIVILEGED:
            - distance from center
        """
        return np.concatenate(
            [
                self.sc.get_full_obs(),
                self._get_privileged_obs(),
            ]
        )

    def _get_info(self) -> dict:
        """
        Return environment info
        """
        return {}

    def reset(self, seed: Optional[int] = None, **kwargs) -> tuple[np.ndarray, None]:
        """
        Starts a new episode
        Args:
            seed: Random seed for reproducible episodes
        """
        super().reset(seed=seed)

        self.sc = Buggy(
            e_utm=self.sc_init_state[0],
            n_utm=self.sc_init_state[1],
            speed=12,
            theta=self.sc_init_state[2],
            wheelbase=SC_WHEELBASE,
        )

        self.terminated = False
        self.prev_dist = 0.0
        self.step_count = 0

        obs = self._get_obs()

        return obs, self._get_info()

    def _dynamics(self, state: np.ndarray, control: np.ndarray, constants: np.ndarray):
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

        return np.array(
            [
                speed * np.cos(theta),
                speed * np.sin(theta),
                0.0,
                speed / wheelbase * np.tan(delta),
            ],
            dtype=np.float32,
        )

    def _update_buggy(self, buggy: Buggy, dt: float) -> None:
        """
        Uses the `dynamics` function to update the buggies state based on the state and steering angle
        """
        state, control, constants = (
            buggy.get_state(),
            buggy.get_control(),
            buggy.get_constants(),
        )

        k1 = self._dynamics(state, control, constants)
        k2 = self._dynamics(state + k1 * dt / 2, control, constants)
        k3 = self._dynamics(state + k2 * dt / 2, control, constants)
        k4 = self._dynamics(state + k3 * dt, control, constants)

        buggy.set_state(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    def _get_reward(self) -> float:
        """
        Returns a reward, currently MSE of how far along you are along the path until the target
        """
        # TODO: Currently no curb constraints, so optimal strategy is beeline to goal flag!
        traj_idx = self.target_traj.get_closest_index_on_path(
            self.sc.e_utm, self.sc.n_utm
        )
        traj_dist = self.target_traj.get_distance_from_index(traj_idx)

        if traj_dist > self.target_traj_dist:
            reward = 1e5 if not self.terminated else 0
            self.terminated = True  # Crossed the finish line
        elif traj_dist > (self.prev_dist + 1):
            # Give a reward for every meter you move forward
            reward = 1
            self.prev_dist = traj_dist
        elif traj_dist < (self.prev_dist - 1):
            # Signifcantly went backwards
            reward = -1
        else:
            # Not going anywhere
            reward = -0.1

        return reward

    def _check_crash(self) -> bool:
        sc_x, sc_y = self.sc.e_utm, self.sc.n_utm
        if (
            self.left_curb.get_distance_to_path(sc_x, sc_y) < 0.1
            or self.right_curb.get_distance_to_path(sc_x, sc_y) < 0.1
        ):
            return True
        return False

    def step(self, sc_steering_percentage):
        """
        Executes one timestep within environment

        Args:
            action: The action to take ([-1 to 1] for steering percentage)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert sc_steering_percentage.shape == (1,)
        self.sc.delta = sc_steering_percentage[0] * self.steer_scale
        self._update_buggy(self.sc, self.dt)

        reward = self._get_reward()

        if self._check_crash():
            self.terminated = True
            reward -= 1e3  # Crash Penalty

        self.step_count += 1

        # Simple environment doesn't have max step limit
        truncated = False

        return self._get_obs(), reward, self.terminated, truncated, self._get_info()

    def _on_close(self, event):
        """Handle window close event."""
        self.window_closed = True

    def render(self):
        """Render the environment for human viewing with step counter.

        Only actually renders every N steps to speed up visualization without
        affecting simulation fidelity.
        """
        # Skip rendering if not on a render frame
        if self.step_count % self.render_every_n_steps != 0:
            return

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            plt.ion()  # Enable interactive mode
            self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.ax.clear()

        # Plot the reference trajectory
        traj_positions = self.target_traj.positions
        self.ax.plot(
            traj_positions[:, 0],
            traj_positions[:, 1],
            "k--",
            linewidth=2,
            label="Reference Trajectory",
            alpha=0.5,
        )

        # Plot curb
        curb_positions = np.concatenate(
            [
                np.array(
                    [
                        self.left_curb.get_position_by_distance(dist)
                        for dist in np.linspace(
                            0,
                            self.left_curb.distances[-1],
                            len(self.left_curb.distances) // 10,
                        )
                    ]
                ),
                np.full(
                    (1, 2), np.nan
                ),  # Splits the line segment so both curbs aren't conjoined
                np.array(
                    [
                        self.right_curb.get_position_by_distance(dist)
                        for dist in np.linspace(
                            0,
                            self.right_curb.distances[-1],
                            len(self.right_curb.distances) // 10,
                        )
                    ]
                ),
            ],
            axis=0,
        )
        self.ax.plot(
            curb_positions[:, 0],
            curb_positions[:, 1],
            "k",
            linewidth=1,
            label="Curbs",
            alpha=0.5,
        )

        # Plot SC buggy (policy-controlled)
        self.ax.plot(
            self.sc.e_utm,
            self.sc.n_utm,
            "bo",
            markersize=6,
            label=f"SC Buggy (Policy) - Speed: {self.sc.speed:.1f} m/s - Steering {np.rad2deg(self.sc.delta):.1f}°",
        )
        # Draw heading arrow for SC
        arrow_length = 8
        self.ax.arrow(
            self.sc.e_utm,
            self.sc.n_utm,
            arrow_length * np.cos(self.sc.theta),
            arrow_length * np.sin(self.sc.theta),
            head_width=2,
            head_length=2,
            fc="blue",
            ec="blue",
        )

        # Plot Previous SC Spot
        prev_east, prev_north = self.target_traj.get_position_by_distance(
            self.prev_dist
        )
        self.ax.plot(
            prev_east,
            prev_north,
            "bo",
            markersize=3,
            label=f"Previous SC Spot",
        )

        # Plot finish line
        finish_e, finish_n = 589693.75 - UTM_EAST_ZERO, 4477191.05 - UTM_NORTH_ZERO
        self.ax.plot(finish_e, finish_n, "g*", markersize=20, label="Finish Line")

        # Add step counter prominently
        time_elapsed = self.step_count * self.dt
        self.ax.text(
            0.02,
            0.98,
            f"Step: {self.step_count}\nTime: {time_elapsed:.2f}s\nRate: {1 / self.dt:.0f} Hz",
            transform=self.ax.transAxes,
            fontsize=14,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        self.ax.set_xlabel("Easting (m)", fontsize=12)
        self.ax.set_ylabel("Northing (m)", fontsize=12)
        self.ax.set_title("Buggy Course Simulation", fontsize=14, fontweight="bold")
        self.ax.legend(loc="upper right")
        self.ax.grid(True, alpha=0.3)
        self.ax.axis("equal")

        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot and process events
