from imitation.policies.serialize import policy_registry
from stable_baselines3.common import policies
from stable_baselines3.common.type_aliases import PyTorchObs
import torch
import numpy as np

from src.util.trajectory import Trajectory


class StanleyPolicy(policies.BasePolicy):
    CROSS_TRACK_GAIN = 1.3
    K_SOFT = 1.0  # m/s
    K_D_YAW = 0.012  # rad / (rad/s)
    WHEELBASE = 1.104

    def __init__(self, venv, reference_traj_path: str, **kwargs) -> None:
        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            **kwargs,
        )
        self.trajectory = Trajectory(
            reference_traj_path, create_kdtree=True, resolution=0.5
        )

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        assert isinstance(observation, torch.Tensor)
        assert observation.ndim == 2 and observation.shape == (
            1,
            8,
        ), f"Dimensions {observation.ndim}, Shape {observation.shape}"

        current_speed = observation[0, 2].item()
        heading = observation[0, 3].item()
        x, y = observation[0, 0].item(), observation[0, 1].item()  # (Easting, Northing)

        front_x = x + self.WHEELBASE * np.cos(heading)
        front_y = y + self.WHEELBASE * np.sin(heading)

        # setting range of indices to search so we don't have to search the entire path
        traj_index = self.trajectory.get_closest_index_on_path(
            front_x,
            front_y,
        )

        if traj_index >= self.trajectory.get_num_points() - 1:
            return torch.Tensor([0], device=observation.device)

        # Use heading at the closest index
        ref_heading = self.trajectory.get_heading_by_index(traj_index)

        error_heading = ref_heading - heading
        error_heading = np.arctan2(
            np.sin(error_heading), np.cos(error_heading)
        )  # Bounds error_heading

        # Calculate cross track error by finding the distance from the buggy to the tangent line of
        # the reference trajectory
        closest_position = self.trajectory.get_position_by_index(traj_index)
        next_position = self.trajectory.get_position_by_index(traj_index + 0.0001)
        x1 = closest_position[0]
        y1 = closest_position[1]
        x2 = next_position[0]
        y2 = next_position[1]
        error_dist = -((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2
        )

        cross_track_component = -np.arctan2(
            self.CROSS_TRACK_GAIN * error_dist,
            current_speed + self.K_SOFT,
        )

        # Determine steering_command
        steering_cmd = error_heading + cross_track_component
        steering_cmd = np.clip(steering_cmd, -np.pi / 9, np.pi / 9) / (np.pi / 9)

        return torch.Tensor(steering_cmd, device=observation.device)


def stanley_policy_loader(venv, reference_traj_path: Trajectory) -> policies.BasePolicy:
    # load your policy here
    return StanleyPolicy(venv, reference_traj_path=reference_traj_path)


policy_registry.register(key="stanley-policy", value=stanley_policy_loader)
