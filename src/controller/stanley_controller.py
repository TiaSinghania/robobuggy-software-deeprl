import numpy as np
import utm

from src.util.buggy import Buggy
from src.util.trajectory import Trajectory


class StanleyController:
    """
    Stanley Controller (front axle used as reference point)
    Referenced from this paper: https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
    """

    CROSS_TRACK_GAIN = 1.3
    K_SOFT = 1.0  # m/s
    K_D_YAW = 0.012  # rad / (rad/s)

    def __init__(self, buggy: Buggy, reference_traj: Trajectory) -> None:
        self.trajectory = reference_traj
        self.current_traj_index = 0
        self.buggy = buggy

    def compute_control(self):
        """Computes the steering angle determined by Stanley controller.
        Does this by looking at the crosstrack error + heading error

        Args:
            buggy: State of buggy computing the control of
            trajectory (Trajectory): reference trajectory

        Returns:
            float (desired steering angle)
        """
        if self.current_traj_index >= self.trajectory.get_num_points() - 1:
            raise Exception("[Stanley]: Ran out of path to follow!")

        current_speed = self.buggy.speed
        heading = self.buggy.theta
        x, y = self.buggy.e_utm, self.buggy.n_utm  # (Easting, Northing)

        front_x = x + self.buggy.wheelbase * np.cos(heading)
        front_y = y + self.buggy.wheelbase * np.sin(heading)

        # setting range of indices to search so we don't have to search the entire path
        traj_index = self.trajectory.get_closest_index_on_path(
            front_x,
            front_y,
            start_index=self.current_traj_index - 20,
            end_index=self.current_traj_index + 50,
        )
        self.current_traj_index = max(traj_index, self.current_traj_index)

        # Use heading at the closest index
        ref_heading = self.trajectory.get_heading_by_index(self.current_traj_index)

        error_heading = ref_heading - heading
        error_heading = np.arctan2(
            np.sin(error_heading), np.cos(error_heading)
        )  # Bounds error_heading

        # Calculate cross track error by finding the distance from the buggy to the tangent line of
        # the reference trajectory
        closest_position = self.trajectory.get_position_by_index(
            self.current_traj_index
        )
        next_position = self.trajectory.get_position_by_index(
            self.current_traj_index + 0.0001
        )
        x1 = closest_position[0]
        y1 = closest_position[1]
        x2 = next_position[0]
        y2 = next_position[1]
        error_dist = -((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2
        )

        cross_track_component = -np.arctan2(
            StanleyController.CROSS_TRACK_GAIN * error_dist,
            current_speed + StanleyController.K_SOFT,
        )

        # Determine steering_command
        steering_cmd = error_heading + cross_track_component
        steering_cmd = np.clip(steering_cmd, -np.pi / 9, np.pi / 9)

        return steering_cmd
