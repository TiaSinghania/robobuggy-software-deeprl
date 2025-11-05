import numpy as np


from util.trajectory import Trajectory
from util.buggy import BuggyObs
from controller_superclass import Controller

import utm


class StanleyController(Controller):
    """
    Stanley Controller (front axle used as reference point)
    Referenced from this paper: https://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf
    """

    CROSS_TRACK_GAIN = 1.3
    K_SOFT = 1.0 # m/s
    K_D_YAW = 0.012 # rad / (rad/s)

    def __init__(self, reference_traj):
        super(StanleyController, self).__init__(reference_traj)

    # TODO: update this once state space is well defined
    def compute_control(self, obs: BuggyObs):
        """Computes the steering angle determined by Stanley controller.
        Does this by looking at the crosstrack error + heading error

        Args:
            state_msg: ros Odometry message
            trajectory (Trajectory): reference trajectory

        Returns:
            float (desired steering angle)
        """
        if self.current_traj_index >= self.trajectory.get_num_points() - 1:
            raise Exception("[Stanley]: Ran out of path to follow!")

        current_rospose = state_msg.pose.pose
        current_speed = np.sqrt(
            state_msg.twist.twist.linear.x**2 + state_msg.twist.twist.linear.y**2
        )
        yaw_rate = state_msg.twist.twist.angular.z
        heading = current_rospose.orientation.z
        x, y = current_rospose.position.x, current_rospose.position.y #(Easting, Northing)

        front_x = x + obs.wheelbase * np.cos(heading)
        front_y = y + obs.wheelbase * np.sin(heading)

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
        error_heading = np.arctan2(np.sin(error_heading), np.cos(error_heading)) #Bounds error_heading

        # Calculate cross track error by finding the distance from the buggy to the tangent line of
        # the reference trajectory
        closest_position = self.trajectory.get_position_by_index(self.current_traj_index)
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
            StanleyController.CROSS_TRACK_GAIN * error_dist, current_speed + StanleyController.K_SOFT
        )

        # Use acceleration at the closest index
        accel_x, accel_y = self.trajectory.get_acceleration_by_index(self.current_traj_index)
        # this works because tan(heading) = dydt/dxdt (do the math)
        dxdt, dydt = np.cos(ref_heading), np.sin(ref_heading)

        # this was dervied by doing the chain rule on the target derivative of theta.
        # dtheta/dt = d/dt (arctan (dydt/dxdt)) << do math.
        r_traj = (1/(1 + (dydt/dxdt)**2)) * (accel_y/dxdt - (dydt * accel_x)/(dxdt ** 2))

        # Calculate yaw rate error
        r_meas = yaw_rate

        yaw = float(StanleyController.K_D_YAW * (r_traj - r_meas))
        # Determine steering_command
        steering_cmd = error_heading + cross_track_component
        if self.usingHeadingRateError:
            steering_cmd += yaw
        steering_cmd = np.clip(steering_cmd, -np.pi / 9, np.pi / 9)

        # Calculate error, where x is in orientation of buggy, y is cross track error
        current_pose = Pose(current_rospose.position.x, current_rospose.position.y, heading)
        reference_error = current_pose.convert_point_from_global_to_local_frame(closest_position)
        reference_error -= np.array([StanleyController.WHEELBASE, 0]) # Translate back to back wheel to get accurate error

        return steering_cmd
