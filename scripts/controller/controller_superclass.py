from abc import ABC, abstractmethod

from nav_msgs.msg import Odometry


from util.trajectory import Trajectory
from util.buggy import BuggyObs

class Controller(ABC):
    """
    Base class for all controllers.

    The controller takes in the current state of the buggy and a reference
    trajectory. It must then compute the desired control output.

    The method that it does this by is up to the implementation, of course.
    Example schemes include Pure Pursuit, Stanley, and LQR.
    """

    def __init__(self, reference_traj: Trajectory) -> None:
        self.trajectory = reference_traj
        self.current_traj_index = 0

    @abstractmethod
    def compute_control(
        self, obs: BuggyObs
    ) -> float:
        """
        Computes the desired control output given the current state and reference trajectory

        Args:
            buggy: (BuggyObs): state of the buggy, including position, attitude and associated twists

        Returns:
            float (desired steering angle)
        """
        raise NotImplementedError