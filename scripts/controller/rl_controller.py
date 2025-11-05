from util.buggy import Buggy, BuggyObs
from util.trajectory import Trajectory
from scripts.controller.controller_superclass import Controller

class RL_Controller(Controller):
    def __init__(self, reference_traj: Trajectory) -> None:
        self.trajectory = reference_traj
        self.current_traj_index = 0

    # this is the policy
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