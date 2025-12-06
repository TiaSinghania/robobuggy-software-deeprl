from abc import ABC, abstractmethod
from typing import Any


class PolicyWrapper(ABC):
    """
    Base class for policy wrappers.

    Provides a common interface for training, saving, and loading policies.
    Subclasses may extend method signatures with additional parameters using **kwargs.
    """

    def __init__(self, env=None, dirpath: str = "./logs"):
        """
        Initialize the policy wrapper.

        Args:
            env: The environment (can be None if wrapper creates its own)
            dirpath: Directory path for saving/loading models and logs
        """
        self.env = env
        self.dirpath = dirpath
        self.policy = None  # Set by child classes

    @abstractmethod
    def train(self, timesteps: int, **kwargs) -> None:
        """
        Train the policy.

        Args:
            timesteps: Number of timesteps to train for (or primary phase timesteps for multi-phase training)
            **kwargs: Additional training parameters (e.g., phase2_timesteps for RMA)
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the current model to dirpath."""
        pass

    @abstractmethod
    def load(self, **kwargs) -> None:
        """
        Load a saved model.

        Args:
            **kwargs: Additional parameters (e.g., phase for RMA)
        """
        pass

    def close(self) -> None:
        """Clean up resources (e.g., close environments)."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
