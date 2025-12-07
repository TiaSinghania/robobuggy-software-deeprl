import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

from src.simulator.environment import rma_phase


class RMAExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        observation_size: int,
        state_action_size: int,
        # phase 1 params
        env_vector_size: int,
        embedding_dim: int = 8,
        embedding_hidden_dim: int = 128,
        # phase 2 params
        lookback_steps: int = 50,
        adaptation_hidden_dim: int = 128,
        adaptation_embedding_dim: int = 8,
        # phase params
        phase: rma_phase = "phase_1",
    ):
        super().__init__(observation_space, observation_size + embedding_dim)

        self.observation_size = observation_size
        self.state_action_size = state_action_size
        self.lookback_steps = lookback_steps
        self.phase = phase

        self.rma_env_vector_size = env_vector_size

        self.rma_embedding = nn.Sequential(
            nn.Linear(env_vector_size, embedding_hidden_dim),
            nn.ReLU(),
            nn.Linear(embedding_hidden_dim, embedding_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_hidden_dim, embedding_dim),
            nn.Tanh(),
        )

        # gets applied to each state action pair in the buffer
        self.adaptation_embedding = nn.Sequential(
            nn.Linear(state_action_size, adaptation_hidden_dim),
            nn.ReLU(),
            nn.Linear(adaptation_hidden_dim, adaptation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adaptation_hidden_dim, adaptation_embedding_dim),
            nn.Tanh(),
        )
        # Then, a 3-layer 1-D CNN
        # convolves the representations across the time dimension to
        # capture temporal correlations in the input. The input channel
        # number, output channel number, kernel size, and stride of each
        # layer are [32, 32, 8, 4], [32, 32, 5, 1], [32, 32, 5, 1]. The flattened
        # CNN output is linearly projected to estimate ˆzt
        num_channels = adaptation_embedding_dim

        seq_len = lookback_steps
        seq_len = (seq_len - 8) // 4 + 1  # after conv1
        seq_len = seq_len - 4  # after conv2
        seq_len = seq_len - 4  # after conv3
        flattened_size = num_channels * seq_len

        # takes in [batch, size, seqs] and outputs [batch, embedding_dim]
        self.output_cnns = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_size, embedding_dim),  # project to embedding_dim
            nn.Tanh(),
        )

    def forward_encoder(self, env_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Phase 1 encoder.
        Args:
            env_vector: Environment parameters [batch_size, env_vector_size]
        Returns:
            embedding: Latent embedding z [batch_size, embedding_dim]
        """
        return self.rma_embedding(env_vector)

    def forward_adaptation(self, state_action_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Phase 2 adaptation module.
        Args:
            state_action_pairs: History of state-action pairs
                                [batch_size, lookback_steps * state_action_size]
                                OR
                                [batch_size, lookback_steps, state_action_size]
        Returns:
            embedding: Estimated latent embedding z_hat [batch_size, embedding_dim]
        """
        # Handle both flattened and structured input
        if state_action_pairs.dim() == 2:
            batch_size = state_action_pairs.shape[0]
            state_action_pairs_B_L_S = state_action_pairs.reshape(
                batch_size, self.lookback_steps, self.state_action_size
            )
        else:
            state_action_pairs_B_L_S = state_action_pairs

        adaptation_embeddings_B_L_AE = self.adaptation_embedding(
            state_action_pairs_B_L_S
        )
        adaptation_embeddings_B_AE_L = adaptation_embeddings_B_L_AE.transpose(1, 2)
        adaptation_estimate_B_E = self.output_cnns(adaptation_embeddings_B_AE_L)
        return adaptation_estimate_B_E

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.phase == "phase_1":
            # obs is of shape [batch, observation_size + rma_env_vector_size]
            assert obs.shape[1] == self.observation_size + self.rma_env_vector_size

            env_vector = obs[:, self.observation_size :]
            assert env_vector.shape[1] == self.rma_env_vector_size

            embedding = self.forward_encoder(env_vector)

            real_obs = obs[:, : self.observation_size]

            return torch.cat([real_obs, embedding], dim=1)

        elif self.phase == "phase_2":
            # obs is of shape [batch, observation_size + state_action_size*lookback_steps]
            assert (
                obs.shape[1]
                == self.observation_size + self.state_action_size * self.lookback_steps
            )

            state_action_pairs = obs[:, self.observation_size :]

            adaptation_estimate_B_E = self.forward_adaptation(state_action_pairs)

            real_obs = obs[:, : self.observation_size]

            return torch.cat([real_obs, adaptation_estimate_B_E], dim=1)
