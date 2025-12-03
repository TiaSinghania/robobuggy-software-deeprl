import os
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

from src.policy_wrappers.policy_wrapper import PolicyWrapper
from src.simulator.environment import RMAConfig, rma_phase
from src.rma_wrappers.rma_wrapper import RMAExtractor


class RMA_PPO_Wrapper(PolicyWrapper):
    """
    PPO Wrapper with RMA (Rapid Motor Adaptation) support.

    Supports unified two-phase training:
    - Phase 1: Train encoder + policy end-to-end with privileged env info
    - Phase 2: Freeze encoder, train adaptation module to mimic encoder from history

    The wrapper handles phase switching and weight transfer between phases.
    Environment workers are reused between phases for efficiency.
    """

    def __init__(
        self,
        dirpath: str,
        # Environment params
        n_envs: int = 10,
        rate: int = 20,
        include_pos_in_obs: bool = False,
        # RMA params
        lookback_steps: int = 50,
        embedding_dim: int = 8,
        embedding_hidden_dim: int = 128,
        adaptation_hidden_dim: int = 128,
        adaptation_embedding_dim: int = 8,
        # PPO params
        device: str = "cpu",
    ):
        # Store configuration
        self.dirpath = dirpath
        self.n_envs = n_envs
        self.rate = rate
        self.include_pos_in_obs = include_pos_in_obs
        self.device = device

        # RMA configuration
        self.lookback_steps = lookback_steps
        self.embedding_dim = embedding_dim
        self.embedding_hidden_dim = embedding_hidden_dim
        self.adaptation_hidden_dim = adaptation_hidden_dim
        self.adaptation_embedding_dim = adaptation_embedding_dim

        # Dimension calculations
        # base_obs_size is the observation size without RMA appendage
        # For include_pos_in_obs=False: 7 (3 state + 4 privileged)
        # For include_pos_in_obs=True: 9 (5 state + 4 privileged)
        self.base_obs_size = 9 if include_pos_in_obs else 7
        self.action_size = 1
        self.state_action_size = self.base_obs_size + self.action_size

        # Get env_vector_size from the environment
        # TODO: This should be retrieved from the actual environment once domain randomization is implemented
        self.env_vector_size = (
            0  # Placeholder until domain randomization is implemented
        )

        # Current phase tracking
        self.current_phase: rma_phase = "phase_1"

        # Initialize Phase 1 environment and policy
        self._init_phase1()

        print(f"RMA PPO Wrapper initialized")
        print(f"  Base obs size: {self.base_obs_size}")
        print(f"  State-action size: {self.state_action_size}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Env vector size: {self.env_vector_size}")
        print(f"  Lookback steps: {self.lookback_steps}")

        # Call parent init (sets self.env and self.dirpath, but we override them)
        # Note: We don't call super().__init__() because we manage env ourselves

    def _create_env(self, phase: rma_phase):
        """Create a vectorized environment for the given phase."""
        rma_config = RMAConfig(
            lookback_steps=self.lookback_steps,
            current_phase=phase,
        )

        env = make_vec_env(
            "BuggyCourseEnv-v1",
            n_envs=self.n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "rate": self.rate,
                "include_pos_in_obs": self.include_pos_in_obs,
                "rma_config": rma_config,
            },
        )

        # Use consistent monitor filename so plot_results can find it
        return VecMonitor(env, self.dirpath + "/monitor.csv")

    def _switch_env_phase(self, phase: rma_phase):
        """
        Switch all environment workers to a new phase without recreating them.

        This updates:
        1. The phase in each worker via env_method
        2. The observation_space on VecEnv wrappers (synced from workers)
        """
        print(f"Switching environment workers to {phase}...")

        # Get the underlying SubprocVecEnv from VecMonitor
        vec_env = self.env.venv if hasattr(self.env, "venv") else self.env

        # Call set_rma_phase on all workers
        vec_env.env_method("set_rma_phase", phase)

        # Fetch updated observation space from workers (they calculate it correctly)
        new_obs_space = vec_env.env_method("get_observation_space")[0]

        # Update observation spaces on VecEnv wrappers
        vec_env.observation_space = new_obs_space
        if hasattr(self.env, "observation_space"):
            self.env.observation_space = new_obs_space

        self.current_phase = phase
        print(
            f"  Environment switched to {phase}, obs_space shape: {new_obs_space.shape}"
        )

    def _create_policy_kwargs(self, phase: rma_phase):
        """Create policy kwargs for the given phase."""
        return dict(
            features_extractor_class=RMAExtractor,
            features_extractor_kwargs=dict(
                observation_size=self.base_obs_size,
                state_action_size=self.state_action_size,
                env_vector_size=self.env_vector_size,
                embedding_dim=self.embedding_dim,
                embedding_hidden_dim=self.embedding_hidden_dim,
                lookback_steps=self.lookback_steps,
                adaptation_hidden_dim=self.adaptation_hidden_dim,
                adaptation_embedding_dim=self.adaptation_embedding_dim,
                phase=phase,
            ),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )

    def _init_phase1(self):
        """Initialize Phase 1: encoder + policy training."""
        self.current_phase = "phase_1"
        self.env = self._create_env("phase_1")

        policy_kwargs = self._create_policy_kwargs("phase_1")

        self.policy: PPO = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            device=self.device,
            policy_kwargs=policy_kwargs,
        )

        print("Phase 1 initialized: Training encoder + policy")

    def _transition_to_phase2(self):
        """
        Transition from Phase 1 to Phase 2:
        1. Extract Phase 1 weights
        2. Switch environment workers to Phase 2 (reuses workers)
        3. Create new Phase 2 policy with updated architecture
        4. Transfer encoder and policy weights from Phase 1
        5. Freeze encoder and policy weights, only train adaptation module
        """
        print("\n" + "=" * 60)
        print("Transitioning to Phase 2...")
        print("=" * 60)

        # 1. Extract Phase 1 weights
        phase1_extractor = self.policy.policy.features_extractor
        phase1_encoder_state = phase1_extractor.rma_embedding.state_dict()

        # Save the full Phase 1 policy network weights (actor + critic)
        phase1_policy_state = self.policy.policy.state_dict()

        # 2. Switch environment workers to Phase 2 (reuses subprocess workers)
        self._switch_env_phase("phase_2")

        # 3. Create new Phase 2 policy (needed because observation space changed)
        policy_kwargs = self._create_policy_kwargs("phase_2")

        self.policy = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            device=self.device,
            policy_kwargs=policy_kwargs,
        )

        # 4. Transfer weights from Phase 1
        phase2_extractor = self.policy.policy.features_extractor

        # Transfer encoder weights (rma_embedding is used in both phases for reference)
        phase2_extractor.rma_embedding.load_state_dict(phase1_encoder_state)

        # Transfer policy network weights (mlp_extractor, action_net, value_net)
        # Selectively transfer weights that have matching shapes in both phases
        phase2_policy_state = self.policy.policy.state_dict()

        transferred, skipped = 0, 0
        for key in phase2_policy_state.keys():
            if key not in phase1_policy_state:
                print(f"    Skipping {key}: not in Phase 1 state")
                skipped += 1
            elif phase1_policy_state[key].shape != phase2_policy_state[key].shape:
                print(
                    f"    Skipping {key}: shape mismatch {phase1_policy_state[key].shape} vs {phase2_policy_state[key].shape}"
                )
                skipped += 1
            else:
                phase2_policy_state[key] = phase1_policy_state[key]
                transferred += 1

        self.policy.policy.load_state_dict(phase2_policy_state)
        print(f"  Transferred {transferred} weight tensors, skipped {skipped}")

        # 5. Freeze everything except adaptation module
        self._freeze_for_phase2(phase2_extractor)

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in self.policy.policy.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.policy.policy.parameters())

        print(f"Phase 2 initialized: Training adaptation module")
        print(f"  Frozen: encoder, policy networks")
        print(f"  Training: adaptation_embedding, output_cnns")
        print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    def _freeze_for_phase2(self, extractor):
        """Freeze encoder and policy networks, keep only adaptation module trainable."""
        # Freeze all parameters first
        for param in self.policy.policy.parameters():
            param.requires_grad = False

        # Unfreeze only the adaptation modules
        for module in [extractor.adaptation_embedding, extractor.output_cnns]:
            for param in module.parameters():
                param.requires_grad = True

    def train(self, timesteps: int, phase2_timesteps: int = 0, **kwargs) -> None:
        """
        Train RMA in two phases.

        Args:
            timesteps: Number of timesteps for Phase 1 (encoder + policy)
            phase2_timesteps: Number of timesteps for Phase 2 (adaptation module)
                              If 0, only Phase 1 is trained.
            **kwargs: Additional arguments (unused, for interface compatibility)
        """
        phase1_timesteps = timesteps
        # Phase 1: Train encoder + policy
        print("\n" + "=" * 60)
        print(f"PHASE 1: Training encoder + policy ({phase1_timesteps:,} timesteps)")
        print("=" * 60)

        self.policy.learn(total_timesteps=phase1_timesteps)

        # Save Phase 1 checkpoint
        self.policy.save(f"{self.dirpath}/model_phase1")
        print(f"Phase 1 model saved to {self.dirpath}/model_phase1")

        # Plot Phase 1 results
        try:
            plot_results(
                [self.dirpath + "/"],
                phase1_timesteps,
                results_plotter.X_TIMESTEPS,
                "RMA PPO Phase 1",
            )
            plt.savefig(self.dirpath + "/rma_ppo_phase1_rewards.png")
            plt.close()
        except Exception as e:
            print(f"Could not plot Phase 1 results: {e}")

        # Phase 2: Train adaptation module (if requested)
        if phase2_timesteps > 0:
            self._transition_to_phase2()

            print("\n" + "=" * 60)
            print(
                f"PHASE 2: Training adaptation module ({phase2_timesteps:,} timesteps)"
            )
            print("=" * 60)

            # Reset timesteps for fresh Phase 2 metrics
            self.policy.learn(
                total_timesteps=phase2_timesteps, reset_num_timesteps=True
            )

            # Save Phase 2 checkpoint
            self.policy.save(f"{self.dirpath}/model_phase2")
            print(f"Phase 2 model saved to {self.dirpath}/model_phase2")

            # Plot Phase 2 results
            try:
                plot_results(
                    [self.dirpath + "/"],
                    phase2_timesteps,
                    results_plotter.X_TIMESTEPS,
                    "RMA PPO Phase 2",
                )
                plt.savefig(self.dirpath + "/rma_ppo_phase2_rewards.png")
                plt.close()
            except Exception as e:
                print(f"Could not plot Phase 2 results: {e}")

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    def save(self) -> None:
        """Save the current model."""
        self.policy.save(f"{self.dirpath}/model")
        print(f"RMA PPO model saved to {self.dirpath}/model")

    def load(self, phase: rma_phase = "phase_2", **kwargs) -> None:
        """
        Load a saved model.

        Args:
            phase: Which phase model to load ("phase_1" or "phase_2")
            **kwargs: Additional arguments (unused, for interface compatibility)
        """
        print(f"Loading RMA PPO model ({phase})...")

        # Switch environment to the appropriate phase (reuses workers)
        if self.current_phase != phase:
            self._switch_env_phase(phase)

        model_path = f"{self.dirpath}/model_{phase}"
        if not os.path.exists(model_path + ".zip"):
            model_path = f"{self.dirpath}/model"

        # Pass policy_kwargs to restore custom RMAExtractor
        policy_kwargs = self._create_policy_kwargs(phase)
        self.policy = PPO.load(
            model_path,
            env=self.env,
            custom_objects={"policy_kwargs": policy_kwargs},
        )
        print(f"Loaded model from {model_path}")

    def close(self) -> None:
        """Clean up resources (close environment workers)."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
            self.env = None
        print("RMA PPO Wrapper closed")

    def get_encoder(self):
        """
        Extract the trained encoder from the policy.
        Returns the encoder network that maps env params -> latent z.
        """
        return self.policy.policy.features_extractor.rma_embedding

    def get_adaptation_module(self):
        """
        Extract the trained adaptation module from the policy.
        Returns both the per-timestep embedding and the CNN.
        """
        features_extractor = self.policy.policy.features_extractor
        return {
            "adaptation_embedding": features_extractor.adaptation_embedding,
            "output_cnns": features_extractor.output_cnns,
        }
