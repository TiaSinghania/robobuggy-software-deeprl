from typing import Literal
import gymnasium as gym
import numpy as np
import torch
from torch import nn
import argparse
import imageio
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from src.policy_wrappers.policy_wrapper import PolicyWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from src.policies.stanley_policy import StanleyPolicy
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from typing import Optional


from sb3_contrib.common.recurrent.type_aliases import RNNStates

try:
    import wandb
except ImportError:
    wandb = None


class TrainDagger:

    def __init__(
        self,
        env: VecEnv,
        policy: RecurrentActorCriticPolicy,
        optimizer: torch.optim,
        expert_policy: StanleyPolicy,
        device="cpu",
    ):
        """
        Initializes the TrainDagger class. Creates necessary data structures.

        Args:
            env: an OpenAI Gym environment.
            model: the model to be trained.
            expert_policy: the expert model that provides the expert actions.
            device: the device to be used for training.

        """
        self.env = env
        self.policy = policy
        self.expert_policy = expert_policy
        self.optimizer = optimizer

        self.loss_fn = nn.MSELoss()
        self.device = device

        self.expert_policy = self.expert_policy.to(self.device)

        self.states = None
        self.policy_states: Optional[RNNStates] = None
        self.actions = None
        self.episode_starts = None
        self.timesteps = None

    def generate_trajectory(self, env: VecEnv, policy: RecurrentActorCriticPolicy):
        """Collects one rollout from the policy in an environment. The environment
        should implement the OpenAI Gym interface. A rollout ends when done=True. The
        number of states and actions should be the same, so you should not include
        the final state when done=True.

        Args:
            env: an OpenAI Gym environment.
            policy: The output of a deep neural network
            render: Whether to store frames from the environment
            Returns:
            states: a list of states visited by the agent.
            actions: a list of actions taken by the agent. Note that these actions should never actually be trained on...
            timesteps: a list of integers, where timesteps[i] is the timestep at which states[i] was visited.
            rewards: list of rewards given by the environment
            rgbs: list of rgb images from the environment for each timestep
        """

        states, old_actions, timesteps, rewards, rgbs, pol_states, episode_starts = (
            [],
            [],
            [],
            [],
            [],
            ([], []),
            [],
        )

        done, trunc = False, False
        cur_state, _ = env.reset()
        cur_pol_state = None
        cur_episode_start = np.ones((1,), dtype=bool)
        t = 0
        while (not done) and (not trunc):
            with torch.no_grad():
                action, next_pol_state = policy.predict(
                    cur_state, cur_pol_state, cur_episode_start
                )
            a = action
            next_state, reward, done, trunc, _ = env.step(a)

            states.append(cur_state)
            if cur_pol_state:
                pol_states[0].append(cur_pol_state[0])
                pol_states[1].append(cur_pol_state[1])
            else:
                pol_states[0].append(None)
                pol_states[1].append(None)
            episode_starts.append(cur_episode_start.item())
            old_actions.append(a)
            timesteps.append(t)
            rewards.append(reward)

            t += 1

            cur_state = next_state
            cur_pol_state = next_pol_state
            episode_starts[0] = done

        pol_states[0][0] = np.zeros_like(pol_states[0][1])
        pol_states[1][0] = np.zeros_like(pol_states[1][1])

        return states, old_actions, timesteps, rewards, rgbs, pol_states, episode_starts

    def call_expert_policy(self, states):
        """
        Calls the expert policy to get an action.

        Args:
            state: the current state of the environment.
        """
        # takes in a np array state and returns an np array action
        with torch.no_grad():
            state_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.expert_policy.predict(state_tensor, deterministic=True)[0]

        assert state_tensor.shape[0] == actions.shape[0]
        return actions

    def update_training_data(self, num_trajectories_per_batch_collection=20):
        """
        Updates the training data by collecting trajectories from the current policy and the expert policy.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        NOTE: you will need to call self.generate_trajectory and self.call_expert_policy in this function.
        NOTE: you should update self.states, self.actions, and self.timesteps in this function.
        """
        # BEGIN STUDENT SOLUTION
        new_states = []
        new_actions = []
        new_timesteps = []
        next_pol_states = ([], [])
        new_episode_starts = []

        for i in range(num_trajectories_per_batch_collection):
            states, actions, timesteps, rewards, rgbs, pol_states, episode_starts = (
                self.generate_trajectory(self.env, self.policy)
            )

            expert_actions = self.call_expert_policy(states)

            new_states.append(states)
            new_actions.append(expert_actions)
            new_timesteps.append(timesteps)
            next_pol_states[0].append(pol_states[0])
            next_pol_states[1].append(pol_states[1])
            new_episode_starts.append(episode_starts)

        new_states_T_S = np.concatenate(new_states, axis=0)
        new_actions_T_A = np.concatenate(new_actions, axis=0)
        new_timesteps_T = np.concatenate(new_timesteps, axis=0)
        new_pol_states_T_S = (
            np.concatenate(next_pol_states[0], axis=0),
            np.concatenate(next_pol_states[1], axis=0),
        )
        new_episode_starts_T = np.concatenate(new_episode_starts, axis=0)

        if self.states is None:
            assert self.actions is None and self.timesteps is None
            self.states = new_states_T_S
            self.actions = new_actions_T_A
            self.timesteps = new_timesteps_T
            self.policy_states = RNNStates(*new_pol_states_T_S)
            self.episode_starts = new_episode_starts_T
        else:
            self.states = np.append(self.states, new_states_T_S, axis=0)
            self.actions = np.append(self.actions, new_actions_T_A, axis=0)
            self.timesteps = np.append(self.timesteps, new_timesteps_T, axis=0)
            self.policy_states = RNNStates(
                np.append(self.policy_states[0], new_pol_states_T_S[0], axis=0),
                np.append(self.policy_states[1], new_pol_states_T_S[1], axis=0),
            )
            self.episode_starts = np.append(
                self.episode_starts, new_episode_starts_T, axis=0
            )

        # return rewards

    def generate_trajectories(self, num_trajectories_per_batch_collection=20):
        """
        Runs inference for a certain number of trajectories.

        Args:
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy.

        Returns:
            average reward per trajectory in a list

        NOTE: you will need to call self.generate_trajectory in this function.
        """
        # BEGIN STUDENT SOLUTION
        rewards = torch.zeros((num_trajectories_per_batch_collection))
        for i in range(num_trajectories_per_batch_collection):
            _, _, _, traj_rewards, _, _, _ = self.generate_trajectory(
                self.env, self.policy
            )
            rewards[i] = torch.tensor(sum(traj_rewards))

        # END STUDENT SOLUTION

        return rewards

    def train(
        self,
        num_batch_collection_steps=20,
        num_training_steps_per_batch_collection=1000,
        num_trajectories_per_batch_collection=20,
        batch_size=64,
        print_every=500,
        save_every=10000,
        wandb_logging=False,
    ):
        """
        Train the model using DAgger

        Args:
            num_batch_collection_steps: the number of times to collecta batch of trajectories from the current policy.
            num_training_steps_per_batch_collection: the number of times to train the model per batch collection.
            num_trajectories_per_batch_collection: the number of trajectories to collect from the current policy per batch.
            batch_size: the batch size to use for training.
            print_every: how often to print the loss during training.
            save_every: how often to save the model during training.
            wandb_logging: whether to log the training to wandb.

        NOTE: for DAgger, you will need to call the self.training_step and self.update_training_data function.
        """

        losses = np.zeros(
            num_batch_collection_steps * num_training_steps_per_batch_collection
        )
        self.policy.train()
        mean_rewards, median_rewards, max_rewards = [], [], []
        # BEGIN STUDENT SOLUTION
        pbar = tqdm(range(num_batch_collection_steps), desc="Batch")
        for i in pbar:
            self.update_training_data(num_trajectories_per_batch_collection)

            pbar_train = tqdm(
                range(num_training_steps_per_batch_collection), desc="Train Loop"
            )
            for j in pbar_train:
                losses[i * num_training_steps_per_batch_collection + j] = (
                    self.training_step(batch_size=batch_size)
                )
                if (j % 10) == 0:
                    pbar_train.set_postfix(
                        {
                            "Loss": losses[
                                i * num_training_steps_per_batch_collection + j
                            ]
                        }
                    )
            # eval
            self.policy.eval()
            rewards = self.generate_trajectories(num_trajectories_per_batch_collection)
            mean_rewards.append(torch.mean(rewards).item())
            median_rewards.append(torch.median(rewards).item())
            max_rewards.append(torch.max(rewards).item())
            out = {
                "Mean Reward": mean_rewards[-1],
                "Max Reward": max_rewards[-1],
                "Loss": {
                    losses[
                        i
                        * num_training_steps_per_batch_collection : (i + 1)
                        * num_training_steps_per_batch_collection
                    ].mean()
                },
            }
            print(out)
            pbar.set_postfix(out)

            self.policy.train()

        # END STUDENT SOLUTION

        return losses

    def training_step(
        self,
        batch_size,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
    ):
        """
        Simple training step implementation

        Args:
            batch_size: the batch size to use for training.
        """
        states, actions, timesteps, pol_states, episode_starts = (
            self.get_training_batch(batch_size=batch_size)
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        timesteps = timesteps.to(self.device)

        self.optimizer.zero_grad()
        # policy.evaluate_actions's type signatures are incorrect.
        predicted_actions, _, _, _ = self.policy.forward(
            obs=states, lstm_states=pol_states, episode_starts=episode_starts  # type: ignore[arg-type]
        )
        loss = self.loss_fn(predicted_actions.squeeze(), actions.squeeze())
        loss.backward()
        # log_prob = log_prob.mean()
        # entropy = entropy.mean() if entropy is not None else None

        # l2_norms = [torch.sum(torch.square(w)) for w in self.policy.parameters()]
        # l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # # # sum of list defaults to float(0) if len == 0.
        # assert isinstance(l2_norm, torch.Tensor)

        # ent_loss = -ent_weight * (entropy if entropy is not None else torch.zeros(1))
        # neglogp = -log_prob
        # print(neglogp)
        # l2_loss = l2_weight * l2_norm
        # loss = neglogp + ent_loss + l2_loss
        # loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def get_training_batch(self, batch_size=64):
        """
        get a training batch

        Args:
            batch_size: the batch size to use for training.
        """
        # get random states, actions, and timesteps
        indices = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = torch.tensor(self.states[indices], device=self.device).float()
        actions = torch.tensor(self.actions[indices], device=self.device).float()
        timesteps = torch.tensor(self.timesteps[indices], device=self.device)
        pol_states = RNNStates(
            torch.tensor(self.policy_states[0][indices], device=self.device).float(),
            torch.tensor(self.policy_states[1][indices], device=self.device).float(),
        )
        episode_starts = torch.tensor(
            self.episode_starts[indices], device=self.device
        ).float()

        return states, actions, timesteps, pol_states, episode_starts


class RecurrentDaggerWrapper(PolicyWrapper):

    def __init__(
        self,
        reference_traj_path,
        policy: RecurrentActorCriticPolicy,
        **kwargs,
    ):
        super().__init__(**kwargs)

        rng = np.random.default_rng(0)

        device = "cpu"

        self.policy: RecurrentActorCriticPolicy = policy

        expert = StanleyPolicy(venv=self.env, reference_traj_path=reference_traj_path)

        optim = torch.optim.AdamW(
            params=policy.parameters(), lr=0.0001, weight_decay=0.0001
        )

        self.trainer = TrainDagger(
            env=self.env,
            policy=self.policy,
            optimizer=optim,
            expert_policy=expert,
            device=device,
        )

    def train(self, timesteps):
        print("Training DAgger model...")
        losses = self.trainer.train(
            num_batch_collection_steps=timesteps // 1000,
            num_training_steps_per_batch_collection=1000,
            num_trajectories_per_batch_collection=20,
            batch_size=128,
        )
        print(f"Final Loss: {losses[-1]}")
        print("Training complete. ")

    def save(
        self,
    ):
        self.policy.save(f"{self.dirpath}/dagger_saved.pt")

    def load(self):
        print("Loading existing DAgger model...")
        self.policy.load(f"{self.dirpath}/dagger_saved.pt", self.policy.device)
