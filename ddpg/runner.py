#!/usr/bin/env python
# coding: utf-8
"""Functions train/visualize models."""

# Standard imports
import random
from collections import deque
from typing import (
    List,
)

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import (
    nn,
    optim,
    Tensor,
)
from unityagents import UnityEnvironment

# Project imports
from .models import QNet, PolicyNet
from .utils import env_reset, env_step, process_action, noise_stdev_generator

# Define a global path to the Unity environment (can be parameterized later, if needed)
APP_PATH = 'Reacher.app'


# Training-specific helpers
def _copy_weights(net: nn.Module, target_net: nn.Module, tau: float = 0.001):
    """Update target_dqn model parameters.

    (Function based on Udacity DeepRL DQN homework code).
    θ_target_dqn = [τ * θ_dqn] + [(1 - τ) * θ_target_dqn]

    Args:
        net: The model to copy params from.
        target_net: The model to mix params into.
        tau: The mixing coefficient (what percentage of the source weights to use).
    """
    for net_params, target_net_params in zip(net.parameters(), target_net.parameters()):
        target_net_params.data.copy_(tau * net_params.data + (1.0 - tau) * target_net_params.data)


# Replay Buffer
class ReplayBuffer:
    """Stores (S, A, R, S') tuples observed during training."""
    def __init__(self, maxlen: int = 1e6):
        # Arbitrarily keep a max of 1M tuples by default (could be tuned in the future)
        self._maxlen = maxlen
        self._buffer = deque(maxlen=self._maxlen)

    def add(self, state: List[float], action: List[float], reward: float, next_state: List[float], done: bool):
        # Store (s, a, r, s', done) pairs
        self._buffer.append((state, action, reward, next_state, int(done)))

    def get_batch(self, batch_size) -> tuple:
        sample = random.sample(self._buffer, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)

        return (
            Tensor(states).float(),
            Tensor(actions).float(),
            Tensor(rewards).float().unsqueeze(-1),
            Tensor(next_states).float(),
            Tensor(dones).float().unsqueeze(-1),
        )

    def __len__(self):
        return len(self._buffer)


# Define main functions
def train_model():
    """Trains a DDPG model."""
    # TODO These could be passed in as arguments, if desired
    # Constants for General setup
    SCORE_WINDOW_SIZE = 100     # The window size to use when calculating the running average
    AVERAGE_SCORE_GOAL = 30     # The running average goal (determines if the environment has been solved)

    # Hyperparams
    RANDOM_EXPLORATION_SPAN = int(10e3)     # The number of steps to do random exploration for before using policy
    MAX_EPISODES = int(5e3)                 # The maximum number of episode sto run for
    BATCH_SIZE = 64                         # The minibatch size to use during SGD
    UPDATE_FREQ = 5                         # How frequently (in terms of steps) the models should be updated
    NUM_INNER_UPDATE = 5                    # When updating the models, how many SGD steps to run
    NOISE_STDEV = 1.0                       # The (initial) standard deviation to use when sampling noise (Normal dist.)
    MAX_REPLAY_SIZE = int(200e3)            # The maximum number of SARS' tuples to store during training (FIFO)
    GAMMA = 0.99                            # The future reward discount factor
    TAU = 1e-2                              # The (soft) mixing factor to use when updating target network weights
    LR_POLICY = 1e-3                        # The learning rate to use for the policy net ("Actor")
    LR_Q = 1e-2                             # The learning rate to use for the q net ("Critic")
    WEIGHT_DECAY = 0.0                      # The strength of L2 regularization

    # Set up environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    noise_stdev_gen = noise_stdev_generator(starting_stdev=NOISE_STDEV)

    # Set up models
    hidden_sizes = [256, 128]

    policy_net = PolicyNet(hidden_sizes=hidden_sizes).to(device)            # "Actor"
    policy_net_target = PolicyNet(hidden_sizes=hidden_sizes).to(device)     # "Actor" target network
    _copy_weights(policy_net, policy_net_target, 1.0)

    q_net = QNet(hidden_sizes=hidden_sizes).to(device)          # "Critic"
    q_net_target = QNet(hidden_sizes=hidden_sizes).to(device)   # "Critic" target network
    _copy_weights(q_net, q_net_target, 1.0)

    # Enable nets for training (for batchnorm)
    policy_net.train()
    q_net.train()

    # Set up training params and optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=LR_POLICY, weight_decay=WEIGHT_DECAY)
    q_optimizer = optim.Adam(q_net.parameters(), lr=LR_Q, weight_decay=WEIGHT_DECAY)

    # Set up required data structures (replay buffer and episode score trackers)
    replay_buffer = ReplayBuffer(MAX_REPLAY_SIZE)
    scores = []
    scores_window = deque(maxlen=SCORE_WINDOW_SIZE)

    # Enable random exploration to start (based on RANDOM_EXPLORATION_SPAN)
    random_exploration_enabled = True

    # Act and learn over series of episodes
    for episode in range(MAX_EPISODES):
        episode_score = 0.0             # A running total of the episode score
        done = False                    # If set to True, episode is finished
        q_loss_log = []                 # For tracking MSE of DQN/critic (to minimize)
        expected_policy_value_log = []  # For tracking expected value of policy/actor (to maximize)
        t = 0                           # For tracking the current timestep within the episode

        # Sample a new noise stdev every episode
        episode_noise_stdev_sample = NOISE_STDEV if random_exploration_enabled else next(noise_stdev_gen)

        # Reset the environment and start a new episode
        state = env_reset(env, brain_name)
        while not done:
            t += 1

            # Convert state np.array to Tensor
            state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(device)

            # Here, we use random actions for a bit
            # (as mentioned here: https://spinningup.openai.com/en/latest/algorithms/ddpg.html)
            do_random_exploration = len(replay_buffer) < RANDOM_EXPLORATION_SPAN
            have_enough_data = len(replay_buffer) >= BATCH_SIZE
            update_on_this_timestep = t % UPDATE_FREQ == 0

            if do_random_exploration:
                action = process_action(np.random.rand(action_size) * 2 - 1, noise_stdev=0.0)
            else:
                if random_exploration_enabled:
                    print('Disabling random exploration...')
                    random_exploration_enabled = False

                # Choose an action by feeding state through "Actor" net
                policy_net.eval()
                with torch.no_grad():
                    action = policy_net.forward(state_tensor).cpu().detach().numpy().squeeze(0)
                action = process_action(action, noise_stdev=episode_noise_stdev_sample)
                policy_net.train()

            # Step in env
            next_state, reward, done = env_step(env, brain_name, action)
            episode_score += reward

            # Store in replay buffer
            replay_buffer.add(state, action.tolist(), reward, next_state, done)
            state = next_state

            # Update if appropriate
            if update_on_this_timestep and have_enough_data:
                for _ in range(NUM_INNER_UPDATE):
                    batch_groups = list(replay_buffer.get_batch(BATCH_SIZE))
                    for k, element in enumerate(batch_groups):
                        batch_groups[k] = element.to(device)

                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch_groups

                    # CALCULATE TARGETS
                    # Get Actor's selection for next best action when in next_state
                    target_next_best_actions = policy_net_target(next_state_batch)

                    # Get Critic's predicted Q value for being in next_state and taking next_best_action
                    target_q_next_state = q_net_target(next_state_batch, target_next_best_actions)

                    # Compute the targets and update both Actor and Critic networks
                    # Note: we don't want to include the predicted value for the next state if we're in a terminal state
                    terminal_state_cancellation = 1 - done_batch
                    target_q = reward_batch + (GAMMA * terminal_state_cancellation * target_q_next_state)

                    # GENERATING PREDICTIONS + APPLYING GRADIENT UPDATES
                    # Update Critic
                    # Minimize mean-squared error between the predicted and target Q(s, a)
                    predicted_q = q_net(state_batch, action_batch)
                    q_loss = F.mse_loss(predicted_q, target_q)
                    q_optimizer.zero_grad()
                    q_loss.backward()  # Minimize MSE between Q and (bootstrapped) target Q
                    q_optimizer.step()

                    # Update Actor
                    # Disable grads for "Critic" (since we don't need them when updating "Actor")
                    q_net.set_requires_grad(False)

                    # Calculate Q
                    next_best_actions = policy_net(state_batch)
                    expected_policy_value = torch.mean(q_net(state_batch, next_best_actions))
                    policy_optimizer.zero_grad()
                    (-expected_policy_value).backward()  # Maximize the expected Q-value of our policy
                    policy_optimizer.step()

                    # Re-enable grads for "Critic" for future backprop calculations
                    q_net.set_requires_grad(True)

                    # Log the two losses
                    q_loss_log.append(q_loss.item())
                    expected_policy_value_log.append(expected_policy_value.item())

                    # UPDATE THE TARGET NETS
                    _copy_weights(policy_net, policy_net_target, TAU)
                    _copy_weights(q_net, q_net_target, TAU)

        # Episode is done
        scores.append(episode_score)
        scores_window.append(episode_score)
        q_loss_avg = np.mean(q_loss_log) if q_loss_log else float('inf')
        expected_policy_value_avg = np.mean(expected_policy_value_log) if expected_policy_value_log else float('-inf')
        score_mean = np.mean(scores_window) if scores_window else 0.0
        print(f"Episode {episode}/{MAX_EPISODES} ({t} steps): Average reward (last 100): {score_mean:.4f} ; "
              f"(Q Loss Avg: {q_loss_avg:.8f} ; E(V) Avg: {expected_policy_value_avg:.4f}) "
              f"[Noise stdev: {episode_noise_stdev_sample}]")

        # Leave the loop if we've reached our goal
        if score_mean >= AVERAGE_SCORE_GOAL:
            print(f"Environment solved at {episode}")
            break

    # Save models
    torch.save(policy_net.state_dict(), 'policy_net.pth')
    torch.save(q_net.state_dict(), 'q_net.pth')

    # Show plot of reward per episode
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.show()


def view_agent(policy_path: str):
    """Visualizes an episode for a pretrained DDPG agent"""
    hidden_sizes = [256, 128]
    policy_net = PolicyNet(hidden_sizes=hidden_sizes)
    policy_net.load_state_dict(torch.load(policy_path))
    policy_net.eval()

    env = UnityEnvironment(file_name=APP_PATH)
    brain_name = env.brain_names[0]

    state = env_reset(env, brain_name, train_mode=False)
    done = False
    total_reward = 0.0
    while not done:
        state_tensor = torch.from_numpy(state).unsqueeze(0).float()
        action = policy_net.forward(state_tensor).detach().numpy().squeeze(0)
        next_state, reward, done = env_step(env, brain_name, action)
        total_reward += reward
        state = next_state

    print(f"Total reward in example episode: {total_reward}")
