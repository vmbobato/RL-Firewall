import numpy as np
import pandas as pd
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.firewall_env import FirewallEnv
from src.utils.results_logger import update_agent_results
from src.utils.metrics import compute_confusion_metrics


class DQNNetwork(nn.Module):
    """
    A simple feedforward neural network that maps the state features to the Q-values for each action.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_dqn_agent(
    data_path: str = "data/preprocessed/Binerized_features.csv",
    max_steps=None,
    num_episodes: int = 50,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    replay_capacity: int = 50_000,
    target_update_freq: int = 500,
):
    """
    Train and evaluate a DQN agent on the FirewallEnv for num_episodes.
    Off-policy learning with epsilon-greedy action selection, with default parameters provided.
    Uses a policy network to choose actions and a target network to calculate target Q-values.
    """

    # Load the data file and the environment
    df_env = pd.read_csv(data_path)
    env = FirewallEnv(df_env, max_steps=max_steps)

    # Define the device being used, allows for the GPU to be used if its available to speed up training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the state dimension from an observation
    obs, info = env.reset(seed=0)
    state_dim = len(obs)
    action_dim = env.action_space.n

    # Initialize the policy and target networks
    policy_network = DQNNetwork(state_dim, action_dim).to(device)
    target_network = DQNNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Initialize the replay buffer
    replay_buffer = deque(maxlen=replay_capacity)

    # Define the other initial variables
    reward_curve = []
    first_cm = None
    first_metrics = None
    epsilon = epsilon_start
    global_step = 0

    # Helper function to convert the state tuple or array into a tensor
    def state_to_tensor(state):
        # Convert into a numpy array first
        x = np.array(state, dtype=np.float32)
        # Then convert to a tensor
        return torch.from_numpy(x).unsqueeze(0).to(device)

    # Run the training episodes
    for ep in range(num_episodes):

        # Define the initial variables
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0
        tp = fp = tn = fn = 0

        # Convert the observation into a tuple
        state = tuple(int(x) for x in obs)

        while True:

            # Increment the global step
            global_step += 1

            # Choose the epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_network(state_to_tensor(state))
                    action = int(torch.argmax(q_vals, dim=1).item())

            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)

            # Update the state and reward from the observation
            next_state = tuple(int(x) for x in next_obs)
            episode_reward += reward

            # Store the transition in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            # Update the confusion matrix
            label = info["label"]
            if label == 1 and action == 1:
                tp += 1
            elif label == 1 and action == 0:
                fn += 1
            elif label == 0 and action == 1:
                fp += 1
            elif label == 0 and action == 0:
                tn += 1

            # Update the state
            state = next_state

            # Actual training step 
            # Check if there's enough data in the replay buffer
            if len(replay_buffer) >= batch_size:

                # Sample a random minibatch of transitions
                batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
                batch_states = []
                batch_actions = []
                batch_rewards = []
                batch_next_states = []
                batch_dones = []

                # Unpack the batch
                for idx in batch:
                    s, a, r, s_next, d = replay_buffer[idx]
                    batch_states.append(s)
                    batch_actions.append(a)
                    batch_rewards.append(r)
                    batch_next_states.append(s_next)
                    batch_dones.append(d)

                # Convert everything to tensors
                states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(batch_actions, dtype=torch.int64, device=device).unsqueeze(-1)
                rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=device).unsqueeze(-1)
                next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=device)
                dones_tensor = torch.tensor(batch_dones, dtype=torch.float32, device=device).unsqueeze(-1)

                # Calculate the current Q-values from the policy network
                q_values = policy_network(states_tensor).gather(1, actions_tensor)

                # Calculate the target Q-values from the target network
                with torch.no_grad():
                    next_q_vals = target_network(next_states_tensor)
                    max_next_q_vals, _ = torch.max(next_q_vals, dim=1, keepdim=True)
                    # target Q-value is the temporal difference target
                    target_q = rewards_tensor + gamma * (1.0 - dones_tensor) * max_next_q_vals

                # Calculate the loss between the current predicted Q-value and the target Q-value and optimize
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Periodically update the target network
            if global_step % target_update_freq == 0:
                target_network.load_state_dict(policy_network.state_dict())

            # Break out of the loop if the terminal state is reached
            if done:
                break

        # Calculate the episode metrics
        cm = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
        metrics = compute_confusion_metrics(tp, fp, tn, fn)
        metrics["reward"] = float(episode_reward)

        # Store the first episode results
        if ep == 0:
            first_cm = cm
            first_metrics = metrics.copy()

        # Add the episode reward to the reward curve
        reward_curve.append(float(episode_reward))

        # Print the episode summary
        print(f"\n=== DQN Training Episode {ep + 1} ===")
        print(f"Reward: {episode_reward:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"Accuracy={metrics['accuracy']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        )

        # Decay the epsilon after each episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Calculate the average reward across all of the training episodes
    avg_reward = float(np.mean(reward_curve))

    # Run the Greedy Evaluation (epsilon = 0)
    print("\n=== Running DQN Greedy Evaluation (epsilon = 0) ===")

    # Define the initial variables
    obs, info = env.reset(seed=999)
    eval_reward = 0.0
    tp = fp = tn = fn = 0

    # Convert the observation into a tuple
    state = tuple(int(x) for x in obs)

    while True:

        # Choose the greedy action
        with torch.no_grad():
            q_vals = policy_network(state_to_tensor(state))
            action = int(torch.argmax(q_vals, dim=1).item())

        # Take a step in the environment
        next_obs, reward, done, info = env.step(action)

        # Update the state and reward from the observation
        next_state = tuple(int(x) for x in next_obs)
        eval_reward += reward

        # Update the confusion matrix
        label = info["label"]
        if label == 1 and action == 1:
            tp += 1
        elif label == 1 and action == 0:
            fn += 1
        elif label == 0 and action == 1:
            fp += 1
        elif label == 0 and action == 0:
            tn += 1

        # Update the state
        state = next_state

        # Break out of the loop if the terminal state is reached
        if done:
            break

    # Print out the final evaluation metrics
    eval_cm = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    eval_metrics = compute_confusion_metrics(tp, fp, tn, fn)
    eval_metrics["reward"] = float(eval_reward)

    print("\n=== DQN Greedy Evaluation Results ===")
    print(f"Reward: {eval_reward:.2f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(
        f"Precision={eval_metrics['precision']:.3f}, "
        f"Recall={eval_metrics['recall']:.3f}, "
        f"Accuracy={eval_metrics['accuracy']:.3f}, "
        f"F1={eval_metrics['f1']:.3f}"
    )

    # Log the results
    update_agent_results(
        agent_name="DQN",
        agent_type="deep_q_network",
        episodes=num_episodes,
        reward_curve=reward_curve,
        first_confusion_matrix=first_cm,
        last_confusion_matrix=eval_cm,
        first_metrics=first_metrics,
        last_metrics=eval_metrics,
        avg_reward=avg_reward,
        hyperparameters={
            "gamma": gamma,
            "lr": lr,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
            "replay_capacity": replay_capacity,
            "target_update_freq": target_update_freq,
        },
    )


if __name__ == "__main__":
    evaluate_dqn_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=None,
        num_episodes=100,
    )
