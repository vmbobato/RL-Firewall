import numpy as np
import pandas as pd
from collections import defaultdict

from src.envs.firewall_env import FirewallEnv
from src.utils.results_logger import update_agent_results
from src.utils.metrics import compute_confusion_metrics


def evaluate_sarsa_agent(
    data_path="data/preprocessed/Binerized_features.csv",
    max_steps=None,
    num_episodes=10,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    epsilon_min=0.01,
    epsilon_decay=0.995,
):
    """
    Train and evaluate a Tabular SARSA agent on the FirewallEnv for num_episodes.
    On-policy learning with epsilon-greedy action selection, with default parameters provided.
    """

    # Load the data file and the environment
    df_env = pd.read_csv(data_path)
    env = FirewallEnv(df_env, max_steps=max_steps)

    # Define the initial variables
    Q = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))
    reward_curve = []
    first_cm = None
    first_metrics = None

    # Run the training episodes
    for ep in range(num_episodes):

        # Define the initial variables
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0
        tp = fp = tn = fn = 0

        # Convert the observation into a tuple so that it can index the Q-table
        state = tuple(int(x) for x in obs)

        # Choose the initial epsilon-greedy action
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        while True:

            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)

            # Update the state and reward from the observation
            episode_reward += reward
            next_state = tuple(int(x) for x in next_obs)

            # Choose the next epsilon-greedy action for the next state
            # Not neccessarily the best action
            if not done:
                if np.random.rand() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = int(np.argmax(Q[next_state]))
            else:
                next_action = 0

            # Update the Q-value using the SARSA update rule
            if done:
                temporal_diff_target = reward
            else:
                temporal_diff_target = reward + gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (temporal_diff_target - Q[state][action])

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

            # Update both the state and action
            state = next_state
            action = next_action

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
        print(f"\n=== SARSA Training Episode {ep + 1} ===")
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
    print("\n=== Running SARSA Greedy Evaluation (epsilon = 0) ===")

    # Define the initial variables
    obs, info = env.reset(seed=999)
    eval_reward = 0.0
    tp = fp = tn = fn = 0

    # Convert the observation into a tuple so that it can index the Q-table
    state = tuple(int(x) for x in obs)

    while True:

        # Choose the greedy action
        action = int(np.argmax(Q[state]))

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

    print("\n=== SARSA Greedy Evaluation Results ===")
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
        agent_name="SARSA",
        agent_type="tabular_sarsa",
        episodes=num_episodes,
        reward_curve=reward_curve,
        first_confusion_matrix=first_cm,
        last_confusion_matrix=eval_cm,
        first_metrics=first_metrics,
        last_metrics=eval_metrics,
        avg_reward=avg_reward,
        hyperparameters={
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        },
    )


if __name__ == "__main__":
    evaluate_sarsa_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=None,
        num_episodes=500,
    )