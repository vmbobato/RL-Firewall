import numpy as np
import pandas as pd
import argparse
from src.envs.firewall_env import FirewallEnv
from src.utils.results_logger import update_agent_results
from src.utils.metrics import compute_confusion_metrics


# Same feature order as in FirewallEnv.build_state
FLOW_DURATION_IDX = 0
TOT_FWD_PKTS_IDX = 1
TOT_BWD_PKTS_IDX = 2
PKT_LEN_MEAN_IDX = 3
FLOW_BYTES_BIN_IDX = 4
ACK_FLAGS_IDX = 5
INIT_FWD_WIN_IDX = 6
SYN_FLAGS_IDX = 7
SYN_RATIO_IDX = 8


def get_state_key(obs):
    """
    Convert observation array to a hashable state key for the Q-table.
    """
    return tuple(int(x) for x in obs)


def epsilon_greedy_action(q_table, state_key, n_actions, epsilon, rng):
    """
    Epsilon-greedy policy over the Q-table.

    q_table: dict[(state_key, action)] -> Q-value
    state_key: tuple representing the discrete state
    n_actions: number of discrete actions (2: allow/deny)
    epsilon: exploration rate
    rng: np.random.Generator
    """
    if rng.random() < epsilon:
        # Explore
        return rng.integers(0, n_actions)

    # Exploit: pick argmax_a Q(s,a)
    q_values = [
        q_table.get((state_key, a), 0.0) for a in range(n_actions)
    ]
    max_q = max(q_values)
    # In case of ties, randomly pick one of the best actions
    best_actions = [a for a, q in enumerate(q_values) if q == max_q]
    return int(rng.choice(best_actions))


def train_q_learning_agent(
    data_path: str,
    max_steps: int = None,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
):
    """
    Train a tabular Q-learning agent on the FirewallEnv and log results.

    Training:
      - num_episodes passes through the dataset with epsilon-greedy policy.
    Evaluation:
      - 1 extra episode with epsilon = 0 (greedy policy), used as "last" metrics.

    Logged:
      - first: episode 1 training metrics
      - last: final greedy evaluation metrics
      - reward_curve: list of training episode rewards
    """
    # Load data and create environment
    df_env = pd.read_csv(data_path)
    env = FirewallEnv(df_env, max_steps=max_steps)

    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n

    # Q-table: dict[(state_key, action)] -> float
    q_table = {}

    reward_curve = []

    first_cm = None
    first_metrics = None

    # ---------- Training loop ----------
    epsilon = epsilon_start

    for ep in range(num_episodes):
        obs, info = env.reset()
        state_key = get_state_key(obs)

        episode_reward = 0.0
        tp = fp = tn = fn = 0

        while True:
            # Select action via epsilon-greedy
            action = epsilon_greedy_action(q_table, state_key, n_actions, epsilon, rng)

            next_obs, reward, terminated, info = env.step(action)
            next_state_key = get_state_key(next_obs)
            episode_reward += reward

            label = info["label"]

            # Confusion matrix update for this episode
            if label == 1 and action == 1:
                tp += 1
            elif label == 1 and action == 0:
                fn += 1
            elif label == 0 and action == 1:
                fp += 1
            elif label == 0 and action == 0:
                tn += 1

            # Q-learning update
            old_q = q_table.get((state_key, action), 0.0)

            if terminated:
                target = reward  # no bootstrap if terminal
                new_q = old_q + alpha * (target - old_q)
                q_table[(state_key, action)] = new_q
                break
            else:
                # Bootstrap with max_a' Q(s', a')
                next_q_values = [
                    q_table.get((next_state_key, a), 0.0) for a in range(n_actions)
                ]
                max_next_q = max(next_q_values)

                target = reward + gamma * max_next_q
                new_q = old_q + alpha * (target - old_q)
                q_table[(state_key, action)] = new_q

            state_key = next_state_key

        # End of episode: compute metrics
        cm = {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
        metrics = compute_confusion_metrics(tp, fp, tn, fn)
        metrics["reward"] = float(episode_reward)

        reward_curve.append(float(episode_reward))

        # Save first-episode metrics
        if ep == 0:
            first_cm = cm
            first_metrics = metrics.copy()

        print(f"\n=== Q-Learning Training Episode {ep + 1} ===")
        print(f"Reward: {episode_reward:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"Accuracy={metrics['accuracy']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        )

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    avg_reward = float(np.mean(reward_curve))

    # ---------- Greedy evaluation (epsilon = 0) ----------
    print("\n=== Q-Learning Greedy Evaluation (epsilon=0) ===")
    obs, info = env.reset()
    state_key = get_state_key(obs)

    eval_reward = 0.0
    tp = fp = tn = fn = 0

    while True:
        # Greedy (no exploration)
        q_values = [q_table.get((state_key, a), 0.0) for a in range(n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        action = int(np.random.choice(best_actions))

        next_obs, reward, terminated, info = env.step(action)
        next_state_key = get_state_key(next_obs)
        eval_reward += reward

        label = info["label"]

        # Confusion matrix update
        if label == 1 and action == 1:
            tp += 1
        elif label == 1 and action == 0:
            fn += 1
        elif label == 0 and action == 1:
            fp += 1
        elif label == 0 and action == 0:
            tn += 1

        if terminated:
            break
        state_key = next_state_key

    eval_cm = {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}
    eval_metrics = compute_confusion_metrics(tp, fp, tn, fn)
    eval_metrics["reward"] = float(eval_reward)

    print(f"Greedy eval reward: {eval_reward:.2f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(
        f"Precision={eval_metrics['precision']:.3f}, "
        f"Recall={eval_metrics['recall']:.3f}, "
        f"Accuracy={eval_metrics['accuracy']:.3f}, "
        f"F1={eval_metrics['f1']:.3f}"
    )

    print("\n========== Q-LEARNING AGENT SUMMARY ==========")
    print(f"Training episodes: {num_episodes}")
    print(f"Avg training reward per episode: {avg_reward:.2f}")
    print(f"Greedy eval reward: {eval_reward:.2f}")

    # Log to results.json
    update_agent_results(
        agent_name="QLearning",
        agent_type="tabular_q_learning",
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
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "seed": seed,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-learning firewall agent")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per episode",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=1.0,
        help="Initial epsilon for e-greedy",
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        default=0.05,
        help="Minimum epsilon",
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        default=0.995,
        help="ε decay rate per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    train_q_learning_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
