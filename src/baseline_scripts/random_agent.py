import numpy as np
import pandas as pd
from pathlib import Path

from src.envs.firewall_env import FirewallEnv


def evaluate_random_agent(data_path, max_steps=None, num_episodes=3):
    # Load preprocessed env dataset
    df_env = pd.read_csv(data_path)

    # Create environment
    env = FirewallEnv(df_env, max_steps=max_steps)

    # Global metrics
    global_tp = global_fp = global_tn = global_fn = 0
    global_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0

        tp = fp = tn = fn = 0

        while True:
            # Random action: 0 = ALLOW, 1 = DENY
            action = env.action_space.sample()

            obs, reward, terminated, info = env.step(action)
            episode_reward += reward

            label = info["label"]  # 0 = benign, 1 = malicious

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

        # Episode metrics
        total = tp + fp + tn + fn + 1e-6
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        accuracy = (tp + tn) / total

        print(f"\n=== Episode {ep+1} ===")
        print(f"Reward: {episode_reward:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Precision={precision:.3f}, Recall={recall:.3f}, Accuracy={accuracy:.3f}")

        global_tp += tp
        global_fp += fp
        global_tn += tn
        global_fn += fn
        global_rewards.append(episode_reward)

    # Aggregate metrics
    g_total = global_tp + global_fp + global_tn + global_fn + 1e-6
    g_precision = global_tp / (global_tp + global_fp + 1e-6)
    g_recall = global_tp / (global_tp + global_fn + 1e-6)
    g_accuracy = (global_tp + global_tn) / g_total
    avg_reward = np.mean(global_rewards)

    print("\n========== RANDOM AGENT SUMMARY ==========")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward per episode: {avg_reward:.2f}")
    print(f"Global TP={global_tp}, FP={global_fp}, TN={global_tn}, FN={global_fn}")
    print(f"Global Precision={g_precision:.3f}")
    print(f"Global Recall   ={g_recall:.3f}")
    print(f"Global Accuracy ={g_accuracy:.3f}")


if __name__ == "__main__":
    evaluate_random_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=None,
        num_episodes=3
    )
