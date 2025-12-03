import numpy as np
import pandas as pd

from src.envs.firewall_env import FirewallEnv
from src.utils.results_logger import update_agent_results
from src.utils.metrics import compute_confusion_metrics


def evaluate_random_agent(data_path, max_steps=None, num_episodes=3):
    # Load preprocessed env dataset
    df_env = pd.read_csv(data_path)

    # Create environment
    env = FirewallEnv(df_env, max_steps=max_steps)

    reward_curve = []

    first_cm = None
    last_cm = None
    first_metrics = None
    last_metrics = None

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

        # Confusion matrix for this episode
        cm = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }

        # Episode metrics using shared util
        metrics = compute_confusion_metrics(tp, fp, tn, fn)
        metrics["reward"] = float(episode_reward)

        precision = metrics["precision"]
        recall = metrics["recall"]
        accuracy = metrics["accuracy"]
        f1 = metrics["f1"]

        print(f"\n=== Episode {ep + 1} ===")
        print(f"Reward: {episode_reward:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"Precision={precision:.3f}, Recall={recall:.3f}, Accuracy={accuracy:.3f}, F1={f1:.3f}")

        # Save first episode metrics
        if ep == 0:
            first_cm = cm
            first_metrics = metrics.copy()

        # Always update last episode metrics
        last_cm = cm
        last_metrics = metrics.copy()

        reward_curve.append(float(episode_reward))

    avg_reward = float(np.mean(reward_curve))

    print("\n========== RANDOM AGENT SUMMARY ==========")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward per episode: {avg_reward:.2f}")
    print("First episode metrics:")
    print(f"  Reward   : {first_metrics['reward']:.2f}")
    print(f"  Precision: {first_metrics['precision']:.3f}")
    print(f"  Recall   : {first_metrics['recall']:.3f}")
    print(f"  Accuracy : {first_metrics['accuracy']:.3f}")
    print(f"  F1       : {first_metrics['f1']:.3f}")
    print("Last episode metrics:")
    print(f"  Reward   : {last_metrics['reward']:.2f}")
    print(f"  Precision: {last_metrics['precision']:.3f}")
    print(f"  Recall   : {last_metrics['recall']:.3f}")
    print(f"  Accuracy : {last_metrics['accuracy']:.3f}")
    print(f"  F1      : {first_metrics['f1']:.3f}")

    # Log to results.json using the new general logger
    update_agent_results(
        agent_name="Random",
        agent_type="baseline_random",
        episodes=num_episodes,
        reward_curve=reward_curve,
        first_confusion_matrix=first_cm,
        last_confusion_matrix=last_cm,
        first_metrics=first_metrics,
        last_metrics=last_metrics,
        avg_reward=avg_reward,
        hyperparameters=None,
    )


if __name__ == "__main__":
    evaluate_random_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=None,
        num_episodes=3,
    )
