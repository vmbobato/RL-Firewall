import numpy as np
import pandas as pd

from src.envs.firewall_env import FirewallEnv
from src.utils.results_logger import update_agent_results
from src.utils.metrics import compute_confusion_metrics

# Flow_Byts/s,Flow_Duration_bin,Tot_Fwd_Pkts_bin,Tot_Bwd_Pkts_bin,Pkt_Len_Mean_bin,Flow_Bytes_s_bin,ACK_Flag_Cnt_bin,Init_Fwd_Win_Byts_bin,SYN_Flag_Cnt_bin,syn_ratio_bin,
FLOW_DURATION_IDX = 0
TOT_FWD_PKTS_IDX = 1
TOT_BWD_PKTS_IDX = 2
PKT_LEN_MEAN_IDX = 3
FLOW_BYTES_BIN_IDX = 4
ACK_FLAGS_IDX = 5
INIT_FWD_WIN_IDX = 6
SYN_FLAGS_IDX = 7
SYN_RATIO_IDX = 8


def rule_based_policy(obs):
    flow_duration_bin = int(obs[FLOW_DURATION_IDX])
    tot_fwd_pkts_bin = int(obs[TOT_FWD_PKTS_IDX])
    tot_bwd_pkts_bin = int(obs[TOT_BWD_PKTS_IDX])
    pkt_len_mean_bin = int(obs[PKT_LEN_MEAN_IDX])
    flow_bytes_bin = int(obs[FLOW_BYTES_BIN_IDX])
    ack_flags_bin = int(obs[ACK_FLAGS_IDX])
    init_fwd_win_bin = int(obs[INIT_FWD_WIN_IDX])
    syn_flags_bin = int(obs[SYN_FLAGS_IDX])
    syn_ratio_bin = int(obs[SYN_RATIO_IDX])
    
    # 1. SYN ratio is extremely predictive
    if syn_ratio_bin in (0, 1, 2):
        return 1  # DENY

    # 2) Strange ACK flag patterns (bins 1–4 are ~97–100% malicious)
    if ack_flags_bin in (1, 2, 3, 4):
        return 1

    # 3) Tot_Bwd_Pkts_bin 1–4 are ~90–99% malicious
    if tot_bwd_pkts_bin in (1, 2, 3, 4):
        return 1

    # 4) Very short flows are ~95% malicious (Flow_Duration_bin == 0)
    if flow_duration_bin == 0:
        return 1

    # 5) Flow_Bytes_s_bin in {1,2,4} are > 80% malicious
    if flow_bytes_bin in (1, 2, 4):
        return 1

    # Default: allow
    return 0  # ALLOW



def evaluate_rule_based_agent(data_path, max_steps=None, num_episodes=1):
    """
    Evaluate the rule-based firewall over num_episodes passes through the dataset.
    Because the policy is deterministic, episodes will be identical 
    """
    df_env = pd.read_csv(data_path)
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
            action = rule_based_policy(obs)
            obs, reward, terminated, info = env.step(action)
            episode_reward += reward

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

        # Confusion matrix for this episode
        cm = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        }

        # Metrics via shared utility
        metrics = compute_confusion_metrics(tp, fp, tn, fn)
        metrics["reward"] = float(episode_reward)

        print(f"\n=== Rule-Based Episode {ep + 1} ===")
        print(f"Reward: {episode_reward:.2f}")
        print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(
            f"Precision={metrics['precision']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"Accuracy={metrics['accuracy']:.3f}, "
            f"F1={metrics['f1']:.3f}"
        )

        # Save first episode metrics
        if ep == 0:
            first_cm = cm
            first_metrics = metrics.copy()

        # Update last episode metrics (for deterministic agent, same as first)
        last_cm = cm
        last_metrics = metrics.copy()

        reward_curve.append(float(episode_reward))

    avg_reward = float(np.mean(reward_curve))

    print("\n========== RULE-BASED AGENT SUMMARY ==========")
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
    print(f"  F1       : {last_metrics['f1']:.3f}")

    # Log to results.json
    update_agent_results(
        agent_name="RuleBased",
        agent_type="rule_based_threshold",
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
    evaluate_rule_based_agent(
        data_path="data/preprocessed/Binerized_features.csv",
        max_steps=None,
        num_episodes=1,
    )
