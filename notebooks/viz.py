import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

with open("./results/results.json", "r") as f:
    results = json.load(f)

def plot_confusion_matrix(cm_array, agent_name):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_array, cmap="Blues")
    
    plt.title(f"Confusion Matrix – {agent_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks([0, 1], ["Positive", "Negative"])
    plt.yticks([0, 1], ["Positive", "Negative"])

    # Print numbers inside cells
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, format(cm_array[i, j], ','),
                ha="center", va="center",
                fontsize=14, color="black"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()


# DQN

agent_name = "DQN"

agent_res = results[agent_name]
reward_curve = agent_res["reward_curve"]

plt.figure()
plt.plot(reward_curve)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"{agent_name} – Reward per Episode")
plt.show()


agent_name = "DQN"
cm = results[agent_name]["summary"]["first"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "DQN Agent")


agent_name = "DQN"
cm = results[agent_name]["summary"]["last"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "DQN Agent")


agent_name = "DQN"

first_metrics = results[agent_name]["summary"]["first"]
last_metrics  = results[agent_name]["summary"]["last"]

metric_names = ["precision", "recall", "accuracy", "f1"]

first_values = [first_metrics[m] for m in metric_names]
last_values  = [last_metrics[m]  for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, first_values, width, label="First episode")
plt.bar(x + width/2, last_values,  width, label="Last episode")

plt.xticks(x, [m.capitalize() for m in metric_names])
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title(f"{agent_name} – Metrics: First vs Last Episode")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Q-Learning Agent Plots

agent_name = "QLearning"

agent_res = results[agent_name]
reward_curve = agent_res["reward_curve"]

plt.figure()
plt.plot(reward_curve)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"{agent_name} – Reward per Episode")
plt.show()


agent_name = "QLearning"
cm = results[agent_name]["summary"]["first"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "Q-Learning Agent")


agent_name = "QLearning"
cm = results[agent_name]["summary"]["last"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "Q-Learning Agent")


agent_name = "QLearning"

first_metrics = results[agent_name]["summary"]["first"]
last_metrics  = results[agent_name]["summary"]["last"]

metric_names = ["precision", "recall", "accuracy", "f1"]

first_values = [first_metrics[m] for m in metric_names]
last_values  = [last_metrics[m]  for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, first_values, width, label="First episode")
plt.bar(x + width/2, last_values,  width, label="Last episode")

plt.xticks(x, [m.capitalize() for m in metric_names])
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title(f"{agent_name} – Metrics: First vs Last Episode")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# SARSA Agent Plots

agent_name = "SARSA"

agent_res = results[agent_name]
reward_curve = agent_res["reward_curve"]

plt.figure()
plt.plot(reward_curve)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"{agent_name} - Reward per Episode")
plt.show()


agent_name = "SARSA"
cm = results[agent_name]["summary"]["first"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "SARSA Agent")


agent_name = "SARSA"
cm = results[agent_name]["summary"]["last"]["confusion_matrix"]
cm_array = np.array([
    [cm["TP"], cm["FN"]],
    [cm["FP"], cm["TN"]]
])
plot_confusion_matrix(cm_array, "SARSA Agent")


agent_name = "SARSA"

first_metrics = results[agent_name]["summary"]["first"]
last_metrics  = results[agent_name]["summary"]["last"]

metric_names = ["precision", "recall", "accuracy", "f1"]

first_values = [first_metrics[m] for m in metric_names]
last_values  = [last_metrics[m]  for m in metric_names]

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, first_values, width, label="First episode")
plt.bar(x + width/2, last_values,  width, label="Last episode")

plt.xticks(x, [m.capitalize() for m in metric_names])
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title(f"{agent_name} – Metrics: First vs Last Episode")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Combined Plots

ql_rewards = results["QLearning"]["reward_curve"]
sarsa_rewards = results["SARSA"]["reward_curve"]

ql_episodes = range(1, len(ql_rewards) + 1)
sarsa_episodes = range(1, len(sarsa_rewards) + 1)

plt.figure(figsize=(10, 6))

plt.plot(ql_episodes, ql_rewards, label="Q-Learning")
plt.plot(sarsa_episodes, sarsa_rewards, label="SARSA")

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward per Episode: Q-Learning vs SARSA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


ql_first = results["QLearning"]["summary"]["first"]
ql_last = results["QLearning"]["summary"]["last"]
sa_first = results["SARSA"]["summary"]["first"]
sa_last = results["SARSA"]["summary"]["last"]

metrics = ["precision", "recall", "accuracy", "f1"]

ql_first_vals = [ql_first[m] for m in metrics]
ql_last_vals = [ql_last[m]  for m in metrics]
sa_first_vals = [sa_first[m] for m in metrics]
sa_last_vals = [sa_last[m]  for m in metrics]

x = np.arange(len(metrics))
width = 0.2

plt.figure(figsize=(10, 6))

plt.bar(x - 1.5*width, ql_first_vals, width, label="QL – First")
plt.bar(x - 0.5*width, ql_last_vals, width, label="QL – Last")
plt.bar(x + 0.5*width, sa_first_vals, width, label="SARSA – First")
plt.bar(x + 1.5*width, sa_last_vals, width, label="SARSA – Last")

plt.xticks(x, [m.capitalize() for m in metrics])
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("QLearning vs SARSA – Metrics: First vs Last Episode")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()