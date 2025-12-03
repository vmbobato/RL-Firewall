import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


RESULTS_PATH = Path("results/results.json")


def _load_results(path: Path = RESULTS_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _save_results(data: Dict[str, Any], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def update_agent_results(
    agent_name: str,
    agent_type: str,
    episodes: int,
    reward_curve: List[float],
    confusion_matrix: Dict[str, int],
    summary_metrics: Dict[str, float],
    per_episode_metrics: Optional[List[Dict[str, Any]]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    results_path: Path = RESULTS_PATH,
) -> None:
    """
    Updates results.json with metrics for a single agent.

    agent_name : str
        Key in the JSON (e.g., "Random", "Q_Learning", "SARSA").
    agent_type : str
        A label like "baseline_random", "QLearning", "SARSA".
    episodes : int
        Number of episodes evaluated.
    reward_curve : list of float
        Total reward per episode.
    confusion_matrix : dict
        Dict with keys: "TP", "FP", "TN", "FN".
    summary_metrics : dict
        Should at least contain: "avg_reward", "precision", "recall", "accuracy", "f1".
    per_episode_metrics : list of dict, optional
        List with per-episode info.
    hyperparameters : dict, optional
        RL hyperparameters (alpha, gamma, epsilon, etc.).
    """
    results = _load_results(results_path)

    avg_reward = summary_metrics.get("avg_reward", 0.0)
    precision = summary_metrics.get("precision", 0.0)
    recall = summary_metrics.get("recall", 0.0)
    accuracy = summary_metrics.get("accuracy", 0.0)
    f1 = summary_metrics.get("f1", 0.0)

    agent_entry = {
        "agent_type": agent_type,
        "timestamp": datetime.utcnow().isoformat(),
        "hyperparameters": hyperparameters,
        "summary": {
            "episodes": episodes,
            "avg_reward": avg_reward,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
        },
        "reward_curve": reward_curve,
        "confusion_matrix": confusion_matrix,
        "per_episode_metrics": per_episode_metrics or [],
    }

    results[agent_name] = agent_entry
    _save_results(results, results_path)
