import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


RESULTS_PATH = Path("results/results.json")


def _load_results(path: Path = RESULTS_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if path.stat().st_size == 0:
        return {}
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _save_results(data: Dict[str, Any], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def update_agent_results(
    agent_name: str,
    agent_type: str,
    episodes: int,
    reward_curve: List[float],
    first_confusion_matrix: Dict[str, int],
    last_confusion_matrix: Dict[str, int],
    first_metrics: Dict[str, float],
    last_metrics: Dict[str, float],
    avg_reward: float,
    hyperparameters: Optional[Dict[str, Any]] = None,
    results_path: Path = RESULTS_PATH,
) -> None:
    """
    General logger for all agents.

    Parameters
    ----------
    agent_name : str
        Key in the JSON (e.g., "Random", "Q_Learning", "SARSA").
    agent_type : str
        A label like "baseline_random", "QLearning", "SARSA".
    episodes : int
        Number of episodes run (training or evaluation).
    reward_curve : list of float
        Total reward per episode.
    first_confusion_matrix : dict
        Confusion matrix for the FIRST episode.
    last_confusion_matrix : dict
        Confusion matrix for the LAST episode.
    first_metrics : dict
        Metrics for the first episode:
        { "reward", "precision", "recall", "accuracy", "f1" }
    last_metrics : dict
        Metrics for the last episode (same keys).
    avg_reward : float
        Average reward across all episodes.
    hyperparameters : dict, optional
        RL hyperparameters (alpha, gamma, epsilon, etc.).
    """
    results = _load_results(results_path)

    summary = {
        "episodes": episodes,
        "avg_reward": float(avg_reward),
        "first": {
            "reward": float(first_metrics.get("reward", 0.0)),
            "precision": float(first_metrics.get("precision", 0.0)),
            "recall": float(first_metrics.get("recall", 0.0)),
            "accuracy": float(first_metrics.get("accuracy", 0.0)),
            "f1": float(first_metrics.get("f1", 0.0)),
            "confusion_matrix": {
                "TP": int(first_confusion_matrix.get("TP", 0)),
                "FP": int(first_confusion_matrix.get("FP", 0)),
                "TN": int(first_confusion_matrix.get("TN", 0)),
                "FN": int(first_confusion_matrix.get("FN", 0)),
            },
        },
        "last": {
            "reward": float(last_metrics.get("reward", 0.0)),
            "precision": float(last_metrics.get("precision", 0.0)),
            "recall": float(last_metrics.get("recall", 0.0)),
            "accuracy": float(last_metrics.get("accuracy", 0.0)),
            "f1": float(last_metrics.get("f1", 0.0)),
            "confusion_matrix": {
                "TP": int(last_confusion_matrix.get("TP", 0)),
                "FP": int(last_confusion_matrix.get("FP", 0)),
                "TN": int(last_confusion_matrix.get("TN", 0)),
                "FN": int(last_confusion_matrix.get("FN", 0)),
            },
        },
    }

    agent_entry = {
        "agent_type": agent_type,
        "timestamp": datetime.utcnow().isoformat(),
        "hyperparameters": hyperparameters,
        "summary": summary,
        "reward_curve": [float(r) for r in reward_curve],
    }

    results[agent_name] = agent_entry
    _save_results(results, results_path)
