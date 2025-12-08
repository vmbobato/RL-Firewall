import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple


class FirewallEnv:
    """
    Env for the RL Firewall
    
    Observation:
        - 9 binned flow feats
        - 1 src_strke_count_bin (0, 1, 2, 3)
        - 1 iface_load_bin (0, 1, 2)
        - Total 11 discrete feats as a vector
    Action:
        - ALLOW = 0
        - DENY  = 1
    Reward:
        - Malicious (label = 1):
            ALLOW -> -20
            DENY  -> +5
        - Benign (label = 0):
            ALLOW -> +0.5
            DENY  -> -2
    """
    metadata = {"render_modes" : []}
    
    def __init__(self, df : pd.DataFrame, iface_window_size : int = 1000, max_steps : Optional[int] = None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.iface_window_size = iface_window_size
        self.max_step = max_steps
        self.bin_cols = [
            "Flow_Duration_bin",
            "Tot_Fwd_Pkts_bin",
            "Tot_Bwd_Pkts_bin",
            "Pkt_Len_Mean_bin",
            "Flow_Bytes_s_bin",
            "ACK_Flag_Cnt_bin",
            "Init_Fwd_Win_Byts_bin",
            "SYN_Flag_Cnt_bin",
            "syn_ratio_bin",
        ]
        bin_sizes = []
        for col in self.bin_cols:
            max_bin = int(self.df[col].max())
            bin_sizes.append(max_bin + 1)
        # src_strike_count_bin: 0..3
        bin_sizes.append(4)
        # iface_load_bin: 0..2
        bin_sizes.append(3)

        self.observation_space = spaces.MultiDiscrete(bin_sizes)
        self.action_space = spaces.Discrete(2)
        
        self.idx = 0
        self.steps = 0
        self.src_strike_count = defaultdict(int)
        self.iface_bytes_window = deque(maxlen=iface_window_size)
        
    def reset(self, *, seed = None, options = None):
        if seed is not None:
            np.random.seed(seed)
        self.idx = 0
        self.steps = 0
        self.src_strike_count.clear()
        self.iface_bytes_window.clear()
        
        if seed is not None:
            self.df = self.df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        obs = self.build_state(self.idx)
        info = {}
        return obs, info
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid Action: {action}"
        row = self.df.iloc[self.idx]
        label = int(row["label"])
        src_ip = row["Src_IP"]
        reward = self.compute_reward(label, action)
        
        if label == 1:
            self.src_strike_count[src_ip] += 1
        if action == 0:
            flow_bps = float(row["Flow_Byts/s"])
            self.iface_bytes_window.append(max(flow_bps, 0.0))
        self.idx += 1
        self.steps += 1
        
        terminated = False
        if self.idx >= self.n:
            terminated = True
        if self.max_step is not None and self.steps >= self.max_step:
            terminated = True
        
        if terminated:
            obs = self.build_terminal_state()
        else:
            obs = self.build_state(self.idx)
            
        info = {
            "label": label,
            "src_ip": src_ip,
            "reward": reward,
            "terminated": terminated,
        }
        return obs, reward, terminated, info
        
        
    def build_state(self, idx):
        row = self.df.iloc[idx]

        obs = [int(row[col]) for col in self.bin_cols]

        src_ip = row["Src_IP"]
        strikes = self.src_strike_count[src_ip]
        strike_bin = self._bin_strike_count(strikes)
        obs.append(strike_bin)

        iface_load = self._compute_iface_load()
        iface_bin = self._bin_iface_load(iface_load)
        obs.append(iface_bin)

        return np.array(obs, dtype=np.int64)
    
    def build_terminal_state(self):
        return -1 * np.ones(11, dtype=np.int64)
    
    def _compute_iface_load(self) -> float:
        if not self.iface_bytes_window:
            return 0.0
        return float(sum(self.iface_bytes_window)) / len(self.iface_bytes_window)
    
    @staticmethod
    def compute_reward(label : int, action : int):
        if label == 1:
            return -20 if action == 0 else 5
        else:
            return 0.5 if action == 0 else -2
        
    @staticmethod
    def _bin_iface_load(load: float):
        if load < 1e3:
            return 0
        elif load < 1e5:
            return 1
        else:
            return 2
        
    @staticmethod
    def _bin_strike_count(strikes: int):
        if strikes == 0:
            return 0
        elif strikes <= 3:
            return 1
        elif strikes <= 10:
            return 2
        else:
            return 3